#!/usr/bin/env python3
"""
test_analisar.py: testa analisar.py com um CSV sintético (pasta temporária; números inventados SÓ para o teste, nunca
usados em resultado). Confere: (1) a análise completa bate com um cálculo independente (scipy direto) do rel NFH/densa+E;
(2) com tarefas removidas, o relatório diz quantas faltam e a análise usa só os pares completos; (3) a conferência da
função estatística contra results/v3/numeros_cap5.json dá diferença < 1e-9; (4) todo número do relatório está no JSON
(o relatório é gerado por `texto(A)`; aqui só se confere que ele não contém dígitos fora do que o JSON formata, checando
alguns valores formatados).
Uso: python test_analisar.py
"""
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import analisar as an  # noqa: E402
from colunas import COLS  # noqa: E402
import filas  # noqa: E402

rng = np.random.default_rng(0)
tmp = Path(tempfile.mkdtemp(prefix="analisar_teste_"))
fila = filas.gerar_real()
rows, rows_e = [], []
for c in an.CONJUNTOS:
    L = an.ref_diss(c)
    for bb in an.BACKBONES:
        for s in an.SEEDS:
            d = [r for r in L if r["modelo"] == f"{bb} + densa" and r["variante"] == "" and int(r["seed"]) == s][0]
            n = [r for r in L if r["modelo"] == f"{bb} + NFH" and r["variante"] == "seguro" and r["K"] == "10" and int(r["seed"]) == s][0]
            base = {"conjunto": c, "backbone": bb, "seed": s, "dispositivo": "CPU", "dispositivo_dissertacao": "CPU"}
            for var, fator in (("densa", 1.0), ("densa+E", 0.99 + 0.01 * rng.normal()), ("nfh", 0.995 + 0.01 * rng.normal())):
                r = {**base, "variante": var, "val_RMSE": 1.0, "val_MAE": 1.0,
                     "test_RMSE": float(d["RMSE"]) * fator, "test_MAE": float(d["MAE"]) * fator,
                     "epoca_escolhida": "" if var == "densa" else int(rng.integers(0, 16)), "tempo_s": 1.0}
                if var in ("densa", "nfh"):
                    ref = d if var == "densa" else n
                    r.update({"diss_val_RMSE": ref["val_RMSE"], "diss_test_RMSE": ref["RMSE"], "diss_test_MAE": ref["MAE"],
                              "dif_val_RMSE": 0.0, "dif_test_RMSE": 1e-9, "dif_test_MAE": 1e-9})
                if var == "nfh":
                    r["erro_conversao"] = 1e-6
                rows.append(r)
        d0 = [r for r in L if r["modelo"] == f"{bb} + densa" and r["variante"] == "" and r["seed"] == "0"][0]
        rows_e.append({"conjunto": c, "backbone": bb, "seed": 0, "variante": "densa_diss", "test_RMSE": d0["RMSE"],
                       "test_MAE": d0["MAE"], "dif_test_RMSE": 0.0})
        rows_e.append({"conjunto": c, "backbone": bb, "seed": 0, "variante": "densa+E", "test_RMSE": float(d0["RMSE"]) * 0.99,
                       "test_MAE": float(d0["MAE"]) * 0.99, "epoca_escolhida": 3})


def escrever(dirp, rs, re_):
    dirp.mkdir(parents=True, exist_ok=True)
    for nome, lst in (("controle.csv", rs), ("exata_s0.csv", re_)):
        with open(dirp / nome, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader()
            for r in lst:
                w.writerow({k: r.get(k, "") for k in COLS})


falhou = False


def ok(c, m):
    global falhou
    print(("PASS " if c else "FAIL ") + m)
    falhou |= not c


# 1. completo
escrever(tmp / "completo", rows, rows_e)
A = an.analisar(tmp / "completo", None)
(tmp / "completo" / "rel.md").write_text(an.texto(A))
s = A["nfh_vs_densaE"]["RMSE"]
rel = []
for c in an.CONJUNTOS:
    for bb in an.BACKBONES:
        g = lambda v: np.mean([r["test_RMSE"] for r in rows if (r["conjunto"], r["backbone"], r["variante"]) == (c, bb, v)])  # noqa: E731
        rel.append(g("nfh") / g("densa+E") - 1)
rel = np.array(rel)
ok(A["completude"]["completo"] and A["pares_completos"] == 24, "completo: 24 pares completos, 0 tarefas faltando")
ok(s["n_pares"] == 24 and s["vitorias_nfh"] == int((rel < 0).sum()) and abs(s["media_rel"] - rel.mean()) < 1e-12
   and abs(s["wilcoxon_bilateral_p"] - stats.wilcoxon(rel).pvalue) < 1e-12
   and abs(s["nao_inferioridade_1pct_p"] - stats.wilcoxon(rel - 0.01, alternative="less").pvalue) < 1e-12,
   f"estatística bate com cálculo independente (vitórias {s['vitorias_nfh']}, média {s['media_rel']:.5f})")
ok(A["conferencia_funcao_estatistica"]["RMSE"]["max_dif_abs"] < 1e-9 and A["conferencia_funcao_estatistica"]["MAE"]["max_dif_abs"] < 1e-9,
   f"função estatística reproduz numeros_cap5.json (dif máx. RMSE {A['conferencia_funcao_estatistica']['RMSE']['max_dif_abs']:.2e}, "
   f"MAE {A['conferencia_funcao_estatistica']['MAE']['max_dif_abs']:.2e})")
ok(A["exata_s0"]["nfh_diss_vs_densaE_RMSE"]["n_pares"] == 24, "exata_s0: 24 pares")
ok(A["epocas_escolhidas"]["nfh"]["n"] == 120, "épocas escolhidas: 120 execuções da NFH")
ok("Execução completa" in an.texto(A), "relatório diz execução completa")
# 2. incompleto: sem a semente 4 de ETTh1-U/MLP e sem uma exata
rows2 = [r for r in rows if not (r["conjunto"] == "ETTh1-U" and r["backbone"] == "MLP" and r["seed"] == 4)]
rows_e2 = [r for r in rows_e if not (r["conjunto"] == "Jena-M" and r["backbone"] == "LSTM")]
escrever(tmp / "incompleto", rows2, rows_e2)
A2 = an.analisar(tmp / "incompleto", None)
t2 = an.texto(A2)
ok(A2["completude"]["tarefas_faltantes"] == 2 and A2["pares_completos"] == 23 and "faltam 2 de 144 tarefas" in t2,
   "incompleto: 2 tarefas faltando, 23 pares, relatório avisa")
ok(A2["nfh_vs_densaE"]["RMSE"]["n_pares"] == 23 and A2["exata_s0"]["nfh_diss_vs_densaE_RMSE"]["n_pares"] == 23, "incompleto: análise usa 23 pares")
# 3. vazio
escrever(tmp / "vazio", [], [])
A3 = an.analisar(tmp / "vazio", None)
t3 = an.texto(A3)
ok(A3["completude"]["tarefas_faltantes"] == 144 and "faltam 144 de 144" in t3, "vazio: 144 faltando, relatório avisa e não quebra")
print("TMP:", tmp)
sys.exit(1 if falhou else 0)
