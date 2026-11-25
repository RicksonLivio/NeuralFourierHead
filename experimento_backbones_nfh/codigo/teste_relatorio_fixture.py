#!/usr/bin/env python3
"""
teste_relatorio_fixture.py: dados FALSOS para testar relatorio.py e explicar_bb.py sem tocar em resultados/ reais.

Tudo é gravado numa pasta temporária (--tmp): res/{t1,t2,confirmacao}.csv com o mesmo esquema do executar.py (números
sorteados, SEM significado científico), res/selecao_T1.md e selecao_T2.md e as filas (geradas pelo selecionar.py real,
via --csv e --saida-dir), e res/modelos/ com modelos PEQUENOS treinados por poucas épocas em CPU para as combinações
sintéticas da fila de confirmação (semente 10), com os nomes que a confirmação usaria
(<stem>_densa.pt, <stem>_<variante>.pt, <stem>_<variante>_pred.npy). Também grava exemplo/ com um
confirmacao.csv/json falso no formato de exemplo_sintetico/.

Uso:
  CUDA_VISIBLE_DEVICES='' .venv/bin/python experimento_backbones_nfh/codigo/teste_relatorio_fixture.py --tmp /tmp/fake_bb [--epocas 3]
Depois:
  .venv/bin/python experimento_backbones_nfh/codigo/relatorio.py --res /tmp/fake_bb/res --saida /tmp/fake_bb/relatorio --exemplo /tmp/fake_bb/exemplo
  CUDA_VISIBLE_DEVICES='' .venv/bin/python experimento_backbones_nfh/codigo/explicar_bb.py --res /tmp/fake_bb/res
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
sys.path.insert(0, str(AQUI))
PY = ROOT / ".venv" / "bin" / "python"

BBS = ["MLP", "Linear", "DLinear", "iTransformer", "TSMixer"]
CONJ = [("SINT-U48", "U", [1.0, 0.25, 0.1]), ("SINT-M", "M", [1.0, 0.25, 0.1]), ("ETTh1", "U", [0.25]), ("ETTh2", "U", [0.25]),
        ("jena", "U", [0.25])]
VAR_T1_REAL = ["densa", "densa+E", "proj_K10", "conv_L1000_K10", "zero_L100_K10", "zero_L1000_K10"]
VAR_T1_SINT = VAR_T1_REAL + ["zero_L1000_K4"]
from selecionar import VAR_T2  # noqa: E402
from executar import COLS  # noqa: E402


def sint(ds):
    return ds.startswith("SINT")


def linha_base(t, var, rng, base_v, base_t, ef_v, ef_t, K="", lam=""):
    """Uma linha do CSV. ef_v/ef_t: efeito relativo na validação/teste (só para variantes NFH)."""
    rv = base_v * (1 + (ef_v if var.startswith(("conv_", "zero_")) else (-0.004 if var == "densa+E" else 0.0)))
    rt = base_t * (1 + (ef_t if var.startswith(("conv_", "zero_")) else (-0.004 if var == "densa+E" else 0.0)))
    return {"dataset": t["dataset"], "modo": t["modo"], "backbone": t["backbone"], "largura": t["largura"], "fracao": f"{t['fracao']:g}",
            "seed": t["seed"], "variante": var, "K": K, "lambda": lam, "val_RMSE": rv, "val_MAE": 0.8 * rv, "test_RMSE": rt,
            "test_MAE": 0.8 * rt * (1 + (ef_t * 0.5 if var.startswith(("conv_", "zero_")) else 0)), "cobertura_mediana": 0.9, "epocas": 20,
            "epoca_escolhida": 12, "tempo_s": 5.0, "params": 1000, "n_tr": 500, "n_val": 300, "n_te": 300, "d": 1,
            "erro_conversao": 1e-7 if var.startswith("conv_") else ""}


def parse_var(v):
    import re
    m = re.match(r"^(?:conv|zero)_L([\d.]+)_K(\d+)$", v)
    return (int(m.group(2)), float(m.group(1))) if m else ("", "")


def injetar_cabecalho(caminho):
    """Simula a corrida na criação do CSV: repete o cabeçalho no meio do arquivo (depois de rodar o selecionar.py)."""
    L = caminho.read_text().splitlines()
    L.insert(len(L) // 2, L[0])
    caminho.write_text("\n".join(L) + "\n")


def escrever_csv(caminho, linhas, cab_repetido=False):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(linhas, columns=COLS)
    txt = df.to_csv(index=False)
    if cab_repetido:       # simula a corrida na criação do CSV: cabeçalho repetido no meio
        L = txt.splitlines()
        L.insert(len(L) // 2, L[0])
        txt = "\n".join(L) + "\n"
    caminho.write_text(txt)


def ingenuas(ds, modo, rng, fase_val):
    out = []
    for v in ("ingenua_media", "ingenua_ultimo", "ingenua_sazonal24"):
        b = rng.uniform(0.9, 1.5)
        out.append({"dataset": ds, "modo": modo, "backbone": "", "largura": "", "fracao": "", "seed": "", "variante": v,
                    "val_RMSE": b, "val_MAE": 0.8 * b, "test_RMSE": b * 1.02, "test_MAE": 0.8 * b * 1.02, "epocas": 0, "params": 0})
    return out


def fake_t1(rng):
    L = []
    for ds, modo, fr in CONJ:
        L += ingenuas(ds, modo, rng, True)
        for bb in BBS:
            for f in fr:
                t = {"dataset": ds, "modo": modo, "backbone": bb, "largura": 1, "fracao": f, "seed": 0}
                bv, bt = rng.uniform(0.3, 0.9), None
                bt = bv * rng.uniform(0.97, 1.05)
                for v in (VAR_T1_SINT if sint(ds) else VAR_T1_REAL):
                    ef_v = rng.normal(-0.04 if sint(ds) else 0.004, 0.02)
                    ef_t = ef_v + rng.normal(0.01, 0.01)
                    K, lam = parse_var(v)
                    L.append(linha_base(t, v, rng, bv, bt, ef_v, ef_t, K=K if K != "" else (10 if v == "proj_K10" else ""), lam=lam))
    return L


def fake_t2(fila, rng):
    L, cache = [], {}
    for t in fila:
        k = (t["dataset"], t["modo"], t["backbone"], t["fracao"], t["largura"])
        if k not in cache:
            cache[k] = rng.normal(-0.04 if sint(t["dataset"]) else -0.004, 0.012)
        bv = rng.uniform(0.3, 0.9)
        bt = bv * rng.uniform(0.97, 1.05)
        for v in t["variantes"]:
            ef_v = cache[k] + rng.normal(0, 0.006) + (0.01 if "zero_L0" in v else 0)
            ef_t = ef_v + rng.normal(0.008, 0.008)
            K, lam = parse_var(v)
            L.append(linha_base(t, v, rng, bv, bt, ef_v, ef_t, K=K if K != "" else (10 if v == "proj_K10" else ""), lam=lam))
    return L


def fake_conf(fila, rng):
    L = []
    for t in fila:
        if t.get("tipo") == "ingenua":
            L += ingenuas(t["dataset"], t["modo"], rng, False)
    comb = {}
    for t in fila:
        if t.get("tipo") == "ingenua":
            continue
        k = (t["dataset"], t["modo"], t["backbone"], t["fracao"], t["largura"])
        comb.setdefault(k, []).append(t)
    for i, (k, ts) in enumerate(comb.items()):
        if sint(k[0]):
            ef, ef_mae = -0.03, -0.03          # SUCESSO por construção
        elif i % 3 == 0:
            ef, ef_mae = -0.01, +0.02          # empate: RMSE médio a favor, MAE médio contra
        else:
            ef, ef_mae = +0.015, +0.015        # derrota
        bt0 = rng.uniform(0.3, 0.9)
        for t in ts:
            bt = bt0 * (1 + rng.normal(0, 0.01))
            for v in t["variantes"]:
                nfh = v.startswith(("conv_", "zero_"))
                K, lam = parse_var(v)
                r = linha_base(t, v, rng, bt, bt, ef + rng.normal(0, 0.005), ef + rng.normal(0, 0.005),
                               K=K if K != "" else (10 if v == "proj_K10" else ""), lam=lam)
                if nfh:
                    r["test_MAE"] = 0.8 * bt * (1 + ef_mae + rng.normal(0, 0.003))
                if v == "proj_K10":
                    r["test_RMSE"] = bt * 1.03
                    r["test_MAE"] = 0.8 * bt * 1.03
                L.append(r)
    return L


def fake_exemplo(pasta, rng):
    pasta.mkdir(parents=True, exist_ok=True)
    L = []
    for s in range(10, 20):
        base = 0.33 + rng.normal(0, 0.01)
        for v, f in (("densa", 1.0), ("densa+E", 0.999), ("NFH", 0.93)):
            L.append({"p": 16, "n": 1000, "semente": s, "variante": v, "val_RMSE": base * f, "val_MAE": 0.8 * base * f,
                      "test_RMSE": base * f, "test_MAE": 0.8 * base * f, "epoca": 30})
    pd.DataFrame(L).to_csv(pasta / "confirmacao.csv", index=False)
    (pasta / "confirmacao.json").write_text(json.dumps({"p": 16, "n": 1000, "variante_nfh": "zero_L1000", "veredito": "SUCESSO"}))


def treinar_modelos(res: Path, fila, epocas):
    """Treina, em CPU e por poucas épocas, a densa e a NFH das combinações SINTÉTICAS da confirmação (semente 10)."""
    import torch
    torch.set_num_threads(2)
    from executar import dados, id_arquivo, n_params, treino_fracao  # noqa: F401
    from backbones import construir_modelo
    from busca import prever
    from treino_bb import train_model_blocos
    from v2_common import set_seed
    from v3_common import CFG3, Q3, H3, densa_para_nfh
    cfg = replace(CFG3, max_epochs=epocas)
    dev = torch.device("cpu")
    (res / "modelos").mkdir(parents=True, exist_ok=True)
    feitos = set()
    for t in fila:
        if t.get("tipo") == "ingenua" or not sint(t["dataset"]) or t["seed"] != 10:
            continue
        nfh_v = [v for v in t["variantes"] if v.startswith(("conv_", "zero_"))][0]
        stem = id_arquivo(t)
        if stem in feitos:
            continue
        feitos.add(stem)
        pacote = dados(t["dataset"], t["modo"])
        ds, trf, va, te, Xv, Yv, Xte, Yte = pacote
        tr = treino_fracao(trf, t["fracao"])
        d = ds["feats"].shape[1]
        lw = int(t["largura"])
        print(f"treinando {stem} ({nfh_v}) n_tr={len(tr)} d={d}", flush=True)
        set_seed(10)
        densa = construir_modelo(t["backbone"], "densa", d, lw)
        densa, _ = train_model_blocos(densa, tr, va, 10, lam=0.0, cfg=cfg, dev=dev)
        torch.save(densa.state_dict(), res / "modelos" / f"{stem}_densa.pt")
        import re
        m = re.match(r"^(conv|zero)_L([\d.]+)_K(\d+)$", nfh_v)
        fam, lam, K = m.group(1), float(m.group(2)), int(m.group(3))
        if fam == "conv":
            nfh = densa_para_nfh(densa, K=K, q=Q3, H=H3)
        else:
            set_seed(10)
            nfh = construir_modelo(t["backbone"], "nfh", d, lw, K=K)
            nfh, _ = train_model_blocos(nfh, tr, va, 10, lam=lam, cfg=cfg, dev=dev)
        torch.save(nfh.state_dict(), res / "modelos" / f"{stem}_{nfh_v}.pt")
        np.save(res / "modelos" / f"{stem}_{nfh_v}_pred.npy", prever(nfh, Xte, dev).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--epocas", type=int, default=3)
    ap.add_argument("--sem-modelos", action="store_true")
    ap.add_argument("--semente", type=int, default=7)
    a = ap.parse_args()
    tmp = Path(a.tmp)
    assert "experimento_backbones_nfh/resultados" not in str(tmp.resolve()), "recuso gravar dentro de resultados/ reais"
    res = tmp / "res"
    rng = np.random.default_rng(a.semente)
    escrever_csv(res / "t1.csv", fake_t1(rng))
    sel = AQUI / "selecionar.py"
    subprocess.run([str(PY), str(sel), "t1", "--csv", str(res / "t1.csv"), "--saida-dir", str(res)], check=True)
    fila2 = json.loads((res / "filas" / "t2.json").read_text())["tarefas"]
    escrever_csv(res / "t2.csv", fake_t2(fila2, rng))
    subprocess.run([str(PY), str(sel), "t2", "--csv", str(res / "t2.csv"), "--saida-dir", str(res)], check=True)
    filac = json.loads((res / "filas" / "confirmacao.json").read_text())["tarefas"]
    escrever_csv(res / "confirmacao.csv", fake_conf(filac, rng))
    for f in ("t1", "t2", "confirmacao"):
        injetar_cabecalho(res / f"{f}.csv")
    fake_exemplo(tmp / "exemplo", rng)
    if not a.sem_modelos:
        treinar_modelos(res, filac, a.epocas)
    print("fixture pronta em", tmp)


if __name__ == "__main__":
    main()
