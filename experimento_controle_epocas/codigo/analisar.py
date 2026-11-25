#!/usr/bin/env python3
"""
analisar.py: análise pré-registrada do controle das épocas extras (INSTRUCOES.md §3) e relatório.

Para cada par (8 conjuntos x 3 backbones), a média do RMSE (e do MAE) de teste nas 5 sementes; rel = NFH / controle - 1
nos 24 pares, controle em {densa+E, densa}. Relata, como src/v3_tabelas.comparacoes: vitórias da NFH (rel < 0), média e
mediana de rel, mínimo e máximo, Wilcoxon bilateral `wilcoxon(rel)`, não inferioridade com margem de 1%
`wilcoxon(rel - 0.01, alternative="less")` (e a sensibilidade às margens 0; 0,25%; 0,5%), por backbone. Também:
épocas escolhidas (densa+E e NFH), reprodução (retreinadas x dissertação), exata_s0 (densa da dissertação + E épocas,
semente 0, contra a NFH da dissertação) e a conferência da função estatística contra results/v3/numeros_cap5.json.

Entrada: <dir>/controle.csv, <dir>/exata_s0.csv. Saída: <dir>/analise.json e o relatório em Markdown. Todo número do
relatório vem do JSON (a função `texto` só formata). Sem argumentos: dir = resultados/, relatório em
relatorio/relatorio_controle.md. Com --dir X: X/analise.json e X/relatorio_controle.md.
Se faltar tarefa, a análise principal usa só os pares com as 5 sementes das 3 variantes e o relatório diz o que falta.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
BASE = ROOT / "experimento_controle_epocas"
RES3 = ROOT / "results" / "v3"
CONJUNTOS = [f"{f}-{s}" for f in ("ETTh1", "ETTh2", "Jena", "AirQ") for s in ("U", "M")]
BACKBONES = ["MLP", "LSTM", "Conv1D+LSTM"]
SEEDS = (0, 1, 2, 3, 4)
MARGENS = (0.0, 0.0025, 0.005, 0.01)
sys.path.insert(0, str(AQUI))


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def ler(f):
    if not Path(f).exists():
        return []
    with open(f, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Estatística (a mesma de src/v3_tabelas.comparacoes)
# ---------------------------------------------------------------------------
def estatistica(rel, backbones=None):
    rel = np.asarray(rel, dtype=float)
    if len(rel) < 5:
        return {"n_pares": int(len(rel)), "erro": "menos de 5 pares: sem testes"}
    out = {"n_pares": int(len(rel)), "media_rel": float(rel.mean()), "media_rel_pct": float(100 * rel.mean()),
           "mediana_rel": float(np.median(rel)), "mediana_rel_pct": float(100 * np.median(rel)),
           "vitorias_nfh": int((rel < 0).sum()), "min_rel": float(rel.min()), "max_rel": float(rel.max()),
           "wilcoxon_bilateral_p": float(stats.wilcoxon(rel).pvalue),
           "nao_inferioridade_1pct_p": float(stats.wilcoxon(rel - 0.01, alternative="less").pvalue),
           "nao_inferioridade_margens_p": {f"{mg:.4f}": float(stats.wilcoxon(rel - mg, alternative="less").pvalue)
                                           for mg in MARGENS}}
    if backbones is not None:
        b = np.asarray(backbones)
        out["por_backbone"] = {bb: {"media_rel": float(rel[b == bb].mean()), "media_rel_pct": float(100 * rel[b == bb].mean()),
                                    "vitorias": int((rel[b == bb] < 0).sum()), "n": int((b == bb).sum())}
                               for bb in BACKBONES if (b == bb).any()}
    return out


# ---------------------------------------------------------------------------
# Médias por par
# ---------------------------------------------------------------------------
def medias_por_par(linhas, variantes, col):
    """Pares completos (5 sementes em todas as variantes): {(conj, bb): {variante: média nas 5 sementes}}."""
    acc = {}
    for r in linhas:
        if r["variante"] in variantes and num(r[col]) is not None:
            acc.setdefault((r["conjunto"], r["backbone"]), {}).setdefault(r["variante"], {})[int(r["seed"])] = num(r[col])
    return {k: {v: float(np.mean(list(d[v].values()))) for v in variantes}
            for k, d in acc.items() if all(v in d and set(d[v]) == set(SEEDS) for v in variantes)}


def comparar(comp_rmse, comp_mae, controle, alvo="nfh"):
    out = {}
    for met, comp in (("RMSE", comp_rmse), ("MAE", comp_mae)):
        pares = [k for k in [(c, b) for c in CONJUNTOS for b in BACKBONES] if k in comp]
        rel = [comp[k][alvo] / comp[k][controle] - 1 for k in pares]
        s = estatistica(rel, backbones=[k[1] for k in pares]) if pares else {"n_pares": 0, "erro": "sem pares completos"}
        s["pares"] = [{"conjunto": k[0], "backbone": k[1], alvo: comp[k][alvo], controle: comp[k][controle], "rel": r}
                      for k, r in zip(pares, rel)]
        out[met] = s
    return out


def ref_diss(conj):
    """Linhas da dissertação sem duplicatas de retomada (fica a última, como src/v3_tabelas.carregar_runs)."""
    ult = {(r["modelo"], r["variante"], r["K"], r["seed"]): r for r in ler(RES3 / "runs" / f"{conj}.csv")}
    return list(ult.values())


def conferencia_funcao():
    """Aplica `estatistica` às linhas da dissertação (NFH seguro K=10 x densa, RMSE e MAE) e compara com
    results/v3/numeros_cap5.json (saída de src/v3_tabelas.py)."""
    out = {}
    try:
        alvo = json.loads((RES3 / "numeros_cap5.json").read_text())["comparacoes"]
    except Exception as e:  # noqa: BLE001
        return {"erro": f"não consegui ler numeros_cap5.json: {e}"}
    for met, chave in (("RMSE", None), ("MAE", "mae")):
        rel, bbs = [], []
        for c in CONJUNTOS:
            L = ref_diss(c)
            for bb in BACKBONES:
                d = [num(r[met]) for r in L if r["modelo"] == f"{bb} + densa" and r["variante"] == "" and int(r["seed"]) >= 0]
                n = [num(r[met]) for r in L if r["modelo"] == f"{bb} + NFH" and r["variante"] == "seguro" and r["K"] == "10"
                     and int(r["seed"]) >= 0]
                rel.append((np.mean(n) - np.mean(d)) / np.mean(d))
                bbs.append(bb)
        mine = estatistica(rel, backbones=bbs)
        ref = alvo if chave is None else alvo[chave]
        dif = {k: abs(mine[k] - ref[k]) for k in ("media_rel", "mediana_rel", "wilcoxon_bilateral_p", "nao_inferioridade_1pct_p")}
        dif["vitorias_nfh"] = abs(mine["vitorias_nfh"] - ref["vitorias_nfh"])
        out[met] = {"dissertacao_recalculada": mine,
                    "numeros_cap5": {k: ref[k] for k in ("media_rel", "mediana_rel", "vitorias_nfh", "wilcoxon_bilateral_p",
                                                         "nao_inferioridade_1pct_p")},
                    "dif_abs": dif, "max_dif_abs": float(max(dif.values()))}
    return out


# ---------------------------------------------------------------------------
# Análise
# ---------------------------------------------------------------------------
def completude(dir_res, fila):
    """Tarefas da fila com todas as variantes gravadas."""
    ja_c = {(r["conjunto"], r["backbone"], r["seed"], r["variante"]) for r in ler(dir_res / "controle.csv")}
    ja_e = {(r["conjunto"], r["backbone"], r["seed"], r["variante"]) for r in ler(dir_res / "exata_s0.csv")}
    falta = []
    for t in fila:
        ja = ja_c if t["tipo"] == "treino" else ja_e
        vs = [v for v in t["variantes"] if (t["conjunto"], t["backbone"], str(t["seed"]), v) not in ja]
        if vs:
            falta.append({"tipo": t["tipo"], "conjunto": t["conjunto"], "backbone": t["backbone"], "seed": t["seed"],
                          "variantes_faltantes": vs})
    n_t = sum(t["tipo"] == "treino" for t in fila)
    return {"tarefas_total": len(fila), "tarefas_treino": n_t, "tarefas_exata": len(fila) - n_t,
            "tarefas_faltantes": len(falta), "faltam_treino": sum(x["tipo"] == "treino" for x in falta),
            "faltam_exata": sum(x["tipo"] == "exata_s0" for x in falta), "completo": len(falta) == 0, "faltantes": falta}


def hist_epocas(linhas, variante, E=15):
    rs = [r for r in linhas if r["variante"] == variante and r["epoca_escolhida"] != ""]
    if not rs:
        return {"n": 0}
    ep = np.array([int(float(r["epoca_escolhida"])) for r in rs])
    por_bb = {bb: {"n": sum(1 for r in rs if r["backbone"] == bb),
                   "na_epoca_0": sum(1 for r in rs if r["backbone"] == bb and int(float(r["epoca_escolhida"])) == 0)}
              for bb in BACKBONES}
    return {"n": int(len(ep)), "na_epoca_0": int((ep == 0).sum()), "na_epoca_E": int((ep == E).sum()),
            "media": float(ep.mean()), "mediana": float(np.median(ep)),
            "histograma": {str(e): int((ep == e).sum()) for e in range(E + 1)}, "por_backbone": por_bb}


def reproducao(linhas):
    out = {}
    for var in ("densa", "nfh"):
        rs = [r for r in linhas if r["variante"] == var and num(r["dif_test_RMSE"]) is not None]
        bloco = {"n": len(rs)}
        if rs:
            for met in ("RMSE", "MAE"):
                d = np.abs(np.array([num(r[f"dif_test_{met}"]) for r in rs]))
                bloco[f"dif_test_{met}"] = {"media_abs": float(d.mean()), "max_abs": float(d.max()),
                                            "mediana_abs": float(np.median(d)), "n_abs_menor_1e-6": int((d < 1e-6).sum()),
                                            "n_abs_menor_1e-3": int((d < 1e-3).sum())}
            d = np.abs(np.array([num(r["dif_val_RMSE"]) for r in rs]))
            bloco["dif_val_RMSE"] = {"media_abs": float(d.mean()), "max_abs": float(d.max())}
            bloco["por_dispositivo"] = {}
            for dev in ("GPU", "CPU"):
                s = [r for r in rs if r["dispositivo"] == dev]
                if s:
                    d = np.abs(np.array([num(r["dif_test_RMSE"]) for r in s]))
                    bloco["por_dispositivo"][dev] = {"n": len(s), "media_abs": float(d.mean()), "max_abs": float(d.max()),
                                                     "n_abs_menor_1e-6": int((d < 1e-6).sum())}
        out[var] = bloco
    return out


def exata(exata_linhas):
    E = {"n_tarefas_gravadas": len({(r["conjunto"], r["backbone"]) for r in exata_linhas if r["variante"] == "densa+E"})}
    pares = []
    for c in CONJUNTOS:
        L0 = ref_diss(c)
        for bb in BACKBONES:
            e = [r for r in exata_linhas if r["conjunto"] == c and r["backbone"] == bb and r["variante"] == "densa+E"]
            dd = [r for r in exata_linhas if r["conjunto"] == c and r["backbone"] == bb and r["variante"] == "densa_diss"]
            n0 = [r for r in L0 if r["modelo"] == f"{bb} + NFH" and r["variante"] == "seguro" and r["K"] == "10" and r["seed"] == "0"]
            d0 = [r for r in L0 if r["modelo"] == f"{bb} + densa" and r["variante"] == "" and r["seed"] == "0"]
            if e and dd and n0 and d0:
                p = {"conjunto": c, "backbone": bb, "densaE_RMSE": num(e[0]["test_RMSE"]), "densaE_MAE": num(e[0]["test_MAE"]),
                     "nfh_diss_RMSE": num(n0[0]["RMSE"]), "nfh_diss_MAE": num(n0[0]["MAE"]),
                     "densa_diss_RMSE": num(d0[0]["RMSE"]), "densa_diss_MAE": num(d0[0]["MAE"]),
                     "epoca_escolhida_densaE": int(float(e[0]["epoca_escolhida"])),
                     "epoca_escolhida_nfh_diss": int(float(n0[0]["epoca_ajuste"])),
                     "dif_densa_carregada_RMSE": num(dd[0]["dif_test_RMSE"])}
                p["rel_RMSE"] = p["nfh_diss_RMSE"] / p["densaE_RMSE"] - 1
                p["rel_MAE"] = p["nfh_diss_MAE"] / p["densaE_MAE"] - 1
                pares.append(p)
    E["pares"] = pares
    if pares:
        bb_ = [p["backbone"] for p in pares]
        E["nfh_diss_vs_densaE_RMSE"] = estatistica([p["rel_RMSE"] for p in pares], bb_)
        E["nfh_diss_vs_densaE_MAE"] = estatistica([p["rel_MAE"] for p in pares], bb_)
        E["nfh_diss_vs_densa_diss_RMSE"] = estatistica([p["nfh_diss_RMSE"] / p["densa_diss_RMSE"] - 1 for p in pares], bb_)
        E["nfh_diss_vs_densa_diss_MAE"] = estatistica([p["nfh_diss_MAE"] / p["densa_diss_MAE"] - 1 for p in pares], bb_)
        E["densaE_vs_densa_diss_RMSE"] = estatistica([p["densaE_RMSE"] / p["densa_diss_RMSE"] - 1 for p in pares], bb_)
        E["epoca_0_densaE"] = int(sum(p["epoca_escolhida_densaE"] == 0 for p in pares))
        E["epoca_0_nfh_diss"] = int(sum(p["epoca_escolhida_nfh_diss"] == 0 for p in pares))
        E["max_abs_dif_densa_carregada_RMSE"] = float(max(abs(p["dif_densa_carregada_RMSE"]) for p in pares))
    return E


def analisar(dir_res: Path, fila_path: Path | None):
    controle, ex = ler(dir_res / "controle.csv"), ler(dir_res / "exata_s0.csv")
    if fila_path and fila_path.exists():
        fila = json.loads(fila_path.read_text())["tarefas"]
    else:
        import filas
        fila = filas.gerar_real()
    A = {"fonte": {"controle": str(dir_res / "controle.csv"), "exata_s0": str(dir_res / "exata_s0.csv"),
                   "fila": str(fila_path) if fila_path else None},
         "completude": completude(dir_res, fila), "n_linhas_controle": len(controle), "n_linhas_exata": len(ex),
         "conferencia_funcao_estatistica": conferencia_funcao()}
    vs = ("densa", "densa+E", "nfh")
    c_rmse, c_mae = medias_por_par(controle, vs, "test_RMSE"), medias_por_par(controle, vs, "test_MAE")
    A["pares_completos"] = len(c_rmse)
    A["pares_incompletos"] = [f"{c}|{b}" for c in CONJUNTOS for b in BACKBONES if (c, b) not in c_rmse]
    A["nfh_vs_densaE"] = comparar(c_rmse, c_mae, "densa+E")
    A["nfh_vs_densa"] = comparar(c_rmse, c_mae, "densa")
    A["densaE_vs_densa_descritivo"] = comparar(c_rmse, c_mae, "densa", alvo="densa+E")   # efeito das épocas sozinho
    A["epocas_escolhidas"] = {"densa+E": hist_epocas(controle, "densa+E"), "nfh": hist_epocas(controle, "nfh")}
    A["reproducao"] = reproducao(controle)
    rel = []
    for k in sorted(c_rmse):
        nd = [num(r["diss_test_RMSE"]) for r in controle if (r["conjunto"], r["backbone"]) == k and r["variante"] == "nfh"]
        dd = [num(r["diss_test_RMSE"]) for r in controle if (r["conjunto"], r["backbone"]) == k and r["variante"] == "densa"]
        if len(nd) == 5 and len(dd) == 5:
            rel.append(np.mean(nd) / np.mean(dd) - 1)
    A["dissertacao_nfh_vs_densa_nos_pares_completos"] = estatistica(rel) if rel else {"n_pares": 0}
    tm = {}
    for v in vs:
        t = [num(r["tempo_s"]) for r in controle if r["variante"] == v and num(r["tempo_s"]) is not None]
        if t:
            tm[v] = {"n": len(t), "media_s": float(np.mean(t)), "soma_h": float(np.sum(t) / 3600)}
    A["tempo"] = tm
    por_exec = {}
    for r in controle:
        por_exec.setdefault((r["conjunto"], r["backbone"], r["seed"]), {})[r["variante"]] = num(r["val_RMSE"])
    viol = {v: sum(1 for d in por_exec.values() if "densa" in d and v in d and d[v] > d["densa"] + 1e-6)
            for v in ("densa+E", "nfh")}
    A["asserts_violados_nas_linhas"] = {
        "erro_conversao_max": max([num(r["erro_conversao"]) for r in controle if num(r["erro_conversao"]) is not None], default=None),
        "val_densaE_maior_densa": viol["densa+E"], "val_nfh_maior_densa": viol["nfh"]}
    A["exata_s0"] = exata(ex)
    return A


# ---------------------------------------------------------------------------
# Relatório (só formata o JSON)
# ---------------------------------------------------------------------------
def f(x, nd=4):
    return "n/d" if x is None else f"{x:.{nd}f}".replace(".", ",")


def pc(x, nd=2):
    return "n/d" if x is None else f"{100 * x:+.{nd}f}".replace(".", ",") + "%"


def pv(x):
    if x is None:
        return "n/d"
    return f"{x:.2e}".replace(".", ",") if (x < 1e-3 and x != 0) else f"{x:.4f}".replace(".", ",")


def linha_stats(nome, s):
    if "erro" in s or s.get("n_pares", 0) < 5:
        return f"| {nome} | {s.get('n_pares', 0)} | sem testes (menos de 5 pares) | | | | | |"
    return (f"| {nome} | {s['n_pares']} | {s['vitorias_nfh']} | {pc(s['media_rel'])} | {pc(s['mediana_rel'])} | "
            f"{pc(s['min_rel'])} a {pc(s['max_rel'])} | {pv(s['wilcoxon_bilateral_p'])} | {pv(s['nao_inferioridade_1pct_p'])} |")


CAB = ("| comparação | pares | vitórias (rel < 0) | média de rel | mediana de rel | mín. a máx. | Wilcoxon bilateral (p) | "
       "não inferioridade 1% (p) |\n|---|---|---|---|---|---|---|---|")


def texto(A):
    c = A["completude"]
    L = ["# Controle das épocas extras: relatório", "",
         "Gerado por `codigo/analisar.py` a partir de `resultados/analise.json`; nenhum número foi digitado à mão. "
         "Protocolo: `INSTRUCOES.md` (§3). `rel` = RMSE (ou MAE) da NFH dividido pelo do controle, menos 1; negativo favorece a NFH. "
         "Nenhuma escolha usou o teste.", ""]
    if not c["completo"]:
        L += [f"**Execução incompleta:** faltam {c['tarefas_faltantes']} de {c['tarefas_total']} tarefas "
              f"({c['faltam_treino']} de treino, de {c['tarefas_treino']}, e {c['faltam_exata']} exata_s0, de {c['tarefas_exata']}). "
              f"A análise abaixo usa só os {A['pares_completos']} pares com as 5 sementes das 3 variantes. "
              "Não use estes números como resultado final.", ""]
    else:
        L += [f"Execução completa: {c['tarefas_total']} tarefas ({c['tarefas_treino']} de treino e {c['tarefas_exata']} exata_s0); "
              f"{A['pares_completos']} pares completos.", ""]
    if A["pares_incompletos"]:
        L += [f"Pares fora da análise principal: {', '.join(A['pares_incompletos'])}.", ""]
    L += ["## 1. Resultado principal: NFH contra a densa com as mesmas épocas", "", CAB]
    for met in ("RMSE", "MAE"):
        L.append(linha_stats(f"NFH × densa+E ({met})", A["nfh_vs_densaE"][met]))
    for met in ("RMSE", "MAE"):
        L.append(linha_stats(f"NFH × densa ({met})", A["nfh_vs_densa"][met]))
    for met in ("RMSE", "MAE"):
        L.append(linha_stats(f"densa+E × densa ({met}), descritivo", A["densaE_vs_densa_descritivo"][met]))
    L += ["", "As duas últimas linhas mostram o efeito das épocas extras sozinho (densa+E contra densa), sem a NFH; "
          "são descritivas e não fazem parte da análise pré-registrada. Nelas, a coluna de vitórias conta os pares em que a densa+E é melhor.", ""]
    for chave, titulo in (("nfh_vs_densaE", "NFH × densa+E"), ("nfh_vs_densa", "NFH × densa")):
        for met in ("RMSE", "MAE"):
            s = A[chave][met]
            if s.get("n_pares", 0) < 5:
                continue
            ni = s["nao_inferioridade_1pct_p"]
            L.append(f"- {titulo}, {met}: a NFH venceu em {s['vitorias_nfh']} de {s['n_pares']} pares; média de rel {pc(s['media_rel'])}; "
                     f"o teste de não inferioridade com margem de 1% deu p = {pv(ni)} "
                     f"({'rejeita' if ni < 0.05 else 'não rejeita'} a hipótese de a NFH ser pior que o controle em mais de 1%, ao nível de 5%); "
                     f"o Wilcoxon bilateral deu p = {pv(s['wilcoxon_bilateral_p'])} "
                     f"({'há' if s['wilcoxon_bilateral_p'] < 0.05 else 'não há'} diferença detectável ao nível de 5%). "
                     "Por backbone (vitórias/pares; média de rel): "
                     + "; ".join(f"{bb} {v['vitorias']}/{v['n']}, {pc(v['media_rel'])}" for bb, v in s["por_backbone"].items()) + ".")
    L += ["", "Sensibilidade à margem da não inferioridade (p-valores):", "", "| comparação | 0% | 0,25% | 0,5% | 1% |", "|---|---|---|---|---|"]
    for chave, titulo in (("nfh_vs_densaE", "NFH × densa+E"), ("nfh_vs_densa", "NFH × densa")):
        for met in ("RMSE", "MAE"):
            s = A[chave][met]
            if s.get("n_pares", 0) >= 5:
                m = s["nao_inferioridade_margens_p"]
                L.append(f"| {titulo} ({met}) | " + " | ".join(pv(m[k]) for k in ("0.0000", "0.0025", "0.0050", "0.0100")) + " |")
    L += ["", "## 2. Valores por par (RMSE de teste, média das 5 sementes, escala padronizada)", "",
          "| conjunto | backbone | densa | densa+E | NFH | rel NFH × densa+E | rel NFH × densa |", "|---|---|---|---|---|---|---|"]
    pd_ = {(p["conjunto"], p["backbone"]): p for p in A["nfh_vs_densa"]["RMSE"].get("pares", [])}
    pe_ = {(p["conjunto"], p["backbone"]): p for p in A["nfh_vs_densaE"]["RMSE"].get("pares", [])}
    for k in sorted(pe_, key=lambda k: (CONJUNTOS.index(k[0]), BACKBONES.index(k[1]))):
        a, b = pe_[k], pd_[k]
        L.append(f"| {k[0]} | {k[1]} | {f(b['densa'])} | {f(a['densa+E'])} | {f(a['nfh'])} | {pc(a['rel'])} | {pc(b['rel'])} |")
    L += ["", "## 3. Épocas escolhidas pela validação", ""]
    for v, nome in (("densa+E", "densa+E"), ("nfh", "NFH (ajuste estrutural, seguro)")):
        h = A["epocas_escolhidas"][v]
        if h["n"]:
            L.append(f"- {nome}: {h['na_epoca_0']} de {h['n']} execuções ficaram na época 0 (a própria densa) e {h['na_epoca_E']} na época 15; "
                     f"média {f(h['media'], 2)}, mediana {f(h['mediana'], 1)}. Na época 0, por backbone: "
                     + "; ".join(f"{bb} {x['na_epoca_0']}/{x['n']}" for bb, x in h["por_backbone"].items())
                     + ". Histograma (época: execuções): " + ", ".join(f"{e}: {n}" for e, n in h["histograma"].items()) + ".")
    L += ["", "## 4. Reprodução da dissertação (retreino com a mesma semente e o mesmo dispositivo)", ""]
    for var, nome in (("densa", "densa retreinada"), ("nfh", "NFH retreinada")):
        r = A["reproducao"][var]
        if r.get("n"):
            L.append(f"- {nome} ({r['n']} linhas): diferença absoluta do RMSE de teste para a dissertação, média {pv(r['dif_test_RMSE']['media_abs'])}, "
                     f"máxima {pv(r['dif_test_RMSE']['max_abs'])}; MAE, média {pv(r['dif_test_MAE']['media_abs'])}, máxima {pv(r['dif_test_MAE']['max_abs'])}; "
                     f"{r['dif_test_RMSE']['n_abs_menor_1e-6']} de {r['n']} com diferença menor que 1e-6 e {r['dif_test_RMSE']['n_abs_menor_1e-3']} menor que 1e-3. "
                     "Por dispositivo (RMSE: média; máxima; n com < 1e-6): "
                     + "; ".join(f"{d} {pv(x['media_abs'])}; {pv(x['max_abs'])}; {x['n_abs_menor_1e-6']}/{x['n']}" for d, x in r["por_dispositivo"].items()) + ".")
    ds = A["dissertacao_nfh_vs_densa_nos_pares_completos"]
    if ds.get("n_pares", 0) >= 5:
        L += ["", f"Nos mesmos {ds['n_pares']} pares, a dissertação (colunas `diss_*`) tem NFH × densa com {ds['vitorias_nfh']} vitórias e média de rel {pc(ds['media_rel'])}; "
              f"a repetição acima, NFH × densa (RMSE), tem {A['nfh_vs_densa']['RMSE']['vitorias_nfh']} vitórias e média {pc(A['nfh_vs_densa']['RMSE']['media_rel'])}."]
    cf = A["conferencia_funcao_estatistica"]
    if "RMSE" in cf:
        L += ["", f"Conferência da função estatística: aplicada às linhas da dissertação, reproduz `results/v3/numeros_cap5.json` "
              f"(NFH × densa, RMSE: {cf['RMSE']['dissertacao_recalculada']['vitorias_nfh']} vitórias, média de rel {pc(cf['RMSE']['dissertacao_recalculada']['media_rel'])}) "
              f"com diferença absoluta máxima {pv(cf['RMSE']['max_dif_abs'])} (RMSE) e {pv(cf['MAE']['max_dif_abs'])} (MAE)."]
    L += ["", "## 5. Comparação exata na semente 0 (densa da dissertação + E épocas × NFH da dissertação)", ""]
    E = A["exata_s0"]
    if E.get("pares"):
        L += [f"{len(E['pares'])} pares gravados. A densa carregada reproduz a linha da dissertação com diferença máxima de RMSE de teste "
              f"{pv(E['max_abs_dif_densa_carregada_RMSE'])}. Época 0 escolhida: densa+E em {E['epoca_0_densaE']} pares, "
              f"NFH da dissertação em {E['epoca_0_nfh_diss']} pares.", "", CAB]
        L.append(linha_stats("NFH × densa+E exata (RMSE)", E["nfh_diss_vs_densaE_RMSE"]))
        L.append(linha_stats("NFH × densa+E exata (MAE)", E["nfh_diss_vs_densaE_MAE"]))
        L.append(linha_stats("NFH × densa da dissertação (RMSE)", E["nfh_diss_vs_densa_diss_RMSE"]))
        L.append(linha_stats("NFH × densa da dissertação (MAE)", E["nfh_diss_vs_densa_diss_MAE"]))
        L.append(linha_stats("densa+E exata × densa da dissertação (RMSE), descritivo", E["densaE_vs_densa_diss_RMSE"]))
    else:
        L.append("Nenhum par exata_s0 completo ainda.")
    ac = A["asserts_violados_nas_linhas"]
    L += ["", "## 6. Verificações e custo", "",
          f"- Erro de conversão máximo nas linhas: {pv(ac['erro_conversao_max'])} (limite 1e-5).",
          f"- Execuções com val_RMSE(densa+E) acima do da densa (tolerância 1e-6): {ac['val_densaE_maior_densa']}; da NFH: {ac['val_nfh_maior_densa']}."]
    for v, t in A["tempo"].items():
        L.append(f"- Tempo, {v}: {t['n']} linhas, média {f(t['media_s'], 1)} s, soma {f(t['soma_h'], 2)} h.")
    L += ["", f"Linhas: {A['n_linhas_controle']} em controle.csv e {A['n_linhas_exata']} em exata_s0.csv.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None, help="pasta com controle.csv e exata_s0.csv (padrão resultados/)")
    ap.add_argument("--fila", default=None)
    a = ap.parse_args()
    d = Path(a.dir) if a.dir else BASE / "resultados"
    fila = Path(a.fila) if a.fila else (BASE / "resultados" / "filas" / "controle.json")
    A = analisar(d, fila)
    (d / "analise.json").write_text(json.dumps(A, indent=1, ensure_ascii=False))
    rel = BASE / "relatorio" / "relatorio_controle.md" if a.dir is None else d / "relatorio_controle.md"
    rel.parent.mkdir(parents=True, exist_ok=True)
    rel.write_text(texto(A))
    c = A["completude"]
    print(f"{d / 'analise.json'}; {rel}; tarefas faltantes {c['tarefas_faltantes']}/{c['tarefas_total']}; pares completos {A['pares_completos']}")


if __name__ == "__main__":
    main()
