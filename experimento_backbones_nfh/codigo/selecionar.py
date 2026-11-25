#!/usr/bin/env python3
"""
selecionar.py: regras de seleção T1 -> T2 e T2 -> confirmação (protocolo §6).

REGRA DE OURO: este módulo lê SOMENTE colunas de identificação e `val_*`. As colunas `test_*` nunca são
carregadas: `ler_val` passa ao pandas uma lista fixa de colunas permitidas (`usecols`), de modo que o
texto das colunas de teste não chega à memória. test_selecionar.py falha se isso mudar.

T1 -> T2. Por combinação (dataset, modo, backbone, fração) em T1 (largura 1, semente 0):
  rel_RMSE(v) = val_RMSE(v)/val_RMSE(densa+E) - 1  e  rel_MAE(v) = val_MAE(v)/val_MAE(densa+E) - 1,
  para as variantes NFH treinadas (conv_* e zero_*; proj_K10 é só controle). A candidata é a de menor
  rel_RMSE. A combinação PASSA se rel_RMSE < 0 e a rel_MAE da mesma candidata <= 0. Vão para T2 as 16 melhores
  pelo rel_RMSE, no máximo 3 por conjunto (dataset, modo) e 4 por backbone, e com ao menos 2 sintéticas se houver
  tantas qualificadas (as duas melhores sintéticas entram primeiro, respeitando os tetos).
  T2: sementes 0, 1, 2; larguras 1 e 2 (só 1 para Linear e DLinear, que não têm largura); variantes
  densa, densa+E, proj_K10, conv_L{100,1000}_K{10,4}, conv_L0_K10 (controle da parametrização, §10 item 4), zero_L{0,100,1000}_K10, zero_L1000_K4.

T2 -> confirmação. Por combinação e largura, para cada variante NFH: média (sobre as 3 sementes) do rel_RMSE e do
  rel_MAE (cada semente contra a densa+E da própria semente). Candidata = menor média de rel_RMSE. Qualifica se
  média rel_RMSE < 0, média rel_MAE <= 0 e rel_RMSE < 0 nas três sementes. Até 8 combinações, no máximo 2 por
  conjunto e 3 por backbone, incluindo a melhor sintética se houver qualificada.
  Confirmação: sementes 10 a 14; densa, densa+E, proj_K10 e a candidata (e as ingênuas dos conjuntos).

Uso: python selecionar.py t1   -> resultados/selecao_T1.md e resultados/filas/t2.json   (lê resultados/t1.csv)
     python selecionar.py t2   -> resultados/selecao_T2.md e resultados/filas/confirmacao.json (lê t2.csv)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
sys.path.insert(0, str(AQUI))

RES = ROOT / "experimento_backbones_nfh" / "resultados"

COLUNAS_PERMITIDAS = ["dataset", "modo", "backbone", "largura", "fracao", "seed", "variante", "K", "lambda",
                      "val_RMSE", "val_MAE", "n_tr", "epoca_escolhida"]

MAX_T2, MAX_POR_CONJ_T1, MAX_POR_BB_T1, MIN_SINT = 12, 3, 3, 2   # §10, 21/11 18h15 (era 16, 3, 4, 2)
N_LARGURA2_T2 = 4   # §10, 21/11 18h15: largura 2 só para as 4 primeiras pelo rel_RMSE de validação
MAX_CONF, MAX_POR_CONJ_CONF, MAX_POR_BB_CONF = 8, 2, 3
SEEDS_T2, SEEDS_CONF = [0, 1, 2], [10, 11, 12, 13, 14]
VAR_T2 = ["densa", "densa+E", "proj_K10", "conv_L100_K10", "conv_L1000_K10", "conv_L1000_K4",  # sem conv_L100_K4 (§10)
          "conv_L0_K10", "zero_L0_K10", "zero_L100_K10", "zero_L1000_K10", "zero_L1000_K4"]


def ler_val(caminho) -> pd.DataFrame:
    """Lê o CSV só com as colunas permitidas (identificação e val_*). Nunca expõe colunas de teste."""
    df = pd.read_csv(caminho, usecols=lambda c: c in COLUNAS_PERMITIDAS,
                     dtype={"dataset": str, "modo": str, "backbone": str, "variante": str})
    assert not any(c.startswith("t" + "est") for c in df.columns)
    # linhas de cabeçalho repetidas (corrida na criação do CSV) são descartadas e as colunas val_* viram número
    df = df[df["dataset"] != "dataset"].copy()
    for c in df.columns:
        if c.startswith("val_") or c in ("largura", "fracao", "seed", "K", "lambda", "n_tr", "epoca_escolhida"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def eh_nfh(v: str) -> bool:
    return v.startswith("conv_") or v.startswith("zero_")


def sintetico(ds: str) -> bool:
    return ds.startswith("SINT")


def tabela_relativa(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (combinação, semente, variante NFH) com rel_RMSE e rel_MAE contra a densa+E da mesma tarefa."""
    d = df[df["backbone"].notna() & (df["backbone"] != "")].copy()
    chave = ["dataset", "modo", "backbone", "largura", "fracao", "seed"]
    ref = d[d["variante"] == "densa+E"][chave + ["val_RMSE", "val_MAE"]].rename(
        columns={"val_RMSE": "ref_RMSE", "val_MAE": "ref_MAE"})
    nfh = d[d["variante"].map(eh_nfh)]
    m = nfh.merge(ref, on=chave, how="inner")
    m["rel_RMSE"] = m["val_RMSE"] / m["ref_RMSE"] - 1
    m["rel_MAE"] = m["val_MAE"] / m["ref_MAE"] - 1
    return m


def _escolher(cand: pd.DataFrame, limite, max_conj, max_bb, min_sint=0, max_sint_prio=0):
    """Guloso por rel_RMSE crescente com tetos por conjunto e por backbone; sintéticas prioritárias."""
    cand = cand.sort_values("rel_RMSE").reset_index(drop=True)
    escolhidos, n_conj, n_bb = [], {}, {}

    def cabe(r):
        return n_conj.get((r["dataset"], r["modo"]), 0) < max_conj and n_bb.get(r["backbone"], 0) < max_bb

    def pegar(i, r):
        escolhidos.append(i)
        n_conj[(r["dataset"], r["modo"])] = n_conj.get((r["dataset"], r["modo"]), 0) + 1
        n_bb[r["backbone"]] = n_bb.get(r["backbone"], 0) + 1
    if min_sint:
        for i, r in cand[cand["dataset"].map(sintetico)].iterrows():
            if len([j for j in escolhidos if sintetico(cand.loc[j, "dataset"])]) >= min_sint:
                break
            if cabe(r):
                pegar(i, r)
    for i, r in cand.iterrows():
        if len(escolhidos) >= limite:
            break
        if i not in escolhidos and cabe(r):
            pegar(i, r)
    return cand.loc[sorted(escolhidos)].reset_index(drop=True), cand


def selecao_t1(df: pd.DataFrame):
    m = tabela_relativa(df)
    m = m[(m["seed"] == 0) & (m["largura"] == 1)]
    chave = ["dataset", "modo", "backbone", "fracao"]
    melhor = m.sort_values("rel_RMSE").groupby(chave, as_index=False).first()
    melhor["qualifica"] = (melhor["rel_RMSE"] < 0) & (melhor["rel_MAE"] <= 0)
    q = melhor[melhor["qualifica"]]
    esc, _ = _escolher(q, MAX_T2, MAX_POR_CONJ_T1, MAX_POR_BB_T1, min_sint=MIN_SINT)
    melhor["vai_T2"] = melhor.apply(lambda r: any((esc["dataset"] == r["dataset"]) & (esc["modo"] == r["modo"])
                                                 & (esc["backbone"] == r["backbone"])
                                                 & (esc["fracao"] == r["fracao"])), axis=1)
    return melhor.sort_values("rel_RMSE").reset_index(drop=True), esc


def fila_t2(esc: pd.DataFrame, usa_largura) -> list:
    tarefas = []
    topo = set(esc.sort_values("rel_RMSE").index[:N_LARGURA2_T2])   # §10: largura 2 só para as 4 melhores
    for i, r in esc.iterrows():
        for lw in ([1, 2] if (usa_largura(r["backbone"]) and i in topo) else [1]):
            for s in SEEDS_T2:
                tarefas.append({"dataset": r["dataset"], "modo": r["modo"], "backbone": r["backbone"], "largura": lw,
                                "fracao": float(r["fracao"]), "seed": s, "variantes": VAR_T2})
    return tarefas


def selecao_t2(df: pd.DataFrame):
    m = tabela_relativa(df)
    m = m[m["seed"].isin(SEEDS_T2)]
    chave = ["dataset", "modo", "backbone", "fracao", "largura"]
    agg = (m.groupby(chave + ["variante", "K", "lambda"], dropna=False)
           .agg(rel_RMSE=("rel_RMSE", "mean"), rel_MAE=("rel_MAE", "mean"), pior_semente=("rel_RMSE", "max"),
                n_sementes=("seed", "nunique")).reset_index())
    agg = agg[agg["n_sementes"] == len(SEEDS_T2)]
    melhor = agg.sort_values("rel_RMSE").groupby(chave, as_index=False).first()
    melhor["qualifica"] = (melhor["rel_RMSE"] < 0) & (melhor["rel_MAE"] <= 0) & (melhor["pior_semente"] < 0)
    q = melhor[melhor["qualifica"]]
    esc, _ = _escolher(q, MAX_CONF, MAX_POR_CONJ_CONF, MAX_POR_BB_CONF, min_sint=1)
    melhor["vai_confirmacao"] = melhor.apply(lambda r: any(
        (esc["dataset"] == r["dataset"]) & (esc["modo"] == r["modo"]) & (esc["backbone"] == r["backbone"])
        & (esc["fracao"] == r["fracao"]) & (esc["largura"] == r["largura"])), axis=1)
    return melhor.sort_values("rel_RMSE").reset_index(drop=True), esc


def fila_confirmacao(esc: pd.DataFrame) -> list:
    tarefas = [{"tipo": "ingenua", "dataset": d, "modo": mo} for d, mo in
               sorted({(r["dataset"], r["modo"]) for _, r in esc.iterrows()})]
    for _, r in esc.iterrows():
        for s in SEEDS_CONF:
            tarefas.append({"dataset": r["dataset"], "modo": r["modo"], "backbone": r["backbone"],
                            "largura": int(r["largura"]), "fracao": float(r["fracao"]), "seed": s,
                            "variantes": ["densa", "densa+E", "proj_K10", r["variante"]]})
    return tarefas


def _pct(x):
    return f"{100 * x:+.2f}%"


def md_t1(melhor, esc, caminho):
    L = ["# Seleção T1 → T2 (somente colunas de validação)", "",
         "Regras: `selecionar.py` (docstring). rel = valor da melhor variante NFH / valor da densa+E − 1, na validação. "
         "Passa se rel_RMSE < 0 e rel_MAE ≤ 0. Vão 16 (máx. 3 por conjunto, 4 por backbone, ≥ 2 sintéticas). "
         "Nenhuma coluna `test_*` foi lida.", "",
         f"Combinações em T1: {len(melhor)}; passam: {int(melhor['qualifica'].sum())}; vão para T2: {len(esc)}.", "",
         "| # | conjunto | modo | backbone | fração | melhor variante NFH | rel RMSE val | rel MAE val | passa | T2 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in melhor.iterrows():
        L.append(f"| {i + 1} | {r['dataset']} | {r['modo']} | {r['backbone']} | {r['fracao']:g} | {r['variante']} | "
                 f"{_pct(r['rel_RMSE'])} | {_pct(r['rel_MAE'])} | {'sim' if r['qualifica'] else 'não'} | "
                 f"{'sim' if r['vai_T2'] else 'não'} |")
    Path(caminho).write_text("\n".join(L) + "\n", encoding="utf-8")


def md_t2(melhor, esc, caminho):
    L = ["# Seleção T2 → confirmação (somente colunas de validação)", "",
         "rel = média, sobre as sementes 0, 1 e 2, de (val da variante NFH / val da densa+E da mesma semente − 1). "
         "Passa se média rel_RMSE < 0, média rel_MAE ≤ 0 e rel_RMSE < 0 nas três sementes. Até 8 (máx. 2 por conjunto, "
         "3 por backbone, incluindo a melhor sintética). Nenhuma coluna `test_*` foi lida.", "",
         f"Combinações × largura: {len(melhor)}; passam: {int(melhor['qualifica'].sum())}; "
         f"vão para a confirmação: {len(esc)}.", "",
         "| # | conjunto | modo | backbone | fração | largura | melhor variante | média rel RMSE | média rel MAE | "
         "pior semente (RMSE) | passa | confirmação |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in melhor.iterrows():
        L.append(f"| {i + 1} | {r['dataset']} | {r['modo']} | {r['backbone']} | {r['fracao']:g} | {int(r['largura'])} | "
                 f"{r['variante']} | {_pct(r['rel_RMSE'])} | {_pct(r['rel_MAE'])} | {_pct(r['pior_semente'])} | "
                 f"{'sim' if r['qualifica'] else 'não'} | {'sim' if r['vai_confirmacao'] else 'não'} |")
    Path(caminho).write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    from backbones import usa_largura
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapa", choices=["t1", "t2"])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--saida-dir", default=str(RES))
    a = ap.parse_args()
    out = Path(a.saida_dir)
    (out / "filas").mkdir(parents=True, exist_ok=True)
    if a.etapa == "t1":
        df = ler_val(a.csv or RES / "t1.csv")
        melhor, esc = selecao_t1(df)
        md_t1(melhor, esc, out / "selecao_T1.md")
        fila = fila_t2(esc, usa_largura)
        (out / "filas" / "t2.json").write_text(json.dumps({"meta": {"fase": "t2"}, "tarefas": fila}, indent=1))
        print(f"T1: {int(melhor['qualifica'].sum())} passam de {len(melhor)}; {len(esc)} vão para T2 ({len(fila)} tarefas)")
    else:
        df = ler_val(a.csv or RES / "t2.csv")
        melhor, esc = selecao_t2(df)
        md_t2(melhor, esc, out / "selecao_T2.md")
        fila = fila_confirmacao(esc)
        (out / "filas" / "confirmacao.json").write_text(json.dumps({"meta": {"fase": "confirmacao"}, "tarefas": fila},
                                                                     indent=1))
        print(f"T2: {int(melhor['qualifica'].sum())} passam de {len(melhor)}; {len(esc)} vão para a confirmação "
              f"({len(fila)} tarefas)")


if __name__ == "__main__":
    main()
