#!/usr/bin/env python3
"""
filas.py: gera a fila de T1 (protocolo §5 e §6) e a fila do smoke test.

T1: todos os backbones, largura 1, todos os conjuntos, frações do §5, semente 0.
  reais (U; M só com --reais-m): frações {1,0; 0,25};  sintéticos (SINT-U48-U, SINT-M-M): {1,0; 0,25; 0,10}.
  variantes: densa, densa+E, proj_K10, conv_L1000_K10, zero_L100_K10, zero_L1000_K10 (+ zero_L1000_K4 nos sintéticos).
As tarefas de ingênuas (uma por conjunto e modo) vão primeiro. As demais são ordenadas da mais cara para a
mais barata (custo = janelas de treino x tempo por janela medido em resultados/tempo_epoca.json, se existir),
o que reduz a cauda do último worker.

Corte do §7 (só se a estimativa de tempo passar de ~4 h): --sem-frac025 ETTh1 jena_climate ... remove a fração 0,25
desses conjuntos reais. O corte aplicado tem de ser registrado no §10 do protocolo.

Corte B2 (decisão do coordenador, protocolo §10, 21/11/2026 16h10): `--b2`. Sintéticos com as três frações; reais com
a fração 0,25 e, com a fração 1,0, só os que têm até LIMITE_FRAC1 = 12100 janelas de treino. Opção M do §10:
`--opcao-m`, acrescenta ETTh1-M e opsd_de_solar-M, fração 0,25, só para iTransformer, TSMixer e PatchTST.
Fila do §10: `python filas.py t1 --b2 --opcao-m --saida resultados/filas/t1_B2.json` (todos os backbones do REGISTRO,
largura 1, semente 0).

Uso: python filas.py t1 [--reais-m] [--sem-frac025 NOME ...] [--b2] [--opcao-m] [--saida resultados/filas/t1.json]
     python filas.py smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
sys.path.insert(0, str(AQUI))

from backbones import REGISTRO  # noqa: E402

RES = ROOT / "experimento_backbones_nfh" / "resultados"
SINTETICOS = [("SINT-U48", "U"), ("SINT-M", "M")]
REAIS = ["ETTh1", "ETTh2", "jena_climate", "air_quality", "bike_sharing_dc", "seoul_bike", "household_power",
         "melbourne_pedestrian_southbank", "electricity_mt156", "opsd_de_solar"]   # os 3 últimos: pesquisa (21/11/2026)
FRAC_REAL, FRAC_SINT = [1.0, 0.25], [1.0, 0.25, 0.10]
VAR_T1 = ["densa", "densa+E", "proj_K10", "conv_L1000_K10", "zero_L100_K10", "zero_L1000_K10"]
VAR_T1_SINT = VAR_T1 + ["zero_L1000_K4"]
BACKBONES = list(REGISTRO)
LIMITE_FRAC1 = 12100                       # B2: fração 1,0 dos reais só até esse nº de janelas de treino
OPCAO_M = [("ETTh1", "M"), ("opsd_de_solar", "M")]
BB_OPCAO_M = ["iTransformer", "TSMixer", "PatchTST"]
FRAC_OPCAO_M = [0.25]


def janelas_treino(dataset, modo):
    from dados_bb import carregar_bb, segmentos_bb
    tr, va, te = segmentos_bb(carregar_bb(f"{dataset}-{modo}"))
    return len(tr), len(va), len(te)


def custo_por_janela():
    f = RES / "tempo_epoca.json"
    if f.exists():
        return json.loads(f.read_text()).get("gpu_s_por_janela_epoca_w1", {})
    return {}


def gerar_t1(reais_m=False, sem_frac025=(), so_frac025_reais=False, b2=False, opcao_m=False):
    cj = custo_por_janela()
    from dados_bb import ler_catalogo_unido
    com_cov = {e["nome"] for e in ler_catalogo_unido() if e.get("covariaveis")}
    conjuntos = [(d, "U", False) for d in REAIS] + [(d, "M", False) for d in (REAIS if reais_m else []) if d in com_cov]
    conjuntos += [(d, m, True) for d, m in SINTETICOS]
    if opcao_m:
        for d, m in OPCAO_M:
            assert d in com_cov, f"{d} sem covariáveis no catálogo"
        conjuntos += [(d, m, False) for d, m in OPCAO_M if (d, m, False) not in conjuntos]
    tarefas, ing = [], []
    for ds, modo, sint in conjuntos:
        ing.append({"tipo": "ingenua", "dataset": ds, "modo": modo})
        ntr, _, _ = janelas_treino(ds, modo)
        so_m = opcao_m and (ds, modo) in OPCAO_M and not reais_m      # entra só pela opção M do §10
        for fr in (FRAC_OPCAO_M if so_m else (FRAC_SINT if sint else FRAC_REAL)):
            if fr == 0.25 and ds in sem_frac025:
                continue
            if so_frac025_reais and not sint and fr != 0.25:
                continue          # corte B (proposta; exige registro no §10): reais só com a fração 0,25
            if b2 and not sint and fr == 1.0 and ntr > LIMITE_FRAC1:
                continue          # corte B2 (§10): fração 1,0 dos reais só até LIMITE_FRAC1 janelas de treino
            for bb in (BB_OPCAO_M if so_m else BACKBONES):
                var = VAR_T1_SINT if sint else VAR_T1
                tarefas.append({"dataset": ds, "modo": modo, "backbone": bb, "largura": 1, "fracao": fr, "seed": 0,
                                "variantes": var, "n_tr_estimado": int(round(ntr * fr)),
                                "custo_rel": ntr * fr * cj.get(bb, 1.0) * (len(var) + 1)})
    tarefas.sort(key=lambda t: -t["custo_rel"])
    return ing + tarefas


def gerar_smoke():
    var = ["densa", "densa+E", "proj_K10", "conv_L1000_K10", "zero_L100_K10", "zero_L1000_K4"]
    conj = [("ETTh1", "U"), ("ETTh1", "M"), ("SINT-U48", "U")]
    out = [{"tipo": "ingenua", "dataset": d, "modo": m} for d, m in conj]
    for d, m in conj:
        for bb in BACKBONES:
            out.append({"dataset": d, "modo": m, "backbone": bb, "largura": 1, "fracao": 0.1, "seed": 0,
                        "variantes": var})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("quem", choices=["t1", "smoke"])
    ap.add_argument("--reais-m", action="store_true", help="inclui os reais no modo M (só se a pesquisa indicar)")
    ap.add_argument("--sem-frac025", nargs="*", default=[], help="corte do §7: conjuntos reais sem a fração 0,25")
    ap.add_argument("--so-frac025-reais", action="store_true",
                    help="corte B (proposta): reais só com fração 0,25; sintéticos com as três (registrar no §10)")
    ap.add_argument("--b2", action="store_true", help="corte B2 do §10 (reais: fração 0,25 e fração 1,0 até 12100 janelas)")
    ap.add_argument("--opcao-m", action="store_true",
                    help="opção M do §10: ETTh1-M e opsd_de_solar-M, fração 0,25, só iTransformer, TSMixer e PatchTST")
    ap.add_argument("--saida", default=None)
    a = ap.parse_args()
    tarefas = gerar_t1(a.reais_m, set(a.sem_frac025), a.so_frac025_reais, a.b2, a.opcao_m) if a.quem == "t1" else gerar_smoke()
    saida = Path(a.saida) if a.saida else RES / "filas" / f"{a.quem}.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    meta = {"fase": a.quem, "reais_m": a.reais_m, "sem_frac025": a.sem_frac025, "so_frac025_reais": a.so_frac025_reais, "b2": a.b2,
            "opcao_m": a.opcao_m, "limite_frac1": LIMITE_FRAC1 if a.b2 else None, "backbones": BACKBONES}
    saida.write_text(json.dumps({"meta": meta, "tarefas": tarefas}, indent=1, ensure_ascii=False))
    n_t = sum(1 for t in tarefas if t.get("tipo") != "ingenua")
    print(f"{saida}: {n_t} tarefas de treino + {len(tarefas) - n_t} de ingênuas")


if __name__ == "__main__":
    main()
