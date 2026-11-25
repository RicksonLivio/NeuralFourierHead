#!/usr/bin/env python3
"""
filas.py: gera a fila do controle das épocas extras.

Fila real (resultados/filas/controle.json): 24 pares (8 conjuntos x 3 backbones) x 5 sementes = 120 tarefas
{densa, densa+E, nfh} + 24 tarefas exata_s0 (densa da dissertação + densa+E, semente 0) = 144.
Cada tarefa leva o dispositivo da dissertação (results/v3/dispositivos.json, 'detalhe').

Ordem: primeiro o estágio da semente 0 (as 24 tarefas da semente 0 e as 24 exata_s0), depois as sementes 1, 2, 3 e 4.
Dentro de cada estágio, da mais cara para a mais barata (custo = tempo medido na dissertação:
tempo da densa + 2 x tempo do ajuste da NFH, em results/v3/runs; exata_s0 = tempo do ajuste). Assim a semente 0
de todos os pares termina antes de qualquer outra semente, e a cauda de cada estágio é curta.

Uso: python filas.py real [--saida resultados/filas/controle.json]
     python filas.py smoke  (resultados/smoke/<fase>/fila.json para as fases smoke e smoke_cpu)
     python filas.py repro  (resultados/repro/fila.json: um par MLP em CPU, semente 0, épocas reais)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
BASE = ROOT / "experimento_controle_epocas" / "resultados"
RES3 = ROOT / "results" / "v3"
FONTES = ["ETTh1", "ETTh2", "Jena", "AirQ"]
CONJUNTOS = [f"{f}-{s}" for f in FONTES for s in ("U", "M")]
BACKBONES = ["MLP", "LSTM", "Conv1D+LSTM"]
SEEDS = (0, 1, 2, 3, 4)
VAR_TREINO = ["densa", "densa+E", "nfh"]
VAR_EXATA = ["densa_diss", "densa+E"]


def dispositivos():
    return json.loads((RES3 / "dispositivos.json").read_text())["detalhe"]


def tempos():
    """{(conj, bb, seed): (tempo_densa_s, tempo_ajuste_nfh_s)} das linhas da dissertação."""
    out = {}
    for conj in CONJUNTOS:
        with open(RES3 / "runs" / f"{conj}.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["backbone"] not in BACKBONES or r["tempo_s"] == "":
                    continue
                k = (conj, r["backbone"], int(r["seed"]))
                a = out.setdefault(k, [0.0, 0.0])
                if r["modelo"].endswith("+ densa") and r["variante"] == "":
                    a[0] = float(r["tempo_s"])
                elif r["modelo"].endswith("+ NFH") and r["variante"] == "seguro" and r["K"] == "10":
                    a[1] = float(r["tempo_s"])
    return out


def gerar_real():
    disp, tp = dispositivos(), tempos()
    tarefas = []
    for conj in CONJUNTOS:
        for bb in BACKBONES:
            for s in SEEDS:
                td, ta = tp[(conj, bb, s)]
                tarefas.append({"tipo": "treino", "conjunto": conj, "backbone": bb, "seed": s,
                                "dispositivo": disp[conj][bb][str(s)], "variantes": VAR_TREINO,
                                "custo_s": td + 2 * ta, "estagio": s})
            td, ta = tp[(conj, bb, 0)]
            tarefas.append({"tipo": "exata_s0", "conjunto": conj, "backbone": bb, "seed": 0,
                            "dispositivo": disp[conj][bb]["0"], "variantes": VAR_EXATA, "custo_s": ta, "estagio": 0})
    tarefas.sort(key=lambda t: (t["estagio"], -t["custo_s"], t["conjunto"], t["backbone"], t["tipo"]))
    return tarefas


def gerar_smoke(dev_alvo):
    """Um par MLP e um LSTM no dispositivo pedido ('GPU': ETTh2-U; 'CPU': AirQ-U), semente 0 e exata_s0."""
    disp = dispositivos()
    conj = "ETTh2-U" if dev_alvo == "GPU" else "AirQ-U"
    out = []
    for bb in ("MLP", "LSTM"):
        assert disp[conj][bb]["0"] == dev_alvo, (conj, bb, disp[conj][bb]["0"])
        out.append({"tipo": "treino", "conjunto": conj, "backbone": bb, "seed": 0, "dispositivo": dev_alvo,
                    "variantes": VAR_TREINO, "custo_s": 0, "estagio": 0})
        out.append({"tipo": "exata_s0", "conjunto": conj, "backbone": bb, "seed": 0, "dispositivo": dev_alvo,
                    "variantes": VAR_EXATA, "custo_s": 0, "estagio": 0})
    return out


def gerar_repro():
    """Reprodução com épocas reais: AirQ-U + MLP, semente 0, CPU (par pequeno; na dissertação: CPU)."""
    disp = dispositivos()
    assert disp["AirQ-U"]["MLP"]["0"] == "CPU"
    return [{"tipo": "treino", "conjunto": "AirQ-U", "backbone": "MLP", "seed": 0, "dispositivo": "CPU",
             "variantes": VAR_TREINO, "custo_s": 0, "estagio": 0},
            {"tipo": "exata_s0", "conjunto": "AirQ-U", "backbone": "MLP", "seed": 0, "dispositivo": "CPU",
             "variantes": VAR_EXATA, "custo_s": 0, "estagio": 0}]


def escrever(caminho, meta, tarefas):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps({"meta": meta, "tarefas": tarefas}, indent=1, ensure_ascii=False))
    n_t = sum(t["tipo"] == "treino" for t in tarefas)
    print(f"{caminho}: {n_t} tarefas de treino + {len(tarefas) - n_t} exata_s0")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("quem", choices=["real", "smoke", "repro"])
    ap.add_argument("--saida", default=None)
    a = ap.parse_args()
    if a.quem == "real":
        t = gerar_real()
        escrever(Path(a.saida) if a.saida else BASE / "filas" / "controle.json",
                 {"fase": "controle", "n": len(t)}, t)
        for d in ("GPU", "CPU"):
            c = sum(x["custo_s"] for x in t if x["dispositivo"] == d)
            n = sum(x["dispositivo"] == d for x in t)
            print(f"  {d}: {n} tarefas; custo da dissertação (densa + 2 x ajuste; soma sob a carga da época) = {c / 3600:.1f} h")
    elif a.quem == "smoke":
        escrever(BASE / "smoke" / "smoke" / "fila.json", {"fase": "smoke"}, gerar_smoke("GPU"))
        escrever(BASE / "smoke" / "smoke_cpu" / "fila.json", {"fase": "smoke_cpu"}, gerar_smoke("CPU"))
    else:
        escrever(BASE / "repro" / "fila.json", {"fase": "repro"}, gerar_repro())


if __name__ == "__main__":
    sys.exit(main())
