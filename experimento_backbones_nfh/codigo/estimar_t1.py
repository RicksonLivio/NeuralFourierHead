#!/usr/bin/env python3
"""
estimar_t1.py: estima o tempo de relógio da T1 (ou de qualquer fila) por simulação da reivindicação dinâmica.

Entradas: resultados/tempo_epoca.json (medir_tempo.py, processo sozinho), arquivos opcionais
resultados/tempo_epoca_carga_*.json (mesma medição com todos os workers rodando ao mesmo tempo; dão o fator de
carga por dispositivo e backbone), a fila (padrão resultados/filas/t1.json) e o número de épocas dos treinos.

Custo de uma tarefa (em épocas-equivalentes) = E*(1 + n_zero) + E_extra + E_ajuste, com
  E = épocas de cada treino do zero (densa e zero_*): 'típica' = mediana de `epocas` da densa do MLP em
      teste_exemplificacao_nfh/resultados/triagem.csv (única referência de épocas disponível); 'pior caso' = 60;
  n_zero = nº de variantes zero_* da tarefa; densa+E = 15 épocas; conv = 15 épocas.
Tempo da tarefa = custo * s_por_janela_epoca[dispositivo, backbone, largura, d] * n_tr * fator_carga + 3 s.
A simulação despacha as tarefas na ordem da fila para o primeiro worker livre (como executar.py).

Uso: python estimar_t1.py [--fila ...] [--gpu 3] [--cpu 6] [--sem-frac025 DATASET ...] [--saida estimativa_t1.json]
"""
from __future__ import annotations

import argparse
import heapq
import json
import statistics
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
RES = ROOT / "experimento_backbones_nfh" / "resultados"


def mediana_epocas_densa():
    d = pd.read_csv(ROOT / "teste_exemplificacao_nfh" / "resultados" / "triagem.csv")
    return float(d[(d["variante"] == "densa") & (d["backbone"] == "MLP")]["epocas"].median())


def carregar_tempos():
    sozinho = json.loads((RES / "tempo_epoca.json").read_text())["medidas"]
    base = {(m["dispositivo"], m["backbone"], m["largura"], 1 if m["d"] == 1 else 5): m["s_por_janela_epoca"]
            for m in sozinho}
    fator = {}
    for f in sorted(RES.glob("tempo_epoca_carga_*.json")):
        for m in json.loads(f.read_text())["medidas"]:
            k = (m["dispositivo"], m["backbone"], m["largura"], 1 if m["d"] == 1 else 5)
            if k in base:
                fator.setdefault(k, []).append(m["s_por_janela_epoca"] / base[k])
    fator = {k: statistics.mean(v) for k, v in fator.items()}
    return base, fator


def tempo_tarefa(t, dev, E, base, fator):
    d = 1 if t["modo"] == "U" else 5
    k = (dev, t["backbone"], t["largura"], d)
    n_zero = sum(1 for v in t["variantes"] if v.startswith("zero_"))
    custo = E * (1 + n_zero) + 15 + 15
    f = fator.get(k)
    if f is None:                       # sem medida sob carga deste backbone: média do dispositivo
        fs = [x for (dv, *_), x in fator.items() if dv == dev]
        f = statistics.mean(fs) if fs else 1.0
    return custo * base[k] * t["n_tr_estimado"] * f + 3.0


def simular(fila, n_gpu, n_cpu, E, base, fator):
    tarefas = [t for t in fila if t.get("tipo") != "ingenua"]
    workers = [(0.0, i, "cuda") for i in range(n_gpu)] + [(0.0, n_gpu + j, "cpu") for j in range(n_cpu)]
    heapq.heapify(workers)
    fim, feitas = 0.0, {"cuda": 0, "cpu": 0}
    for t in tarefas:
        livre, i, dev = heapq.heappop(workers)
        termina = livre + tempo_tarefa(t, dev, E, base, fator)
        feitas[dev] += 1
        fim = max(fim, termina)
        heapq.heappush(workers, (termina, i, dev))
    return fim / 3600.0, feitas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fila", default=str(RES / "filas" / "t1.json"))
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--cpu", type=int, default=6)
    ap.add_argument("--sem-frac025", nargs="*", default=[])
    ap.add_argument("--saida", default=None)
    a = ap.parse_args()
    fila = json.loads(Path(a.fila).read_text())["tarefas"]
    fila = [t for t in fila if not (t.get("fracao") == 0.25 and t.get("dataset") in set(a.sem_frac025))]
    base, fator = carregar_tempos()
    E_tip = mediana_epocas_densa()
    out = {"fila": a.fila, "n_tarefas": sum(1 for t in fila if t.get("tipo") != "ingenua"), "gpu": a.gpu, "cpu": a.cpu,
           "sem_frac025": a.sem_frac025, "E_tipica": E_tip, "fatores_de_carga": {str(k): v for k, v in fator.items()},
           "cenarios": {}}
    for nome, E in (("tipico", E_tip), ("pior_caso", 60.0)):
        h, feitas = simular(fila, a.gpu, a.cpu, E, base, fator)
        h_gpu, _ = simular(fila, a.gpu, 0, E, base, fator)
        out["cenarios"][nome] = {"E": E, "horas": h, "tarefas_por_dispositivo": feitas, "horas_so_gpu": h_gpu}
        print(f"{nome}: E={E:g} -> {h:.2f} h com {a.gpu} GPU + {a.cpu} CPU "
              f"(GPU {feitas['cuda']} tarefas, CPU {feitas['cpu']}); só GPU: {h_gpu:.2f} h")
    if a.saida:
        Path(a.saida).write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
