#!/usr/bin/env python3
"""
medir_memoria.py: pico de memória da GPU e val_RMSE de `train_model` (original, validação inteira de uma vez) contra
`train_model_blocos` (validação em blocos de 1024), no mesmo conjunto, backbone, semente e número de épocas.

Uso: python medir_memoria.py <backbone> <orig|blocos> [--dataset opsd_de_solar] [--fracao 0.25] [--epocas 3] [--cpu]
Imprime uma linha JSON: pico alocado/reservado (MiB), val_RMSE por época (17 algarismos) e n_val.
Cada medição deve rodar num processo próprio (o pico é por processo).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import executar as X  # noqa: E402
from backbones import construir_modelo  # noqa: E402
from treino_bb import train_model_blocos  # noqa: E402
from v2_common import set_seed, train_model  # noqa: E402
from v3_common import CFG3  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone")
    ap.add_argument("versao", choices=["orig", "blocos"])
    ap.add_argument("--dataset", default="opsd_de_solar")
    ap.add_argument("--fracao", type=float, default=0.25)
    ap.add_argument("--epocas", type=int, default=3)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cpu" if a.cpu else "cuda")
    torch.set_num_threads(2)
    cfg = replace(CFG3, max_epochs=a.epocas, patience=99)
    pac = X.dados(a.dataset, "U")
    tr, va = X.treino_fracao(pac[1], a.fracao), pac[2]
    set_seed(0)
    m = construir_modelo(a.backbone, "densa", 1, 1)
    f = train_model if a.versao == "orig" else train_model_blocos
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    m, info = f(m, tr, va, 0, lam=0.0, cfg=cfg, dev=dev)
    r = {"backbone": a.backbone, "versao": a.versao, "dataset": a.dataset, "n_tr": len(tr), "n_val": len(va),
         "epocas": a.epocas, "val_MSE_por_epoca": [repr(h[1]) for h in info["history"]]}
    if dev.type == "cuda":
        r["pico_alocado_MiB"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        r["pico_reservado_MiB"] = round(torch.cuda.max_memory_reserved() / 2 ** 20)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
