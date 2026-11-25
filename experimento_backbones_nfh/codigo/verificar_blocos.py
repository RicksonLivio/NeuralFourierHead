#!/usr/bin/env python3
"""
verificar_blocos.py: para o MESMO modelo treinado, compara o val_RMSE com a validação inteira de uma vez
(como em `train_model`) e em blocos de 1024 (`treino_bb.saida_em_blocos`). Isola o efeito da mudança de validação do ruído de treino da GPU (o treino
não é determinístico entre execuções).

Uso: python verificar_blocos.py [--dataset opsd_de_solar] [--cpu] [--backbones ...]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import executar as X  # noqa: E402
from backbones import REGISTRO, construir_modelo  # noqa: E402
from treino_bb import saida_em_blocos, train_model_blocos  # noqa: E402
from v3_common import CFG3  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--backbones", nargs="+", default=list(REGISTRO))
    a = ap.parse_args()
    dev = torch.device("cpu" if a.cpu else "cuda")
    torch.set_num_threads(2)
    pac = X.dados(a.dataset, "U")
    tr, va = X.treino_fracao(pac[1], 0.1), pac[2]
    Xv, Yv = pac[4].to(dev), pac[5].double().numpy()
    cfg = replace(CFG3, max_epochs=1, patience=99)
    pior = 0.0
    for bb in a.backbones:
        m = construir_modelo(bb, "densa", pac[0]["feats"].shape[1], 1)
        m, _ = train_model_blocos(m, tr, va, 0, lam=0.0, cfg=cfg, dev=dev)
        with torch.no_grad():
            inteira = m(Xv)["forecast"].double().cpu().numpy()
            blocos = saida_em_blocos(m, Xv)["forecast"].double().cpu().numpy()
        r1 = float(np.sqrt(np.mean((inteira - Yv) ** 2)))
        r2 = float(np.sqrt(np.mean((blocos - Yv) ** 2)))
        pior = max(pior, abs(r1 - r2))
        print(f"{bb:12s} n_val={len(Xv)} val_RMSE inteira={r1:.9f} blocos={r2:.9f} dif={abs(r1 - r2):.2e} "
              f"max|dif previsão|={np.abs(inteira - blocos).max():.2e}")
    print(f"maior diferença de val_RMSE: {pior:.2e}")


if __name__ == "__main__":
    main()
