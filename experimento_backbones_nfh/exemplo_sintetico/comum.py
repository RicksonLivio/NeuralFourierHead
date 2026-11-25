"""Código comum do exemplo sintético: reutiliza (sem editar) o exemplo de livro do teste de exemplificação.

Série: exemplo_livro/gerar.py com sigma = 0,3. Protocolo de treino: exemplo_livro/treinar.py (Adam, taxa 3e-3, até 400
épocas, paciência 40, E = 50 épocas extras com taxa 3e-4, lote 64, colunas normalizadas por D, seleção segura).
Só CPU, 1 thread. Nada aqui escolhe nada pelo teste.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["LIVRO_SIGMA"] = "0.3"
os.environ["OMP_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True          # não gravar __pycache__ dentro de exemplo_livro/

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
LIVRO = RAIZ / "teste_exemplificacao_nfh" / "exemplo_livro"
sys.path.insert(0, str(LIVRO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
import treinar as tr  # noqa: E402  (exemplo_livro/treinar.py, só leitura)
from gerar import H, K, L, D, Q, SIGMA, bases, oraculo, particoes  # noqa: E402

assert SIGMA == 0.3
LAMBDA = 1000.0
VARIANTES = ["densa", "densa+E", "conv_L1000", "zero_L0", "zero_L1000"]


def dados():
    part = particoes()
    T = {k: (torch.tensor(v[0], dtype=torch.float32), torch.tensor(v[1], dtype=torch.float32)) for k, v in part.items()}
    return part, T


def mae(m, X, Y):
    with torch.no_grad():
        return float((m(X) - Y).abs().mean())


def metricas(m, T):
    return {"val_RMSE": tr.rmse(m, *T["val"]), "val_MAE": mae(m, *T["val"]),
            "test_RMSE": tr.rmse(m, *T["teste"]), "test_MAE": mae(m, *T["teste"])}


def treinar_variantes(p, n, semente, T, quais=VARIANTES):
    """Treina as variantes pedidas (a densa é sempre treinada, pois densa+E e conv partem dela).
    Devolve {variante: (modelo, epoca, metricas)}. Determinístico em CPU com 1 thread."""
    tr.P_LAT = p
    Xtr, Ytr = T["treino"][0][:n], T["treino"][1][:n]
    Xv, Yv = T["val"]
    out = {}
    torch.manual_seed(semente)
    densa, _, ep = tr.treinar(tr.Modelo("densa"), Xtr, Ytr, Xv, Yv, semente, tr.LR, tr.EPOCAS, paciencia=tr.PACIENCIA)
    out["densa"] = (densa, ep, metricas(densa, T))
    if "densa+E" in quais:
        de, _, ep = tr.treinar(__import__("copy").deepcopy(densa), Xtr, Ytr, Xv, Yv, semente, tr.LR_AJ, tr.E_EXTRA,
                               incluir_inicio=True)
        out["densa+E"] = (de, ep, metricas(de, T))
    if "conv_L1000" in quais:
        conv = tr.converter(densa)
        assert abs(tr.rmse(conv, *T["teste"]) - tr.rmse(densa, *T["teste"])) < 1e-5
        aj, ep = tr.ajuste_seguro(conv, Xtr, Ytr, Xv, Yv, semente, LAMBDA)
        out["conv_L1000"] = (aj, ep, metricas(aj, T))
    for nome, lam in (("zero_L0", 0.0), ("zero_L1000", LAMBDA)):
        if nome in quais:
            torch.manual_seed(semente)
            z, _, ep = tr.treinar(tr.Modelo("nfh"), Xtr, Ytr, Xv, Yv, semente, tr.LR, tr.EPOCAS, lam=lam,
                                  paciencia=tr.PACIENCIA)
            out[nome] = (z, ep, metricas(z, T))
    return out
