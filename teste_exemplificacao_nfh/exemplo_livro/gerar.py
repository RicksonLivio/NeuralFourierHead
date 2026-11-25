"""Exemplo sintético "de livro" da NFH (versão 2 do exemplo do teste de exemplificação).

Por que existe uma versão 2: na primeira versão (pasta exemplo_sintetico/), o futuro da série não era previsível a partir
da janela (os coeficientes dependiam de valores futuros de passeios aleatórios) e a "tendência" era um dente de serra
de período 8; os modelos ficavam com RMSE cinco vezes o piso de ruído. Aqui o futuro de cada janela é, por construção,
uma função LINEAR conhecida da janela e fica (quase) inteiro no espaço estrutural T + S da NFH.

Série (passo de tempo s; H = 8, q = 1, K = 2 com períodos 8 e 4; L = 16; d = 4 entradas):
  x1 (condutor da tendência)  = 1 + 0,6 sen(2 pi s / 480 + 0,4) + 0,3 sen(2 pi s / 1300 + 1,1)      (lento)
  x2 (condutor do ciclo)      = A_s [cos(2 pi s / 8 + 0,7) + 0,3 cos(2 pi s / 4 - 0,5)],
                                A_s = 1 + 0,4 sen(2 pi s / 700 + 2)                              (amplitude lenta)
  x3 (distrator lento)        = 0,8 sen(2 pi s / 600 + 2,5) + 0,4 sen(2 pi s / 1700)
  x4 (distrator oscilante)    = B_s cos(2 pi s / 8 + 2,6),  B_s = 1 + 0,4 sen(2 pi s / 900)
  alvo  y_t = x1_{t-8} + x2_{t-8} + e_t,  e_t ~ N(0, SIGMA^2) independente;  y NÃO é entrada.
Como o atraso (8) é igual ao horizonte, y_{t+1..t+8} = x1 + x2 nas 8 últimas posições da janela + ruído:
o oráculo E[y | janela] é a soma das duas últimas colunas-bloco, e seu RMSE é SIGMA. Sobre 8 passos, x1 é quase uma reta
(entra em T) e x2 é quase uma soma exata de cossenos de períodos 8 e 4 (entra em T + S); a média de x2 num período
inteiro é zero, então o nível previsto g_T depende só de x1, e a energia sazonal g_S, de x2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from v2_common import nfh_bases  # noqa: E402

H, Q, K, L, D = 8, 1, 2, 16, 4
import os
SIGMA = float(os.environ.get("LIVRO_SIGMA", "0.1"))
N = 9000
SEMENTE_DADOS = 2026


def gerar(n=N, sigma=SIGMA, semente=SEMENTE_DADOS):
    s = np.arange(n, dtype=np.float64)
    x1 = 1 + 0.6 * np.sin(2 * np.pi * s / 480 + 0.4) + 0.3 * np.sin(2 * np.pi * s / 1300 + 1.1)
    A = 1 + 0.4 * np.sin(2 * np.pi * s / 700 + 2)
    x2 = A * (np.cos(2 * np.pi * s / 8 + 0.7) + 0.3 * np.cos(2 * np.pi * s / 4 - 0.5))
    x3 = 0.8 * np.sin(2 * np.pi * s / 600 + 2.5) + 0.4 * np.sin(2 * np.pi * s / 1700)
    B = 1 + 0.4 * np.sin(2 * np.pi * s / 900)
    x4 = B * np.cos(2 * np.pi * s / 8 + 2.6)
    X = np.stack([x1, x2, x3, x4], 1)
    limpo = np.full(n, np.nan)
    limpo[H:] = x1[:-H] + x2[:-H]
    rng = np.random.default_rng(semente)
    y = limpo + rng.normal(0, sigma, n)
    return X, y, limpo


def janelas(X, y, limpo, ini, fim):
    """Janelas (X_{i..i+L-1}, y_{i+L..i+L+H-1}) com o início do horizonte em [ini + L, fim - H]."""
    idx = np.arange(max(ini, H), fim - L - H + 1)
    Xw = np.stack([X[i:i + L] for i in idx])
    Yw = np.stack([y[i + L:i + L + H] for i in idx])
    Cw = np.stack([limpo[i + L:i + L + H] for i in idx])
    return Xw, Yw, Cw


def particoes():
    X, y, limpo = gerar()
    a, b = int(0.6 * N), int(0.8 * N)
    return {"treino": janelas(X, y, limpo, 0, a), "val": janelas(X, y, limpo, a, b),
            "teste": janelas(X, y, limpo, b, N)}


def oraculo(Xw):
    """Previsão ótima: x1 + x2 nas 8 últimas posições da janela (sem ruído)."""
    return Xw[:, -H:, 0] + Xw[:, -H:, 1]


def bases():
    P, F, Ft, B, Qp = nfh_bases(H, K, Q)
    M = np.concatenate([P.T, Ft.T, Qp], 1)
    return P, F, Ft, Qp, M


if __name__ == "__main__":
    part = particoes()
    P, F, Ft, Qp, M = bases()
    for nome, (Xw, Yw, Cw) in part.items():
        Ot = oraculo(Xw)
        assert np.allclose(Ot, Cw)
        R = Ot @ (Qp @ Qp.T).T
        Oc = Ot - Ot.mean(1, keepdims=True)
        kappa = 1 - (R ** 2).sum(1) / (Oc ** 2).sum(1)
        print(nome, Xw.shape, "RMSE do oráculo", float(np.sqrt(np.mean((Ot - Yw) ** 2))),
              "cobertura do oráculo: mediana", float(np.median(kappa)), "mínimo", float(kappa.min()))
    json.dump({"H": H, "q": Q, "K": K, "L": L, "d": D, "sigma": SIGMA, "N": N, "semente_dados": SEMENTE_DADOS,
               "cortes": [0.6, 0.8]}, open(AQUI / "config.json", "w"), indent=1)
