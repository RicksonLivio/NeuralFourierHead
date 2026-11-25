"""Verifica, num vetor de previsão com H = 48 passos e período s = 24, o alcance da decomposição clássica
por média móvel centrada 2 x 24 (Hyndman e Athanasopoulos) comparada à projeção ortogonal da NFH, para
responder à pergunta "por que não aplicar uma decomposição tradicional sobre a previsão?".
Usa a previsão de teste da NFH Conv1D+LSTM (semente 0, Jena-U, janela central). Implementação direta em
numpy (statsmodels não está instalado no ambiente).
Saída: results/v3/verif_decomp_classica.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))


def media_movel_2xs(y, s):
    """Média móvel centrada 2 x s (s par): pesos [1/2, 1, ..., 1, 1/2] / s sobre s + 1 pontos."""
    w = np.r_[0.5, np.ones(s - 1), 0.5] / s
    t = np.full(len(y), np.nan)
    h = s // 2
    for j in range(h, len(y) - h):
        t[j] = np.dot(w, y[j - h:j + h + 1])
    return t


def main():
    from v3_common import Projetor

    P = np.load(RAIZ / "results" / "v3" / "previsoes" / "Jena-U__Conv1D+LSTM+NFH__s0.npy")
    y = P[len(P) // 2].astype(float)
    H, s = len(y), 24
    t = media_movel_2xs(y, s)
    out = {"H": H, "periodo": s, "classica_passos_sem_tendencia": int(np.isnan(t).sum()),
           "classica_fracao_sem_tendencia": float(np.isnan(t).mean())}
    pr = Projetor(H, 10, 2)
    T, S, R = pr.decompor(y[None, :])
    yc, Tc = y - y.mean(), T[0] - T[0].mean()
    out["projecao_erro_reconstrucao"] = float(np.abs(T[0] + S[0] + R[0] - y).max())
    out["projecao_soma_participacoes"] = float((Tc @ Tc + S[0] @ S[0] + R[0] @ R[0]) / (yc @ yc))
    z = np.random.default_rng(0).normal(size=H)
    Tz = pr.decompor(z[None, :])[0][0]
    Tyz = pr.decompor((y + z)[None, :])[0][0]
    out["projecao_desvio_linearidade"] = float(np.abs(Tyz - T[0] - Tz).max())
    (RAIZ / "results" / "v3" / "verif_decomp_classica.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
