"""Verificação: prioridade da tendência na decomposição da NFH (Seções 4.3.5 e 5.11).

Como a base sazonal é ortogonalizada contra a tendência (Ft = F (I - Pi_T)), a parte
polinomial de cada harmônico passa à tendência T = Pi_T y. Para uma previsão que é uma
senoide pura, amplitude e fase continuam exatas (Teorema 4.6), mas parte da energia da
forma aparece em epsilon_T. Este script mede essa parte para senoides puras dos dois
primeiros harmônicos nas configurações do texto:

  * H = 48, q = 2, K = 10 e K = 2 (experimentos do Capítulo 5);
  * H = 8,  q = 1, K = 2         (exemplos do Capítulo 4).

Saída: results/v3/verif_prioridade_tendencia.json
"""
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from v2_common import nfh_bases, time_grid  # noqa: E402


def decompor(y, H, K, q):
    P, F, Ft, B, Qp = nfh_bases(H, K, q)
    PiT = P.T @ np.linalg.solve(P @ P.T, P)
    T = PiT @ y
    R = Qp @ (Qp.T @ y)
    S = y - T - R
    thS = np.linalg.solve(Ft @ Ft.T, Ft @ y)
    a, b = thS[0::2], thS[1::2]
    A = np.hypot(a, b)
    fase = np.arctan2(b, a)
    ybar = y.mean()
    yc, Tc = y - ybar, T - ybar
    E = float(yc @ yc)
    # participação de cada harmônico, e_k = <S_k, S>/||y_c||^2, com S_k = (I - Pi_T)(a_k c_k + b_k s_k)
    e = [float(((a[k] * Ft[2 * k] + b[k] * Ft[2 * k + 1]) @ S) / E) for k in range(K)]
    return {"eps_T": float(Tc @ Tc) / E, "eps_S": float(S @ S) / E, "eps_R": float(R @ R) / E,
            "A": A[:2].tolist(), "fase": fase[:2].tolist(), "e": e[:2],
            "erro_reconstrucao": float(np.abs(T + S + R - y).max())}


def main():
    out = {}
    for H, q, K in ((48, 2, 10), (48, 2, 2), (8, 1, 2)):
        tau = time_grid(H)
        cfg = {}
        for k in (1, 2):
            for nome, y in ((f"cos_k{k}", np.cos(2 * np.pi * k * tau)), (f"sen_k{k}", np.sin(2 * np.pi * k * tau))):
                cfg[nome] = decompor(y, H, K, q)
                cfg[nome]["periodo"] = H / k
        out[f"H{H}_q{q}_K{K}"] = cfg
    destino = RAIZ / "results" / "v3" / "verif_prioridade_tendencia.json"
    destino.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for c, v in out.items():
        for nome, r in v.items():
            print(c, nome, "periodo", r["periodo"], "eps_T %.4f eps_S %.4f eps_R %.2e" % (r["eps_T"], r["eps_S"], r["eps_R"]),
                  "A", np.round(r["A"], 6), "fase", np.round(r["fase"], 6), "e", np.round(r["e"], 4),
                  "erro %.1e" % r["erro_reconstrucao"])


if __name__ == "__main__":
    main()
