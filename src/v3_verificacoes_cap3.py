"""Versão 3, Cap. 3: verificação computacional das afirmações novas sobre N-BEATS-I e NFT.

1. N-BEATS-I (bases de ORESHKIN et al., 2020, com floor(H/2-1) harmônicos, como em
   src/v2_common.nbeats_seasonal_basis): dimensão da interseção entre o espaço de tendência
   T_nb = Img([1, tau, ..., tau^p]) e o sazonal S_nb, comparada com a fórmula do Corolário
   (H par: dim = p_nb), e o vetor compartilhado tau*(tau - (H-1)/H) para p_nb = 2.
2. NFT (KOREN; RADINSKY, 2024, Eqs. 4 e 5): a matriz constante 1_M 1_H^T pertence tanto ao
   espaço do bloco de tendência {A P} quanto ao do bloco sazonal {F_M^T C F_H}; as linhas
   sin(2 pi 0 .) de F_M e F_H são nulas (e sin(2 pi (M/2) .) de F_M, com M par).
Saída: results/v3/verif_cap3.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from v2_common import nbeats_seasonal_basis  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAIDA = ROOT / "results" / "v3" / "verif_cap3.json"
TOL = 1e-9


def posto(A):
    return int(np.linalg.matrix_rank(A, tol=TOL * max(A.shape) * max(1.0, np.abs(A).max())))


def base_tendencia(H, p):
    tau = np.arange(H) / H
    return np.stack([tau ** i for i in range(p + 1)], 1)          # H x (p+1)


def dim_intersecao(A, B):
    return posto(A) + posto(B) - posto(np.concatenate([A, B], 1))


def nbeats():
    linhas = []
    for H in (6, 7, 8, 13, 14, 18, 30, 31, 48):
        S = nbeats_seasonal_basis(H).T                            # H x (1 + 2 floor(H/2-1))
        for p in (1, 2, 3):
            T = base_tendencia(H, p)
            d = dim_intersecao(T, S)
            previsto = p if H % 2 == 0 else None
            linhas.append({"H": H, "p_nb": p, "dim_S": posto(S), "dim_T": posto(T), "dim_intersecao": d,
                           "formula_H_par": previsto, "confere": (previsto is None) or (d == previsto)})
    # Vetor compartilhado para p_nb = 2 e H par: v = tau (tau - (H-1)/H), ortogonal ao vetor alternado
    compart = []
    for H in (8, 30, 48):
        tau = np.arange(H) / H
        v = tau * (tau - (H - 1) / H)
        S = nbeats_seasonal_basis(H).T
        coef, *_ = np.linalg.lstsq(S, v, rcond=None)
        alt = (-1.0) ** np.arange(H)
        compart.append({"H": H, "erro_v_em_S": float(np.abs(S @ coef - v).max()),
                        "prod_tau_alternado": float(tau @ alt), "prod_1_alternado": float(np.ones(H) @ alt)})
    return linhas, compart


def nft(M=4, H=8, N=4, grau=2):
    """Bases do NFT conforme exibidas no artigo: cossenos e senos para k = 0, ..., M/2 (F_M) e
    k = 0, ..., N/2 (F_H); tendência A P com P de Vandermonde (potências 0..grau)."""
    def fourier(n_freq_meio, n_pts):
        t = np.arange(n_pts) / n_pts
        ks = np.arange(n_freq_meio + 1)
        return np.concatenate([np.cos(2 * np.pi * np.outer(ks, t)), np.sin(2 * np.pi * np.outer(ks, t))], 0)

    FM = fourier(M // 2, M)                                      # (M+2) x M, como exibida
    FH = fourier(N // 2, H)                                      # (N+2) x H
    t = np.arange(H) / H
    P = np.stack([t ** i for i in range(grau + 1)], 0)           # (grau+1) x H
    # Espaço sazonal: {FM^T C FH}, vetorizado por kron; espaço de tendência: {A P}
    Bsaz = np.kron(FM.T, FH.T)                                   # vec(FM^T C FH) = (FM^T kron FH^T) vec(C) (ordem por linhas)
    Btend = np.kron(np.eye(M), P.T)                              # vec(A P) por linhas
    um = np.ones(M * H)
    c_saz, *_ = np.linalg.lstsq(Bsaz, um, rcond=None)
    c_tend, *_ = np.linalg.lstsq(Btend, um, rcond=None)
    linhas_nulas_FM = [int(i) for i in np.where(np.abs(FM).max(1) < 1e-12)[0]]
    linhas_nulas_FH = [int(i) for i in np.where(np.abs(FH).max(1) < 1e-12)[0]]
    return {"M": M, "H": H, "N": N, "grau_tendencia": grau,
            "formato_FM": list(FM.shape), "formato_FH": list(FH.shape),
            "erro_1_no_sazonal": float(np.abs(Bsaz @ c_saz - um).max()),
            "erro_1_na_tendencia": float(np.abs(Btend @ c_tend - um).max()),
            "dim_intersecao": dim_intersecao(Btend, Bsaz),
            "posto_sazonal": posto(Bsaz), "colunas_sazonal": Bsaz.shape[1],
            "linhas_nulas_FM": linhas_nulas_FM, "linhas_nulas_FH": linhas_nulas_FH}


def main():
    linhas, compart = nbeats()
    res = {"nbeats_intersecao": linhas, "nbeats_vetor_compartilhado": compart, "nft": nft()}
    assert all(r["confere"] for r in linhas), "fórmula do Corolário não confere"
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for r in linhas:
        print(r)
    print(compart)
    print(res["nft"])


if __name__ == "__main__":
    main()
