"""Verificação numérica da forma real da DFT (Cap. 2) e da nova demonstração real do
Lema de independência linear da base (Cap. 4) e do Corolário da dimensão da interseção
no N-BEATS-I (Cap. 3).

Saída: results/v3/verif_dft_real.json

Itens verificados (todos com tolerância em ponto flutuante float64):
  1. eq:dft + eq:dft-inversa: reconstrução exata para N par e ímpar;
  2. Parseval na forma real;
  3. Lema das somas C(a), D(a);
  4. Corolário de ortogonalidade (inclusive o vetor de Nyquist c_{N/2});
  5. eq:energia-harmonico;
  6. Δ[A cos(ωj-φ)] = 2 sin(ω/2) A cos(ωj - φ + ω/2 + π/2);
  7. recorrência do polinômio χ(x) = Π (x² - 2cos ω_k x + 1), mônico de grau 2K;
  8. posto completo de B = [P^T F^T] (base do código) para m <= H;
  9. Corolário 3.3: a = c_{H/2} ortogonal às colunas sazonais do N-BEATS-I (H par).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from v2_common import fourier_basis, poly_basis  # noqa: E402

SAIDA = RAIZ / "results" / "v3" / "verif_dft_real.json"
rng = np.random.default_rng(2026)


def dft_real(x):
    N = len(x)
    n = np.arange(N)
    a0 = x.mean()
    ks = [k for k in range(1, N) if 2 * k < N]
    a = {k: 2 / N * np.sum(x * np.cos(2 * np.pi * k * n / N)) for k in ks}
    b = {k: 2 / N * np.sum(x * np.sin(2 * np.pi * k * n / N)) for k in ks}
    aN = (1 / N) * np.sum(x * np.cos(np.pi * n)) if N % 2 == 0 else 0.0
    return a0, a, b, aN


def inversa_real(a0, a, b, aN, N):
    n = np.arange(N)
    x = np.full(N, a0, dtype=float)
    for k in a:
        x += a[k] * np.cos(2 * np.pi * k * n / N) + b[k] * np.sin(2 * np.pi * k * n / N)
    if N % 2 == 0:
        x += aN * np.cos(np.pi * n)
    return x


def main():
    out = {}

    # 1-2. reconstrução e Parseval
    err_rec, err_pars = 0.0, 0.0
    for N in (1, 2, 3, 7, 8, 48, 95, 96):
        for _ in range(20):
            x = rng.normal(size=N)
            a0, a, b, aN = dft_real(x)
            err_rec = max(err_rec, np.max(np.abs(inversa_real(a0, a, b, aN, N) - x)))
            energia = N * a0 ** 2 + N / 2 * sum(a[k] ** 2 + b[k] ** 2 for k in a) + (N * aN ** 2 if N % 2 == 0 else 0)
            err_pars = max(err_pars, abs(energia - np.sum(x ** 2)) / max(1.0, np.sum(x ** 2)))
    out["reconstrucao_erro_max"] = float(err_rec)
    out["parseval_erro_rel_max"] = float(err_pars)

    # 3. somas C(a), D(a)
    err_C, err_D = 0.0, 0.0
    for N in range(1, 25):
        n = np.arange(N)
        for aa in range(-60, 61):
            C = np.sum(np.cos(2 * np.pi * aa * n / N))
            D = np.sum(np.sin(2 * np.pi * aa * n / N))
            alvo = N if aa % N == 0 else 0
            err_C, err_D = max(err_C, abs(C - alvo)), max(err_D, abs(D))
    out["somas_C_erro_max"], out["somas_D_erro_max"] = float(err_C), float(err_D)

    # 4. Gram da base real (inclusive Nyquist) = diag(N, N/2, ..., N/2, [N])
    err_gram = 0.0
    for N in (2, 5, 6, 7, 8, 30, 48, 96):
        n = np.arange(N)
        cols, normas = [np.ones(N)], [N]
        for k in range(1, N):
            if 2 * k < N:
                cols += [np.cos(2 * np.pi * k * n / N), np.sin(2 * np.pi * k * n / N)]
                normas += [N / 2, N / 2]
        if N % 2 == 0:
            cols.append(np.cos(np.pi * n))
            normas.append(N)
        G = np.stack(cols, 1)
        assert G.shape == (N, N)
        err_gram = max(err_gram, np.max(np.abs(G.T @ G - np.diag(normas))))
    out["gram_base_real_erro_max"] = float(err_gram)

    # 5. energia de um harmônico
    err_en = 0.0
    for N in (7, 48):
        n = np.arange(N)
        for k in range(1, (N + 1) // 2):
            if 2 * k >= N:
                continue
            aa, bb = rng.normal(size=2)
            v = aa * np.cos(2 * np.pi * k * n / N) + bb * np.sin(2 * np.pi * k * n / N)
            err_en = max(err_en, abs(v @ v - N / 2 * (aa ** 2 + bb ** 2)))
    out["energia_harmonico_erro_max"] = float(err_en)

    # 6. operador de diferença sobre uma senoide
    err_delta = 0.0
    j = np.arange(200)
    for _ in range(200):
        w = rng.uniform(1e-3, np.pi - 1e-3)
        A, phi = rng.uniform(0.1, 3), rng.uniform(-np.pi, np.pi)
        u = A * np.cos(w * (j + 1) - phi) - A * np.cos(w * j - phi)
        alvo = 2 * np.sin(w / 2) * A * np.cos(w * j - phi + w / 2 + np.pi / 2)
        err_delta = max(err_delta, np.max(np.abs(u - alvo)))
    out["delta_senoide_erro_max"] = float(err_delta)

    # 7. recorrência de chi: monic, grau 2K, anula somas de senoides
    err_rec2, monico = 0.0, True
    for H in (12, 48):
        for K in (1, 2, 4, 10):
            if 2 * K >= H:
                continue
            ws = 2 * np.pi * np.arange(1, K + 1) / H
            chi = np.array([1.0])
            for w in ws:
                chi = np.convolve(chi, np.array([1.0, -2 * np.cos(w), 1.0]))  # coef. do maior grau primeiro
            rho = chi[::-1]                                                     # rho_0..rho_2K
            monico &= bool(len(rho) == 2 * K + 1 and abs(rho[-1] - 1) < 1e-15 and abs(rho[0] - 1) < 1e-12)
            jj = np.arange(4 * H)
            A, phi = rng.uniform(0.1, 2, K), rng.uniform(-np.pi, np.pi, K)
            vt = sum(A[k] * np.cos(ws[k] * jj - phi[k]) for k in range(K))
            res = [np.dot(rho, vt[t:t + 2 * K + 1]) for t in range(len(jj) - 2 * K)]
            err_rec2 = max(err_rec2, float(np.max(np.abs(res))))
    out["recorrencia_chi_residuo_max"] = err_rec2
    out["recorrencia_chi_monico"] = monico

    # 8. posto de B = [P^T F^T] (bases do código) para todo (H, q, K) com m <= H, H <= 40,
    #    e nas configurações experimentais (H = 48, q = 2, K em {2, 4, 10})
    falhas, testados = [], 0
    for H in range(2, 41):
        for q in range(0, 4):
            for K in range(1, H):
                m = q + 1 + 2 * K
                if m > H:
                    continue
                B = np.concatenate([poly_basis(H, q), fourier_basis(H, K)], 0).T
                testados += 1
                if np.linalg.matrix_rank(B) != m:
                    falhas.append((H, q, K))
    exper = {}
    for K in (2, 4, 10):
        B = np.concatenate([poly_basis(48, 2), fourier_basis(48, K)], 0).T
        s = np.linalg.svd(B, compute_uv=False)
        exper[f"K{K}"] = {"m": int(B.shape[1]), "posto": int(np.linalg.matrix_rank(B)),
                          "menor_valor_singular": float(s.min())}
    out["posto_B"] = {"configuracoes_testadas": testados, "falhas": falhas, "experimentos_H48_q2": exper}

    # 9. Corolário 3.3: a = c_{H/2} ortogonal às colunas sazonais do N-BEATS-I
    err_nb = 0.0
    for H in (6, 8, 14, 18, 30, 48):
        tau = np.arange(H) / H
        a = (-1.0) ** np.arange(H)
        cols = [np.ones(H)] + [f(2 * np.pi * k * tau) for k in range(1, H // 2) for f in (np.cos, np.sin)]
        err_nb = max(err_nb, max(abs(a @ c) for c in cols))
    out["cor_nbeats_a_ortogonal_erro_max"] = float(err_nb)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
