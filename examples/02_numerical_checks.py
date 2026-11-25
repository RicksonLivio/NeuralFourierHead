"""Example 02: numerical checks of Lemma 1, Theorem 1 and Proposition 1 of the paper.

Lemma 1: if m = q + 1 + 2K <= H, the columns of [P^T F^T] are linearly independent.
Theorem 1 (identifiability): T, S, R are pairwise orthogonal, R^H = T + S + R and the representation of every
    forecast is unique, with theta_T = (P P^T)^-1 P y, theta_S = (F~ F~^T)^-1 F~ y, u = Q^T y.
Proposition 1 (N-BEATS-I): for even H > p_nb >= 1 the trend and seasonal spans of N-BEATS-I intersect in a
    subspace of dimension p_nb, so the split trend + seasonal is not unique.

    uv run python examples/02_numerical_checks.py
"""
from __future__ import annotations

import numpy as np
import torch

from _toy import check
from nfh import NeuralFourierHead, build_bases, check_dimensions

FAILED = []


def ok(passed: bool) -> None:
    if not passed:
        FAILED.append(1)


def nbeats_i_bases(H: int, p_nb: int):
    """N-BEATS-I trend [1, tau, ..., tau^p_nb] and seasonal [1, cos(2 pi i tau), sin(2 pi i tau)], i = 1..floor(H/2 - 1)
    (Oreshkin et al., 2020, eq. 3). Same bases as ``src/v2_common.py::nbeats_seasonal_basis``."""
    tau = np.arange(H) / H
    trend = np.stack([tau ** i for i in range(p_nb + 1)], 1)                      # H x (p_nb + 1)
    kmax = int(np.floor(H / 2 - 1))
    seas = np.stack([np.ones(H)] + [np.cos(2 * np.pi * i * tau) for i in range(1, kmax + 1)]
                    + [np.sin(2 * np.pi * i * tau) for i in range(1, kmax + 1)], 1)
    return trend, seas


def rank(A: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(A, tol=1e-9 * max(A.shape) * max(1.0, np.abs(A).max())))


def main() -> None:
    H, K, q = 48, 10, 2
    b = build_bases(H, K, q, dtype=torch.float64)
    print(f"Configuration of the paper's main study: H = {H}, K = {K}, q = {q}, m = {b.m}")

    print("\nLemma 1: rank [P^T F^T] = m whenever m <= H")
    B = torch.cat([b.P, b.F], 0).T
    ok(int(torch.linalg.matrix_rank(B)) == b.m)
    print(f"   rank = {int(torch.linalg.matrix_rank(B))}, m = {b.m}  [{'ok' if int(torch.linalg.matrix_rank(B)) == b.m else 'FAIL'}]")
    for bad_K in (23, 24):
        try:
            check_dimensions(H, bad_K, q)
            ok(False)
        except ValueError:
            print(f"   K = {bad_K} is rejected (2K >= H or m > H): the hypothesis of the lemma is enforced  [ok]")

    print("\nTheorem 1: orthogonality, decomposition of R^H and uniqueness")
    for name, v in (("P F~^T", b.P @ b.Ft.T), ("P Q", b.P @ b.Q), ("F~ Q", b.Ft @ b.Q)):
        ok(check(f"max |{name}|", float(v.abs().max()), 1e-10))
    ok(int(torch.linalg.matrix_rank(b.M)) == H)
    print(f"   rank M = {int(torch.linalg.matrix_rank(b.M))} = H (M = [P^T F~^T Q] is invertible)")
    torch.manual_seed(0)
    y = torch.randn(7, H, dtype=torch.float64)                        # ANY vector in R^H has a decomposition
    thT = torch.linalg.solve(b.P @ b.P.T, b.P @ y.T).T               # (P P^T)^-1 P y
    thS = torch.linalg.solve(b.Ft @ b.Ft.T, b.Ft @ y.T).T            # (F~ F~^T)^-1 F~ y
    u = y @ b.Q                                                       # Q^T y
    rec = thT @ b.P + thS @ b.Ft + u @ b.Q.T
    ok(check("max |y - (P^T theta_T + F~^T theta_S + Q u)|", float((rec - y).abs().max()), 1e-10))
    head = NeuralFourierHead(16, H, K, q).double()
    dec = head.decompose(y)                                           # library: (theta_T, theta_S, u) = M^-1 y
    for nm, a, c in (("theta_T", dec.theta_T, thT), ("theta_S", dec.theta_S, thS), ("u", dec.u, u)):
        ok(check(f"max |library {nm} - closed form|", float((a - c).abs().max()), 1e-9))
    # a dense layer with totally different weights gives the same decomposition of the same forecast
    h = torch.randn(5, 16, dtype=torch.float64)
    with torch.no_grad():
        out = head(h)
        dense = head.to_linear()
        yd = dense(h)
    ok(check("max |decompose(dense forecast) - NFH amplitudes|", float((head.decompose(yd).amplitude - out.amplitude).abs().max()), 1e-9))

    print("\nProposition 1: dim(T_nb and S_nb) = p_nb for even H > p_nb >= 1 (N-BEATS-I bases)")
    print(f"   {'H':>3s} {'p_nb':>4s} {'dim T':>6s} {'dim S':>6s} {'dim intersection':>17s}")
    for Hn in (6, 8, 14, 18, 30, 48):
        for p_nb in (1, 2, 3):
            T, S = nbeats_i_bases(Hn, p_nb)
            d = rank(T) + rank(S) - rank(np.concatenate([T, S], 1))
            ok(d == p_nb)
            print(f"   {Hn:3d} {p_nb:4d} {rank(T):6d} {rank(S):6d} {d:17d}  [{'ok' if d == p_nb else 'FAIL'}]")
    # the consequence: moving v from T to S changes neither the forecast nor validity of the split
    T, S = nbeats_i_bases(H, 2)
    tau = np.arange(H) / H
    v = tau * (tau - (H - 1) / H)                                     # lies in both spans (H even, p_nb = 2)
    in_S = np.abs(S @ np.linalg.lstsq(S, v, rcond=None)[0] - v).max()
    ok(check("distance of v = tau (tau - (H-1)/H) to span(S_nb)", float(in_S), 1e-9))
    print("   so (T* + v, S* - v) is another valid N-BEATS-I split of the same forecast; NFH has none (S and T are orthogonal).")

    print("\nAll checks passed." if not FAILED else f"\n{len(FAILED)} CHECK(S) FAILED.")
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
