"""Example 03: exact dense -> NFH conversion, then structural adjustment with safe selection.

Paper: Section 2 (equivalence with the dense layer, "pure decomposition", Training paragraph).
  1. train a model with a DENSE output layer (MSE only);
  2. convert it to an NFH: same forecast (epoch 0 = pure decomposition of the dense forecast);
  3. structural adjustment: E more epochs on J_lambda = MSE + lambda (M H)^-1 sum ||R||^2;
  4. safe selection: among the epochs with validation RMSE r_nu <= r_0, take the lowest validation J_lambda.
Epoch 0 is always feasible, so the selected NFH is never worse than the dense model on VALIDATION data
(on test data it can be).

    uv run python examples/03_dense_to_nfh_and_adjustment.py
"""
from __future__ import annotations

import numpy as np
import torch

from _toy import check, make_splits, pct, synthetic_univariate, train_dense
from nfh import AdjustConfig, energy_balance, structural_adjustment

L, H, q = 48, 24, 1                # the toy series has cycles of 24 and 8 steps, i.e. harmonics k = 1 and k = 3 of H = 24


def run(split, K: int) -> None:
    Xte, Yte = split.test.tensors()
    rmse = lambda P: float(torch.sqrt(((P - Yte) ** 2).mean()))  # noqa: E731
    print(f"\n=== K = {K} harmonics (m = q + 1 + 2K = {q + 1 + 2 * K} <= H = {H}) ===")
    print("1. Dense model (MLP backbone + nn.Linear) trained with MSE")
    dense = train_dense(split, "mlp", L, H, K, q, n_features=1, epochs=15)
    with torch.no_grad():
        yd = dense(Xte).forecast
    print(f"   test RMSE (standardized) = {rmse(yd):.4f}")

    print("2. Exact conversion to NFH: W_NFH = (M D)^-1 W_dense")
    nfh0 = dense.to_nfh().eval()
    with torch.no_grad():
        o0 = nfh0(Xte)
    check("max |y_dense - y_NFH| over all test windows", float((yd - o0.forecast).abs().max()), 1e-4)
    check("max |y_NFH - (T + S + R)|", float((o0.forecast - o0.trend - o0.seasonal - o0.residual).abs().max()), 1e-4)
    cov0 = energy_balance(o0)["coverage"]
    print(f"   pure decomposition (epoch 0): median kappa = {pct(np.median(cov0))}, minimum = {pct(cov0.min())}")

    print("3. Structural adjustment (lambda = 1000, rate 1e-4, E = 6 epochs) with safe selection")
    res = structural_adjustment(dense.to_nfh(), split.train, split.val,
                                AdjustConfig(epochs=6, lr=1e-4, lam=1000.0, batch_size=64, device="cpu"))
    print(f"   {'epoch':>5s} {'val RMSE':>9s} {'val J':>9s} {'kappa':>7s}  feasible (r <= r_0)")
    for t in res.trajectory:
        mark = "  <- safe selection" if t["epoch"] == res.selected_epoch else ""
        print(f"   {t['epoch']:5d} {t['rmse']:9.4f} {t['J']:9.4f} {pct(t['kappa']):>7s}  {'yes' if t['feasible'] else 'no '}{mark}")
    sel = res.selected
    print(f"   selected epoch nu* = {res.selected_epoch}; r_nu* = {sel['rmse']:.4f} <= r_0 = {res.rmse_val_dense:.4f}")
    if not (sel["feasible"] and sel["rmse"] <= res.rmse_val_dense + 1e-12):
        raise SystemExit("safe selection violated (this should never happen)")
    with torch.no_grad():
        o1 = res.model.eval()(Xte)
    cov1 = energy_balance(o1)["coverage"]
    print(f"   test RMSE: dense {rmse(yd):.4f} -> NFH {rmse(o1.forecast):.4f} (no guarantee on test data)")
    print(f"   median kappa: {pct(np.median(cov0))} -> {pct(np.median(cov1))}")


def main() -> None:
    split = make_splits(synthetic_univariate(), "y", ["y"], L, H)
    run(split, K=2)    # the basis cannot hold the 8-step cycle: adjusting costs accuracy, safe selection keeps epoch 0
    run(split, K=4)    # the basis holds both cycles: the pure decomposition already covers the shape


if __name__ == "__main__":
    main()
