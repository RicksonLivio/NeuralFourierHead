"""Example 01: couple the NFH to any backbone and read T, S, R, A_k and the shares.

Paper: Section 2 (the layer; y_hat = T + S + R). Runs on CPU in a few seconds.

    uv run python examples/01_quickstart.py
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from _toy import check, make_splits, pct, synthetic_univariate
from nfh import NeuralFourierHead, attach_nfh, energy_balance, harmonic_shares

L, H, K, q = 48, 24, 4, 1          # window, horizon, harmonics, trend degree; m = q + 1 + 2K = 10 <= H
torch.manual_seed(0)


class MyForecaster(nn.Module):
    """Any backbone Phi (here a GRU) followed by the NFH instead of nn.Linear(p, H)."""

    def __init__(self, d: int, p: int = 32):
        super().__init__()
        self.gru = nn.GRU(d, p, batch_first=True)
        self.head = NeuralFourierHead(p, H, n_harmonics=K, degree=q)     # same H(p + 1) parameters as nn.Linear(p, H)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])             # NFHOutput: forecast, trend, seasonal, residual, theta_T, theta_S, u, ...


def main() -> None:
    split = make_splits(synthetic_univariate(), "y", ["y"], L, H)
    Xtr, Ytr = split.train.tensors()
    Xte, Yte = split.test.tensors()
    model = MyForecaster(1)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(3):                      # a few epochs are enough for the demonstration
        for i in torch.randperm(len(Xtr)).split(128):
            loss = nn.functional.mse_loss(model(Xtr[i]).forecast, Ytr[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        out = model(Xte)

    print("1. Components of every forecast (shape = batch x H):")
    print(f"   forecast {tuple(out.forecast.shape)}, T {tuple(out.trend.shape)}, S {tuple(out.seasonal.shape)}, "
          f"R {tuple(out.residual.shape)}")
    check("max |y_hat - (T + S + R)|", float((out.forecast - out.trend - out.seasonal - out.residual).abs().max()), 1e-5)

    print("2. Coefficients: theta_T (q+1), theta_S = (a_1, b_1, ..., a_K, b_K) and u (H - m):")
    print(f"   theta_T {tuple(out.theta_T.shape)}, theta_S {tuple(out.theta_S.shape)}, u {tuple(out.u.shape)}")

    print("3. Amplitudes A_k = sqrt(a_k^2 + b_k^2) of the first test window (period H/k steps):")
    for k, a in enumerate(out.amplitude[0].tolist(), start=1):
        print(f"   k={k} (period {H / k:5.1f}): A_k = {a:.3f}")

    print("4. Shares of the centered forecast energy (they add up to one):")
    e = energy_balance(out)
    print(f"   mean eps_T {pct(e['eps_T'].mean())}, eps_S {pct(e['eps_S'].mean())}, eps_R {pct(e['eps_R'].mean())}; "
          f"kappa = 1 - eps_R = {pct(e['coverage'].mean())}")
    check("max |eps_T + eps_S + eps_R - 1|", float(np.abs(e["eps_T"] + e["eps_S"] + e["eps_R"] - 1).max()), 1e-6)
    ek = harmonic_shares(out, model.head.Ft)
    check("max |sum_k e_k - eps_S|", float(np.abs(ek.sum(1) - e["eps_S"]).max()), 1e-6)

    print("5. An already trained dense model: swap its last nn.Linear for an equivalent NFH (same predictions).")
    class Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Flatten(), nn.Linear(L, 32), nn.ReLU())
            self.out = nn.Linear(32, H)

    dense = Dense()
    x = Xte[:8]
    with torch.no_grad():
        before = dense.out(dense.body(x))
    head = attach_nfh(dense, "out", n_harmonics=K, degree=q)     # dense.out is now a NeuralFourierHead
    with torch.no_grad():
        after = dense.out(dense.body(x)).forecast
    check("max |dense - NFH| predictions", float((before - after).abs().max()), 1e-4)


if __name__ == "__main__":
    main()
