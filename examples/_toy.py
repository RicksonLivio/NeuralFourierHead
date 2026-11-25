"""Small helpers shared by the examples: synthetic series, a short training loop and printing.

Everything here is CPU-friendly (each example runs in well under two minutes). The library itself
(``nfh``) has no dependency on this file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from nfh import Forecaster
from nfh.data import WindowDataset, make_splits
from nfh.training import TrainConfig, fit


def check(label: str, value: float, tol: float) -> bool:
    """Print a numerical check (|value| <= tol) and return whether it passed."""
    ok = abs(value) <= tol
    print(f"   [{'ok' if ok else 'FAIL'}] {label}: {value:.2e} (tolerance {tol:.0e})")
    return ok


def pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def synthetic_univariate(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """y_t = slow trend + 1.0 sin(2 pi t/24) + 0.5 cos(2 pi t/8) + AR(1) noise (known components)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.6 * e[i - 1] + rng.normal(0, 0.15)
    y = 0.0004 * t + 0.5 * np.sin(2 * np.pi * t / 2000) + np.sin(2 * np.pi * t / 24) + 0.5 * np.cos(2 * np.pi * t / 8) + e
    return pd.DataFrame({"y": y})


def _ar1(rng, n, phi, sigma):
    z = np.zeros(n)
    e = rng.normal(0, sigma, n)
    for i in range(1, n):
        z[i] = phi * z[i - 1] + e[i]
    return z


def sint_m(n: int = 5000, lag: int = 48, seed: int = 2026) -> pd.DataFrame:
    """Small SINT-M: same equations as ``src/v3_sintetico_multivariado.py::gerar`` with a smaller ``n``.

    x1 slow AR(1) (drives the trend), x2 daily oscillation with varying amplitude (drives the cycle),
    x3 and x4 distractors of the same two types; y_t = x1_{t-lag} + x2_{t-lag} + e_t (AR(1) noise).
    The target is NOT an input, so the ground truth is unambiguous.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    box = np.ones(24) / 24
    x1 = np.convolve(_ar1(rng, n, 0.998, 0.06), box, mode="same")
    x3 = np.convolve(_ar1(rng, n, 0.998, 0.06), box, mode="same")
    x2 = np.exp(0.4 * _ar1(rng, n, 0.99, 0.14)) * np.sin(2 * np.pi * t / 24 + 0.3)
    x4 = np.exp(0.4 * _ar1(rng, n, 0.99, 0.14)) * np.sin(2 * np.pi * t / 24 + 1.4)
    y = np.full(n, np.nan)
    y[lag:] = x1[:-lag] + x2[:-lag]
    y = y + _ar1(rng, n, 0.5, 0.15)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y}).iloc[lag:].reset_index(drop=True)


class Splits:
    """Chronological 70/15/15 windows; inputs and target standardized with TRAIN statistics."""

    def __init__(self, df: pd.DataFrame, inputs: list[str], target: str, L: int, H: int):
        n = len(df)
        a, b = int(0.70 * n), int(0.85 * n)
        X, y = df[inputs].to_numpy(float), df[target].to_numpy(float)
        X = ((X - X[:a].mean(0)) / X[:a].std(0)).astype(np.float32)
        y = ((y - y[:a].mean()) / y[:a].std()).astype(np.float32)
        self.train = WindowDataset(X[:a], y[:a], L, H)
        self.val = WindowDataset(X[a:b], y[a:b], L, H)
        self.test = WindowDataset(X[b:], y[b:], L, H)


def train_dense(split, backbone: str, L: int, H: int, K: int, q: int, n_features: int, target_idx: int = 0,
                epochs: int = 6, seed: int = 0, revin=True) -> Forecaster:
    """Train a Forecaster with a DENSE output layer by MSE (the starting point of the NFH procedure)."""
    torch.manual_seed(seed)
    kw = {"backbone_kwargs": {"lookback": L}} if backbone == "mlp" else {}
    dense = Forecaster(backbone, n_features=n_features, horizon=H, target_idx=target_idx, head="dense",
                       n_harmonics=K, degree=q, revin=revin, **kw)
    cfg = TrainConfig(max_epochs=epochs, patience=epochs, batch_size=64, device="cpu", seed=seed)
    return fit(dense, split.train, split.val, cfg).model.eval()


__all__ = ["check", "pct", "synthetic_univariate", "sint_m", "train_dense", "make_splits", "Splits"]
