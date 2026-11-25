"""Example 05: synthetic series with known drivers (a small SINT-M).

Paper: "Known drivers" paragraph. x1 (slow) drives the predicted trend, x2 (daily oscillation) drives the cycle,
x3 and x4 are distractors of the same two types; y_t = x1_{t-48} + x2_{t-48} + e_t and the target is not an input.
An attribution is correct when the variable with the largest absolute share is the true driver.

This is a SMALL, CPU version (shorter series, MLP backbone, few epochs, 30 windows, one seed). Its numbers are
not the paper's; they only show the protocol. The paper's numbers come from ``src/v3_sintetico_multivariado.py``
(stored in ``results/v3/sint_m.json``, see ``docs/PAPER_TO_CODE.md``).

    uv run python examples/05_synthetic_known_drivers.py
"""
from __future__ import annotations

import numpy as np

from _toy import Splits, pct, sint_m, train_dense
from nfh import AdjustConfig, integrated_gradients, structural_adjustment

L, H, K, q = 96, 48, 10, 2
FEATURES = ["x1", "x2", "x3", "x4"]
N_WINDOWS = 30


def main() -> None:
    split = Splits(sint_m(4500), FEATURES, "y", L, H)
    Xte, _ = split.test.tensors()
    dense = train_dense(split, "mlp", L, H, K, q, n_features=4, epochs=8, revin=False)
    model = structural_adjustment(dense.to_nfh(), split.train, split.val,
                                  AdjustConfig(epochs=5, lr=1e-4, lam=1000.0, batch_size=64, device="cpu")).model.eval()
    idx = np.linspace(0, len(Xte) - 1, N_WINDOWS).astype(int)
    hit = {"T": 0, "S": 0}
    for j in idx:
        a = integrated_gradients(model.inner, Xte[j:j + 1], FEATURES, steps=32)
        for comp, truth in (("T", "x1"), ("S", "x2")):
            best = max(a[comp]["share"], key=a[comp]["share"].get)
            hit[comp] += best == truth
    print(f"Trend driver (x1) found on g_T in {hit['T']} of {N_WINDOWS} windows ({pct(hit['T'] / N_WINDOWS)})")
    print(f"Cycle driver (x2) found on g_S in {hit['S']} of {N_WINDOWS} windows ({pct(hit['S'] / N_WINDOWS)})")
    print("Chance level with four variables is 25%.")


if __name__ == "__main__":
    main()
