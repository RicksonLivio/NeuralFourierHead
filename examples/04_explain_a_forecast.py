"""Example 04: explain one forecast: components, amplitudes, shares and, as a complement, Captum attribution.

Paper: Section 2 (shares eps_T, eps_S, eps_R; amplitudes A_k; per-component attribution with g_T and g_S).
The decomposition is the explanation (T, S, R, A_k, eps). Attribution to the INPUT variables is a second, optional
layer: Integrated Gradients on g_T(X) = mean of T (the level) and g_S(X) = ||S||^2 (seasonal energy), with the
reference X' = 0. The library implements it (``nfh.integrated_gradients``, Gauss-Legendre) and the same functionals
can be passed to Captum, which is what this example does to compare both.

    uv run python examples/04_explain_a_forecast.py
"""
from __future__ import annotations

import numpy as np
import torch
from captum.attr import IntegratedGradients

from _toy import Splits, check, pct, sint_m, train_dense
from nfh import AdjustConfig, explain, integrated_gradients, narrative, structural_adjustment

L, H, K, q = 96, 48, 10, 2
FEATURES = ["x1", "x2", "x3", "x4"]


def main() -> None:
    split = Splits(sint_m(3500), FEATURES, "y", L, H)
    Xte, Yte = split.test.tensors()
    dense = train_dense(split, "mlp", L, H, K, q, n_features=4, epochs=5, revin=False)
    model = structural_adjustment(dense.to_nfh(), split.train, split.val,
                                  AdjustConfig(epochs=4, lr=1e-4, lam=1000.0, batch_size=64, device="cpu")).model.eval()

    i = 10
    x = Xte[i:i + 1]
    exp = explain(model, x)
    print(f"1. Forecast of test window {i}: {H} steps = T + S + R")
    print(f"   level (mean of T) = {exp.level[0]:+.3f}; trend goes from {exp.trend[0, 0]:+.3f} to {exp.trend[0, -1]:+.3f}")
    print(f"   eps_T = {pct(exp.eps_T[0])}, eps_S = {pct(exp.eps_S[0])}, eps_R = {pct(exp.eps_R[0])}, kappa = {pct(exp.coverage[0])}")
    check("eps_T + eps_S + eps_R - 1", float(exp.eps_T[0] + exp.eps_S[0] + exp.eps_R[0] - 1), 1e-6)

    print("2. Harmonics: amplitude A_k, phase and additive share e_k (sum of e_k = eps_S; negative = cancellation)")
    for k in np.argsort(-exp.shares[0])[:3]:
        print(f"   k={k + 1:2d} (period {exp.periods[k]:5.1f} steps): A_k = {exp.amplitude[0, k]:.3f}, "
              f"phase {exp.phase[0, k]:+.2f} rad, e_k = {pct(exp.shares[0, k])}")

    print("3. Text summary")
    for line in narrative(exp, 0, unit="steps", target="standardized y"):   # text is in Portuguese, as in the library
        print(f"   - {line}")

    print("4. Which input variable sustains each component? (explanation of g_T and g_S, reference X' = 0)")
    ours = integrated_gradients(model.inner, x, FEATURES, steps=64)       # library, Gauss-Legendre quadrature
    for comp, name in (("T", "level g_T"), ("S", "seasonal energy g_S")):
        top = sorted(ours[comp]["share"].items(), key=lambda kv: -kv[1])
        print(f"   nfh IG on {name:19s}: " + ", ".join(f"{k} {pct(v)}" for k, v in top[:3])
              + f"  (completeness error {ours[comp]['completeness_error']:.1e})")

    print("5. Complement: the same two scalar functions explained with Captum's IntegratedGradients")
    g = {"T": lambda xx: model.inner(xx).trend.mean(-1), "S": lambda xx: (model.inner(xx).seasonal ** 2).sum(-1)}
    for comp, name in (("T", "g_T"), ("S", "g_S")):
        ig = IntegratedGradients(g[comp])
        a = ig.attribute(x.clone(), baselines=torch.zeros_like(x), n_steps=64, method="gausslegendre")
        per_var = a[0].sum(0).detach().numpy()                              # sum over lags: I_c(v)
        mine = np.array([ours[comp]["I"][v] for v in FEATURES])
        share = np.abs(per_var) / np.abs(per_var).sum()
        print(f"   Captum IG on {name}: " + ", ".join(f"{v} {pct(s)}" for v, s in zip(FEATURES, share))
              + f"  | max |Captum - nfh| = {np.abs(per_var - mine).max():.1e}")
    print("   (an attribution is a local reading of the model, not a causal statement)")


if __name__ == "__main__":
    main()
