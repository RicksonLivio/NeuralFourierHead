"""Testes das garantias da NFH (cada teste corresponde a um resultado da dissertação, Cap. 4)."""
import math

import numpy as np
import pytest
import torch

from nfh import (Forecaster, NeuralFourierHead, attach_nfh, build_bases, check_dimensions, energy_balance,
                 explain, harmonic_shares, integrated_gradients, max_harmonics)

CONFIGS = [(30, 10, 2), (24, 6, 1), (8, 2, 1), (48, 10, 2), (168, 20, 2)]


@pytest.mark.parametrize("H,K,q", CONFIGS)
def test_subspaces_are_orthogonal_and_complete(H, K, q):
    """Lema da decomposição ortogonal: 𝒯 ⊥ 𝒮 ⊥ ℛ e R^H = 𝒯 ⊕ 𝒮 ⊕ ℛ."""
    b = build_bases(H, K, q, dtype=torch.float64)
    assert torch.allclose(b.P @ b.Ft.T, torch.zeros(q + 1, 2 * K, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(b.P @ b.Q, torch.zeros(q + 1, H - b.m, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(b.Ft @ b.Q, torch.zeros(2 * K, H - b.m, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(b.Q.T @ b.Q, torch.eye(H - b.m, dtype=torch.float64), atol=1e-10)
    assert int(torch.linalg.matrix_rank(b.M)) == H  # M invertível (Teorema da equivalência)
    ones = torch.ones(H, dtype=torch.float64)
    assert torch.allclose(b.Pi_T @ ones, ones, atol=1e-10)  # 𝟙 ∈ 𝒯


@pytest.mark.parametrize("H,K,q", [(30, 15, 2), (30, 14, 2), (10, 4, 2)])
def test_dimension_condition_is_enforced(H, K, q):
    """Nyquist (2K < H) e m = q + 1 + 2K ≤ H são verificados."""
    with pytest.raises(ValueError):
        check_dimensions(H, K, q)


def test_max_harmonics_is_the_largest_valid_k():
    for H in range(4, 60):
        for q in range(0, 3):
            k = max_harmonics(H, q)
            if k >= 1:
                check_dimensions(H, k, q)
                with pytest.raises(ValueError):
                    check_dimensions(H, k + 1, q)


@pytest.mark.parametrize("normalize", [True, False])
def test_exactness_identifiability_and_energy(normalize):
    """Exatidão, identificabilidade e balanço de energia para uma head aleatória."""
    torch.manual_seed(0)
    head = NeuralFourierHead(32, 30, 10, 2, normalize_columns=normalize).double()
    h = torch.randn(64, 32, dtype=torch.float64)
    with torch.no_grad():
        out = head(h)
    # Exatidão: ŷ = T + S + R e S = Σ_k S_k
    assert torch.allclose(out.forecast, out.trend + out.seasonal + out.residual, atol=1e-12)
    # Identificabilidade: os coeficientes são recuperados apenas a partir de ŷ
    dec = head.decompose(out.forecast)
    for a, b in ((dec.theta_T, out.theta_T), (dec.theta_S, out.theta_S), (dec.u, out.u)):
        assert torch.allclose(a, b, atol=1e-9)
    # Balanço de energia: ε_T + ε_S + ε_R = 1, sem termo cruzado
    e = energy_balance(out)
    assert np.allclose(e["eps_T"] + e["eps_S"] + e["eps_R"], 1.0, atol=1e-10)
    assert np.abs(e["iota"]).max() < 1e-10
    # Nível = média da tendência
    assert np.allclose(e["level"], out.trend.mean(-1).numpy(), atol=1e-12)
    # Σ e_k = ε_S
    ek = harmonic_shares(out, head.Ft)
    assert np.allclose(ek.sum(1), e["eps_S"], atol=1e-10)
    # Energia própria de cada harmônico ≤ (H/2) A_k²
    Sk_norm2 = ((out.theta_S[:, 0::2, None] * head.Ft[None, 0::2] +
                 out.theta_S[:, 1::2, None] * head.Ft[None, 1::2]) ** 2).sum(-1)
    assert torch.all(Sk_norm2 <= 15 * out.amplitude ** 2 + 1e-9)


def test_same_parameter_count_as_dense():
    head = NeuralFourierHead(128, 30, 10, 2)
    assert sum(p.numel() for p in head.parameters()) == 30 * (128 + 1) == 3870


def test_dense_equivalence_both_directions():
    """Teorema da equivalência: from_linear / to_linear preservam a função."""
    torch.manual_seed(1)
    dense = torch.nn.Linear(16, 24).double()
    h = torch.randn(10, 16, dtype=torch.float64)
    nfh = NeuralFourierHead.from_linear(dense, n_harmonics=6, degree=1)
    with torch.no_grad():
        assert torch.allclose(nfh(h).forecast, dense(h), atol=1e-10)
        assert torch.allclose(nfh.to_linear()(h), dense(h), atol=1e-10)


def test_attach_nfh_keeps_predictions():
    torch.manual_seed(2)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = torch.nn.Sequential(torch.nn.Linear(5, 32), torch.nn.Tanh())
            self.out = torch.nn.Linear(32, 30)

        def forward(self, x):
            o = self.out(self.body(x))
            return o.forecast if hasattr(o, "forecast") else o

    net = Net()
    x = torch.randn(7, 5)
    before = net(x).detach()
    attach_nfh(net, "out", n_harmonics=10, degree=2)
    assert isinstance(net.out, NeuralFourierHead)
    assert torch.allclose(net(x).detach(), before, atol=1e-4)


def test_gradient_flows_end_to_end():
    """Teorema do fluxo de gradiente: o backbone recebe gradiente."""
    torch.manual_seed(3)
    model = Forecaster("lstm", n_features=3, horizon=24, n_harmonics=6, degree=1)
    x, y = torch.randn(4, 48, 3), torch.randn(4, 24)
    out = model(x)
    loss = torch.nn.functional.mse_loss(out.forecast, y) + 10 * model.residual_penalty(out)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)


def test_revin_denormalization_preserves_decomposition():
    """Observação RevIN: após desnormalizar, ŷ = T + S + R continua exata e ε somam 1."""
    torch.manual_seed(4)
    model = Forecaster("transformer", n_features=2, horizon=24, n_harmonics=6, degree=1).double().eval()
    x = torch.randn(5, 72, 2, dtype=torch.float64) * 3 + 10
    with torch.no_grad():
        out = model(x)
    assert torch.allclose(out.forecast, out.trend + out.seasonal + out.residual, atol=1e-9)
    e = energy_balance(out)
    assert np.allclose(e["eps_T"] + e["eps_S"] + e["eps_R"], 1.0, atol=1e-9)


def test_explain_and_integrated_gradients_completeness():
    """Proposição da completude: Σ_v I_c(v) = g_c(X) − g_c(X')."""
    torch.manual_seed(5)
    model = Forecaster("lstm", n_features=4, horizon=24, n_harmonics=6, degree=1).eval()
    x = torch.randn(1, 48, 4)
    exp = explain(model, x)
    assert exp.shares.shape == (1, 6) and math.isclose(exp.eps_T[0] + exp.eps_S[0] + exp.eps_R[0], 1, rel_tol=1e-6)
    xn, _, _ = model.normalize(x)
    attr = integrated_gradients(model.inner, xn, ["a", "b", "c", "d"], steps=128)
    for c in ("T", "S", "R"):
        scale = abs(attr[c]["delta"]) + 1e-6
        assert attr[c]["completeness_error"] / scale < 0.05
        assert math.isclose(sum(attr[c]["share"].values()), 1.0, rel_tol=1e-6)
