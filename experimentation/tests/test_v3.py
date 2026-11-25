"""Testes dos resultados da versão 3 da dissertação: conversão exata, seleção segura, explicações abdutivas, IG."""
import numpy as np
import pandas as pd
import pytest
import torch

from nfh import (AdjustConfig, Forecaster, abductive_explanations, integrated_gradients, latent_box,
                 structural_adjustment)
from nfh.abductive import brute_force_minimum, manual_minimal_explanation, minimal_explanation
from nfh.data import make_splits
from nfh.explain import quadrature
from nfh.training import TrainConfig, fit


def _toy_split(n=1500, L=48, H=24, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = np.sin(2 * np.pi * t / 24) + 0.3 * np.sin(2 * np.pi * t / 8) + 0.002 * t + 0.2 * rng.standard_normal(n)
    x = np.cos(2 * np.pi * t / 24) + 0.1 * rng.standard_normal(n)
    return make_splits(pd.DataFrame({"y": y, "x": x}), "y", ["y", "x"], L, H)


@pytest.mark.parametrize("backbone", ["mlp", "lstm", "convlstm", "transformer"])
def test_dense_to_nfh_conversion_is_exact(backbone):
    """Teorema da equivalência: a NFH convertida faz exatamente a mesma previsão que a densa."""
    torch.manual_seed(0)
    kw = {"backbone_kwargs": {"lookback": 48}} if backbone == "mlp" else {}
    dense = Forecaster(backbone, n_features=2, horizon=24, head="dense", n_harmonics=6, degree=1, **kw).eval()
    nfh = dense.to_nfh().eval()
    x = torch.randn(16, 48, 2) * 3 + 1
    with torch.no_grad():
        yd, yn = dense(x).forecast, nfh(x)
    assert nfh.head_type == "nfh"
    assert torch.allclose(yd, yn.forecast, atol=2e-5)
    assert torch.allclose(yn.forecast, yn.trend + yn.seasonal + yn.residual, atol=1e-5)
    # o backbone foi copiado, não compartilhado
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(dense.backbone.parameters(), nfh.backbone.parameters()))


def test_safe_selection_never_worse_than_dense_on_validation():
    """Proposição 4.10: RMSE_val(NFH selecionada) <= RMSE_val(densa) e J selecionado <= J da época 0."""
    split = _toy_split()
    dense = Forecaster("lstm", n_features=2, horizon=24, head="dense", n_harmonics=6, degree=1)
    dense = fit(dense, split.train, split.val, TrainConfig(max_epochs=3, patience=3, batch_size=64, device="cpu")).model
    res = structural_adjustment(dense.to_nfh(), split.train, split.val,
                                AdjustConfig(epochs=3, lam=100.0, lr=1e-3, batch_size=64, device="cpu"))
    traj = res.trajectory
    Xv, Yv = split.val.tensors()
    with torch.no_grad():
        r_dense = float(torch.sqrt(((dense(Xv).forecast - Yv) ** 2).mean()))
    assert abs(traj[0]["rmse"] - r_dense) < 1e-5            # época 0 = a densa convertida
    sel = res.selected
    assert sel["feasible"] and sel["rmse"] <= res.rmse_val_dense + 1e-12
    assert sel["J"] <= traj[0]["J"] + 1e-12
    feasible = [t for t in traj if t["feasible"]]
    assert sel["J"] == min(t["J"] for t in feasible)          # mínimo entre as viáveis


def test_minimal_abductive_explanation_matches_brute_force():
    """Proposição 2.4: a ordenação dos ganhos dá o menor conjunto suficiente (conferido por enumeração)."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        p = 7
        w, h = rng.normal(size=p), rng.uniform(-1, 1, p)
        lo, hi = np.full(p, -1.0), np.full(p, 1.0)
        w0 = rng.normal()
        val = float(w @ h + w0)
        xi = val - abs(rng.normal()) * 0.5          # propriedade satisfeita em h*
        r = minimal_explanation(w, w0, h, lo, hi, xi)
        assert r["k"] == brute_force_minimum(w, w0, h, lo, hi, xi)
        m = manual_minimal_explanation(w.tolist(), w0, h.tolist(), lo.tolist(), hi.tolist(), xi)
        assert m["k"] == r["k"] and set(m["neurons"]) == set(r["neurons"])


def test_abductive_explanations_are_sufficient_on_a_model():
    """Fixados os neurônios da explicação, nenhum completamento da caixa viola P_T ou P_S."""
    split = _toy_split()
    torch.manual_seed(0)
    model = Forecaster("lstm", n_features=2, horizon=24, n_harmonics=6, degree=2).eval()
    Xtr, _ = split.train.tensors()
    Xte, _ = split.test.tensors()
    box = latent_box(model, Xtr)
    with torch.no_grad():
        h = model(Xte[:1]).latent[0].double().numpy()
    ab = abductive_explanations(model, h, box, k_cycle=1, n_check=5000)
    assert set(ab) == {"P_T", "P_S"}
    for a in ab.values():
        assert a["k"] >= 0 and a["violations"] == 0
        assert a["k"] <= a["k_ig"]                  # a ordem dos IG nunca precisa de menos neurônios


def test_gauss_legendre_quadrature():
    """Pesos somam 1 e a regra integra polinômios de grau 2n-1 exatamente; completude melhor que o ponto médio."""
    a, w = quadrature(8)
    assert abs(w.sum() - 1) < 1e-14 and ((a > 0) & (a < 1)).all()
    assert abs((w * a ** 15).sum() - 1 / 16) < 1e-14
    torch.manual_seed(0)
    model = Forecaster("lstm", n_features=3, horizon=24, n_harmonics=6, degree=1).eval()
    xn, _, _ = model.normalize(torch.randn(1, 48, 3))
    gl = integrated_gradients(model.inner, xn, ["a", "b", "c"], steps=16)
    mp = integrated_gradients(model.inner, xn, ["a", "b", "c"], steps=16, method="midpoint")
    for c in ("T", "S"):
        assert gl[c]["completeness_error"] <= mp[c]["completeness_error"] + 1e-9
