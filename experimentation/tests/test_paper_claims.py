"""Tests that tie the library to the statements of the paper (names follow docs/PAPER_TO_CODE.md)."""
import numpy as np
import pytest
import torch

from nfh import NeuralFourierHead, build_bases, check_dimensions, energy_balance


def _nbeats_i(H, p_nb):
    tau = np.arange(H) / H
    trend = np.stack([tau ** i for i in range(p_nb + 1)], 1)
    kmax = int(np.floor(H / 2 - 1))
    seas = np.stack([np.ones(H)] + [np.cos(2 * np.pi * i * tau) for i in range(1, kmax + 1)]
                    + [np.sin(2 * np.pi * i * tau) for i in range(1, kmax + 1)], 1)
    return trend, seas


def _rank(A):
    return int(np.linalg.matrix_rank(A, tol=1e-9 * max(A.shape) * max(1.0, np.abs(A).max())))


@pytest.mark.parametrize("H", [6, 8, 14, 18, 30, 48])
@pytest.mark.parametrize("p_nb", [1, 2, 3])
def test_proposition_1_nbeats_i_intersection_has_dimension_p_nb(H, p_nb):
    T, S = _nbeats_i(H, p_nb)
    assert _rank(T) + _rank(S) - _rank(np.concatenate([T, S], 1)) == p_nb


def test_lemma_1_rank_equals_m_and_fails_beyond_h():
    b = build_bases(48, 10, 2, dtype=torch.float64)
    assert int(torch.linalg.matrix_rank(torch.cat([b.P, b.F], 0).T)) == b.m == 23
    with pytest.raises(ValueError):
        check_dimensions(48, 24, 2)


def test_theorem_1_closed_form_coefficients():
    b = build_bases(24, 4, 1, dtype=torch.float64)
    y = torch.randn(5, 24, dtype=torch.float64)
    thT = torch.linalg.solve(b.P @ b.P.T, b.P @ y.T).T
    thS = torch.linalg.solve(b.Ft @ b.Ft.T, b.Ft @ y.T).T
    u = y @ b.Q
    assert torch.allclose(thT @ b.P + thS @ b.Ft + u @ b.Q.T, y, atol=1e-10)
    dec = NeuralFourierHead(8, 24, 4, 1).double().decompose(y)
    assert torch.allclose(dec.theta_T, thT, atol=1e-9) and torch.allclose(dec.theta_S, thS, atol=1e-9)
    assert torch.allclose(dec.u, u, atol=1e-9)


def test_pythagoras_shares_sum_to_one_and_lie_in_unit_interval():
    torch.manual_seed(1)
    head = NeuralFourierHead(16, 48, 10, 2).double()
    with torch.no_grad():
        e = energy_balance(head(torch.randn(32, 16, dtype=torch.float64)))
    assert np.allclose(e["eps_T"] + e["eps_S"] + e["eps_R"], 1.0, atol=1e-10)
    for k in ("eps_T", "eps_S", "eps_R"):
        assert (e[k] >= -1e-12).all() and (e[k] <= 1 + 1e-12).all()
    assert np.allclose(e["coverage"], 1 - e["eps_R"])


def test_forecast_mean_is_the_mean_of_trend():
    head = NeuralFourierHead(16, 48, 10, 2).double()
    with torch.no_grad():
        out = head(torch.randn(8, 16, dtype=torch.float64))
    assert torch.allclose(out.forecast.mean(-1), out.trend.mean(-1), atol=1e-10)
