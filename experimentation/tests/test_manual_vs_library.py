"""A implementação à mão (``nfh.manual``) e a vetorizada (``nfh.bases``/``nfh.head``) coincidem.

Observação: a base ortonormal Q⊥ do resíduo é única apenas a menos de uma
rotação; por isso compara-se o projetor Q⊥Q⊥ᵀ e as componentes T, S, R obtidas
por decomposição de uma previsão, que são únicas (Teorema da identificabilidade).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from nfh import NeuralFourierHead, build_bases
from nfh import manual

CONFIGS = [(30, 10, 2), (24, 6, 1), (8, 2, 1)]


@pytest.mark.parametrize("H,K,q", CONFIGS)
def test_bases_manual_equals_library(H, K, q):
    b = build_bases(H, K, q, dtype=torch.float64)
    v = manual.nfh_vectors(H, K, q)
    assert torch.allclose(torch.stack(v["trend"]), b.P, atol=1e-12)
    assert torch.allclose(torch.stack(v["seasonal"]), b.Ft, atol=1e-10)
    Qm = torch.stack(v["resid"], 1)
    assert torch.allclose(Qm @ Qm.T, b.Q @ b.Q.T, atol=1e-10)
    assert max(manual.orthogonality_report(v)) < 1e-10
    assert np.allclose(manual.column_norms(v)[: b.m], b.D[: b.m].numpy(), atol=1e-12)


@pytest.mark.parametrize("H,K,q", CONFIGS)
def test_forward_and_explanation_manual_equals_library(H, K, q):
    torch.manual_seed(0)
    head = NeuralFourierHead(12, H, K, q, normalize_columns=True).double()
    h = torch.randn(1, 12, dtype=torch.float64)
    with torch.no_grad():
        out = head(h)
    v = manual.nfh_vectors(H, K, q)
    # usa o mesmo Q⊥ da biblioteca para comparar o forward coordenada a coordenada
    v_lib = dict(v, resid=[head.Q[:, j] for j in range(head.Q.shape[1])])
    m = manual.head_forward(h[0], head.lin.weight.detach(), head.lin.bias.detach(), v_lib, head.D.tolist())
    assert torch.allclose(m["forecast"], out.forecast[0], atol=1e-10)
    assert np.allclose(m["amplitude"], out.amplitude[0].numpy(), atol=1e-10)
    # explicação manual (com o Q⊥ próprio, obtido por Gram-Schmidt) = biblioteca
    d = manual.decompose(out.forecast[0], v)
    assert torch.allclose(d["T"], out.trend[0], atol=1e-9)
    assert torch.allclose(d["S"], out.seasonal[0], atol=1e-9)
    assert torch.allclose(d["R"], out.residual[0], atol=1e-9)
    eb = manual.energy_balance(d["T"], d["S"], d["R"])
    assert abs(eb["eps_T"] + eb["eps_S"] + eb["eps_R"] - 1) < 1e-10 and abs(eb["cross"]) < 1e-10
    ek = manual.harmonic_shares(d["theta_S"], v, out.forecast[0])
    assert abs(sum(ek) - eb["eps_S"]) < 1e-10


THESIS_SRC = Path(__file__).resolve().parents[2] / "src" / "v2_common.py"


@pytest.mark.skipif(not THESIS_SRC.exists(), reason="código da dissertação (src/v2_common.py) indisponível")
def test_library_equals_thesis_reference():
    """Mesmas bases e mesma saída que a NFHHead usada nos experimentos da dissertação."""
    pytest.importorskip("pandas")
    sys.path.insert(0, str(THESIS_SRC.parent))
    try:
        import v2_common as v2
    except Exception as exc:  # pragma: no cover - dependências extras do código da tese
        pytest.skip(f"não foi possível importar v2_common: {exc}")
    P, F, Ft, B, Qp = v2.nfh_bases(30, 10, 2)
    b = build_bases(30, 10, 2, dtype=torch.float64)
    assert np.allclose(b.P.numpy(), P) and np.allclose(b.Ft.numpy(), Ft, atol=1e-12)
    assert np.allclose((b.Q @ b.Q.T).numpy(), Qp @ Qp.T, atol=1e-12)
    torch.manual_seed(0)
    mine = NeuralFourierHead(128, 30, 10, 2, normalize_columns=True)
    ref = v2.NFHHead(128, 30, 10, 2, normalize=True)
    ref.lin.load_state_dict(mine.lin.state_dict())
    h = torch.randn(8, 128)
    with torch.no_grad():
        a, r = mine(h), ref(h)
    assert torch.allclose(a.trend, r["trend"], atol=1e-5)
    assert torch.allclose(a.seasonal, r["seasonal"], atol=1e-5)
    assert torch.allclose(a.amplitude, r["amplitude"], atol=1e-5)
