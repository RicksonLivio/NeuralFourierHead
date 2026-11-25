"""A biblioteca ``nfh`` e o núcleo do web app (``web_app/backend/nfh_core``) implementam a mesma NFH."""
import importlib.util
from pathlib import Path

import pytest
import torch

from nfh import NeuralFourierHead, build_bases

CORE = Path(__file__).resolve().parents[2] / "web_app" / "backend" / "nfh_core" / "networks.py"


@pytest.mark.skipif(not CORE.exists(), reason="web_app/backend/nfh_core ausente")
def test_library_equals_web_app_core():
    spec = importlib.util.spec_from_file_location("nfh_core_networks", CORE)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    P, F, Ft, Qp = core.nfh_bases(30, 10, 2)
    b = build_bases(30, 10, 2, dtype=torch.float64)
    assert torch.allclose(torch.tensor(Ft), b.Ft, atol=1e-12)
    assert torch.allclose(torch.tensor(Qp @ Qp.T), b.Q @ b.Q.T, atol=1e-12)
    torch.manual_seed(0)
    mine = NeuralFourierHead(64, 30, 10, 2, normalize_columns=True)
    app = core.NeuralFourierHead(64, 30, 10, 2, normalize_columns=True)
    app.lin.load_state_dict(mine.lin.state_dict())
    h = torch.randn(5, 64)
    with torch.no_grad():
        a, o = mine(h), app(h)
    for x, y in ((a.forecast, o["forecast"]), (a.trend, o["trend"]), (a.seasonal, o["seasonal"]),
                 (a.amplitude, o["amplitude"])):
        assert torch.allclose(x, y, atol=1e-5)
