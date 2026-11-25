"""Salvaguardas contra vazamento temporal e carga do conjunto Bike Sharing."""
import numpy as np
import pandas as pd
import pytest

from nfh.data import BIKE_FEATURES, DATA_DIR, load_bike_sharing, make_splits


def test_splits_are_chronological_and_use_train_stats_only():
    n = 1000
    df = pd.DataFrame({"y": np.arange(n, dtype=float), "x": np.linspace(0, 1, n)})
    s = make_splits(df, "y", ["y", "x"], lookback=20, horizon=10)
    a, b = s.bounds
    assert (a, b) == (700, 850)
    assert np.isclose(s.mean[0], np.arange(700).mean())  # média só do treino
    X, Y = s.train.tensors()
    # a última janela de treino termina antes do fim do trecho de treino
    assert s.to_original(Y[-1].numpy())[-1] == pytest.approx(699, abs=1e-3)
    Xt, Yt = s.test.tensors()
    assert s.to_original(Xt[0, 0, 0].numpy()) == pytest.approx(850, abs=1e-3)
    assert len(s.train) == 700 - 30 + 1 and len(s.val) == 150 - 30 + 1


@pytest.mark.skipif(not (DATA_DIR / "bike_sharing" / "hour.csv").exists(), reason="hour.csv ausente")
def test_bike_sharing_loader():
    d = load_bike_sharing()
    assert len(d) == 17544  # 2 anos completos de horas, com as 165 ausentes preenchidas
    assert list(d.columns[: len(BIKE_FEATURES)]) == BIKE_FEATURES
    assert d.isna().sum().sum() == 0
    assert np.allclose(np.expm1(d["cnt"]), d["cnt_raw"])
    assert d.index.is_monotonic_increasing and d.index.freqstr in ("h", "H")
