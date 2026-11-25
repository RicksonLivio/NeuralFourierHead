"""
Preparação de dados para previsão multi-passo sem vazamento temporal.

Três salvaguardas (as mesmas da dissertação):

1. **Divisão cronológica** treino / validação / teste (padrão 70/15/15), nunca aleatória.
2. **Padronização com estatísticas só do treino** (média e desvio estimados no
   trecho de treino e aplicados aos três trechos).
3. **Janelas dentro do próprio trecho**: entrada (x_{t-L+1..t}) e alvo
   (y_{t+1..t+H}) nunca atravessam a fronteira entre trechos.

``load_bike_sharing`` prepara o conjunto UCI Bike Sharing (horário), usado no
tutorial: contagem de aluguéis por hora com clima e calendário.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = ["WindowDataset", "SplitData", "make_splits", "load_bike_sharing", "BIKE_FEATURES"]

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class WindowDataset(Dataset):
    """Janelas deslizantes (X: (L, d), y: (H,)) dentro de um único trecho."""

    def __init__(self, features: np.ndarray, target: np.ndarray, lookback: int, horizon: int, stride: int = 1):
        self.f = torch.as_tensor(np.ascontiguousarray(features), dtype=torch.float32)
        self.t = torch.as_tensor(np.ascontiguousarray(target), dtype=torch.float32)
        self.L, self.H, self.stride = lookback, horizon, stride
        self.n = max(0, (len(features) - lookback - horizon) // stride + 1)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        s = i * self.stride
        return self.f[s:s + self.L], self.t[s + self.L:s + self.L + self.H]

    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Todas as janelas empilhadas: X (N, L, d), Y (N, H)."""
        idx = torch.arange(self.n) * self.stride
        X = self.f[idx[:, None] + torch.arange(self.L)]
        Y = self.t[idx[:, None] + self.L + torch.arange(self.H)]
        return X, Y


@dataclass
class SplitData:
    """Resultado de ``make_splits``: datasets + metadados para desnormalizar."""

    train: WindowDataset
    val: WindowDataset
    test: WindowDataset
    feature_names: List[str]
    target_name: str
    target_idx: int
    mean: np.ndarray            # (d,) média de treino de cada variável
    std: np.ndarray             # (d,)
    bounds: Tuple[int, int]     # índices de fim do treino e da validação
    timestamps: Optional[pd.DatetimeIndex] = None
    info: Dict = field(default_factory=dict)

    def to_original(self, y_std: np.ndarray) -> np.ndarray:
        """Leva valores do alvo padronizado de volta à escala original."""
        return y_std * self.std[self.target_idx] + self.mean[self.target_idx]

    @property
    def target_std(self) -> float:
        return float(self.std[self.target_idx])


def make_splits(df: pd.DataFrame, target: str, features: Sequence[str], lookback: int, horizon: int,
                train_ratio: float = 0.70, val_ratio: float = 0.15, test_stride: int = 1,
                standardize: Sequence[str] | None = None) -> SplitData:
    """Divide, padroniza (estatísticas de treino) e janela um DataFrame ordenado no tempo.

    Args:
        df: tabela ordenada cronologicamente.
        target: coluna prevista (deve estar em ``features``: o backbone também a vê).
        features: colunas de entrada.
        standardize: colunas a padronizar (padrão: todas). Variáveis já em [0, 1]
            ou cíclicas (sen/cos) podem ficar de fora.
        test_stride: passo entre janelas de teste (1 = todas).
    """
    features = list(features)
    if target not in features:
        raise ValueError("O alvo deve fazer parte das variáveis de entrada.")
    raw = df[features].to_numpy(dtype=np.float64)
    n = len(raw)
    a, b = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    std_cols = set(features if standardize is None else standardize) | {target}
    mean = np.zeros(len(features))
    std = np.ones(len(features))
    for j, c in enumerate(features):
        if c in std_cols:
            mean[j] = raw[:a, j].mean()
            std[j] = raw[:a, j].std() + 1e-12
    z = ((raw - mean) / std).astype(np.float32)
    ti = features.index(target)
    y = z[:, ti]
    ts = df.index if isinstance(df.index, pd.DatetimeIndex) else None
    return SplitData(
        train=WindowDataset(z[:a], y[:a], lookback, horizon),
        val=WindowDataset(z[a:b], y[a:b], lookback, horizon),
        test=WindowDataset(z[b:], y[b:], lookback, horizon, stride=test_stride),
        feature_names=features, target_name=target, target_idx=ti, mean=mean, std=std, bounds=(a, b),
        timestamps=ts, info={"n": n, "train_ratio": train_ratio, "val_ratio": val_ratio},
    )


# --------------------------------------------------------------------------- #
# UCI Bike Sharing (hourly)
# --------------------------------------------------------------------------- #
BIKE_FEATURES = ["cnt", "temp", "atemp", "hum", "windspeed", "weathersit", "workingday", "holiday",
                 "hr_sin", "hr_cos", "wd_sin", "wd_cos"]


def load_bike_sharing(path: Optional[Path] = None, log_target: bool = True) -> pd.DataFrame:
    """Lê ``hour.csv``, completa as horas ausentes e cria variáveis de calendário.

    Fonte: Fanaee-T, H. & Gama, J. (2014), Event labeling combining ensemble
    detectors and background knowledge, Progress in Artificial Intelligence 2, 113–127.
    UCI Machine Learning Repository, dataset 275, licença CC BY 4.0.

    * 17.379 registros horários (2011-01-01 a 2012-12-31); faltam 165 horas, que
      são inseridas com contagem 0 (o sistema não registrou aluguéis) e clima
      preenchido para frente.
    * ``cnt`` (aluguéis por hora) vira log1p(cnt) se ``log_target`` (reduz a
      assimetria; desfaça com ``np.expm1``).
    * hora do dia e dia da semana entram como seno/cosseno (ciclos de 24 h e 7 dias).
    """
    path = path or DATA_DIR / "bike_sharing" / "hour.csv"
    d = pd.read_csv(path, parse_dates=["dteday"])
    d.index = d["dteday"] + pd.to_timedelta(d["hr"], unit="h")
    full = pd.date_range(d.index.min(), d.index.max(), freq="h")
    d = d.reindex(full)
    d["cnt"] = d["cnt"].fillna(0.0)
    for c in ("temp", "atemp", "hum", "windspeed", "weathersit", "workingday", "holiday"):
        d[c] = d[c].ffill().bfill()
    hr, wd = d.index.hour.to_numpy(), d.index.dayofweek.to_numpy()
    d["hr_sin"], d["hr_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    d["wd_sin"], d["wd_cos"] = np.sin(2 * np.pi * wd / 7), np.cos(2 * np.pi * wd / 7)
    d["cnt_raw"] = d["cnt"]
    if log_target:
        d["cnt"] = np.log1p(d["cnt"])
    return d[BIKE_FEATURES + ["cnt_raw"]].astype(float)
