"""
Treino e avaliação (mesmo protocolo da dissertação, ``src/v2_common.py::train_model``).

Objetivo:  J_λ = MSE(ŷ, y) + λ · (1/(B H)) Σ_n ‖R_n‖²

* Adam (lr 1e-3, weight decay 1e-5), lotes de 32, recorte de gradiente 1,0.
* Parada antecipada na validação (paciência 10), restaurando a melhor época.
* A penalização λ só afeta a NFH: favorece previsões expressas por tendência e
  ciclos (Proposição da monotonicidade em λ: λ maior → resíduo menor no treino).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import WindowDataset

__all__ = ["TrainConfig", "set_seed", "fit", "predict", "metrics", "naive_forecasts"]


def set_seed(seed: int) -> None:
    """Fixa as sementes de Python, NumPy e PyTorch (CPU e GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 60
    patience: int = 10
    grad_clip: float = 1.0
    lam: float = 0.0
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = False


@dataclass
class FitResult:
    model: nn.Module
    history: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    train_time_s: float = 0.0


def _objective(model, out, y, lam: float, mse: nn.MSELoss) -> torch.Tensor:
    loss = mse(out.forecast, y)
    if lam > 0:
        loss = loss + lam * model.residual_penalty(out)
    return loss


def fit(model: nn.Module, train: WindowDataset, val: WindowDataset, cfg: TrainConfig,
        on_epoch: Optional[Callable[[Dict[str, float]], None]] = None) -> FitResult:
    """Treina com parada antecipada e devolve o modelo na melhor época de validação.

    O critério de parada é o objetivo J_λ na validação (como na dissertação).
    """
    set_seed(cfg.seed)
    dev = torch.device(cfg.device)
    model = model.to(dev)
    g = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, generator=g, drop_last=False)
    Xv, Yv = (t.to(dev) for t in val.tensors())
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()
    best, best_state, bad, best_epoch = float("inf"), None, 0, 0
    history: List[Dict[str, float]] = []
    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = _objective(model, model(x), y, cfg.lam, mse)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            out = model(Xv)
            v_mse = float(mse(out.forecast, Yv))
            v_obj = float(_objective(model, out, Yv, cfg.lam, mse))
        row = {"epoch": epoch, "train_obj": float(np.mean(losses)), "val_mse": v_mse, "val_obj": v_obj}
        history.append(row)
        if on_epoch:
            on_epoch(row)
        if cfg.verbose:
            print(f"  época {epoch:3d}  treino {row['train_obj']:.4f}  val MSE {v_mse:.4f}")
        if v_obj < best - 1e-7:
            best, bad, best_epoch = v_obj, 0, epoch
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return FitResult(model, history, best_epoch, time.time() - t0)


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, batch_size: int = 512, device: Optional[str] = None):
    """Previsões e componentes em lotes; devolve dict de tensores na CPU."""
    dev = torch.device(device) if device else next(model.parameters()).device
    model.eval()
    parts: Dict[str, List[torch.Tensor]] = {}
    for i in range(0, len(X), batch_size):
        out = model(X[i:i + batch_size].to(dev))
        for k, v in out.as_dict().items():
            parts.setdefault(k, []).append(v.detach().cpu())
    return {k: torch.cat(v) for k, v in parts.items()}


def metrics(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """RMSE e MAE sobre todas as janelas e passos."""
    e = np.asarray(pred) - np.asarray(y)
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e)))}


def naive_forecasts(X: np.ndarray, horizon: int, target_idx: int, season: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Baselines ingênuos sobre janelas X (N, L, d) já padronizadas.

    * ``media``      : média de treino (0 na escala padronizada)
    * ``ultimo``     : persistência do último valor observado
    * ``sazonal``    : repete o último período sazonal (ex.: 24 h, 168 h)
    """
    hist = X[:, :, target_idx]
    out = {"media": np.zeros((len(X), horizon)), "ultimo": np.repeat(hist[:, -1:], horizon, axis=1)}
    if season and hist.shape[1] >= season:
        reps = int(np.ceil(horizon / season))
        out[f"sazonal_{season}"] = np.tile(hist[:, -season:], (1, reps))[:, :horizon]
    return out
