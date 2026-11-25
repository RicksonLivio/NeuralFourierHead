"""
Procedimento da versão 3 da dissertação (Seção 4.3.7): conversão exata + ajuste estrutural com seleção segura.

    1. Treina-se o modelo com camada DENSA pelo MSE (``training.fit`` com ``head="dense"``, λ = 0).
    2. Converte-se a camada densa EXATAMENTE em NFH (``Forecaster.to_nfh``): mesma previsão, agora com
       coordenadas (θ_T, θ_S, u). Esta é a época 0 = DECOMPOSIÇÃO PURA da previsão densa.
    3. AJUSTE ESTRUTURAL: continua-se o treino do modelo inteiro por E épocas com a perda penalizada
           J = MSE + λ · (1/(B H)) Σ ‖R‖²          (λ = 1000 e taxa 10⁻⁴ na dissertação)
    4. SELEÇÃO SEGURA: entre as épocas ν = 0..E, escolhe-se
           ν* ∈ argmin { J_ν : r_ν ≤ r_0 }        (r = RMSE de validação, J = r² + λ ρ)
       Como a época 0 é viável, r_ν* ≤ r_0 (= RMSE de validação da densa) e J_ν* ≤ J_0: a NFH nunca é pior
       que a densa NA VALIDAÇÃO (Proposição 4.10). No teste não há garantia.

Também se devolve a "estrutura completa" (menor J sem a restrição), para comparação.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import WindowDataset
from .explain import energy_balance
from .training import set_seed

__all__ = ["AdjustConfig", "AdjustResult", "evaluate_structure", "structural_adjustment"]


@dataclass
class AdjustConfig:
    epochs: int = 15          # E
    lr: float = 1e-4
    weight_decay: float = 1e-5
    lam: float = 1000.0
    batch_size: int = 32
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = False


@dataclass
class AdjustResult:
    model: nn.Module                       # NFH da seleção segura
    selected_epoch: int                    # ν*
    rmse_val_dense: float                  # r_0 (época 0 = densa convertida)
    trajectory: List[Dict[str, float]] = field(default_factory=list)   # (época, r, J, κ, viável)
    full_structure_state: Optional[Dict] = None                         # pesos da "estrutura completa"
    full_structure_epoch: int = 0
    time_s: float = 0.0

    @property
    def selected(self) -> Dict[str, float]:
        return self.trajectory[self.selected_epoch]


@torch.no_grad()
def evaluate_structure(model: nn.Module, X: torch.Tensor, Y: torch.Tensor, lam: float, batch: int = 1024) -> Dict:
    """RMSE, J = RMSE² + λ ρ (ρ = média ‖R‖²/H) e cobertura mediana κ, na escala da saída do modelo."""
    model.eval()
    dev = next(model.parameters()).device
    outs = [model(X[i:i + batch].to(dev)) for i in range(0, len(X), batch)]
    P = torch.cat([o.forecast for o in outs]).double()
    R = torch.cat([o.residual for o in outs]).double()
    rmse = float(torch.sqrt(((P - Y.to(dev).double()) ** 2).mean()))
    rho = float((R ** 2).sum(1).mean() / P.shape[1])
    from .components import NFHOutput

    full = NFHOutput(forecast=P, trend=torch.cat([o.trend for o in outs]).double(),
                     seasonal=torch.cat([o.seasonal for o in outs]).double(), residual=R,
                     theta_T=P[:, :0], theta_S=P[:, :0], u=P[:, :0], amplitude=P[:, :0], phase=P[:, :0])
    kappa = float(np.median(energy_balance(full)["coverage"]))
    return {"rmse": rmse, "J": rmse ** 2 + lam * rho, "rho": rho, "kappa": kappa}


def structural_adjustment(nfh_model: nn.Module, train: WindowDataset, val: WindowDataset,
                          cfg: AdjustConfig = AdjustConfig()) -> AdjustResult:
    """Ajuste estrutural com seleção segura a partir de uma NFH CONVERTIDA (``dense.to_nfh()``)."""
    set_seed(cfg.seed)
    dev = torch.device(cfg.device)
    model = nfh_model.to(dev)
    Xv, Yv = val.tensors()
    ev0 = evaluate_structure(model, Xv, Yv, cfg.lam)
    r0 = ev0["rmse"]
    traj = [{"epoch": 0, **ev0, "feasible": True}]
    safe = {"epoch": 0, "J": ev0["J"], "state": copy.deepcopy(model.state_dict())}
    full = {"epoch": 0, "J": ev0["J"], "state": copy.deepcopy(model.state_dict())}
    loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(cfg.seed))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = mse(out.forecast, y) + cfg.lam * model.residual_penalty(out)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        ev = evaluate_structure(model, Xv, Yv, cfg.lam)
        feasible = ev["rmse"] <= r0                      # restrição da seleção segura
        traj.append({"epoch": ep, **ev, "feasible": feasible})
        if ev["J"] < full["J"]:
            full = {"epoch": ep, "J": ev["J"], "state": copy.deepcopy(model.state_dict())}
        if feasible and ev["J"] < safe["J"]:
            safe = {"epoch": ep, "J": ev["J"], "state": copy.deepcopy(model.state_dict())}
        if cfg.verbose:
            print(f"  ajuste {ep:2d}: RMSE val {ev['rmse']:.4f} (≤ {r0:.4f}? {feasible}) J {ev['J']:.4f} κ {ev['kappa']:.4f}")
    model.load_state_dict(safe["state"])
    model.eval()
    return AdjustResult(model, safe["epoch"], r0, traj, full["state"], full["epoch"], time.time() - t0)
