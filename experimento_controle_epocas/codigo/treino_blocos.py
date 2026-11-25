"""
treino_blocos.py: cópia local de `train_model` (src/v2_common.py) com UMA diferença: a validação de cada época é
feita em blocos de 1024 janelas (menos memória de GPU). Mesma lógica do original linha a linha (semente, gerador do
DataLoader, Adam, MSE, clip, parada por objetivo de validação com tolerância 1e-7, restauração do melhor estado).
Cópia de experimento_backbones_nfh/codigo/treino_bb.py (que não é alterado nem importado, para este experimento não
depender do outro). Só muda a ordem das contas da validação (últimos algarismos do float32).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from v2_common import CONFIG, device, residual_penalty, set_seed, stack_ds  # noqa: E402

BLOCO_VAL = 1024


@torch.no_grad()
def saida_em_blocos(model, X, bloco=BLOCO_VAL):
    """Passa X (já no dispositivo) pelo modelo em blocos e concatena 'forecast' e 'residual' (se existir)."""
    fs, rs = [], []
    for i in range(0, len(X), bloco):
        o = model(X[i:i + bloco])
        fs.append(o["forecast"])
        if "residual" in o:
            rs.append(o["residual"])
    out = {"forecast": torch.cat(fs)}
    if rs:
        out["residual"] = torch.cat(rs)
    return out


def train_model_blocos(model, tr_ds, va_ds, seed: int, lam: float = 0.0, cfg=CONFIG, dev=None, verbose=False,
                       bloco=BLOCO_VAL):
    """Igual a v2_common.train_model, com a validação em blocos de `bloco` janelas."""
    dev = dev or device()
    set_seed(seed)
    model = model.to(dev)
    g = torch.Generator()
    g.manual_seed(seed)
    tl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, generator=g)
    Xv, Yv = stack_ds(va_ds)
    Xv, Yv = Xv.to(dev), Yv.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()
    best, best_state, bad, hist = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(cfg.max_epochs):
        model.train()
        tr_losses = []
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = model(x)
            loss = mse(out["forecast"], y)
            if lam > 0:
                loss = loss + lam * residual_penalty(out, cfg.horizon)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tr_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            ov = saida_em_blocos(model, Xv, bloco)          # única diferença para train_model
            v = mse(ov["forecast"], Yv).item()
            if lam > 0:
                v_obj = v + lam * float(residual_penalty(ov, cfg.horizon))
            else:
                v_obj = v
        hist.append((float(np.mean(tr_losses)), v))
        if v_obj < best - 1e-7:
            best, bad = v_obj, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if verbose:
            print(ep, hist[-1])
        if bad >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"epochs": len(hist), "train_time_s": time.time() - t0, "history": hist}
