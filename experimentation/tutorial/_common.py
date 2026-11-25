"""Utilidades compartilhadas pelos scripts do tutorial (impressão didática e argumentos)."""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]           # experimentation/
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))               # permite rodar sem instalar o pacote

WIDTH = 92


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--graficos", action="store_true", help="gera os gráficos (PNG) ao final, em results/")
    p.add_argument("--saida", type=Path, default=None, help="pasta de saída (padrão: results/<script>)")
    return p


def header(title: str, subtitle: str = "") -> None:
    print("=" * WIDTH)
    print(title)
    if subtitle:
        print(textwrap.fill(subtitle, WIDTH))
    print("=" * WIDTH)


def step(n: int | str, title: str, explanation: str = "") -> None:
    print()
    print(f"── PASSO {n} · {title} " + "─" * max(0, WIDTH - len(str(n)) - len(title) - 12))
    if explanation:
        for par in textwrap.dedent(explanation).strip().split("\n\n"):
            print(textwrap.fill(" ".join(par.split()), WIDTH))
            print()


def show_matrix(name: str, M, decimals: int = 3, max_cols: int = 12) -> None:
    A = M.detach().cpu().numpy() if isinstance(M, torch.Tensor) else np.asarray(M)
    A = np.atleast_2d(A)
    print(f"{name}  (shape {tuple(A.shape)})")
    cols = A[:, :max_cols]
    for row in cols:
        print("   [" + " ".join(f"{v:+.{decimals}f}" for v in row) + (" …" if A.shape[1] > max_cols else "") + "]")


def check(label: str, value: float, tol: float) -> bool:
    ok = abs(value) <= tol
    print(f"   {'✓' if ok else '✗'} {label}: {value:.2e} (tolerância {tol:.0e})")
    return ok


def pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def out_dir(args, name: str) -> Path:
    d = args.saida or (RESULTS / name)
    d.mkdir(parents=True, exist_ok=True)
    return d
