"""
Gráficos (matplotlib). Todas as funções devolvem a ``Figure`` e, se ``path`` for
informado, salvam em PNG (150 dpi). Paleta do projeto: roxo (tendência), rosa
(sazonalidade), azul-lavanda (resíduo), verde (real).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # funciona sem display; use plt.show() por conta própria em notebooks
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .bases import NFHBases  # noqa: E402
from .explain import Explanation  # noqa: E402

__all__ = ["plot_bases", "plot_decomposition", "plot_energy", "plot_harmonics", "plot_attribution",
           "plot_learning_curves", "plot_comparison", "COLORS"]

COLORS = {"trend": "#744b93", "seasonal": "#c889b5", "residual": "#736eae", "true": "#1f9d55",
          "forecast": "#2a1f33", "history": "#9a90b5", "dense": "#e67e22"}


def _save(fig, path: Optional[str | Path]):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_bases(b: NFHBases, path=None, harmonics: Sequence[int] = (1, 2, 5)):
    """Linhas de P, F e F̃ (mostra o efeito de remover a parte polinomial)."""
    tau = np.arange(b.horizon)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for i in range(b.degree + 1):
        ax[0].plot(tau, b.P[i].numpy(), lw=2, label=f"τ^{i}")
    ax[0].set_title("P: tendência (polinômios)")
    for k in harmonics:
        ax[1].plot(tau, b.F[2 * (k - 1)].numpy(), lw=1.8, label=f"cos k={k}")
        ax[2].plot(tau, b.Ft[2 * (k - 1)].numpy(), lw=1.8, label=f"cos k={k}")
    ax[1].set_title("F: senoides originais")
    ax[2].set_title("F̃ = F(I − Π_T): sem parte polinomial")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
        a.set_xlabel("passo j do horizonte")
    fig.tight_layout()
    return _save(fig, path)


def plot_decomposition(exp: Explanation, i: int = 0, history: Optional[np.ndarray] = None,
                       truth: Optional[np.ndarray] = None, title: str = "", ylabel: str = "", path=None):
    """Previsão, real e componentes T, S, R (forma centrada no painel de baixo)."""
    H = exp.forecast.shape[1]
    t = np.arange(1, H + 1)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=False, gridspec_kw={"height_ratios": [1.3, 1]})
    if history is not None:
        L = len(history)
        ax[0].plot(np.arange(-L + 1, 1), history, color=COLORS["history"], lw=1.4, label="histórico")
    if truth is not None:
        ax[0].plot(t, truth, color=COLORS["true"], lw=1.8, ls="--", label="real")
    ax[0].plot(t, exp.forecast[i], color=COLORS["forecast"], lw=2.4, label="previsão NFH")
    ax[0].plot(t, exp.trend[i], color=COLORS["trend"], lw=1.6, ls=":", label="tendência T (com nível)")
    ax[0].axvline(0.5, color="#999", lw=0.8)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title(title or "Previsão e tendência")
    ax[0].legend(fontsize=8, ncol=4)
    ax[0].grid(alpha=0.3)
    lvl = exp.level[i]
    ax[1].plot(t, exp.trend[i] - lvl, color=COLORS["trend"], lw=2, label=f"T − nível (ε_T {100 * exp.eps_T[i]:.1f}%)")
    ax[1].plot(t, exp.seasonal[i], color=COLORS["seasonal"], lw=2, label=f"S (ε_S {100 * exp.eps_S[i]:.1f}%)")
    ax[1].plot(t, exp.residual[i], color=COLORS["residual"], lw=1.6, label=f"R (ε_R {100 * exp.eps_R[i]:.1f}%)")
    ax[1].axhline(0, color="#999", lw=0.8)
    ax[1].set_title("Forma da previsão: componentes ortogonais")
    ax[1].set_xlabel("passo à frente")
    ax[1].legend(fontsize=8, ncol=3)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def plot_energy(exp: Explanation, idx: Optional[Sequence[int]] = None, labels: Optional[List[str]] = None, path=None):
    """Barras empilhadas ε_T / ε_S / ε_R (somam 100%)."""
    idx = list(range(min(len(exp.eps_T), 12))) if idx is None else list(idx)
    labels = labels or [f"#{i}" for i in idx]
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(idx) + 2), 3.4))
    left = np.zeros(len(idx))
    for key, c, lab in (("eps_T", "trend", "ε_T"), ("eps_S", "seasonal", "ε_S"), ("eps_R", "residual", "ε_R")):
        v = 100 * getattr(exp, key)[idx]
        ax.bar(labels, v, bottom=left, color=COLORS[c], label=lab)
        left += v
    ax.set_ylim(0, 100)
    ax.set_ylabel("% da energia da forma")
    ax.set_title("Balanço de energia (soma = 100%)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_harmonics(exp: Explanation, i: int = 0, unit: str = "passos", path=None):
    """Participação e_k por harmônico (barras negativas = cancelamento parcial)."""
    e = 100 * exp.shares[i]
    labels = [f"k={k + 1}\n{exp.periods[k]:.1f}" for k in range(len(e))]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(labels, e, color=[COLORS["seasonal"] if v >= 0 else "#e8467c" for v in e])
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_ylabel("e_k (% da forma)")
    ax.set_title(f"Participação de cada harmônico (Σ e_k = ε_S = {100 * exp.eps_S[i]:.1f}%) — período em {unit}")
    fig.tight_layout()
    return _save(fig, path)


def plot_attribution(attr: Dict[str, Dict], path=None, components: Sequence[str] = ("T", "S")):
    """Participação π_c(v) de cada variável nas componentes (Integrated Gradients)."""
    names = list(attr["T"]["share"].keys())
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(names) + 1.6))
    w = 0.8 / len(components)
    cols = {"T": COLORS["trend"], "S": COLORS["seasonal"], "R": COLORS["residual"]}
    labs = {"T": "tendência (g_T)", "S": "sazonalidade (g_S)", "R": "resíduo (g_R)"}
    for j, c in enumerate(components):
        ax.barh(y + j * w, [100 * attr[c]["share"][n] for n in names], height=w, color=cols[c], label=labs[c])
    ax.set_yticks(y + w * (len(components) - 1) / 2, names)
    ax.invert_yaxis()
    ax.set_xlabel("participação π_c(v) (%)")
    err = max(attr[c]["completeness_error"] for c in components)
    ax.set_title(f"Que variáveis sustentam cada componente (erro de completude ≤ {err:.1e})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_learning_curves(histories: Dict[str, List[Dict]], path=None):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for name, h in histories.items():
        ax.plot([r["epoch"] for r in h], [r["val_mse"] for r in h], lw=1.8, label=name)
    ax.set_xlabel("época")
    ax.set_ylabel("MSE de validação")
    ax.set_title("Curvas de aprendizado")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_comparison(table, metric: str = "RMSE", path=None, title: str = ""):
    """Barras com média ± desvio entre sementes. ``table``: DataFrame com colunas model, mean, std."""
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(table) + 1.4))
    colors = [COLORS["trend"] if "NFH" in m else (COLORS["dense"] if "densa" in m else "#9a90b5")
              for m in table["model"]]
    ax.barh(table["model"], table["mean"], xerr=table["std"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric} de teste (escala padronizada)")
    ax.set_title(title or f"{metric} de teste (média ± desvio entre sementes)")
    lo = float((table["mean"] - table["std"]).min())
    ax.set_xlim(left=max(0.0, lo * 0.9))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)
