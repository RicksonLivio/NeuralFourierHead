"""
Figura do Capítulo 5 da versão 3: RMSE de teste de cada modelo relativo à MELHOR das três previsões
ingênuas de cada conjunto, (RMSE_modelo - RMSE_ingênuo) / RMSE_ingênuo, em %, com média e desvio sobre as
cinco sementes; painéis (a) univariados e (b) multivariados. Valores negativos = erro menor que o do
melhor ingênuo. Paleta e padrão da tese (sem título embutido).
Saída: versao3_correcao2_aplicada/figuras/graficos/fig_rmse_relativo_v3.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from v3_common import BACKBONES, FONTES, ROOT  # noqa: E402
from v3_tabelas import NAIVES, ROT, carregar_runs, naive  # noqa: E402

FIG = ROOT / "versao3_correcao2_aplicada" / "figuras" / "graficos"
PNG = Path.home() / ".cache" / "v2work" / "figs_v3"
DEEP, TEALD, BLUEG, MAUVE, LILAC = "#270a33", "#3d7b80", "#8daec3", "#725b75", "#81749c"
plt.rcParams.update({"font.size": 10.5, "text.color": DEEP, "axes.edgecolor": DEEP, "axes.labelcolor": DEEP,
                     "xtick.color": DEEP, "ytick.color": DEEP})
HATCH = {"MLP": "", "LSTM": "///", "Conv1D+LSTM": "..."}


def rel_seeds(df, conj, modelo, variante, K, ref):
    s = df[(df.conjunto == conj) & (df.modelo == modelo) & (df.variante == variante) & (df.K == K) & (df.seed >= 0)]
    v = 100 * (s.RMSE.values - ref) / ref
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0) if len(v) else (np.nan, np.nan)


def main():
    df = carregar_runs()
    series = [("N-BEATS-I", "", -1, MAUVE, "")]
    for bb in BACKBONES:
        series += [(f"{bb} + densa", "", -1, BLUEG, HATCH[bb]), (f"{bb} + NFH", "seguro", 10, TEALD, HATCH[bb])]
    fig, axs = plt.subplots(1, 2, figsize=(13.0, 4.6), sharey=True)
    largura = 0.11
    for ax, modo, letra in zip(axs, ("U", "M"), ("a", "b")):
        conjs = [f"{f}-{modo}" for f in FONTES]
        x = np.arange(len(conjs))
        for i, (mod, var, K, cor, hatch) in enumerate(series):
            vals = [rel_seeds(df, c, mod, var, K, np.nanmin([naive(df, c, k) for k in NAIVES])) for c in conjs]
            m = [v[0] for v in vals]
            e = [v[1] for v in vals]
            ax.bar(x + (i - (len(series) - 1) / 2) * largura, m, largura, yerr=e, color=cor, hatch=hatch,
                   edgecolor=DEEP, lw=0.5, error_kw={"lw": 0.8, "capsize": 1.5, "ecolor": DEEP})
        ax.axhline(0, color=DEEP, lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([ROT[c.split("-")[0]] for c in conjs])
        ax.grid(axis="y", alpha=0.25, color=LILAC)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.text(0.01, 0.98, f"({letra}) {'univariados' if modo == 'U' else 'multivariados'}", transform=ax.transAxes,
                va="top", ha="left", fontsize=10.5)
    import matplotlib.ticker as mticker
    for ax in axs:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:g}".replace(".", ",").replace("-", "\u2212")))
    axs[0].set_ylabel("RMSE relativo ao melhor ingênuo (%)")
    leg = [Patch(facecolor=MAUVE, edgecolor=DEEP, label="N-BEATS-I"), Patch(facecolor=BLUEG, edgecolor=DEEP, label="Camada densa"),
           Patch(facecolor=TEALD, edgecolor=DEEP, label="NFH")]
    leg += [Patch(facecolor="white", edgecolor=DEEP, hatch=HATCH[bb], label=bb) for bb in BACKBONES]
    fig.legend(handles=leg, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_rmse_relativo_v3.pdf", bbox_inches="tight")
    fig.savefig(PNG / "fig_rmse_relativo_v3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("ok fig_rmse_relativo_v3")


if __name__ == "__main__":
    main()
