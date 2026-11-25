"""
Diagramas da versão 3 (mesmo estilo e paleta de src/v2_diagrams.py; sem título embutido).
  fig_nfh_arquitetura   : Figura 4 corrigida (itens M28-M33 do orientador): entrada X e backbone Phi;
                          h ligado diretamente a theta_T, theta_S e u (sem a caixa "Camada afim").
  fig_nbeats_arquitetura: arquitetura do N-BEATS-I (item M17).
  fig_divisao_temporal  : divisão cronológica treino/validação/teste de cada conjunto (item M44).
Saída: versao3_correcao2_aplicada/figuras/imagens/*.pdf (PNG de conferência em ~/.cache/v2work/figs_v3/).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "versao3_correcao2_aplicada" / "figuras" / "imagens"
PNG = Path.home() / ".cache" / "v2work" / "figs_v3"
OUT.mkdir(parents=True, exist_ok=True)
PNG.mkdir(parents=True, exist_ok=True)

DEEP, WINE, PLUM = "#270a33", "#451a3e", "#4d3e6b"
TEALD, TEALM, TEALL = "#3d7b80", "#77b1a9", "#b1e6d1"
MAUVE, LILAC, ROSE = "#725b75", "#81749c", "#c1a2a0"
BLUEG, CYAN, SAND, CREAM = "#8daec3", "#c5dfe0", "#e3cebd", "#fcfdeb"
plt.rcParams.update({"font.size": 11, "text.color": DEEP, "axes.edgecolor": DEEP, "axes.labelcolor": DEEP,
                     "xtick.color": DEEP, "ytick.color": DEEP, "mathtext.fontset": "cm"})


def box(ax, x, y, w, h, text, fc, fs=10, tc=DEEP, lw=1.1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=DEEP, lw=lw, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3)


def arrow(ax, p, q, color=DEEP, lw=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, color=color, lw=lw, zorder=1))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(PNG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("ok", name)


def nfh_arquitetura():
    """Mesmas caixas, cores e fontes da Figura 4 da v2, com a entrada e o backbone à esquerda
    e sem a caixa intermediária: h alimenta diretamente as três projeções afins."""
    # figura mais estreita e fontes maiores: impressa a 0,95 da largura do texto, o texto fica com ~8 pt
    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    ax.axis("off")
    fs = 11.5
    box(ax, 0.0, 2.2, 1.45, 1.2, "$\\mathbf{X}\\in\\mathbb{R}^{L\\times d}$\njanela de\nentrada", CYAN, fs)
    arrow(ax, (1.45, 2.8), (1.8, 2.8))
    box(ax, 1.8, 2.2, 1.2, 1.2, "$\\Phi$\n$\\mathit{backbone}$", BLUEG, fs)
    arrow(ax, (3.0, 2.8), (3.35, 2.8))
    box(ax, 3.35, 2.2, 2.2, 1.2, "$\\mathbf{h}\\in\\mathbb{R}^{p}$\nrepresentação\nlatente", LILAC, fs)
    rows = [(4.2, "$\\boldsymbol{\\theta}_T=\\mathbf{W}_T\\mathbf{h}+\\mathbf{b}_T$\n$\\in\\mathbb{R}^{q+1}$",
             "$\\mathbf{T}=\\mathbf{P}^{\\top}\\boldsymbol{\\theta}_T\\in\\mathcal{T}$\n(tendência)"),
            (2.25, "$\\boldsymbol{\\theta}_S=\\mathbf{W}_S\\mathbf{h}+\\mathbf{b}_S$\n$(a_k,b_k)_{k=1}^{K}\\in\\mathbb{R}^{2K}$",
             "$\\mathbf{S}=\\widetilde{\\mathbf{F}}^{\\top}\\boldsymbol{\\theta}_S\\in\\mathcal{S}$\n(sazonalidade)"),
            (0.3, "$\\mathbf{u}=\\mathbf{W}_R\\mathbf{h}+\\mathbf{b}_R$\n$\\in\\mathbb{R}^{H-m}$",
             "$\\mathbf{R}=\\mathbf{Q}_{\\perp}\\mathbf{u}\\in\\mathcal{R}$\n(resíduo)")]  # \bQ = Q_perp no texto
    x1, x2, x3 = 6.0, 9.05, 11.95
    for y, t1, t2 in rows:
        box(ax, x1, y, 2.75, 1.1, t1, CYAN, fs)
        arrow(ax, (5.55, 2.8), (x1, y + 0.55))
        box(ax, x2, y, 2.55, 1.1, t2, TEALL, fs)
        arrow(ax, (x1 + 2.75, y + 0.55), (x2, y + 0.55))
        arrow(ax, (x2 + 2.55, y + 0.55), (x3, 2.8))
    box(ax, x3, 2.2, 1.95, 1.2, "$\\hat{\\mathbf{y}}=\\mathbf{T}+\\mathbf{S}+\\mathbf{R}$\n$\\in\\mathbb{R}^{H}$", TEALD, fs,
        tc="white")
    ax.text(x2 + 1.3, -0.2, "$\\mathcal{T}\\perp\\mathcal{S}\\perp\\mathcal{R}$, $\\ \\mathcal{T}\\oplus\\mathcal{S}\\oplus\\mathcal{R}=\\mathbb{R}^{H}$",
            ha="center", fontsize=fs, color=PLUM)
    ax.set_xlim(-0.1, 14.0)
    ax.set_ylim(-0.5, 5.6)
    save(fig, "fig_nfh_arquitetura")


def nbeats_arquitetura():
    # figura mais estreita e fontes maiores, para ~7 a 8 pt na impressão
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    ax.axis("off")
    box(ax, 0.0, 3.3, 1.4, 1.0, "Janela\n$\\mathbf{x}\\in\\mathbb{R}^{L}$", CYAN, 12)
    pilhas = [(1.9, "Pilha de tendência ($\\ell\\in\\mathcal{L}_T$)\nbase $\\mathbf{T}_{\\mathrm{nb}}$: polinômio de grau $p_{\\mathrm{nb}}$", LILAC),
              (7.45, "Pilha de sazonalidade ($\\ell\\in\\mathcal{L}_S$)\nbase $\\mathbf{S}_{\\mathrm{nb}}$: Fourier, $\\lfloor H/2-1\\rfloor$ harmônicos", BLUEG)]
    xs = []
    for xp, rot, cor in pilhas:
        ax.add_patch(FancyBboxPatch((xp, 2.75), 5.15, 2.25, boxstyle="round,pad=0.03", fc="none", ec=PLUM, lw=1.3,
                                    ls="--"))
        ax.text(xp + 2.575, 5.2, rot, ha="center", va="bottom", fontsize=11.5, color=PLUM)
        for j in range(3):
            xb = xp + 0.2 + j * 1.7
            box(ax, xb, 3.1, 1.4, 1.4, f"Bloco\n4 densas\n+ ReLU\n$\\to\\boldsymbol{{\\theta}}^{{b}},\\boldsymbol{{\\theta}}^{{f}}$", cor, 11)
            xs.append(xb)
    arrow(ax, (1.4, 3.8), (xs[0], 3.8))
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 1.4, 3.8), (b, 3.8))
    ax.text((xs[0] + 1.4 + xs[1]) / 2, 4.62, "$\\mathbf{x}_{\\ell+1}=\\mathbf{x}_{\\ell}-\\hat{\\mathbf{x}}_{\\ell}$",
            ha="center", fontsize=11.5, color=WINE)
    for xb in xs:
        arrow(ax, (xb + 0.7, 3.1), (xb + 0.7, 1.6))
    ax.text(xs[0] + 0.85, 2.25, "$\\hat{\\mathbf{y}}_{\\ell}$", fontsize=12, color=WINE)
    box(ax, xs[0] - 0.1, 0.55, xs[-1] + 1.5 - xs[0] + 0.1, 1.05,
        "Soma das previsões parciais: $\\hat{\\mathbf{y}}=\\sum_{\\ell}\\hat{\\mathbf{y}}_{\\ell}"
        "=\\mathbf{T}_{\\mathrm{nb}}\\boldsymbol{\\theta}_{T,\\mathrm{nb}}+\\mathbf{S}_{\\mathrm{nb}}\\boldsymbol{\\theta}_{S,\\mathrm{nb}}"
        "\\in\\mathbb{R}^{H}$", TEALD, 12, tc="white")
    ax.text((xs[0] + xs[-1] + 1.4) / 2, 0.35,
            "Em cada bloco $\\ell$: $\\mathit{backcast}$ $\\hat{\\mathbf{x}}_{\\ell}=\\sum_i\\theta^{b}_{\\ell,i}\\mathbf{v}^{b}_i$"
            "  e  $\\mathit{forecast}$ $\\hat{\\mathbf{y}}_{\\ell}=\\sum_i\\theta^{f}_{\\ell,i}\\mathbf{v}^{f}_i$",
            fontsize=11.5, ha="center", va="top", color=DEEP)
    ax.set_xlim(-0.1, 12.8)
    ax.set_ylim(-0.35, 6.3)
    save(fig, "fig_nbeats_arquitetura")


def divisao_temporal():
    """Barras proporcionais ao tempo: treino | validação | teste, com datas e número de janelas."""
    def datas_fin(sub, nome):
        d = pd.read_csv(ROOT / "data" / sub / f"{nome}.csv", usecols=["date"])["date"]
        return pd.to_datetime(d.iloc[1:]).reset_index(drop=True)  # retornos começam na 2a data
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from v3_common import carregar, segmentos  # noqa: E402
    linhas = []
    # Conjuntos principais primeiro; as séries financeiras (experimento secundário) por último.
    for rot, conj in (("ETTh1 e ETTh2", "ETTh1-U"), ("Jena 2009–2011", "Jena-U"), ("Air Quality", "AirQ-U")):
        ds = carregar(conj)
        tr, va, te = segmentos(ds)
        dt = pd.to_datetime(pd.Series(ds["datas"]))
        linhas.append((rot, dt, ds["cortes"], [len(tr), len(va), len(te)], "horária"))
    for rot, sub, nome, L, H in (("Criptomoedas (5)", "univariate", "BTC_USD", 60, 30),
                                 ("Ações (5)", "multivariate", "AAPL", 60, 30)):
        dt = datas_fin(sub, nome)
        n = len(dt)
        a, b = int(0.7 * n), int(0.85 * n)
        jan = [a - L - H + 1, b - a - L - H + 1, n - b - L - H + 1]
        linhas.append((rot, dt, (a, b), jan, "diária"))
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    cores = [(TEALD, "Treino"), (LILAC, "Validação"), (SAND, "Teste")]
    for i, (rot, dt, (a, b), jan, freq) in enumerate(linhas):
        y = len(linhas) - 1 - i
        n = len(dt)
        fr = [a / n, (b - a) / n, (n - b) / n]
        x0 = 0.0
        for (cor, nome), f_, nj in zip(cores, fr, jan):
            ax.barh(y, f_, left=x0, color=cor, edgecolor=DEEP, height=0.62, lw=0.7)
            ax.text(x0 + f_ / 2, y, f"{100 * f_:.0f}%\n{nj:,} janelas".replace(",", "."), ha="center", va="center",
                    fontsize=9.5, color="white" if cor == TEALD else DEEP)
            x0 += f_
        ax.text(-0.01, y + 0.13, rot, ha="right", va="center", fontsize=11)
        ax.text(-0.01, y - 0.2, f"{dt.iloc[0]:%d/%m/%Y} a {dt.iloc[-1]:%d/%m/%Y} ({freq})", ha="right", va="center",
                fontsize=9, color=PLUM)
        for c in (a, b):
            ax.text(c / n, y + 0.36, f"{dt.iloc[c]:%d/%m/%Y}", ha="center", va="bottom", fontsize=8.5, color=PLUM)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(linhas) - 0.25)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Fração das observações, em ordem cronológica")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in cores]
    ax.legend(handles, [n for _, n in cores], loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.17), frameon=False)
    save(fig, "fig_divisao_temporal")


if __name__ == "__main__":
    nfh_arquitetura()
    nbeats_arquitetura()
    divisao_temporal()
