"""Etapa 5: escreve exemplo_passo_a_passo.tex (e figuras) a partir de passo_a_passo.json, confirmacao.json/.csv,
selecao.json/.csv, e compila com tectonic. Nenhum número do texto é digitado: todos saem desses arquivos."""
import json
import math
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
FIG = AQUI / "figuras"
FIG.mkdir(exist_ok=True)
d = json.load(open(AQUI / "passo_a_passo.json"))
cf = json.load(open(AQUI / "confirmacao.json"))
sl = json.load(open(AQUI / "selecao.json"))
conf = pd.read_csv(AQUI / "confirmacao.csv")
selc = pd.read_csv(AQUI / "selecao.csv", usecols=["p", "n", "semente", "variante", "val_RMSE", "val_MAE"])  # só val_*
proj = pd.read_csv(AQUI / "projecao_confirmacao.csv")           # controle adicional (projetar_confirmacao.py)
pex = json.load(open(AQUI / "protocolo_exemplo.json"))          # E, taxas, lote, épocas, paciência do exemplo
cfg16 = json.load(open(AQUI.parents[1] / "teste_exemplificacao_nfh" / "exemplo_livro" / "config.json"))  # série do teste de 16/11
_ins = (AQUI.parent / "INSTRUCOES.md").read_text(encoding="utf-8")
_m1 = re.search(r"Adam, taxa (1e-3), \*weight decay\* 1e-5, lote (\d+), \*clip\* 1,0\), até (\d+) épocas e paciência (\d+)", _ins)
_m2 = re.search(r"`densa\+E`: a densa treinada por mais E = (\d+) épocas, Adam com taxa (1e-4)", _ins)
assert _m1 and _m2, "protocolo da T1 não encontrado em INSTRUCOES.md"
T1P = dict(lr=_m1.group(1), lote=_m1.group(2), ep=_m1.group(3), pac=_m1.group(4), E=_m2.group(1), lraj=_m2.group(2))
A = lambda x: np.asarray(x, dtype=float)  # noqa: E731

# ------------------------------------------------------------------ formatação
def fmt(x, sig=4):
    """Número com `sig` algarismos significativos, vírgula decimal (o pacote icomma cuida do espaço), seguro em modo matemático."""
    x = float(x)
    if x == 0 or (sig != 2 and abs(x) < 1e-13):   # sig == 2 é reservado a erros numéricos, que nunca são zerados
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    if abs(x) < 1e-3 or abs(x) >= 1e5:
        m = x / 10 ** e
        ms = f"{m:.{sig - 1}f}".replace(".", ",")
        return f"{ms}\\times 10^{{{e}}}"
    dec = max(0, sig - 1 - e)
    return f"{x:.{dec}f}".replace(".", ",")


def N(x, sig=4):
    return "\x01" + fmt(x, sig) + "\x02"


def N4(x):
    return "\x01" + f"{float(x):.4f}".replace(".", ",") + "\x02"


def pct(x, sig=3):
    return "\x01" + fmt(100 * float(x), sig) + "\\,\\%\x02"


def finalizar(doc):
    """Marcadores \x01..\x02: fora do modo matemático viram $...$; dentro (cifrão aberto, \\[, align) são removidos."""
    import re
    saida, mat_, i = [], False, 0
    tokens = re.split(r"(\\\$|\$|\\\[|\\\]|\\begin\{align\*\}|\\end\{align\*\}|\x01|\x02)", doc)
    for t in tokens:
        if t == "\\$":
            saida.append(t)
        elif t == "$":
            mat_ = not mat_; saida.append(t)
        elif t in ("\\[", "\\begin{align*}"):
            mat_ = True; saida.append(t)
        elif t in ("\\]", "\\end{align*}"):
            mat_ = False; saida.append(t)
        elif t in ("\x01", "\x02"):
            saida.append("" if mat_ else "$")
        else:
            saida.append(t)
    return "".join(saida)


def vec(v, sig=4):
    return "(" + ";\\allowbreak\\ ".join(fmt(t, sig) for t in v) + ")"


def mat(M_, sig=4):
    return "\\begin{bmatrix}" + " \\\\ ".join(" & ".join(fmt(t, sig) for t in r) for r in M_) + "\\end{bmatrix}"


def tex(_t, **kw):
    for k, v in kw.items():
        _t = _t.replace(f"<<{k}>>", str(v))
    assert "<<" not in _t, _t[_t.index("<<"):_t.index("<<") + 40]
    return _t


def tabela(legenda, rotulo, cab, linhas, spec, fonte=None, escala=None, tam="\\small"):
    """Tabela ABNT: título acima, fonte abaixo. `escala` = largura (resizebox) para tabelas largas."""
    t = ["\\begin{table}[H]", "\\centering", f"\\caption{{{legenda}}}\\label{{{rotulo}}}", tam]
    corpo = ["\\begin{tabular}{" + spec + "}", "\\toprule", " & ".join(cab) + " \\\\", "\\midrule"]
    corpo += [" & ".join(str(c) for c in r) + " \\\\" for r in linhas]
    corpo += ["\\bottomrule", "\\end{tabular}"]
    if escala:
        corpo = ["\\resizebox{" + escala + "}{!}{%"] + corpo + ["}"]
    t += corpo + ["\\par\\vspace{2pt}{\\footnotesize Fonte: elaborado pelo autor" + (f"; {fonte}" if fonte else "") + ".}", "\\end{table}"]
    return "\n".join(t)


def figura(arq, legenda, rotulo, larg="0.85"):
    return "\n".join(["\\begin{figure}[H]", "\\centering", f"\\caption{{{legenda}}}\\label{{{rotulo}}}",
                      f"\\includegraphics[width={larg}\\textwidth]{{figuras/{arq}}}",
                      "\\par\\vspace{2pt}{\\footnotesize Fonte: elaborado pelo autor.}", "\\end{figure}"])

# ------------------------------------------------------------------ dados curtos
cfg = d["configuracao"]; B = d["bases"]; J = d["janela"]; C = d["camada"]; NF = d["nfh"]; V = d["verdade"]; EV = d["erros_nfh_vs_verdade"]
DS = d["densa_mesma_semente"]; CV = d["conversao"]; NB = d["nbeats_i"]; IG = d["integrated_gradients"]; AB = d["abdutiva"]; G = d["gerador"]
H, L, DD, p, q, K = cfg["H"], cfg["L"], cfg["d"], cfg["p"], cfg["q"], cfg["K"]
m = q + 1 + 2 * K
Hm = H - m
tau = A(B["tau"]); Pm, Fm, Ftm, Qm, Mm = A(B["P"]), A(B["F"]), A(B["Ft"]), A(B["Q"]), A(B["M"])
Dv, Nv = A(B["D"]), A(B["normas_D"])
X = A(J["X"]); W = A(C["W"]); bb = A(C["b"]); h = A(C["h"]); cpr = A(C["c_linha"]); c = A(C["c"])
yh = A(NF["previsao"]); Tn, Sn, Rn = A(NF["T"]), A(NF["S"]), A(NF["R"])
To, So, Ro = A(V["T"]), A(V["S"]), A(V["R"])
yor, yre = A(J["oraculo_futuro"]), A(J["y_real_futuro"])
thT, thS, uu = A(NF["theta_T"]), A(NF["theta_S"]), A(NF["u"])
tT0, tS0, u0 = A(V["theta_T"]), A(V["theta_S"]), A(V["u"])
vn = ["x_1", "x_2", "x_3", "x_4"]
cinza = {"preto": "0.0", "cinza escuro": "0.35", "cinza médio": "0.6", "cinza claro": "0.8"}

def cmed(v, k):
    return cf[f"{v}_test_{k}_media"], cf[f"{v}_test_{k}_dp"]

# ------------------------------------------------------------------ figuras (tons de cinza, sem título na imagem)
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
sr = d["janela"]["trecho_serie"]; Xs, ys, ls = A(sr["X"]), A(sr["y"]), A(sr["limpo"])
t0 = sr["inicio"]; tt = np.arange(t0, t0 + len(Xs))
i0 = sr["i0"]
fig, ax = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True)
for k, (g, ls_) in enumerate(zip(["0.0", "0.35", "0.65", "0.8"], ["-", "--", "-", "-."])):
    ax[0].plot(tt, Xs[:, k], ls_, color=g, lw=1.6 if k < 2 else 1.2, label=f"$x_{k + 1}$")
ax[0].axvspan(i0, i0 + L - 0.5, color="0.92", zorder=0)
ax[0].set_ylabel("variáveis de entrada"); ax[0].legend(ncol=4, loc="lower left")
ax[1].plot(tt, ys, "o", ms=3, color="0.6", label="$y$ (com ruído)")
ax[1].plot(tt, ls, "-", color="0.0", lw=1.4, label="$y$ sem ruído")
ax[1].plot(np.arange(i0 + L, i0 + L + H), yh, "s", ms=5, color="0.0", mfc="white", label="previsão da NFH")
ax[1].axvspan(i0, i0 + L - 0.5, color="0.92", zorder=0)
ax[1].axvline(i0 + L - 0.5, color="0.4", lw=0.8, ls=":")
ax[1].set_xlabel("instante $s$ da série"); ax[1].set_ylabel("alvo"); ax[1].legend(loc="lower left", ncol=3)
fig.tight_layout(); fig.savefig(FIG / "fig_serie_janela.pdf"); plt.close(fig)

fig, ax = plt.subplots(3, 1, figsize=(7, 6.2), sharex=True)
jj = np.arange(H)
for a_, (nm, mod, ver) in zip(ax, [("$T$", Tn, To), ("$S$", Sn, So), ("$R$", Rn, Ro)]):
    a_.plot(jj, ver, "o--", color="0.55", ms=5, label="verdade (oráculo)")
    a_.plot(jj, mod, "s-", color="0.0", ms=4, label="NFH")
    a_.set_ylabel(nm)
ax[0].legend(loc="best"); ax[2].set_xlabel("hora do horizonte $j$")
fig.tight_layout(); fig.savefig(FIG / "fig_TSR.pdf"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
w_ = 0.38
for a_, nome, tit in zip(ax, ["g_T", "g_S"], ["(a) $g_T$", "(b) $g_S$"]):
    a_.bar(np.arange(DD) - w_ / 2, A(IG[nome]["participacao"]), w_, color="0.15", label="NFH (IG)")
    a_.bar(np.arange(DD) + w_ / 2, A(IG[nome + "_oraculo"]["participacao"]), w_, color="0.75", edgecolor="0.3", label="verdade (IG do oráculo)")
    a_.set_xticks(range(DD)); a_.set_xticklabels([f"$x_{k + 1}$" for k in range(DD)]); a_.set_xlabel(tit)
ax[0].set_ylabel("participação (fração do total)"); ax[0].legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig_IG.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 3.8))
kk = np.arange(p + 1)
ax.plot(kk, AB["acumulado"], "o-", color="0.0", ms=4, label="ordem gulosa (maior ganho primeiro)")
ax.plot(kk, AB["acumulado_IG"], "s--", color="0.5", ms=4, label="ordem dos Integrated Gradients")
ax.axhline(AB["xi"], color="0.3", lw=1, ls=":"); ax.text(0.2, AB["xi"], "  $\\xi$", va="bottom")
ax.axvline(AB["k_estrela"], color="0.7", lw=1); ax.axhline(0, color="0.8", lw=0.8)
ax.set_xlabel("número $k$ de neurônios fixados"); ax.set_ylabel("pior valor garantido de $s\\,\\theta_{T,1}$")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig_abdutiva.pdf"); plt.close(fig)

# ------------------------------------------------------------------ pré-cálculos do texto
ven = cf["veredito"]
gT, gS, gTo, gSo = IG["g_T"], IG["g_S"], IG["g_T_oraculo"], IG["g_S_oraculo"]
pT, pS, pTo, pSo = A(gT["participacao"]), A(gS["participacao"]), A(gTo["participacao"]), A(gSo["participacao"])
en, ev, ed = NF["energia"], V["energia"], DS["energia"]
nd = lambda i: i + 1  # noqa: E731

def agrup(df):
    out = []
    for (pp, nn), g in df.groupby(["p", "n"]):
        base = g[g.variante == "densa+E"].set_index("semente"); dens = g[g.variante == "densa"].set_index("semente")
        for v in ["conv_L1000", "zero_L0", "zero_L1000"]:
            x = g[g.variante == v].set_index("semente")
            rr = ((x.val_RMSE - base.val_RMSE) / base.val_RMSE).mean(); rm = ((x.val_MAE - base.val_MAE) / base.val_MAE).mean()
            mb = int((x.val_RMSE < base.val_RMSE).sum())
            out.append((int(pp), int(nn), v, x.val_RMSE.mean(), base.val_RMSE.mean(), dens.val_RMSE.mean(), rr, rm, mb, bool(rr < 0 and rm <= 0 and mb == 5)))
    return out

grade = agrup(selc)
n_qual = sum(r[-1] for r in grade)
ordem_gulosa = AB["ordem_gulosa"]; ordem_ig = AB["ordem_IG"]
# comparação zero_L0 x zero_L1000 (só val_*), na combinação escolhida e em n = 300
PPL, NNL = int(sl["p"]), int(sl["n"])
def _zz(pp_, nn_):
    g = selc[(selc.p == pp_) & (selc.n == nn_)]
    a = g[g.variante == "zero_L0"].set_index("semente").val_RMSE; b = g[g.variante == "zero_L1000"].set_index("semente").val_RMSE
    return a.mean(), b.mean(), int((b < a).sum()), len(a)
_a0, _a1, _w, _nn = _zz(PPL, NNL)
_n3 = int(selc.n.min()); _b0, _b1, _w3, _ = _zz(PPL, _n3)
CMPL = dict(a0=_a0, a1=_a1, w=_w, b0=_b0, b1=_b1)
N3L = _n3
CMP_CURTO = "com $p=" + str(PPL) + "$ e $n=" + str(NNL) + "$, RMSE de validação " + N(_a0) + " sem penalização e " + N(_a1) + " com $\\lambda=" + fmt(cfg["lambda"], 4) + "$, menor em " + str(_w) + " das " + str(_nn) + " sementes"
Aset = AB["A"]
kS, kI = AB["k_estrela"], AB["k_IG"]
# a série e as janelas de teste são as do teste de 16/11 (exemplo_livro/config.json)? Só se afirma no texto porque confere.
assert cfg16["semente_dados"] == G["semente_dados"] and cfg16["N"] == G["N"] and cfg16["cortes"] == G["cortes"] and cfg16["sigma"] == cfg["sigma"]
# projeção da densa em T ⊕ S (controle adicional)
assert sorted(proj.semente) == sorted(set(conf.semente))
def _ms(x):
    return float(x.mean()), float(x.std(ddof=1))
PJ = {k: _ms(proj[k]) for k in ("densa_test_RMSE", "densa_test_MAE", "proj_test_RMSE", "proj_test_MAE")}
_nfh = conf[conf.variante == "NFH"].sort_values("semente")
_pj = proj.sort_values("semente")
PJ_ganho_proj = (PJ["densa_test_RMSE"][0] - PJ["proj_test_RMSE"][0]) / PJ["densa_test_RMSE"][0]
PJ_ganho_nfh = (cf["densa_test_RMSE_media"] - cf["NFH_test_RMSE_media"]) / cf["densa_test_RMSE_media"]
PJ_fracao = PJ_ganho_proj / PJ_ganho_nfh                                  # fração da distância densa -> NFH que a projeção percorre
PJ_prm = int((_pj.proj_test_RMSE.values < _pj.densa_test_RMSE.values).sum())
PJ_nfh_vs_proj = int((_nfh.test_RMSE.values < _pj.proj_test_RMSE.values).sum())
PJ_nfh_vs_proj_mae = int((_nfh.test_MAE.values < _pj.proj_test_MAE.values).sum())
PJ_frac_res = float(proj.frac_energia_residual_prev.mean())
PJ_NAO_EXPLICA = PJ_fracao < 0.25

# ------------------------------------------------------------------ documento
S = []
add = S.append
add(r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[brazilian]{babel}
\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage{icomma}
\usepackage{booktabs,longtable,array}
\usepackage{graphicx}
\usepackage[a4paper,margin=2.3cm]{geometry}
\usepackage[hidelinks]{hyperref}
\usepackage{float}
\usepackage[font=small,labelfont=bf,skip=4pt]{caption}
\newcommand{\bh}{\mathbf{h}}\newcommand{\bW}{\mathbf{W}}\newcommand{\bb}{\mathbf{b}}\newcommand{\bc}{\mathbf{c}}
\newcommand{\bP}{\mathbf{P}}\newcommand{\bF}{\mathbf{F}}\newcommand{\bFt}{\widetilde{\mathbf{F}}}\newcommand{\bQ}{\mathbf{Q}}
\newcommand{\bM}{\mathbf{M}}\newcommand{\bD}{\mathbf{D}}\newcommand{\bT}{\mathbf{T}}\newcommand{\bS}{\mathbf{S}}
\newcommand{\bR}{\mathbf{R}}\newcommand{\byh}{\hat{\mathbf{y}}}\newcommand{\bX}{\mathbf{X}}
\newcommand{\thT}{\boldsymbol{\theta}_T}\newcommand{\thS}{\boldsymbol{\theta}_S}\newcommand{\bu}{\mathbf{u}}
\newcommand{\bI}{\mathbf{I}}\newcommand{\R}{\mathbb{R}}\newcommand{\En}{\mathcal{E}}
\DeclareMathOperator{\sen}{sen}
\setlength{\emergencystretch}{3em}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\title{Neural Fourier Head: um exemplo sintético passo a passo,\\com todas as contas à mão}
\author{Experimento \texttt{exemplo\_sintetico} da rodada de 21/11/2026}
\date{}
\begin{document}
\maketitle
""")

# ---- primeira página
rn, rd, re_ = cmed("NFH", "RMSE"), cmed("densa", "RMSE"), cmed("densa+E", "RMSE")
mn, md_, me = cmed("NFH", "MAE"), cmed("densa", "MAE"), cmed("densa+E", "MAE")
nS = len(set(conf.semente))
add(r"\section*{Resultado da confirmação (leia primeiro)}")
if ven == "SUCESSO":
    add(tex(r"""Este exemplo foi construído para mostrar a Neural Fourier Head (NFH) \textbf{num caso favorável e pré-registrado} (regra e critério fixados por escrito antes de rodar). Isso \textbf{não é uma regra geral}: em dados reais da dissertação, a NFH treinada do zero perdeu da camada densa. Aqui, a combinação foi escolhida só pela validação (sementes 0 a 4) e depois reavaliada no teste com as sementes <<s0>> a <<s1>>, que não participaram da seleção. Essas sementes mudam só a inicialização dos pesos e a ordem dos lotes de treino: \textbf{a série e as janelas de teste são as mesmas} da seleção e do teste de exemplificação de 16/11 (mesma semente de dados <<sd>>, $N=<<Nn>>$, cortes <<c1>> e <<c2>>). Por isso o resultado mostra estabilidade entre inicializações, e não um resultado em dados novos. Em cada par ``média $\pm$ desvio'' abaixo, o $\pm$ é o desvio-padrão amostral (denominador $n-1$) entre as <<ns>> sementes, e todos os valores têm quatro casas decimais. RMSE é a raiz do erro quadrático médio das previsões e MAE é o erro absoluto médio, ambos na escala do alvo; menor é melhor. Os números da confirmação estão na Tabela~\ref{tab:prim}. O RMSE de teste da NFH fica <<rel_nfh>> acima do RMSE do oráculo (<<ro>>, a referência amostral das mesmas janelas de teste) e o da camada densa (cabeça linear comum, que escreve direto os valores do horizonte previsto), <<rel_d>> acima. A densa+E é a densa treinada por mais $E=<<E>>$ épocas (a T1 usa outro $E$; ver Seção~\ref{sec:modelo}). A NFH teve RMSE menor que a densa em <<vrd>> das <<ns>> sementes e menor que a densa+E em <<vre>>; no MAE, <<vmd>> e <<vme>>. O critério pré-registrado (médias de RMSE e de MAE da NFH menores ou iguais às da densa e às da densa+E, e RMSE menor que o da densa+E em pelo menos 8 sementes) foi \textbf{cumprido: veredito SUCESSO}. O modelo é pequeno: o backbone (a parte que lê a janela) tem uma camada com $p=<<p>>$ neurônios e <<nb>> parâmetros, e a cabeça tem mais <<npar>>, nas duas variantes, para que todas as contas caibam nas páginas seguintes.""",
        s0=conf.semente.min(), s1=conf.semente.max(), ns=nS, rn=N4(rn[0]), rns=N4(rn[1]), rd=N4(rd[0]), rds=N4(rd[1]), re=N4(re_[0]), res=N4(re_[1]),
        mn=N4(mn[0]), mns=N4(mn[1]), md=N4(md_[0]), mds=N4(md_[1]), me=N4(me[0]), mes=N4(me[1]),
        sd=G["semente_dados"], Nn=G["N"], c1=pct(G["cortes"][0], 2), c2=pct(G["cortes"][1], 2),
        vrd=cf["vitorias_test_RMSE_vs_densa"], vre=cf["vitorias_test_RMSE_vs_densa+E"], vmd=cf["vitorias_test_MAE_vs_densa"],
        vme=cf["vitorias_test_MAE_vs_densa+E"], p=p, npar=cfg["n_parametros_cabeca"], nb=cfg["n_parametros_backbone"], ro=N4(d["oraculo_teste"]["RMSE"]),
        rel_nfh=pct(rn[0] / d["oraculo_teste"]["RMSE"] - 1, 2), rel_d=pct(rd[0] / d["oraculo_teste"]["RMSE"] - 1, 2), E=pex["E_EXTRA"]))
else:
    add(tex(r"""Este exemplo foi construído para mostrar a NFH num caso favorável e pré-registrado, mas \textbf{a confirmação falhou}: nas sementes <<s0>> a <<s1>>, o RMSE de teste foi <<rn>> $\pm$ <<rns>> (NFH), <<rd>> $\pm$ <<rds>> (densa) e <<re>> $\pm$ <<res>> (densa+E); o MAE, <<mn>>, <<md>> e <<me>>. A NFH teve RMSE menor que a densa+E em <<vre>> de <<ns>> sementes. Portanto, \textbf{neste exemplo a NFH não superou a densa} pelo critério pré-registrado. O passo a passo vale como explicação do funcionamento, não como vitória.""",
        s0=conf.semente.min(), s1=conf.semente.max(), ns=nS, rn=N4(rn[0]), rns=N4(rn[1]), rd=N4(rd[0]), rds=N4(rd[1]), re=N4(re_[0]), res=N4(re_[1]),
        mn=N4(mn[0]), md=N4(md_[0]), me=N4(me[0]), vre=cf["vitorias_test_RMSE_vs_densa+E"]))
def _lt(v, k):
    return N4(cf[f"{v}_test_{k}_media"]) + " $\\pm$ " + N4(cf[f"{v}_test_{k}_dp"])
_nvit = lambda v, k: str(cf[f"vitorias_test_{k}_vs_{v}"]) + f"/{nS}"  # noqa: E731
_linhas1 = [["NFH (\\emph{" + cf["variante_nfh"].replace("_", "\\_") + "})", _lt("NFH", "RMSE"), _lt("NFH", "MAE"), "--", "--"],
            ["camada densa", _lt("densa", "RMSE"), _lt("densa", "MAE"), _nvit("densa", "RMSE"), _nvit("densa", "MAE")],
            ["densa+E", _lt("densa+E", "RMSE"), _lt("densa+E", "MAE"), _nvit("densa+E", "RMSE"), _nvit("densa+E", "MAE")],
            ["densa projetada em $\\bT\\oplus\\bS$ (controle)", N4(PJ["proj_test_RMSE"][0]) + " $\\pm$ " + N4(PJ["proj_test_RMSE"][1]),
             N4(PJ["proj_test_MAE"][0]) + " $\\pm$ " + N4(PJ["proj_test_MAE"][1]), str(PJ_nfh_vs_proj) + f"/{nS}", str(PJ_nfh_vs_proj_mae) + f"/{nS}"]]
add(tabela("Confirmação no teste: média $\\pm$ desvio-padrão amostral entre as " + str(nS) + " sementes (menor é melhor). As duas últimas colunas contam em quantas sementes a NFH tem erro estritamente menor que o da variante da linha (RMSE e MAE). A última linha é um controle relatado depois da confirmação e não entra no critério (Seção~\\ref{sec:resumo}).", "tab:prim",
           ["variante", "RMSE de teste", "MAE de teste", "NFH menor (RMSE)", "NFH menor (MAE)"], _linhas1, "lcccc", escala="\\textwidth",
           fonte="\\texttt{confirmacao.csv} e \\texttt{projecao\\_confirmacao.csv}", tam="\\footnotesize"))
add(tex(r"""O passo a passo usa a semente <<sem>> (a primeira da confirmação): nela, RMSE e MAE de teste são <<a>> e <<a2>> para a NFH, <<b>> e <<b2>> para a densa e <<c>> e <<c2>> para a densa+E. O retreino determinístico reproduziu esses valores (diferença máxima <<dif>>). Para que o texto se leia como uma aula, cada passo diz o que se calcula, por que, mostra a fórmula, substitui os números e mostra o resultado. Todo número vem de \texttt{passo\_a\_passo.json}, \texttt{confirmacao.csv} ou \texttt{selecao.csv}, gerados por código.""",
        sem=cfg["semente"], a=N(d["conferencia_teste"]["NFH"]["retreino_test_RMSE"]), a2=N(d["conferencia_teste"]["NFH"]["retreino_test_MAE"]),
        b=N(d["conferencia_teste"]["densa"]["retreino_test_RMSE"]), b2=N(d["conferencia_teste"]["densa"]["retreino_test_MAE"]),
        c=N(d["conferencia_teste"]["densa+E"]["retreino_test_RMSE"]), c2=N(d["conferencia_teste"]["densa+E"]["retreino_test_MAE"]),
        dif=N(max(abs(v["retreino_test_RMSE"] - v["csv_test_RMSE"]) for v in d["conferencia_teste"].values()), 2)))
add(r"\tableofcontents\newpage")

# ---- 1. o problema
g1, g2, g3, g4 = G["x1"], G["x2"], G["x3"], G["x4"]
add(r"\section{O problema e a série sintética}\label{sec:serie}")
add(tex(r"""Uma previsão multi-passo olha uma \emph{janela} do passado e escreve de uma vez os próximos $H=<<H>>$ valores. Aqui a janela é uma matriz $\bX\in\R^{<<L>>\times <<d>>}$: <<L>> instantes seguidos de <<d>> variáveis $x_1,\dots,x_<<d>>$. A saída é o vetor $\byh\in\R^{<<H>>}$. A ideia da NFH é fazer a camada final escrever esse vetor já dividido em três partes que se somam: uma \emph{tendência} $\bT$ (uma reta), uma \emph{sazonalidade} $\bS$ (<<K>> harmônicos) e um \emph{resíduo} $\bR$ (o que sobra, ortogonal às duas primeiras). Os coeficientes são $\thT=(\theta_{T,0},\theta_{T,1})$ (nível e inclinação da reta), $\thS$ (um cosseno e um seno por harmônico) e $\bu$ (os do resíduo). Por isso a explicação não é calculada depois: ela já está nos coeficientes que o modelo produz.""", H=H, L=L, d=DD, K=K))
add(tex(r"""Para saber se a explicação está certa, precisamos de uma série cuja resposta conhecemos. Gera-se, para cada instante $s=0,1,\dots,<<N>>-1$,
\begin{align*}
x_1(s) &= <<n1>> + <<a1>>\,\sen\!\left(\tfrac{2\pi s}{<<pe1>>}+<<f1>>\right) + <<a2>>\,\sen\!\left(\tfrac{2\pi s}{<<pe2>>}+<<f2>>\right) \\
x_2(s) &= A_s\left[\cos\!\left(\tfrac{2\pi s}{<<c1>>}+<<fc1>>\right) + <<w2>>\cos\!\left(\tfrac{2\pi s}{<<c2>>}<<fc2>>\right)\right],\\
A_s &= <<ab>>+<<am>>\,\sen\!\left(\tfrac{2\pi s}{<<pm>>}+<<fm>>\right),\\
x_3(s) &= <<a31>>\,\sen\!\left(\tfrac{2\pi s}{<<p31>>}+<<f31>>\right) + <<a32>>\,\sen\!\left(\tfrac{2\pi s}{<<p32>>}\right)\\
x_4(s) &= B_s\cos\!\left(\tfrac{2\pi s}{<<p4>>}+<<f4>>\right),\\
B_s &= <<bb>>+<<bm>>\,\sen\!\left(\tfrac{2\pi s}{<<pm4>>}\right)
\end{align*}
Aqui $x_1$ é lenta e conduz a tendência, $x_2$ conduz o ciclo, $x_3$ é um distrator lento e $x_4$, um distrator oscilante. O alvo é $y_s = x_1(s-<<at>>)+x_2(s-<<at>>)+e_s$, com ruído $e_s\sim\mathcal N(0,\sigma^2)$ independente e $\sigma=<<sig>>$. A série $y$ \emph{não} é entrada do modelo. Como o atraso ($<<at>>$) é igual ao horizonte, os próximos $H$ valores de $y$ são exatamente $x_1+x_2$ nas últimas $H$ linhas da janela, mais ruído. O melhor previsor possível, o \emph{oráculo}, é essa soma, e seu erro é o ruído. Como a soma de dois cossenos de períodos <<c1>> e <<c2>> cabe nos <<K>> harmônicos da NFH, e $x_1$ é quase uma reta em $H$ passos, o futuro mora quase todo no espaço $\bT\oplus\bS$ (o espaço gerado pelas colunas da tendência e da sazonalidade juntas). É de propósito: um caso favorável.""",
        N=G["N"], n1=fmt(g1["nivel"]), a1=fmt(g1["a1"]), pe1=g1["per1"], f1=fmt(g1["ph1"]), a2=fmt(g1["a2"]), pe2=g1["per2"], f2=fmt(g1["ph2"]),
        c1=g2["per_c1"], fc1=fmt(g2["ph_c1"]), w2=fmt(g2["peso_c2"]), c2=g2["per_c2"], fc2=("-" if g2["ph_c2"] < 0 else "+") + fmt(abs(g2["ph_c2"])),
        ab=fmt(g2["a_base"]), am=fmt(g2["a_mod"]), pm=g2["per_mod"], fm=fmt(g2["ph_mod"]), a31=fmt(g3["a1"]), p31=g3["per1"], f31=fmt(g3["ph1"]),
        a32=fmt(g3["a2"]), p32=g3["per2"], p4=g4["per"], f4=fmt(g4["ph"]), bb=fmt(g4["b_base"]), bm=fmt(g4["b_mod"]), pm4=g4["per_mod"],
        at=G["atraso"], sig=fmt(cfg["sigma"]), K=K))
npj = d["n_janelas"]
add(tex(r"""A série tem $<<N>>$ instantes, cortados em treino (primeiros <<c1>>), validação (até <<c2>>) e teste (o restante), o que dá <<ntr>> janelas de treino, <<nva>> de validação e <<nte>> de teste. O treino do exemplo usa as primeiras $<<n>>$ janelas de treino. O RMSE do oráculo nas janelas de teste é <<ro>> e o MAE, <<mo>>: é a \emph{referência amostral} que nenhum modelo consegue bater de forma consistente.""",
        N=G["N"], c1=pct(G["cortes"][0], 2), c2=pct(G["cortes"][1], 2), ntr=npj["treino"], nva=npj["val"], nte=npj["teste"], n=cfg["n_treino"],
        ro=N(d["oraculo_teste"]["RMSE"]), mo=N(d["oraculo_teste"]["MAE"])))

# ---- 2. escolha da combinação
add(r"\section{Como a combinação foi escolhida (antes de olhar o teste)}\label{sec:selecao}")
add(tex(r"""Antes de qualquer execução, fixou-se no protocolo (registro de 21/11/2026, 17h20, em \texttt{experimento\_backbones\_nfh/INSTRUCOES.md}) a grade, a regra e o critério de sucesso. A grade variou o número de neurônios $p\in\{8,16\}$ e o número de janelas de treino $n\in\{300,1000\}$, com as variantes \emph{densa} (cabeça linear comum), \emph{densa+E} (a mesma densa treinada por mais $E=<<E>>$ épocas, o controle de ``mais épocas''; atenção: o protocolo de treino do exemplo difere do da T1, ver Seção~\ref{sec:modelo}), \emph{conv\_L1000} (a densa convertida exatamente para a forma NFH e ajustada por $E$ épocas com penalização $\lambda=<<lam>>$ do resíduo, com seleção segura), \emph{zero\_L0} (NFH treinada do zero sem penalização) e \emph{zero\_L1000} (NFH treinada do zero com $\lambda=<<lam>>$). Cada célula foi repetida nas sementes 0 a 4. A regra olha \emph{só} a validação: para cada combinação $(p,n,\text{variante NFH})$ calcula-se a diferença relativa $(\text{NFH}-\text{densa+E})/\text{densa+E}$ do RMSE e do MAE de validação, média das 5 sementes. Qualifica-se se a média do RMSE relativo é negativa, a do MAE relativo é $\le 0$ e a NFH vence a densa+E na validação nas 5 sementes. Escolhe-se a de menor média do RMSE relativo; empate, o menor $p$.""", E=pex["E_EXTRA"], lam=fmt(cfg["lambda"], 4)))
linhas = [(r[0], r[1], r[2].replace("_", r"\_"), N(r[3]), N(r[4]), N(r[5]), pct(r[6]), pct(r[7]), f"{r[8]}/5", "sim" if r[9] else "não") for r in grade]
add(tex(r"""A Tabela~\ref{tab:selecao} mostra todas as <<nc>> combinações testadas, inclusive as que perderam. Qualificaram-se <<nq>>. A escolhida foi $p=<<pp>>$, $n=<<nn>>$, variante <<var>>, com RMSE relativo médio de validação <<rr>> e MAE relativo <<rm>>.""",
        nc=len(grade), nq=n_qual, pp=sl["p"], nn=sl["n"], var=r"\emph{" + sl["variante"].replace("_", r"\_") + "}", rr=N(sl["rel_RMSE"]), rm=N(sl["rel_MAE"])))
add(tabela("Seleção só pela validação (médias de 5 sementes). Diferenças relativas em relação à densa+E; negativo favorece a NFH.", "tab:selecao",
           ["$p$", "$n$", "NFH", "RMSE val. NFH", "RMSE val. densa+E", "RMSE val. densa", "rel. RMSE", "rel. MAE", "NFH melhor", "qualifica"],
           linhas, "rrlrrrrrcc", fonte="\\texttt{selecao.csv}, colunas \\texttt{val\\_*} (regra aplicada por \\texttt{selecionar.py})", tam="\\footnotesize", escala="\\textwidth"))
n300q = sum(r[-1] for r in grade if r[1] == 300)
add("O resultado desfavorável também aparece: " + str(n_qual) + " das " + str(len(grade)) + " combinações qualificaram-se, e em $n=300$ " + ("nenhuma" if n300q == 0 else str(n300q)) + " qualificou. A combinação escolhida é a que a regra aponta, não a que daria o melhor resultado de teste.")
add(tex(r"\paragraph{O que a grade diz sobre a penalização.} As variantes \emph{zero\_L0} e \emph{zero\_L1000} diferem só em $\lambda$. Com $p=<<pp>>$ e $n=<<nn>>$, o RMSE de validação médio (5 sementes) foi <<a0>> sem penalização e <<a1>> com $\lambda=<<lam>>$, e com $\lambda=<<lam>>$ o RMSE foi menor em <<w>> das 5 sementes. Em $n=<<n3>>$ e $p=<<pp>>$ a ordem se inverteu (<<b0>> sem penalização contra <<b1>> com ela). Portanto, com $p=<<pp>>$, a vantagem da penalização aparece com mais janelas de treino e não com $n=<<n3>>$, e as demais células da grade (Tabela~\ref{tab:selecao}) não são todas favoráveis; estes números são de validação e não decidem nada sobre o teste.", pp=PPL, nn=NNL, a0=N(CMPL["a0"]), a1=N(CMPL["a1"]), lam=fmt(cfg["lambda"], 4), w=CMPL["w"], n3=N3L, b0=N(CMPL["b0"]), b1=N(CMPL["b1"])))

# ---- 3. bases
add(r"\section{As bases: o alfabeto da decomposição}\label{sec:bases}")
add(tex(r"""A NFH escreve a previsão como combinação de colunas fixas. Elas são calculadas uma vez, a partir só de $H$, $q$ e $K$, e não têm parâmetro treinável. Usamos o tempo normalizado $\tau_j=j/H$, $j=0,\dots,<<Hm1>>$: $\tau=<<tau>>$.""", Hm1=H - 1, tau=vec(tau, 3)))
add(r"\subsection{A base de tendência $\bP$}")
add(tex(r"""A tendência é um polinômio de grau $q=<<q>>$, ou seja, uma reta. A matriz $\bP\in\R^{<<q1>>\times <<H>>}$ tem como linhas $\tau^0$ e $\tau^1$:
\[\bP=<<P>>.\]
Por que isso? Porque uma reta é o jeito mais simples de dizer ``o nível e a inclinação do futuro''. A primeira linha, toda de uns, faz com que o coeficiente $\theta_{T,0}$ (o primeiro de $\thT$, multiplicado por ela) seja o nível, e $\theta_{T,1}$, multiplicado por $\tau$, a inclinação.""", q=q, q1=q + 1, H=H, P=mat(Pm)))
add(r"\subsection{A base de Fourier $\bF$}")
add(tex(r"""Para os ciclos usamos, para cada harmônico $k=1,\dots,<<K>>$, um cosseno e um seno de período $H/k$: $F_{2k-2,j}=\cos(2\pi k\tau_j)$ e $F_{2k-1,j}=\sen(2\pi k\tau_j)$. Com $K=<<K>>$ há $2K=<<K2>>$ linhas (períodos <<per1>> e <<per2>> passos). Exige-se $2K<H$ (critério de Nyquist), aqui $<<K2>>\,<\,<<H>>$:
\[\bF=<<F>>.\]""", K=K, K2=2 * K, per1=fmt(H / 1), per2=fmt(H / 2), H=H, F=mat(Fm)))
add(r"\subsection{Tirar a reta dos ciclos: a parte polinomial e $\bFt$}")
PPt, PPti, alf = A(B["PPt"]), A(B["PPt_inv"]), A(B["parte_polinomial_de_F"])
add(tex(r"""Um cosseno de período $H$ medido em apenas $H$ pontos não é ortogonal à reta: parte dele \emph{parece} tendência. Se deixássemos assim, $\bT$ e $\bS$ disputariam a mesma energia. A solução é projetar cada linha de $\bF$ sobre as linhas de $\bP$ e subtrair. Primeiro a matriz de Gram $\bP\bP^\top$ e sua inversa:
\[\bP\bP^\top=<<G>>,\qquad (\bP\bP^\top)^{-1}=<<Gi>>.\]
A parte polinomial da linha $i$ de $\bF$ é $\boldsymbol\alpha_i=(\bP\bP^\top)^{-1}\bP\,\bF_i^\top$, os coeficientes da melhor reta que aproxima essa linha. Colocando as <<K2>> linhas lado a lado, $\boldsymbol\alpha=\left[\boldsymbol\alpha_1\cdots\boldsymbol\alpha_{<<K2>>}\right]$ (uma coluna por linha de $\bF$):
\[\boldsymbol\alpha=<<alf>>.\]
A base sazonal é $\bFt=\bF-\boldsymbol\alpha^\top\bP$, isto é, cada ciclo sem a sua reta:
\[\bFt=<<Ft>>.\]
Conferência numérica: o maior valor absoluto de $\bFt\bP^\top$ é <<err>> (zero, a menos do arredondamento), logo $\bFt\perp\bP$. A reta que saiu dos ciclos não se perde: volta para $\bT$. Veremos isso na Seção~\ref{sec:verdade}.""",
        G=mat(PPt), Gi=mat(PPti), K2=2 * K, alf=mat(alf.T), Ft=mat(Ftm), err=N(B["erro_Ft_ortogonal_a_P"], 2)))
add(r"\subsection{A base residual $\bQ$ por Gram--Schmidt}")
gs = B["gram_schmidt"]
Bm_ = A(B["B"])
add(tex(r"""Tendência e ciclos ocupam $m=q+1+2K=<<m>>$ dimensões das $H=<<H>>$ disponíveis. As $H-m=<<Hm>>$ restantes formam o espaço do \emph{resíduo}: tudo o que a previsão pode conter além de $\bT$ e $\bS$. A base $\bQ\in\R^{<<H>>\times<<Hm>>}$ é uma base ortonormal desse complemento. \textbf{Aviso:} a Tabela~\ref{tab:gs} ilustra o método de Gram--Schmidt, mas as contas das seções seguintes usam \emph{outra base do mesmo espaço} (QR de Householder, explicada depois da tabela). Quem refizer à mão as contas com os $q$ da tabela obterá coeficientes $\bu$ diferentes dos mostrados; o resíduo $\bR$ e o projetor $\bQ\bQ^\top$ são os mesmos. Calcula-se por Gram--Schmidt: junta-se as colunas $\mathbf B=[\bP^\top\ \bF^\top]\in\R^{<<H>>\times<<m>>}$ (as seis direções já ocupadas), e para cada vetor $v$ da lista (primeiro as seis colunas de $\mathbf B$, depois $e_1,e_2,\dots$) subtrai-se sua projeção sobre os $q_i$ já obtidos, $w=v-\sum_i(q_i^\top v)q_i$, e normaliza-se, $q=w/\|w\|$. Se $\|w\|\approx0$ o vetor já estava no espaço e é descartado. A Tabela~\ref{tab:gs} mostra os coeficientes $q_i^\top v$ e a norma de $w$ em cada passo. Aqui <<nac>> dos <<nv>> vetores foram aceitos<<desc>>; $q_7$ vem de <<v7>> e $q_8$ de <<v8>>.""",
        m=m, H=H, Hm=Hm, nac=sum(g_["aceito"] for g_ in gs), nv=len(gs), desc=(" (nenhum foi descartado)" if all(g_["aceito"] for g_ in gs) else ""),
        v7="$" + gs[6]["vetor"] + "$", v8="$" + gs[7]["vetor"] + "$"))
cab = ["passo", "$v$"] + [f"$q_{i + 1}^\\top v$" for i in range(H - 1)] + ["$\\|w\\|$"]
linhas = []
for k_, ps in enumerate(gs):
    pr = [fmt(t, 3) for t in ps["projecoes"]] + ["--"] * (H - 1 - len(ps["projecoes"]))
    linhas.append([k_ + 1, ps["vetor"].replace("coluna ", "col. ").replace("e_", "$e_").replace("de B", "de $\\mathbf B$") + ("$" if ps["vetor"].startswith("e_") else "")] + [f"${t}$" if t != "--" else t for t in pr] + [N(ps["norma"], 3)])
add(tabela("Gram--Schmidt passo a passo (3 algarismos significativos). Os primeiros seis vetores são as colunas de $\\mathbf B$; os dois últimos completam a base com $e_1$ e $e_2$.",
           "tab:gs", cab, linhas, "rl" + "r" * (H - 1) + "r", tam="\\scriptsize", escala="\\textwidth"))
qgs = A(B["Q_GS"]); Og = A(B["O_GS_para_Q"])
add(tex(r"""Os dois últimos vetores aceitos, $q_7$ e $q_8$, formam a base residual por Gram--Schmidt:
\[\bQ_{\mathrm{GS}}=<<Q>>.\]
O código usa a base $\bQ$ obtida por decomposição QR de Householder (função \texttt{nfh\_bases}), que gera outra base do mesmo espaço: $\bQ=\bQ_{\mathrm{GS}}\mathbf O$ com a rotação $2\times2$ $\mathbf O=<<O>>$. Por isso os coeficientes $\bu$ dependem da base, mas o resíduo $\bR=\bQ\bu$ e o projetor $\bQ\bQ^\top$ não: o maior desvio entre $\bQ_{\mathrm{GS}}\bQ_{\mathrm{GS}}^\top$ e $\bQ\bQ^\top$ é <<ep>>. A base usada nas contas seguintes é
\[\bQ=<<Qm>>,\]
com $\max|\bQ^\top\bQ-\bI|=<<eo>>$ e $\max|\bQ^\top\mathbf B|=<<eb>>$ (ambos zero a menos do arredondamento).""",
        Q=mat(qgs), O=mat(Og), ep=N(B["erro_projetor_GS_vs_Q"], 2), Qm=mat(Qm), eo=N(B["erro_ortogonalidade_Q"], 2), eb=N(B["erro_Q_ortogonal_a_B"], 2)))
add(r"\subsection{A matriz de síntese $\bM$ e a normalização $\bD$}")
add(tex(r"""Juntando as três bases lado a lado, a matriz de síntese é $\bM=[\bP^\top\ \ \bFt^\top\ \ \bQ]\in\R^{<<H>>\times<<H>>}$: dados os coeficientes $\bc=(\thT,\thS,\bu)$, a previsão é $\byh=\bM\bc$. A Tabela~\ref{tab:M} mostra $\bM$; seu número de condição é $\mathrm{cond}(\bM)=<<cond>>$, pequeno, o que torna a conversão com a camada densa estável (Seção~\ref{sec:conv}). Para que a rede aprenda com passos parecidos em todas as colunas, o modelo trabalha com coeficientes $\bc'$ de escala neutra e define $\bc=\bD\bc'$, onde $\bD$ é diagonal com $D_{ii}=1/\|\bM_{:,i}\|$ para as colunas de tendência e sazonalidade (a norma da linha correspondente de $\bP$ ou $\bFt$) e $D_{ii}=1$ para o resíduo. As normas são $<<nor>>$, e portanto $\bD=\mathrm{diag}<<D>>$.""",
        H=H, cond=N(B["cond_M"]), nor=vec(Nv, 4), D=vec(Dv, 4)))
add(tabela("Matriz de síntese $\\bM=[\\bP^\\top\\ \\bFt^\\top\\ \\bQ]$ (as 2 primeiras colunas são a tendência, as 4 seguintes a sazonalidade, as 2 últimas o resíduo).", "tab:M",
           ["$j$"] + [f"col. {i + 1}" for i in range(H)], [[j] + [N(t, 4) for t in Mm[j]] for j in range(H)], "r" + "r" * H, tam="\\footnotesize", escala="\\textwidth"))

# ---- 4. modelo
add(r"\section{O modelo treinado}\label{sec:modelo}")
add(tex(r"""O modelo tem duas partes. O \emph{backbone} lê a janela achatada $\mathrm{vec}(\bX)\in\R^{<<LD>>}$ (as linhas de $\bX$ uma após outra) e produz $\bh=\tanh(\bW_\Phi\,\mathrm{vec}(\bX)+\bb_\Phi)\in\R^{<<p>>}$, com $\bW_\Phi\in\R^{<<p>>\times<<LD>>}$ (<<nb>> parâmetros contando o viés). A \emph{cabeça} NFH calcula $\bc'=\bW\bh+\bb$ com $\bW\in\R^{<<H>>\times<<p>>}$ e $\bb\in\R^{<<H>>}$ (<<nc>> parâmetros, o mesmo número da cabeça densa), depois $\bc=\bD\bc'$ e $\byh=\bM\bc=\bT+\bS+\bR$. A cabeça densa equivalente teria a mesma forma $\byh=\bW'\bh+\bb'$, sem a estrutura.""",
        LD=L * DD, p=p, nb=cfg["n_parametros_backbone"], H=H, nc=cfg["n_parametros_cabeca"]))
add(tex(r"""Treino (mesmo protocolo para densa e NFH, com a mesma semente <<sem>>): otimizador Adam com taxa $<<lr>>$ e decaimento de pesos $10^{-5}$, lotes de <<lote>>, até <<ep>> épocas com paciência <<pac>> sobre o RMSE de validação, ficando com a época de menor RMSE de validação. A NFH do zero minimiza $\overline{(\byh-\mathbf y)^2}+\lambda\,\overline{\bR^2}$ com $\lambda=<<lam>>$: o segundo termo \emph{pune o resíduo}. A época escolhida foi <<e1>> para a NFH e <<e2>> para a densa. A densa+E parte da densa e treina mais $E=<<E>>$ épocas com taxa $<<lra>>$ (época escolhida <<e3>> de <<E>>). \textbf{Aviso:} esse protocolo (taxa $<<lr>>$, lotes de <<lote>>, até <<ep>> épocas, paciência <<pac>>, e $E=<<E>>$ com taxa $<<lra>>$ no ajuste) é o do exemplo de livro do teste de 16/11, mantido porque a série e o modelo são pequenos, e \emph{não é} o da T1 (taxa $<<t1lr>>$, lotes de <<t1lote>>, até <<t1ep>> épocas, paciência <<t1pac>>; densa+E com $E=<<t1E>>$ épocas e taxa $<<t1lra>>$). Quem comparar este exemplo com a dissertação ou com o artigo deve lembrar que a densa+E daqui tem $E=<<E>>$, e não $<<t1E>>$. O teste nunca escolhe nada. As $n$ janelas de treino são as \emph{primeiras} $n$ do segmento de treino, como no exemplo de livro (\texttt{exemplo\_livro/treinar.py}); o experimento principal sobre backbones (T1) usa as \emph{últimas} (\texttt{INSTRUCOES.md}), mais próximas da validação, então este exemplo não replica esse detalhe da T1.""",
        sem=cfg["semente"], lr=fmt(cfg["lr"]), lote=cfg["lote"], ep=cfg["epocas_max"], pac=cfg["paciencia"], lam=fmt(cfg["lambda"]),
        e1=cfg["epocas_escolhidas"]["zero_L1000"], e2=cfg["epocas_escolhidas"]["densa"], e3=cfg["epocas_escolhidas"]["densa+E"], lra=fmt(cfg["lr_ajuste"]), E=pex["E_EXTRA"], t1lr=T1P["lr"].replace("e-3", "\\times10^{-3}"), t1lote=T1P["lote"], t1ep=T1P["ep"],
        t1pac=T1P["pac"], t1E=T1P["E"], t1lra=T1P["lraj"].replace("e-4", "\\times10^{-4}")))

# ---- 5. janela
add(r"\section{A janela que vamos explicar}\label{sec:janela}")
add(tex(r"""Escolhemos, antes de olhar qualquer resultado, a janela central do conjunto de teste: o índice $\lfloor<<nt>>/2\rfloor=<<j>>$ entre as <<nt>> janelas de teste. Ela começa no instante $s=<<i0>>$ da série. A Tabela~\ref{tab:X} traz $\bX$ e a Figura~\ref{fig:serie} mostra a janela (faixa cinza-clara) e o futuro. Os últimos <<H>> valores de $x_1$ e $x_2$ (linhas <<l0>> a <<L>>) são o que o oráculo usa.""",
        nt=cfg["n_janelas_teste"], j=cfg["janela_teste"], i0=cfg["inicio_janela_na_serie"], H=H, l0=L - H + 1, L=L))
add(tabela(f"A janela $\\bX$ ({L} linhas $\\times$ {DD} variáveis).", "tab:X", ["$\\ell$"] + [f"${v}$" for v in vn],
           [[l + 1] + [N(t) for t in X[l]] for l in range(L)], "r" + "r" * DD))
add(figura("fig_serie_janela.pdf", "Série em torno da janela explicada. Painel superior: as quatro variáveis de entrada (preto contínuo $x_1$; cinza escuro tracejado $x_2$; cinza médio contínuo $x_3$; cinza claro traço-ponto $x_4$); a faixa cinza clara marca as " + str(L) + " linhas da janela. Painel inferior: alvo com ruído (pontos cinza médio), alvo sem ruído (linha preta) e previsão da NFH nas " + str(H) + " horas seguintes (quadrados vazados).", "fig:serie"))
add(tex(r"""O futuro real (com ruído) e o oráculo (sem ruído), em cada hora $j=0,\dots,<<Hm1>>$, estão na Tabela~\ref{tab:fut}.""", Hm1=H - 1))
add(tabela("Futuro real e oráculo.", "tab:fut", ["$j$"] + [str(j) for j in range(H)],
           [["$y_j$ (real)"] + [N(t) for t in yre], ["oráculo $x_1+x_2$"] + [N(t) for t in yor]], "l" + "r" * H, tam="\\footnotesize", escala="\\textwidth"))

# ---- 6. h
add(r"\section{Passo 1: o backbone calcula $\bh$}\label{sec:h}")
nr = C["neuronio"]
add(tex(r"""Cada um dos $<<p>>$ neurônios de $\bh$ faz o mesmo: multiplica cada um dos $<<LD>>$ números de $\mathrm{vec}(\bX)$ por um peso, soma tudo, soma um viés e aplica $\tanh$, que comprime o resultado para $(-1,1)$. Mostramos as contas completas do neurônio $<<i>>$ (o primeiro; fixado antes de olhar): $h_<<i>>=\tanh\big(\sum_{k=1}^{<<LD>>}W_{\Phi,<<i>>k}\,x_k+b_{\Phi,<<i>>}\big)$. A Tabela~\ref{tab:neuronio} lista, para cada posição $\ell$ e variável, o peso $w$ e o produto $w\cdot x$ (o $x$ está na Tabela~\ref{tab:X}).""", p=p, LD=L * DD, i=nr["indice"]))
pr = nr["produtos"]
lin = []
for l in range(L):
    r_ = [l + 1]
    for v in range(DD):
        e_ = pr[l * DD + v]
        r_ += [N(e_["w"], 3), N(e_["w_x"], 3)]
    lin.append(r_)
cab = ["$\\ell$"] + [t for v in vn for t in (f"$w$ ($\\!{v}$)", f"$w\\,x$ ($\\!{v}$)")]
add(tabela(f"Os {L * DD} produtos $w\\cdot x$ do neurônio {nr['indice']} (3 algarismos significativos).", "tab:neuronio", cab, lin, "r" + "rr" * DD, tam="\\footnotesize"))
sums = [sum(pr[l * DD + v]["w_x"] for l in range(L)) for v in range(DD)]
add(tex(r"""Somando os <<LD>> produtos: <<s1>> (para $x_1$), <<s2>> ($x_2$), <<s3>> ($x_3$) e <<s4>> ($x_4$), total $\sum w\,x=<<tot>>$. Somando o viés $b_{\Phi,<<i>>}=<<vies>>$ obtém-se a pré-ativação $z=<<z>>$, e então
\[h_<<i>>=\tanh(<<z>>)=<<h>>.\]
Os outros <<pm1>> neurônios seguem a mesma conta com seus próprios pesos. O vetor completo é, na Tabela~\ref{tab:h}, $\bh\in(-1,1)^{<<p>>}$ (diferença máxima de <<eh>> contra o modelo em ponto flutuante de 32 bits).""",
        LD=L * DD, s1=N(sums[0]), s2=N(sums[1]), s3=N(sums[2]), s4=N(sums[3]), tot=N(nr["soma_produtos"]), i=nr["indice"], vies=N(nr["vies"]), z=N(nr["z"]),
        h=N(nr["h"]), pm1=p - 1, p=p, eh=N(C["erro_h_vs_modelo"], 2)))
add(tabela("A representação latente $\\bh$.", "tab:h", ["neurônio"] + [str(i + 1) for i in range(p)], [["$h$"] + [N(t, 3) for t in h]], "l" + "r" * p, tam="\\scriptsize", escala="\\textwidth"))

# ---- 7. c' = W h + b
add(r"\section{Passo 2: a cabeça calcula $\bc'=\bW\bh+\bb$}\label{sec:cprime}")
add(tex(r"""\textbf{Aviso de índices:} daqui em diante, as linhas de $\bW$, as coordenadas de $\bc'$, $\bc$ e $\bD$ e as horas $j$ do horizonte começam em 0, enquanto os neurônios $k$ e as posições $\ell$ da janela começam em 1. Para conferir à mão, basta refazer a linha $i=0$; as outras seguem a mesma conta. A cabeça é uma camada linear: cada uma das $<<H>>$ linhas de $\bW$ faz o produto escalar com $\bh$ e soma o viés. As Tabelas~\ref{tab:W0} e~\ref{tab:W1} mostram $\bW$ e $\bb$; as Tabelas~\ref{tab:prod0} e~\ref{tab:prod1} mostram os produtos $W_{ik}h_k$ de todas as linhas, a soma de cada uma, o viés $b_i$ e o resultado $c'_i$.""", H=H))
for pt, (k0, k1) in enumerate([(0, p // 2), (p // 2, p)]):
    extra = ["$b_i$"] if pt == 1 else []
    add(tabela(f"A matriz $\\bW\\in\\R^{{{H}\\times{p}}}$, colunas {k0 + 1} a {k1}" + (" e viés $\\bb$" if pt == 1 else "") + " (3 algarismos significativos).", f"tab:W{pt}",
               ["$i$"] + [str(k + 1) for k in range(k0, k1)] + extra,
               [[i] + [N(t, 3) for t in W[i][k0:k1]] + ([N(bb[i], 3)] if pt == 1 else []) for i in range(H)], "r" + "r" * (k1 - k0) + ("r" if pt == 1 else ""), tam="\\footnotesize"))
lc = C["linhas_c"]
for pt, (k0, k1) in enumerate([(0, p // 2), (p // 2, p)]):
    extra = ["$\\sum_k$", "$b_i$", "$c'_i$"] if pt == 1 else []
    add(tabela(f"Produtos $W_{{ik}}h_k$ para $k={k0 + 1},\\dots,{k1}$" + (", soma sobre todos os $k$, viés e $c'_i$" if pt == 1 else "") + " (3 algarismos significativos). A linha $i=0$ define o nível da tendência e a linha $i=1$, sua inclinação.", f"tab:prod{pt}",
               ["$i$"] + [str(k + 1) for k in range(k0, k1)] + extra,
               [[i] + [N(t, 3) for t in lc[i]["produtos"][k0:k1]] + ([N(lc[i]["soma"]), N(lc[i]["vies"]), N(lc[i]["c_linha"])] if pt == 1 else []) for i in range(H)],
               "r" + "r" * (k1 - k0) + ("rrr" if pt == 1 else ""), tam="\\footnotesize", escala=("\\textwidth" if pt == 1 else None)))
def conta_linha(i):
    pr_ = lc[i]["produtos"]
    termos = " + ".join(f"({fmt(W[i][k], 3)})({fmt(h[k], 3)})" for k in (0, 1)) + " + \\cdots + " + f"({fmt(W[i][p - 1], 3)})({fmt(h[p - 1], 3)})"
    return (f"c'_{{{i}}} &= \\sum_{{k=1}}^{{{p}}}W_{{{i},k}}h_k+b_{i} = {termos} + {fmt(b[i] if False else bb[i])}\\\\\n"
            f"&= {fmt(lc[i]['soma'])} + ({fmt(lc[i]['vies'])}) = {fmt(lc[i]['c_linha'])}")
add(tex(r"""Por escrito, para as duas primeiras linhas (as reticências escondem termos que estão nas Tabelas~\ref{tab:prod0} e~\ref{tab:prod1}):
\begin{align*}
<<l0>>\\[4pt]
<<l1>>
\end{align*}
O vetor completo é $\bc'=<<cp>>$.""", l0=conta_linha(0), l1=conta_linha(1), cp=vec(cpr)))
add(tex(r"""\subsection*{Normalização e divisão}
Multiplicando termo a termo por $\bD=\mathrm{diag}<<D>>$ obtém-se $\bc=\bD\bc'=<<c>>$. Os $q+1=<<q1>>$ primeiros números são $\thT$, os $2K=<<K2>>$ seguintes são $\thS=(a_1,b_1,a_2,b_2)$ e os $H-m=<<Hm>>$ últimos são $\bu$:
\begin{align*}
\thT&=<<tT>> &&\text{(nível $\theta_{T,0}$ e inclinação $\theta_{T,1}$)}\\
\thS&=<<tS>> &&\text{(cosseno e seno dos harmônicos 1 e 2)}\\
\bu&=<<u>> &&\text{(coeficientes do resíduo)}
\end{align*}""", D=vec(Dv), c=vec(c), q1=q + 1, K2=2 * K, Hm=Hm, tT=vec(thT), tS=vec(thS), u=vec(uu)))

# ---- 8. T S R
add(r"\section{Passo 3: sintetizar $\bT$, $\bS$, $\bR$ e conferir a soma}\label{sec:TSR}")
jx = H // 2
add(tex(r"""Agora cada componente sai de um produto simples: $\bT=\bP^\top\thT$, $\bS=\bFt^\top\thS$, $\bR=\bQ\bu$ e $\byh=\bT+\bS+\bR$. Exemplo na hora $j=<<j>>$ (fixada em $\lfloor H/2\rfloor$), com $\tau=<<tau>>$:
\begin{align*}
T_{<<j>>}&=\theta_{T,0}\cdot 1+\theta_{T,1}\,\tau=(<<t0>>)(1)+(<<t1>>)(<<tau>>)=<<T>>,\\
S_{<<j>>}&=\sum_{i=1}^{<<K2>>}\widetilde F_{i,<<j>>}\,\theta_{S,i}\\
&=<<sS>>=<<S>>,\\
R_{<<j>>}&=\sum_{i=1}^{<<Hm>>}Q_{<<j>>,i}\,u_i=(<<q0>>)(<<u0>>)+(<<q1>>)(<<u1>>)=<<R>>,\\
\hat y_{<<j>>}&=T_{<<j>>}+S_{<<j>>}+R_{<<j>>}=<<T>>+(<<S>>)+(<<R>>)=<<yh>>.
\end{align*}
A Tabela~\ref{tab:TSR} repete a conta para as $<<H>>$ horas. O maior erro de $T+S+R$ contra $\byh$ calculado por $\bM\bc$ é <<es>>, a soma é exata a menos do arredondamento.""",
        j=jx, tau=fmt(tau[jx], 3), t0=fmt(thT[0]), t1=fmt(thT[1]), T=fmt(Tn[jx]), K2=2 * K, Hm=Hm,
        sS=" + ".join(f"({fmt(Ftm[i][jx], 3)})({fmt(thS[i], 3)})" for i in range(2 * K)), S=fmt(Sn[jx]),
        q0=fmt(Qm[jx][0], 3), u0=fmt(uu[0], 3), q1=fmt(Qm[jx][1], 3), u1=fmt(uu[1], 3), R=fmt(Rn[jx]), yh=fmt(yh[jx]), H=H, es=N(NF["erro_soma"], 2)))
add(tabela("Decomposição da previsão da NFH hora a hora.", "tab:TSR", ["$j$", "$T_j$", "$S_j$", "$R_j$", "$T_j+S_j+R_j$", "$\\hat y_j$", "oráculo", "real"],
           [[j, N(Tn[j]), N(Sn[j]), N(Rn[j]), N(Tn[j] + Sn[j] + Rn[j]), N(yh[j]), N(yor[j]), N(yre[j])] for j in range(H)], "rrrrrrrr"))
add(tex(r"""\textbf{A média de $\bT$ é a média da previsão.} Como $\bFt\perp\bP$ e $\bQ\perp\bP$ e a primeira linha de $\bP$ é toda de uns, $\bS$ e $\bR$ têm média zero. Numericamente: média de $\bT$ = <<mt>> e média de $\byh$ = <<my>>. Por isso o nível médio previsto fica todo em $\bT$ e o ciclo não tem como ``roubar'' nível.""", mt=N(NF["media_T"]), my=N(NF["media_previsao"])))

# ---- 9. energia e harmônicos
add(r"\section{Passo 4: balanço de energia e harmônicos}\label{sec:energia}")
add(tex(r"""Como $\bT$ (sem a média), $\bS$ e $\bR$ são ortogonais, a energia da previsão centrada, $\En=\|\byh-\bar y\|^2$ (não confundir com as $E$ épocas extras da Seção~\ref{sec:selecao}), se reparte sem sobras: $\varepsilon_T=\|\bT-\bar y\|^2/\En$, $\varepsilon_S=\|\bS\|^2/\En$, $\varepsilon_R=\|\bR\|^2/\En$, com $\varepsilon_T+\varepsilon_S+\varepsilon_R=1$. Aqui $\bar y=<<yb>>$ (a média dos $H$ valores previstos) e $\En=<<E>>$. Resulta $\varepsilon_T=<<eT>>$, $\varepsilon_S=<<eS>>$ e $\varepsilon_R=<<eR>>$ (soma <<soma>>). Ou seja, na previsão centrada, a quase totalidade da variação é sazonal, e o resíduo é desprezível: a previsão já \emph{nasce} dentro do espaço estrutural. Esta é uma janela só; que isso se deva à penalização $\lambda$ não sai daqui, e a comparação que a sustenta, feita na validação com 5 sementes, está na Seção~\ref{sec:selecao} (<<cmp0>>).""",
        yb=N(en["media"]), E=N(en["norma2_centrada"]), eT=pct(en["eps_T"]), eS=pct(en["eps_S"]), eR=pct(en["eps_R"]), soma=N(en["eps_T"] + en["eps_S"] + en["eps_R"], 8), cmp0=CMP_CURTO))
hm = NF["harmonicos"]; hv = V["harmonicos"]
add(tex(r"""Cada harmônico $k$ tem coeficientes $(a_k,b_k)$ do cosseno e do seno; $a_k\cos(2\pi k\tau)+b_k\sen(2\pi k\tau)=A_k\cos(2\pi k\tau-\varphi_k)$ com amplitude $A_k=\sqrt{a_k^2+b_k^2}$ e fase $\varphi_k=\operatorname{atan2}(b_k,a_k)$. A participação $e_k=\langle \bS_k,\bS\rangle/\En$ ($\bS_k$ é a parte de $\bS$ do harmônico $k$) é a fatia da energia que o harmônico $k$ carrega. Na Tabela~\ref{tab:harm}, as colunas da NFH e da verdade (Seção~\ref{sec:verdade}) permitem comparar.""", ))
add(tabela("Harmônicos da NFH e da verdade (oráculo) nesta janela.", "tab:harm",
           ["$k$", "período", "$a_k$", "$b_k$", "$A_k$", "$\\varphi_k$ (rad)", "$e_k$", "$A_k$ verdade", "$\\varphi_k$ verdade", "$e_k$ verdade"],
           [[hm[k]["k"], N(hm[k]["periodo_passos"], 3), N(hm[k]["a"]), N(hm[k]["b"]), N(hm[k]["A"]), N(hm[k]["fase_rad"]), pct(hm[k]["e_k"]),
             N(hv[k]["A"]), N(hv[k]["fase_rad"]), pct(hv[k]["e_k"])] for k in range(K)], "rrrrrrrrrr", tam="\\footnotesize"))

# ---- 10. verdade
add(r"\section{A verdade: a mesma decomposição aplicada ao oráculo}\label{sec:verdade}")
Ftv = Ftm @ Ftm.T
add(tex(r"""Podemos decompor qualquer vetor de $\R^H$ pelas mesmas bases, sem rede nenhuma: $\thT=(\bP\bP^\top)^{-1}\bP\mathbf y$, $\thS=(\bFt\bFt^\top)^{-1}\bFt\mathbf y$, $\bu=\bQ^\top\mathbf y$ (as ``fórmulas de identificabilidade'': dada a previsão, os coeficientes são únicos). Aplicadas ao oráculo $\mathbf y^\star=x_1+x_2$ (Tabela~\ref{tab:fut}) dão $\thT^\star=<<tT>>$, $\thS^\star=<<tS>>$ e $\bu^\star=<<u>>$; o asterisco marca, daqui em diante, a mesma grandeza calculada sobre o oráculo (a ``verdade'') em vez da previsão da NFH. A energia da verdade se reparte em $\varepsilon_T^\star=<<eT>>$, $\varepsilon_S^\star=<<eS>>$ e $\varepsilon_R^\star=<<eR>>$, definidos como $\varepsilon_T,\varepsilon_S,\varepsilon_R$ do Passo 4, com $\mathbf y^\star$ no lugar de $\byh$.""",
        tT=vec(tT0), tS=vec(tS0), u=vec(u0), eT=pct(ev["eps_T"]), eS=pct(ev["eps_S"]), eR=pct(ev["eps_R"])))
bg, ax1 = A(V["beta_gerador_com_A_medio"]), A(V["reta_de_x1_no_horizonte"])
tTg = A(V["theta_T_previsto_pelo_gerador"])
add(tex(r"""\paragraph{De onde deveriam vir esses números.} No horizonte, $x_2(s_0+j)=A_s[\cos(2\pi j/8+\psi_1)+0{,}3\cos(2\pi j/4+\psi_2)]$ com $\psi_1=2\pi s_0/8+<<f1>>\equiv<<psi1>>$ e $\psi_2=2\pi s_0/4<<f2>>\equiv<<psi2>>$ (módulo $2\pi$; aqui $s_0=<<s0>>$). Com a amplitude trocada pela média no horizonte, $\bar A=<<Am>>$, e $\cos(\omega j+\psi)=\cos\psi\cos\omega j-\sen\psi\sen\omega j$, os coeficientes esperados do seno e do cosseno são $\beta=\bar A(\cos\psi_1,\,-\sen\psi_1,\,0{,}3\cos\psi_2,\,-0{,}3\sen\psi_2)=<<beta>>$, que se comparam a $\thS^\star=<<tS>>$ (diferem porque $A_s$ varia no horizonte). Já a reta de $x_1$ no horizonte é $\boldsymbol\alpha_{x_1}=<<ax>>$ (nível e inclinação em $\tau$). Os ciclos trazem uma reta embutida, $\boldsymbol\alpha\boldsymbol\beta=<<ab>>$, que o desenho de $\bFt$ tirou da sazonalidade e devolveu à tendência: $\thT^\star\approx\boldsymbol\alpha_{x_1}+\boldsymbol\alpha\boldsymbol\beta=<<tg>>$, contra $\thT^\star=<<tT>>$ calculado diretamente.""",
        f1=fmt(G["x2"]["ph_c1"]), psi1=fmt(V["psi1"] % (2 * math.pi)), f2=("-" if G["x2"]["ph_c2"] < 0 else "+") + fmt(abs(G["x2"]["ph_c2"])), psi2=fmt(V["psi2"] % (2 * math.pi)), s0=cfg["s0"],
        Am=N(V["A_medio"]), beta=vec(bg), tS=vec(tS0), ax=vec(ax1), ab=vec(A(V["parte_polinomial_dos_ciclos"])), tg=vec(tTg), tT=vec(tT0)))
add(tex(r"""\paragraph{Erro da NFH contra a verdade.} Na Figura~\ref{fig:TSR} aparecem os componentes; a diferença dos coeficientes é $\thT-\thT^\star=<<dT>>$ e $\thS-\thS^\star=<<dS>>$. Em unidade de previsão, o erro máximo é <<mx>>, o RMSE da NFH contra o oráculo nesta janela é <<rn>> e contra o futuro real, <<rr>>. Por componente, o maior desvio absoluto é <<xT>> em $\bT$, <<xS>> em $\bS$ e <<xR>> em $\bR$.""",
        dT=vec(A(EV["theta_T"])), dS=vec(A(EV["theta_S"])), mx=N(EV["max_abs_previsao"]), rn=N(EV["rmse_janela_nfh_vs_oraculo"]), rr=N(EV["rmse_janela_nfh_vs_real"]),
        xT=N(EV["max_abs_T"]), xS=N(EV["max_abs_S"]), xR=N(EV["max_abs_R"])))
add(figura("fig_TSR.pdf", "Componentes da NFH (quadrados pretos, linha contínua) e da verdade (círculos cinza médio, linha tracejada) em cada hora do horizonte: tendência $T$, sazonalidade $S$ e resíduo $R$.", "fig:TSR"))

# ---- 11. densa
add(r"\section{A camada densa da mesma semente: só dá para decompor depois}\label{sec:densa}")
add(tex(r"""A densa da semente <<sem>> usa o mesmo backbone treinado do mesmo jeito, mas sua cabeça escreve $\hat{\mathbf y}$ direto. Para explicá-la é preciso decompor a previsão depois, pelas mesmas fórmulas de identificabilidade. A previsão da densa nesta janela é $<<yd>>$. Decomposta: $\thT=<<tT>>$, $\thS=<<tS>>$, $\bu=<<u>>$. As participações de energia são $\varepsilon_T=<<eT>>$, $\varepsilon_S=<<eS>>$, $\varepsilon_R=<<eR>>$. O resíduo da densa carrega <<fr>> da energia centrada, contra <<fn>> na NFH. Isso é compatível com o efeito de penalizar o resíduo, mas uma janela e uma semente não o demonstram; o que a seleção traz sobre a penalização está na Seção~\ref{sec:selecao} (<<cmp0>>). O RMSE contra o oráculo nesta janela é <<rd>> (densa) e <<rn>> (NFH); contra o futuro real, <<rdr>> e <<rnr>>. Uma janela só ilustra; as duas são comparadas no teste inteiro na Seção~\ref{sec:resumo}.""",
        sem=cfg["semente"], yd=vec(A(DS["previsao"])), tT=vec(A(DS["theta_T"])), tS=vec(A(DS["theta_S"])), u=vec(A(DS["u"])), eT=pct(ed["eps_T"]), eS=pct(ed["eps_S"]), eR=pct(ed["eps_R"]),
        fr=pct(ed["eps_R"]), fn=pct(en["eps_R"]), cmp0=CMP_CURTO, rd=N(EV["rmse_janela_densa_vs_oraculo"]), rn=N(EV["rmse_janela_nfh_vs_oraculo"]),
        rdr=N(EV["rmse_janela_densa_vs_real"]), rnr=N(EV["rmse_janela_nfh_vs_real"])))

# ---- 12. conversão
add(r"\section{Conversão entre a NFH e a camada densa}\label{sec:conv}")
add(tex(r"""A NFH não é um modelo diferente da densa: é a mesma família linear com os coeficientes reorganizados. Como $\byh=\bM\bD(\bW\bh+\bb)$, uma camada densa equivalente tem $\bW'=\bM\bD\bW$ e $\bb'=\bM\bD\bb$. Por exemplo, o elemento $W'_{0,0}=\sum_i(\bM\bD)_{0,i}W_{i,0}$ é a soma de <<H>> produtos, e a densa equivalente produz, para esta janela, a mesma previsão da NFH com diferença máxima <<e1>>. A volta é $\bW=(\bM\bD)^{-1}\bW'$ e $\bb=(\bM\bD)^{-1}\bb'$; como $\bM\bD$ é inversível (condição <<cond>>), $\bW$ e $\bb$ voltam com erro máximo <<e2>> e <<e3>>. A primeira linha da densa equivalente é $\bW'_{0,:}=<<w0>>$ (<<p>> números) e $b'_0=<<b0>>$. Na direção densa $\to$ NFH, aplicada à densa treinada da semente <<sem>>, a NFH obtida prevê o mesmo que a densa em todo o teste, com diferença máxima <<e4>> (precisão de 32 bits). Isso é o que permite, em outros experimentos, converter uma rede já treinada e só depois ajustá-la.""",
        H=H, e1=N(CV["erro_previsao"], 2), cond=N(B["cond_M"]), e2=N(CV["erro_volta_W"], 2), e3=N(CV["erro_volta_b"], 2), w0=vec(A(CV["W_densa_equivalente"])[0], 3),
        b0=N(A(CV["b_densa_equivalente"])[0]), p=p, sem=cfg["semente"], e4=N(CV["densa_para_nfh_erro_previsao_teste"], 2)))

# ---- 13. N-BEATS-I
add(r"\section{A ambiguidade do N-BEATS-I na mesma previsão}\label{sec:nbeats}")
add(tex(r"""O N-BEATS interpretável também divide a previsão em tendência e sazonalidade, mas com blocos que somam bases \emph{sem exigir ortogonalidade}. Para ver o problema, usa-se para a tendência um polinômio de grau 2 ($<<cT>>$ colunas) e para a sazonalidade uma constante mais os harmônicos 1 a 3 ($<<cS>>$ colunas). As duas bases juntas têm posto $<<pos>>$ em $\R^{<<H>>}$, então há mais colunas que dimensões, e a mesma previsão $\byh$ admite infinitas divisões. A Tabela~\ref{tab:nb} mostra duas, ambas com erro de soma de no máximo <<er>>: a divisão (a) é a de menor norma; a (b) passa uma constante $+1$ da sazonalidade para a tendência. A média da sazonalidade vai de <<ma>> em (a) para <<mb>> em (b), e a tendência de <<ta>> para <<tb>>: o ``nível'' depende de uma escolha arbitrária. Na NFH a divisão é única, e a tendência vale <<tn>> em média, igual à média da previsão (Seção~\ref{sec:TSR}).""",
        cT=NB["colunas_T"], cS=NB["colunas_S"], pos=NB["posto_conjunto"], H=H, er=N(max(NB["erro_soma_a"], NB["erro_soma_b"]), 2), ma=N(NB["media_S_a"]), mb=N(NB["media_S_b"]),
        ta=N(np.mean(NB["T_a"])), tb=N(np.mean(NB["T_b"])), tn=N(NF["media_T"])))
add(tabela("Duas divisões da mesma previsão no estilo N-BEATS-I (a primeira e a segunda) e a divisão única da NFH.", "tab:nb", ["$j$", "$T$ (a)", "$S$ (a)", "$T$ (b)", "$S$ (b)", "$T$ NFH", "$S+R$ NFH", "$\\hat y_j$"],
           [[j, N(NB["T_a"][j]), N(NB["S_a"][j]), N(NB["T_b"][j]), N(NB["S_b"][j]), N(Tn[j]), N(Sn[j] + Rn[j]), N(yh[j])] for j in range(H)], "rrrrrrrr", tam="\\footnotesize"))

# ---- 14. IG
add(r"\section{Integrated Gradients por componente}\label{sec:ig}")
add(tex(r"""Até aqui explicamos \emph{o que} a previsão contém. Falta dizer \emph{quais entradas} produzem cada parte. Para qualquer função escalar $g$ da janela, o Integrated Gradients (IG) com referência nula atribui à entrada $x_{\ell v}$ o valor $\mathrm{IG}_{\ell v}=x_{\ell v}\int_0^1\partial g/\partial x_{\ell v}(\alpha\bX)\,d\alpha$ e satisfaz a \emph{completude}: $\sum_{\ell v}\mathrm{IG}_{\ell v}=g(\bX)-g(\mathbf 0)$. A integral é calculada pela regra do ponto médio com <<ps>> passos. Como a NFH expõe $\bT$ e $\bS$ como saídas, podemos aplicar o IG a um funcional de cada uma, em vez de a um número só: $g_T(\bX)=\overline{\bT}$ (o nível médio da tendência) e $g_S(\bX)=\|\bS\|^2$ (a energia sazonal). Somando as atribuições sobre as <<L>> linhas, obtém-se a contribuição de cada variável.""", ps=cfg["passos_IG"], L=L))
add(tex(r"""\paragraph{Verdade de referência.} Por construção, $x_1$ conduz a tendência e $x_2$ o ciclo. Para ter uma referência numérica e não só uma frase, aplica-se o mesmo IG à função do oráculo, $\mathbf y^\star(\bX)=x_1+x_2$ (últimas <<H>> linhas), com os mesmos $g_T$ e $g_S$ (a energia sazonal calculada pelo projetor sobre $\bFt$). Essa é a atribuição ``certa'' que um bom modelo deveria reproduzir. Os resultados estão na Tabela~\ref{tab:ig} e na Figura~\ref{fig:IG}.""", H=H))
rows = []
for v in range(DD):
    rows.append([f"${vn[v]}$", N(gT["por_variavel"][v]), pct(pT[v]), N(gTo["por_variavel"][v]), pct(pTo[v]), N(gS["por_variavel"][v]), pct(pS[v]), N(gSo["por_variavel"][v]), pct(pSo[v])])
rows.append(["soma", N(gT["soma"]), "", N(gTo["soma"]), "", N(gS["soma"]), "", N(gSo["soma"]), ""])
rows.append(["$g(\\bX)-g(\\mathbf 0)$", N(gT["g_X_menos_g_ref"]), "", N(gTo["g_X_menos_g_ref"]), "", N(gS["g_X_menos_g_ref"]), "", N(gSo["g_X_menos_g_ref"]), ""])
rows.append(["erro de completude", N(gT["erro_completude"], 2), "", N(gTo["erro_completude"], 2), "", N(gS["erro_completude"], 2), "", N(gSo["erro_completude"], 2), ""])
add(tabela("IG por variável (somado sobre as linhas da janela). Participação é o valor absoluto da atribuição da variável dividido pela soma dos valores absolutos. Colunas ``verdade'': IG da função do oráculo.", "tab:ig",
           ["variável", "$g_T$ NFH", "part.", "$g_T$ verdade", "part.", "$g_S$ NFH", "part.", "$g_S$ verdade", "part."], rows, "lrrrrrrrr", tam="\\footnotesize", escala="\\textwidth"))
add(tex(r"""Lendo a tabela: para a tendência, $x_1$ recebe <<a1>> da atribuição de $g_T$ na NFH, contra <<a2>> na verdade de referência; para o ciclo, $x_2$ recebe <<b1>> de $g_S$ na NFH, contra <<b2>> na referência. Os distratores $x_3$ e $x_4$ somam <<c1>> em $g_T$ e <<c2>> em $g_S$. A atribuição da NFH aponta as mesmas variáveis que a construção da série, e a pequena parte que vai para os distratores é erro de estimação do modelo (a referência dá a eles <<d1>> e <<d2>>). A completude vale com erro de ordem <<ec>> (a regra do ponto médio com <<ps>> passos). Conferência extra: o IG do nível médio da previsão, $\overline{\byh}$, coincide com o de $g_T$ (diferença máxima entre as atribuições <<dm>>), como deve ser, pois $\overline{\bT}=\overline{\byh}$.""",
        a1=pct(pT[0]), a2=pct(pTo[0]), b1=pct(pS[1]), b2=pct(pSo[1]), c1=pct(pT[2] + pT[3]), c2=pct(pS[2] + pS[3]), d1=pct(pTo[2] + pTo[3]), d2=pct(pSo[2] + pSo[3]),
        ec=N(max(gT["erro_completude"], gS["erro_completude"]), 2), ps=cfg["passos_IG"], dm=N(IG["media_da_previsao"]["max_dif_vs_g_T"], 2)))
add(figura("fig_IG.pdf", "Participação de cada variável nos funcionais $g_T$ (painel a) e $g_S$ (painel b): barras cinza escuro para a NFH e barras cinza claro com contorno para a verdade de referência (IG da função do oráculo).", "fig:IG", "0.9"))

# ---- 15. abdutiva
add(r"\section{Explicação abdutiva: quais neurônios sustentam a inclinação da tendência}\label{sec:abd}")
add(tex(r"""Os IG mostram quais \emph{entradas} pesam, mas não dão garantia. Uma explicação \emph{abdutiva} procura um conjunto mínimo de características que, mantidas fixas, \emph{garantem} uma propriedade, quaisquer que sejam os valores das outras. Aqui as características são os $<<p>>$ neurônios de $\bh$, cada um em $[-1,1]$ (a imagem de $\tanh$). A propriedade é: ``a inclinação da tendência $\theta_{T,1}(\bh')=\sum_k w_k h'_k+w_0$ mantém o sinal e pelo menos metade do valor atual''. Como $\theta_{T,1}(\bh)=<<th>>$ é <<sg>>, escrevemos $s=<<s>>$ e exigimos $s\,\theta_{T,1}(\bh')\ge\xi=|\theta_{T,1}(\bh)|/2=<<xi>>$. Os pesos da inclinação são $w_k=s\,D_{11}W_{1,k}$ e $w_0=s\,D_{11}b_1$, com $D_{11}=<<D>>$ (a coordenada de índice 1 de $\bD$; os índices de $\bc'$, $\bD$ e das linhas de $\bW$ começam em 0, como nas tabelas do Passo 2, enquanto os dos neurônios $k$ começam em 1).""",
        p=p, th=N(AB["theta_T1"]), sg="positiva" if AB["sinal"] > 0 else "negativa", s=fmt(AB["sinal"], 1), xi=N(AB["xi"]), D=N(Dv[1])))
add(tex(r"""\paragraph{Margens.} Se nenhum neurônio é fixado, o pior caso põe cada $h'_k=-\operatorname{sinal}(w_k)$ e vale $\Gamma_\emptyset=w_0-\sum_k|w_k|=<<g0>>$, abaixo de $\xi$. Fixar o neurônio $k$ no seu valor real $h_k$ melhora esse pior caso em $\delta_k=w_kh_k+|w_k|\ge0$ (o ganho). Para um conjunto $\mathcal A$ de neurônios fixados, o pior caso é $\Gamma(\mathcal A)=\Gamma_\emptyset+\sum_{k\in\mathcal A}\delta_k$, e $\mathcal A$ \emph{garante} a propriedade se $\Gamma(\mathcal A)\ge\xi$. Como os ganhos se somam, o menor conjunto que garante a propriedade vem de escolher os maiores ganhos primeiro (ordem gulosa). A Tabela~\ref{tab:gan} traz $w_k$, $h_k$, o ganho e a posição na ordem gulosa.""", g0=N(AB["Gamma_vazio"])))
wv_, ga = A(AB["w"]), A(AB["ganhos"])
pos = {n_: i + 1 for i, n_ in enumerate(ordem_gulosa)}
add(tabela("Ganhos $\\delta_k=w_kh_k+|w_k|$ e posição na ordem gulosa.", "tab:gan", ["$k$", "$w_k$", "$h_k$", "$\\delta_k$", "posição"],
           [[k + 1, N(wv_[k]), N(h[k]), N(ga[k]), pos[k + 1]] for k in range(p)], "rrrrr", tam="\\footnotesize"))
ac = A(AB["acumulado"])
first_ok = int(np.argmax(ac >= AB["xi"]))
add(tex(r"""Acumulando os ganhos nessa ordem (<<ord>>), o pior caso garantido passa de $<<g0>>$ para $<<a1>>$, $<<a2>>$, $\dots$ e cruza $\xi=<<xi>>$ com $k^*=<<ks>>$ neurônios, em que $\Gamma=<<gk>>$. Com um neurônio a menos ($k^*-1=<<k1>>$), $\Gamma=<<gk1>><\xi$, então <<ks>> é o mínimo. A explicação é o conjunto $\mathcal A^*=\{<<A>>\}$: se esses <<ks>> neurônios forem mantidos nos valores atuais, a inclinação mantém o sinal e pelo menos metade do valor para \emph{qualquer} valor dos outros <<ro>> neurônios em $[-1,1]$. Note o custo: $k^*=<<ks>>$ de <<p>> neurônios é <<pc>> da camada, de modo que a explicação mínima fixa a maior parte dela; é uma garantia, não uma explicação curta. A Figura~\ref{fig:abd} mostra o acumulado.""",
        ord=", ".join(str(t) for t in ordem_gulosa[:kS]) + (r", \ldots" if kS < p else ""), g0=fmt(ac[0]), a1=fmt(ac[1]), a2=fmt(ac[2]), xi=N(AB["xi"]), ks=kS, gk=fmt(ac[kS]), k1=kS - 1,
        gk1=fmt(ac[kS - 1]), A=", ".join(str(t) for t in Aset), ro=p - kS, p=p, pc=pct(kS / p, 2)))
add(figura("fig_abdutiva.pdf", "Pior valor garantido de $s\\,\\theta_{T,1}$ conforme se fixam neurônios, na ordem gulosa (pretos, linha contínua) e na ordem dos Integrated Gradients (cinza médio, tracejada). A linha pontilhada é $\\xi$; a vertical cinza claro marca $k^*$ da ordem gulosa.", "fig:abd", "0.75"))
add(tex(r"""\paragraph{Verificação por amostragem.} Para conferir sem confiar só na álgebra, sorteiam-se $10\,000$ vetores $\bh'$ uniformes em $[-1,1]^{<<p>>}$, com os neurônios de $\mathcal A^*$ fixados nos valores reais: o número de violações da propriedade é <<v>>. Retirando o último neurônio incluído (o <<ult>>), a mesma amostragem encontra <<v2>> violações em $10\,000$, e o pior completamento (cada neurônio livre no extremo desfavorável) leva $s\,\theta_{T,1}$ a <<pc>>, abaixo de $\xi$; calculando $\theta_{T,1}$ diretamente com os pesos da cabeça nesse $\bh'$ obtém-se <<pcd>>, o mesmo valor. A amostragem não prova nada (só a conta das margens prova), mas confirma que a conta e o código concordam.""",
        p=p, v=AB["violacoes_10000_completamentos"], ult=AB["ultimo_neuronio_incluido"], v2=AB["violacoes_10000_sem_o_ultimo"], pc=N(AB["pior_caso_sem_ultimo"]), pcd=N(AB["theta_T1_do_pior_h"])))
add(tex(r"""\paragraph{Comparação com a ordem dos Integrated Gradients.} Foi testada uma única alternativa à ordem gulosa: fixar os neurônios em ordem decrescente de $|w_k(h_k-\bar h_k)|$, com $\bar h_k$ a média de $h_k$ nas janelas de treino. É uma atribuição ``estilo IG'' (linear, relativa à média), não o Integrated Gradients da seção anterior. A ordem obtida é <<oi>> e o limiar $\xi$ só é atingido com $k_{\mathrm{IG}}=<<kI>>$ neurônios, contra $k^*=<<ks>>$ da ordem gulosa. <<concl>>""",
        oi=", ".join(str(t) for t in ordem_ig[:6]) + r", \ldots", kI=kI, ks=kS,
        concl=("Neste exemplo, essa ordem precisou de mais neurônios que o mínimo. Isso mostra que, aqui, o que pesa na média não coincide com o que sustenta a garantia; é um caso, e outra ordem ou outra janela podem dar outro resultado." if kI > kS else
               ("As duas ordens coincidem em tamanho nesta janela." if kI == kS else "A ordem do IG fica abaixo do mínimo, o que não deveria ocorrer; ver o JSON."))))
add(r"""\paragraph{Limites da garantia.} A garantia vale na última camada (sobre $\bh$ como se fosse livre em $[-1,1]^p$, sem passar pelo backbone) e é relativa à caixa $[-1,1]^p$. Ela não diz quais entradas de $\bX$ geram esses neurônios.""")

# ---- 16. resumo no teste
add(r"\section{De uma janela para o teste inteiro}\label{sec:resumo}")
tabrow = []
for s_ in sorted(set(conf.semente)):
    g = conf[conf.semente == s_].set_index("variante")
    tabrow.append([s_] + [N(g.loc[v, k]) for k in ("test_RMSE", "test_MAE") for v in ("densa", "densa+E", "NFH")])
    _r = proj[proj.semente == s_].iloc[0]; tabrow[-1] += [N(_r.proj_test_RMSE), N(_r.proj_test_MAE)]
add(tex(r"""Uma janela só ilustra. A Tabela~\ref{tab:conf} traz o teste inteiro, em cada semente da confirmação, com as duas últimas colunas do controle adicional descrito a seguir. Nela, a semente <<sem>> é a explicada neste documento. As médias e os desvios estão na primeira página.""", sem=cfg["semente"]))
_t = ("A projeção não explica o ganho da NFH: ela percorre " + pct(PJ_fracao, 2) + " da distância entre a densa e a NFH, e a causa do ganho não foi isolada." if PJ_NAO_EXPLICA
      else "Os números mostram que a projeção percorre " + pct(PJ_fracao, 2) + " da distância entre a densa e a NFH; a leitura de que a projeção explica parte do ganho tem de ser discutida, e a causa não foi isolada.")
add(tex(r"""\paragraph{Controle adicional: a densa projetada em $\bT\oplus\bS$.} Depois da confirmação, e a pedido de uma auditoria externa, calculou-se com o código do projeto (o projetor $\bQ\bQ^\top$ de \texttt{exemplo\_livro/treinar.py}) o que acontece se a previsão da camada densa for projetada em $\bT\oplus\bS$ \emph{sem treino}: $\byh_{\mathrm{proj}}=\byh-\bQ\bQ^\top\byh$, isto é, a decomposição pura com o resíduo descartado. Para cada semente <<s0>> a <<s1>> retreinou-se a densa (a mesma do critério; o RMSE e o MAE reproduzem os de \texttt{confirmacao.csv} com diferença máxima de <<dif>> no RMSE) e avaliou-se a projeção no teste (\texttt{projecao\_confirmacao.csv}). O RMSE de teste médio passa de <<d1>> (densa) para <<p1>> (projetada), e o MAE, de <<d2>> para <<p2>>; a projeção tem RMSE menor que a densa em <<prm>> das <<ns>> sementes, mas a NFH (<<n1>>) tem RMSE menor que a densa projetada em <<nvp>> das <<ns>>. A parte da energia da previsão da densa que cai no espaço residual é só <<fr>> em média. <<txt>> Este é um controle adicional, relatado depois dos resultados: não entrou na seleção e não altera o veredito pré-registrado. Não há, neste exemplo, uma densa com regularização equivalente (por exemplo, ridge ou parada antecipada mais curta), de modo que não se sabe se o ganho é da estrutura NFH ou de qualquer regularização.""",
        s0=conf.semente.min(), s1=conf.semente.max(), ns=nS, dif=N(float(proj.dif_vs_confirmacao_RMSE.max()), 2), d1=N4(PJ["densa_test_RMSE"][0]), p1=N4(PJ["proj_test_RMSE"][0]),
        d2=N4(PJ["densa_test_MAE"][0]), p2=N4(PJ["proj_test_MAE"][0]), prm=PJ_prm, n1=N4(cf["NFH_test_RMSE_media"]), nvp=PJ_nfh_vs_proj, fr=pct(PJ_frac_res, 2), txt=_t))
add(tabela("RMSE e MAE de teste por semente (sementes de confirmação).", "tab:conf",
           ["semente", "RMSE densa", "RMSE densa+E", "RMSE NFH", "MAE densa", "MAE densa+E", "MAE NFH", "RMSE densa proj.", "MAE densa proj."], tabrow, "rrrrrrrrr", fonte="\\texttt{confirmacao.csv} e \\texttt{projecao\\_confirmacao.csv}", tam="\\footnotesize", escala="\\textwidth"))

# ---- 17. o que mostra / não mostra
add(r"\section{O que este exemplo mostra e o que não mostra}\label{sec:limites}")
add(tex(r"""\subsection*{O que mostra}
Mostra, com todas as contas, como a NFH transforma <<p>> números de um backbone em uma previsão já dividida em $\bT$, $\bS$ e $\bR$; que a soma é exata e a média de $\bT$ é a média da previsão; que a divisão é única e pode ser comparada com a verdade conhecida (nesta janela, o erro máximo da previsão contra o oráculo é <<mx>>); que a conversão para uma camada densa equivalente tem erro de ordem <<ec>>; que o IG por componente aponta $x_1$ para a tendência e $x_2$ para o ciclo, como na referência; e que é possível dar uma garantia formal (a explicação abdutiva, com $k^*=<<ks>>$ de <<p>> neurônios, isto é, <<kpc>> da camada: uma garantia, não uma explicação curta). Mostra também que, nesta série e com este protocolo pré-registrado, a combinação escolhida pela validação confirmou-se no teste: <<ven>>.""",
        p=p, mx=N(EV["max_abs_previsao"]), ec=N(max(CV["erro_previsao"], CV["erro_volta_W"]), 2), ks=kS, kpc=pct(kS / p, 2),
        ven=("a NFH teve RMSE e MAE de teste médios de " + N(rn[0]) + " e " + N(mn[0]) + ", contra " + N(rd[0]) + " e " + N(md_[0]) + " da densa, e vitória em " + str(cf["vitorias_test_RMSE_vs_densa+E"]) + " de " + str(nS) + " sementes sobre a densa+E em RMSE") if ven == "SUCESSO" else "a confirmação falhou, como dito na primeira página"))
add(tex(r"""\subsection*{O que não mostra}
Não mostra que a NFH é melhor que a camada densa em geral. A série foi \emph{escolhida} para ser favorável: o futuro é uma função linear da janela e cai quase todo em $\bT\oplus\bS$ (verdade com $\varepsilon_R^\star=<<eR>>$ da energia), e o ruído ($\sigma=<<sg>>$) com poucas janelas de treino é o regime em que, no teste de 16/11, a penalização do resíduo pareceu ajudar (isto é uma hipótese; a projeção da densa em $\bT\oplus\bS$ não a sustenta, ver Seção~\ref{sec:resumo}). Em dados reais da dissertação, a NFH treinada do zero perdeu da densa no teste de 16/11 (\texttt{teste\_exemplificacao\_nfh/resultados/zero.csv}; compara médias simples das linhas do arquivo, sem pareamento por semente, e as variantes têm números diferentes de linhas, então a direção é indicativa, não uma comparação pareada). A grade de seleção (Tabela~\ref{tab:selecao}) mostra que, mesmo aqui, só <<nq>> de <<nc>> combinações superaram a densa+E na validação pela regra, e em $n=300$, <<n3>>. Também não mostra nada sobre backbones grandes: o backbone é uma camada. O treino usa as primeiras $n$ janelas do segmento de treino (a T1 usa as últimas), outro motivo para não tomar este exemplo como réplica da T1. A atribuição por IG foi comparada com uma verdade construída pelo próprio autor; ela vale para esta série e para a janela explicada, e uma única janela não mede a qualidade média da explicação. A explicação abdutiva vale na última camada, não nas entradas. Por último, a escolha da série veio do teste de exemplificação de 16/11, em que a NFH já tinha ido bem \emph{nesta mesma série} (mesma semente de dados, mesmos cortes e mesmas janelas de teste); antes desta confirmação o coordenador também viu, por engano de formato, diferenças de teste de 36 linhas de outro experimento (registrado em \texttt{INSTRUCOES.md}, \S10). A seleção desta pasta, porém, usou só as colunas de validação, e a confirmação usou sementes que não participaram da seleção, mas elas só mudam a inicialização e a ordem dos lotes: o resultado não foi confirmado em outra realização do ruído nem em outra série. A projeção da densa em $\bT\oplus\bS$ (Seção~\ref{sec:resumo}) não explica o ganho da NFH e a causa do ganho não foi isolada; $E=<<E>>$ épocas e taxa $<<lra>>$ da densa+E também diferem da T1 (Seção~\ref{sec:modelo}).""",
        eR=pct(ev["eps_R"]), sg=fmt(cfg["sigma"]), E=pex["E_EXTRA"], lra=fmt(cfg["lr_ajuste"]), nq=n_qual, nc=len(grade), n3=("nenhuma" if n300q == 0 else str(n300q))))
add(r"\end{document}")

(AQUI / "exemplo_passo_a_passo.tex").write_text(finalizar("\n\n".join(S)), encoding="utf-8")
r = subprocess.run([str(Path.home() / ".cache/v2work/bin/tectonic"), "exemplo_passo_a_passo.tex"], cwd=AQUI, capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-3000:])
raise SystemExit(r.returncode)
