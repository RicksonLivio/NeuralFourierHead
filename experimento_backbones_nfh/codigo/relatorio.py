#!/usr/bin/env python3
"""
relatorio.py: gerador do relatório do experimento de backbones da NFH (protocolo: ../INSTRUCOES.md).

Lê resultados/t1.csv, t2.csv, confirmacao.csv (descarta linhas de cabeçalho repetidas, dataset == 'dataset'),
as filas (resultados/filas/*.json), resultados/selecao_T1.md e selecao_T2.md, o §10 do protocolo, o exemplo
passo a passo (exemplo_sintetico/confirmacao.{csv,json}) e, se existirem, resultados/explicacoes/*.json.
Escreve em relatorio/ (ou --saida): numeros.json (todo número que entra no texto), tabelas .tex, figuras em
tons de cinza, relatorio.tex e relatorio.pdf (compilado com tectonic).

Nenhum número do texto é digitado à mão: tudo sai dos arquivos acima. As únicas constantes são as do protocolo
(limiares do §6 e do §10), declaradas em PROTOCOLO e gravadas em numeros.json. Fase ausente ou vazia não quebra:
o relatório diz que a fase não rodou.

IMPORTANTE: este script lê colunas test_*. Só deve ser rodado depois da confirmação (a seleção T1/T2 continua
feita só por selecionar.py, que lê val_*).

Uso (a partir da raiz):
  .venv/bin/python experimento_backbones_nfh/codigo/relatorio.py
  .venv/bin/python experimento_backbones_nfh/codigo/relatorio.py --res /tmp/fake/resultados --saida /tmp/fake/relatorio \
      --exemplo /tmp/fake/exemplo      # teste com dados falsos
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
B = ROOT / "experimento_backbones_nfh"
TECTONIC = Path.home() / ".cache" / "v2work" / "bin" / "tectonic"

COLS = ["dataset", "modo", "backbone", "largura", "fracao", "seed", "variante", "K", "lambda",
        "val_RMSE", "val_MAE", "test_RMSE", "test_MAE", "cobertura_mediana", "epocas", "epoca_escolhida",
        "tempo_s", "params", "n_tr", "n_val", "n_te", "d", "erro_conversao"]
NUM = ["largura", "fracao", "seed", "K", "lambda", "val_RMSE", "val_MAE", "test_RMSE", "test_MAE",
       "cobertura_mediana", "epocas", "epoca_escolhida", "tempo_s", "params", "n_tr", "n_val", "n_te", "d",
       "erro_conversao"]

# ---------------------------------------------------------------------------
# Constantes do protocolo: um único dicionário, conferido contra o código que executa o experimento.
# Sementes de T2/confirmação vêm de selecionar.py (importado). As demais são lidas por AST (sem importar torch) de
# executar.py, src/v3_common.py, src/v2_common.py e exemplo_sintetico/confirmar.py. Só o critério do §6 (mínimo de
# vitórias) e o ``rótulo'' das taxas em texto não existem como constante no código e ficam declarados aqui.
# ---------------------------------------------------------------------------
import ast  # noqa: E402

sys.path.insert(0, str(AQUI))
import selecionar as _sel  # noqa: E402  (só pandas; não executa nada ao importar)


def _ast_const(arquivo: Path, nome: str, classe: str | None = None):
    """Valor literal de uma constante (atribuição de módulo, campo de dataclass, ou os.environ.get(..., padrão))."""
    arvore = ast.parse(Path(arquivo).read_text(encoding="utf-8"))
    corpo = arvore.body
    if classe:
        corpo = next((n.body for n in arvore.body if isinstance(n, ast.ClassDef) and n.name == classe), [])

    def avalia(v):
        if isinstance(v, ast.Call) and getattr(v.func, "id", "") == "range":
            return list(range(*[ast.literal_eval(a) for a in v.args]))
        if isinstance(v, ast.Call) and v.args:               # os.environ.get("X", 15) -> 15; int(...) -> interno
            return avalia(v.args[-1])
        return ast.literal_eval(v)
    for n in corpo:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == nome and n.value is not None:
            return avalia(n.value)
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == nome:
                    return avalia(n.value)
                if isinstance(t, ast.Tuple) and isinstance(n.value, ast.Tuple):
                    for e, v in zip(t.elts, n.value.elts):
                        if isinstance(e, ast.Name) and e.id == nome:
                            return avalia(v)
    raise KeyError(f"{nome} não encontrado em {arquivo}")


def _montar_protocolo() -> dict:
    ex, v3, v2 = AQUI / "executar.py", ROOT / "src" / "v3_common.py", ROOT / "src" / "v2_common.py"
    cf = B / "exemplo_sintetico" / "confirmar.py"
    P_ = {
        "sementes_t2": list(_sel.SEEDS_T2), "sementes_confirmacao": list(_sel.SEEDS_CONF),
        "minimo_vitorias_rmse_vs_densaE": 4,                       # §6 (INSTRUCOES.md): pelo menos 4 das 5 sementes
        "H": int(_ast_const(v3, "H3")), "L": int(_ast_const(v3, "L3")), "q": int(_ast_const(v3, "Q3")),
        "K_proj": int(_ast_const(v3, "K3")),
        "proporcao_treino": float(_ast_const(v2, "train_ratio", "Config")), "proporcao_val": float(_ast_const(v2, "val_ratio", "Config")),
        "taxa_aprendizado": float(_ast_const(v2, "lr", "Config")), "taxa_ajuste_densaE": float(_ast_const(ex, "LR_AJUSTE")),
        "epocas_extra": int(_ast_const(ex, "E_EXTRA")), "epocas_max": int(_ast_const(v2, "max_epochs", "Config")),
        "paciencia": int(_ast_const(v2, "patience", "Config")),
        "exemplo_H": 8,                                            # exemplo_sintetico (livro); §10 de 21/11
    }
    P_["proporcao_teste"] = round(1.0 - P_["proporcao_treino"] - P_["proporcao_val"], 10)
    sem_ex = list(_ast_const(cf, "SEMENTES"))
    P_["exemplo_sementes"] = len(sem_ex)
    P_["exemplo_sementes_faixa"] = [min(sem_ex), max(sem_ex)]
    m = re.search(r"vitorias_ok\"\]\s*=\s*[^\n]*>=\s*(\d+)", cf.read_text(encoding="utf-8"))
    if not m:
        raise KeyError("limiar de vitórias não encontrado em exemplo_sintetico/confirmar.py")
    P_["exemplo_minimo_vitorias"] = int(m.group(1))
    # conferência cruzada: a seleção usa as mesmas sementes e o executor não mudou a taxa do §3
    assert P_["sementes_confirmacao"] == list(_sel.SEEDS_CONF) and P_["sementes_t2"] == list(_sel.SEEDS_T2)
    return P_


PROTOCOLO = _montar_protocolo()


def faixa(xs) -> str:
    """[10, 11, 12, 13, 14] -> '10 a 14'; [0, 1, 2] -> '0, 1 e 2' (texto gerado das constantes)."""
    xs = list(xs)
    if len(xs) > 3 and xs == list(range(xs[0], xs[-1] + 1)):
        return f"{xs[0]} a {xs[-1]}"
    return ", ".join(map(str, xs[:-1])) + f" e {xs[-1]}" if len(xs) > 1 else str(xs[0])


def pot10(x: float) -> str:
    """1e-3 -> '$10^{-3}$' (a partir da constante)."""
    e = int(round(np.log10(x)))
    return f"$10^{{{e}}}$"


def divisao() -> str:
    return "/".join(str(int(round(100 * PROTOCOLO[k]))) for k in ("proporcao_treino", "proporcao_val", "proporcao_teste"))


CH = ["dataset", "modo", "backbone", "largura", "fracao", "seed"]
CH_COMB = ["dataset", "modo", "backbone", "largura", "fracao"]
VAR_ING = ["ingenua_media", "ingenua_ultimo", "ingenua_sazonal24"]


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------
def ler_csv(caminho) -> tuple[pd.DataFrame, dict]:
    """Lê um CSV de resultados. Devolve (df, info). Ausente ou vazio -> df vazio com as colunas."""
    info = {"arquivo": str(caminho), "existe": False, "linhas_brutas": 0, "cabecalhos_repetidos": 0,
            "duplicadas_descartadas": 0, "linhas": 0}
    p = Path(caminho)
    vazio = pd.DataFrame(columns=COLS)
    if not p.exists() or p.stat().st_size == 0:
        return vazio, info
    info["existe"] = True
    try:
        df = pd.read_csv(p, dtype=str, keep_default_na=False, on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        return vazio, info
    if "dataset" not in df.columns:
        return vazio, info
    info["linhas_brutas"] = int(len(df))
    rep = df["dataset"] == "dataset"
    info["cabecalhos_repetidos"] = int(rep.sum())
    df = df[~rep].copy()
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("dataset", "modo", "backbone", "variante"):
        df[c] = df[c].astype(str)
    antes = len(df)
    df = df.drop_duplicates(subset=CH + ["variante"], keep="last")
    info["duplicadas_descartadas"] = int(antes - len(df))
    info["linhas"] = int(len(df))
    return df.reset_index(drop=True), info


def eh_nfh(v: str) -> bool:
    return v.startswith("conv_") or v.startswith("zero_")


def parse_variante(v: str):
    m = re.match(r"^(conv|zero)_L(\d+(?:\.\d+)?)_K(\d+)$", v)
    return (m.group(1), float(m.group(2)), int(m.group(3))) if m else (None, None, None)


def so_treinados(df):
    return df[df["backbone"] != ""]


def ler_json(p):
    p = Path(p)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_selecao_md(caminho) -> dict:
    """Lê selecao_T1.md / selecao_T2.md. Devolve {'existe', 'cabecalho' (contagens), 'linhas' (DataFrame)}."""
    p = Path(caminho)
    out = {"existe": False, "contagens": {}, "linhas": pd.DataFrame()}
    if not p.exists() or p.stat().st_size == 0:
        return out
    out["existe"] = True
    txt = p.read_text(encoding="utf-8")
    m = re.search(r"Combinações[^:]*:\s*(\d+);\s*passam:\s*(\d+);\s*vão para [^:]*:\s*(\d+)", txt)
    if m:
        out["contagens"] = {"total": int(m.group(1)), "passam": int(m.group(2)), "seguem": int(m.group(3))}
    linhas = []
    cab = None
    for ln in txt.splitlines():
        if not ln.startswith("|"):
            continue
        cel = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cab is None:
            cab = cel
            continue
        if set("".join(cel)) <= set("-: "):
            continue
        if len(cel) == len(cab):
            linhas.append(dict(zip(cab, cel)))
    df = pd.DataFrame(linhas)
    if len(df):
        def pc(s):
            return float(s.replace("%", "").replace("+", "")) / 100.0
        df = df.rename(columns={"conjunto": "dataset", "fração": "fracao"})
        df["fracao"] = df["fracao"].astype(float)
        if "largura" in df.columns:
            df["largura"] = df["largura"].astype(int)
        for c in df.columns:
            if c.startswith("rel ") or c.startswith("média rel") or c.startswith("pior semente"):
                df[c] = df[c].map(pc)
        passa = df["passa"] == "sim"
        df["passa"] = passa
        for c in ("T2", "confirmação"):
            if c in df.columns:
                df[c] = df[c] == "sim"
    out["linhas"] = df
    return out


def ler_secao10() -> list[str]:
    p = B / "INSTRUCOES.md"
    if not p.exists():
        return []
    txt = p.read_text(encoding="utf-8")
    i = txt.find("## 10.")
    if i < 0:
        return []
    corpo = txt[i:].split("\n", 1)[1]
    itens, atual = [], None
    for ln in corpo.splitlines():
        if ln.startswith("- "):
            if atual is not None:
                itens.append(atual)
            atual = ln[2:].strip()
        elif atual is not None and ln.strip():
            atual += " " + ln.strip()
    if atual is not None:
        itens.append(atual)
    return itens


# ---------------------------------------------------------------------------
# LaTeX: escape e formatação numérica (vírgula decimal)
# ---------------------------------------------------------------------------
UNI = {"≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$", "−": r"$-$", "→": r"$\to$", "λ": r"$\lambda$", "φ": r"$\varphi$", "ϕ": r"$\phi$", "Δ": r"$\Delta$", "δ": r"$\delta$", "α": r"$\alpha$", "β": r"$\beta$", "μ": r"$\mu$", "ω": r"$\omega$",
       "ε": r"$\varepsilon$", "σ": r"$\sigma$", "ρ": r"$\rho$", "Φ": r"$\Phi$", "‖": r"$\|$", "²": r"$^2$",
       "≈": r"$\approx$", "∈": r"$\in$", "ℝ": r"$\mathbb{R}$", "⊕": r"$\oplus$", "≠": r"$\neq$",
       "±": r"$\pm$", "κ": r"$\kappa$", "ₑ": "e", "′": "'", "\u00a0": " ", "↑": r"$\uparrow$", "↓": r"$\downarrow$",
       "∑": r"$\sum$", "Σ": r"$\Sigma$", "θ": r"$\theta$", "τ": r"$\tau$", "·": r"$\cdot$", "…": "...",
       "‑": "-", "–": "--", "→": r"$\to$", "⁻": r"$^{-}$", "¹": r"$^1$", "₀": r"$_0$", "₁": r"$_1$"}
ESP = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
       "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(s, quebra=False) -> str:
    out = []
    for ch in str(s):
        if ch in ESP:
            out.append(ESP[ch])
            if quebra and ch in "_":
                out.append(r"\allowbreak{}")
        elif ch in UNI:
            out.append(UNI[ch])
        elif ord(ch) > 0x2000 and ch not in "“”‘’—":
            out.append("?")
        else:
            out.append(ch)
            if quebra and ch in "/.-":
                out.append(r"\allowbreak{}")
    return "".join(out)


def md2tex(s: str) -> str:
    """Markdown mínimo (`código`, **negrito**) para LaTeX."""
    partes = re.split(r"(`[^`]*`|\*\*[^*]+\*\*|(?<![*\w])\*[^*\s](?:[^*]*[^*\s])?\*(?![*\w]))", s)
    out = []
    for p in partes:
        if p.startswith("`") and p.endswith("`") and len(p) >= 2:
            out.append(r"\texttt{" + esc(p[1:-1], quebra=True) + "}")
        elif p.startswith("**") and p.endswith("**") and len(p) >= 4:
            out.append(r"\textbf{" + esc(p[2:-2]) + "}")
        elif p.startswith("*") and p.endswith("*") and len(p) >= 3 and not p.startswith("**"):
            out.append(r"\emph{" + esc(p[1:-1]) + "}")
        else:
            out.append(esc(p))
    return "".join(out)


def nf(x, nd=4, sinal=False) -> str:
    """Número com vírgula decimal (e \\textminus para negativos)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/d"
    t = f"{x:+.{nd}f}" if sinal else f"{x:.{nd}f}"
    t = t.replace(".", ",")
    return t.replace("-", r"\textminus{}")


def pc(x, nd=2) -> str:
    """Fração -> percentual com sinal."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/d"
    return nf(100 * x, nd, sinal=True) + r"\%"


def pcn(x, nd=2) -> str:
    """Fração -> percentual sem sinal obrigatório."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/d"
    return nf(100 * x, nd) + r"\%"


def mdp(m, s, nd=4) -> str:
    if m is None or not np.isfinite(m):
        return "n/d"
    return f"{nf(m, nd)} $\\pm$ {nf(s, nd) if np.isfinite(s) else 'n/d'}"


def quart(x, nd=2) -> str:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "n/d"
    q1, md, q3 = np.percentile(x, [25, 50, 75])
    return f"{pc(md, nd)} [{nf(100 * q1, nd, True)}; {nf(100 * q3, nd, True)}]"


def cab_linhas(c: str) -> str:
    """Quebra o cabeçalho em linhas de até duas palavras (\\shortstack)."""
    pal = c.split(" ")
    if len(pal) < 2:
        return c
    ls = [" ".join(pal[i:i + 2]) for i in range(0, len(pal), 2)]
    return r"\shortstack[l]{" + r"\\ ".join(ls) + "}"


def dec(x) -> str:
    """Número curto com vírgula decimal (0,1; 0,25; 1)."""
    return f"{x:g}".replace(".", ",")


def lista_pt(itens) -> str:
    itens = [str(i) for i in itens]
    if len(itens) <= 1:
        return "".join(itens)
    return ", ".join(itens[:-1]) + " e " + itens[-1]


def tabela(nome, legenda, cab, linhas, alinh, saida: Path, fonte, longa=False, rotulo=None, tam=r"\scriptsize"):
    """Grava um .tex com a tabela completa (legenda acima, fonte abaixo) e devolve o texto de \\input."""
    rotulo = rotulo or f"tab:{nome}"
    cab = [cab_linhas(c) for c in cab]
    L = []
    if longa:
        L += [tam, r"\setlength{\tabcolsep}{2pt}",
              r"\begin{longtable}{" + alinh + "}", r"\caption{" + legenda + r"}\label{" + rotulo + r"}\\",
              r"\toprule", " & ".join(cab) + r" \\ \midrule", r"\endfirsthead",
              r"\toprule", " & ".join(cab) + r" \\ \midrule", r"\endhead",
              r"\bottomrule", r"\multicolumn{" + str(len(cab)) + r"}{l}{\textit{" + "Fonte: " + fonte + r"}}\\", r"\endlastfoot"]
        L += [" & ".join(l) + r" \\" for l in linhas]
        L += [r"\end{longtable}", r"\normalsize"]
    else:
        L += [r"\begin{table}[htbp]", r"\centering", tam, r"\setlength{\tabcolsep}{2pt}",
              r"\caption{" + legenda + r"}\label{" + rotulo + "}",
              r"\begin{tabular}{" + alinh + "}", r"\toprule", " & ".join(cab) + r" \\ \midrule"]
        L += [" & ".join(l) + r" \\" for l in linhas]
        L += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip{\scriptsize Fonte: " + fonte + "}",
              r"\end{table}"]
    (saida / f"{nome}.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    return r"\input{" + nome + "}\n"


# ---------------------------------------------------------------------------
# Análises
# ---------------------------------------------------------------------------
def relativas(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (tarefa, variante NFH) com a diferença relativa contra a densa+E (rE_*) e contra a densa (rD_*)."""
    d = so_treinados(df)
    if d.empty:
        return pd.DataFrame()
    cols = ["val_RMSE", "val_MAE", "test_RMSE", "test_MAE"]
    refE = d[d["variante"] == "densa+E"][CH + cols].rename(columns={c: "E_" + c for c in cols})
    refD = d[d["variante"] == "densa"][CH + cols].rename(columns={c: "D_" + c for c in cols})
    nfh = d[d["variante"].map(eh_nfh)]
    m = nfh.merge(refE, on=CH, how="inner").merge(refD, on=CH, how="inner")
    for c in cols:
        m["rE_" + c] = m[c] / m["E_" + c] - 1
        m["rD_" + c] = m[c] / m["D_" + c] - 1
    return m


def conj(r) -> str:
    return f"{r['dataset']}-{r['modo']}"


def melhor_t1(m: pd.DataFrame) -> pd.DataFrame:
    """Melhor variante NFH por (conjunto, modo, backbone, fração), escolhida só pelo rE_val_RMSE (como selecionar.py)."""
    if m.empty:
        return pd.DataFrame()
    mm = m[(m["seed"] == 0) & (m["largura"] == 1)]
    if mm.empty:
        return pd.DataFrame()
    ch = ["dataset", "modo", "backbone", "fracao"]
    # a linha inteira da melhor variante (head(1)), não first(), que misturaria colunas de variantes diferentes quando há NaN
    best = (mm.dropna(subset=["rE_val_RMSE"]).sort_values("rE_val_RMSE", kind="mergesort")
            .groupby(ch, sort=False).head(1).reset_index(drop=True))
    best["conjunto"] = best.apply(conj, axis=1)
    return best.sort_values("rE_val_RMSE").reset_index(drop=True)


def cruzar_t1_md(best: pd.DataFrame, md: dict) -> tuple[pd.DataFrame, dict]:
    """Acrescenta as flags passa/vai_T2 lidas do selecao_T1.md e confere com o CSV."""
    conf = {"md_existe": md["existe"]}
    best = best.copy()
    best["passa"] = np.nan
    best["vai_T2"] = np.nan
    if not md["existe"] or md["linhas"].empty or best.empty:
        return best, conf
    L = md["linhas"][["dataset", "modo", "backbone", "fracao", "melhor variante NFH", "rel RMSE val", "passa", "T2"]]
    L = L.rename(columns={"melhor variante NFH": "var_md", "rel RMSE val": "rel_md", "T2": "vai_md", "passa": "passa_md"})
    j = best.merge(L, on=["dataset", "modo", "backbone", "fracao"], how="left")
    best["passa"] = j["passa_md"].values
    best["vai_T2"] = j["vai_md"].values
    ok = j["var_md"].notna()
    conf["linhas_no_md"] = int(len(L))
    conf["combinacoes_csv_sem_linha_no_md"] = int((~ok).sum())
    conf["variante_diferente_do_md"] = int((ok & (j["var_md"] != j["variante"])).sum())
    conf["rel_val_diferente_do_md_(>0,006)"] = int((ok & ((j["rel_md"] - j["rE_val_RMSE"]).abs() > 0.00006)).sum())
    return best, conf


def estat_grupo(best: pd.DataFrame, chave) -> list[dict]:
    out = []
    for g, d in best.groupby(chave, sort=True):
        reg = {"grupo": g if isinstance(g, str) else " ".join(map(str, g)), "n": int(len(d)),
               "vence_val": int((d["rE_val_RMSE"] < 0).sum()), "vence_test": int((d["rE_test_RMSE"] < 0).sum()),
               "vence_val_ambos": int(((d["rE_val_RMSE"] < 0) & (d["rE_val_MAE"] < 0)).sum()),
               "vence_test_ambos": int(((d["rE_test_RMSE"] < 0) & (d["rE_test_MAE"] < 0)).sum()),
               "qualificaram": int(d["passa"].fillna(False).astype(bool).sum()) if d["passa"].notna().any() else None,
               "foram_T2": int(d["vai_T2"].fillna(False).astype(bool).sum()) if d["vai_T2"].notna().any() else None}
        for sp in ("val", "test"):
            for ref, nr in (("E", "densaE"), ("D", "densa")):
                for met in ("RMSE", "MAE"):
                    x = d[f"r{ref}_{sp}_{met}"].to_numpy(float)
                    q = np.nanpercentile(x, [25, 50, 75]) if np.isfinite(x).any() else [np.nan] * 3
                    reg[f"{sp}_{nr}_{met}"] = {"q1": float(q[0]), "mediana": float(q[1]), "q3": float(q[2])}
        out.append(reg)
    return out


def tab_estat(estat, tipo) -> tuple[list, list]:
    """Linhas (strings LaTeX) para a tabela de mediana e quartis; tipo 'val' ou 'test'."""
    linhas = []
    for r in estat:
        c = lambda k: (f"{pc(r[f'{tipo}_{k}']['mediana'])} [{nf(100 * r[f'{tipo}_{k}']['q1'], 2, True)}; "
                       f"{nf(100 * r[f'{tipo}_{k}']['q3'], 2, True)}]")
        linhas.append([esc(r["grupo"]), str(r["n"]), c("densaE_RMSE"), c("densaE_MAE"), c("densa_RMSE"), c("densa_MAE")])
    return linhas


def variantes_t1(m: pd.DataFrame, best: pd.DataFrame) -> list[dict]:
    """Por variante NFH: em quantas combinações venceu a densa+E (val/teste) e quantas vezes foi a escolhida."""
    mm = m[(m["seed"] == 0) & (m["largura"] == 1)]
    esc_ = best["variante"].value_counts().to_dict() if not best.empty else {}
    out = []
    for v, d in mm.groupby("variante"):
        fam, lam, K = parse_variante(v)
        out.append({"variante": v, "familia": fam, "lambda": lam, "K": K, "n": int(len(d)),
                    "vence_val": int((d["rE_val_RMSE"] < 0).sum()), "vence_test": int((d["rE_test_RMSE"] < 0).sum()),
                    "mediana_rel_test_RMSE": float(np.nanmedian(d["rE_test_RMSE"])),
                    "escolhida_na_validacao": int(esc_.get(v, 0))})
    return sorted(out, key=lambda r: (-r["vence_test"], r["variante"]))


def agrega_variantes(vs: list[dict]) -> list[dict]:
    out = []
    for campo, rot in (("familia", "família"), ("lambda", "λ"), ("K", "K")):
        g = {}
        for r in vs:
            k = r[campo]
            a = g.setdefault(k, {"n": 0, "vence_val": 0, "vence_test": 0, "escolhida": 0})
            a["n"] += r["n"]
            a["vence_val"] += r["vence_val"]
            a["vence_test"] += r["vence_test"]
            a["escolhida"] += r["escolhida_na_validacao"]
        for k, a in sorted(g.items(), key=lambda kv: str(kv[0])):
            out.append({"campo": rot, "valor": k, **a})
    return out


# ---------- T2 ----------
def resumo_t2(df: pd.DataFrame, md: dict) -> tuple[pd.DataFrame, dict]:
    """Por (combinação, largura): a melhor variante pela média de rE_val_RMSE (como selecionar.py) e as médias sobre as sementes."""
    m = relativas(df)
    if m.empty:
        return pd.DataFrame(), {}
    m = m[m["seed"].isin(PROTOCOLO["sementes_t2"])]
    if m.empty:
        return pd.DataFrame(), {}
    ch = ["dataset", "modo", "backbone", "fracao", "largura", "variante", "K", "lambda"]
    g = m.groupby(ch, dropna=False)
    agg = g.agg(n_sem=("seed", "nunique"), rE_val_RMSE=("rE_val_RMSE", "mean"), rE_val_MAE=("rE_val_MAE", "mean"),
                rE_test_RMSE=("rE_test_RMSE", "mean"), rE_test_MAE=("rE_test_MAE", "mean"),
                rD_test_RMSE=("rD_test_RMSE", "mean"), rD_test_MAE=("rD_test_MAE", "mean"),
                rD_val_RMSE=("rD_val_RMSE", "mean"), pior_sem_val=("rE_val_RMSE", "max"),
                val_RMSE=("val_RMSE", "mean"), test_RMSE=("test_RMSE", "mean"), test_MAE=("test_MAE", "mean"),
                E_val_RMSE=("E_val_RMSE", "mean"), E_test_RMSE=("E_test_RMSE", "mean"), E_test_MAE=("E_test_MAE", "mean"),
                D_test_RMSE=("D_test_RMSE", "mean"), D_test_MAE=("D_test_MAE", "mean")).reset_index()
    agg = agg[agg["n_sem"] == len(PROTOCOLO["sementes_t2"])]
    if agg.empty:
        return pd.DataFrame(), {}
    ch2 = ["dataset", "modo", "backbone", "fracao", "largura"]
    best = (agg.dropna(subset=["rE_val_RMSE"]).sort_values("rE_val_RMSE", kind="mergesort")
            .groupby(ch2, sort=False).head(1).reset_index(drop=True))
    best["conjunto"] = best.apply(conj, axis=1)
    best["passa_calc"] = (best["rE_val_RMSE"] < 0) & (best["rE_val_MAE"] <= 0) & (best["pior_sem_val"] < 0)
    conf = {"md_existe": md["existe"]}
    best["passa"] = np.nan
    best["vai_conf"] = np.nan
    if md["existe"] and not md["linhas"].empty:
        L = md["linhas"][["dataset", "modo", "backbone", "fracao", "largura", "melhor variante", "passa", "confirmação"]]
        L = L.rename(columns={"melhor variante": "var_md", "passa": "passa_md", "confirmação": "conf_md"})
        j = best.merge(L, on=ch2, how="left")
        best["passa"] = j["passa_md"].values
        best["vai_conf"] = j["conf_md"].values
        ok = j["var_md"].notna()
        conf["variante_diferente_do_md"] = int((ok & (j["var_md"] != j["variante"])).sum())
        conf["passa_diferente_do_calculo"] = int((ok & (j["passa_md"] != j["passa_calc"])).sum())
        conf["combinacoes_sem_linha_no_md"] = int((~ok).sum())
    return best.sort_values("rE_val_RMSE").reset_index(drop=True), conf


# ---------- Confirmação ----------
def confirmacao_por_combinacao(df: pd.DataFrame) -> list[dict]:
    """Uma entrada por combinação com as médias ± dp, vitórias por semente, comparação com a melhor ingênua e o veredito (§6)."""
    d = so_treinados(df)
    if d.empty:
        return []
    ing = df[df["variante"].isin(VAR_ING)]
    out = []
    sem_esp = PROTOCOLO["sementes_confirmacao"]
    for chave, g in d.groupby(CH_COMB, sort=True):
        dataset, modo, backbone, largura, fracao = chave
        nfhs = sorted(v for v in g["variante"].unique() if eh_nfh(v))
        reg = {"dataset": dataset, "modo": modo, "backbone": backbone, "largura": int(largura), "fracao": float(fracao),
               "conjunto": f"{dataset}-{modo}", "id": f"{dataset}|{modo}|{backbone}|w{int(largura)}|f{float(fracao):g}"}
        if not nfhs:
            reg.update({"veredito": "INCOMPLETA", "motivo": "sem variante NFH no CSV", "variante_nfh": None,
                        "n_sementes": 0})
            out.append(reg)
            continue
        v = nfhs[0]
        reg["variante_nfh"] = v
        reg["varias_variantes_nfh"] = nfhs if len(nfhs) > 1 else None
        piv = {}
        reg["sementes_fora_do_protocolo"] = sorted(int(x) for x in g["seed"].unique() if int(x) not in sem_esp) or None
        for nome, var in (("densa", "densa"), ("densaE", "densa+E"), ("proj", "proj_K10"), ("nfh", v)):
            # só as sementes exatas de selecionar.SEEDS_CONF entram nas médias (item 5 da revisão)
            x = g[(g["variante"] == var) & (g["seed"].isin(sem_esp))].drop_duplicates("seed").set_index("seed")
            piv[nome] = x
        sem = sorted(set(piv["densa"].index) & set(piv["densaE"].index) & set(piv["nfh"].index))
        reg["sementes"] = [int(s) for s in sem]
        reg["n_sementes"] = len(sem)
        # densa+E com época escolhida 0 é o mesmo modelo da densa (as 15 épocas extras não melhoraram a validação)
        e0 = piv["densaE"].reindex(sem)
        z = e0["epoca_escolhida"] == 0
        reg["densaE_epoca0_sementes"] = [int(s) for s in e0.index[z.values]]
        reg["densaE_epoca0_n"] = int(z.sum())
        reg["densaE_epoca0_rmse_igual_densa"] = int(sum(
            np.isclose(e0.loc[s, "test_RMSE"], piv["densa"].loc[s, "test_RMSE"], rtol=0, atol=1e-9) for s in e0.index[z.values]))
        for nome in piv:
            for met in ("RMSE", "MAE"):
                col = f"test_{met}"
                x = piv[nome].reindex(sem if nome != "proj" else sorted(set(sem) & set(piv["proj"].index)))[col].dropna()
                reg[f"{nome}_{met}_media"] = float(x.mean()) if len(x) else None
                reg[f"{nome}_{met}_dp"] = float(x.std(ddof=1)) if len(x) > 1 else None
                reg[f"{nome}_{met}_n"] = int(len(x))
        if sem:
            nf_r, e_r = piv["nfh"].loc[sem, "test_RMSE"], piv["densaE"].loc[sem, "test_RMSE"]
            nf_m, e_m = piv["nfh"].loc[sem, "test_MAE"], piv["densaE"].loc[sem, "test_MAE"]
            d_r, d_m = piv["densa"].loc[sem, "test_RMSE"], piv["densa"].loc[sem, "test_MAE"]
            reg["vitorias_RMSE_vs_densaE"] = int((nf_r < e_r).sum())
            reg["vitorias_MAE_vs_densaE"] = int((nf_m < e_m).sum())
            reg["vitorias_RMSE_vs_densa"] = int((nf_r < d_r).sum())
            reg["vitorias_MAE_vs_densa"] = int((nf_m < d_m).sum())
            reg["rel_RMSE_por_semente_vs_densaE"] = {int(s): float(nf_r[s] / e_r[s] - 1) for s in sem}
            reg["rel_RMSE_media_vs_densaE"] = reg["nfh_RMSE_media"] / reg["densaE_RMSE_media"] - 1
            reg["rel_MAE_media_vs_densaE"] = reg["nfh_MAE_media"] / reg["densaE_MAE_media"] - 1
            reg["rel_RMSE_media_vs_densa"] = reg["nfh_RMSE_media"] / reg["densa_RMSE_media"] - 1
            reg["rel_MAE_media_vs_densa"] = reg["nfh_MAE_media"] / reg["densa_MAE_media"] - 1
            reg["cobertura_mediana_nfh_media"] = float(piv["nfh"].loc[sem, "cobertura_mediana"].mean())
            reg["val_rel_RMSE_media_vs_densaE"] = float((piv["nfh"].loc[sem, "val_RMSE"] / piv["densaE"].loc[sem, "val_RMSE"] - 1).mean())
            c_ = {"medias_RMSE_ok_densa": reg["nfh_RMSE_media"] <= reg["densa_RMSE_media"],
                  "medias_RMSE_ok_densaE": reg["nfh_RMSE_media"] <= reg["densaE_RMSE_media"],
                  "medias_MAE_ok_densa": reg["nfh_MAE_media"] <= reg["densa_MAE_media"],
                  "medias_MAE_ok_densaE": reg["nfh_MAE_media"] <= reg["densaE_MAE_media"],
                  "vitorias_ok": reg["vitorias_RMSE_vs_densaE"] >= PROTOCOLO["minimo_vitorias_rmse_vs_densaE"]}
            c_ = {k: bool(x) for k, x in c_.items()}
            reg["criterios"] = c_
            reg["medias_ok"] = all(c_[k] for k in c_ if k.startswith("medias"))
            if sorted(sem) != sorted(sem_esp):
                reg["veredito"] = "INCOMPLETA"
                reg["motivo"] = f"só {len(sem)} das {len(sem_esp)} sementes do protocolo concluídas"
            elif reg["medias_ok"] and c_["vitorias_ok"]:
                reg["veredito"] = "SUCESSO"
                reg["motivo"] = ""
            else:
                reg["veredito"] = "FALHA"
                falhas = []
                if not c_["medias_RMSE_ok_densa"]:
                    falhas.append("média de RMSE da NFH maior que a da densa")
                if not c_["medias_RMSE_ok_densaE"]:
                    falhas.append("média de RMSE da NFH maior que a da densa+E")
                if not c_["medias_MAE_ok_densa"]:
                    falhas.append("média de MAE da NFH maior que a da densa")
                if not c_["medias_MAE_ok_densaE"]:
                    falhas.append("média de MAE da NFH maior que a da densa+E")
                if not c_["vitorias_ok"]:
                    falhas.append("vitórias de RMSE contra a densa+E abaixo do mínimo")
                reg["motivo"] = "; ".join(falhas)
                # derrota = a média de RMSE da NFH perde para a densa ou para a densa+E;
                # falha pelo MAE = RMSE médio a favor, mas a média de MAE perde para a densa ou para a densa+E;
                # falha pelas vitórias = médias a favor, mas menos vitórias por semente que o mínimo.
                # (22/11/2026, auditoria: o rótulo antigo "empate" escondia uma derrota no MAE.)
                if not (c_["medias_RMSE_ok_densa"] and c_["medias_RMSE_ok_densaE"]):
                    reg["tipo_falha"] = "derrota"
                elif not (c_["medias_MAE_ok_densa"] and c_["medias_MAE_ok_densaE"]):
                    reg["tipo_falha"] = "falha pelo MAE"
                else:
                    reg["tipo_falha"] = "falha pelas vitórias"
        else:
            reg["veredito"] = "INCOMPLETA"
            reg["motivo"] = "nenhuma semente com densa, densa+E e NFH"
        # melhor ingênua
        ig = ing[(ing["dataset"] == dataset) & (ing["modo"] == modo)]
        if len(ig):
            r_i = ig.sort_values("test_RMSE").iloc[0]
            m_i = ig.sort_values("test_MAE").iloc[0]
            reg["ingenua_RMSE"] = {"variante": r_i["variante"], "valor": float(r_i["test_RMSE"])}
            reg["ingenua_MAE"] = {"variante": m_i["variante"], "valor": float(m_i["test_MAE"])}
            u = ig[ig["variante"] == "ingenua_ultimo"]
            if len(u):
                reg["ingenua_ultimo_RMSE"] = float(u.iloc[0]["test_RMSE"])
                reg["ingenua_ultimo_MAE"] = float(u.iloc[0]["test_MAE"])
                for met in ("RMSE", "MAE"):
                    vals = [reg.get(f"{k}_{met}_media") for k in ("densa", "densaE", "proj", "nfh")]
                    vals = [x for x in vals if x is not None and np.isfinite(x)]
                    reg[f"todos_acima_ultimo_{met}"] = bool(vals) and all(x > reg[f"ingenua_ultimo_{met}"] for x in vals)
            if reg.get("nfh_RMSE_media") is not None:
                reg["nfh_vs_ingenua_RMSE"] = reg["nfh_RMSE_media"] / reg["ingenua_RMSE"]["valor"] - 1
                reg["nfh_vs_ingenua_MAE"] = reg["nfh_MAE_media"] / reg["ingenua_MAE"]["valor"] - 1
                reg["densa_vs_ingenua_RMSE"] = reg["densa_RMSE_media"] / reg["ingenua_RMSE"]["valor"] - 1
        out.append(reg)
    return out


def exemplo_h8(pasta: Path) -> dict | None:
    """Exemplo passo a passo (H = 8, outro protocolo): recalcula as médias do confirmacao.csv e lê o veredito do json."""
    csv_ = pasta / "confirmacao.csv"
    js = ler_json(pasta / "confirmacao.json")
    if not csv_.exists() or csv_.stat().st_size == 0:
        return None
    df = pd.read_csv(csv_)
    reg = {"arquivo": str(csv_), "p": int(df["p"].iloc[0]), "n": int(df["n"].iloc[0]),
           "variante_nfh": (js or {}).get("variante_nfh"), "veredito_json": (js or {}).get("veredito")}
    sem = sorted(set(df[df["variante"] == "NFH"]["semente"]) & set(df[df["variante"] == "densa"]["semente"])
                 & set(df[df["variante"] == "densa+E"]["semente"]))
    reg["n_sementes"] = len(sem)
    reg["sementes"] = [int(s) for s in sem]
    pv = {v: df[df["variante"] == v].set_index("semente").loc[sem] for v in ("densa", "densa+E", "NFH")}
    for nome, v in (("densa", "densa"), ("densaE", "densa+E"), ("nfh", "NFH")):
        for met in ("RMSE", "MAE"):
            x = pv[v][f"test_{met}"]
            reg[f"{nome}_{met}_media"] = float(x.mean())
            reg[f"{nome}_{met}_dp"] = float(x.std(ddof=1))
    reg["vitorias_RMSE_vs_densaE"] = int((pv["NFH"]["test_RMSE"] < pv["densa+E"]["test_RMSE"]).sum())
    reg["vitorias_MAE_vs_densaE"] = int((pv["NFH"]["test_MAE"] < pv["densa+E"]["test_MAE"]).sum())
    reg["vitorias_RMSE_vs_densa"] = int((pv["NFH"]["test_RMSE"] < pv["densa"]["test_RMSE"]).sum())
    reg["vitorias_MAE_vs_densa"] = int((pv["NFH"]["test_MAE"] < pv["densa"]["test_MAE"]).sum())
    medias_ok = all(reg[f"nfh_{m}_media"] <= reg[f"{r}_{m}_media"] for m in ("RMSE", "MAE") for r in ("densa", "densaE"))
    reg["medias_ok"] = bool(medias_ok)
    reg["veredito_recalculado"] = ("SUCESSO" if medias_ok and len(sem) >= PROTOCOLO["exemplo_sementes"]
                                   and reg["vitorias_RMSE_vs_densaE"] >= PROTOCOLO["exemplo_minimo_vitorias"] else "FALHA")
    reg["concorda_com_json"] = (reg["veredito_recalculado"] == reg["veredito_json"]) if js else None
    # SUCESSO só se o recálculo e (quando existe) o confirmacao.json dizem SUCESSO
    reg["veredito_final"] = "SUCESSO" if (reg["veredito_recalculado"] == "SUCESSO" and (not js or reg["veredito_json"] == "SUCESSO")) else (
        reg["veredito_recalculado"] if reg["veredito_recalculado"] != "SUCESSO" else str(reg["veredito_json"]))
    return reg


# ---------------------------------------------------------------------------
# Figuras (tons de cinza, sem título)
# ---------------------------------------------------------------------------
def figuras(best_t1, conf, saida: Path) -> dict:
    out = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return out
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#222222", "axes.labelcolor": "#111111"})
    fdir = saida / "figuras"
    fdir.mkdir(parents=True, exist_ok=True)
    if best_t1 is not None and len(best_t1):
        bbs = sorted(best_t1["backbone"].unique())
        dados_t = [100 * best_t1[best_t1["backbone"] == b]["rE_test_RMSE"].dropna().to_numpy() for b in bbs]
        dados_v = [100 * best_t1[best_t1["backbone"] == b]["rE_val_RMSE"].dropna().to_numpy() for b in bbs]
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        pos = np.arange(len(bbs))
        b1 = ax.boxplot(dados_v, positions=pos - 0.18, widths=0.3, patch_artist=True, showfliers=False)
        b2 = ax.boxplot(dados_t, positions=pos + 0.18, widths=0.3, patch_artist=True, showfliers=False)
        for p in b1["boxes"]:
            p.set(facecolor="#bdbdbd", edgecolor="#222222")
        for p in b2["boxes"]:
            p.set(facecolor="#555555", edgecolor="#222222")
        for k in ("medians",):
            for b_ in (b1, b2):
                for p in b_[k]:
                    p.set(color="#000000")
        ax.axhline(0, color="#000000", lw=0.8, ls="--")
        ax.set_xticks(pos)
        ax.set_xticklabels(bbs, rotation=30, ha="right")
        ax.set_ylabel("RMSE relativo à densa+E (%)")
        ax.legend([b1["boxes"][0], b2["boxes"][0]], ["validação", "teste"], loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(fdir / "t1_por_backbone.png", dpi=220)
        plt.close(fig)
        out["t1_por_backbone"] = "figuras/t1_por_backbone.png"
    cs = [c for c in conf if c.get("rel_RMSE_por_semente_vs_densaE")]
    if cs:
        fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(cs) + 1.4))
        for i, c in enumerate(cs):
            v = 100 * np.array(list(c["rel_RMSE_por_semente_vs_densaE"].values()))
            ax.scatter(v, np.full(len(v), i), s=18, color="#777777", zorder=3)
            ax.scatter([100 * c["rel_RMSE_media_vs_densaE"]], [i], s=60, marker="D", color="#000000", zorder=4)
        ax.axvline(0, color="#000000", lw=0.8, ls="--")
        ax.set_yticks(range(len(cs)))
        ax.set_yticklabels([f"{c['conjunto']} {c['backbone']} f={c['fracao']:g} ({c['veredito']})" for c in cs], fontsize=8)
        ax.set_xlabel("RMSE de teste da NFH relativo à densa+E (%); losango = média, ponto = semente")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(fdir / "confirmacao_por_semente.png", dpi=220)
        plt.close(fig)
        out["confirmacao"] = "figuras/confirmacao_por_semente.png"
    return out


# ---------------------------------------------------------------------------
# Montagem
# ---------------------------------------------------------------------------
def aus(txt):
    return r"\textit{" + esc(txt) + "}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", default=str(B / "resultados"), help="pasta com t1.csv, t2.csv, confirmacao.csv, filas/, selecao_*.md")
    ap.add_argument("--saida", default=str(B / "relatorio"))
    ap.add_argument("--exemplo", default=str(B / "exemplo_sintetico"))
    ap.add_argument("--sem-pdf", action="store_true")
    a = ap.parse_args()
    RES = Path(a.res)
    SAI = Path(a.saida)
    SAI.mkdir(parents=True, exist_ok=True)
    N: dict = {"protocolo": PROTOCOLO, "res": str(RES), "arquivos": {}}

    t1, i1 = ler_csv(RES / "t1.csv")
    t2, i2 = ler_csv(RES / "t2.csv")
    cf, i3 = ler_csv(RES / "confirmacao.csv")
    N["arquivos"] = {"t1": i1, "t2": i2, "confirmacao": i3}
    md1 = parse_selecao_md(RES / "selecao_T1.md")
    md2 = parse_selecao_md(RES / "selecao_T2.md")
    N["selecao_T1_md"] = {"existe": md1["existe"], **md1["contagens"]}
    N["selecao_T2_md"] = {"existe": md2["existe"], **md2["contagens"]}
    filas = {}
    for nome in ("t1_B2", "t1", "t2", "confirmacao"):
        j = ler_json(RES / "filas" / f"{nome}.json")
        if j:
            tar = j.get("tarefas", [])
            filas[nome] = {"tarefas": len(tar), "tarefas_treino": sum(1 for t in tar if t.get("tipo") != "ingenua"),
                           "combinacoes": len({(t["dataset"], t["modo"], t["backbone"], t["fracao"], t.get("largura"))
                                               for t in tar if t.get("tipo") != "ingenua"})}
    N["filas"] = filas
    itens10 = ler_secao10()

    # ---- T1 ----
    m1 = relativas(t1)
    best1 = melhor_t1(m1)
    best1, conf_md1 = cruzar_t1_md(best1, md1)
    N["conferencia_T1_md"] = conf_md1
    t1_rodou = len(best1) > 0
    N["t1"] = {"rodou": bool(t1_rodou)}
    if t1_rodou:
        qual = best1["passa"].notna().any()
        N["t1"].update({
            "combinacoes": int(len(best1)),
            "backbones": int(best1["backbone"].nunique()), "conjuntos": int(best1.groupby(["dataset", "modo"]).ngroups),
            "nomes_conjunto": int(best1["dataset"].nunique()),
            "nomes_com_dois_modos": sorted(g for g, k in best1.groupby("dataset")["modo"].nunique().items() if k > 1),
            "vence_val_RMSE": int((best1["rE_val_RMSE"] < 0).sum()),
            "vence_test_RMSE": int((best1["rE_test_RMSE"] < 0).sum()),
            "vence_val_ambos": int(((best1["rE_val_RMSE"] < 0) & (best1["rE_val_MAE"] < 0)).sum()),
            "vence_test_ambos": int(((best1["rE_test_RMSE"] < 0) & (best1["rE_test_MAE"] < 0)).sum()),
            "qualificaram": int(best1["passa"].fillna(False).astype(bool).sum()) if qual else None,
            "foram_T2": int(best1["vai_T2"].fillna(False).astype(bool).sum()) if qual else None,
            "ganhou_val_e_perdeu_teste": int(((best1["rE_val_RMSE"] < 0) & (best1["rE_test_RMSE"] >= 0)).sum()),
            "mediana_rel_test_RMSE_vs_densaE": float(best1["rE_test_RMSE"].median()),
            "mediana_rel_test_RMSE_vs_densa": float(best1["rD_test_RMSE"].median()),
            "tarefas_treino_csv": int(so_treinados(t1).groupby(CH).ngroups),
            "linhas_ingenuas": int((t1["backbone"] == "").sum()),
        })
        N["t1"]["estat_por_backbone"] = estat_grupo(best1, "backbone")
        N["t1"]["estat_por_conjunto_fracao"] = estat_grupo(best1.assign(cf=best1.apply(lambda r: f"{r['conjunto']} f={r['fracao']:g}", axis=1)), "cf")
        N["t1"]["estat_por_conjunto"] = estat_grupo(best1, "conjunto")
        vs = variantes_t1(m1, best1)
        N["t1"]["variantes"] = vs
        mm1 = m1[(m1["seed"] == 0) & (m1["largura"] == 1)]
        nv = mm1.groupby(["dataset", "modo", "backbone", "fracao"])["variante"].nunique()
        N["t1"]["n_variantes_nfh_total"] = int(len(vs))
        N["t1"]["n_variantes_nfh_por_combinacao_min"] = int(nv.min()) if len(nv) else 0
        N["t1"]["n_variantes_nfh_por_combinacao_max"] = int(nv.max()) if len(nv) else 0
        N["t1"]["variantes_agregadas"] = agrega_variantes(vs)

    # ---- T2 ----
    best2, conf_md2 = resumo_t2(t2, md2)
    N["conferencia_T2_md"] = conf_md2
    t2_rodou = len(best2) > 0
    N["t2"] = {"rodou": bool(t2_rodou)}
    if t2_rodou:
        q2 = best2["passa"].notna().any()
        N["t2"].update({
            "combinacoes_largura": int(len(best2)),
            "combinacoes": int(best2.groupby(["dataset", "modo", "backbone", "fracao"]).ngroups),
            "vence_val_RMSE": int((best2["rE_val_RMSE"] < 0).sum()),
            "vence_test_RMSE": int((best2["rE_test_RMSE"] < 0).sum()),
            "vence_test_ambos": int(((best2["rE_test_RMSE"] < 0) & (best2["rE_test_MAE"] < 0)).sum()),
            "qualificaram": int(best2["passa"].fillna(False).astype(bool).sum()) if q2 else int(best2["passa_calc"].sum()),
            "qualificaram_fonte": "selecao_T2.md" if q2 else "cálculo próprio (selecao_T2.md ausente)",
            "foram_confirmacao": int(best2["vai_conf"].fillna(False).astype(bool).sum()) if q2 else None,
            "tarefas_treino_csv": int(so_treinados(t2).groupby(CH).ngroups),
        })
        _q = best2["passa"].fillna(best2["passa_calc"]).astype(bool)
        _v = best2["vai_conf"].fillna(False).astype(bool)
        _perd = (best2["rE_test_RMSE"] > 0) | (best2["rE_test_MAE"] > 0)
        _sel_ = best2[_q & ~_v & _perd]
        N["t2"]["qualificou_sem_confirmacao_perdeu_teste"] = int(len(_sel_))
        N["t2"]["qualificou_sem_confirmacao_perdeu_teste_pares"] = [
            f"{r['conjunto']}|{r['backbone']}|f{r['fracao']:g}|w{int(r['largura'])}" for _, r in _sel_.iterrows()]
        N["t2"]["qualificou_sem_confirmacao"] = int((_q & ~_v).sum())

    # ---- Confirmação ----
    conf = confirmacao_por_combinacao(cf)
    cf_rodou = len(conf) > 0
    N["confirmacao"] = {"rodou": bool(cf_rodou), "combinacoes": len(conf)}
    avisos_conf = []
    for c in conf:
        if c.get("varias_variantes_nfh"):
            avisos_conf.append(f"{c['id']}: mais de uma variante NFH na confirmação ({', '.join(c['varias_variantes_nfh'])}); "
                               f"o veredito usa só {c['variante_nfh']} (a primeira em ordem alfabética)")
        if c.get("sementes_fora_do_protocolo"):
            avisos_conf.append(f"{c['id']}: sementes fora de selecionar.SEEDS_CONF ignoradas ({c['sementes_fora_do_protocolo']})")
    N["confirmacao"]["avisos"] = avisos_conf
    for a_ in avisos_conf:
        print("AVISO:", a_, file=sys.stderr)
    if cf_rodou:
        N["confirmacao"].update({"sucesso": sum(c["veredito"] == "SUCESSO" for c in conf),
                                 "falha": sum(c["veredito"] == "FALHA" for c in conf),
                                 "incompleta": sum(c["veredito"] == "INCOMPLETA" for c in conf),
                                 "falha_derrota": sum(c.get("tipo_falha") == "derrota" for c in conf),
                                 # chave antiga mantida: falhas sem derrota no RMSE médio (antes chamadas de "empate")
                                 "falha_empate": sum(c.get("tipo_falha") in ("falha pelo MAE", "falha pelas vitórias") for c in conf),
                                 "falha_pelo_mae": sum(c.get("tipo_falha") == "falha pelo MAE" for c in conf),
                                 "falha_pelas_vitorias": sum(c.get("tipo_falha") == "falha pelas vitórias" for c in conf),
                                 "combinacoes_detalhe": conf})
    exemplo = exemplo_h8(Path(a.exemplo))
    N["exemplo_h8"] = exemplo
    # explicações
    # Só entram como "explicações de SUCESSO" os JSON cuja combinação tem SUCESSO no veredito RECALCULADO de confirmacao.csv.
    # Os demais (gerados com --lista, ou de uma confirmação anterior) vão para a subseção de controle, com o veredito impresso;
    # JSON sem combinação em confirmacao.csv ficam como "sem_veredito" e não entram em nenhuma lista de SUCESSO.
    ver_conf = {c["id"]: c["veredito"] for c in conf}
    expl, expl_ctrl, expl_erro, expl_desc = [], [], [], []
    for p in sorted((RES / "explicacoes").glob("*.json")) if (RES / "explicacoes").exists() else []:
        j = ler_json(p)
        if not j:
            continue
        if "erro" in j:
            expl_erro.append({"combinacao": j.get("combinacao"), "erro": j.get("erro")})
            continue
        v = ver_conf.get(j.get("id"))
        j["veredito_recalculado"] = v or "SEM_VEREDITO"
        if v is not None and j.get("veredito_confirmacao") not in (None, v):
            expl_desc.append({"combinacao": j.get("combinacao"), "json": j.get("veredito_confirmacao"), "recalculado": v})
        (expl if v == "SUCESSO" else expl_ctrl).append(j)
    N["explicacoes"] = {"arquivos": len(expl) + len(expl_ctrl), "combinacoes": [e.get("combinacao") for e in expl],
                        "controle_sem_sucesso": [{"combinacao": e.get("combinacao"), "veredito": e["veredito_recalculado"]} for e in expl_ctrl],
                        "veredito_json_desatualizado": expl_desc, "com_erro": expl_erro}

    figs = figuras(best1 if t1_rodou else None, conf, SAI)
    N["figuras"] = figs

    # ================== TEXTO ==================
    S: list[str] = []
    P = S.append
    P(r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[brazilian]{babel}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,longtable,array}
\usepackage{graphicx}
\usepackage[a4paper,margin=1.8cm]{geometry}
\usepackage[hidelinks]{hyperref}
\usepackage[font=small,labelfont=bf,skip=4pt]{caption}
\setlength{\emergencystretch}{3em}\setlength{\parindent}{1.2em}\setlength{\parskip}{3pt}
\title{Backbones para a Neural Fourier Head: relat\'orio do experimento de 21/11/2026}
\author{Gerado por \texttt{codigo/relatorio.py} a partir dos CSV de \texttt{resultados/}}
\date{}
\begin{document}
\maketitle
""")
    # --- aviso e visão geral
    P(r"\section*{Como ler este relatório}")
    P("Este relatório foi gerado por código; nenhum número do texto foi digitado à mão (todos estão em "
      r"\texttt{numeros.json}). Os casos de vitória da NFH foram \emph{procurados} (exemplificação): eles mostram onde a "
      "NFH iguala ou supera a camada densa, e não valem como regra geral. Todas as combinações testadas e todas as derrotas "
      "aparecem aqui. Nas tabelas, ``densa+E'' é a camada densa treinada por mais épocas, o controle das épocas extras "
      "que a NFH recebe no ajuste. Diferenças relativas são $\\text{NFH}/\\text{referência}-1$: valores negativos favorecem a NFH.\n")
    ex = N["arquivos"]
    for nome, inf in (("T1", i1), ("T2", i2), ("confirmação", i3)):
        pass
    P(f"Arquivos lidos: \\texttt{{t1.csv}} ({i1['linhas']} linhas válidas), \\texttt{{t2.csv}} ({i2['linhas']}) e "
      f"\\texttt{{confirmacao.csv}} ({i3['linhas']}). Linhas de cabeçalho repetidas descartadas: {i1['cabecalhos_repetidos']}, "
      f"{i2['cabecalhos_repetidos']} e {i3['cabecalhos_repetidos']}. Linhas duplicadas (mesma tarefa e variante) descartadas: "
      f"{i1['duplicadas_descartadas']}, {i2['duplicadas_descartadas']} e {i3['duplicadas_descartadas']}.\n")
    faltam = [n for n, f in (("T1", t1_rodou), ("T2", t2_rodou), ("confirmação", cf_rodou)) if not f]
    if faltam:
        P(r"\textbf{Fases sem dados neste relatório:} " + ", ".join(faltam) +
          ". Nessas fases o relatório diz que a fase não rodou (ou não deixou resultados) e os resumos que dependem delas ficam vazios.\n")

    # --- Resumo dos resultados (tudo calculado; nenhum número digitado)
    P(r"\section*{Resumo dos resultados}")
    if t1_rodou and t2_rodou and cf_rodou:
        n1, n2, nc = N["t1"], N["t2"], N["confirmacao"]
        qual1 = "n/d" if n1["qualificaram"] is None else str(n1["qualificaram"])
        P(f"Funil: {n1['combinacoes']} combinações na T1, {qual1} qualificadas pela validação, {n1['foram_T2'] if n1['foram_T2'] is not None else 'n/d'} "
          f"na T2 ({n2['combinacoes_largura']} pares combinação e largura), {n2['qualificaram']} pares qualificados, "
          f"{n2['foram_confirmacao'] if n2['foram_confirmacao'] is not None else 'n/d'} na confirmação, "
          f"{nc['sucesso']} com SUCESSO e {nc['falha']} com FALHA.\n")
    suc_r = [c for c in conf if c["veredito"] == "SUCESSO"]
    fal_r = [c for c in conf if c["veredito"] == "FALHA"]
    def idc(c):
        nome = c["dataset"] if c["dataset"].startswith("SINT") else c["conjunto"]
        return f"{esc(nome)}/{esc(c['backbone'])} (largura {c['largura']})"
    if suc_r:
        sint_r = [c for c in suc_r if c["dataset"].startswith("SINT")]
        real_r = [c for c in suc_r if not c["dataset"].startswith("SINT")]
        frs = sorted({c["fracao"] for c in suc_r})
        ser_r = sorted({c["conjunto"] for c in real_r})
        P(f"Dos {len(suc_r)} SUCESSO, {len(sint_r)} são sintéticos ({lista_pt(idc(c) for c in sint_r) or 'nenhum'}) e {len(real_r)} são de dados reais "
          f"({lista_pt(idc(c) for c in real_r) or 'nenhum'}), de {len(ser_r)} {'série real' if len(ser_r) == 1 else 'séries reais'}: {lista_pt(esc(x) for x in ser_r) or 'nenhuma'}. "
          f"Todos usam fração de treino de {dec(frs[0])} a {dec(frs[-1])}"
          + (" (nenhum usa o treino inteiro)" if frs[-1] < 1 else "") + ". "
          "Eles foram procurados entre as combinações da T1 e valem só para esse recorte: não mostram que a NFH iguala a densa "
          "em dados reais em geral nem em \\emph{backbones} em geral.\n")
    if fal_r:
        P(f"As {len(fal_r)} FALHAS são: " + "; ".join(
            f"{idc(c)}, tipo ``{esc(c.get('tipo_falha', 'n/d'))}'' (RMSE {pc(c.get('rel_RMSE_media_vs_densaE'))} e MAE "
            f"{pc(c.get('rel_MAE_media_vs_densaE'))} contra a densa+E)" for c in fal_r) + ".")
        ult_r = [c for c in fal_r if c.get("todos_acima_ultimo_RMSE")]
        if ult_r:
            P(" Em " + lista_pt(idc(c) for c in ult_r) + ", todos os modelos (densa, densa+E, \\texttt{proj\\_K10} e NFH) ficam acima da ingênua do "
              "último valor no RMSE médio de teste.")
        P("\n")
    if t1_rodou:
        n1 = N["t1"]
        P(f"Na T1, a NFH vence a densa+E no teste nas duas métricas (RMSE e MAE) em {n1['vence_test_ambos']} de {n1['combinacoes']} combinações, "
          f"e a mediana da diferença relativa de RMSE de teste da melhor variante é {pc(n1['mediana_rel_test_RMSE_vs_densaE'])} (zero na precisão mostrada) "
          f"contra a densa+E. Em {n1['ganhou_val_e_perdeu_teste']} combinações a NFH ganhou na validação e perdeu no teste.\n")
    ex_sm = [e for e in expl + expl_ctrl if e.get("verdade_sint_m")]
    if ex_sm:
        P("\\textbf{Explicabilidade na SINT-M da varredura (resultado negativo).} Aqui o $y$ passado entra na janela ($d=5$; na dissertação $d=4$, sem $y$). "
          "``Acerto'' é a fração das janelas de teste em que a variável de maior atribuição nos Integrated Gradients é a condutora conhecida. ")
        for e in ex_sm:
            v = e["verdade_sint_m"]
            P(f"Em {esc(e['combinacao'], True)}, o acerto com as 5 colunas é {nf(v['tendencia']['acerto_5_colunas'], 2)} para a tendência e "
              f"{nf(v['ciclo']['acerto_5_colunas'], 2)} para o ciclo (participação do $y$ passado na atribuição: {pcn(v['tendencia']['participacao_y_passado'], 2)} e "
              f"{pcn(v['ciclo']['participacao_y_passado'], 2)}); só com as 4 covariáveis, {nf(v['tendencia']['acerto_4_covariaveis'], 2)} e "
              f"{nf(v['ciclo']['acerto_4_covariaveis'], 2)}. ")
        P("A atribuição não recupera a verdade conhecida quando o alvo passado está na entrada. Isso limita a alegação de explicação conferida nesta varredura; "
          "o resultado positivo da dissertação é de outra configuração.\n")
    if t2_rodou and N["t2"].get("qualificou_sem_confirmacao_perdeu_teste") is not None:
        P(f"Na T2, {N['t2']['qualificou_sem_confirmacao_perdeu_teste']} pares qualificaram, não foram à confirmação e tiveram RMSE ou MAE de teste pior "
          "que o da densa+E; eles estão na Seção~\\ref{sec:naoganhou}.\n")

    # --- 1. protocolo
    P(r"\section{Protocolo em uma página}")
    P(f"O protocolo foi escrito em 21/11/2026, antes de qualquer execução (\\texttt{{INSTRUCOES.md}}). A pergunta é em que "
      f"\\emph{{backbones}} e conjuntos a NFH, no lugar da camada densa final, tem RMSE \\textbf{{e}} MAE de teste iguais ou menores "
      f"que os da densa. Todos os modelos têm a forma \\texttt{{InstanceNorm(Backbone + cabeça)}}, com janela $L={PROTOCOLO['L']}$, "
      f"horizonte $H={PROTOCOLO['H']}$, $q={PROTOCOLO['q']}$, divisão cronológica {divisao()} e a mesma semente, o mesmo "
      f"otimizador (Adam, taxa {pot10(PROTOCOLO['taxa_aprendizado'])}) e os mesmos dados para a densa e a NFH; a cabeça NFH tem os mesmos $H(p+1)$ parâmetros da densa.\n")
    P(r"\textbf{Seleção só na validação.} Em cada etapa, a combinação e a variante NFH candidata são escolhidas pelo RMSE de validação "
      "relativo à densa+E (e o MAE de validação relativo precisa ser menor ou igual a zero); \\texttt{selecionar.py} nem carrega "
      "as colunas de teste. O teste só entra na confirmação e neste relatório.\n")
    P(f"\\textbf{{Controle densa+E.}} A densa é treinada por mais {PROTOCOLO['epocas_extra']} épocas com taxa {pot10(PROTOCOLO['taxa_ajuste_densaE'])}, sem penalização, escolhendo a época de "
      "menor RMSE de validação. A NFH é comparada com ela e com a densa simples; a densa+E tem, por construção, RMSE de validação "
      "menor ou igual ao da densa.\n")
    P(r"\textbf{Etapas.} T1 (triagem: todos os \emph{backbones}, largura 1, semente 0), T2 (as combinações selecionadas, sementes "
      f"{', '.join(map(str, PROTOCOLO['sementes_t2']))}, larguras 1 e 2) e confirmação no teste com \\textbf{{sementes novas}} "
      f"({PROTOCOLO['sementes_confirmacao'][0]} a {PROTOCOLO['sementes_confirmacao'][-1]}), com a densa, a densa+E, a projeção "
      f"\\texttt{{proj\\_K10}} e a variante NFH escolhida.\n")
    P(r"\textbf{Critério de sucesso (§6).} Média do RMSE de teste \emph{e} média do MAE de teste da NFH menores ou iguais às da densa "
      f"\\emph{{e}} às da densa+E, e RMSE de teste da NFH menor que o da densa+E em pelo menos {PROTOCOLO['minimo_vitorias_rmse_vs_densaE']} "
      f"das {len(PROTOCOLO['sementes_confirmacao'])} sementes. Só as combinações com SUCESSO são citadas como exemplos em que a NFH "
      "iguala ou supera a densa; as demais são derrotas ou falhas do critério (pelo MAE ou pelas vitórias).\n")
    if itens10:
        P(r"\textbf{Mudanças registradas no §10} (texto integral no Apêndice~\ref{ap:s10}):")
        P(r"\begin{itemize}\setlength{\itemsep}{0pt}")
        for it in itens10:
            m = re.match(r"^(\d{2}/\d{2}/\d{4}(?:, \d+h\d*)?)(?: \(([^)]*)\))?[.,]?\s*(.*)$", it, flags=re.S)
            if m:
                quando, quem, resto = m.group(1), m.group(2), m.group(3)
                resto = re.sub(r"\s+", " ", resto).lstrip(" :.,")
                corte = resto[:330] + ("..." if len(resto) > 330 else "")
                P(r"\item \textbf{" + esc(quando) + r"}" + (f" ({esc(quem)})" if quem else "") + ": " + md2tex(corte))
            else:
                P(r"\item " + md2tex(it[:330]))
        P(r"\end{itemize}")
    else:
        P(aus("O §10 de INSTRUCOES.md não foi encontrado.") + "\n")

    # --- 2. contagens
    P(r"\section{Contagens}")
    if t1_rodou:
        n1 = N["t1"]
        vmin, vmax = n1["n_variantes_nfh_por_combinacao_min"], n1["n_variantes_nfh_por_combinacao_max"]
        nv_txt = str(vmin) if vmin == vmax else f"{vmin} a {vmax}"
        P(f"\\textbf{{T1.}} A T1 testou {n1['combinacoes']} combinações (conjunto, modo, \\emph{{backbone}}, fração) em "
          f"{n1['backbones']} \\emph{{backbones}} e {n1['conjuntos']} conjuntos ({n1['nomes_conjunto']} nomes de conjunto; "
          f"{lista_pt(esc(x) for x in n1['nomes_com_dois_modos'])} entra{'m' if len(n1['nomes_com_dois_modos']) > 1 else ''} nos modos U e M, "
          f"e cada modo conta como um conjunto, o que dá {n1['conjuntos']}). Para cada combinação a melhor variante NFH foi "
          f"escolhida só pelo RMSE de validação. Contra a densa+E, essa variante venceu em RMSE de validação em {n1['vence_val_RMSE']} "
          f"combinações e, separadamente, em RMSE de teste em {n1['vence_test_RMSE']}; venceu em RMSE \\emph{{e}} MAE em "
          f"{n1['vence_val_ambos']} (validação) e {n1['vence_test_ambos']} (teste). Em {n1['ganhou_val_e_perdeu_teste']} "
          f"combinações a NFH ganhou na validação e perdeu no teste (a escolha da melhor entre as variantes NFH da T1, {nv_txt} por combinação, favorece a NFH na "
          "validação, o que explica parte da diferença).")
        if n1["qualificaram"] is not None:
            P(f" Qualificaram pelo critério da validação {n1['qualificaram']} combinações e {n1['foram_T2']} foram para a T2 "
              f"(\\texttt{{selecao\\_T1.md}}).")
        else:
            P(" \\texttt{selecao\\_T1.md} não foi encontrado: não há contagem de qualificadas nem das que foram para a T2.")
        if "t2" in filas:
            P(f" A fila da T2 tem {filas['t2']['tarefas_treino']} tarefas de treino em {filas['t2']['combinacoes']} combinações com largura.")
        P("\n")
    else:
        P(aus("A T1 não rodou (ou t1.csv está ausente/vazio): sem contagens da T1.") + "\n")
    if t2_rodou:
        n2 = N["t2"]
        P(f"\\textbf{{T2.}} A T2 deixou resultados completos (as sementes {', '.join(map(str, PROTOCOLO['sementes_t2']))}) em {n2['combinacoes_largura']} pares (combinação, largura), "
          f"de {n2['combinacoes']} combinações. A melhor variante (pela média do RMSE de validação relativo) venceu a densa+E no RMSE de "
          f"validação em {n2['vence_val_RMSE']} pares e, separadamente, no RMSE de teste em {n2['vence_test_RMSE']} "
          f"(em RMSE e MAE: {n2['vence_test_ambos']}). Qualificaram {n2['qualificaram']} pares ({esc(n2['qualificaram_fonte'])})")
        if n2["foram_confirmacao"] is not None:
            P(f" e {n2['foram_confirmacao']} foram para a confirmação.")
        else:
            P(".")
        P("\n")
    else:
        P(aus("A T2 não rodou (ou t2.csv está ausente/vazio): sem contagens da T2.") + "\n")
    if cf_rodou:
        nc = N["confirmacao"]
        P(f"\\textbf{{Confirmação.}} {nc['combinacoes']} combinações foram confirmadas com sementes novas: {nc['sucesso']} com SUCESSO e "
          f"{nc['falha']} com FALHA ({nc['falha_derrota']} derrotas no RMSE médio, {nc['falha_pelo_mae']} falhas pelo MAE e "
          f"{nc['falha_pelas_vitorias']} falhas pelas vitórias, definidas na Seção~\\ref{{sec:naoganhou}})")
        if nc["incompleta"]:
            P(f"; {nc['incompleta']} ainda sem as {len(PROTOCOLO['sementes_confirmacao'])} sementes (INCOMPLETA)")
        P(".\n")
    else:
        P(aus("A confirmação não rodou (ou confirmacao.csv está ausente/vazio): sem contagens da confirmação.") + "\n")

    # --- 3. T1
    P(r"\section{Triagem T1}")
    if t1_rodou:
        est_bb = N["t1"]["estat_por_backbone"]
        est_cj = N["t1"]["estat_por_conjunto_fracao"]
        P("A Tabela~\\ref{tab:t1_contagem_bb} dá as contagens por \\emph{backbone} e a Tabela~\\ref{tab:t1_contagem_cj} por conjunto e "
          "fração. As Tabelas~\\ref{tab:t1_bb_val} e \\ref{tab:t1_bb_test} resumem, por \\emph{backbone}, a diferença relativa da melhor "
          "variante NFH (escolhida na validação) contra a densa+E e contra a densa, em RMSE e MAE: mediana e, entre colchetes, "
          "primeiro e terceiro quartis. As Tabelas~\\ref{tab:t1_cj_val} e \\ref{tab:t1_cj_test} fazem o mesmo por conjunto e fração.\n")
        for nome, est, rot in (("t1_contagem_bb", est_bb, "backbone"), ("t1_contagem_cj", est_cj, "conjunto e fração")):
            linhas = [[esc(r["grupo"]), str(r["n"]), str(r["vence_val"]), str(r["vence_test"]), str(r["vence_val_ambos"]),
                       str(r["vence_test_ambos"]), "n/d" if r["qualificaram"] is None else str(r["qualificaram"]),
                       "n/d" if r["foram_T2"] is None else str(r["foram_T2"])] for r in est]
            S.append(tabela(nome, f"T1: combina\\c{{c}}\\~oes testadas e vit\\'orias da NFH sobre a densa+E, por {rot}. "
                            "``Vence'' = RMSE (ou RMSE e MAE) menor que o da densa+E, da melhor variante NFH escolhida na valida\\c{c}\\~ao.",
                            [rot.capitalize(), "Comb.", "RMSE val", "RMSE teste", "RMSE+MAE val", "RMSE+MAE teste", "Qualif.", "Foram T2"],
                            linhas, "lrrrrrrr", SAI, "resultados/t1.csv e selecao\\_T1.md.", longa=len(linhas) > 24))
        for nome, est, tipo, rot in (("t1_bb_val", est_bb, "val", "backbone"), ("t1_bb_test", est_bb, "test", "backbone"),
                                      ("t1_cj_val", est_cj, "val", "conjunto e fração"), ("t1_cj_test", est_cj, "test", "conjunto e fração")):
            nome_sp = "valida\\c{c}\\~ao" if tipo == "val" else "teste"
            S.append(tabela(nome, f"T1, {nome_sp}: diferen\\c{{c}}a relativa da melhor variante NFH (escolhida na valida\\c{{c}}\\~ao), "
                            f"mediana [Q1; Q3], por {rot}.", [rot.capitalize(), "n", "RMSE vs densa+E", "MAE vs densa+E",
                                                              "RMSE vs densa", "MAE vs densa"],
                            tab_estat(est, tipo), "lrllll", SAI, "resultados/t1.csv.", longa=len(est) > 24))
        if "t1_por_backbone" in figs:
            P(r"\begin{figure}[htbp]\centering\caption{T1: RMSE da melhor variante NFH relativo à densa+E, por \emph{backbone}. "
              r"Caixas cinza-claro: validação; caixas cinza-escuro: teste; linha tracejada: igualdade com a densa+E.}\label{fig:t1bb}"
              r"\includegraphics[width=0.92\linewidth]{" + figs["t1_por_backbone"] + r"}\par\smallskip{\scriptsize Fonte: resultados/t1.csv.}\end{figure}")
        # variantes
        vs = N["t1"]["variantes"]
        top = vs[0] if vs else None
        top_esc = max(vs, key=lambda r: r["escolhida_na_validacao"]) if vs else None
        P("\\subsection{Qual variante NFH mais vence}")
        if top:
            P(f"A Tabela~\\ref{{tab:t1_variantes}} conta, para cada variante, em quantas combinações ela venceu a densa+E no RMSE de validação "
              f"e no de teste (cada variante olhada sozinha) e quantas vezes foi a escolhida na validação. A que mais venceu no teste foi "
              f"\\texttt{{{esc(top['variante'])}}}, com {top['vence_test']} vitórias em {top['n']} combinações; a mais escolhida na validação "
              f"foi \\texttt{{{esc(top_esc['variante'])}}} ({top_esc['escolhida_na_validacao']} vezes). A Tabela~\\ref{{tab:t1_variantes_agr}} "
              "agrega por família (\\texttt{conv} = conversão exata da densa e ajuste; \\texttt{zero} = NFH treinada do zero), por $\\lambda$ e por $K$.\n")
        linhas = [[r"\texttt{" + esc(r["variante"]) + "}", str(r["n"]), str(r["vence_val"]), str(r["vence_test"]),
                   pc(r["mediana_rel_test_RMSE"]), str(r["escolhida_na_validacao"])] for r in vs]
        S.append(tabela("t1_variantes", "T1: variantes NFH, vit\\'orias sobre a densa+E (RMSE) e escolhas pela valida\\c{c}\\~ao.",
                        ["Variante", "Comb.", "Vence val", "Vence teste", "Mediana rel. RMSE teste", "Escolhida"], linhas,
                        "lrrrrr", SAI, "resultados/t1.csv (semente 0, largura 1)."))
        linhas = [[esc(r["campo"]), esc(r["valor"]), str(r["n"]), str(r["vence_val"]), str(r["vence_test"]), str(r["escolhida"])]
                  for r in N["t1"]["variantes_agregadas"]]
        S.append(tabela("t1_variantes_agr", "T1: vit\\'orias por fam\\'ilia, $\\lambda$ e $K$ (soma sobre as variantes do grupo).",
                        ["Campo", "Valor", "Pares", "Vence val", "Vence teste", "Escolhida"], linhas, "llrrrr", SAI,
                        "resultados/t1.csv."))
    else:
        P(aus("A T1 não rodou: sem tabelas da T1.") + "\n")

    # --- 4. T2
    P(r"\section{Aprofundamento T2}")
    if t2_rodou:
        P("A Tabela~\\ref{tab:t2_rel} mostra, para cada par (combinação, largura), a melhor variante pela média do RMSE de validação "
          f"relativo à densa+E nas {len(PROTOCOLO['sementes_t2'])} sementes, e as médias das diferenças relativas na validação e no teste. A Tabela~\\ref{{tab:t2_abs}} "
          "traz as médias absolutas de RMSE e MAE. ``Passa'' e ``Conf.'' vêm de \\texttt{selecao\\_T2.md}.\n")
        sn = lambda x: "n/d" if pd.isna(x) else ("sim" if bool(x) else "n\\~ao")
        linhas = [[esc(r["conjunto"]), esc(r["backbone"]), f"{r['fracao']:g}", str(int(r["largura"])),
                   r"\texttt{" + esc(r["variante"], True) + "}", pc(r["rE_val_RMSE"]), pc(r["rE_val_MAE"]), pc(r["pior_sem_val"]),
                   pc(r["rE_test_RMSE"]), pc(r["rE_test_MAE"]), pc(r["rD_test_RMSE"]),
                   sn(r["passa"] if pd.notna(r["passa"]) else np.nan), sn(r["vai_conf"])] for _, r in best2.iterrows()]
        S.append(tabela("t2_rel", f"T2: m\\'edias sobre as sementes {faixa(PROTOCOLO['sementes_t2'])} da diferen\\c{{c}}a relativa da melhor variante NFH "
                        "(contra a densa+E, salvo a \\'ultima coluna de teste).", ["Conjunto", "Backbone", "Fr.", "L", "Variante",
                        "RMSE val", "MAE val", "Pior sem. val", "RMSE teste", "MAE teste", "RMSE teste vs densa", "Passa", "Conf."],
                        linhas, "llrrlrrrrrrll", SAI, "resultados/t2.csv e selecao\\_T2.md.", longa=True, tam=r"\tiny"))
        linhas = [[esc(r["conjunto"]), esc(r["backbone"]), f"{r['fracao']:g}", str(int(r["largura"])),
                   nf(r["E_val_RMSE"]), nf(r["val_RMSE"]), nf(r["D_test_RMSE"]), nf(r["E_test_RMSE"]), nf(r["test_RMSE"]),
                   nf(r["D_test_MAE"]), nf(r["E_test_MAE"]), nf(r["test_MAE"])] for _, r in best2.iterrows()]
        S.append(tabela("t2_abs", f"T2: m\\'edias sobre as {len(PROTOCOLO['sementes_t2'])} sementes de RMSE e MAE (escala padronizada pelo treino).",
                        ["Conjunto", "Backbone", "Fr.", "L", "RMSE val densa+E", "RMSE val NFH", "RMSE teste densa", "RMSE teste densa+E",
                         "RMSE teste NFH", "MAE teste densa", "MAE teste densa+E", "MAE teste NFH"], linhas, "llrrrrrrrrrr", SAI,
                        "resultados/t2.csv.", longa=True, tam=r"\tiny"))
    else:
        P(aus("A T2 não rodou (ou t2.csv está ausente/vazio): sem tabelas da T2.") + "\n")

    # --- 5. Confirmação
    P(r"\section{Confirmação com sementes novas}")
    if cf_rodou or exemplo:
        P("As Tabelas~\\ref{tab:conf_rmse} e \\ref{tab:conf_mae} dão, por combinação, a média $\\pm$ desvio-padrão amostral "
          f"(denominador $n-1$) entre as sementes de RMSE e MAE de teste da densa, da densa+E, da \\texttt{{proj\\_K10}} e da NFH. "
          "A Tabela~\\ref{tab:conf_vit} traz as vitórias da NFH por semente (estritamente menor), a comparação com a melhor ingênua "
          "(média de treino, último valor ou sazonal de 24 h, a de menor erro de teste em cada métrica) e o veredito pelo critério do §6.\n")
        if exemplo:
            P(f"\\textbf{{Atenção:}} a linha marcada com $\\dagger$ é o exemplo passo a passo, com \\textbf{{outro protocolo}}: $H={PROTOCOLO['exemplo_H']}$, "
              f"série de livro, {exemplo['n_sementes']} sementes ({faixa(PROTOCOLO['exemplo_sementes_faixa'])}), critério com pelo menos {PROTOCOLO['exemplo_minimo_vitorias']} vitórias, "
              "sem \\texttt{proj\\_K10} nem ingênuas. Ela não é comparável às demais.\n")
    if cf_rodou or exemplo:
        P("\\textbf{O que as sementes variam, e o que não.} Na T1, na T2 e na confirmação, a semente muda só a inicialização dos pesos e a ordem dos lotes. "
          f"A divisão cronológica ({divisao()}) e as séries sintéticas são fixas, e a confirmação avalia o \\emph{{mesmo bloco de teste}} da T1 e da T2: "
          "as sementes são novas, o teste não. O desvio-padrão das tabelas mede, portanto, a variação do treino, e \\emph{não} mede a incerteza de amostragem do teste. "
          "Ganhos pequenos podem não se repetir em outra série ou em outro bloco de teste.\n")
    for a_ in avisos_conf:
        P(f"\\textbf{{Aviso:}} {esc(a_)}.\n")
    if cf_rodou or exemplo:
        def lin_rmse(c):
            return [esc(c["conjunto"]), esc(c["backbone"]), f"{c['fracao']:g}", str(c["largura"]), r"\texttt{" + esc(c.get("variante_nfh") or "n/d", True) + "}",
                    str(c.get("n_sementes", 0)), f"{c.get('densaE_epoca0_n', 'n/d')}/{c.get('n_sementes', 0)}",
                    mdp(c.get("densa_RMSE_media"), c.get("densa_RMSE_dp") or np.nan),
                    mdp(c.get("densaE_RMSE_media"), c.get("densaE_RMSE_dp") or np.nan),
                    mdp(c.get("proj_RMSE_media"), c.get("proj_RMSE_dp") or np.nan),
                    mdp(c.get("nfh_RMSE_media"), c.get("nfh_RMSE_dp") or np.nan)]

        def lin_mae(c):
            return [esc(c["conjunto"]), esc(c["backbone"]), f"{c['fracao']:g}", str(c["largura"]),
                    mdp(c.get("densa_MAE_media"), c.get("densa_MAE_dp") or np.nan), mdp(c.get("densaE_MAE_media"), c.get("densaE_MAE_dp") or np.nan),
                    mdp(c.get("proj_MAE_media"), c.get("proj_MAE_dp") or np.nan), mdp(c.get("nfh_MAE_media"), c.get("nfh_MAE_dp") or np.nan)]
        L_r = [lin_rmse(c) for c in conf]
        L_m = [lin_mae(c) for c in conf]
        L_v = []
        for c in conf:
            ni = c.get("ingenua_RMSE")
            L_v.append([esc(c["conjunto"]), esc(c["backbone"]), f"{c['fracao']:g}", str(c["largura"]),
                        f"{c.get('vitorias_RMSE_vs_densaE', 'n/d')}/{c.get('n_sementes', 0)}", f"{c.get('vitorias_MAE_vs_densaE', 'n/d')}/{c.get('n_sementes', 0)}",
                        f"{c.get('vitorias_RMSE_vs_densa', 'n/d')}/{c.get('n_sementes', 0)}", f"{c.get('vitorias_MAE_vs_densa', 'n/d')}/{c.get('n_sementes', 0)}",
                        (f"{nf(c['ingenua_RMSE']['valor'])} ({esc(c['ingenua_RMSE']['variante'].replace('ingenua_', '')[:4])})" if ni else "n/d"),
                        (pc(c["nfh_vs_ingenua_RMSE"]) if "nfh_vs_ingenua_RMSE" in c else "n/d"),
                        (pc(c["nfh_vs_ingenua_MAE"]) if "nfh_vs_ingenua_MAE" in c else "n/d"),
                        r"\textbf{" + c["veredito"] + "}"])
        if exemplo:
            e = exemplo
            r_ = ["SINT-H8 (livro)$^\\dagger$", f"1 camada, p={e['p']}", f"n={e['n']}", "--", r"\texttt{" + esc(e["variante_nfh"] or "NFH", True) + "}",
                  str(e["n_sementes"]), "n/d", mdp(e["densa_RMSE_media"], e["densa_RMSE_dp"]), mdp(e["densaE_RMSE_media"], e["densaE_RMSE_dp"]),
                  "n/d", mdp(e["nfh_RMSE_media"], e["nfh_RMSE_dp"])]
            L_r.append(r_)
            L_m.append(["SINT-H8 (livro)$^\\dagger$", f"1 camada, p={e['p']}", f"n={e['n']}", "--", mdp(e["densa_MAE_media"], e["densa_MAE_dp"]),
                        mdp(e["densaE_MAE_media"], e["densaE_MAE_dp"]), "n/d", mdp(e["nfh_MAE_media"], e["nfh_MAE_dp"])])
            ns = e["n_sementes"]
            L_v.append(["SINT-H8 (livro)$^\\dagger$", f"1 camada, p={e['p']}", f"n={e['n']}", "--", f"{e['vitorias_RMSE_vs_densaE']}/{ns}",
                        f"{e['vitorias_MAE_vs_densaE']}/{ns}", f"{e['vitorias_RMSE_vs_densa']}/{ns}", f"{e['vitorias_MAE_vs_densa']}/{ns}",
                        "n/d", "n/d", "n/d", r"\textbf{" + esc(e["veredito_final"]) + "}$^\\dagger$"])
        S.append(tabela("conf_rmse", "Confirma\\c{c}\\~ao: RMSE de teste, m\\'edia $\\pm$ desvio-padr\\~ao entre as sementes (escala padronizada "
                        "pelo treino; menor \\'e melhor). A coluna ``Sem. com densa+E = densa'' conta as sementes em que a \\'epoca escolhida da densa+E \\'e 0, "
                        "isto \\'e, a densa+E \\'e o mesmo modelo da densa.", ["Conjunto", "Backbone", "Fr.", "L", "Variante NFH", "Sem.", "Sem. com densa+E = densa (época 0)", "densa", "densa+E",
                        "proj\\_K10", "NFH"], L_r, "llrrlrrllll", SAI, "resultados/confirmacao.csv; $\\dagger$ = exemplo\\_sintetico/confirmacao.csv (outro protocolo, $H=8$).",
                        longa=True, tam=r"\tiny"))
        S.append(tabela("conf_mae", "Confirma\\c{c}\\~ao: MAE de teste, m\\'edia $\\pm$ desvio-padr\\~ao entre as sementes.",
                        ["Conjunto", "Backbone", "Fr.", "L", "densa", "densa+E", "proj\\_K10", "NFH"], L_m, "llrrllll", SAI,
                        "resultados/confirmacao.csv; $\\dagger$ = exemplo (outro protocolo).", longa=True, tam=r"\tiny"))
        S.append(tabela("conf_vit", "Confirma\\c{c}\\~ao: vit\\'orias da NFH por semente (estritamente menor), compara\\c{c}\\~ao com a melhor ing\\^enua e veredito do \\S6.",
                        ["Conjunto", "Backbone", "Fr.", "L", "RMSE vs densa+E", "MAE vs densa+E", "RMSE vs densa", "MAE vs densa",
                         "Melhor ing\\^enua (RMSE)", "NFH vs ing. RMSE", "NFH vs ing. MAE", "Veredito"], L_v, "llrrrrrrlrrl", SAI,
                        "resultados/confirmacao.csv (ing\\^enuas: linhas \\texttt{ingenua\\_*}); $\\dagger$ = exemplo (outro protocolo).",
                        longa=True, tam=r"\tiny"))
        e0_ = [c for c in conf if c.get("densaE_epoca0_n")]
        if e0_:
            P("\\textbf{Controle das épocas extras sem efeito em parte das sementes.} Quando a época escolhida da densa+E é 0, as 15 épocas extras não melhoraram "
              "a validação e a densa+E é o mesmo modelo da densa"
              + (" (RMSE de teste igual nessas sementes)" if all(c["densaE_epoca0_rmse_igual_densa"] == c["densaE_epoca0_n"] for c in e0_)
                 else " (o RMSE de teste de alguma dessas sementes difere: ver numeros.json)")
              + ". Isso ocorre, em número de sementes, em: "
              + "; ".join(f"{idc(c)}: {c['densaE_epoca0_n']} de {c['n_sementes']}" for c in e0_)
              + ". Nessas sementes o controle não acrescenta informação; a comparação continua válida, porque a NFH é comparada com o melhor que a densa "
                "conseguiu na validação, mas a coluna da densa+E repete a da densa.\n")
        if "confirmacao" in figs:
            P(r"\begin{figure}[htbp]\centering\caption{Confirmação: RMSE de teste da NFH relativo à densa+E, por semente (pontos cinza) "
              r"e na média (losango preto). Valores negativos favorecem a NFH.}\label{fig:conf}"
              r"\includegraphics[width=0.95\linewidth]{" + figs["confirmacao"] + r"}\par\smallskip{\scriptsize Fonte: resultados/confirmacao.csv.}\end{figure}")
    else:
        P(aus("A confirmação não rodou: sem tabelas de confirmação.") + "\n")

    def par_exemplo():
        P(r"\paragraph{Exemplo passo a passo (outro protocolo).} " +
              f"O exemplo didático ($H={PROTOCOLO['exemplo_H']}$, $p={exemplo['p']}$, $n={exemplo['n']}$ janelas de treino, variante "
              f"\\texttt{{{esc(exemplo['variante_nfh'] or 'NFH', True)}}}) tem veredito {esc(exemplo['veredito_final'])} no seu próprio "
              f"critério ({exemplo['n_sementes']} sementes, pelo menos {PROTOCOLO['exemplo_minimo_vitorias']} vitórias): RMSE de teste "
              f"{mdp(exemplo['nfh_RMSE_media'], exemplo['nfh_RMSE_dp'])} (NFH), {mdp(exemplo['densa_RMSE_media'], exemplo['densa_RMSE_dp'])} (densa) e "
              f"{mdp(exemplo['densaE_RMSE_media'], exemplo['densaE_RMSE_dp'])} (densa+E). Foi avaliado no próprio protocolo (sem as ingênuas nem a \\texttt{{proj\\_K10}}); "
              "não é uma das combinações da T1, da T2 ou da confirmação de \\emph{backbones}.\n")

    # --- 6. Exemplos
    P(r"\section{Exemplos em que a NFH iguala ou supera a densa}\label{sec:exemplos}")
    suc = [c for c in conf if c["veredito"] == "SUCESSO"]
    if not cf_rodou:
        P(aus("A confirmação não rodou: nenhum exemplo a listar.") + "\n")
    elif not suc:
        P("Nenhuma combinação de \\emph{backbones} teve SUCESSO no critério do §6. Não há, portanto, exemplo a citar entre elas; as combinações confirmadas "
          "estão na Seção~\\ref{sec:naoganhou}.\n")
    else:
        P("Esta seção lista \\emph{somente} as combinações com SUCESSO: média de RMSE e de MAE de teste da NFH menores ou iguais às da densa "
          f"e da densa+E, e vitória em RMSE contra a densa+E em pelo menos {PROTOCOLO['minimo_vitorias_rmse_vs_densaE']} das "
          f"{len(PROTOCOLO['sementes_confirmacao'])} sementes. Elas foram \\emph{{procuradas}} (T1, T2 e confirmação) e não representam o comportamento geral.\n")
        for c in suc:
            ni = c.get("ingenua_RMSE")
            P(f"\\paragraph{{{esc(c['conjunto'])}, {esc(c['backbone'])}, fração {c['fracao']:g}, largura {c['largura']}, variante "
              f"\\texttt{{{esc(c['variante_nfh'], True)}}}.}} "
              f"RMSE de teste (média $\\pm$ dp, {c['n_sementes']} sementes): NFH {mdp(c['nfh_RMSE_media'], c['nfh_RMSE_dp'])}; densa "
              f"{mdp(c['densa_RMSE_media'], c['densa_RMSE_dp'])}; densa+E {mdp(c['densaE_RMSE_media'], c['densaE_RMSE_dp'])}; "
              f"\\texttt{{proj\\_K10}} {mdp(c.get('proj_RMSE_media'), c.get('proj_RMSE_dp') or np.nan)}. "
              f"MAE de teste: NFH {mdp(c['nfh_MAE_media'], c['nfh_MAE_dp'])}; densa {mdp(c['densa_MAE_media'], c['densa_MAE_dp'])}; "
              f"densa+E {mdp(c['densaE_MAE_media'], c['densaE_MAE_dp'])}. A diferença relativa das médias da NFH é {pc(c['rel_RMSE_media_vs_densaE'])} "
              f"(RMSE) e {pc(c['rel_MAE_media_vs_densaE'])} (MAE) contra a densa+E, e {pc(c['rel_RMSE_media_vs_densa'])} e "
              f"{pc(c['rel_MAE_media_vs_densa'])} contra a densa. Vitórias da NFH: RMSE {c['vitorias_RMSE_vs_densaE']} de {c['n_sementes']} "
              f"e MAE {c['vitorias_MAE_vs_densaE']} de {c['n_sementes']} contra a densa+E; RMSE {c['vitorias_RMSE_vs_densa']} e MAE "
              f"{c['vitorias_MAE_vs_densa']} contra a densa. Cobertura mediana $1-\\varepsilon_R$ da NFH: {nf(c.get('cobertura_mediana_nfh_media'), 3)}. ")
            if ni:
                P(f"Melhor ingênua em RMSE: {esc(ni['variante'])} ({nf(ni['valor'])}); a NFH fica {pc(c['nfh_vs_ingenua_RMSE'])} dela em RMSE "
                  f"e {pc(c['nfh_vs_ingenua_MAE'])} em MAE (melhor ingênua em MAE: {esc(c['ingenua_MAE']['variante'])}). "
                  + ("A densa fica " + pc(c["densa_vs_ingenua_RMSE"]) + " da melhor ingênua em RMSE." if "densa_vs_ingenua_RMSE" in c else ""))
            else:
                P("Não há linha de ingênuas para este conjunto na confirmação.")
            P("\n")
    ex_ok = bool(exemplo) and exemplo["veredito_final"] == "SUCESSO"
    if ex_ok:
        par_exemplo()

    # --- 7. Onde não ganhou
    P(r"\section{Onde a NFH não ganhou}\label{sec:naoganhou}")
    P("\\emph{Derrota}: a média de RMSE de teste da NFH é maior que a da densa ou a da densa+E. \\emph{Falha pelo MAE}: as médias de RMSE não "
      "pioram, mas a média de MAE de teste da NFH é maior que a da densa ou a da densa+E; é uma derrota parcial. \\emph{Falha pelas vitórias}: "
      "as médias de RMSE e de MAE não pioram, mas a NFH vence a densa+E em menos sementes que o mínimo. Não há limiar de tolerância, e este relatório "
      "não chama nenhuma combinação de empate.\n")
    if exemplo and not ex_ok:
        par_exemplo()
    nao = [c for c in conf if c["veredito"] != "SUCESSO"]
    if cf_rodou and nao:
        linhas = []
        for c in nao:
            linhas.append([esc(c["conjunto"]), esc(c["backbone"]), f"{c['fracao']:g}", str(c["largura"]),
                           r"\texttt{" + esc(c.get("variante_nfh") or "n/d", True) + "}", esc(c.get("tipo_falha", c["veredito"].lower())),
                           pc(c.get("rel_RMSE_media_vs_densaE")), pc(c.get("rel_MAE_media_vs_densaE")),
                           pc(c.get("rel_RMSE_media_vs_densa")), f"{c.get('vitorias_RMSE_vs_densaE', 'n/d')}/{c.get('n_sementes', 0)}",
                           esc(c.get("motivo", ""))])
        S.append(tabela("nao_ganhou_conf", "Confirma\\c{c}\\~ao: combina\\c{c}\\~oes sem SUCESSO (diferen\\c{c}as relativas das m\\'edias de teste).",
                        ["Conjunto", "Backbone", "Fr.", "L", "Variante", "Tipo", "RMSE vs densa+E", "MAE vs densa+E", "RMSE vs densa",
                         "Vit. RMSE", "Falhas do crit\\'erio"], linhas, "llrrllrrrlp{3.6cm}", SAI, "resultados/confirmacao.csv.",
                        longa=True, tam=r"\tiny"))
        for c in nao:
            if c.get("todos_acima_ultimo_RMSE"):
                mae_txt = (f" e, no MAE, {nf(c['ingenua_ultimo_MAE'])} contra {nf(c['nfh_MAE_media'])} da NFH"
                           if c.get("todos_acima_ultimo_MAE") else "")
                P(f"Em {idc(c)}, fração {dec(c['fracao'])}: \\textbf{{todos os modelos ficam acima da ingênua do último valor}}: RMSE médio de teste de "
                  f"{nf(c['ingenua_ultimo_RMSE'])} para a ingênua contra {nf(c['nfh_RMSE_media'])} (NFH), {nf(c['densa_RMSE_media'])} (densa) e "
                  f"{nf(c['densaE_RMSE_media'])} (densa+E){mae_txt}. A NFH não bate nem a ingênua nessa combinação; a diferença para a densa+E não muda isso.\n")
    elif cf_rodou:
        P("Todas as combinações confirmadas tiveram SUCESSO; não há derrota nem falha pelo critério na confirmação.\n")
    else:
        P(aus("A confirmação não rodou: sem derrotas da confirmação a listar.") + "\n")
    if t2_rodou:
        nq = best2[~best2["passa"].fillna(best2["passa_calc"]).astype(bool)]
        P(f"Na T2, {len(nq)} dos {len(best2)} pares (combinação, largura) não qualificaram para a confirmação (Tabela~\\ref{{tab:nao_ganhou_t2}}).\n")
        if len(nq):
            linhas = [[esc(r["conjunto"]), esc(r["backbone"]), f"{r['fracao']:g}", str(int(r["largura"])), r"\texttt{" + esc(r["variante"], True) + "}",
                       pc(r["rE_val_RMSE"]), pc(r["rE_val_MAE"]), pc(r["pior_sem_val"]), pc(r["rE_test_RMSE"]), pc(r["rE_test_MAE"])]
                      for _, r in nq.iterrows()]
            S.append(tabela("nao_ganhou_t2", "T2: pares sem qualifica\\c{c}\\~ao (m\\'edias sobre as sementes; contra a densa+E).",
                            ["Conjunto", "Backbone", "Fr.", "L", "Variante", "RMSE val", "MAE val", "Pior sem. val", "RMSE teste", "MAE teste"],
                            linhas, "llrrlrrrrr", SAI, "resultados/t2.csv.", longa=True, tam=r"\tiny"))
        qual_ = best2["passa"].fillna(best2["passa_calc"]).astype(bool)
        conf_ = best2["vai_conf"].fillna(False).astype(bool)
        perd_ = (best2["rE_test_RMSE"] > 0) | (best2["rE_test_MAE"] > 0)
        pq = best2[qual_ & ~conf_ & perd_]
        P(f"Também na T2, {int((qual_ & ~conf_).sum())} pares qualificaram pela validação e não foram à confirmação (limites da seleção da T2); "
          f"{len(pq)} deles tiveram RMSE ou MAE de teste médio \\emph{{pior}} que o da densa+E (Tabela~\\ref{{tab:t2_qualif_perdeu}}). Eles não passaram pela confirmação, "
          "e a perda no teste da T2 é média de 3 sementes, sem sementes novas.\n")
        if len(pq):
            def _pior(r):
                a, b = r["rE_test_RMSE"] > 0, r["rE_test_MAE"] > 0
                return "RMSE e MAE" if a and b else ("RMSE" if a else "MAE")
            linhas = [[esc(r["conjunto"]), esc(r["backbone"]), f"{r['fracao']:g}", str(int(r["largura"])), r"\texttt{" + esc(r["variante"], True) + "}",
                       pc(r["rE_val_RMSE"]), pc(r["rE_val_MAE"]), pc(r["rE_test_RMSE"]), pc(r["rE_test_MAE"]), _pior(r)] for _, r in pq.iterrows()]
            S.append(tabela("t2_qualif_perdeu", "T2: pares que qualificaram, n\\~ao foram \\`a confirma\\c{c}\\~ao e perderam no teste "
                            "(m\\'edias sobre as sementes; contra a densa+E).",
                            ["Conjunto", "Backbone", "Fr.", "L", "Variante", "RMSE val", "MAE val", "RMSE teste", "MAE teste", "Pior em"],
                            linhas, "llrrlrrrrl", SAI, "resultados/t2.csv e selecao\\_T2.md.", longa=True, tam=r"\tiny"))
    else:
        P(aus("A T2 não rodou: sem derrotas da T2.") + "\n")
    if t1_rodou:
        n1 = N["t1"]
        nao_val = n1["combinacoes"] - n1["vence_val_RMSE"]
        P(f"Na T1, a melhor variante NFH não venceu a densa+E em RMSE de validação em {nao_val} das {n1['combinacoes']} combinações e não a venceu "
          f"em RMSE de teste em {n1['combinacoes'] - n1['vence_test_RMSE']}. A mediana da diferença relativa de RMSE de teste da melhor variante "
          f"é {pc(n1['mediana_rel_test_RMSE_vs_densaE'])} contra a densa+E e {pc(n1['mediana_rel_test_RMSE_vs_densa'])} contra a densa. "
          "Todas as combinações da T1 estão no Apêndice~\\ref{ap:t1}.\n")

    # --- 8. explicações
    def bloco_expl(lista, rot_tab, controle):
        linhas = []
        for e in lista:
            n_, d_ = e["nfh"], e["densa_projetada"]
            f = lambda b, k: (f"{nf(b[k]['mediana'], 3)} [{nf(b[k]['q1'], 3)}; {nf(b[k]['q3'], 3)}]")
            ig = e.get("ig", {})
            lin = [esc(e["combinacao"], True)] + ([esc(e["veredito_recalculado"])] if controle else [])
            linhas.append(lin + ["NFH", f(n_, "eps_T"), f(n_, "eps_S"), f(n_, "eps_R"),
                                 nf(ig.get("g_T", {}).get("completude_rel_mediano"), 4), nf(ig.get("g_S", {}).get("completude_rel_mediano"), 4)])
            linhas.append([""] * (2 if controle else 1) + ["densa (Projetor)", f(d_, "eps_T"), f(d_, "eps_S"), f(d_, "eps_R"), "--", "--"])
        cab = ["Combina\\c{c}\\~ao"] + (["Veredito"] if controle else []) + ["Modelo", "$\\varepsilon_T$", "$\\varepsilon_S$",
                                                                          "$\\varepsilon_R$", "Compl. $g_T$", "Compl. $g_S$"]
        S.append(tabela(rot_tab, ("Controles sem SUCESSO: " if controle else "") + "Explica\\c{c}\\~oes (semente 10, teste): participa\\c{c}\\~oes de energia, "
                        "mediana [Q1; Q3], e erro relativo mediano de completude dos IG (s\\'o NFH).", cab, linhas,
                        "lllllllrr" if controle else "lllllrr", SAI, "resultados/explicacoes/*.json.", longa=True, tam=r"\tiny"))
        for e in lista:
            P(r"\subsection{" + esc(e["combinacao"], True) + (f" (veredito {esc(e['veredito_recalculado'])})" if controle else "") + "}")
            P(e.get("texto_relatorio_tex") or "")
            fg = e.get("figura")
            if fg and (RES / "explicacoes" / fg).exists():
                dst = SAI / "figuras" / ("expl_" + fg)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(RES / "explicacoes" / fg, dst)
                P(r"\begin{figure}[htbp]\centering\caption{" + esc(e["combinacao"], True) + ": uma janela de teste (previs\\~ao, $T$, $S$, $R$ e, nos sint\\'eticos, a verdade).}"
                  r"\includegraphics[width=0.95\linewidth]{figuras/expl_" + fg + r"}\par\smallskip{\scriptsize Fonte: explicar\_bb.py.}\end{figure}")

    P(r"\section{Explicações das combinações com SUCESSO}")
    P("Esta seção só inclui as combinações cujo veredito recalculado a partir de \\texttt{confirmacao.csv} é SUCESSO; o campo de veredito "
      "gravado nos arquivos de explicação não é usado para essa escolha.\n")
    if expl:
        P(f"Para {len(expl)} combinações, \\texttt{{explicar\\_bb.py}} usou o modelo da semente 10 salvo pela confirmação e calculou, no teste, a decomposição "
          "$T$, $S$, $R$, as participações $\\varepsilon_T,\\varepsilon_S,\\varepsilon_R$ (mediana e quartis), a decomposição pura da densa pelo "
          "Projetor e os Integrated Gradients de $g_T$ e $g_S$ (referência zero) com o erro de completude.\n")
        bloco_expl(expl, "expl_eps", False)
    else:
        P(aus("Não há explicação de combinação com SUCESSO em resultados/explicacoes/ (explicar_bb.py não rodou ou não houve SUCESSO).") + "\n")
    if expl_ctrl:
        P(r"\subsection*{Controles, sem SUCESSO}")
        P("Os arquivos abaixo foram gerados com \\texttt{--lista} (ou vêm de uma confirmação anterior) para combinações que \\textbf{não} têm SUCESSO "
          "na confirmação atual. Cada um traz o seu veredito. Não são exemplos de vitória da NFH.\n")
        bloco_expl(expl_ctrl, "expl_eps_ctrl", True)
    for d_ in expl_desc:
        P(f"Aviso: o arquivo de explicação de \\texttt{{{esc(d_['combinacao'], True)}}} diz {esc(d_['json'])}, mas a confirmação atual dá {esc(d_['recalculado'])}; vale a confirmação atual.\n")
    for ee in expl_erro:
        P(f"A explicação de \\texttt{{{esc(ee['combinacao'], True)}}} falhou: {esc(ee['erro'])}.\n")

    # --- Apêndices
    P(r"\appendix")
    P(r"\section{Todas as combinações da T1}\label{ap:t1}")
    if t1_rodou:
        linhas = []
        for _, r in best1.iterrows():
            q = "n/d" if pd.isna(r["passa"]) else ("sim" if bool(r["passa"]) else "n\\~ao")
            t = "n/d" if pd.isna(r["vai_T2"]) else ("sim" if bool(r["vai_T2"]) else "n\\~ao")
            linhas.append([esc(r["conjunto"]), esc(r["backbone"]), f"{r['fracao']:g}", r"\texttt{" + esc(r["variante"], True) + "}",
                           pc(r["rE_val_RMSE"]), pc(r["rE_val_MAE"]), pc(r["rE_test_RMSE"]), pc(r["rE_test_MAE"]),
                           pc(r["rD_test_RMSE"]), pc(r["rD_test_MAE"]), q, t])
        S.append(tabela("t1_todas", "T1: todas as combina\\c{c}\\~oes (melhor variante NFH pelo RMSE de valida\\c{c}\\~ao; diferen\\c{c}as contra a densa+E, "
                        "e contra a densa nas colunas ``vs densa'').", ["Conjunto", "Backbone", "Fr.", "Variante", "RMSE val", "MAE val",
                        "RMSE teste", "MAE teste", "RMSE teste vs densa", "MAE teste vs densa", "Qualif.", "T2"], linhas,
                        "llrlrrrrrrll", SAI, "resultados/t1.csv e selecao\\_T1.md.", longa=True, tam=r"\tiny"))
    else:
        P(aus("A T1 não rodou.") + "\n")
    P(r"\section{Diferença por semente na confirmação}")
    if cf_rodou:
        sem = PROTOCOLO["sementes_confirmacao"]
        linhas = []
        for c in conf:
            d_ = c.get("rel_RMSE_por_semente_vs_densaE", {})
            linhas.append([esc(c["conjunto"]), esc(c["backbone"]), f"{c['fracao']:g}"] + [pc(d_.get(s)) if s in d_ else "n/d" for s in sem])
        S.append(tabela("conf_semente", "Confirma\\c{c}\\~ao: RMSE de teste da NFH relativo \\`a densa+E em cada semente.",
                        ["Conjunto", "Backbone", "Fr."] + [f"s{s}" for s in sem], linhas, "llr" + "r" * len(sem), SAI,
                        "resultados/confirmacao.csv."))
    else:
        P(aus("A confirmação não rodou.") + "\n")
    P(r"\section{Registro do §10 do protocolo (texto integral)}\label{ap:s10}")
    if itens10:
        P(r"\begin{itemize}\setlength{\itemsep}{2pt}\small")
        for it in itens10:
            P(r"\item " + md2tex(it))
        P(r"\end{itemize}")
    else:
        P(aus("Não encontrado.") + "\n")
    P(r"\end{document}")

    # tabelas escritas durante a montagem estão em S (strings \input) misturadas ao texto
    (SAI / "numeros.json").write_text(json.dumps(N, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                                      encoding="utf-8")
    (SAI / "relatorio.tex").write_text(("\n".join(S) + "\n").replace("§", r"\S{}"), encoding="utf-8")
    for tx in SAI.glob("*.tex"):
        tx.write_text(tx.read_text(encoding="utf-8").replace("§", r"\S{}"), encoding="utf-8")
    print(f"relatorio.tex e numeros.json escritos em {SAI}")
    if a.sem_pdf:
        return
    if not TECTONIC.exists():
        print(f"AVISO: {TECTONIC} não existe; PDF não gerado")
        return
    r = subprocess.run([str(TECTONIC), "relatorio.tex"], cwd=SAI, capture_output=True, text=True)
    (SAI / "tectonic.log").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0 or not (SAI / "relatorio.pdf").exists():
        print("ERRO na compilação; ver tectonic.log")
        print(r.stderr[-3000:])
        sys.exit(1)
    print("relatorio.pdf gerado")


if __name__ == "__main__":
    main()
