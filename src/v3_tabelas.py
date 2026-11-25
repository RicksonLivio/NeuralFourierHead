"""
Tabelas e números do Capítulo 5 da versão 3, gerados a partir de results/v3/ (nada digitado à mão).

Saídas:
  versao3_correcao2_aplicada/fontes/tabelas/tab_v3_*.tex   (corpo das tabelas, incluídas com \\input)
  results/v3/numeros_cap5.json                              (todos os números citados no texto)
Uso: python src/v3_tabelas.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from v3_common import BACKBONES, CONJUNTOS, RES3, ROOT  # noqa: E402

TAB = ROOT / "versao3_correcao2_aplicada" / "fontes" / "tabelas"
TAB.mkdir(parents=True, exist_ok=True)
NAIVES = ["Ingênuo (média de treino)", "Ingênuo (último valor)", "Ingênuo sazonal (24 h)"]
ROT = {"ETTh1": "ETTh1", "ETTh2": "ETTh2", "Jena": "Jena", "AirQ": "Air Quality"}
NUM = {}


def n(x, nd=3, pct=False, sinal=False):
    """Número no padrão brasileiro dentro de modo matemático."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    v = 100 * x if pct else x
    s = f"{v:+.{nd}f}" if sinal else f"{v:.{nd}f}"
    return "$" + s.replace(".", "{,}").replace("-", "-") + "$"


def rot(conj):
    f, m = conj.split("-")
    return f"{ROT[f]}-{m}"


def carregar_runs():
    dfs = []
    for c in CONJUNTOS:
        f = RES3 / "runs" / f"{c}.csv"
        if f.exists():
            dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    df["variante"] = df["variante"].fillna("")
    df["K"] = df["K"].fillna(-1).astype(int)
    # proteção contra linhas duplicadas por retomada: fica a última, a mesma execução que gravou os
    # arquivos de partes e de modelos (a retomada refaz a semente inteira)
    return df.drop_duplicates(subset=["conjunto", "modelo", "variante", "K", "seed"], keep="last")


def media_seeds(df, conj, modelo, variante="", K=-1, col="RMSE"):
    s = df[(df.conjunto == conj) & (df.modelo == modelo) & (df.variante == variante) & (df.K == K) & (df.seed >= 0)]
    return (float(s[col].mean()), float(s[col].std()), int(len(s))) if len(s) else (np.nan, np.nan, 0)


def naive(df, conj, nome):
    s = df[(df.conjunto == conj) & (df.modelo == nome)]
    return float(s.RMSE.iloc[0]) if len(s) else np.nan


def tab_acuracia(df):
    """RMSE de teste por conjunto: três ingênuos, N-BEATS-I e cada backbone com camada densa e com NFH."""
    linhas, num = [], {}
    cols = [("N-BEATS-I", "")] + [(f"{bb} + {cab}", var) for bb in BACKBONES for cab, var in (("densa", ""), ("NFH", "seguro"))]
    for conj in CONJUNTOS:
        nv = [naive(df, conj, k) for k in NAIVES]
        best_nv = np.nanmin(nv)
        vals = []
        for mod, var in cols:
            K = 10 if var == "seguro" else -1
            vals.append(media_seeds(df, conj, mod, var, K))
        medias = [v[0] for v in vals]
        melhor = np.nanmin(medias + nv)
        cel = []
        for x in nv + medias:
            s = n(x, 3)
            if np.isfinite(x) and np.isclose(x, melhor):
                s = "$\\mathbf{" + s.strip("$") + "}$"
            cel.append(s)
        linhas.append(f"{rot(conj)} & " + " & ".join(cel) + r" \\")
        num[conj] = {"ingenuos": dict(zip(NAIVES, nv)), "melhor_ingenuo": best_nv,
                     "modelos": {f"{m}|{v}": {"media": a, "dp": b, "n": c} for (m, v), (a, b, c) in zip(cols, vals)}}
    corpo = "\n".join(linhas)
    cab = (r"\begin{tabular}{l" + "c" * (3 + len(cols)) + "}\n\\toprule\n"
           r" & \multicolumn{3}{c}{Ingênuos} & & \multicolumn{2}{c}{MLP} & \multicolumn{2}{c}{LSTM} & \multicolumn{2}{c}{Conv1D + LSTM} \\"
           "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}" "\n"
           r"Conjunto & Média & Último & Sazonal & N-BEATS-I & Densa & NFH & Densa & NFH & Densa & NFH \\" "\n\\midrule\n")
    (TAB / "tab_v3_acuracia.tex").write_text(cab + corpo + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    NUM["acuracia"] = num
    return num


def comparacoes(df):
    """NFH (seguro, K = 10) contra a densa no mesmo backbone e todos os modelos contra o melhor ingênuo."""
    pares, ganho_ing = [], []
    for conj in CONJUNTOS:
        best_nv = np.nanmin([naive(df, conj, k) for k in NAIVES])
        for bb in BACKBONES:
            rd = media_seeds(df, conj, f"{bb} + densa")[0]
            rn = media_seeds(df, conj, f"{bb} + NFH", "seguro", 10)[0]
            if np.isfinite(rd) and np.isfinite(rn):
                pares.append({"conjunto": conj, "backbone": bb, "rel": (rn - rd) / rd, "densa": rd, "nfh": rn})
            for cab, var, K in (("densa", "", -1), ("NFH", "seguro", 10)):
                r = media_seeds(df, conj, f"{bb} + {cab}", var, K)[0]
                if np.isfinite(r):
                    ganho_ing.append({"conjunto": conj, "modelo": f"{bb} + {cab}", "rel_ingenuo": (r - best_nv) / best_nv})
        r = media_seeds(df, conj, "N-BEATS-I")[0]
        if np.isfinite(r):
            ganho_ing.append({"conjunto": conj, "modelo": "N-BEATS-I", "rel_ingenuo": (r - best_nv) / best_nv})
    P = pd.DataFrame(pares)
    G = pd.DataFrame(ganho_ing)
    out = {"n_pares": int(len(P))}
    if len(P) >= 5:
        rel = P.rel.values
        out.update({"media_rel": float(rel.mean()), "mediana_rel": float(np.median(rel)),
                    "vitorias_nfh": int((rel < 0).sum()), "min_rel": float(rel.min()), "max_rel": float(rel.max()),
                    "wilcoxon_bilateral_p": float(stats.wilcoxon(rel).pvalue),
                    "nao_inferioridade_1pct_p": float(stats.wilcoxon(rel - 0.01, alternative="less").pvalue),
                    # sensibilidade à margem (só para a defesa; a margem do texto continua 1%)
                    "nao_inferioridade_margens_p": {f"{mg:.4f}": float(stats.wilcoxon(rel - mg, alternative="less").pvalue)
                                                    for mg in (0.0, 0.0025, 0.005, 0.01)},
                    "por_backbone": {bb: {"media_rel": float(P[P.backbone == bb].rel.mean()),
                                          "vitorias": int((P[P.backbone == bb].rel < 0).sum()),
                                          "n": int((P.backbone == bb).sum())} for bb in BACKBONES},
                    "pares": P.to_dict("records")})
    if len(G):
        out["vs_melhor_ingenuo"] = {m: {"media_rel": float(g.rel_ingenuo.mean()), "max_rel": float(g.rel_ingenuo.max()),
                                        "n_melhor_que_ingenuo": int((g.rel_ingenuo < 0).sum()), "n": int(len(g))}
                                    for m, g in G.groupby("modelo")}
        out["todos_melhores_que_ingenuo"] = bool((G.rel_ingenuo < 0).all())
        out["vs_ingenuo_por_conjunto"] = G.to_dict("records")
    # agregação por fonte (média dos seis pares U/M x backbone de cada fonte), Seção 5.4
    if len(P):
        fonte = P.conjunto.str.split("-").str[0]
        por_fonte = P.groupby(fonte).rel.mean()
        out["por_fonte"] = {f: float(v) for f, v in por_fonte.items()}
        if len(por_fonte) >= 2:
            out["por_fonte_wilcoxon_bilateral_p"] = float(stats.wilcoxon(por_fonte.values).pvalue)
    # mesma comparação NFH x densa com o MAE (citada no texto da Seção 5.4)
    rel_mae = []
    for conj in CONJUNTOS:
        for bb in BACKBONES:
            md = media_seeds(df, conj, f"{bb} + densa", col="MAE")[0]
            mn = media_seeds(df, conj, f"{bb} + NFH", "seguro", 10, col="MAE")[0]
            if np.isfinite(md) and np.isfinite(mn):
                rel_mae.append((mn - md) / md)
    if len(rel_mae) >= 5:
        rm = np.array(rel_mae)
        out["mae"] = {"n_pares": int(len(rm)), "media_rel": float(rm.mean()), "mediana_rel": float(np.median(rm)),
                      "vitorias_nfh": int((rm < 0).sum()), "min_rel": float(rm.min()), "max_rel": float(rm.max()),
                      "wilcoxon_bilateral_p": float(stats.wilcoxon(rm).pvalue),
                      "nao_inferioridade_1pct_p": float(stats.wilcoxon(rm - 0.01, alternative="less").pvalue)}
    NUM["comparacoes"] = out
    return out


def tab_variantes(df):
    """Ablação de K e variantes: decomposição pura, projeção, NFH segura e estrutura completa (relativo à densa)."""
    linhas, num, piores = [], {}, {}
    for K in (10, 4, 2):
        reg = {"kappa_densa": [], "rel_proj": [], "rel_seg": [], "kappa_seg": [], "rel_comp": [], "kappa_comp": [],
               "ep0": [], "val_ok": [], "teste_pior": [], "teste_melhor": []}
        for conj in CONJUNTOS:
            for bb in BACKBONES:
                for s in range(5):
                    q = df[(df.conjunto == conj) & (df.seed == s)]
                    dz = q[(q.modelo == f"{bb} + densa") & (q.variante == "")]
                    dp = q[(q.modelo == f"{bb} + densa") & (q.variante == "decomposição pura") & (q.K == K)]
                    pr = q[(q.modelo == f"{bb} + densa") & (q.variante == "projeção") & (q.K == K)]
                    sg = q[(q.modelo == f"{bb} + NFH") & (q.variante == "seguro") & (q.K == K)]
                    cp = q[(q.modelo == f"{bb} + NFH") & (q.variante == "completo") & (q.K == K)]
                    if not (len(dz) and len(dp) and len(pr) and len(sg) and len(cp)):
                        continue
                    rd = float(dz.RMSE.iloc[0])
                    reg["kappa_densa"].append(float(dp.cobertura_mediana.iloc[0]))
                    reg["rel_proj"].append((float(pr.RMSE.iloc[0]) - rd) / rd)
                    reg["rel_seg"].append((float(sg.RMSE.iloc[0]) - rd) / rd)
                    reg["kappa_seg"].append(float(sg.cobertura_mediana.iloc[0]))
                    reg["rel_comp"].append((float(cp.RMSE.iloc[0]) - rd) / rd)
                    reg["kappa_comp"].append(float(cp.cobertura_mediana.iloc[0]))
                    reg["ep0"].append(int(sg.epoca_ajuste.iloc[0]) == 0)
                    reg["val_ok"].append(float(sg.val_RMSE.iloc[0]) <= float(sg.rmse_val_densa.iloc[0]) + 1e-12)
                    # tolerância relativa de 1e-6: na época 0 a NFH reproduz a densa a menos do erro de conversão
                    # (diferenças de até 2e-8), o que é empate, e não derrota nem vitória
                    reg["teste_pior"].append(float(sg.RMSE.iloc[0]) > rd * (1 + 1e-6))
                    reg["teste_melhor"].append(float(sg.RMSE.iloc[0]) < rd * (1 - 1e-6))
                    piores.setdefault(K, []).append(
                        {"conjunto": conj, "backbone": bb, "seed": s, "rel_teste": (float(sg.RMSE.iloc[0]) - rd) / rd,
                         "rel_validacao": (float(sg.val_RMSE.iloc[0]) - float(sg.rmse_val_densa.iloc[0]))
                         / float(sg.rmse_val_densa.iloc[0]), "epoca": int(sg.epoca_ajuste.iloc[0])})
        if not reg["rel_seg"]:
            continue
        r = {k: np.array(v, dtype=float) for k, v in reg.items()}
        num[K] = {"n": int(len(r["rel_seg"])), "kappa_densa_mediana": float(np.median(r["kappa_densa"])),
                  "kappa_densa_min": float(r["kappa_densa"].min()),
                  "rel_proj_media": float(r["rel_proj"].mean()), "rel_proj_max": float(r["rel_proj"].max()),
                  "rel_seg_media": float(r["rel_seg"].mean()), "rel_seg_max": float(r["rel_seg"].max()),
                  "rel_seg_min": float(r["rel_seg"].min()),
                  "kappa_seg_mediana": float(np.median(r["kappa_seg"])),
                  "rel_comp_media": float(r["rel_comp"].mean()), "rel_comp_max": float(r["rel_comp"].max()),
                  "kappa_comp_mediana": float(np.median(r["kappa_comp"])),
                  "fracao_epoca0": float(r["ep0"].mean()), "fracao_val_ok": float(r["val_ok"].mean()),
                  "fracao_teste_pior": float(r["teste_pior"].mean()),
                  "n_teste_melhor": int(r["teste_melhor"].sum()), "n_teste_pior": int(r["teste_pior"].sum()),
                  "n_teste_empate": int(len(r["rel_seg"]) - r["teste_melhor"].sum() - r["teste_pior"].sum()),
                  "kappa_seg_mediana_epoca0": float(np.median(r["kappa_seg"][r["ep0"] > 0])) if r["ep0"].any() else None,
                  "kappa_seg_mediana_demais": float(np.median(r["kappa_seg"][r["ep0"] == 0])) if (r["ep0"] == 0).any() else None,
                  # diferença média de RMSE de teste só nas execuções que saíram da época 0 (Seção 5.5)
                  "n_demais": int((r["ep0"] == 0).sum()),
                  "rel_seg_media_demais": float(r["rel_seg"][r["ep0"] == 0].mean()) if (r["ep0"] == 0).any() else None}
        u = num[K]
        linhas.append(f"$K = {K}$ & {n(u['kappa_densa_mediana'], 2, pct=True)} & {n(u['rel_proj_media'], 2, pct=True, sinal=True)} & "
                      f"{n(u['rel_seg_media'], 2, pct=True, sinal=True)} & {n(u['kappa_seg_mediana'], 2, pct=True)} & "
                      f"{n(u['fracao_epoca0'], 0, pct=True)} & ${u['n_teste_melhor']}/{u['n_teste_empate']}/{u['n_teste_pior']}$ & "
                      f"{n(u['rel_comp_media'], 2, pct=True, sinal=True)} & {n(u['kappa_comp_mediana'], 2, pct=True)} \\\\")
    cab = (r"\begin{tabular}{lcccccccc}" "\n\\toprule\n"
           r" & Decomp. pura & Projeção & \multicolumn{4}{c}{NFH (seleção segura)} & \multicolumn{2}{c}{Estrutura completa} \\"
           "\n" r"\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-7}\cmidrule(lr){8-9}" "\n"
           r" & $\kappa$ & $\Delta$RMSE & $\Delta$RMSE & $\kappa$ & época $0$ & \shortstack{teste: melhor/\\igual/pior} & $\Delta$RMSE & $\kappa$ \\"
           "\n\\midrule\n")
    (TAB / "tab_v3_variantes.tex").write_text(cab + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    NUM["variantes"] = {str(k): v for k, v in num.items()}
    # as três maiores perdas de teste da NFH segura em cada K (citadas na Seção 5.5)
    NUM["variantes_piores"] = {str(k): sorted(v, key=lambda r: -r["rel_teste"])[:3] for k, v in piores.items()}
    return num


def tab_xai():
    f = RES3 / "xai_resumo.csv"
    if not f.exists():
        return None
    r = pd.read_csv(f)
    ordem = ["KernelSHAP", "LIME", "Integrated Gradients", "Saliency", "Feature Ablation"]
    # A explicação estrutural (eps_T, e_1..e_K, eps_R) não ordena entradas, logo o AOPC não se aplica a ela;
    # seus números (tempo e robustez) vão para o texto, e a tabela traz só atribuições de entrada.
    est = r[r.metodo == "NFH (estrutural)"].iloc[0]
    linhas = []
    num = {"NFH (estrutural)": {"tempo_ms": 1e3 * est.tempo_media, "tempo_dp_ms": 1e3 * est.tempo_dp,
                                "robustez": est.rob_media, "robustez_dp": est.rob_dp}}
    for m in ordem:
        gT = r[(r.metodo == m) & (r.funcional == "g_T")].iloc[0]
        gS = r[(r.metodo == m) & (r.funcional == "g_S")].iloc[0]
        rotulo = m + (r"$^{a}$" if m == "Integrated Gradients" else "")
        pm = lambda a, b, nd: n(a, nd)[:-1] + r" \pm " + n(b, nd)[1:]  # noqa: E731  "$a \pm b$"
        ms = lambda a, b: "$" + f"{a:,.0f}".replace(",", ".") + r" \pm " + f"{b:,.0f}".replace(",", ".") + "$"  # noqa: E731
        linhas.append(f"{rotulo} & {ms(1e3 * gT.tempo_media, 1e3 * gT.tempo_dp)} & {pm(gT.rob_media, gT.rob_dp, 2)} & "
                      f"{pm(gT.aopc_media, gT.aopc_dp, 2)} & {pm(gS.rob_media, gS.rob_dp, 2)} & "
                      f"{pm(gS.aopc_media, gS.aopc_dp, 2)} \\\\")
        num[m] = {"g_T": gT.to_dict(), "g_S": gS.to_dict()}
    cab = (r"\begin{tabular}{lccccc}" "\n\\toprule\n"
           r" & & \multicolumn{2}{c}{Nível previsto $g_T$} & \multicolumn{2}{c}{Energia sazonal $g_S$} \\"
           "\n" r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}" "\n"
           r"Método & Tempo (ms) $\downarrow$ & Robustez $\uparrow$ & AOPC rel. $\uparrow$ & Robustez $\uparrow$ & AOPC rel. $\uparrow$ \\"
           "\n\\midrule\n")
    (TAB / "tab_v3_xai.tex").write_text(cab + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    NUM["xai"] = num
    return num


def tab_estabilidade():
    f = RES3 / "estabilidade.json"
    if not f.exists():
        return None
    e = json.loads(f.read_text())
    linhas, num = [], {}
    for conj in CONJUNTOS:
        if conj not in e or "N-BEATS-I nativo" not in e[conj]:
            continue
        nat, prj = e[conj]["N-BEATS-I nativo"], e[conj]["N-BEATS-I projetado"]
        nfh = e[conj].get("Conv1D+LSTM + NFH", {})
        den = e[conj].get("Conv1D+LSTM + densa (decomposição pura)", {})
        dp = lambda r_: n(r_.get("mediana_dp_entre_sementes", np.nan), 1, pct=True)  # noqa: E731
        linhas.append(f"{rot(conj)} & {n(nat['mediana_valor'], 1, pct=True)} & {n(prj['mediana_valor'], 1, pct=True)} & "
                      f"{n(nat['fracao_fora_0_1'], 1, pct=True)} & {dp(nat)} & {dp(prj)} & {dp(nfh)} & {dp(den)} \\\\")
        num[conj] = {"nativo": nat, "projetado": prj, "nfh_conv": nfh, "densa_conv": den}
    cab = (r"\begin{tabular}{lccccccc}" "\n\\toprule\n"
           r" & \multicolumn{5}{c}{N-BEATS-I (mesmas previsões)} & \multicolumn{2}{c}{Conv1D + LSTM} \\"
           "\n" r"\cmidrule(lr){2-6}\cmidrule(lr){7-8}" "\n"
           r"Conjunto & $\rho_T$ & $\varepsilon_T$ & $\rho_T \notin [0, 1]$ & DP $\rho_T$ & DP $\varepsilon_T$ & DP NFH & DP densa \\"
           "\n\\midrule\n")
    (TAB / "tab_v3_estabilidade.tex").write_text(cab + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    NUM["estabilidade"] = num
    # pares (conjunto, backbone) em que a dispersão da NFH foi menor que a da decomposição pura (Seção 5.8)
    menor, total = 0, 0
    for conj in CONJUNTOS:
        for bb in BACKBONES:
            a, b = e.get(conj, {}).get(f"{bb} + NFH"), e.get(conj, {}).get(f"{bb} + densa (decomposição pura)")
            if a and b:
                total += 1
                menor += int(a["mediana_dp_entre_sementes"] < b["mediana_dp_entre_sementes"])
    # dispersão relativa ao próprio nível (desvio mediano entre sementes / mediana da participação), Seção 5.8
    rel_nat = [e[c]["N-BEATS-I nativo"]["mediana_dp_entre_sementes"] / e[c]["N-BEATS-I nativo"]["mediana_valor"]
               for c in CONJUNTOS if c in e and "N-BEATS-I nativo" in e[c]]
    rel_prj = [e[c]["N-BEATS-I projetado"]["mediana_dp_entre_sementes"] / e[c]["N-BEATS-I projetado"]["mediana_valor"]
               for c in CONJUNTOS if c in e and "N-BEATS-I projetado" in e[c]]
    NUM["estabilidade_resumo"] = {"pares": total, "nfh_menos_dispersa": menor,
                                  "dp_relativo_nativo": [float(min(rel_nat)), float(max(rel_nat))] if rel_nat else None,
                                  "dp_relativo_projetado": [float(min(rel_prj)), float(max(rel_prj))] if rel_prj else None}
    return num


def tab_sint_m():
    f = RES3 / "sint_m.json"
    if not f.exists():
        return None
    s = json.loads(f.read_text())
    met = {}
    for sd in s["sementes"].values():
        for m, v in sd["metodos"].items():
            met.setdefault(m, []).append(v)
    ordem = ["NFH | IG sobre g_T e g_S", "densa + projeção | IG sobre g_T e g_S", "NFH | KernelSHAP sobre g_T e g_S",
             "NFH | Feature Ablation sobre g_T e g_S", "densa | IG sobre a média", "densa | KernelSHAP sobre a média",
             "densa | LIME sobre a média", "densa | Saliency sobre a média", "densa | Feature Ablation sobre a média",
             "densa | IG sobre a energia da forma"]
    nomes = {"NFH | IG sobre g_T e g_S": r"NFH, IG sobre $g_T$ e $g_S$",
             "densa + projeção | IG sobre g_T e g_S": r"Densa + projeção, IG sobre $g_T$ e $g_S$",
             "NFH | KernelSHAP sobre g_T e g_S": r"NFH, KernelSHAP sobre $g_T$ e $g_S$$^{a}$",
             "NFH | Feature Ablation sobre g_T e g_S": r"NFH, Feature Ablation sobre $g_T$ e $g_S$$^{a}$",
             "densa | IG sobre a média": r"Densa, IG sobre a média",
             "densa | KernelSHAP sobre a média": r"Densa, KernelSHAP sobre a média$^{a}$",
             "densa | LIME sobre a média": r"Densa, LIME sobre a média$^{a}$",
             "densa | Saliency sobre a média": r"Densa, Saliency sobre a média$^{a}$",
             "densa | Feature Ablation sobre a média": r"Densa, Feature Ablation sobre a média$^{a}$",
             "densa | IG sobre a energia da forma": r"Densa, IG sobre $\norm{\byh_c}^2$"}
    linhas, num = [], {}
    for m in ordem:
        if m not in met:
            continue
        v = met[m]
        at = np.array([x["acerto_tendencia"] for x in v])
        ac = np.array([x["acerto_ciclo"] for x in v])
        p1 = np.array([x["participacao_x1_na_tendencia"] for x in v])
        p2 = np.array([x["participacao_x2_no_ciclo"] for x in v])
        fmt = (lambda a: n(a.mean(), 0, pct=True) + (r" $\pm$ " + n(a.std(ddof=1), 0, pct=True) if len(a) > 1 else ""))
        linhas.append(f"{nomes[m]} & {fmt(at)} & {fmt(ac)} & {fmt(p1)} & {fmt(p2)} \\\\")
        num[m] = {"acerto_tendencia": float(at.mean()), "acerto_ciclo": float(ac.mean()),
                  "part_x1_T": float(p1.mean()), "part_x2_S": float(p2.mean()), "n_sementes": len(v)}
    cab = (r"\begin{tabular}{p{6.0cm}cccc}" "\n\\toprule\n"
           r"Método & \shortstack{Acerto\\tend. (\%)} & \shortstack{Acerto\\ciclo (\%)} & "
           r"\shortstack{$\pi(x_1)$ na\\tend. (\%)} & \shortstack{$\pi(x_2)$ no\\ciclo (\%)} \\"
           "\n\\midrule\n")
    (TAB / "tab_v3_sint_m.tex").write_text(cab + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    NUM["sint_m"] = {"config": s["config"], "metodos": num,
                     "rmse": {k: {"densa": v["RMSE_densa"], "nfh": v["RMSE_nfh"], "kappa_nfh": v["kappa_nfh"]}
                              for k, v in s["sementes"].items()}}
    return num


def tab_abdutivas():
    f = RES3 / "abdutivas.json"
    if not f.exists():
        return None
    a = json.loads(f.read_text())
    linhas, num = [], {}
    for bb in BACKBONES:
        regs = [v for k, v in a.items() if k.endswith("|" + bb)]
        if not regs:
            continue
        p = regs[0]["p"]
        med = lambda chave, prop: float(np.median([r[prop][chave] for r in regs]))  # noqa: E731
        num[bb] = {"p": p, "n_conjuntos": len(regs),
                   "P_T_fracao_p": med("fracao_p_mediana", "P_T"), "P_S_fracao_p": med("fracao_p_mediana", "P_S"),
                   "P_T_razao_IG": med("razao_kIG_k_media", "P_T"), "P_S_razao_IG": med("razao_kIG_k_media", "P_S"),
                   "P_T_topk_IG": med("fracao_topk_ig_suficiente", "P_T"), "P_S_topk_IG": med("fracao_topk_ig_suficiente", "P_S"),
                   "P_T_trivial": med("fracao_trivial", "P_T"), "P_S_trivial": med("fracao_trivial", "P_S"),
                   "tempo_ms": float(np.median([r["tempo_medio_ms"] for r in regs])),
                   "iguais_min": float(min(r["explicacoes_iguais_densa_proj"] for r in regs)),
                   "violacoes": int(sum(r["violacoes_amostragem"] for r in regs)),
                   "fora_caixa_max": float(max(r["fracao_h_fora_caixa_treino"] for r in regs))}
        u = num[bb]
        linhas.append(f"{bb} & ${p}$ & {n(u['P_T_fracao_p'], 1, pct=True)} & {n(u['P_T_razao_IG'], 2)} & "
                      f"{n(u['P_S_fracao_p'], 1, pct=True)} & {n(u['P_S_razao_IG'], 2)} & {n(u['tempo_ms'], 3)} \\\\")
    cab = (r"\begin{tabular}{lcccccc}" "\n\\toprule\n"
           r" & & \multicolumn{2}{c}{Tendência ($P_T$)} & \multicolumn{2}{c}{Ciclo diário ($P_S$)} & \\"
           "\n" r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}" "\n"
           r"\textit{Backbone} & $p$ & $|\mathcal{A}|/p$ & $k_{\mathrm{IG}}/|\mathcal{A}|$ & $|\mathcal{A}|/p$ & $k_{\mathrm{IG}}/|\mathcal{A}|$ & Tempo (ms) \\"
           "\n\\midrule\n")
    (TAB / "tab_v3_abdutivas.tex").write_text(cab + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    NUM["abdutivas"] = num
    return num


def tab_jena():
    f = RES3 / "exemplo_jena.json"
    if not f.exists():
        return None
    e = json.loads(f.read_text())
    nomes = e["variaveis"]
    rotulos = {"p": "$p$ (pressão)", "rh": "$rh$ (umidade relativa)", "Wx": "$W_x$ (vento, leste)",
               "Wy": "$W_y$ (vento, norte)", "T": "$T$ (temperatura, alvo)"}
    linhas = []
    for v in nomes:
        cel = [n(e["atribuicao_ig"]["T"]["participacao_pct"][v] / 100, 1, pct=True),
               n(e["atribuicao_ig"]["S"]["participacao_pct"][v] / 100, 1, pct=True)]
        for m in ("KernelSHAP", "LIME", "Feature Ablation"):
            cel += [n(e["outros_metodos"]["T"][m][v] / 100, 1, pct=True), n(e["outros_metodos"]["S"][m][v] / 100, 1, pct=True)]
        cel += [n(e["agregado_50"]["T"]["media"][v] / 100, 1, pct=True), n(e["agregado_50"]["S"]["media"][v] / 100, 1, pct=True)]
        linhas.append(f"{rotulos.get(v, v)} & " + " & ".join(cel) + r" \\")
    cab = (r"\begin{tabular}{lcccccccccc}" "\n\\toprule\n"
           r" & \multicolumn{2}{c}{NFH (IG)} & \multicolumn{2}{c}{KernelSHAP} & \multicolumn{2}{c}{LIME} & "
           r"\multicolumn{2}{c}{Feat. Ablation} & \multicolumn{2}{c}{IG, 50 janelas} \\"
           "\n" r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}" "\n"
           r"Variável & $g_T$ & $g_S$ & $g_T$ & $g_S$ & $g_T$ & $g_S$ & $g_T$ & $g_S$ & $g_T$ & $g_S$ \\" "\n\\midrule\n")
    (TAB / "tab_v3_jena.tex").write_text(cab + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    NUM["jena"] = e
    return e


def tab_conjuntos(df):
    """Descrição dos conjuntos; janelas de treino/validação/teste lidas dos resultados."""
    info = [("ETTh1", "OT (temperatura do óleo)", "HUFL, HULL, MUFL, MULL, LUFL, LULL", "01/07/2016 a 20/02/2018", "12/4/4 meses"),
            ("ETTh2", "OT (temperatura do óleo)", "as mesmas seis cargas do ETTh1", "01/07/2016 a 20/02/2018", "12/4/4 meses"),
            ("Jena", "T (temperatura do ar)", "p, rh, $W_x$, $W_y$", "01/01/2009 a 01/01/2012", "70/15/15\\%"),
            ("AirQ", "C$_6$H$_6$ (benzeno)", "S1, S3, S4, S5, T, RH, AH", "10/03/2004 a 04/04/2005", "70/15/15\\%")]
    linhas = []
    for f, alvo, cov, per, div in info:
        s = df[df.conjunto == f"{f}-U"]
        jan = (f"{int(s.n_treino.iloc[0]):,}/{int(s.n_val.iloc[0]):,}/{int(s.n_teste.iloc[0]):,}".replace(",", ".")
               if len(s) else "--")
        linhas.append(f"{ROT[f]} & {alvo} & {cov} & {per} & {div} & {jan} \\\\")
    rr = r">{\raggedright\arraybackslash}"   # colunas alinhadas à esquerda, sem espaços esticados
    cab = (r"\begin{tabular}{" + "".join(f"{rr}p{{{w}cm}}" for w in (1.5, 2.3, 2.7, 2.2, 1.7, 2.9)) + "}"
           "\n\\toprule\n"
           r"Fonte & Alvo & Covariáveis (conjunto M) & Período & Divisão & Janelas (treino/val./teste) \\" "\n\\midrule\n")
    (TAB / "tab_v3_conjuntos.tex").write_text(cab + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def tab_parametros(df):
    """Parâmetros treináveis por modelo: d = 1 (conjuntos U) e cada conjunto M."""
    colunas = [("ETTh1-U", r"\shortstack{U\\($d = 1$)}"), ("ETTh1-M", r"\shortstack{ETT-M\\($d = 7$)}"),
               ("Jena-M", r"\shortstack{Jena-M\\($d = 5$)}"), ("AirQ-M", r"\shortstack{Air Quality-M\\($d = 8$)}")]
    modelos = [f"{bb} + {cab}" for bb in reversed(BACKBONES) for cab in ("NFH", "densa")] + ["N-BEATS-I"]
    rot_mod = {m: m for m in modelos}
    linhas, num = [], {}
    for mod in modelos:
        cel = []
        for conj, _ in colunas:
            s = df[(df.conjunto == conj) & (df.modelo == mod) & df.params.notna() & (df.params != "")]
            v = int(float(s.params.iloc[0])) if len(s) else None
            num[f"{mod}|{conj}"] = v
            cel.append(f"{v:,}".replace(",", ".") if v is not None else "--")
        linhas.append(f"{rot_mod[mod]} & " + " & ".join(cel) + r" \\")
    cab = (r"\begin{tabular}{lrrrr}" "\n\\toprule\n" "Modelo & " + " & ".join(c for _, c in colunas) + r" \\" "\n\\midrule\n")
    (TAB / "tab_v3_parametros.tex").write_text(cab + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    NUM["parametros"] = num
    return num


def piloto():
    """Números do piloto descartado (results/v3_piloto/), citados na justificativa do protocolo."""
    f = ROOT / "results" / "v3_piloto" / "benchmark_runs.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    out = {}
    for bb in ("MLP", "LSTM"):
        nf = d[(d.modelo == f"{bb} + NFH")].set_index("seed")
        de = d[(d.modelo == f"{bb} + densa")].set_index("seed")
        rel = ((nf.RMSE - de.RMSE) / de.RMSE).dropna()
        out[bb] = {"n": int(len(rel)), "epocas": nf.epoca_ajuste.astype(int).tolist(),
                   "rel_teste": rel.tolist(), "max_rel": float(rel.max()), "n_pior": int((rel > 0).sum())}
    eps = out["MLP"]["epocas"] + out["LSTM"]["epocas"]
    out["n_epoca_maior_igual_13"] = int(sum(e >= 13 for e in eps))
    out["n_total"] = len(eps)
    NUM["piloto"] = out
    return out


def contagens(df):
    out = {}
    for conj in CONJUNTOS:
        s = df[df.conjunto == conj]
        if len(s):
            r = s.iloc[0]
            out[conj] = {"n_treino": int(r.n_treino), "n_val": int(r.n_val), "n_teste": int(r.n_teste),
                         "n_teste_total": int(r.n_teste_total), "d": int(r.d)}
    p = {}
    for bb in BACKBONES + ["N-BEATS-I"]:
        for modo in ("U", "M"):
            s = df[(df.modo == modo) & (df.backbone == bb) & df.params.notna()]
            if len(s):
                p[f"{bb}|{modo}"] = sorted({int(x) for x in s.params})
    out["parametros"] = p
    NUM["contagens"] = out
    return out


def main():
    df = carregar_runs()
    contagens(df)
    tab_conjuntos(df)
    tab_parametros(df)
    piloto()
    tab_acuracia(df)
    comparacoes(df)
    tab_variantes(df)
    for fn in (tab_xai, tab_estabilidade, tab_sint_m, tab_abdutivas, tab_jena):
        fn()
    (RES3 / "numeros_cap5.json").write_text(json.dumps(NUM, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({k: NUM[k] for k in ("comparacoes",) if k in NUM}, indent=1, ensure_ascii=False, default=float)[:4000])
    print(json.dumps(NUM.get("variantes", {}), indent=1, default=float))


if __name__ == "__main__":
    main()
