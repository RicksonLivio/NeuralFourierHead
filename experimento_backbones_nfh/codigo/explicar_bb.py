#!/usr/bin/env python3
"""
explicar_bb.py: explicações das combinações do experimento de backbones em que a NFH teve SUCESSO na confirmação
(critério do §6 do protocolo), para a semente 10.

Para cada combinação (SUCESSO em resultados/confirmacao.csv; e, opcionalmente, as passadas em --lista) o script:
  1. carrega os modelos salvos pela confirmação (resultados/modelos/<stem>_densa.pt e <stem>_<variante NFH>.pt, semente 10),
     reconstrói os dados com dados_bb (mesmo carregador, mesma divisão) e confere a previsão da NFH contra
     <stem>_<variante>_pred.npy quando ele existe;
  2. no TESTE: decomposição T, S, R da NFH (da própria cabeça), participações eps_T, eps_S, eps_R (mediana e quartis),
     amplitudes por harmônico (mediana) e a decomposição PURA da densa da mesma semente pelo Projetor (v3_common),
     para comparar;
  3. Integrated Gradients (captum, referência 0, 64 passos, 50 janelas de teste igualmente espaçadas) de
     g_T = média de T (nível previsto) e g_S = ||S||^2, somados por variável (modo M) ou por bloco de 24 h de defasagens
     (modo U), com o erro de completude |soma das atribuições - (g(x) - g(0))|;
  4. SINT-M: acerto por variável contra a verdade (x1 conduz a tendência, x2 conduz o ciclo, como em
     src/v3_sintetico_multivariado.py). ATENÇÃO: aqui d = 5 e a ÚLTIMA coluna é o próprio y passado (exigência do
     catálogo; na v3 era d = 4, sem y). Como y_{t-48} = x1_{t-48} + x2_{t-48} + ruído, o y passado carrega a mesma
     informação, e por isso o acerto é reportado de duas formas: com as 5 colunas e só com as 4 covariáveis;
  5. SINT-U48: decomposição do sinal SEM RUÍDO do futuro (dados/SINT-U48_com_sinal.csv, coluna sinal_sem_ruido,
     padronizado com a média e o desvio do treino) pelas mesmas fórmulas, e o erro da NFH (e da densa projetada) contra ela.
Grava resultados/explicacoes/<combinação>.json e <combinação>.png (cinza; uma janela de teste: previsão, T, S, R e, nos
sintéticos, a verdade). CPU com 2 threads.

Uso (raiz do repositório):
  CUDA_VISIBLE_DEVICES='' .venv/bin/python experimento_backbones_nfh/codigo/explicar_bb.py
  CUDA_VISIBLE_DEVICES='' .venv/bin/python experimento_backbones_nfh/codigo/explicar_bb.py --lista 'SINT-U48|U|MLP|w1|f0.25'
  ... --res /tmp/fake_bb/res            (teste com dados falsos)
  ... --janelas 50 --passos 64 --max-combinacoes 3
Identificador da combinação: dataset|modo|backbone|w<largura>|f<fração>  (como em numeros.json do relatório).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
B = ROOT / "experimento_backbones_nfh"
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "teste_exemplificacao_nfh" / "codigo"))

import relatorio as rel  # noqa: E402  (leitura dos CSV e veredito; não importa torch)
from backbones import construir_modelo  # noqa: E402
from busca import prever  # noqa: E402
from dados_bb import carregar_bb, segmentos_bb  # noqa: E402
from v2_common import stack_ds  # noqa: E402
from v3_common import H3, L3, Q3, Projetor  # noqa: E402

SEED = 10
UM_DIA = 24


def id_arquivo_comb(c) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", c["id"])


def stem_tarefa(c, seed=SEED) -> str:
    """Mesmo nome que executar.id_arquivo: '<dataset>|<modo>|<backbone>|w<l>|f<fração>|s<seed>' com [^A-Za-z0-9_.+-] -> '_'."""
    s = f"{c['dataset']}|{c['modo']}|{c['backbone']}|w{c['largura']}|f{c['fracao']:g}|s{seed}"
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", s)


def resumo(x) -> dict:
    x = np.asarray(x, dtype=float)
    q1, md, q3 = np.percentile(x, [25, 50, 75])
    return {"mediana": float(md), "q1": float(q1), "q3": float(q3)}


def eps_de(forecast, T, S, R) -> dict:
    """Participações de energia da previsão centrada: eps_T, eps_S, eps_R (mesmas fórmulas de Projetor.partes)."""
    Yc = forecast - forecast.mean(1, keepdims=True)
    E = (Yc ** 2).sum(1) + 1e-30
    Tc = T - T.mean(1, keepdims=True)
    return {"eps_T": (Tc ** 2).sum(1) / E, "eps_S": (S ** 2).sum(1) / E, "eps_R": (R ** 2).sum(1) / E}


def amplitudes(theta_S):
    A, Bc = theta_S[:, 0::2], theta_S[:, 1::2]
    return np.sqrt(A ** 2 + Bc ** 2)


@torch.no_grad()
def saidas_nfh(modelo, X, bs=512):
    ch = {k: [] for k in ("forecast", "trend", "seasonal", "residual", "amplitude")}
    for i in range(0, len(X), bs):
        o = modelo(X[i:i + bs])
        for k in ch:
            ch[k].append(o[k].double().numpy())
    return {k: np.concatenate(v) for k, v in ch.items()}


class Funcional(torch.nn.Module):
    """Saída escalar a explicar, em função da janela achatada (L*d): g_T = média de T; g_S = ||S||^2."""

    def __init__(self, m, qual, L, d):
        super().__init__()
        self.m, self.qual, self.L, self.d = m, qual, L, d

    def forward(self, flat):
        o = self.m(flat.reshape(flat.shape[0], self.L, self.d))
        if self.qual == "T":
            return o["trend"].mean(1, keepdim=True)
        return (o["seasonal"] ** 2).sum(1, keepdim=True)


def grupos(modo, d, nomes):
    """Índices (sobre a janela achatada L*d, ordem t-major) e rótulos dos grupos: por variável (M) ou bloco de 24 h (U)."""
    if modo == "M" and d > 1:
        return [np.arange(j, L3 * d, d) for j in range(d)], [str(n) for n in nomes]
    idx, rot = [], []
    for b in range(L3 // UM_DIA):
        pos = np.arange(UM_DIA * b, UM_DIA * (b + 1))
        idx.append(pos * d + (d - 1))
        rot.append(f"defasagens {L3 - UM_DIA * (b + 1) + 1} a {L3 - UM_DIA * b} h")
    return idx, rot


def ig_janelas(modelo, Xte, janelas, idx_g, passos, d):
    """IG de g_T e g_S; devolve atribuições por grupo (n_janelas x G, com sinal) e os erros de completude."""
    from captum.attr import IntegratedGradients
    res = {}
    for qual in ("T", "S"):
        f = Funcional(modelo, qual, L3, d).eval()
        ig = IntegratedGradients(f)
        A = np.zeros((len(janelas), len(idx_g)))
        err, rel_err, dg = [], [], []
        for n, w in enumerate(janelas):
            x = Xte[w].reshape(1, -1).clone()
            base = torch.zeros_like(x)
            at = ig.attribute(x, baselines=base, n_steps=passos, internal_batch_size=passos)
            at = at[0].detach().numpy().astype(np.float64)
            with torch.no_grad():
                gx, g0 = float(f(x)), float(f(base))
            for g, ix in enumerate(idx_g):
                A[n, g] = at[ix].sum()
            err.append(abs(float(at.sum()) - (gx - g0)))
            dg.append(abs(gx - g0))
            rel_err.append(err[-1] / (dg[-1] + 1e-12))
        v = np.abs(A)
        part = v / (v.sum(1, keepdims=True) + 1e-12)
        res[qual] = {"atribuicao": A, "abs_mediana": np.median(v, 0), "participacao_mediana": np.median(part, 0),
                     "completude_erro_mediano": float(np.median(err)), "completude_erro_max": float(np.max(err)),
                     "completude_rel_mediano": float(np.median(rel_err)), "completude_rel_max": float(np.max(rel_err)),
                     "delta_g_mediano": float(np.median(dg))}
    return res


def verdade_sint_u48(ds, te, proj, nfh_o, dens_dec):
    """Decomposição do sinal sem ruído do futuro e erro da NFH (e da densa projetada) contra ela."""
    csv_ = B / "dados" / "SINT-U48_com_sinal.csv"
    if not csv_.exists():
        return {"erro": f"{csv_} não existe"}
    df = pd.read_csv(csv_)
    mu, sd = ds["mu_alvo"], ds["sd_alvo"]
    y_z = (df["y"].to_numpy() - mu) / sd
    s_z = (df["sinal_sem_ruido"].to_numpy() - mu) / sd
    desvio = float(np.abs(y_z - ds["alvo"]).max())
    b = ds["cortes"][1]
    fut = np.stack([s_z[b + te.idx[j] + L3:b + te.idx[j] + L3 + H3] for j in range(len(te))])
    Tt, St, Rt = proj.decompor(fut)
    eps_t = eps_de(fut, Tt, St, Rt)
    amp_t = amplitudes(fut @ proj.thS.T)

    def erro(prev, Tp, Sp, Rp, amp):
        r = lambda a_, b_: float(np.sqrt(np.mean((a_ - b_) ** 2)))
        return {"RMSE_previsao_vs_sinal": r(prev, fut), "RMSE_T_vs_T_verdade": r(Tp, Tt), "RMSE_S_vs_S_verdade": r(Sp, St),
                "RMSE_R_vs_R_verdade": r(Rp, Rt),
                "erro_relativo_S_mediano": float(np.median(np.linalg.norm(Sp - St, axis=1) / (np.linalg.norm(St, axis=1) + 1e-12))),
                "erro_abs_amplitude_por_harmonico_mediano": np.median(np.abs(amp - amp_t), 0).tolist()}
    out = {"alinhamento_max_abs_y_csv_vs_dados": desvio, "n_janelas_teste": int(len(te)),
           "verdade": {"eps_T": resumo(eps_t["eps_T"]), "eps_S": resumo(eps_t["eps_S"]), "eps_R": resumo(eps_t["eps_R"]),
                       "amplitude_mediana_por_harmonico": np.median(amp_t, 0).tolist()},
           "nfh": erro(nfh_o["forecast"], nfh_o["trend"], nfh_o["seasonal"], nfh_o["residual"], nfh_o["amplitude"]),
           "densa_projetada": erro(dens_dec["forecast"], dens_dec["T"], dens_dec["S"], dens_dec["R"], dens_dec["amplitude"])}
    orj = rel.ler_json(B / "dados" / "sint_u48_oraculo.json")
    if orj:
        out["referencia_amostral_RMSE_oraculo_janelas_teste"] = orj.get("RMSE_oraculo_padronizado_janelas_teste")
    return out, fut, (Tt, St, Rt)


def figura(caminho, jan, Yobs, nfh_o, dens_dec, verdade=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10})
    h = np.arange(H3)
    fig, ax = plt.subplots(4, 1, figsize=(7.0, 8.0), sharex=True)
    est = {"nfh": dict(color="#000000", ls="-", marker="o", ms=3, lw=1.2, label="NFH"),
           "densa": dict(color="#666666", ls="--", marker="s", ms=3, lw=1.1, label="densa (projetada)"),
           "ver": dict(color="#aaaaaa", ls="-", lw=2.6, label="verdade (sinal sem ruído)"),
           "obs": dict(color="#888888", ls=":", marker=".", ms=4, lw=1.0, label="alvo observado")}
    ax[0].plot(h, Yobs[jan], **est["obs"])
    if verdade is not None:
        ax[0].plot(h, verdade[0][jan], **est["ver"])
    ax[0].plot(h, dens_dec["forecast"][jan], **est["densa"])
    ax[0].plot(h, nfh_o["forecast"][jan], **est["nfh"])
    ax[0].set_ylabel("(a) previsão")
    for k, (rot, nn, dd, vv) in enumerate([("(b) T", "trend", "T", 0), ("(c) S", "seasonal", "S", 1), ("(d) R", "residual", "R", 2)], 1):
        if verdade is not None:
            ax[k].plot(h, verdade[1][vv][jan], **est["ver"])
        ax[k].plot(h, dens_dec[dd][jan], **est["densa"])
        ax[k].plot(h, nfh_o[nn][jan], **est["nfh"])
        ax[k].set_ylabel(rot)
    ax[3].set_xlabel("passo do horizonte (h)")
    ax[0].legend(frameon=False, fontsize=8, ncol=2, loc="best")
    for a_ in ax:
        a_.grid(color="#dddddd", lw=0.5)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def explicar(c, res: Path, saida: Path, n_jan, passos) -> dict:
    t0 = time.time()
    var = c["variante_nfh"]
    fam, lam, K = rel.parse_variante(var)
    stem = stem_tarefa(c)
    mdir = res / "modelos"
    arq_d, arq_n = mdir / f"{stem}_densa.pt", mdir / f"{stem}_{var}.pt"
    base = {"combinacao": id_arquivo_comb(c), "id": c["id"], "dataset": c["dataset"], "modo": c["modo"], "backbone": c["backbone"],
            "largura": c["largura"], "fracao": c["fracao"], "variante_nfh": var, "K": K, "lambda": lam, "seed": SEED,
            "veredito_confirmacao": c["veredito"]}
    faltam = [str(p) for p in (arq_d, arq_n) if not p.exists()]
    if faltam:
        return {**base, "erro": "modelo(s) ausente(s): " + ", ".join(faltam)}
    ds = carregar_bb(f"{c['dataset']}-{c['modo']}")
    tr, va, te = segmentos_bb(ds)
    Xte, Yte = stack_ds(te)
    d = ds["feats"].shape[1]
    densa = construir_modelo(c["backbone"], "densa", d, c["largura"])
    densa.load_state_dict(torch.load(arq_d, map_location="cpu"))
    nfh = construir_modelo(c["backbone"], "nfh", d, c["largura"], K=K)
    nfh.load_state_dict(torch.load(arq_n, map_location="cpu"))
    for m in (densa, nfh):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    proj = Projetor(H3, K, Q3)
    Y = Yte.double().numpy()
    o = saidas_nfh(nfh, Xte)
    pd_ = prever(densa, Xte, torch.device("cpu"))
    Td, Sd, Rd = proj.decompor(pd_)
    amp_d = amplitudes(pd_ @ proj.thS.T)
    dens_dec = {"forecast": pd_, "T": Td, "S": Sd, "R": Rd, "amplitude": amp_d}
    conf = {"max_abs_T+S+R_menos_previsao": float(np.abs(o["trend"] + o["seasonal"] + o["residual"] - o["forecast"]).max())}
    Tp, Sp, Rp = proj.decompor(o["forecast"])
    conf["max_abs_cabeca_vs_projetor"] = float(max(np.abs(Tp - o["trend"]).max(), np.abs(Sp - o["seasonal"]).max(),
                                                   np.abs(Rp - o["residual"]).max()))
    parq = mdir / f"{stem}_{var}_pred.npy"
    if parq.exists():
        conf["max_abs_previsao_vs_pred_npy_salvo"] = float(np.abs(np.load(parq).astype(np.float64) - o["forecast"]).max())
    rm = lambda P: float(np.sqrt(np.mean((P - Y) ** 2)))
    ma = lambda P: float(np.mean(np.abs(P - Y)))
    metricas = {"nfh": {"test_RMSE": rm(o["forecast"]), "test_MAE": ma(o["forecast"])},
                "densa": {"test_RMSE": rm(pd_), "test_MAE": ma(pd_)}}
    e_n = eps_de(o["forecast"], o["trend"], o["seasonal"], o["residual"])
    e_p = proj.partes(o["forecast"])
    e_d = proj.partes(pd_)
    out = {**base, "d": int(d), "n_janelas_teste": int(len(Xte)), "periodo_h_por_harmonico": [H3 / k for k in range(1, K + 1)],
           "nfh": {**{k: resumo(v) for k, v in e_n.items()},
                   "amplitude_mediana_por_harmonico": np.median(o["amplitude"], 0).tolist(),
                   "eps_via_projetor": {k: resumo(e_p[k]) for k in ("eps_T", "eps_S", "eps_R")},
                   "cobertura_mediana": float(np.median(1 - e_n["eps_R"]))},
           "densa_projetada": {**{k: resumo(e_d[k]) for k in ("eps_T", "eps_S", "eps_R")},
                               "amplitude_mediana_por_harmonico": np.median(amp_d, 0).tolist(),
                               "cobertura_mediana": float(np.median(1 - e_d["eps_R"]))},
           "metricas_teste_semente10": metricas, "conferencias": conf}
    janelas = np.unique(np.linspace(0, len(Xte) - 1, n_jan).astype(int))
    idx_g, rot = grupos(c["modo"], d, ds["nomes"])
    ig = ig_janelas(nfh, Xte, janelas, idx_g, passos, d)
    out["ig"] = {"n_janelas": int(len(janelas)), "n_passos": passos, "baseline": 0.0, "janelas_teste_indices": janelas.tolist(),
                 "agrupamento": ("por variável (soma sobre as 96 defasagens)" if (c["modo"] == "M" and d > 1)
                                 else "por bloco de 24 h de defasagens"), "grupos": rot}
    for q, nome in (("T", "g_T"), ("S", "g_S")):
        r = ig[q]
        out["ig"][nome] = {"abs_mediana_por_grupo": r["abs_mediana"].tolist(),
                           "participacao_mediana_por_grupo": r["participacao_mediana"].tolist(),
                           **{k: r[k] for k in ("completude_erro_mediano", "completude_erro_max", "completude_rel_mediano",
                                                "completude_rel_max", "delta_g_mediano")}}
    verdade_fig = None
    if c["dataset"] == "SINT-M" and c["modo"] == "M" and d == 5:
        g = {}
        for rot_q, q, certo in (("tendencia", "T", 0), ("ciclo", "S", 1)):
            v = np.abs(ig[q]["atribuicao"])
            p5, p4 = v / (v.sum(1, keepdims=True) + 1e-12), v[:, :4] / (v[:, :4].sum(1, keepdims=True) + 1e-12)
            g[rot_q] = {"variavel_correta": rot[certo],
                        "acerto_5_colunas": float(np.mean(v.argmax(1) == certo)),
                        "acerto_4_covariaveis": float(np.mean(v[:, :4].argmax(1) == certo)),
                        "participacao_correta_5_colunas": float(np.mean(p5[:, certo])),
                        "participacao_correta_4_covariaveis": float(np.mean(p4[:, certo])),
                        "participacao_y_passado": float(np.mean(p5[:, 4]))}
        out["verdade_sint_m"] = {
            "verdade": "y_t = x1_{t-48} + x2_{t-48} + ruído AR(1): x1 conduz a tendência, x2 conduz o ciclo (src/v3_sintetico_multivariado.py)",
            "aviso_d5": "Aqui d = 5 e a última coluna é o próprio y passado (o catálogo exige o alvo entre as variáveis); na v3, d = 4, sem y. "
                        "Como o y passado é x1 + x2 defasados mais ruído, ele pode absorver atribuição: por isso o acerto é reportado "
                        "com as 5 colunas e só com as 4 covariáveis.", **g}
    if c["dataset"] == "SINT-U48" and c["modo"] == "U":
        r = verdade_sint_u48(ds, te, proj, o, dens_dec)
        if isinstance(r, tuple):
            out["verdade_sint_u48"], fut, dec_t = r
            verdade_fig = (fut, dec_t)
        else:
            out["verdade_sint_u48"] = r
    saida.mkdir(parents=True, exist_ok=True)
    nome_fig = f"{id_arquivo_comb(c)}.png"
    j_fig = int(janelas[len(janelas) // 2])
    figura(saida / nome_fig, j_fig, Y, o, dens_dec, verdade_fig)
    out["figura"] = nome_fig
    out["janela_da_figura"] = j_fig
    out["tempo_s"] = time.time() - t0
    out["texto_relatorio_tex"] = texto_tex(out)
    return out


def texto_tex(e) -> str:
    """Parágrafo do relatório (todos os números vêm de `e`)."""
    nf, esc = rel.nf, rel.esc
    pcn = lambda x: rel.pcn(x, 1)
    n, dn = e["nfh"], e["densa_projetada"]
    f3 = lambda b: f"{nf(b['mediana'], 3)} [{nf(b['q1'], 3)}; {nf(b['q3'], 3)}]"
    ig = e["ig"]
    k_amp = int(np.argmax(n["amplitude_mediana_por_harmonico"]))
    m = e["metricas_teste_semente10"]
    T = [f"Semente 10, {e['n_janelas_teste']} janelas de teste. Participações de energia da NFH (mediana [Q1; Q3]): "
         f"$\\varepsilon_T={f3(n['eps_T'])}$, $\\varepsilon_S={f3(n['eps_S'])}$, $\\varepsilon_R={f3(n['eps_R'])}$. A densa da mesma semente, decomposta "
         f"só pelo Projetor, dá $\\varepsilon_T={f3(dn['eps_T'])}$, $\\varepsilon_S={f3(dn['eps_S'])}$, $\\varepsilon_R={f3(dn['eps_R'])}$. "
         f"O harmônico de maior amplitude mediana da NFH é $k={k_amp + 1}$ (período {nf(e['periodo_h_por_harmonico'][k_amp], 1)} h, "
         f"amplitude {nf(n['amplitude_mediana_por_harmonico'][k_amp], 3)}). RMSE e MAE de teste desta semente: NFH "
         f"{nf(m['nfh']['test_RMSE'])} e {nf(m['nfh']['test_MAE'])}; densa {nf(m['densa']['test_RMSE'])} e {nf(m['densa']['test_MAE'])}."]
    for nome in ("g_T", "g_S"):
        part = ig[nome]["participacao_mediana_por_grupo"]
        j = int(np.argmax(part))
        T.append(f" IG de ${nome[0]}_{nome[2]}$ ({ig['n_janelas']} janelas, {ig['n_passos']} passos, referência zero): maior participação mediana em "
                 f"``{esc(ig['grupos'][j])}'' ({pcn(part[j])}); erro de completude mediano {nf(ig[nome]['completude_erro_mediano'], 6)} "
                 f"(relativo {nf(ig[nome]['completude_rel_mediano'], 4)}).")
    if "verdade_sint_m" in e:
        v = e["verdade_sint_m"]
        T.append(f" SINT-M: acerto da tendência (x1) {nf(v['tendencia']['acerto_5_colunas'], 2)} com as 5 colunas e "
                 f"{nf(v['tendencia']['acerto_4_covariaveis'], 2)} só com as 4 covariáveis; acerto do ciclo (x2) "
                 f"{nf(v['ciclo']['acerto_5_colunas'], 2)} e {nf(v['ciclo']['acerto_4_covariaveis'], 2)}. Aviso: aqui $d=5$ e a última "
                 f"coluna é o próprio $y$ passado (na v3, $d=4$); sua participação mediana é {pcn(v['tendencia']['participacao_y_passado'])} em $g_T$ "
                 f"e {pcn(v['ciclo']['participacao_y_passado'])} em $g_S$.")
    if "verdade_sint_u48" in e and "nfh" in e["verdade_sint_u48"]:
        v = e["verdade_sint_u48"]
        T.append(f" SINT-U48, contra o sinal sem ruído do futuro decomposto pelas mesmas fórmulas: RMSE da previsão da NFH contra o sinal "
                 f"{nf(v['nfh']['RMSE_previsao_vs_sinal'])} (densa projetada {nf(v['densa_projetada']['RMSE_previsao_vs_sinal'])}); RMSE de $T$ "
                 f"{nf(v['nfh']['RMSE_T_vs_T_verdade'])} e de $S$ {nf(v['nfh']['RMSE_S_vs_S_verdade'])}; erro relativo mediano de $S$ "
                 f"{nf(v['nfh']['erro_relativo_S_mediano'], 3)}. Referência amostral do oráculo (janelas de teste): "
                 f"{nf(v.get('referencia_amostral_RMSE_oraculo_janelas_teste'))}.")
    return "".join(T) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", default=str(B / "resultados"))
    ap.add_argument("--lista", nargs="*", default=[], help="ids 'dataset|modo|backbone|w<l>|f<fr>' extras (qualquer veredito)")
    ap.add_argument("--so-lista", action="store_true", help="não incluir as combinações com SUCESSO, só as da lista")
    ap.add_argument("--janelas", type=int, default=50)
    ap.add_argument("--passos", type=int, default=64)
    ap.add_argument("--max-combinacoes", type=int, default=0, help="0 = todas")
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.enabled = False
    res = Path(a.res)
    cf, info = rel.ler_csv(res / "confirmacao.csv")
    conf = rel.confirmacao_por_combinacao(cf)
    por_id = {c["id"]: c for c in conf}
    alvo = [] if a.so_lista else [c for c in conf if c["veredito"] == "SUCESSO"]
    for i in a.lista:
        if i not in por_id:
            print(f"AVISO: combinação '{i}' não está em confirmacao.csv; ignorada")
        elif por_id[i] not in alvo:
            alvo.append(por_id[i])
    if a.max_combinacoes:
        alvo = alvo[:a.max_combinacoes]
    print(f"confirmacao.csv: {len(conf)} combinações; a explicar: {len(alvo)}")
    saida = res / "explicacoes"
    for c in alvo:
        print(f"[{time.strftime('%H:%M:%S')}] {c['id']} ({c['variante_nfh']}, {c['veredito']})", flush=True)
        try:
            r = explicar(c, res, saida, a.janelas, a.passos)
        except Exception as ex:
            r = {"combinacao": id_arquivo_comb(c), "id": c["id"], "erro": f"{type(ex).__name__}: {ex}", "traceback": traceback.format_exc()}
            print("  ERRO:", r["erro"], flush=True)
        saida.mkdir(parents=True, exist_ok=True)
        (saida / f"{id_arquivo_comb(c)}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
        if "erro" not in r:
            print(f"  ok em {r['tempo_s']:.0f}s; eps NFH T/S/R = {r['nfh']['eps_T']['mediana']:.3f}/{r['nfh']['eps_S']['mediana']:.3f}/"
                  f"{r['nfh']['eps_R']['mediana']:.3f}; completude rel. g_T {r['ig']['g_T']['completude_rel_mediano']:.2e}", flush=True)


if __name__ == "__main__":
    main()
