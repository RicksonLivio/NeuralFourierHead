"""Etapa 4: todos os números de uma janela de teste, para a semente 10 da combinação confirmada (passo_a_passo.json).

Retreina (determinístico, CPU, 1 thread) densa, densa+E e a NFH escolhida com a semente 10 e confere que o RMSE de teste
bate com confirmacao.csv. Janela explicada: a central do teste, índice len(teste)//2 (fixado antes de olhar).
Neurônio detalhado do backbone: o de índice 0 (o primeiro, fixado antes de olhar). Linhas de c' detalhadas: todas as 8.
"""
import json
import math

import numpy as np
import pandas as pd
import torch

from comum import AQUI, H, K, L, D, Q, SIGMA, bases, dados, oraculo, tr, treinar_variantes
from gerar import gerar, N

SEMENTE = 10
PASSOS_IG = 256
NEURONIO = 0
LISTA = lambda a: np.asarray(a, dtype=float).tolist()  # noqa: E731


def gram_schmidt(Bm, H):
    """Gram-Schmidt clássico, passo a passo, nas colunas de B e depois em e_1, e_2, ... até completar H vetores."""
    qs, passos = [], []
    cand = [(f"coluna {j + 1} de B", Bm[:, j].copy()) for j in range(Bm.shape[1])] + \
           [(f"e_{i + 1}", np.eye(H)[:, i]) for i in range(H)]
    for nome, v in cand:
        if len(qs) == H:
            break
        proj = [float(q @ v) for q in qs]
        w = v - sum((pj * q for pj, q in zip(proj, qs)), np.zeros(H))
        nw = float(np.linalg.norm(w))
        aceito = nw > 1e-8
        passos.append({"vetor": nome, "bruto": LISTA(v), "projecoes": proj, "apos_subtrair": LISTA(w), "norma": nw,
                       "aceito": aceito, "q": LISTA(w / nw) if aceito else None})
        if aceito:
            qs.append(w / nw)
    return np.stack(qs, 1), passos


GERADOR = {  # constantes de exemplo_livro/gerar.py (conferidas abaixo: a série refeita com elas é igual à de gerar())
    "x1": {"nivel": 1.0, "a1": 0.6, "per1": 480, "ph1": 0.4, "a2": 0.3, "per2": 1300, "ph2": 1.1},
    "x2": {"a_base": 1.0, "a_mod": 0.4, "per_mod": 700, "ph_mod": 2.0, "per_c1": 8, "ph_c1": 0.7, "peso_c2": 0.3, "per_c2": 4, "ph_c2": -0.5},
    "x3": {"a1": 0.8, "per1": 600, "ph1": 2.5, "a2": 0.4, "per2": 1700, "ph2": 0.0},
    "x4": {"b_base": 1.0, "b_mod": 0.4, "per_mod": 900, "per": 8, "ph": 2.6},
    "atraso": 8, "N": 9000, "cortes": [0.6, 0.8], "semente_dados": 2026}


def confere_gerador():
    g, s = GERADOR, np.arange(GERADOR["N"], dtype=float)
    a, b, c3, d4 = g["x1"], g["x2"], g["x3"], g["x4"]
    x1 = a["nivel"] + a["a1"] * np.sin(2 * np.pi * s / a["per1"] + a["ph1"]) + a["a2"] * np.sin(2 * np.pi * s / a["per2"] + a["ph2"])
    A = b["a_base"] + b["a_mod"] * np.sin(2 * np.pi * s / b["per_mod"] + b["ph_mod"])
    x2 = A * (np.cos(2 * np.pi * s / b["per_c1"] + b["ph_c1"]) + b["peso_c2"] * np.cos(2 * np.pi * s / b["per_c2"] + b["ph_c2"]))
    x3 = c3["a1"] * np.sin(2 * np.pi * s / c3["per1"] + c3["ph1"]) + c3["a2"] * np.sin(2 * np.pi * s / c3["per2"] + c3["ph2"])
    B = d4["b_base"] + d4["b_mod"] * np.sin(2 * np.pi * s / d4["per_mod"])
    x4 = B * np.cos(2 * np.pi * s / d4["per"] + d4["ph"])
    Xr, _, _ = gerar()
    assert np.allclose(np.stack([x1, x2, x3, x4], 1), Xr, atol=1e-12) and N == GERADOR["N"]


def main():
    confere_gerador()
    P, F, Ft, Qp, M = bases()
    m = Q + 1 + 2 * K
    part, T = dados()
    Xte, Yte = T["teste"]
    res = treinar_variantes(16, 1000, SEMENTE, T, quais=["densa", "densa+E", "zero_L1000"])
    sel = json.load(open(AQUI / "selecao.json"))
    conf = pd.read_csv(AQUI / "confirmacao.csv")
    assert (sel["p"], sel["n"], sel["variante"]) == (16, 1000, "zero_L1000")
    confere = {}
    for v, nome in (("densa", "densa"), ("densa+E", "densa+E"), ("zero_L1000", "NFH")):
        ref = conf[(conf.semente == SEMENTE) & (conf.variante == nome)].iloc[0]
        r = res[v][2]
        confere[nome] = {"retreino_test_RMSE": r["test_RMSE"], "csv_test_RMSE": float(ref.test_RMSE),
                         "retreino_test_MAE": r["test_MAE"], "csv_test_MAE": float(ref.test_MAE)}
        assert abs(r["test_RMSE"] - ref.test_RMSE) < 1e-9 and abs(r["test_MAE"] - ref.test_MAE) < 1e-9, confere[nome]
    nfh, densa = res["zero_L1000"][0], res["densa"][0]
    rmse_or = float(np.sqrt(np.mean((oraculo(part["teste"][0]) - part["teste"][1]) ** 2)))
    mae_or = float(np.mean(np.abs(oraculo(part["teste"][0]) - part["teste"][1])))

    # ---- bases
    PPt = P @ P.T
    poli_F = np.linalg.solve(PPt, P @ F.T)                      # (q+1, 2K): parte polinomial de cada linha de F
    Bm = np.concatenate([P, F], 0).T                             # (H, m)
    Qgs, passos_gs = gram_schmidt(Bm, H)
    Qgs_perp = Qgs[:, m:]
    O = Qgs_perp.T @ Qp                                          # Qp = Qgs_perp O
    erro_proj = float(np.abs(Qgs_perp @ Qgs_perp.T - Qp @ Qp.T).max())
    NORMAS = tr.NORMAS
    Dn = 1.0 / NORMAS
    MD = M * Dn[None, :]

    # ---- janela
    j = len(Xte) // 2
    i0 = int(np.arange(max(int(0.8 * N), H), N - L - H + 1)[j])      # início da janela na série
    s0 = i0 + L - H                                                  # instante de x que aparece em y_{t+1}
    x = part["teste"][0][j]                                          # (L, d)
    y_real, y_or = part["teste"][1][j], oraculo(part["teste"][0][j:j + 1])[0]
    Xs, ys, limpo = gerar()
    assert np.allclose(Xs[i0:i0 + L], x) and np.allclose(limpo[i0 + L:i0 + L + H], y_or)
    trecho = {"inicio": i0 - 8, "X": LISTA(Xs[i0 - 8:i0 + L + H]), "y": LISTA(ys[i0 - 8:i0 + L + H]),
              "limpo": LISTA(limpo[i0 - 8:i0 + L + H]), "i0": i0, "L": L, "H": H}

    # ---- backbone e cabeça em dupla precisão (parâmetros float32 do modelo, contas em float64)
    Wphi = nfh.phi.weight.detach().double().numpy(); bphi = nfh.phi.bias.detach().double().numpy()
    W = nfh.lin.weight.detach().double().numpy(); b = nfh.lin.bias.detach().double().numpy()
    xv = x.astype(np.float32).astype(np.float64).reshape(-1)         # vec(X), linha a linha (ell, variável)
    z = Wphi @ xv + bphi
    h = np.tanh(z)
    with torch.no_grad():
        h_modelo = nfh.h(Xte[j:j + 1])[0].double().numpy()
        y_modelo = nfh(Xte[j:j + 1])[0].double().numpy()
        y_densa_modelo = densa(Xte[j:j + 1])[0].double().numpy()
    assert np.abs(h - h_modelo).max() < 1e-5
    nomes_var = ["x1", "x2", "x3", "x4"]
    prod = Wphi[NEURONIO] * xv
    neuronio = {"indice": NEURONIO + 1, "produtos": [{"ell": int(k // D) + 1, "var": nomes_var[k % D], "w": float(Wphi[NEURONIO, k]),
                                                      "x": float(xv[k]), "w_x": float(prod[k])} for k in range(L * D)],
                "soma_produtos": float(prod.sum()), "vies": float(bphi[NEURONIO]), "z": float(z[NEURONIO]),
                "h": float(h[NEURONIO])}
    cprime = W @ h + b
    linhas_c = [{"i": i, "produtos": LISTA(W[i] * h), "soma": float((W[i] * h).sum()), "vies": float(b[i]), "c_linha": float(cprime[i])}
                for i in range(H)]
    c = Dn * cprime
    yhat = M @ c
    q1 = Q + 1
    thT, thS, u = c[:q1], c[q1:m], c[m:]
    Tv, Sv, Rv = P.T @ thT, Ft.T @ thS, Qp @ u

    def energia(y, T_, S_, R_):
        yb = y.mean(); yc = y - yb; E = float(yc @ yc)
        return {"media": float(yb), "norma2_centrada": E, "eps_T": float((T_ - yb) @ (T_ - yb) / E),
                "eps_S": float(S_ @ S_ / E), "eps_R": float(R_ @ R_ / E)}

    def decompor(y):
        tT = np.linalg.solve(PPt, P @ y); tS = np.linalg.solve(Ft @ Ft.T, Ft @ y); uu = Qp.T @ y
        return tT, tS, uu, P.T @ tT, Ft.T @ tS, Qp @ uu

    def harmonicos(tS, S_, y):
        yc = y - y.mean(); E = float(yc @ yc); out = []
        for k in range(K):
            a, bb = tS[2 * k], tS[2 * k + 1]
            Sk = Ft[2 * k:2 * k + 2].T @ tS[2 * k:2 * k + 2]
            out.append({"k": k + 1, "periodo_passos": H / (k + 1), "a": float(a), "b": float(bb), "A": math.hypot(a, bb),
                        "fase_rad": math.atan2(bb, a), "e_k": float(Sk @ S_ / E), "S_k": LISTA(Sk)})
        return out

    dec_nfh = decompor(yhat)
    tT0, tS0, u0, T0, S0, R0 = decompor(y_or)
    A_s = 1 + 0.4 * np.sin(2 * np.pi * (s0 + np.arange(H)) / 700 + 2)
    A_meio = float(A_s.mean())
    psi1, psi2 = 2 * np.pi * s0 / 8 + 0.7, 2 * np.pi * s0 / 4 - 0.5
    beta = np.array([A_meio * math.cos(psi1), -A_meio * math.sin(psi1), 0.3 * A_meio * math.cos(psi2), -0.3 * A_meio * math.sin(psi2)])
    x1f, x2f = x[-H:, 0], x[-H:, 1]
    alpha_x1 = np.polyfit(np.arange(H) / H, x1f, 1)[::-1]
    # densa da mesma semente
    yd = y_densa_modelo
    tTd, tSd, ud, Td, Sd, Rd = decompor(yd)
    # conversão
    Wd, bd = MD @ W, MD @ b
    y_conv = Wd @ h + bd
    W_volta = np.linalg.solve(MD, Wd); b_volta = np.linalg.solve(MD, bd)
    conv_densa = tr.converter(densa)                                  # densa treinada -> forma NFH
    with torch.no_grad():
        err_dn = float((conv_densa(Xte).double() - densa(Xte).double()).abs().max())
    Wdn = conv_densa.lin.weight.detach().double().numpy()
    # N-BEATS-I: a mesma previsão com duas divisões T/S diferentes
    tau = np.arange(H) / H
    Tnb = np.stack([np.ones(H), tau, tau ** 2], 1)
    Snb = np.concatenate([np.ones((H, 1))] + [np.stack([np.cos(2 * np.pi * k * tau), np.sin(2 * np.pi * k * tau)], 1) for k in range(1, H // 2)], 1)
    coef_nb, *_ = np.linalg.lstsq(np.concatenate([Tnb, Snb], 1), yhat, rcond=None)
    T_a, S_a = Tnb @ coef_nb[:3], Snb @ coef_nb[3:]
    um = np.ones(H)
    nbeats = {"T_a": LISTA(T_a), "S_a": LISTA(S_a), "T_b": LISTA(T_a + um), "S_b": LISTA(S_a - um),
              "erro_soma_a": float(np.abs(T_a + S_a - yhat).max()), "erro_soma_b": float(np.abs(T_a + um + S_a - um - yhat).max()),
              "colunas_T": Tnb.shape[1], "colunas_S": Snb.shape[1], "posto_conjunto": int(np.linalg.matrix_rank(np.concatenate([Tnb, Snb], 1))),
              "media_S_a": float(S_a.mean()), "media_S_b": float((S_a - um).mean())}

    # ---- Integrated Gradients (dupla precisão, referência nula, regra do ponto médio)
    td = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
    tWphi, tbphi, tW, tb, tDn = td(Wphi), td(bphi), td(W), td(b), td(Dn)
    tP, tFt, tM = td(P), td(Ft), td(M)
    ccomp = lambda zz: (torch.tanh(zz.reshape(len(zz), -1) @ tWphi.T + tbphi) @ tW.T + tb) * tDn  # noqa: E731
    Pi_S = torch.tensor(Ft.T @ np.linalg.solve(Ft @ Ft.T, Ft), dtype=torch.float64)
    gT = lambda zz: (ccomp(zz)[:, :q1] @ tP).mean()                                   # noqa: E731
    gS = lambda zz: ((ccomp(zz)[:, q1:m] @ tFt) ** 2).sum()                           # noqa: E731
    yor_fn = lambda zz: zz[:, -H:, 0] + zz[:, -H:, 1]                                  # noqa: E731
    gT_or = lambda zz: yor_fn(zz).mean()                                               # noqa: E731
    gS_or = lambda zz: ((yor_fn(zz) @ Pi_S.T) ** 2).sum()                              # noqa: E731
    xt = torch.tensor(x, dtype=torch.float64)[None]

    def ig(f):
        tot = torch.zeros_like(xt)
        for r in range(PASSOS_IG):
            zz = (((r + 0.5) / PASSOS_IG) * xt).requires_grad_(True)
            g, = torch.autograd.grad(f(zz), zz)
            tot += g
        return (xt * tot / PASSOS_IG)[0].numpy()

    res_ig = {}
    for nome, f in (("g_T", gT), ("g_S", gS), ("g_T_oraculo", gT_or), ("g_S_oraculo", gS_or)):
        a = ig(f)
        with torch.no_grad():
            delta = float(f(xt) - f(torch.zeros_like(xt)))
        pv = a.sum(0)
        res_ig[nome] = {"atribuicao_LxD": LISTA(a), "por_variavel": LISTA(pv), "participacao": LISTA(np.abs(pv) / np.abs(pv).sum()),
                        "soma": float(a.sum()), "g_X": float(f(xt)), "g_ref": float(f(torch.zeros_like(xt))),
                        "g_X_menos_g_ref": delta, "erro_completude": abs(float(a.sum()) - delta)}
    # g_T = média de ŷ, conferência
    ig_media = ig(lambda zz: (ccomp(zz) @ tM.T).mean())
    res_ig["media_da_previsao"] = {"por_variavel": LISTA(ig_media.sum(0)), "soma": float(ig_media.sum()),
                                   "max_dif_vs_g_T": float(np.abs(ig_media - np.array(res_ig["g_T"]["atribuicao_LxD"])).max())}

    # ---- explicação abdutiva sobre h: s * theta_T1(h') >= |theta_T1(h)| / 2 para todo h' em [-1,1]^p
    wT1, w0 = Dn[1] * W[1], Dn[1] * b[1]
    th1 = float(wT1 @ h + w0)
    s_ = 1.0 if th1 >= 0 else -1.0
    wv, w0v, xi = s_ * wT1, s_ * w0, abs(th1) / 2
    gam0 = float(w0v - np.abs(wv).sum())
    ganhos = wv * h + np.abs(wv)
    ordem = np.argsort(-ganhos)
    acum = np.concatenate([[gam0], gam0 + np.cumsum(ganhos[ordem])])
    kstar = int(np.argmax(acum >= xi))
    with torch.no_grad():
        mu = nfh.h(T["treino"][0][:1000]).double().numpy().mean(0)
    ordem_ig = np.argsort(-np.abs(wv * (h - mu)))
    acum_ig = np.concatenate([[gam0], gam0 + np.cumsum(ganhos[ordem_ig])])
    k_ig = int(np.argmax(acum_ig >= xi))
    rng = np.random.default_rng(0)
    A_set = ordem[:kstar]
    Hs = rng.uniform(-1, 1, (10000, len(h))); Hs[:, A_set] = h[A_set]
    viol = int(((Hs @ wv + w0v) < xi - 1e-12).sum())
    A_menos = ordem[:kstar - 1] if kstar > 0 else ordem[:0]
    h_pior = -np.sign(wv) * 1.0; h_pior[A_menos] = h[A_menos]       # pior completamento de A sem o último
    pior_menos = float(h_pior @ wv + w0v)
    A_menos_viol = bool(pior_menos < xi)
    Hm = rng.uniform(-1, 1, (10000, len(h))); Hm[:, A_menos] = h[A_menos]
    viol_menos_amostra = int(((Hm @ wv + w0v) < xi).sum())
    # confere pelo cálculo direto do modelo: theta_T1 de h_pior
    with torch.no_grad():
        th1_pior = float(((torch.tensor(W[1]) @ torch.tensor(h_pior) + torch.tensor(b[1])) * float(Dn[1])))
    abd = {"propriedade": "s * theta_T1(h') >= |theta_T1(h)| / 2 para todo h' em [-1,1]^p com h'_i = h_i em A",
           "theta_T1": th1, "sinal": s_, "xi": xi, "w": LISTA(wv), "w0": float(w0v), "Gamma_vazio": gam0, "ganhos": LISTA(ganhos),
           "ordem_gulosa": (ordem + 1).tolist(), "acumulado": LISTA(acum), "k_estrela": kstar, "A": (np.sort(A_set) + 1).tolist(),
           "ordem_IG": (ordem_ig + 1).tolist(), "acumulado_IG": LISTA(acum_ig), "k_IG": k_ig,
           "violacoes_10000_completamentos": viol, "violacoes_10000_sem_o_ultimo": viol_menos_amostra,
           "pior_caso_sem_ultimo": pior_menos, "pior_caso_sem_ultimo_viola": A_menos_viol, "theta_T1_do_pior_h": s_ * th1_pior,
           "h_medio_treino": LISTA(mu), "ultimo_neuronio_incluido": int(ordem[kstar - 1]) + 1 if kstar > 0 else None}

    out = {
        "configuracao": {"H": H, "q": Q, "K": K, "L": L, "d": D, "p": 16, "sigma": SIGMA, "n_treino": 1000, "semente": SEMENTE,
                         "variante_nfh": "zero_L1000", "lambda": tr.LAMBDA if hasattr(tr, "LAMBDA") else 1000.0,
                         "epocas_escolhidas": {k: res[k][1] for k in res}, "janela_teste": j, "n_janelas_teste": len(Xte),
                         "inicio_janela_na_serie": i0, "s0": s0, "passos_IG": PASSOS_IG, "neuronio_detalhado": NEURONIO + 1,
                         "lr": tr.LR, "lr_ajuste": tr.LR_AJ, "epocas_max": tr.EPOCAS, "paciencia": tr.PACIENCIA, "lote": tr.LOTE,
                         "n_parametros_cabeca": int(W.size + b.size), "n_parametros_backbone": int(Wphi.size + bphi.size)},
        "gerador": GERADOR, "n_janelas": {k: len(part[k][0]) for k in part},
        "conferencia_teste": confere, "oraculo_teste": {"RMSE": rmse_or, "MAE": mae_or},
        "bases": {"tau": LISTA(tau), "P": LISTA(P), "F": LISTA(F), "PPt": LISTA(PPt), "PPt_inv": LISTA(np.linalg.inv(PPt)),
                  "parte_polinomial_de_F": LISTA(poli_F.T), "Ft": LISTA(Ft), "B": LISTA(Bm), "gram_schmidt": passos_gs,
                  "Q_GS": LISTA(Qgs_perp), "Q": LISTA(Qp), "O_GS_para_Q": LISTA(O), "erro_projetor_GS_vs_Q": erro_proj,
                  "M": LISTA(M), "normas_D": LISTA(NORMAS), "D": LISTA(Dn), "cond_M": float(np.linalg.cond(M)),
                  "erro_Ft_ortogonal_a_P": float(np.abs(Ft @ P.T).max()), "erro_ortogonalidade_Q": float(np.abs(Qp.T @ Qp - np.eye(H - m)).max()),
                  "erro_Q_ortogonal_a_B": float(np.abs(Qp.T @ Bm).max())},
        "janela": {"X": LISTA(x), "nomes": nomes_var, "y_real_futuro": LISTA(y_real), "oraculo_futuro": LISTA(y_or), "trecho_serie": trecho},
        "camada": {"neuronio": neuronio, "h": LISTA(h), "erro_h_vs_modelo": float(np.abs(h - h_modelo).max()),
                   "W": LISTA(W), "b": LISTA(b), "linhas_c": linhas_c, "c_linha": LISTA(cprime), "c": LISTA(c),
                   "erro_previsao_vs_modelo_float32": float(np.abs(yhat - y_modelo).max())},
        "nfh": {"theta_T": LISTA(thT), "theta_S": LISTA(thS), "u": LISTA(u), "T": LISTA(Tv), "S": LISTA(Sv), "R": LISTA(Rv),
                "previsao": LISTA(yhat), "erro_soma": float(np.abs(Tv + Sv + Rv - yhat).max()),
                "erro_identificabilidade": float(max(np.abs(dec_nfh[0] - thT).max(), np.abs(dec_nfh[1] - thS).max(), np.abs(dec_nfh[2] - u).max())),
                "energia": energia(yhat, Tv, Sv, Rv), "harmonicos": harmonicos(thS, Sv, yhat),
                "media_T": float(Tv.mean()), "media_previsao": float(yhat.mean()),
                "erro_vs_real_RMSE_janela": float(np.sqrt(np.mean((yhat - y_real) ** 2))),
                "erro_vs_real_MAE_janela": float(np.mean(np.abs(yhat - y_real)))},
        "verdade": {"theta_T": LISTA(tT0), "theta_S": LISTA(tS0), "u": LISTA(u0), "T": LISTA(T0), "S": LISTA(S0), "R": LISTA(R0),
                    "energia": energia(y_or, T0, S0, R0), "harmonicos": harmonicos(tS0, S0, y_or),
                    "beta_gerador_com_A_medio": LISTA(beta), "A_medio": A_meio, "psi1": psi1, "psi2": psi2,
                    "reta_de_x1_no_horizonte": LISTA(alpha_x1), "x1_futuro": LISTA(x1f), "x2_futuro": LISTA(x2f),
                    "theta_T_previsto_pelo_gerador": LISTA(alpha_x1 + poli_F @ beta),
                    "parte_polinomial_dos_ciclos": LISTA(poli_F @ tS0)},
        "erros_nfh_vs_verdade": {"theta_T": LISTA(thT - tT0), "theta_S": LISTA(thS - tS0), "max_abs_previsao": float(np.abs(yhat - y_or).max()),
                                 "rmse_janela_nfh_vs_oraculo": float(np.sqrt(np.mean((yhat - y_or) ** 2))),
                                 "rmse_janela_densa_vs_oraculo": float(np.sqrt(np.mean((yd - y_or) ** 2))),
                                 "max_abs_T": float(np.abs(Tv - T0).max()), "max_abs_S": float(np.abs(Sv - S0).max()),
                                 "max_abs_R": float(np.abs(Rv - R0).max()),
                                 "rmse_janela_densa_vs_real": float(np.sqrt(np.mean((yd - y_real) ** 2))),
                                 "rmse_janela_nfh_vs_real": float(np.sqrt(np.mean((yhat - y_real) ** 2)))},
        "densa_mesma_semente": {"previsao": LISTA(yd), "theta_T": LISTA(tTd), "theta_S": LISTA(tSd), "u": LISTA(ud), "T": LISTA(Td),
                                "S": LISTA(Sd), "R": LISTA(Rd), "energia": energia(yd, Td, Sd, Rd), "harmonicos": harmonicos(tSd, Sd, yd)},
        "conversao": {"W_densa_equivalente": LISTA(Wd), "b_densa_equivalente": LISTA(bd), "erro_previsao": float(np.abs(y_conv - yhat).max()),
                      "erro_volta_W": float(np.abs(W_volta - W).max()), "erro_volta_b": float(np.abs(b_volta - b).max()),
                      "densa_para_nfh_erro_previsao_teste": err_dn, "W_densa_treinada_em_forma_nfh_linha0": LISTA(Wdn[0]),
                      "densa_treinada_W_linha0": LISTA(densa.lin.weight.detach().double().numpy()[0])},
        "nbeats_i": nbeats, "integrated_gradients": res_ig, "abdutiva": abd,
    }
    json.dump(out, open(AQUI / "passo_a_passo.json", "w"), indent=1, ensure_ascii=False)
    print("conferência RMSE de teste:", json.dumps(confere))
    print("janela", j, "i0", i0, "| erro soma", out["nfh"]["erro_soma"], "| ident", out["nfh"]["erro_identificabilidade"])
    print("energia NFH", out["nfh"]["energia"]); print("energia verdade", out["verdade"]["energia"])
    print("theta_S NFH", LISTA(thS), "\n verdade", LISTA(tS0), "\n gerador", LISTA(beta))
    print("theta_T NFH", LISTA(thT), "verdade", LISTA(tT0), "gerador", out["verdade"]["theta_T_previsto_pelo_gerador"])
    for k_, v in res_ig.items():
        print(k_, "participação", [round(zz, 3) for zz in v.get("participacao", [])], "completude", v.get("erro_completude"), v.get("max_dif_vs_g_T"))
    print("abdutiva k*", kstar, "de", len(h), "k_IG", k_ig, "violações", viol, "sem último", viol_menos_amostra, pior_menos, xi)
    print("erro conv", out["conversao"]["erro_previsao"], out["conversao"]["erro_volta_W"], err_dn, "erro_proj_GS", erro_proj)
    print("RMSE janela nfh/densa vs oráculo", out["erros_nfh_vs_verdade"]["rmse_janela_nfh_vs_oraculo"], out["erros_nfh_vs_verdade"]["rmse_janela_densa_vs_oraculo"])


if __name__ == "__main__":
    main()
