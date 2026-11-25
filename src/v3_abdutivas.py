"""
E5-v3: explicações abdutivas estruturais (Seção 4.5) sobre os neurônios da última camada dos modelos
treinados (NFH, K = 10, seleção segura, semente 0), nos 8 conjuntos e nos três backbones.

Propriedades (lineares em h; Proposição 4.x):
  P_T : s* (T_{H-1} - T_0)(h) >= |T_{H-1} - T_0|(h*) / 2   (tendência mantém o sentido e >= metade da variação)
  P_S : <e*, (a_2, b_2)(h)> >= A_2* / 2                     (ciclo diário, k = 2 com H = 48, >= metade da
                                                             amplitude na fase prevista)
Avaliadas na escala normalizada da camada (a normalização reversível multiplica a previsão por sigma_y > 0
e soma mu_y 1, o que não altera nenhuma das duas desigualdades).
Caixa empírica: mínimo e máximo de cada h_i nas janelas de treino, estendidos para conter h*.
Para cada janela de teste: tamanho k* da explicação mínima (ordenação dos ganhos), k_IG obtido acrescentando
neurônios na ordem de |w_i (h*_i - mu_i)| (IG = valor de Shapley exato de uma função afim com referência na
média de treino mu), e se os k* primeiros nessa ordem já são suficientes. Verificações: amostragem de
completamentos (nenhuma violação esperada) e invariância à parametrização (camada densa equivalente com
projeção dá os mesmos coeficientes e as mesmas explicações).
Saída: results/v3/abdutivas_janelas.csv, results/v3/abdutivas.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from v2_common import stack_ds  # noqa: E402
from v3_common import BACKBONES, CONJUNTOS, H3, K3, Q3, RES3, carregar, construir, segmentos  # noqa: E402

K_CICLO = 2           # com H = 48 h, o harmônico k = 2 tem período de 24 h
N_VERIF = 20          # janelas com verificação por amostragem
N_AMOSTRAS = 10000


@torch.no_grad()
def latentes(modelo, X, dev, bs=1024):
    return torch.cat([modelo(X[i:i + bs].to(dev))["latent"].cpu() for i in range(0, len(X), bs)]).double().numpy()


def coeficientes(head):
    W = head.lin.weight.detach().cpu().double().numpy()
    b = head.lin.bias.detach().cpu().double().numpy()
    if head.normalize:
        D = head.D.cpu().double().numpy()
        W, b = D[:, None] * W, D * b
    return W, b


def explicar(w, w0, h, lo, hi, xi, mu):
    """Uma janela: ganhos, k* (ótimo), k_IG e suficiência do top-k* pela ordem de |w (h - mu)|."""
    pior = np.minimum(w * lo, w * hi)
    delta = w * h - pior
    g0 = w0 + pior.sum()
    ordem = np.argsort(-delta, kind="stable")
    acum = g0 + np.concatenate([[0.0], np.cumsum(delta[ordem])])
    k_otimo = int(np.argmax(acum >= xi - 1e-12)) if (acum >= xi - 1e-12).any() else -1
    ordem_ig = np.argsort(-np.abs(w * (h - mu)), kind="stable")
    acum_ig = g0 + np.concatenate([[0.0], np.cumsum(delta[ordem_ig])])
    k_ig = int(np.argmax(acum_ig >= xi - 1e-12)) if (acum_ig >= xi - 1e-12).any() else -1
    topk_ig_suf = bool(acum_ig[k_otimo] >= xi - 1e-12) if k_otimo >= 0 else False
    # folga relativa no corte: quão perto do limiar estão as margens com k* e k* - 1 neurônios
    folga = float("inf")
    if k_otimo >= 0:
        folga = acum[k_otimo] - xi
        if k_otimo >= 1:
            folga = min(folga, xi - acum[k_otimo - 1])
        folga = folga / (abs(xi) + 1e-30)
    return k_otimo, k_ig, topk_ig_suf, ordem[:max(k_otimo, 0)], g0, folga


def propriedades(W, b, h, q1, P):
    """(w, w0, xi) das propriedades P_T e P_S para a instância h."""
    tH = P[:, -1] - P[:, 0]                    # T_{H-1} - T_0 = (P_{:,H-1} - P_{:,0})^T theta_T
    WT, bT = W[:q1], b[:q1]
    dT = float(tH @ (WT @ h + bT))
    s = 1.0 if dT >= 0 else -1.0
    props = {"P_T": (s * (WT.T @ tH), s * float(tH @ bT), abs(dT) / 2)}
    ia, ib = q1 + 2 * (K_CICLO - 1), q1 + 2 * (K_CICLO - 1) + 1
    a, bb_ = float(W[ia] @ h + b[ia]), float(W[ib] @ h + b[ib])
    A = float(np.hypot(a, bb_))
    e = np.array([a, bb_]) / (A + 1e-30)
    props["P_S"] = (e[0] * W[ia] + e[1] * W[ib], e[0] * b[ia] + e[1] * b[ib], A / 2)
    return props


def main(conjuntos=None):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    linhas, resumo = [], {}
    for conj in conjuntos or CONJUNTOS:
        ds = carregar(conj)
        tr, va, te = segmentos(ds)
        d = ds["feats"].shape[1]
        Xtr, _ = stack_ds(tr)
        Xte, _ = stack_ds(te)
        for bb in BACKBONES:
            arq = RES3 / "modelos" / f"{conj}__{bb}+NFH__s0.pt"
            if not arq.exists():
                continue
            modelo = construir(bb, "nfh", d)
            modelo.load_state_dict(torch.load(arq, map_location="cpu"))
            modelo = modelo.to(dev).eval()
            head = modelo.inner.head
            W, b = coeficientes(head)
            q1, m = head.q + 1, head.m
            P = head.P.cpu().double().numpy()
            Ft = head.Ft.cpu().double().numpy()
            Qp = head.Qp.cpu().double().numpy()
            Htr, Hte = latentes(modelo, Xtr, dev), latentes(modelo, Xte, dev)
            p = Htr.shape[1]
            lo0, hi0, mu = Htr.min(0), Htr.max(0), Htr.mean(0)
            # Invariância: camada densa equivalente (W' = M W) e coordenadas por projeção (Teorema 4.x)
            M = np.concatenate([P.T, Ft.T, Qp], 1)
            Wd, bd = M @ W, M @ b
            Wproj = np.concatenate([np.linalg.solve(P @ P.T, P) @ Wd, np.linalg.solve(Ft @ Ft.T, Ft) @ Wd])
            bproj = np.concatenate([np.linalg.solve(P @ P.T, P) @ bd, np.linalg.solve(Ft @ Ft.T, Ft) @ bd])
            dif_coef = float(max(np.abs(Wproj - W[:m]).max(), np.abs(bproj - b[:m]).max()))
            iguais, fora_caixa, viol, tempos, folgas_dif = 0, 0, 0, [], []
            for j in range(len(Hte)):
                h = Hte[j]
                lo, hi = np.minimum(lo0, h), np.maximum(hi0, h)
                fora_caixa += int((h < lo0).any() or (h > hi0).any())
                pr = propriedades(W, b, h, q1, P)
                prd = propriedades(np.concatenate([Wproj, W[m:]]), np.concatenate([bproj, b[m:]]), h, q1, P)
                for nome, (w, w0, xi) in pr.items():
                    t0 = time.perf_counter()
                    k_ot, k_ig, suf, A, g0, folga = explicar(w, w0, h, lo, hi, xi, mu)
                    tempos.append(time.perf_counter() - t0)
                    wd, w0d, xid = prd[nome]
                    k_d, _, _, A_d, _, _ = explicar(wd, w0d, h, lo, hi, xid, mu)
                    igual = k_d == k_ot and set(A_d.tolist()) == set(A.tolist())
                    iguais += int(igual)
                    if not igual:
                        folgas_dif.append(folga)
                    if j < N_VERIF and k_ot >= 0:
                        amostra = rng.uniform(lo, hi, size=(N_AMOSTRAS, p))
                        amostra[:, A] = h[A]
                        viol += int(((amostra @ w + w0) < xi - 1e-9).sum())
                    linhas.append({"conjunto": conj, "backbone": bb, "propriedade": nome, "janela": j, "p": p,
                                   "k_otimo": k_ot, "k_ig": k_ig, "topk_ig_suficiente": suf,
                                   "trivial": k_ot == 0, "margem_vazia": g0 - xi, "limiar": xi})
            dfj = pd.DataFrame([r for r in linhas if r["conjunto"] == conj and r["backbone"] == bb])
            res = {"p": p, "n_janelas": int(len(Hte)), "dif_coef_densa_proj": dif_coef,
                   "explicacoes_iguais_densa_proj": iguais / (2 * len(Hte)),
                   "max_folga_relativa_nas_diferencas": float(max(folgas_dif)) if folgas_dif else 0.0,
                   "fracao_h_fora_caixa_treino": fora_caixa / len(Hte), "violacoes_amostragem": viol,
                   "tempo_medio_ms": 1e3 * float(np.mean(tempos))}
            for nome in ("P_T", "P_S"):
                s = dfj[dfj.propriedade == nome]
                nt = s[s.k_otimo > 0]
                res[nome] = {"k_mediana": float(s.k_otimo.median()), "k_media": float(s.k_otimo.mean()),
                             "fracao_p_mediana": float((s.k_otimo / p).median()),
                             "fracao_trivial": float(s.trivial.mean()), "fracao_todos": float((s.k_otimo == p).mean()),
                             "razao_kIG_k_media": float((nt.k_ig / nt.k_otimo).mean()) if len(nt) else float("nan"),
                             "fracao_topk_ig_suficiente": float(nt.topk_ig_suficiente.mean()) if len(nt) else float("nan")}
            resumo[f"{conj}|{bb}"] = res
            print(f"[{time.strftime('%H:%M:%S')}] {conj} {bb} p={p}: P_T k* mediana {res['P_T']['k_mediana']:.0f} "
                  f"({100 * res['P_T']['fracao_p_mediana']:.1f}% de p), trivial {res['P_T']['fracao_trivial']:.2f}, "
                  f"kIG/k* {res['P_T']['razao_kIG_k_media']:.2f}; P_S k* mediana {res['P_S']['k_mediana']:.0f} "
                  f"({100 * res['P_S']['fracao_p_mediana']:.1f}%), kIG/k* {res['P_S']['razao_kIG_k_media']:.2f}; "
                  f"iguais {res['explicacoes_iguais_densa_proj']:.3f}, dif {dif_coef:.1e}, viol {viol}, "
                  f"{res['tempo_medio_ms']:.3f} ms", flush=True)
    pd.DataFrame(linhas).to_csv(RES3 / "abdutivas_janelas.csv", index=False)
    (RES3 / "abdutivas.json").write_text(json.dumps(resumo, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
