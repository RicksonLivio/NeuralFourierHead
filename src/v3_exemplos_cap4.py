"""Versão 3, Cap. 4: números novos dos exemplos com d = 1 e d = 2 (L = 12, H = 8, K = 2, q = 1, p = 4).

1. M43 ("quais instantes da janela pesaram mais", d = 1): distribuição, pelas 12 defasagens, da
   atribuição por Integrated Gradients do nível previsto g_T, lida de results/v2/small_example.json.
2. Explicação abdutiva estrutural (Seção nova do Cap. 4) sobre os p = 4 neurônios da última camada.
   O modelo pequeno é retreinado exatamente como em src/v2_small_example.py (CPU, sementes fixas) e
   conferido contra o JSON. Propriedades (lineares em h, Proposição do Cap. 2):
     P_T: s * theta_{T,1}(h) >= |theta*_{T,1}| / 2   (tendência mantém o sentido e >= metade da inclinação)
     P_S: <e*, (a_1, b_1)(h)> >= A*_1 / 2            (ciclo de 8 h mantém >= metade da amplitude na fase prevista)
   Caixas: formal [-1, 1]^4 (tanh, vale para qualquer janela) e empírica (mín./máx. no treino).
   Verificações: guloso = força bruta nos 2^4 subconjuntos; amostragem de completamentos; invariância
   à parametrização (camada densa equivalente + projeção dá a mesma explicação).
   Volta à entrada (d = 2): IG do funcional restrito g_A(X) = sum_{i in A} w_i h_i(X) por variável.
Saída: results/v3/exemplos_cap4.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import v2_small_example as ex  # noqa: E402
from v2_common import WindowDataset, load_etth1, set_seed, stack_ds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAIDA = ROOT / "results" / "v3" / "exemplos_cap4.json"
JSON_V2 = ROOT / "results" / "v2" / "small_example.json"


def distribuicao_defasagens(v2):
    """Participação de cada defasagem na atribuição absoluta de g_T (d = 1)."""
    ig = np.array(v2["d1"]["atribuicao"]["T"]["ig"], dtype=float)[:, 0]   # ordem: t-11, ..., t
    absol = np.abs(ig)
    part = absol / absol.sum()
    return {"ig_por_defasagem_mais_antiga_primeiro": ig.tolist(),
            "participacao_abs_pct": (100 * part).round(1).tolist(),
            "ultima_hora_pct": float(100 * part[-1]), "primeira_hora_pct": float(100 * part[0]),
            "intermediarias_pct": float(100 * part[1:-1].sum()),
            "ultima_hora_ig": float(ig[-1]), "soma_ig": float(ig.sum()),
            "soma_intermediarias_ig": float(ig[1:-1].sum())}


def treinar(d):
    """Reproduz o treino de v2_small_example.run(d) até o melhor estado de validação."""
    set_seed(0)
    feats, target, names = load_etth1(multivariate=True)
    cols = [names.index("OT")] if d == 1 else [names.index("OT"), names.index("HUFL")]
    f = feats[:, cols]
    tr = WindowDataset(f[:8640], target[:8640], ex.L, ex.H)
    va = WindowDataset(f[8640:11520], target[8640:11520], ex.L, ex.H)
    te = WindowDataset(f[11520:], target[11520:], ex.L, ex.H)
    model = ex.Tiny(d)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    Xtr, Ytr = stack_ds(tr)
    Xva, Yva = stack_ds(va)
    best, state = 1e9, None
    g = torch.Generator().manual_seed(0)
    for _ in range(40):
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = ((model(Xtr[idx])["forecast"] - Ytr[idx]) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = float(((model(Xva)["forecast"] - Yva) ** 2).mean())
        if v < best:
            best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    Xte, _ = stack_ds(te)
    return model, Xtr, Xte


def afim_da_cabeca(model):
    """theta_T, theta_S como funções afins de h: (W_T, b_T), (W_S, b_S); bases P, Ft, Qp."""
    head = model.head
    W = head.lin.weight.detach().double().numpy()
    b = head.lin.bias.detach().double().numpy()
    if head.normalize:
        D = head.D.double().numpy()
        W, b = D[:, None] * W, D * b
    q1, m = head.q + 1, head.m
    return W, b, q1, m


def ganhos(w, w0, h_ast, lo, hi):
    pior_livre = np.minimum(w * lo, w * hi)
    delta = w * h_ast - pior_livre
    gamma0 = w0 + pior_livre.sum()
    return delta, gamma0


def guloso(delta, gamma0, xi):
    ordem = np.argsort(-delta, kind="stable")
    acum = gamma0
    if acum >= xi:
        return [], gamma0
    for k, i in enumerate(ordem, 1):
        acum += delta[i]
        if acum >= xi:
            return sorted(int(j) for j in ordem[:k]), float(acum)
    raise RuntimeError("a propriedade não vale na própria instância")


def forca_bruta(w, w0, h_ast, lo, hi, xi):
    """Menor conjunto suficiente por enumeração de subconjuntos e de vértices da caixa livre."""
    p = len(w)
    melhores = None
    for r in range(p + 1):
        achados = []
        for A in itertools.combinations(range(p), r):
            livres = [i for i in range(p) if i not in A]
            pior = np.inf
            for vert in itertools.product(*[(lo[i], hi[i]) for i in livres]):
                hh = h_ast.copy()
                for i, v in zip(livres, vert):
                    hh[i] = v
                pior = min(pior, float(w @ hh + w0))
            if pior >= xi - 1e-12:
                achados.append(list(A))
        if achados:
            melhores = achados
            break
    return melhores


def ordem_atribuicao(w, h_ast, ref, gamma0, delta, xi):
    """Quantos neurônios são necessários, adicionados na ordem de |w_i (h*_i - ref_i)| (IG = SHAP exato
    para função linear com referência ref), até o conjunto ficar suficiente."""
    phi = w * (h_ast - ref)
    ordem = np.argsort(-np.abs(phi), kind="stable")
    acum = gamma0
    if acum >= xi:
        return 0, phi
    for k, i in enumerate(ordem, 1):
        acum += delta[i]
        if acum >= xi:
            return k, phi
    return len(w), phi


def amostrar_completamentos(w, w0, h_ast, lo, hi, A, xi, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    hh = rng.uniform(lo, hi, size=(n, len(w)))
    hh[:, A] = h_ast[A]
    val = hh @ w + w0
    return int((val < xi - 1e-12).sum()), float(val.min())


def ig_restrito(model, x, coef, steps=512):
    """IG do funcional g_A(X) = sum_i coef_i h_i(X) (referência nula), por variável."""
    base = torch.zeros_like(x)
    c = torch.tensor(coef, dtype=torch.float32)
    grads = torch.zeros_like(x, dtype=torch.float64)
    for a in (torch.arange(steps, dtype=torch.float64) + 0.5) / steps:
        xi = (base + a.float() * (x - base)).detach().requires_grad_(True)
        gA = (model(xi)["latent"] * c).sum()
        grads += torch.autograd.grad(gA, xi)[0].double()
    ig = ((x - base).double() * grads / steps).squeeze(0).numpy()      # L x d
    with torch.no_grad():
        delta = float((model(x)["latent"] * c).sum() - (model(base)["latent"] * c).sum())
    return ig, delta


def explicar(d, v2):
    model, Xtr, Xte = treinar(d)
    x = Xte[0:1]
    with torch.no_grad():
        out = model(x)
        Htr = model(Xtr)["latent"].double().numpy()
    h_ast = out["latent"][0].double().numpy()
    ref = v2[f"d{d}"]
    conf = {"max_dif_h": float(np.abs(h_ast - np.array(ref["h"])).max()),
            "max_dif_thetaT": float(np.abs(out["theta_T"][0].double().numpy() - np.array(ref["theta_T"])).max()),
            "max_dif_thetaS": float(np.abs(out["theta_S"][0].double().numpy() - np.array(ref["theta_S"])).max())}
    W, b, q1, m = afim_da_cabeca(model)
    head = model.head
    # Propriedade P_T: s * theta_{T,1}(h) >= |theta*_{T,1}| / 2
    wT, bT = W[1], b[1]
    th1 = float(wT @ h_ast + bT)
    s = 1.0 if th1 >= 0 else -1.0
    props = {"P_T": (s * wT, s * bT, abs(th1) / 2, {"theta_T1": th1, "sinal": s})}
    # Propriedade P_S: <e*, (a_1, b_1)> >= A*_1 / 2, harmônico k = 1 (período 8 h)
    a1w, b1w = W[q1], W[q1 + 1]
    a1, b1 = float(a1w @ h_ast + b[q1]), float(b1w @ h_ast + b[q1 + 1])
    A1 = float(np.hypot(a1, b1))
    e = np.array([a1, b1]) / A1
    props["P_S"] = (e[0] * a1w + e[1] * b1w, e[0] * b[q1] + e[1] * b[q1 + 1], A1 / 2,
                    {"a1": a1, "b1": b1, "A1": A1, "fase_rad": float(np.arctan2(b1, a1))})
    caixas = {"formal": (-np.ones(4), np.ones(4)), "empirica": (Htr.min(0), Htr.max(0))}
    # Camada densa equivalente + projeção (Proposição de invariância)
    Pm, Ftm, Qpm = (head.P.double().numpy(), head.Ft.double().numpy(), head.Qp.double().numpy())
    M = np.concatenate([Pm.T, Ftm.T, Qpm], 1)
    Wd, bd = M @ W, M @ b
    PT = np.linalg.solve(Pm @ Pm.T, Pm)                    # theta_T = (P P^T)^{-1} P y
    PS = np.linalg.solve(Ftm @ Ftm.T, Ftm)                 # theta_S = (Ft Ft^T)^{-1} Ft y
    wT_d, bT_d = (PT @ Wd)[1], (PT @ bd)[1]
    a1w_d, b1w_d = (PS @ Wd)[0], (PS @ Wd)[1]
    res = {"conferencia_json_v2": conf, "h": h_ast.tolist(), "propriedades": {}}
    for nome, (w, w0, xi, info) in props.items():
        res["propriedades"][nome] = {"info": info, "limiar": xi, "w": w.tolist(), "w0": float(w0),
                                     "valor_na_instancia": float(w @ h_ast + w0), "caixas": {}}
        if nome == "P_T":
            w_eq, w0_eq = s * wT_d, s * bT_d
        else:
            w_eq = e[0] * a1w_d + e[1] * b1w_d
            w0_eq = e[0] * (PS @ bd)[0] + e[1] * (PS @ bd)[1]
        for cx, (lo, hi) in caixas.items():
            delta, g0 = ganhos(w, w0, h_ast, lo, hi)
            A, margem = guloso(delta, g0, xi)
            bf = forca_bruta(w, w0, h_ast, lo, hi, xi)
            k_ig0, phi0 = ordem_atribuicao(w, h_ast, np.zeros(4), g0, delta, xi)
            k_igm, phim = ordem_atribuicao(w, h_ast, Htr.mean(0), g0, delta, xi)
            viol, vmin = amostrar_completamentos(w, w0, h_ast, lo, hi, A, xi)
            # sem o último elemento do prefixo guloso, a propriedade pode falhar?
            menor = None
            if A:
                ordem = list(np.argsort(-delta, kind="stable")[:len(A)])
                menor_A = sorted(int(j) for j in ordem[:-1])
                menor = amostrar_completamentos(w, w0, h_ast, lo, hi, menor_A, xi)
            d_eq, g0_eq = ganhos(w_eq, w0_eq, h_ast, lo, hi)
            A_eq, _ = guloso(d_eq, g0_eq, xi)
            res["propriedades"][nome]["caixas"][cx] = {
                "lo": np.asarray(lo).tolist(), "hi": np.asarray(hi).tolist(), "ganhos": delta.tolist(),
                "gamma_vazio": float(g0), "explicacao_gulosa": A, "margem_final": margem,
                "forca_bruta_minimos": bf, "guloso_e_minimo": (bf is not None and A in bf and len(A) == len(bf[0])),
                "k_ordem_IG_ref_nula": k_ig0, "phi_ref_nula": phi0.tolist(),
                "k_ordem_IG_ref_media": k_igm, "phi_ref_media": phim.tolist(),
                "violacoes_em_200k_completamentos": viol, "min_valor_amostrado": vmin,
                "sem_ultimo_elemento_violacoes_e_min": menor,
                "explicacao_densa_mais_projecao": A_eq,
                "max_dif_coef_densa_vs_nfh": float(max(np.abs(w_eq - w).max(), abs(w0_eq - w0)))}
    # Volta à entrada: IG restrito por variável, para P_S na caixa formal
    if d == 2:
        w, w0, xi, _ = props["P_S"]
        A = res["propriedades"]["P_S"]["caixas"]["formal"]["explicacao_gulosa"]
        coef = np.zeros(4)
        coef[A] = w[A]
        ig, delta = ig_restrito(model, x, coef)
        por_var = ig.sum(0)
        res["volta_entrada_P_S_formal"] = {
            "neuronios": A, "por_variavel": por_var.tolist(), "variaveis": ["OT", "HUFL"],
            "participacao_pct": (100 * np.abs(por_var) / np.abs(por_var).sum()).tolist(),
            "soma": float(ig.sum()), "delta": delta, "erro_completude": float(abs(ig.sum() - delta)),
            "HUFL_ultimas4h_pct_da_HUFL": float(100 * np.abs(ig[-4:, 1]).sum() / (np.abs(ig[:, 1]).sum() + 1e-30))}
    return res


def main():
    torch.set_num_threads(2)
    v2 = json.loads(JSON_V2.read_text())
    res = {"M43_defasagens_d1": distribuicao_defasagens(v2), "d1": explicar(1, v2), "d2": explicar(2, v2)}
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res["M43_defasagens_d1"], indent=1))
    for d in ("d1", "d2"):
        print(d, res[d]["conferencia_json_v2"])
        for nome, pr in res[d]["propriedades"].items():
            print(" ", nome, pr["info"], "limiar", round(pr["limiar"], 5), "valor", round(pr["valor_na_instancia"], 5))
            for cx, c in pr["caixas"].items():
                print("    ", cx, "A", c["explicacao_gulosa"], "bf", c["forca_bruta_minimos"], "k_IG0", c["k_ordem_IG_ref_nula"],
                      "k_IGm", c["k_ordem_IG_ref_media"], "viol", c["violacoes_em_200k_completamentos"],
                      "sem_ultimo", c["sem_ultimo_elemento_violacoes_e_min"], "eq", c["explicacao_densa_mais_projecao"],
                      "dif", f"{c['max_dif_coef_densa_vs_nfh']:.1e}", "g0", round(c["gamma_vazio"], 4),
                      "ganhos", np.round(c["ganhos"], 4).tolist())
    if "volta_entrada_P_S_formal" in res["d2"]:
        print(res["d2"]["volta_entrada_P_S_formal"])


if __name__ == "__main__":
    main()
