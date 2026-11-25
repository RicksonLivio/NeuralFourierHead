"""
Exemplo pequeno e totalmente rastreável (pedido do orientador, itens 6.6 e 8.8):
f(x_1, ..., x_L) = (y_1, ..., y_H) com L, d, H pequenos, dados reais (ETTh1).

Casos: d = 1 (OT) e d = 2 (OT, HUFL). L = 12 h, H = 8 h, K = 2 (períodos 8 h e
4 h; 2K = 4 < 8), q = 1 (tendência afim), p = 4 (dimensão latente).
Backbone mínimo Phi(x) = tanh(W vec(x) + b) para que todos os números possam
ser impressos. Atribuição por variável/defasagem via Integrated Gradients
aplicado a funcionais escalares de cada componente, com verificação numérica
da completude (soma das atribuições = g(x) - g(x')).

Saída: results/v2/small_example.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from v2_common import RES, NFHHead, WindowDataset, load_etth1, save_json, set_seed, stack_ds  # noqa: E402

L, H, K, Q, P_LAT = 12, 8, 2, 1, 4


class Tiny(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(L * d, P_LAT)
        self.head = NFHHead(P_LAT, H, K, Q)

    def forward(self, x):
        h = torch.tanh(self.W(x.reshape(x.shape[0], -1)))
        out = self.head(h)
        out["latent"] = h
        return out


def functionals(out):
    """g_T = média de T no horizonte (nível previsto), g_S = ||S||^2, g_R = ||R||^2."""
    return {"T": out["trend"].mean(1), "S": (out["seasonal"] ** 2).sum(1),
            "R": (out["residual"] ** 2).sum(1)}


def integrated_gradients(model, x, key, steps=512):
    base = torch.zeros_like(x)
    alphas = (torch.arange(steps, dtype=torch.float64) + 0.5) / steps
    grads = torch.zeros_like(x, dtype=torch.float64)
    for a in alphas:
        xi = (base + a.float() * (x - base)).detach().requires_grad_(True)
        g = functionals(model(xi))[key].sum()
        grads += torch.autograd.grad(g, xi)[0].double()
    ig = (x - base).double() * grads / steps
    with torch.no_grad():
        delta = float(functionals(model(x))[key] - functionals(model(base))[key])
    return ig.squeeze(0).numpy(), delta


def run(d):
    set_seed(0)
    feats, target, names = load_etth1(multivariate=True)
    cols = [names.index("OT")] if d == 1 else [names.index("OT"), names.index("HUFL")]
    f = feats[:, cols]
    tr = WindowDataset(f[:8640], target[:8640], L, H)
    va = WindowDataset(f[8640:11520], target[8640:11520], L, H)
    te = WindowDataset(f[11520:], target[11520:], L, H)
    model = Tiny(d)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    Xtr, Ytr = stack_ds(tr)
    Xva, Yva = stack_ds(va)
    best, state = 1e9, None
    g = torch.Generator().manual_seed(0)
    for ep in range(40):
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
    Xte, Yte = stack_ds(te)
    with torch.no_grad():
        rmse = float(torch.sqrt(((model(Xte)["forecast"] - Yte) ** 2).mean()))
    # amostra explicada: primeira janela do conjunto de teste
    i0 = 0
    x = Xte[i0:i0 + 1]
    with torch.no_grad():
        out = model(x)
    head = model.head
    r = lambda a, n=4: np.round(np.asarray(a, dtype=float), n).tolist()
    yhat, T, S, R = (out[k][0].numpy() for k in ("forecast", "trend", "seasonal", "residual"))
    # identificabilidade: recuperar coeficientes a partir de y_hat
    Pn, Ftn, Fn, Qpn = (head.P.numpy().astype(np.float64), head.Ft.numpy().astype(np.float64),
                        head.F.numpy().astype(np.float64), head.Qp.numpy().astype(np.float64))
    Bfull = np.concatenate([Pn.T, Ftn.T, Qpn], 1)
    coef = np.linalg.solve(Bfull, yhat.astype(np.float64))
    internal = np.concatenate([out["theta_T"][0].numpy(), out["theta_S"][0].numpy(), out["u"][0].numpy()])
    yc = yhat - yhat.mean()
    Tc = T - T.mean()
    E = float((yc ** 2).sum())
    energy = {"eps_T": float((Tc ** 2).sum() / E), "eps_S": float((S ** 2).sum() / E),
              "eps_R": float((R ** 2).sum() / E), "iota": float(2 * (Tc * S).sum() / E)}
    # contraexemplo motivador: mesma previsão decomposta na base NÃO ortogonalizada [P, F, Qp]
    craw = np.linalg.solve(np.concatenate([Pn.T, Fn.T, Qpn], 1), yhat.astype(np.float64))
    Traw, Sraw = craw[:Q + 1] @ Pn, craw[Q + 1:Q + 1 + 2 * K] @ Fn
    Trc = Traw - Traw.mean()
    energy_raw = {"eps_T": float((Trc ** 2).sum() / E), "eps_S": float((Sraw ** 2).sum() / E),
                  "eps_R": float((R ** 2).sum() / E), "iota": float(2 * (Trc * Sraw).sum() / E)}
    amps = out["amplitude"][0].numpy()
    th = out["theta_S"][0].numpy()
    harm = []
    for k in range(K):
        sk = th[2 * k] * Ftn[2 * k] + th[2 * k + 1] * Ftn[2 * k + 1]
        harm.append({"k": k + 1, "periodo_h": H / (k + 1), "a": float(th[2 * k]), "b": float(th[2 * k + 1]),
                     "A": float(amps[k]), "fase_rad": float(out["phase"][0, k]),
                     "contrib_S": float((sk * S).sum() / E), "energia_propria": float((sk ** 2).sum() / E),
                     "limite_superior": float(H / 2 * amps[k] ** 2 / E)})
    attrib = {}
    for key in ("T", "S", "R"):
        ig, delta = integrated_gradients(model, x, key)
        net = ig.sum(0)  # I_c(v) = soma sobre defasagens (definição do Cap. 4)
        attrib[key] = {"ig": r(ig, 5), "soma_ig": float(ig.sum()), "delta_g": delta,
                       "erro_completude": float(abs(ig.sum() - delta)),
                       "por_variavel": r(net, 5),
                       "participacao_liquida_pct": r(100 * np.abs(net) / (np.abs(net).sum() + 1e-30), 1),
                       "por_variavel_pct": r(100 * np.abs(ig).sum(0) / (np.abs(ig).sum() + 1e-12), 1)}
    return {
        "d": d, "variaveis": [names[c] for c in cols], "L": L, "H": H, "K": K, "q": Q, "p": P_LAT,
        "rmse_teste": rmse, "n_teste": len(te),
        "x": r(x[0].numpy()), "y_real": r(Yte[i0].numpy()),
        "h": r(out["latent"][0].numpy()), "theta_T": r(out["theta_T"][0].numpy()),
        "theta_S": r(out["theta_S"][0].numpy()), "u": r(out["u"][0].numpy()),
        "y_hat": r(yhat), "T": r(T), "S": r(S), "R": r(R),
        "exatidao_max_err": float(np.abs(yhat - (T + S + R)).max()),
        "identificabilidade_max_err": float(np.abs(coef - internal).max()),
        "energia": energy, "soma_energia": float(sum(energy.values())),
        "energia_base_nao_ortogonal": energy_raw, "T_base_nao_ortogonal": r(Traw),
        "S_base_nao_ortogonal": r(Sraw),
        "harmonicos": harm, "media_y_hat": float(yhat.mean()), "media_T": float(T.mean()),
        "atribuicao": attrib,
        "P": r(head.P.numpy()), "F": r(head.F.numpy()), "Ft": r(head.Ft.numpy()), "Qp": r(head.Qp.numpy()),
        "n_parametros": int(sum(p.numel() for p in model.parameters())),
    }


if __name__ == "__main__":
    res = {"d1": run(1), "d2": run(2)}
    save_json(res, RES / "small_example.json")
    for k, v in res.items():
        print(k, "rmse", round(v["rmse_teste"], 4), "energia", {a: round(b, 3) for a, b in v["energia"].items()},
              "raw", {a: round(b, 3) for a, b in v["energia_base_nao_ortogonal"].items()},
              "harm", [(h["k"], round(h["contrib_S"], 3), round(h["limite_superior"], 3)) for h in v["harmonicos"]],
              "exat", v["exatidao_max_err"], "ident", v["identificabilidade_max_err"],
              {c: (v["atribuicao"][c]["por_variavel_pct"], round(v["atribuicao"][c]["erro_completude"], 6)) for c in "TSR"})
