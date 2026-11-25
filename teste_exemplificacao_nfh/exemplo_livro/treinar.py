"""Treino no exemplo de livro: densa, densa+E, NFH por conversão + ajuste e NFH do zero, com o mesmo backbone.

Backbone: h = tanh(W_phi vec(X) + b_phi), p = 6. Cabeça densa: Linear(p, 8). NFH: Linear(p, 8) seguida de
c -> M D c (colunas normalizadas, como na dissertação), mesmos 8(p+1) parâmetros.
Critério de parada e de escolha de época: RMSE de validação (o teste nunca é usado para escolher nada).
Grade pré-registrada: n_treino em {100, 300, 1000, todas}; sementes 0..9; lambda da NFH em {1000}; E = 50 épocas extras
(densa+E e ajuste estrutural, Adam com taxa 1/10 da original). NFH do zero com lambda em {0, 1000}.
Saída: resultados_treino.csv (uma linha por variante, n_treino e semente) e modelos da semente 0 com n = 1000.
"""
from __future__ import annotations

import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from gerar import H, K, L, D, Q, bases, oraculo, particoes  # noqa: E402

torch.set_num_threads(1)
import os
P_LAT = int(os.environ.get("LIVRO_P", "6"))
LR, LR_AJ, EPOCAS, PACIENCIA, E_EXTRA, LOTE = 3e-3, 3e-4, 400, 40, 50, 64
LAMBDA = 1000.0
N_TREINO = [None if x == "todas" else int(x) for x in os.environ.get("LIVRO_N", "100,300,1000,todas").split(",")]
SEMENTES = list(range(10))
P, F, Ft, Qp, M = bases()
NORMAS = np.concatenate([np.linalg.norm(P, axis=1), np.linalg.norm(Ft, axis=1), np.ones(H - (Q + 1 + 2 * K))])
MD = torch.tensor(M / NORMAS[None, :], dtype=torch.float32)        # c' -> y = M D c'
QQ = torch.tensor(Qp @ Qp.T, dtype=torch.float32)


class Modelo(nn.Module):
    def __init__(self, cabeca):
        super().__init__()
        self.phi = nn.Linear(L * D, P_LAT)
        self.lin = nn.Linear(P_LAT, H)
        self.cabeca = cabeca

    def h(self, x):
        return torch.tanh(self.phi(x.reshape(len(x), -1)))

    def forward(self, x):
        c = self.lin(self.h(x))
        return c @ MD.T if self.cabeca == "nfh" else c


def residuo(yhat):
    return yhat @ QQ.T


def rmse(m, X, Y):
    with torch.no_grad():
        return float(torch.sqrt(((m(X) - Y) ** 2).mean()))


def treinar(m, Xtr, Ytr, Xv, Yv, semente, lr, epocas, lam=0.0, paciencia=None, incluir_inicio=False):
    """Adam; a cada época mede o RMSE de validação; devolve o estado de menor RMSE de validação.
    incluir_inicio=True torna o estado inicial (época 0) candidato, como na seleção segura."""
    g = torch.Generator().manual_seed(semente)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-5)
    melhor, estado, ep_melhor, ruins = (rmse(m, Xv, Yv), copy.deepcopy(m.state_dict()), 0, 0) if incluir_inicio \
        else (float("inf"), None, -1, 0)
    for ep in range(1, epocas + 1):
        m.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), LOTE):
            j = perm[i:i + LOTE]
            o = m(Xtr[j])
            perda = ((o - Ytr[j]) ** 2).mean() + lam * (residuo(o) ** 2).mean()
            opt.zero_grad(); perda.backward(); opt.step()
        r = rmse(m, Xv, Yv)
        if r < melhor - 1e-7:
            melhor, estado, ep_melhor, ruins = r, copy.deepcopy(m.state_dict()), ep, 0
        else:
            ruins += 1
            if paciencia and ruins >= paciencia:
                break
    m.load_state_dict(estado)
    return m, melhor, ep_melhor


def converter(densa):
    """Conversão exata (Teorema de equivalência): W_NFH = (MD)^{-1} W', b_NFH = (MD)^{-1} b'."""
    nfh = copy.deepcopy(densa); nfh.cabeca = "nfh"
    MDi = torch.linalg.inv(MD.double())
    nfh.lin.weight.data = (MDi @ densa.lin.weight.data.double()).float()
    nfh.lin.bias.data = (MDi @ densa.lin.bias.data.double()).float()
    return nfh


def ajuste_seguro(nfh, Xtr, Ytr, Xv, Yv, semente, lam):
    """Ajuste estrutural com seleção segura: menor J de validação entre as épocas com RMSE_val <= RMSE_val(época 0)."""
    g = torch.Generator().manual_seed(semente)
    opt = torch.optim.Adam(nfh.parameters(), lr=LR_AJ, weight_decay=1e-5)

    def J(m):
        with torch.no_grad():
            o = m(Xv)
            return float(((o - Yv) ** 2).mean() + lam * (residuo(o) ** 2).mean()), float(torch.sqrt(((o - Yv) ** 2).mean()))
    j0, r0 = J(nfh)
    melhor = (j0, copy.deepcopy(nfh.state_dict()), 0)
    for ep in range(1, E_EXTRA + 1):
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), LOTE):
            k = perm[i:i + LOTE]
            o = nfh(Xtr[k])
            perda = ((o - Ytr[k]) ** 2).mean() + lam * (residuo(o) ** 2).mean()
            opt.zero_grad(); perda.backward(); opt.step()
        j, r = J(nfh)
        if r <= r0 and j < melhor[0]:
            melhor = (j, copy.deepcopy(nfh.state_dict()), ep)
    nfh.load_state_dict(melhor[1])
    return nfh, melhor[2]


def cobertura(m, X):
    with torch.no_grad():
        o = m(X).double()
    R = o @ QQ.double().T
    oc = o - o.mean(1, keepdim=True)
    return float(torch.median(1 - (R ** 2).sum(1) / (oc ** 2).sum(1)))


def main():
    part = particoes()
    T = {k: (torch.tensor(v[0], dtype=torch.float32), torch.tensor(v[1], dtype=torch.float32)) for k, v in part.items()}
    Xv, Yv = T["val"]; Xte, Yte = T["teste"]
    or_te = float(np.sqrt(np.mean((oraculo(part["teste"][0]) - part["teste"][1]) ** 2)))
    saida = AQUI / os.environ.get("LIVRO_SAIDA", "resultados_treino.csv")
    campos = ["n_treino", "semente", "variante", "rmse_teste", "rmse_val", "razao_oraculo", "cobertura_teste", "epoca", "tempo_s"]
    with open(saida, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos); w.writeheader()
        w.writerow({"n_treino": "", "semente": "", "variante": "oráculo", "rmse_teste": or_te, "razao_oraculo": 1.0})
        for n in N_TREINO:
            Xtr, Ytr = T["treino"]
            if n is not None:
                Xtr, Ytr = Xtr[:n], Ytr[:n]
            for s in SEMENTES:
                def linha(var, m, ep, t0):
                    r = rmse(m, Xte, Yte)
                    w.writerow({"n_treino": len(Xtr), "semente": s, "variante": var, "rmse_teste": r,
                                "rmse_val": rmse(m, Xv, Yv), "razao_oraculo": r / or_te,
                                "cobertura_teste": cobertura(m, Xte), "epoca": ep, "tempo_s": time.time() - t0})
                t0 = time.time(); torch.manual_seed(s)
                densa, _, ep = treinar(Modelo("densa"), Xtr, Ytr, Xv, Yv, s, LR, EPOCAS, paciencia=PACIENCIA)
                linha("densa", densa, ep, t0)
                t0 = time.time()
                de, _, ep = treinar(copy.deepcopy(densa), Xtr, Ytr, Xv, Yv, s, LR_AJ, E_EXTRA, incluir_inicio=True)
                linha("densa+E", de, ep, t0)
                t0 = time.time()
                conv = converter(densa)
                assert abs(rmse(conv, Xte, Yte) - rmse(densa, Xte, Yte)) < 1e-5
                aj, ep = ajuste_seguro(conv, Xtr, Ytr, Xv, Yv, s, LAMBDA)
                linha("NFH (conversão + ajuste)", aj, ep, t0)
                for lam in (0.0, LAMBDA):
                    t0 = time.time(); torch.manual_seed(s)
                    z, _, ep = treinar(Modelo("nfh"), Xtr, Ytr, Xv, Yv, s, LR, EPOCAS, lam=lam, paciencia=PACIENCIA)
                    linha(f"NFH do zero (lambda={lam:g})", z, ep, t0)
                    if n == 1000 and s == 0 and lam == LAMBDA and "LIVRO_SAIDA" not in os.environ:
                        torch.save(z.state_dict(), AQUI / "modelo_nfh_livro.pt")
                if n == 1000 and s == 0 and "LIVRO_SAIDA" not in os.environ:
                    torch.save(densa.state_dict(), AQUI / "modelo_densa_livro.pt")
                f.flush()
                print(f"n={len(Xtr)} s={s} ok", flush=True)


if __name__ == "__main__":
    main()
# Regimes pré-registrados (16/11/2026, antes de ver os resultados do regime B):
#   A (padrão): p = 6 = m, sigma = 0,1, n em {100, 300, 1000, todas}: cabeça do tamanho do espaço estrutural.
#   B: p = 32 > H, sigma = 0,3, n em {300, 1000, 3000}: cabeça larga e ruído alto, em que a densa pode ajustar ruído
#      nas direções do resíduo e a penalização da NFH deveria reduzir variância. Saída: resultados_treino_B.csv
