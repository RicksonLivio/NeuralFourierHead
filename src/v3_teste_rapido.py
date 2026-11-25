"""Teste rápido da infraestrutura v3: carrega os 8 conjuntos, confere janelas e a conversão
densa -> NFH em um modelo não treinado (erro deve ser ~1e-6), e o projetor (exatidão)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from v2_common import stack_ds  # noqa: E402
from v3_common import CONJUNTOS, Projetor, carregar, construir, densa_para_nfh, segmentos  # noqa: E402

for nome in CONJUNTOS:
    ds = carregar(nome)
    tr, va, te = segmentos(ds)
    print(nome, "d", ds["feats"].shape[1], "N", len(ds["alvo"]), "cortes", ds["cortes"], "janelas", len(tr), len(va),
          len(te), "teste_total", te.n_total, "obs", round(float(ds["observado"].mean()), 4),
          "nan", int(np.isnan(ds["feats"]).sum()), "datas", str(ds["datas"][0])[:16], str(ds["datas"][-1])[:16],
          flush=True)
ds = carregar("Jena-M")
tr, va, te = segmentos(ds)
X, Y = stack_ds(va)
for bb in ("MLP", "LSTM", "Conv1D+LSTM"):
    torch.manual_seed(0)
    m = construir(bb, "densa", X.shape[2]).eval()
    n = densa_para_nfh(m).eval()
    with torch.no_grad():
        a, b = m(X[:256])["forecast"], n(X[:256])
    print(bb, "erro conversão", float((a - b["forecast"]).abs().max()),
          "exatidão T+S+R", float((b["trend"] + b["seasonal"] + b["residual"] - b["forecast"]).abs().max()),
          "params", sum(p.numel() for p in m.parameters()), sum(p.numel() for p in n.parameters()))
pj = Projetor()
Z = np.random.default_rng(0).normal(size=(5, 48))
T, S, R = pj.decompor(Z)
print("projetor: soma", np.abs(T + S + R - Z).max(), "<T,S>", np.abs((T * S).sum(1)).max(), "<T,R>",
      np.abs((T * R).sum(1)).max(), "<S,R>", np.abs((S * R).sum(1)).max())
