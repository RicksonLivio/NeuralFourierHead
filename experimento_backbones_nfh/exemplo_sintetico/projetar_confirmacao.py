"""Controle adicional (21/11, noite; auditoria externa, item 2): projeção da camada densa em T ⊕ S, sem treino, nas
sementes 10 a 19 (as da confirmação). Para cada semente, retreina a densa (mesmo código de comum.treinar_variantes,
mesmos hiperparâmetros) e avalia, no teste, ŷ_proj = ŷ - Q Qᵀ ŷ, isto é, a decomposição pura com o resíduo descartado
(tr.QQ é o projetor sobre o espaço residual, do código do exemplo de livro). Nada é treinado depois da projeção.
É um controle relatado: não participa da seleção nem muda o veredito pré-registrado. Grava projecao_confirmacao.csv e .json.
Confere que a densa retreinada reproduz test_RMSE/test_MAE de confirmacao.csv (tolerância 1e-5)."""
import csv
import json

import numpy as np
import pandas as pd
import torch

from comum import AQUI, dados, tr, treinar_variantes

SEMENTES = range(10, 20)


def main():
    sel = json.load(open(AQUI / "selecao.json"))
    p, n = sel["p"], sel["n"]
    _, T = dados()
    Xte, Yte = T["teste"]
    conf = pd.read_csv(AQUI / "confirmacao.csv")
    linhas = []
    for s in SEMENTES:
        densa = treinar_variantes(p, n, s, T, quais=["densa"])["densa"][0]
        with torch.no_grad():
            yh = densa(Xte)
            yp = yh - yh @ tr.QQ.T                    # ŷ - Q Qᵀ ŷ (QQ é simétrica)
            r = lambda y: float(torch.sqrt(((y - Yte) ** 2).mean()))   # noqa: E731
            m = lambda y: float((y - Yte).abs().mean())               # noqa: E731
            lin = {"p": p, "n": n, "semente": s, "densa_test_RMSE": r(yh), "densa_test_MAE": m(yh),
                   "proj_test_RMSE": r(yp), "proj_test_MAE": m(yp),
                   "frac_energia_residual_prev": float(((yh @ tr.QQ.T) ** 2).sum() / (yh ** 2).sum())}
        c = conf[(conf.semente == s) & (conf.variante == "densa")].iloc[0]
        assert abs(lin["densa_test_RMSE"] - c.test_RMSE) < 1e-5 and abs(lin["densa_test_MAE"] - c.test_MAE) < 1e-5, (s, lin, c)
        lin["dif_vs_confirmacao_RMSE"] = abs(lin["densa_test_RMSE"] - c.test_RMSE)
        linhas.append(lin)
        print("semente", s, {k: round(v, 5) for k, v in lin.items() if isinstance(v, float)}, flush=True)
    with open(AQUI / "projecao_confirmacao.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader(); w.writerows(linhas)
    json.dump({"E_EXTRA": tr.E_EXTRA, "LR": tr.LR, "LR_AJ": tr.LR_AJ, "EPOCAS": tr.EPOCAS, "PACIENCIA": tr.PACIENCIA,
               "LOTE": tr.LOTE}, open(AQUI / "protocolo_exemplo.json", "w"), indent=1)


if __name__ == "__main__":
    main()
