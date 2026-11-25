"""Etapa 2: regra do pré-registro (INSTRUCOES.md §10, 21/11/2026 17h20), aplicada LENDO SÓ as colunas val_*.

Para cada (p, n, variante NFH em {conv_L1000, zero_L0, zero_L1000}) e cada semente: diferença relativa do RMSE (e do MAE)
de validação em relação à densa+E, (NFH - densa+E) / densa+E. Qualifica se: média do RMSE relativo < 0, média do MAE
relativo <= 0 e NFH com RMSE de validação menor que o da densa+E nas 5 sementes. Escolha: menor média do RMSE relativo
entre as qualificadas; empate (diferença < 1e-12) pelo menor p. Se nenhuma qualificar, usa-se a de menor média do RMSE
relativo de validação, marcada como 'não qualificada' (o documento então diz que a NFH não superou a densa).
Saída: selecao.md e selecao.json (para confirmar.py)."""
import json

import pandas as pd

from comum import AQUI

NFHS = ["conv_L1000", "zero_L0", "zero_L1000"]


def main():
    d = pd.read_csv(AQUI / "selecao.csv", usecols=["p", "n", "semente", "variante", "val_RMSE", "val_MAE"])
    assert not any(c.startswith("test") for c in d.columns)
    linhas = []
    for (p, n), g in d.groupby(["p", "n"]):
        base = g[g.variante == "densa+E"].set_index("semente")
        dens = g[g.variante == "densa"].set_index("semente")
        for v in NFHS:
            x = g[g.variante == v].set_index("semente")
            rr = (x.val_RMSE - base.val_RMSE) / base.val_RMSE
            rm = (x.val_MAE - base.val_MAE) / base.val_MAE
            linhas.append({"p": int(p), "n": int(n), "variante": v, "rel_RMSE": rr.mean(), "rel_MAE": rm.mean(),
                           "melhor_em": int((x.val_RMSE < base.val_RMSE).sum()),
                           "val_RMSE_nfh": x.val_RMSE.mean(), "val_RMSE_densaE": base.val_RMSE.mean(),
                           "val_RMSE_densa": dens.val_RMSE.mean(), "val_MAE_nfh": x.val_MAE.mean(),
                           "val_MAE_densaE": base.val_MAE.mean()})
    t = pd.DataFrame(linhas)
    t["qualifica"] = (t.rel_RMSE < 0) & (t.rel_MAE <= 0) & (t.melhor_em == 5)
    pool = t[t.qualifica] if t.qualifica.any() else t
    menor = pool.rel_RMSE.min()
    emp = pool[pool.rel_RMSE - menor < 1e-12].sort_values("p")
    esc = emp.iloc[0]
    qual = bool(t.qualifica.any())
    json.dump({"p": int(esc.p), "n": int(esc.n), "variante": esc.variante, "qualificada": qual,
               "rel_RMSE": float(esc.rel_RMSE), "rel_MAE": float(esc.rel_MAE), "empate_com": int(len(emp) - 1)},
              open(AQUI / "selecao.json", "w"), indent=1)
    f = lambda x: f"{x:.4f}".replace(".", ",")  # noqa: E731
    md = ["# Seleção da combinação (só validação)", "",
          "Regra pré-registrada em `experimento_backbones_nfh/INSTRUCOES.md` §10 (21/11/2026, 17h20). Este script lê só as colunas",
          "`val_*` de `selecao.csv`. Diferença relativa = (NFH − densa+E) / densa+E, média das 5 sementes (0 a 4).", "",
          "| p | n | variante NFH | RMSE val NFH | RMSE val densa+E | RMSE val densa | rel. RMSE | rel. MAE | sementes com NFH melhor | qualifica |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.sort_values(["p", "n", "variante"]).iterrows():
        md.append(f"| {r.p} | {r.n} | {r.variante} | {f(r.val_RMSE_nfh)} | {f(r.val_RMSE_densaE)} | {f(r.val_RMSE_densa)} | "
                  f"{f(r.rel_RMSE)} | {f(r.rel_MAE)} | {r.melhor_em}/5 | {'sim' if r.qualifica else 'não'} |")
    md += ["", "## Escolha", "",
           f"**p = {int(esc.p)}, n = {int(esc.n)}, variante {esc.variante}** (rel. RMSE de validação {f(esc.rel_RMSE)}; rel. MAE {f(esc.rel_MAE)}).", ""]
    if qual:
        md.append(f"Motivo: entre as {int(t.qualifica.sum())} combinações que cumprem a regra (média do RMSE relativo < 0, média do MAE relativo ≤ 0, "
                  "NFH melhor na validação nas 5 sementes), é a de menor média do RMSE relativo de validação."
                  + (f" Houve empate exato com {len(emp)-1} combinação(ões); venceu o menor p." if len(emp) > 1 else ""))
    else:
        md.append("Motivo: **nenhuma combinação cumpre a regra** de qualificação. Usa-se, só para que o documento exista, a de menor média do "
                  "RMSE relativo de validação, marcada como não qualificada. O documento dirá que a NFH não superou a densa na validação.")
    (AQUI / "selecao.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
