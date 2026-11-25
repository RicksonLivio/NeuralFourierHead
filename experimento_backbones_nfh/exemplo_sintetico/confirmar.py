"""Etapa 3: confirmação com sementes novas (10 a 19) da combinação escolhida em selecao.json (densa, densa+E, NFH escolhida).
Critério pré-registrado: SUCESSO se a média do RMSE e a do MAE de teste da NFH forem <= as da densa e <= as da densa+E, e a
NFH tiver RMSE de teste menor que o da densa+E em >= 8 das 10 sementes. Caso contrário, FALHA. Grava confirmacao.csv/.md/.json."""
import csv
import json
import time

import numpy as np

from comum import AQUI, dados, treinar_variantes

SEMENTES = range(10, 20)


def main():
    sel = json.load(open(AQUI / "selecao.json"))
    p, n, nfh = sel["p"], sel["n"], sel["variante"]
    _, T = dados()
    campos = ["p", "n", "semente", "variante", "val_RMSE", "val_MAE", "test_RMSE", "test_MAE", "epoca"]
    linhas = []
    for s in SEMENTES:
        t0 = time.time()
        res = treinar_variantes(p, n, s, T, quais=["densa", "densa+E", nfh])
        for v in ("densa", "densa+E", nfh):
            _, ep, m = res[v]
            linhas.append({"p": p, "n": n, "semente": s, "variante": "NFH" if v == nfh else v, **m, "epoca": ep})
        print(f"semente {s} ok ({time.time()-t0:.1f}s)", flush=True)
    with open(AQUI / "confirmacao.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)
    A = {v: {k: np.array([r[k] for r in linhas if r["variante"] == v]) for k in ("test_RMSE", "test_MAE")}
         for v in ("densa", "densa+E", "NFH")}
    out = {"p": p, "n": n, "variante_nfh": nfh, "qualificada_na_selecao": sel["qualificada"]}
    for v in A:
        for k in ("test_RMSE", "test_MAE"):
            out[f"{v}_{k}_media"], out[f"{v}_{k}_dp"] = float(A[v][k].mean()), float(A[v][k].std(ddof=1))
    for b in ("densa", "densa+E"):
        for k in ("test_RMSE", "test_MAE"):
            out[f"vitorias_{k}_vs_{b}"] = int((A["NFH"][k] < A[b][k]).sum())
    cond = [out[f"NFH_{k}_media"] <= out[f"{b}_{k}_media"] for k in ("test_RMSE", "test_MAE") for b in ("densa", "densa+E")]
    out["medias_ok"] = bool(all(cond))
    out["vitorias_ok"] = out["vitorias_test_RMSE_vs_densa+E"] >= 8
    out["veredito"] = "SUCESSO" if out["medias_ok"] and out["vitorias_ok"] else "FALHA"
    json.dump(out, open(AQUI / "confirmacao.json", "w"), indent=1)
    f3 = lambda x: f"{x:.4f}".replace(".", ",")  # noqa: E731
    md = ["# Confirmação (sementes 10 a 19, teste)", "",
          f"Combinação escolhida na validação (`selecao.md`): p = {p}, n = {n}, NFH = {nfh}. Critério pré-registrado (INSTRUCOES.md §10): "
          "média do RMSE e do MAE de teste da NFH ≤ as da densa e as da densa+E, e NFH com RMSE de teste menor que o da densa+E em ≥ 8 de 10 sementes.", "",
          "| variante | RMSE teste (média ± dp) | MAE teste (média ± dp) |", "|---|---|---|"]
    for v in A:
        md.append(f"| {v} | {f3(out[f'{v}_test_RMSE_media'])} ± {f3(out[f'{v}_test_RMSE_dp'])} | {f3(out[f'{v}_test_MAE_media'])} ± {f3(out[f'{v}_test_MAE_dp'])} |")
    md += ["", "Vitórias da NFH por semente (estritamente menor), de 10:", "",
           "| contra | RMSE | MAE |", "|---|---|---|"]
    for b in ("densa", "densa+E"):
        md.append(f"| {b} | {out[f'vitorias_test_RMSE_vs_{b}']} | {out[f'vitorias_test_MAE_vs_{b}']} |")
    md += ["", "Por semente (RMSE de teste):", "", "| semente | densa | densa+E | NFH |", "|---|---|---|---|"]
    for i, s in enumerate(SEMENTES):
        md.append(f"| {s} | {f3(A['densa']['test_RMSE'][i])} | {f3(A['densa+E']['test_RMSE'][i])} | {f3(A['NFH']['test_RMSE'][i])} |")
    md += ["", "Por semente (MAE de teste):", "", "| semente | densa | densa+E | NFH |", "|---|---|---|---|"]
    for i, s in enumerate(SEMENTES):
        md.append(f"| {s} | {f3(A['densa']['test_MAE'][i])} | {f3(A['densa+E']['test_MAE'][i])} | {f3(A['NFH']['test_MAE'][i])} |")
    md += ["", f"Médias cumprem o critério: {out['medias_ok']}. Vitórias ≥ 8: {out['vitorias_ok']}.", "", f"**Veredito: {out['veredito']}**"]
    (AQUI / "confirmacao.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
