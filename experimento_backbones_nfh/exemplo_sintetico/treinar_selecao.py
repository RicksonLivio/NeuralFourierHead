"""Etapa 1 (pré-registro de 21/11/2026, 17h20): grade de seleção, p in {8,16}, n in {300,1000}, sementes 0 a 4.
Grava selecao.csv com val_RMSE, val_MAE, test_RMSE, test_MAE por linha (o teste é gravado mas NUNCA lido por selecionar.py)."""
import csv
import time

from comum import AQUI, VARIANTES, dados, treinar_variantes

PS, NS, SEMENTES = (8, 16), (300, 1000), range(5)


def main():
    _, T = dados()
    campos = ["p", "n", "semente", "variante", "val_RMSE", "val_MAE", "test_RMSE", "test_MAE", "epoca", "tempo_s"]
    with open(AQUI / "selecao.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for p in PS:
            for n in NS:
                for s in SEMENTES:
                    t0 = time.time()
                    res = treinar_variantes(p, n, s, T)
                    dt = time.time() - t0
                    for v in VARIANTES:
                        _, ep, m = res[v]
                        w.writerow({"p": p, "n": n, "semente": s, "variante": v, **m, "epoca": ep, "tempo_s": round(dt, 2)})
                    f.flush()
                    print(f"p={p} n={n} s={s} ok ({dt:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
