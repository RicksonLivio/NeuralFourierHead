#!/usr/bin/env python3
"""completo.py: confere se todas as variantes de uma fila estão no CSV da fase.

Uso: python completo.py <fase> <fila.json>   -> imprime 'faltam=<n>' (variantes de treino sem linha; ingênuas contam
como uma variante cada) e termina com código 0 se faltam=0, 1 caso contrário. Não lê nenhuma métrica.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "resultados"


def fmt_frac(f):
    return f"{float(f):g}"


def main():
    fase, fila = sys.argv[1], sys.argv[2]
    tarefas = json.load(open(fila))["tarefas"]
    f = RES / f"{fase}.csv"
    linhas = []
    if f.exists():
        with open(f, newline="") as fh:
            linhas = [r for r in csv.DictReader(fh) if r.get("dataset") != "dataset"]
    chaves = {(r["dataset"], r["modo"], r["backbone"], str(r["largura"]), str(r["fracao"]), str(r["seed"]), r["variante"])
              for r in linhas}
    ing = {(r["dataset"], r["modo"]) for r in linhas if r["variante"].startswith("ingenua")}
    faltam = 0
    for t in tarefas:
        if t.get("tipo") == "ingenua":
            faltam += (t["dataset"], t["modo"]) not in ing
            continue
        for v in t["variantes"]:
            k = (t["dataset"], t["modo"], t["backbone"], str(t["largura"]), fmt_frac(t["fracao"]), str(t["seed"]), v)
            faltam += k not in chaves
    print(f"faltam={faltam}")
    sys.exit(0 if faltam == 0 else 1)


if __name__ == "__main__":
    main()
