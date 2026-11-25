#!/usr/bin/env python3
"""conferir_smoke.py <fase> [fila.json]: conta linhas, falhas, duplicatas e confere os asserts de um CSV de smoke."""
import csv, json, sys
from pathlib import Path
RES = Path(__file__).resolve().parents[1] / "resultados"
fase = sys.argv[1]
fila = Path(sys.argv[2]) if len(sys.argv) > 2 else RES / "filas" / "smoke.json"
rows = list(csv.DictReader(open(RES / f"{fase}.csv")))
tarefas = [t for t in json.loads(fila.read_text())["tarefas"] if t.get("tipo") != "ingenua"]
jl = [json.loads(x) for x in (RES / "logs" / f"{fase}_tarefas.jsonl").read_text().splitlines() if x.strip()]
tr = [r for r in jl if r["backbone"]]
chaves = [(r["dataset"], r["modo"], r["backbone"], r["largura"], r["fracao"], r["seed"], r["variante"]) for r in rows]
print(f"fase {fase}: linhas {len(rows)}, duplicadas {len(chaves) - len(set(chaves))}, tarefas de treino {len(tr)} "
      f"(ok {sum(r['ok'] for r in tr)}, com erro {sum(not r['ok'] for r in tr)}), esperadas {len(tarefas)}")
bbs = sorted({r["backbone"] for r in rows if r["backbone"]})
print("backbones:", len(bbs), bbs)
by = {}
for r in rows:
    if r["backbone"]:
        by.setdefault((r["dataset"], r["modo"], r["backbone"]), {})[r["variante"]] = r
esperadas = {v for t in tarefas for v in t["variantes"]}
faltam = [(k, v) for k, d in by.items() for v in esperadas if v not in d]
print("variantes faltantes:", len(faltam))
em, ec, ed = [], [], []
for k, d in by.items():
    ed.append(float(d["densa+E"]["val_RMSE"]) - float(d["densa"]["val_RMSE"]))
    for v, r in d.items():
        if v.startswith("conv_"):
            em.append(float(r["erro_conversao"])); ec.append(float(r["val_RMSE"]) - float(d["densa"]["val_RMSE"]))
print(f"erro máximo da conversão: {max(em):.3g} (limite 1e-5); max(val densa+E - val densa) = {max(ed):.3g} (assert <= 1e-7); "
      f"max(val conv - val densa) = {max(ec):.3g} (assert <= 1e-7)")
assert max(em) < 1e-5 and max(ed) <= 1e-7 and max(ec) <= 1e-7 and not faltam
print("asserts conferidos: OK")
