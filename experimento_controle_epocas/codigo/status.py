#!/usr/bin/env python3
"""
status.py: progresso do controle das épocas extras, por dispositivo e por worker, com o que falta e um ETA grosseiro.

Uso: python status.py [fase]      (padrão: controle; também repro, smoke, smoke_cpu)
     python status.py controle --fila caminho.json

Progresso = tarefas com todas as variantes gravadas / tarefas da fila (treino e exata_s0). O ETA soma, sobre as tarefas
que faltam, a duração média observada por (dispositivo, backbone) (média por dispositivo se ainda não houver) dividida
pelos workers vivos daquele dispositivo; se não há duração observada, usa o custo da dissertação (densa + 2 x ajuste)
como estimativa, e o número dessas tarefas é impresso.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
BASE = ROOT / "experimento_controle_epocas" / "resultados"


def pasta(fase):
    return BASE if fase == "controle" else (BASE / "repro" if fase == "repro" else BASE / "smoke" / fase)


def vivo(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True


def ler(f):
    if not Path(f).exists():
        return []
    with open(f, newline="") as fh:
        return list(csv.DictReader(fh))


def san(s):
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", s)


def idt(t):
    return (f"exata_s0|{t['conjunto']}|{t['backbone']}|s0" if t["tipo"] == "exata_s0"
            else f"{t['conjunto']}|{t['backbone']}|s{t['seed']}")


def main():
    args = sys.argv[1:]
    fila_arg = None
    if "--fila" in args:
        i = args.index("--fila")
        fila_arg = args[i + 1]
        del args[i:i + 2]
    fase = args[0] if args else "controle"
    d = pasta(fase)
    fila_p = Path(fila_arg) if fila_arg else d / ("filas/controle.json" if fase == "controle" else "fila.json")
    if not fila_p.exists():
        print(f"fila não encontrada: {fila_p} (rode codigo/filas.py)")
        return
    fila = json.loads(fila_p.read_text())["tarefas"]
    ja = {"treino": {(r["conjunto"], r["backbone"], r["seed"], r["variante"]) for r in ler(d / "controle.csv")},
          "exata_s0": {(r["conjunto"], r["backbone"], r["seed"], r["variante"]) for r in ler(d / "exata_s0.csv")}}

    def completa(t):
        return all((t["conjunto"], t["backbone"], str(t["seed"]), v) in ja[t["tipo"]] for v in t["variantes"])

    feitas = [t for t in fila if completa(t)]
    falta = [t for t in fila if not completa(t)]
    print(f"=== fase {fase} ({time.strftime('%F %T')}) ===")
    print(f"tarefas: {len(feitas)}/{len(fila)} ({100 * len(feitas) / max(1, len(fila)):.1f}%); "
          f"linhas: {sum(len(v) for v in ja.values())}")
    for dev in ("GPU", "CPU"):
        tot = [t for t in fila if t["dispositivo"] == dev]
        print(f"  {dev}: {sum(completa(t) for t in tot)}/{len(tot)}")
    print(f"FALTA: {len(falta)} tarefa(s): {sum(t['tipo'] == 'treino' for t in falta)} de treino, "
          f"{sum(t['tipo'] == 'exata_s0' for t in falta)} exata_s0")
    s0 = [t for t in fila if t["estagio"] == 0]
    print(f"  estágio da semente 0 (treino + exata_s0): {sum(completa(t) for t in s0)}/{len(s0)}")
    parc = [t for t in falta if any((t["conjunto"], t["backbone"], str(t["seed"]), v) in ja[t["tipo"]] for v in t["variantes"])]
    for t in parc[:10]:
        fl = [v for v in t["variantes"] if (t["conjunto"], t["backbone"], str(t["seed"]), v) not in ja[t["tipo"]]]
        print(f"  parcial {idt(t)}: faltam {', '.join(fl)}")
    erros = sorted((d / "logs").glob("erro_*.txt")) if (d / "logs").exists() else []
    if erros:
        print(f"ERROS: {len(erros)} arquivo(s) em {d / 'logs'}: " + ", ".join(e.name for e in erros[:6]) + (" ..." if len(erros) > 6 else ""))
    jl = d / "logs" / "tarefas.jsonl"
    reg = [json.loads(x) for x in jl.read_text().splitlines() if x.strip()] if jl.exists() else []
    claims = {}
    cd = d / "claims"
    if cd.exists():
        for c in cd.glob("*.claim"):
            try:
                claims[c.stem] = json.loads(c.read_text())
            except Exception:  # noqa: BLE001
                pass
    vivos = {}
    pd_ = BASE / "pids"
    for p in (pd_.glob(f"{fase}_[gc][0-9]*.pid") if pd_.exists() else []):
        try:
            pid = int(p.read_text().strip())
        except ValueError:
            continue
        if vivo(pid):
            vivos[p.stem[len(fase) + 1:]] = pid
    print(f"workers vivos (PID files): {len(vivos)} ({', '.join(sorted(vivos))}); "
          f"GPU {sum(w.startswith('g') for w in vivos)}, CPU {sum(w.startswith('c') for w in vivos)}")
    por_w = defaultdict(list)
    for r in reg:
        por_w[(r["worker"], r["device"])].append(r)
    print("por worker/dispositivo: concluídas | com erro | duração média (s)")
    for (w, dev), rs in sorted(por_w.items()):
        ok = [r["duracao_s"] for r in rs if r.get("ok")]
        print(f"  {w}/{dev}: {len(ok)} | {sum(1 for r in rs if not r.get('ok'))} | {(sum(ok) / len(ok) if ok else 0):.0f}")
    ids_reg = {san(r["tarefa"]) for r in reg}
    for c, v in sorted(claims.items(), key=lambda kv: kv[1].get("t0", 0)):
        if vivo(v.get("pid")) and c not in ids_reg:
            print(f"  em andamento (pid {v['pid']}): {c} há {(time.time() - v['t0']) / 60:.1f} min")
    dur = defaultdict(list)
    for r in reg:
        if r.get("ok"):
            dur[(r["device"], r["backbone"])].append(r["duracao_s"])
    dur_dev = defaultdict(list)
    for (dev, _), v in dur.items():
        dur_dev[dev] += v
    resta, estimado = {"cuda": 0.0, "cpu": 0.0}, {"cuda": 0, "cpu": 0}
    for t in falta:
        dev = "cuda" if t["dispositivo"] == "GPU" else "cpu"
        v = dur.get((dev, t["backbone"])) or dur_dev.get(dev)
        if v:
            resta[dev] += sum(v) / len(v)
        else:
            resta[dev] += t.get("custo_s", 0)
            estimado[dev] += 1
    for dev, nome in (("cuda", "GPU"), ("cpu", "CPU")):
        n = max(1, sum(1 for w in vivos if w.startswith("g" if dev == "cuda" else "c")))
        print(f"ETA {nome}: ~{resta[dev] / n / 3600:.2f} h com {n} worker(s) "
              f"({estimado[dev]} tarefa(s) sem duração observada usam o custo da dissertação)")


if __name__ == "__main__":
    main()
