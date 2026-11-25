#!/usr/bin/env python3
"""
status.py: progresso por fase e por worker, com ETA.

Uso: python status.py [fase ...] [--fila caminho.json]   (padrão: t1 t2 confirmacao, só as que têm fila ou CSV)
  --fila: fila a comparar com o CSV (padrão resultados/filas/<fase>.json; para a T1 do corte B2 use filas/t1_B2.json).
Também mostra o que FALTA em relação à fila: tarefas sem nenhuma linha, tarefas parciais (com as variantes que faltam),
por backbone e por conjunto.

Progresso = tarefas com todas as variantes em resultados/<fase>.csv / tarefas da fila (ingênuas incluídas).
Por worker: tarefas concluídas, tarefa atual (reivindicação mais recente sem conclusão), duração média, PID vivo.
ETA = soma, sobre as tarefas restantes, da duração média observada para o mesmo backbone (média geral se ainda
não houver) dividida pelo número de workers vivos. É uma estimativa grosseira; tarefas grandes entram primeiro.
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
RES = ROOT / "experimento_backbones_nfh" / "resultados"
sys.path.insert(0, str(AQUI))


def vivo(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError, ValueError, TypeError):
        return False


def chave(t, v=None):
    if t.get("tipo") == "ingenua":
        return None
    return (t["dataset"], t["modo"], t["backbone"], str(t["largura"]), f"{float(t['fracao']):g}", str(t["seed"]))


def concluidas(fase):
    f = RES / f"{fase}.csv"
    feitas = set()
    if f.exists():
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                feitas.add((r["dataset"], r["modo"], r["backbone"], str(r["largura"]), str(r["fracao"]),
                            str(r["seed"]), r["variante"]))
    return feitas


def tarefa_completa(t, feitas):
    if t.get("tipo") == "ingenua":
        return all((t["dataset"], t["modo"], "", "", "", "", v) in feitas
                   for v in ("ingenua_media", "ingenua_ultimo", "ingenua_sazonal24"))
    k = chave(t)
    return all((*k, v) in feitas for v in t["variantes"])


def faltas(fila, feitas):
    """(tarefas sem nenhuma linha, parciais {id: [variantes que faltam]}, nº de variantes faltantes) em relação à fila."""
    sem, parc, n_var = [], {}, 0
    for t in fila:
        if t.get("tipo") == "ingenua":
            vs = [v for v in ("ingenua_media", "ingenua_ultimo", "ingenua_sazonal24")
                  if (t["dataset"], t["modo"], "", "", "", "", v) not in feitas]
            nome = f"ing|{t['dataset']}|{t['modo']}"
            tot = 3
        else:
            k = chave(t)
            vs = [v for v in t["variantes"] if (*k, v) not in feitas]
            nome = "|".join(k)
            tot = len(t["variantes"])
        n_var += len(vs)
        if vs and len(vs) == tot:
            sem.append((nome, t))
        elif vs:
            parc[nome] = vs
    return sem, parc, n_var


def relatorio(fase, fila_arg=None):
    fila_p = Path(fila_arg) if fila_arg else RES / "filas" / f"{fase}.json"
    if not fila_p.exists() and not (RES / f"{fase}.csv").exists():
        return None
    out = [f"=== fase {fase} ==="]
    feitas = concluidas(fase)
    jl = RES / "logs" / f"{fase}_tarefas.jsonl"
    reg = [json.loads(x) for x in jl.read_text().splitlines() if x.strip()] if jl.exists() else []
    claims = {}
    cd = RES / "claims" / fase
    if cd.exists():
        for c in cd.glob("*.claim"):
            try:
                claims[c.stem] = json.loads(c.read_text())
            except Exception:
                pass
    vivos = {c["pid"] for c in claims.values() if vivo(c.get("pid"))}
    out.append(f"linhas no CSV: {len(feitas)}; workers vivos (por reivindicação): {len(vivos)}")
    erros = list((RES / "logs").glob(f"erro_{fase}_*.txt"))
    if erros:
        out.append(f"ERROS: {len(erros)} (ver resultados/logs/erro_{fase}_*.txt)")
    if fila_p.exists():
        fila = json.loads(fila_p.read_text())["tarefas"]
        feitas_n = sum(tarefa_completa(t, feitas) for t in fila)
        out.append(f"tarefas: {feitas_n}/{len(fila)} ({100 * feitas_n / max(1, len(fila)):.1f}%)")
        sem, parc, n_var = faltas(fila, feitas)
        out.append(f"FALTA em relação à fila ({fila_p.name}): {len(sem)} tarefa(s) sem nenhuma linha, "
                   f"{len(parc)} parcial(is), {n_var} variante(s) no total")
        if sem:
            pb = defaultdict(int)
            pc = defaultdict(int)
            for _, t in sem:
                pb[t.get("backbone") or "(ingênuas)"] += 1
                pc[f"{t['dataset']}-{t['modo']}"] += 1
            out.append("  sem nenhuma linha, por backbone: " + ", ".join(f"{k} {v}" for k, v in sorted(pb.items())))
            out.append("  sem nenhuma linha, por conjunto: " + ", ".join(f"{k} {v}" for k, v in sorted(pc.items())))
        for nome, vs in list(parc.items())[:10]:
            out.append(f"  parcial {nome}: faltam {', '.join(vs)}")
        if len(parc) > 10:
            out.append(f"  ... e mais {len(parc) - 10} parcial(is)")
        dur_bb = defaultdict(list)
        for r in reg:
            if r.get("ok"):
                dur_bb[r["backbone"]].append(r["duracao_s"])
        geral = [x for v in dur_bb.values() for x in v]
        media_g = sum(geral) / len(geral) if geral else None
        resta = 0.0
        sem_estimativa = 0
        for t in fila:
            if tarefa_completa(t, feitas) or t.get("tipo") == "ingenua":
                continue
            v = dur_bb.get(t["backbone"])
            m = sum(v) / len(v) if v else media_g
            if m is None:
                sem_estimativa += 1
            else:
                resta += m
        n = max(1, len(vivos))
        if media_g is None:
            out.append("ETA: sem tarefas concluídas ainda")
        else:
            out.append(f"ETA: ~{resta / n / 3600:.2f} h com {n} worker(s) vivo(s) "
                       f"({sem_estimativa} tarefas sem média do backbone)")
    por_w = defaultdict(list)
    for r in reg:
        por_w[(r["worker"], r["device"])].append(r)
    out.append("por worker: id/dispositivo | concluídas | erros | duração média (s)")
    for (w, dev), rs in sorted(por_w.items()):
        ok = [r["duracao_s"] for r in rs if r.get("ok")]
        out.append(f"  w{w}/{dev}: {len(ok)} | {sum(1 for r in rs if not r.get('ok'))} | "
                   f"{(sum(ok) / len(ok) if ok else 0):.0f}")
    feitos_stem = {re.sub(r"[^A-Za-z0-9_.+-]", "_", r["tarefa"]) for r in reg}
    ativos = [(c["t0"], k, c) for k, c in claims.items() if vivo(c.get("pid")) and k not in feitos_stem]
    for t0, k, c in sorted(ativos):
        out.append(f"  em andamento (pid {c['pid']}): {k} há {(time.time() - t0) / 60:.1f} min")
    return "\n".join(out[: 200])


def main():
    args = sys.argv[1:]
    fila = None
    if "--fila" in args:
        i = args.index("--fila")
        fila = args[i + 1]
        del args[i:i + 2]
    fases = args or ["t1", "t2", "confirmacao"]
    achou = False
    for f in fases:
        r = relatorio(f, fila)
        if r:
            achou = True
            print(r)
    if not achou:
        print("nenhuma fase com fila ou CSV")


if __name__ == "__main__":
    main()
