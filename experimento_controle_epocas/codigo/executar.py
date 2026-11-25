#!/usr/bin/env python3
"""
executar.py: executor retomável do controle das épocas extras (protocolo: ../INSTRUCOES.md).

Uma TAREFA = (conjunto, backbone, semente) com as variantes
  densa    retreinada do zero como em src/v3_benchmark.rodar (set_seed, construir, train_model com CFG3);
  densa+E  a partir dessa densa, E = 15 épocas com o otimizador, a taxa (1e-4), o weight decay, o lote, o clip e o
           gerador de embaralhamento de v3_common.ajuste_estrutural, perda MSE SEM penalização; escolhe a época 0..15
           de menor RMSE de validação (a época 0 é a própria densa; empate fica com a época mais cedo);
  nfh      densa_para_nfh(densa, K=10) + ajuste_estrutural com os padrões da v3 ("seguro"), como na dissertação.
ou uma tarefa exata_s0 (só semente 0): carrega results/v3/modelos/<conj>__<bb>+densa__s0.pt (a densa da
dissertação), avalia ("densa_diss") e roda o mesmo densa+E. Nada é retreinado.

Cada tarefa roda no dispositivo que a dissertação usou (results/v3/dispositivos.json). Um worker --device cuda só
pega tarefas GPU e um --device cpu só pega tarefas CPU, a menos de --qualquer.

Saídas (por fase; a fase real é "controle"): resultados/controle.csv, resultados/exata_s0.csv (uma linha por
variante concluída, flock, retomável), resultados/logs/ (jsonl por tarefa e erros), resultados/claims/<fase>/.
As fases "smoke*" gravam em resultados/smoke/<fase>/ e podem usar TESTE_MAX_EPOCAS e TESTE_E_EXTRA; as fases
"controle" e "repro" (reprodução com épocas reais num par pequeno) RECUSAM rodar se alguma estiver definida.

Asserts por variante (falha vai para logs/erro_*.txt, a linha não é gravada, as outras variantes seguem, e o que
falhou é refeito no próximo lançamento): erro de conversão < 1e-5; val_RMSE(densa+E) <= val_RMSE(densa);
val_RMSE(nfh) <= val_RMSE(densa). A tolerância das comparações é 1e-6 (a conversão exata muda o RMSE de validação
na 8.ª casa: ver rmse_val_densa x val_RMSE nas linhas da dissertação).

Validação em blocos de 1024 janelas (treino_blocos.py): --val-blocos auto = sim na GPU, não na CPU.

Uso: python executar.py --fase controle --fila resultados/filas/controle.json --worker 0 --device cuda
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import os
import re
import socket
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(ROOT / "src"))

from v2_common import set_seed, stack_ds, train_model  # noqa: E402
from v3_benchmark import metricas, prever  # noqa: E402  (importados, nunca editados)
from v3_common import (CFG3, H3, K3, LR_AJUSTE, Q3, RES3, ajuste_estrutural, avaliar, carregar,  # noqa: E402
                       construir, densa_para_nfh, segmentos)
from colunas import COLS  # noqa: E402
from treino_blocos import train_model_blocos  # noqa: E402

BASE = ROOT / "experimento_controle_epocas" / "resultados"
FASES_REAIS = ("controle", "repro")
K_NFH = 10
E_EXTRA = int(os.environ.get("TESTE_E_EXTRA", 15))
TOL_CONVERSAO = 1e-5
TOL_COMPARA = 1e-6
VARIANTES = ("densa", "densa+E", "nfh")




# ---------------------------------------------------------------------------
# Pastas e CSV (várias escritoras: flock)
# ---------------------------------------------------------------------------
def pasta(fase):
    return BASE if fase == "controle" else (BASE / "repro" if fase == "repro" else BASE / "smoke" / fase)


def csv_path(fase, tipo):
    return pasta(fase) / ("controle.csv" if tipo == "treino" else "exata_s0.csv")


def chave_linha(r):
    return (r["conjunto"], r["backbone"], str(r["seed"]), r["variante"])


def feitos(fase, tipo):
    f = csv_path(fase, tipo)
    if not f.exists():
        return set()
    with open(f, newline="") as fh:
        return {chave_linha(r) for r in csv.DictReader(fh)}


def gravar(fase, tipo, linha):
    f = csv_path(fase, tipo)
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            w = csv.DictWriter(fh, fieldnames=COLS)
            fh.seek(0, os.SEEK_END)      # com 'a', tell() é a posição da abertura; outra escritora pode ter gravado o cabeçalho
            if fh.tell() == 0:
                w.writeheader()
            w.writerow({c: linha.get(c, "") for c in COLS})
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def log_jsonl(fase, obj):
    d = pasta(fase) / "logs"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "tarefas.jsonl", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()


def san(s):
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", s)


def id_tarefa(t):
    if t["tipo"] == "exata_s0":
        return f"exata_s0|{t['conjunto']}|{t['backbone']}|s0"
    return f"{t['conjunto']}|{t['backbone']}|s{t['seed']}"


# ---------------------------------------------------------------------------
# Referência: as linhas da dissertação (results/v3/runs)
# ---------------------------------------------------------------------------
_diss = {}


def linhas_dissertacao(conj):
    if conj not in _diss:
        with open(RES3 / "runs" / f"{conj}.csv", newline="") as fh:
            rs = list(csv.DictReader(fh))
        # como src/v3_tabelas.carregar_runs: linhas duplicadas por retomada -> fica a ÚLTIMA (a que gravou os modelos .pt)
        ult = {(r["modelo"], r["variante"], r["K"], r["seed"]): r for r in rs}
        _diss[conj] = list(ult.values())
    return _diss[conj]


def ref_dissertacao(conj, bb, seed, variante):
    """Linha da dissertação: 'densa' = '<bb> + densa' sem variante; 'nfh' = '<bb> + NFH', seguro, K = 10."""
    for r in linhas_dissertacao(conj):
        if int(r["seed"]) != int(seed):
            continue
        if variante in ("densa", "densa_diss") and r["modelo"] == f"{bb} + densa" and r["variante"] == "":
            return r
        if variante == "nfh" and r["modelo"] == f"{bb} + NFH" and r["variante"] == "seguro" and r["K"] == str(K_NFH):
            return r
    return None


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
_cache = {}


def dados(conj):
    if conj not in _cache:
        _cache.clear()
        ds = carregar(conj)
        tr, va, te = segmentos(ds)
        Xv, Yv = stack_ds(va)
        Xte, Yte = stack_ds(te)
        _cache[conj] = (ds, tr, va, te, Xv, Yv, Xte, Yte, Yte.double().numpy())
    return _cache[conj]


def cfg_atual():
    return replace(CFG3, max_epochs=int(os.environ.get("TESTE_MAX_EPOCAS", CFG3.max_epochs)))


def n_params(m):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


def medir(modelo, pacote, dev):
    """val_RMSE como na dissertação (v3_common.avaliar), val_MAE, test_RMSE e test_MAE (v3_benchmark.prever/metricas)."""
    _, _, _, _, Xv, Yv, Xte, _, Y = pacote
    rv, _, Pv, _ = avaliar(modelo, Xv, Yv, dev)
    vm = float(np.mean(np.abs(Pv - Yv.double().numpy())))
    rt, mt = metricas(prever(modelo, Xte, dev), Y)
    return {"val_RMSE": rv, "val_MAE": vm, "test_RMSE": rt, "test_MAE": mt}


# ---------------------------------------------------------------------------
# densa+E: o laço de ajuste_estrutural, sem penalização
# ---------------------------------------------------------------------------
def densa_mais_E(densa, tr, va, Xv, Yv, seed, dev, E=None, cfg=CFG3):
    """E épocas a partir de `densa` (cópia; a original não muda). Mesmos otimizador, taxa (LR_AJUSTE), weight decay,
    lote, clip, gerador de embaralhamento (Generator com a semente) e set_seed(seed) de ajuste_estrutural; perda
    MSE pura. Escolhe a época 0..E de menor RMSE de validação (estritamente menor: empate fica com a mais cedo)."""
    E = E_EXTRA if E is None else E
    modelo = copy.deepcopy(densa).to(dev)
    set_seed(seed)
    r0 = avaliar(modelo, Xv, Yv, dev, H=cfg.horizon)[0]
    melhor = {"epoca": 0, "rmse": r0, "estado": copy.deepcopy(modelo.state_dict())}
    traj = [(0, r0)]
    g = torch.Generator()
    g.manual_seed(seed)
    tl = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, generator=g)
    opt = torch.optim.Adam(modelo.parameters(), lr=LR_AJUSTE, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()
    t0 = time.time()
    for ep in range(1, E + 1):
        modelo.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            perda = mse(modelo(x)["forecast"], y)
            perda.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), cfg.grad_clip)
            opt.step()
        r = avaliar(modelo, Xv, Yv, dev, H=cfg.horizon)[0]
        traj.append((ep, r))
        if r < melhor["rmse"]:
            melhor = {"epoca": ep, "rmse": r, "estado": copy.deepcopy(modelo.state_dict())}
    modelo.load_state_dict(melhor.pop("estado"))
    modelo.eval()
    return modelo, {"epoca": melhor["epoca"], "rmse": melhor["rmse"], "trajetoria": traj, "tempo_s": time.time() - t0}


# ---------------------------------------------------------------------------
# Tarefas
# ---------------------------------------------------------------------------
def _erro(fase, t, variante, exc, extra=None):
    d = pasta(fase) / "logs"
    d.mkdir(parents=True, exist_ok=True)
    arq = d / f"erro_{san(id_tarefa(t))}__{san(variante)}.txt"
    arq.write_text(f"tarefa: {id_tarefa(t)}\nvariante: {variante}\n"
                   + (f"extra: {json.dumps(extra, ensure_ascii=False)}\n" if extra else "")
                   + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return {"variante": variante, "tipo": type(exc).__name__, "mensagem": str(exc)[:300], **(extra or {}),
            "arquivo": arq.name}


def montar_linha(t, variante, met, ctx, ref=None, extra=None):
    conj, bb = t["conjunto"], t["backbone"]
    seed = int(t.get("seed", 0))
    L = {"conjunto": conj, "backbone": bb, "seed": seed, "variante": variante, "dispositivo": ctx["dev_nome"],
         "dispositivo_dissertacao": t["dispositivo"], **met, **ctx["info"], **(extra or {})}
    if ref is not None:
        L.update({"diss_val_RMSE": ref["val_RMSE"], "diss_test_RMSE": ref["RMSE"], "diss_test_MAE": ref["MAE"],
                  "dif_val_RMSE": met["val_RMSE"] - float(ref["val_RMSE"]),
                  "dif_test_RMSE": met["test_RMSE"] - float(ref["RMSE"]),
                  "dif_test_MAE": met["test_MAE"] - float(ref["MAE"])})
    return L


def executar_tarefa(t, fase, dev, worker, blocos):
    tipo = t["tipo"]
    cfg = cfg_atual()
    conj, bb = t["conjunto"], t["backbone"]
    seed = int(t.get("seed", 0))
    variantes = list(t["variantes"])
    ja = feitos(fase, tipo)
    pend = [v for v in variantes if (conj, bb, str(seed), v) not in ja]
    if not pend:
        return {"pulada": True}
    pacote = dados(conj)
    ds, tr, va, te, Xv, Yv, Xte, _, Y = pacote
    d = ds["feats"].shape[1]
    ctx = {"dev_nome": "GPU" if dev.type == "cuda" else "CPU",
           "info": {"n_tr": len(tr), "n_val": len(va), "n_te": len(te), "d": d, "val_blocos": int(blocos),
                    "max_epocas": cfg.max_epochs, "e_extra": E_EXTRA, "worker": worker}}
    reg = {"asserts": {}, "erros": [], "avisos": []}
    ck = pasta(fase) / "modelos"
    ck.mkdir(parents=True, exist_ok=True)
    stem = san(id_tarefa(t))

    def falha(v, exc, extra=None):
        reg["erros"].append(_erro(fase, t, v, exc, extra))
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    def gr(linha):
        gravar(fase, tipo, linha)

    # --- a densa de partida ---
    densa = met_d = None
    try:
        if tipo == "exata_s0":
            arq = RES3 / "modelos" / f"{conj}__{bb}+densa__s0.pt"
            densa = construir(bb, "densa", d)
            densa.load_state_dict(torch.load(arq, map_location="cpu"))
            densa.to(dev).eval()
            met_d = medir(densa, pacote, dev)
            if "densa_diss" in pend:
                gr(montar_linha(t, "densa_diss", met_d, ctx, ref_dissertacao(conj, bb, seed, "densa_diss"),
                                {"epocas": "", "epoca_escolhida": "", "tempo_s": 0, "params": n_params(densa)}))
        else:
            arq = ck / f"{stem}_densa.pt"
            if "densa" not in pend and arq.exists():
                densa = construir(bb, "densa", d)
                densa.load_state_dict(torch.load(arq, map_location="cpu"))
                densa.to(dev).eval()
                info_d = None
            else:
                if "densa" not in pend:
                    reg["avisos"].append("densa já estava no CSV mas o checkpoint não existe: foi RETREINADA")
                    print(f"AVISO {id_tarefa(t)}: checkpoint da densa ausente; retreinando", flush=True)
                set_seed(seed)
                densa = construir(bb, "densa", d)
                t0 = time.time()
                treinar = train_model_blocos if blocos else train_model
                densa, info_d = treinar(densa, tr, va, seed, lam=0.0, cfg=cfg, dev=dev)
                info_d["tempo"] = time.time() - t0
                torch.save(densa.state_dict(), arq)
            met_d = medir(densa, pacote, dev)
            if "densa" in pend and info_d is not None:
                gr(montar_linha(t, "densa", met_d, ctx, ref_dissertacao(conj, bb, seed, "densa"),
                                {"epocas": info_d["epochs"], "epoca_escolhida": "", "tempo_s": info_d["tempo"],
                                 "params": n_params(densa)}))
        reg["val_RMSE_densa"] = met_d["val_RMSE"]
    except Exception as exc:
        densa = None
        falha("densa_diss" if tipo == "exata_s0" else "densa", exc)

    # --- densa+E ---
    if "densa+E" in pend and densa is not None:
        try:
            de, ie = densa_mais_E(densa, tr, va, Xv, Yv, seed, dev, E=E_EXTRA, cfg=cfg)
            met_e = medir(de, pacote, dev)
            reg["asserts"]["val_densaE_menos_densa"] = met_e["val_RMSE"] - met_d["val_RMSE"]
            assert met_e["val_RMSE"] <= met_d["val_RMSE"] + TOL_COMPARA, \
                f"densa+E pior que a densa na validação: {met_e['val_RMSE']} > {met_d['val_RMSE']}"
            gr(montar_linha(t, "densa+E", met_e, ctx, None,
                            {"epocas": E_EXTRA, "epoca_escolhida": ie["epoca"], "tempo_s": ie["tempo_s"],
                             "params": n_params(de)}))
        except Exception as exc:
            falha("densa+E", exc)

    # --- nfh (K = 10) ---
    if "nfh" in pend and densa is not None:
        erro = None
        try:
            nfh = densa_para_nfh(densa, K=K_NFH, q=Q3, H=H3)
            nfh.eval()
            with torch.no_grad():
                b = Xv[:512].to(dev)
                erro = float((nfh(b)["forecast"] - densa(b)["forecast"]).abs().max())
            reg["asserts"]["erro_conversao"] = erro
            assert erro < TOL_CONVERSAO, f"conversão inexata: erro máximo {erro} >= {TOL_CONVERSAO}"
            nfh, _, aj = ajuste_estrutural(nfh, tr, va, seed, epocas=E_EXTRA, cfg=cfg, dev=dev)
            met_n = medir(nfh, pacote, dev)
            reg["asserts"]["val_nfh_menos_densa"] = met_n["val_RMSE"] - met_d["val_RMSE"]
            assert met_n["val_RMSE"] <= met_d["val_RMSE"] + TOL_COMPARA, \
                f"nfh: val_RMSE {met_n['val_RMSE']} > densa {met_d['val_RMSE']} (seleção segura violada)"
            gr(montar_linha(t, "nfh", met_n, ctx, ref_dissertacao(conj, bb, seed, "nfh"),
                            {"epocas": E_EXTRA, "epoca_escolhida": aj["seguro"]["epoca"],
                             "tempo_s": aj["tempo_ajuste_s"], "erro_conversao": erro, "K": K_NFH,
                             "params": n_params(nfh)}))
        except Exception as exc:
            falha("nfh", exc, {"erro_conversao": erro, "dispositivo": ctx["dev_nome"]})

    # o checkpoint da densa só some se nada falhou (uma nova tentativa precisa da MESMA densa)
    if tipo == "treino" and not reg["erros"]:
        try:
            (ck / f"{stem}_densa.pt").unlink()
        except FileNotFoundError:
            pass
    return reg


# ---------------------------------------------------------------------------
# Reivindicação dinâmica (um arquivo por tarefa; flock evita corrida ao retirar reivindicação de worker morto)
# ---------------------------------------------------------------------------
def vivo(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True


def reivindicar(fase, t):
    d = pasta(fase) / "claims"
    d.mkdir(parents=True, exist_ok=True)
    alvo = d / f"{san(id_tarefa(t))}.claim"
    with open(d / ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            if alvo.exists():
                try:
                    c = json.loads(alvo.read_text())
                except Exception:
                    c = {}
                if c.get("host") == socket.gethostname() and not vivo(c.get("pid")):
                    alvo.unlink()                    # sobra de worker morto: a tarefa volta para a fila
                else:
                    return False
            alvo.write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "t0": time.time()}))
            return True
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def tarefa_completa(t, fase):
    ja = feitos(fase, t["tipo"])
    s = str(t.get("seed", 0))
    return all((t["conjunto"], t["backbone"], s, v) in ja for v in t["variantes"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fase", required=True, choices=["controle", "repro", "smoke", "smoke_cpu"])
    ap.add_argument("--fila", required=True, help="JSON com {'tarefas': [...]} (filas.py)")
    ap.add_argument("--worker", default="0")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--qualquer", action="store_true", help="pega tarefas de qualquer dispositivo (muda o dispositivo!)")
    ap.add_argument("--val-blocos", choices=["auto", "sim", "nao"], default="auto")
    a = ap.parse_args()

    if a.fase in FASES_REAIS:
        vars_teste = [v for v in ("TESTE_MAX_EPOCAS", "TESTE_E_EXTRA") if v in os.environ]
        if vars_teste:
            sys.exit(f"RECUSADO: {', '.join(vars_teste)} definida(s); só valem nas fases smoke*. "
                     f"Rode `unset {' '.join(vars_teste)}` e lance de novo (fase {a.fase}).")
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("RECUSADO: --device cuda sem GPU disponível")
    dev = torch.device(a.device)
    torch.set_num_threads(2)                      # 2 fios, como a dissertação (V3_THREADS = 2); também nos workers GPU
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True     # como v3_benchmark.main
    blocos = {"auto": dev.type == "cuda", "sim": True, "nao": False}[a.val_blocos]
    fila = json.loads(Path(a.fila).read_text())["tarefas"]
    pasta(a.fase).mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] fase={a.fase} worker={a.worker} dev={dev} tarefas={len(fila)} "
          f"E_EXTRA={E_EXTRA} MAX_EPOCAS={cfg_atual().max_epochs} val_blocos={int(blocos)} qualquer={a.qualquer} "
          f"threads={torch.get_num_threads()} (TESTE_E_EXTRA={os.environ.get('TESTE_E_EXTRA', '-')}, "
          f"TESTE_MAX_EPOCAS={os.environ.get('TESTE_MAX_EPOCAS', '-')})", flush=True)
    alvo_dev = "GPU" if dev.type == "cuda" else "CPU"
    for t in fila:
        if not a.qualquer and t["dispositivo"] != alvo_dev:
            continue
        if tarefa_completa(t, a.fase):
            continue
        if not reivindicar(a.fase, t):
            continue
        tid = id_tarefa(t)
        t0 = time.monotonic()
        try:
            reg = executar_tarefa(t, a.fase, dev, a.worker, blocos)
            dur = time.monotonic() - t0
            erros = reg.get("erros", [])
            log_jsonl(a.fase, {"tarefa": tid, "worker": a.worker, "device": a.device, "duracao_s": dur,
                               "backbone": t["backbone"], "ok": not erros, "registro": reg})
            print(f"[{time.strftime('%H:%M:%S')}] " + (f"ERRO em {len(erros)} variante(s) de {tid} "
                  f"({', '.join(e['variante'] for e in erros)}); ver logs/erro_*" if erros else f"ok {tid} {dur:.0f}s"),
                  flush=True)
        except Exception:
            dur = time.monotonic() - t0
            (pasta(a.fase) / "logs").mkdir(parents=True, exist_ok=True)
            (pasta(a.fase) / "logs" / f"erro_{san(tid)}.txt").write_text(traceback.format_exc())
            log_jsonl(a.fase, {"tarefa": tid, "worker": a.worker, "device": a.device, "duracao_s": dur,
                               "backbone": t["backbone"], "ok": False})
            print(f"[{time.strftime('%H:%M:%S')}] ERRO {tid}: ver logs/erro_{san(tid)}.txt", flush=True)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    print(f"[{time.strftime('%H:%M:%S')}] worker {a.worker} terminou", flush=True)


if __name__ == "__main__":
    main()
