#!/usr/bin/env python3
"""
executar.py: executor retomável do experimento de backbones da NFH (protocolo: ../INSTRUCOES.md §3 e §6).

Uma TAREFA = (dataset, modo, backbone, largura, fracao, seed) + lista de variantes; ou uma tarefa de
ingênuas por (dataset, modo). Uma linha por variante concluída em resultados/<fase>.csv; o que já existe é
pulado; erro numa tarefa vai para resultados/logs/erro_<fase>_<tarefa>.txt e o worker segue.

Variantes (nomes exatos do §3):
  densa            treinada do zero, lambda = 0 (Adam 1e-3, wd 1e-5, lote 32, clip 1, até 60 épocas, paciência 10);
  densa+E          a densa mais E = 15 épocas (Adam 1e-4, sem penalização), época 0..15 de menor RMSE de validação;
  proj_K10         projeção da previsão da densa sobre T + S (sem treino; o resíduo é descartado);
  conv_L{l}_K{K}   conversão exata da densa (erro máximo < 1e-5, gravado) + ajuste estrutural de E = 15 épocas
                   (lr 1e-4, perda MSE + l*||R||^2/H) com seleção segura (RMSE de validação <= o da densa);
  zero_L{l}_K{K}   NFH treinada do zero com o mesmo protocolo da densa e perda MSE + l*||R||^2/H.
A fração de treino usa as ÚLTIMAS janelas do segmento de treino; validação e teste não mudam.

Asserts (a falha de uma variante vai para resultados/logs/erro_<fase>_<tarefa>__<variante>.txt com o erro_conversao,
as outras variantes da tarefa seguem; o que falhou é refeito no próximo lançamento):
  conversão: erro máximo < 1e-5; RMSE_val(densa+E) <= RMSE_val(densa); RMSE_val(conv) <= RMSE_val(densa).
Restrição da NFH: q + 1 + 2K <= H (checada ao ler a fila).

Variáveis de ambiente: TESTE_MAX_EPOCAS (padrão 60) e TESTE_E_EXTRA (padrão 15) só valem nas fases smoke*; nas fases
t1, t2 e confirmacao o executor recusa rodar se alguma estiver definida. O cabeçalho do log mostra E_EXTRA e MAX_EPOCAS efetivos.
A validação de cada época é feita em blocos de 1024 janelas (treino_bb.train_model_blocos).

Uso: python executar.py --fase t1 --fila resultados/filas/t1.json --worker 0 --n-workers 9 --device cuda
Por padrão as tarefas são reivindicadas dinamicamente (um arquivo de reivindicação por tarefa); com --estatico
o worker i faz as tarefas de índice i mod n.
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

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "teste_exemplificacao_nfh" / "codigo"))

from backbones import construir_modelo  # noqa: E402
from busca import cobertura, prever, treinar_densa_e, verificar_conversao  # noqa: E402
from dados_bb import carregar_bb, segmentos_bb  # noqa: E402
from treino_bb import train_model_blocos  # noqa: E402  (validação em blocos de 1024; ver treino_bb.py)
from v2_common import set_seed, stack_ds  # noqa: E402
from v3_common import CFG3, H3, Q3, Projetor, ajuste_estrutural, densa_para_nfh, ingenuas  # noqa: E402

RES = ROOT / "experimento_backbones_nfh" / "resultados"
E_EXTRA = int(os.environ.get("TESTE_E_EXTRA", 15))
FASES_REAIS = ("t1", "t2", "confirmacao")     # nessas fases TESTE_MAX_EPOCAS e TESTE_E_EXTRA são proibidas
LR_AJUSTE = 1e-4
TOL_CONVERSAO = 1e-5
TOL_COMPARA = 1e-7          # tolerância numérica (float) nas comparações de RMSE de validação

COLS = ["dataset", "modo", "backbone", "largura", "fracao", "seed", "variante", "K", "lambda",
        "val_RMSE", "val_MAE", "test_RMSE", "test_MAE", "cobertura_mediana", "epocas", "epoca_escolhida",
        "tempo_s", "params", "n_tr", "n_val", "n_te", "d", "erro_conversao"]

RE_CONV = re.compile(r"^conv_L(\d+(?:\.\d+)?)_K(\d+)$")
RE_ZERO = re.compile(r"^zero_L(\d+(?:\.\d+)?)_K(\d+)$")
RE_PROJ = re.compile(r"^proj_K(\d+)$")


def cfg_atual():
    return replace(CFG3, max_epochs=int(os.environ.get("TESTE_MAX_EPOCAS", CFG3.max_epochs)))


def fmt_frac(f):
    return f"{float(f):g}"


def id_tarefa(t):
    if t.get("tipo") == "ingenua":
        return f"ing|{t['dataset']}|{t['modo']}"
    return f"{t['dataset']}|{t['modo']}|{t['backbone']}|w{t['largura']}|f{fmt_frac(t['fracao'])}|s{t['seed']}"


def id_arquivo(t):
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", id_tarefa(t))


def validar_variante(v):
    """Checa o nome e a restrição q + 1 + 2K <= H da NFH."""
    if v in ("densa", "densa+E"):
        return
    for rx in (RE_CONV, RE_ZERO, RE_PROJ):
        m = rx.match(v)
        if m:
            K = int(m.groups()[-1])
            if not (Q3 + 1 + 2 * K <= H3):
                raise ValueError(f"{v}: q+1+2K = {Q3 + 1 + 2 * K} > H = {H3}")
            return
    raise ValueError(f"variante desconhecida: {v}")


# ---------------------------------------------------------------------------
# CSV (várias escritoras ao mesmo tempo: flock)
# ---------------------------------------------------------------------------
def csv_path(fase):
    return RES / f"{fase}.csv"


def chave_linha(r):
    return (r["dataset"], r["modo"], r["backbone"], str(r["largura"]), str(r["fracao"]), str(r["seed"]),
            r["variante"])


def feitos(fase):
    f = csv_path(fase)
    if not f.exists():
        return set()
    with open(f, newline="") as fh:
        return {chave_linha(r) for r in csv.DictReader(fh)}


def gravar(fase, linha):
    RES.mkdir(parents=True, exist_ok=True)
    f = csv_path(fase)
    with open(f, "a", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            w = csv.DictWriter(fh, fieldnames=COLS)
            fh.seek(0, os.SEEK_END)  # com 'a', tell() é a posição da abertura; outra escritora pode ter gravado o cabeçalho
            if fh.tell() == 0:
                w.writeheader()
            w.writerow({c: linha.get(c, "") for c in COLS})
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def log_jsonl(fase, obj):
    (RES / "logs").mkdir(parents=True, exist_ok=True)
    with open(RES / "logs" / f"{fase}_tarefas.jsonl", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
_cache = {}


def dados(dataset, modo):
    k = (dataset, modo)
    if k not in _cache:
        _cache.clear()
        ds = carregar_bb(f"{dataset}-{modo}")
        tr, va, te = segmentos_bb(ds)
        Xv, Yv = stack_ds(va)
        Xte, Yte = stack_ds(te)
        _cache[k] = (ds, tr, va, te, Xv, Yv, Xte, Yte)
    return _cache[k]


def treino_fracao(tr, fracao):
    """Últimas janelas do treino (as mais próximas da validação). Devolve uma cópia rasa do dataset."""
    n = len(tr)
    m = max(1, int(round(n * float(fracao)))) if float(fracao) < 1.0 else n
    t2 = copy.copy(tr)
    t2.idx = tr.idx[-m:]
    return t2


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def rmse_mae(P, Y):
    e = P - Y
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))


def avaliar_modelo(modelo, pacote, dev, K_cob=10):
    """val_RMSE, val_MAE, test_RMSE, test_MAE, cobertura mediana (teste) e as previsões de val e teste."""
    _, _, _, _, Xv, Yv, Xte, Yte = pacote
    Pv, Pt = prever(modelo, Xv, dev), prever(modelo, Xte, dev)
    vr, vm = rmse_mae(Pv, Yv.double().numpy())
    tr_, tm = rmse_mae(Pt, Yte.double().numpy())
    return {"val_RMSE": vr, "val_MAE": vm, "test_RMSE": tr_, "test_MAE": tm,
            "cobertura_mediana": cobertura(Pt, K=K_cob)}, Pv, Pt


# ---------------------------------------------------------------------------
# Tarefa
# ---------------------------------------------------------------------------
def executar_ingenua(t, fase, dev):
    ja = feitos(fase)
    ds, tr, va, te, Xv, Yv, Xte, Yte = dados(t["dataset"], t["modo"])
    nomes = {"Ingênuo (média de treino)": "ingenua_media", "Ingênuo (último valor)": "ingenua_ultimo",
             "Ingênuo sazonal (24 h)": "ingenua_sazonal24"}
    Pv, Pt = ingenuas(Xv, H3), ingenuas(Xte, H3)
    for nome, var in nomes.items():
        base = {"dataset": t["dataset"], "modo": t["modo"], "backbone": "", "largura": "", "fracao": "",
                "seed": "", "variante": var}
        if chave_linha(base) in ja:
            continue
        vr, vm = rmse_mae(Pv[nome], Yv.double().numpy())
        tr_, tm = rmse_mae(Pt[nome], Yte.double().numpy())
        gravar(fase, {**base, "val_RMSE": vr, "val_MAE": vm, "test_RMSE": tr_, "test_MAE": tm,
                      "epocas": 0, "tempo_s": 0, "params": 0, "n_val": len(va), "n_te": len(te),
                      "d": ds["feats"].shape[1]})
    return {"ingenua": True}


def _erro_variante(fase, t, variante, exc, extra=None):
    """Grava o traceback da variante em resultados/logs/ e devolve o registro de erro (vai para o jsonl)."""
    (RES / "logs").mkdir(parents=True, exist_ok=True)
    arq = RES / "logs" / f"erro_{fase}_{id_arquivo(t)}__{re.sub(r'[^A-Za-z0-9_.+-]', '_', variante)}.txt"
    arq.write_text(f"tarefa: {id_tarefa(t)}\nvariante: {variante}\n"
                   + (f"extra: {json.dumps(extra, ensure_ascii=False)}\n" if extra else "")
                   + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return {"variante": variante, "tipo": type(exc).__name__, "mensagem": str(exc)[:300], **(extra or {}),
            "arquivo": arq.name}


def executar_tarefa(t, fase, dev):
    """Executa as variantes pendentes da tarefa. Cada variante tem o seu try/except: uma falha (assert de
    conversão, seleção segura, falta de memória) grava o erro em resultados/logs/ e NÃO descarta as demais.
    O que falhou não vai para o CSV e é refeito no próximo lançamento (as outras são puladas)."""
    if t.get("tipo") == "ingenua":
        return executar_ingenua(t, fase, dev)
    cfg = cfg_atual()
    dataset, modo, bb = t["dataset"], t["modo"], t["backbone"]
    lw, frac, seed = int(t["largura"]), float(t["fracao"]), int(t["seed"])
    variantes = list(t["variantes"])
    for v in variantes:
        validar_variante(v)
    ja = feitos(fase)
    base_key = dict(dataset=dataset, modo=modo, backbone=bb, largura=str(lw), fracao=fmt_frac(frac), seed=str(seed))
    pend = [v for v in variantes if chave_linha({**base_key, "variante": v}) not in ja]
    if not pend:
        return {"pulada": True}
    pacote = dados(dataset, modo)
    ds, tr_full, va, te, Xv, Yv, Xte, Yte = pacote
    tr = treino_fracao(tr_full, frac)
    d = ds["feats"].shape[1]
    info_base = {**base_key, "d": d, "n_tr": len(tr), "n_val": len(va), "n_te": len(te)}
    reg = {"asserts": {}, "erros": [], "avisos": []}
    ckpt = RES / "modelos"
    ckpt.mkdir(parents=True, exist_ok=True)
    stem = id_arquivo(t)

    def linha(var, met, K="", lam="", extra=None):
        gravar(fase, {**info_base, "variante": var, "K": K, "lambda": lam, **met, **(extra or {})})

    def falha(v, exc, extra=None):
        reg["erros"].append(_erro_variante(fase, t, v, exc, extra))
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # --- densa (necessária para densa, densa+E, proj e conv) ---
    precisa_densa = any(v == "densa" or v == "densa+E" or RE_PROJ.match(v) or RE_CONV.match(v) for v in pend)
    densa = met_d = Pv_d = Pt_d = None
    if precisa_densa:
        try:
            arq = ckpt / f"{stem}_densa.pt"
            info_d = None
            if "densa" not in pend:
                # a densa já está no CSV: só serve se o checkpoint existir; senão a retreinada pode diferir da gravada
                if not arq.exists():
                    reg["avisos"].append("densa já estava no CSV mas o checkpoint não existe: a densa foi RETREINADA; "
                                         "densa+E/conv novas partem dela, que pode diferir (ruído de GPU) da densa do CSV")
                    print(f"AVISO {id_tarefa(t)}: checkpoint da densa ausente; retreinando (ver 'avisos' no jsonl)", flush=True)
            if "densa" not in pend and arq.exists():
                densa = construir_modelo(bb, "densa", d, lw)
                densa.load_state_dict(torch.load(arq, map_location="cpu"))
                densa.to(dev).eval()
            else:
                set_seed(seed)
                densa = construir_modelo(bb, "densa", d, lw)
                t0 = time.monotonic()
                densa, info_d = train_model_blocos(densa, tr, va, seed, lam=0.0, cfg=cfg, dev=dev)
                info_d["tempo"] = time.monotonic() - t0
                torch.save(densa.state_dict(), arq)
            met_d, Pv_d, Pt_d = avaliar_modelo(densa, pacote, dev)
            if "densa" in pend and info_d is not None:
                hist = [h[1] for h in info_d["history"]]
                linha("densa", met_d, extra={"epocas": info_d["epochs"], "epoca_escolhida": int(np.argmin(hist)) + 1,
                                             "tempo_s": info_d["tempo"], "params": n_params(densa)})
            reg["val_RMSE_densa"] = met_d["val_RMSE"]
        except Exception as exc:
            densa = None
            falha("densa", exc)
    dependentes = [v for v in pend if v == "densa+E" or RE_PROJ.match(v) or RE_CONV.match(v)]

    # --- densa+E ---
    if "densa+E" in pend and densa is not None:
        try:
            t0 = time.monotonic()
            de, info_e = treinar_densa_e(densa, tr, va, seed, E=E_EXTRA, cfg=cfg, dev=dev)
            met_e, _, _ = avaliar_modelo(de, pacote, dev)
            reg["asserts"]["densaE_val_menos_densa_val"] = met_e["val_RMSE"] - met_d["val_RMSE"]
            assert met_e["val_RMSE"] <= met_d["val_RMSE"] + TOL_COMPARA, \
                f"densa+E pior que a densa na validação: {met_e['val_RMSE']} > {met_d['val_RMSE']}"
            linha("densa+E", met_e, extra={"epocas": E_EXTRA, "epoca_escolhida": info_e["melhor_epoca"],
                                           "tempo_s": time.monotonic() - t0, "params": n_params(de)})
            if fase == "confirmacao":
                torch.save(de.state_dict(), ckpt / f"{stem}_densa+E.pt")
                np.save(ckpt / f"{stem}_densa+E_pred.npy", prever(de, Xte, dev).astype(np.float32))
        except Exception as exc:
            falha("densa+E", exc)

    # --- proj_K10 ---
    for v in [v for v in pend if RE_PROJ.match(v)]:
        if densa is None:
            continue
        try:
            K = int(RE_PROJ.match(v).group(1))
            proj = Projetor(H3, K, Q3)

            def so_TS(P):
                _, _, R = proj.decompor(P)
                return P - R
            Pv_p, Pt_p = so_TS(Pv_d), so_TS(Pt_d)
            vr, vm = rmse_mae(Pv_p, Yv.double().numpy())
            tr_, tm = rmse_mae(Pt_p, Yte.double().numpy())
            met = {"val_RMSE": vr, "val_MAE": vm, "test_RMSE": tr_, "test_MAE": tm,
                   "cobertura_mediana": cobertura(Pt_p, K=K)}
            linha(v, met, K=K, extra={"epocas": 0, "tempo_s": 0, "params": n_params(densa)})
            if fase == "confirmacao":
                np.save(ckpt / f"{stem}_{v}_pred.npy", Pt_p.astype(np.float32))
        except Exception as exc:
            falha(v, exc)

    # --- conv ---
    for v in [v for v in pend if RE_CONV.match(v)]:
        if densa is None:
            continue
        erro = None
        try:
            lam, K = float(RE_CONV.match(v).group(1)), int(RE_CONV.match(v).group(2))
            nfh = densa_para_nfh(densa, K=K, q=Q3, H=H3)
            nfh.eval()
            erro = verificar_conversao(densa, nfh, Xv, dev)
            reg["asserts"][f"erro_conversao_{v}"] = erro
            assert erro < TOL_CONVERSAO, f"conversão inexata: erro máximo {erro} >= {TOL_CONVERSAO}"
            t0 = time.monotonic()
            seg, _, info_a = ajuste_estrutural(nfh, tr, va, seed, lam=lam, lr=LR_AJUSTE, epocas=E_EXTRA, cfg=cfg, dev=dev)
            met_c, _, Pt_c = avaliar_modelo(seg, pacote, dev, K_cob=K)
            reg["asserts"][f"val_{v}_menos_densa"] = met_c["val_RMSE"] - met_d["val_RMSE"]
            assert met_c["val_RMSE"] <= met_d["val_RMSE"] + TOL_COMPARA, \
                f"{v}: RMSE de validação {met_c['val_RMSE']} > densa {met_d['val_RMSE']} (seleção segura violada)"
            linha(v, met_c, K=K, lam=lam, extra={"epocas": E_EXTRA, "epoca_escolhida": info_a["seguro"]["epoca"],
                                                 "tempo_s": time.monotonic() - t0, "params": n_params(seg),
                                                 "erro_conversao": erro})
            if fase == "confirmacao":
                torch.save(seg.state_dict(), ckpt / f"{stem}_{v}.pt")
                np.save(ckpt / f"{stem}_{v}_pred.npy", Pt_c.astype(np.float32))
        except Exception as exc:
            falha(v, exc, {"erro_conversao": erro, "dispositivo": str(dev)})

    # --- zero ---
    for v in [v for v in pend if RE_ZERO.match(v)]:
        try:
            lam, K = float(RE_ZERO.match(v).group(1)), int(RE_ZERO.match(v).group(2))
            set_seed(seed)
            nfh = construir_modelo(bb, "nfh", d, lw, K=K)
            t0 = time.monotonic()
            nfh, info_z = train_model_blocos(nfh, tr, va, seed, lam=lam, cfg=cfg, dev=dev)
            tempo = time.monotonic() - t0
            met_z, _, Pt_z = avaliar_modelo(nfh, pacote, dev, K_cob=K)
            esc = int(np.argmin([h[1] for h in info_z["history"]])) + 1 if lam == 0 else ""
            linha(v, met_z, K=K, lam=lam, extra={"epocas": info_z["epochs"], "epoca_escolhida": esc,
                                                 "tempo_s": tempo, "params": n_params(nfh)})
            if fase == "confirmacao":
                torch.save(nfh.state_dict(), ckpt / f"{stem}_{v}.pt")
                np.save(ckpt / f"{stem}_{v}_pred.npy", Pt_z.astype(np.float32))
        except Exception as exc:
            falha(v, exc)

    # o checkpoint da densa só é apagado se nada falhou (uma nova tentativa precisa da MESMA densa)
    if fase != "confirmacao" and precisa_densa and not reg["erros"]:
        try:
            (ckpt / f"{stem}_densa.pt").unlink()
        except FileNotFoundError:
            pass
    return reg


# ---------------------------------------------------------------------------
# Fila e CLI
# ---------------------------------------------------------------------------
def reivindicar(fase, t):
    d = RES / "claims" / fase
    d.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(d / f"{id_arquivo(t)}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "t0": time.time()}).encode())
    os.close(fd)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fase", required=True, choices=["t1", "t2", "confirmacao", "smoke", "smoke_cpu", "smoke2", "smoke2_cpu"])
    ap.add_argument("--fila", required=True, help="JSON com {'tarefas': [...]}")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--estatico", action="store_true", help="partição fixa i mod n em vez de reivindicação dinâmica")
    a = ap.parse_args()

    if a.fase in FASES_REAIS:
        vars_teste = [v for v in ("TESTE_MAX_EPOCAS", "TESTE_E_EXTRA") if v in os.environ]
        if vars_teste:
            sys.exit(f"RECUSADO: {', '.join(vars_teste)} definida(s) no ambiente; essas variáveis só valem nas fases "
                     f"smoke*. Rode `unset {' '.join(vars_teste)}` e lance de novo (fase {a.fase}).")

    dev = torch.device(a.device)
    torch.set_num_threads(2)             # 2 fios também nos workers GPU (o laço de treino é limitado pela CPU)
    fila = json.loads(Path(a.fila).read_text())["tarefas"]
    for t in fila:                       # valida tudo antes de começar
        for v in t.get("variantes", []):
            validar_variante(v)
    (RES / "logs").mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] fase={a.fase} worker={a.worker}/{a.n_workers} dev={dev} "
          f"tarefas={len(fila)} E_EXTRA={E_EXTRA} MAX_EPOCAS={cfg_atual().max_epochs} "
          f"(TESTE_E_EXTRA={os.environ.get('TESTE_E_EXTRA', '-')}, TESTE_MAX_EPOCAS={os.environ.get('TESTE_MAX_EPOCAS', '-')})",
          flush=True)
    for i, t in enumerate(fila):
        tid = id_tarefa(t)
        if a.estatico and i % a.n_workers != a.worker:
            continue
        if t.get("tipo") != "ingenua":
            ja = feitos(a.fase)
            if all(chave_linha({"dataset": t["dataset"], "modo": t["modo"], "backbone": t["backbone"],
                                "largura": str(t["largura"]), "fracao": fmt_frac(t["fracao"]),
                                "seed": str(t["seed"]), "variante": v}) in ja for v in t["variantes"]):
                continue
        if not a.estatico and not reivindicar(a.fase, t):
            continue
        t0 = time.monotonic()
        try:
            reg = executar_tarefa(t, a.fase, dev)
            dur = time.monotonic() - t0
            erros = reg.get("erros", []) if isinstance(reg, dict) else []
            log_jsonl(a.fase, {"tarefa": tid, "worker": a.worker, "device": a.device, "duracao_s": dur,
                               "backbone": t.get("backbone", ""), "ok": not erros, "registro": reg})
            if erros:
                print(f"[{time.strftime('%H:%M:%S')}] ERRO em {len(erros)} variante(s) de {tid} "
                      f"({', '.join(e['variante'] for e in erros)}); as demais foram gravadas; ver resultados/logs/erro_{a.fase}_*",
                      flush=True)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] ok {tid} {dur:.0f}s", flush=True)
        except Exception:
            dur = time.monotonic() - t0
            (RES / "logs" / f"erro_{a.fase}_{id_arquivo(t)}.txt").write_text(traceback.format_exc())
            log_jsonl(a.fase, {"tarefa": tid, "worker": a.worker, "device": a.device, "duracao_s": dur,
                               "backbone": t.get("backbone", ""), "ok": False})
            print(f"[{time.strftime('%H:%M:%S')}] ERRO {tid}: ver resultados/logs/erro_{a.fase}_{id_arquivo(t)}.txt",
                  flush=True)
            if a.device == "cuda":
                torch.cuda.empty_cache()
    print(f"[{time.strftime('%H:%M:%S')}] worker {a.worker} terminou", flush=True)


if __name__ == "__main__":
    main()
