#!/usr/bin/env python3
"""
medir_tempo.py: mede o tempo por época de cada backbone (largura 1 e 2, GPU e CPU) e grava
resultados/tempo_epoca.json.

Método: para cada (backbone, largura, dispositivo, conjunto) treina com `train_model_blocos` (a mesma rotina dos
experimentos, com avaliação de validação a cada época) por 1 época e por 3 épocas, na fração 0,25 do treino
(últimas janelas); tempo marginal por época = (t3 - t1)/2, sem a inicialização. Normaliza por janela de treino
(s por janela por época) para escalar a qualquer conjunto: a validação, que é fixa, fica embutida (estimativa
levemente conservadora para treinos grandes).
Conjuntos: SINT-U48-U (d = 1) e SINT-M-M (d = 5).

Uso: python medir_tempo.py [--dispositivos cuda cpu] [--larguras 1 2] [--saida ...] [--rotulo sozinho|carga]
  --rotulo carga: usado quando outros processos rodam ao mesmo tempo; grava em tempo_epoca_<rotulo>_<id>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
sys.path.insert(0, str(AQUI))

import executar as X  # noqa: E402
from backbones import REGISTRO, construir_modelo  # noqa: E402
from treino_bb import train_model_blocos as train_model  # noqa: E402  (a rotina real dos experimentos)
from v3_common import CFG3  # noqa: E402


def medir(bb, lw, dev, pacote, tr, cfg1, cfg3, d):
    ts = []
    for cfg in (cfg1, cfg3):
        m = construir_modelo(bb, "densa", d, lw)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.monotonic()
        train_model(m, tr, pacote[2], 0, lam=0.0, cfg=cfg, dev=dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.monotonic() - t0)
    return (ts[1] - ts[0]) / 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dispositivos", nargs="+", default=["cuda", "cpu"])
    ap.add_argument("--larguras", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--backbones", nargs="+", default=list(REGISTRO))
    ap.add_argument("--rotulo", default="sozinho")
    ap.add_argument("--saida", default=None)
    ap.add_argument("--acrescentar", action="store_true",
                    help="junta as medidas dos --backbones dados ao tempo_epoca.json existente (substitui os do mesmo backbone)")
    a = ap.parse_args()
    cfg1 = replace(CFG3, max_epochs=1, patience=99)
    cfg3 = replace(CFG3, max_epochs=3, patience=99)
    res = {"metodo": __doc__.split("Método:")[1].split("Uso:")[0].strip(), "rotulo": a.rotulo,
           "medidas": []}
    for dispositivo in a.dispositivos:
        dev = torch.device(dispositivo)
        torch.set_num_threads(2)
        w = X.dados("SINT-U48", "U")          # aquecimento (CUDA, cuDNN, alocador), descartado
        medir("MLP", 1, dev, w, X.treino_fracao(w[1], 0.1), cfg1, cfg1, 1)
        for ds, modo in (("SINT-U48", "U"), ("SINT-M", "M")):
            pacote = X.dados(ds, modo)
            d = pacote[0]["feats"].shape[1]
            tr = X.treino_fracao(pacote[1], 0.25)
            for lw in a.larguras:
                for bb in a.backbones:
                    t = medir(bb, lw, dev, pacote, tr, cfg1, cfg3, d)
                    res["medidas"].append({"dispositivo": dispositivo, "dataset": ds, "d": d, "backbone": bb,
                                           "largura": lw, "n_tr": len(tr), "n_val": len(pacote[2]),
                                           "s_por_epoca": t, "s_por_janela_epoca": t / len(tr)})
                    print(f"{dispositivo} {ds} d={d} {bb} w{lw}: {t:.2f} s/época ({len(tr)} janelas)", flush=True)
    saida = Path(a.saida) if a.saida else X.RES / (
        "tempo_epoca.json" if a.rotulo == "sozinho" else f"tempo_epoca_{a.rotulo}_{os.getpid()}.json")
    saida.parent.mkdir(parents=True, exist_ok=True)
    if a.acrescentar and saida.exists():
        antigo = json.loads(saida.read_text())
        novos = {m["backbone"] for m in res["medidas"]}
        res["medidas"] = [m for m in antigo["medidas"] if m["backbone"] not in novos] + res["medidas"]
    # resumo usado por filas.py (GPU, largura 1, d = 1)
    res["gpu_s_por_janela_epoca_w1"] = {m["backbone"]: m["s_por_janela_epoca"] for m in res["medidas"]
                                        if m["dispositivo"] == "cuda" and m["largura"] == 1 and m["d"] == 1}
    saida.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("gravado", saida)


if __name__ == "__main__":
    main()
