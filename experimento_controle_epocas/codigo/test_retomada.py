#!/usr/bin/env python3
"""
test_retomada.py: testes do executor numa pasta temporária (CPU, 2 épocas; não toca nos resultados):
  1. uma tarefa completa grava 3 linhas, com cabeçalho uma só vez; rodar de novo não grava nada (retomada);
  2. apagar as linhas densa+E e nfh e rodar de novo regrava só essas duas (a densa do CSV fica), com aviso do
     checkpoint ausente; a densa retreinada na CPU dá a mesma linha de densa+E que a primeira vez (determinismo);
  3. falha forçada (TOL_CONVERSAO = 1e-12): a nfh não é gravada, vai para logs/erro_*.txt, densa e densa+E ficam,
     o checkpoint da densa permanece; relançada com a tolerância normal, a nfh é gravada e o checkpoint some;
  4. falha forçada de densa+E (TOL_COMPARA negativa): densa+E não é gravada; a nfh também falha (mesma tolerância);
  5. fases reais recusam TESTE_MAX_EPOCAS;
  6. dois processos gravando ao mesmo tempo produzem um só cabeçalho (flock).
Uso: CUDA_VISIBLE_DEVICES= python test_retomada.py
"""
import csv
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["TESTE_MAX_EPOCAS"] = "2"
os.environ["TESTE_E_EXTRA"] = "2"
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import torch  # noqa: E402

torch.set_num_threads(2)
import executar as ex  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="controle_teste_"))
ex.BASE = TMP
T = {"tipo": "treino", "conjunto": "AirQ-U", "backbone": "MLP", "seed": 0, "dispositivo": "CPU",
     "variantes": ["densa", "densa+E", "nfh"]}
DEV = torch.device("cpu")
FASE = "smoke_cpu"


def linhas():
    f = ex.csv_path(FASE, "treino")
    return list(csv.DictReader(open(f))) if f.exists() else []


def reescrever(manter):
    f = ex.csv_path(FASE, "treino")
    rs = [r for r in linhas() if r["variante"] in manter]
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ex.COLS)
        w.writeheader()
        w.writerows(rs)


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        ok.falhou = True


ok.falhou = False


def _escritora(i):
    for j in range(5):
        ex.gravar("smoke_cpu", "exata_s0", {"conjunto": "X", "backbone": "b", "seed": i, "variante": f"v{j}"})


if __name__ == "__main__":
    # 1
    reg = ex.executar_tarefa(T, FASE, DEV, "t", False)
    ok(len(linhas()) == 3 and not reg["erros"], f"1. primeira execução grava 3 linhas ({[r['variante'] for r in linhas()]})")
    txt = ex.csv_path(FASE, "treino").read_text()
    ok(txt.count("conjunto,backbone") == 1, "1. cabeçalho uma só vez")
    v0 = {r["variante"]: r for r in linhas()}
    reg = ex.executar_tarefa(T, FASE, DEV, "t", False)
    ok(reg.get("pulada") and len(linhas()) == 3, "1. segunda execução pula tudo (retomada)")
    ck = ex.pasta(FASE) / "modelos" / f"{ex.san(ex.id_tarefa(T))}_densa.pt"
    ok(not ck.exists(), "1. checkpoint da densa removido após sucesso")
    # 2
    reescrever({"densa"})
    reg = ex.executar_tarefa(T, FASE, DEV, "t", False)
    v1 = {r["variante"]: r for r in linhas()}
    ok(set(v1) == {"densa", "densa+E", "nfh"} and len(linhas()) == 3 and bool(reg["avisos"]),
       f"2. regrava só densa+E e nfh (avisos: {reg['avisos']})")
    ok(v1["densa+E"]["test_RMSE"] == v0["densa+E"]["test_RMSE"] and v1["nfh"]["test_RMSE"] == v0["nfh"]["test_RMSE"],
       "2. densa retreinada na CPU reproduz densa+E e nfh bit a bit")
    # 3
    reescrever({"densa"})
    ex.TOL_CONVERSAO = 1e-12
    reg = ex.executar_tarefa(T, FASE, DEV, "t", False)
    vs = [r["variante"] for r in linhas()]
    ok(sorted(vs) == ["densa", "densa+E"] and [e["variante"] for e in reg["erros"]] == ["nfh"],
       f"3. conversão forçada a falhar: linhas {vs}, erro em {[e['variante'] for e in reg['erros']]}")
    ok(any(ex.pasta(FASE).joinpath("logs").glob("erro_*nfh*.txt")), "3. erro gravado em logs/erro_*nfh*.txt")
    ok(ck.exists(), "3. checkpoint da densa mantido após falha")
    ex.TOL_CONVERSAO = 1e-5
    reg = ex.executar_tarefa(T, FASE, DEV, "t", False)
    ok(len(linhas()) == 3 and not reg["erros"] and not ck.exists(), "3. relançada, grava a nfh e remove o checkpoint")
    # 4
    reescrever({"densa"})
    ex.TOL_COMPARA = -1.0
    reg = ex.executar_tarefa(T, FASE, DEV, "t", False)
    vs = sorted(r["variante"] for r in linhas())
    ok(vs == ["densa"] and sorted(e["variante"] for e in reg["erros"]) == ["densa+E", "nfh"],
       f"4. asserts de validação forçados a falhar: linhas {vs}, erros {[e['variante'] for e in reg['erros']]}")
    ex.TOL_COMPARA = 1e-6
    # 5
    env = {k: v for k, v in os.environ.items()}
    env["TESTE_MAX_EPOCAS"] = "2"
    p = subprocess.run([sys.executable, str(AQUI / "executar.py"), "--fase", "controle", "--fila", "x.json",
                        "--device", "cpu"], env=env, capture_output=True, text=True)
    ok(p.returncode != 0 and "RECUSADO" in (p.stdout + p.stderr), "5. fase controle recusa TESTE_MAX_EPOCAS")
    p = subprocess.run([sys.executable, str(AQUI / "executar.py"), "--fase", "repro", "--fila", "x.json",
                        "--device", "cpu"], env={**env, "TESTE_E_EXTRA": "2"}, capture_output=True, text=True)
    ok(p.returncode != 0 and "RECUSADO" in (p.stdout + p.stderr), "5. fase repro recusa TESTE_E_EXTRA")
    # 6
    with mp.get_context("fork").Pool(4) as pool:
        pool.map(_escritora, range(4))
    t6 = ex.csv_path("smoke_cpu", "exata_s0").read_text()
    n = len(list(csv.DictReader(open(ex.csv_path("smoke_cpu", "exata_s0")))))
    ok(t6.count("conjunto,backbone") == 1 and n == 20, f"6. 4 processos x 5 linhas: cabeçalhos={t6.count('conjunto,backbone')}, linhas={n}")
    print("TMP:", TMP)
    sys.exit(1 if ok.falhou else 0)
