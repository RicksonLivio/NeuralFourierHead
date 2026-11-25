#!/usr/bin/env python3
"""
test_falha_variante.py: confere que a falha de UMA variante não descarta as outras da tarefa.
Injeta (monkeypatch) uma conversão inexata (erro 1.0 em verificar_conversao, vale para as duas conv_*) e uma exceção em zero_L100_K10, roda uma
tarefa pequena em CPU num diretório temporário e confere: (a) as variantes sem falha estão no CSV; (b) as falhas
têm arquivo em logs/ com o erro_conversao; (c) o registro devolvido lista os erros; (d) relançar sem a injeção
completa só o que falhou (as outras são puladas) e o checkpoint da densa é reaproveitado.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["TESTE_MAX_EPOCAS"] = "1"
os.environ["TESTE_E_EXTRA"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import executar as X  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="test_falha_"))
X.RES = tmp
torch.set_num_threads(2)
dev = torch.device("cpu")
t = {"dataset": "ETTh1", "modo": "U", "backbone": "MLP", "largura": 1, "fracao": 0.05, "seed": 0,
     "variantes": ["densa", "densa+E", "proj_K10", "conv_L1000_K10", "conv_L0_K10", "zero_L100_K10", "zero_L1000_K10"]}

orig_vc, orig_train = X.verificar_conversao, X.train_model_blocos
X.verificar_conversao = lambda *a, **k: 1.0 if X._so_lambda_1000 else orig_vc(*a, **k)
X._so_lambda_1000 = True   # (nome histórico: liga a falha de conversão em todas as conv_*)


def treino_falho(model, tr, va, seed, lam=0.0, **k):
    if lam == 100.0:
        raise RuntimeError("falha injetada em zero_L100_K10")
    return orig_train(model, tr, va, seed, lam=lam, **k)


X.train_model_blocos = treino_falho
reg = X.executar_tarefa(t, "teste", dev)
feitas = {r[-1] for r in X.feitos("teste")}
print("variantes gravadas:", sorted(feitas))
print("erros:", [(e["variante"], e["tipo"], e.get("erro_conversao")) for e in reg["erros"]])
assert feitas == {"densa", "densa+E", "proj_K10", "zero_L1000_K10"}, feitas
assert {e["variante"] for e in reg["erros"]} == {"conv_L1000_K10", "conv_L0_K10", "zero_L100_K10"}
conv = [e for e in reg["erros"] if e["variante"] == "conv_L1000_K10"][0]
assert conv["erro_conversao"] == 1.0 and conv["tipo"] == "AssertionError"
assert (tmp / "modelos" / f"{X.id_arquivo(t)}_densa.pt").exists(), "checkpoint da densa deve ficar quando há erro"
for e in reg["erros"]:
    assert (tmp / "logs" / e["arquivo"]).exists()
    print("log:", e["arquivo"], "|", (tmp / "logs" / e["arquivo"]).read_text().splitlines()[2][:90])

# relança sem injeção: só as três variantes que faltavam rodam
X._so_lambda_1000 = False
X.train_model_blocos = orig_train
n_antes = len(X.feitos("teste"))
reg2 = X.executar_tarefa(t, "teste", dev)
n_depois = len(X.feitos("teste"))
print("relançada: linhas", n_antes, "->", n_depois, "| erros:", reg2["erros"], "| avisos:", reg2["avisos"])
assert n_depois == n_antes + 3 and not reg2["erros"] and not reg2["avisos"]
assert not (tmp / "modelos" / f"{X.id_arquivo(t)}_densa.pt").exists(), "checkpoint apagado após sucesso"

# aviso: densa já no CSV, sem checkpoint, variante nova
t2 = dict(t, variantes=t["variantes"] + ["conv_L100_K4"])
reg3 = X.executar_tarefa(t2, "teste", dev)
print("aviso de densa ausente:", reg3["avisos"])
assert len(reg3["avisos"]) == 1 and not reg3["erros"]
print("test_falha_variante: OK")
