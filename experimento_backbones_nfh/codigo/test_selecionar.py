#!/usr/bin/env python3
"""
test_selecionar.py: o selecionador NÃO pode ler colunas test_*. Falha se ler.

Testes:
  1. `ler_val` devolve só colunas de identificação e val_*.
  2. O resultado é idêntico se as colunas test_* forem trocadas por lixo (texto, NaN, valores extremos).
  3. O resultado é idêntico se as colunas test_* forem removidas do CSV (um selecionador que as lesse levaria KeyError).
  4. Espião em pandas.read_csv: toda chamada tem de passar `usecols` que exclui test_*.
  5. Sanidade das regras: uma combinação em que a NFH ganha na validação passa; uma que perde no MAE não passa.
Uso: python test_selecionar.py   (ou pytest)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selecionar as S  # noqa: E402

COLS = ["dataset", "modo", "backbone", "largura", "fracao", "seed", "variante", "K", "lambda", "val_RMSE", "val_MAE",
        "test_RMSE", "test_MAE", "cobertura_mediana", "epocas", "epoca_escolhida", "tempo_s", "params", "n_tr", "n_val",
        "n_te", "d", "erro_conversao"]


def csv_falso(caminho, com_teste=True, lixo=None):
    rng = np.random.default_rng(0)
    linhas = []
    casos = [("SINT-U48", "U", "MLP", 0.1, 0.90, 0.95), ("SINT-U48", "U", "TCN", 0.1, 1.05, 0.95),   # TCN: perde no RMSE
             ("ETTh1", "U", "MLP", 1.0, 0.99, 1.02),                                                  # perde no MAE
             ("SINT-M", "M", "LSTM", 1.0, 0.97, 0.98)]
    for ds, modo, bb, fr, f_rmse, f_mae in casos:
        base = dict(dataset=ds, modo=modo, backbone=bb, largura=1, fracao=fr, seed=0)
        linhas.append({**base, "variante": "densa", "val_RMSE": 1.0, "val_MAE": 0.8})
        linhas.append({**base, "variante": "densa+E", "val_RMSE": 0.98, "val_MAE": 0.79})
        linhas.append({**base, "variante": "zero_L1000_K10", "K": 10, "lambda": 1000,
                       "val_RMSE": 0.98 * f_rmse, "val_MAE": 0.79 * f_mae})
        linhas.append({**base, "variante": "proj_K10", "K": 10, "val_RMSE": 0.5, "val_MAE": 0.5})
    df = pd.DataFrame(linhas).reindex(columns=COLS)
    df["test_RMSE"] = rng.uniform(0.1, 9, len(df)) if lixo is None else lixo
    df["test_MAE"] = rng.uniform(0.1, 9, len(df)) if lixo is None else lixo
    if not com_teste:
        df = df.drop(columns=["test_RMSE", "test_MAE"])
    df.to_csv(caminho, index=False)


def resumo(caminho):
    melhor, esc = S.selecao_t1(S.ler_val(caminho))
    return melhor.to_json(), esc.to_json()


def test_tudo():
    with tempfile.TemporaryDirectory() as td:
        a, b, c, e = (Path(td) / n for n in ("a.csv", "b.csv", "c.csv", "e.csv"))
        csv_falso(a)
        csv_falso(b, lixo="lixo")
        csv_falso(c, com_teste=False)
        csv_falso(e, lixo=1e30)
        # 1
        assert not [x for x in S.ler_val(a).columns if x.startswith("test")]
        # 2 e 3
        ref = resumo(a)
        assert resumo(b) == ref and resumo(e) == ref, "o resultado mudou quando as colunas de teste mudaram"
        assert resumo(c) == ref, "o resultado mudou quando as colunas de teste foram removidas"
        # 4
        chamadas = []
        orig = pd.read_csv

        def espiao(*args, **kw):
            chamadas.append(kw.get("usecols"))
            return orig(*args, **kw)
        pd.read_csv = espiao
        try:
            S.ler_val(a)
        finally:
            pd.read_csv = orig
        assert chamadas and all(callable(u) or isinstance(u, (list, tuple)) for u in chamadas)
        for u in chamadas:
            nomes = [n for n in COLS if (u(n) if callable(u) else n in u)]
            assert not [n for n in nomes if n.startswith("test")], nomes
        # 5
        melhor, esc = S.selecao_t1(S.ler_val(a))
        q = {(r["dataset"], r["backbone"]): bool(r["qualifica"]) for _, r in melhor.iterrows()}
        assert q[("SINT-U48", "MLP")] and q[("SINT-M", "LSTM")]
        assert not q[("SINT-U48", "TCN")] and not q[("ETTh1", "MLP")]
        assert len(esc) == 2


if __name__ == "__main__":
    test_tudo()
    print("test_selecionar: OK (o selecionador não lê colunas de teste)")
