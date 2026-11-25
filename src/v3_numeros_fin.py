"""Versão 3, Seção 5.6 (séries financeiras, experimento secundário): números novos sobre a figura da v2
(item M55 do orientador), lidos dos mesmos arquivos que geram fig_rmse_por_conjunto (src/v2c_figuras.py).
  * quais barras são negativas (modelo com RMSE menor que a previsão ingênua pela média de treino);
  * diferença relativa NFH x densa por conjunto (a "única exceção" do texto).
Saída: results/v3/numeros_financeiro.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "v2"
MODELOS = {"NFHn-l1000": "Conv1D+LSTM + NFH", "ConvLSTM+Densa": "Conv1D+LSTM + densa", "N-BEATS-I": "N-BEATS-I",
           "LSTM": "LSTM", "MLP": "MLP"}
CONJ = ["BNB_USD", "BTC_USD", "ETH_USD", "SOL_USD", "XRP_USD", "AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]


def main():
    runs = pd.read_csv(RES / "forecast_runs.csv")
    naive = pd.read_csv(RES / "forecast_naive.csv")
    naive = naive[naive.model == "Ingênuo (média)"].set_index("dataset").RMSE
    neg, rel_nfh = [], {}
    for ds in CONJ:
        for m, rot in MODELOS.items():
            v = runs[(runs.dataset == ds) & (runs.model == m)].RMSE.values
            r = 100 * (v.mean() / naive[ds] - 1)
            if r < 0:
                neg.append({"conjunto": ds, "modelo": rot, "rel_pct": round(float(r), 3)})
        nf = runs[(runs.dataset == ds) & (runs.model == "NFHn-l1000")].RMSE.mean()
        de = runs[(runs.dataset == ds) & (runs.model == "ConvLSTM+Densa")].RMSE.mean()
        rel_nfh[ds] = round(float(100 * (nf - de) / de), 3)
    out = {"barras_negativas": neg, "rel_nfh_vs_densa_pct": rel_nfh,
           "conjuntos_nfh_pior_que_densa": [k for k, v in rel_nfh.items() if v > 0]}
    (ROOT / "results" / "v3" / "numeros_financeiro.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                                      encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
