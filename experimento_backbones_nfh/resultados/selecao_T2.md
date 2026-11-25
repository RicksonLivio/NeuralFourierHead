# Seleção T2 → confirmação (somente colunas de validação)

rel = média, sobre as sementes 0, 1 e 2, de (val da variante NFH / val da densa+E da mesma semente − 1). Passa se média rel_RMSE < 0, média rel_MAE ≤ 0 e rel_RMSE < 0 nas três sementes. Até 8 (máx. 2 por conjunto, 3 por backbone, incluindo a melhor sintética). Nenhuma coluna `test_*` foi lida.

Combinações × largura: 16; passam: 14; vão para a confirmação: 8.

| # | conjunto | modo | backbone | fração | largura | melhor variante | média rel RMSE | média rel MAE | pior semente (RMSE) | passa | confirmação |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | SINT-M | M | iTransformer | 0.25 | 2 | conv_L1000_K10 | -9.89% | -6.41% | -7.29% | sim | sim |
| 2 | ETTh1 | M | PatchTST | 0.25 | 2 | zero_L1000_K4 | -8.66% | -7.06% | -7.66% | sim | sim |
| 3 | ETTh1 | M | PatchTST | 0.25 | 1 | zero_L1000_K4 | -6.98% | -5.54% | -3.70% | sim | sim |
| 4 | SINT-M | M | PatchTST | 0.1 | 1 | conv_L1000_K4 | -5.61% | -5.38% | -3.98% | sim | sim |
| 5 | ETTh1 | M | iTransformer | 0.25 | 1 | zero_L1000_K4 | -5.47% | -5.10% | -4.36% | sim | não |
| 6 | SINT-M | M | iTransformer | 0.25 | 1 | conv_L1000_K10 | -4.63% | -2.71% | -3.59% | sim | não |
| 7 | air_quality | U | PatchTST | 1 | 1 | zero_L1000_K10 | -4.31% | -4.77% | -3.47% | sim | não |
| 8 | SINT-M | M | PatchTST | 0.1 | 2 | conv_L1000_K10 | -3.80% | -3.29% | -2.98% | sim | não |
| 9 | ETTh2 | U | iTransformer | 0.25 | 1 | zero_L1000_K4 | -2.94% | -3.13% | -2.38% | sim | sim |
| 10 | SINT-U48 | U | Linear | 0.1 | 1 | zero_L1000_K4 | -2.53% | -2.48% | -2.37% | sim | sim |
| 11 | SINT-M | M | Conv1D+LSTM | 1 | 2 | zero_L1000_K10 | -2.23% | -2.22% | -1.12% | sim | não |
| 12 | SINT-U48 | U | DLinear | 0.1 | 1 | zero_L1000_K4 | -2.13% | -2.14% | -2.06% | sim | sim |
| 13 | SINT-M | M | Conv1D+LSTM | 1 | 1 | conv_L1000_K4 | -1.70% | -1.51% | -0.00% | sim | não |
| 14 | seoul_bike | U | Conv1D+LSTM | 1 | 1 | conv_L1000_K10 | -1.62% | -1.22% | +0.00% | não | não |
| 15 | ETTh2 | U | FreqMLP | 0.25 | 1 | conv_L100_K10 | -1.39% | -1.69% | -0.70% | sim | sim |
| 16 | SINT-U48 | U | FreqMLP | 0.1 | 1 | zero_L100_K10 | -0.27% | -0.02% | +2.20% | não | não |
