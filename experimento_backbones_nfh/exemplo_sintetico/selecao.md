# Seleção da combinação (só validação)

Regra pré-registrada em `experimento_backbones_nfh/INSTRUCOES.md` §10 (21/11/2026, 17h20). Este script lê só as colunas
`val_*` de `selecao.csv`. Diferença relativa = (NFH − densa+E) / densa+E, média das 5 sementes (0 a 4).

| p | n | variante NFH | RMSE val NFH | RMSE val densa+E | RMSE val densa | rel. RMSE | rel. MAE | sementes com NFH melhor | qualifica |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 300 | conv_L1000 | 0,4680 | 0,4679 | 0,4686 | 0,0003 | -0,0000 | 2/5 | não |
| 8 | 300 | zero_L0 | 0,5485 | 0,4679 | 0,4686 | 0,1810 | 0,1804 | 1/5 | não |
| 8 | 300 | zero_L1000 | 0,5361 | 0,4679 | 0,4686 | 0,1489 | 0,1532 | 1/5 | não |
| 8 | 1000 | conv_L1000 | 0,3165 | 0,3165 | 0,3176 | -0,0002 | -0,0005 | 3/5 | não |
| 8 | 1000 | zero_L0 | 0,3208 | 0,3165 | 0,3176 | 0,0135 | 0,0133 | 0/5 | não |
| 8 | 1000 | zero_L1000 | 0,3149 | 0,3165 | 0,3176 | -0,0053 | -0,0047 | 5/5 | sim |
| 16 | 300 | conv_L1000 | 0,4215 | 0,4212 | 0,4215 | 0,0006 | 0,0012 | 2/5 | não |
| 16 | 300 | zero_L0 | 0,4147 | 0,4212 | 0,4215 | -0,0119 | -0,0128 | 3/5 | não |
| 16 | 300 | zero_L1000 | 0,4206 | 0,4212 | 0,4215 | -0,0022 | 0,0023 | 3/5 | não |
| 16 | 1000 | conv_L1000 | 0,3245 | 0,3277 | 0,3284 | -0,0100 | -0,0099 | 5/5 | sim |
| 16 | 1000 | zero_L0 | 0,3342 | 0,3277 | 0,3284 | 0,0197 | 0,0186 | 0/5 | não |
| 16 | 1000 | zero_L1000 | 0,3148 | 0,3277 | 0,3284 | -0,0395 | -0,0379 | 5/5 | sim |

## Escolha

**p = 16, n = 1000, variante zero_L1000** (rel. RMSE de validação -0,0395; rel. MAE -0,0379).

Motivo: entre as 3 combinações que cumprem a regra (média do RMSE relativo < 0, média do MAE relativo ≤ 0, NFH melhor na validação nas 5 sementes), é a de menor média do RMSE relativo de validação.
