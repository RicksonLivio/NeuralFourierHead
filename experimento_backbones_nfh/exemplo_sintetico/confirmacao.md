# Confirmação (sementes 10 a 19, teste)

Combinação escolhida na validação (`selecao.md`): p = 16, n = 1000, NFH = zero_L1000. Critério pré-registrado (INSTRUCOES.md §10): média do RMSE e do MAE de teste da NFH ≤ as da densa e as da densa+E, e NFH com RMSE de teste menor que o da densa+E em ≥ 8 de 10 sementes.

| variante | RMSE teste (média ± dp) | MAE teste (média ± dp) |
|---|---|---|
| densa | 0,3282 ± 0,0116 | 0,2625 ± 0,0088 |
| densa+E | 0,3285 ± 0,0124 | 0,2625 ± 0,0093 |
| NFH | 0,3038 ± 0,0015 | 0,2445 ± 0,0011 |

Vitórias da NFH por semente (estritamente menor), de 10:

| contra | RMSE | MAE |
|---|---|---|
| densa | 10 | 10 |
| densa+E | 10 | 10 |

Por semente (RMSE de teste):

| semente | densa | densa+E | NFH |
|---|---|---|---|
| 10 | 0,3399 | 0,3365 | 0,3039 |
| 11 | 0,3543 | 0,3601 | 0,3018 |
| 12 | 0,3259 | 0,3251 | 0,3066 |
| 13 | 0,3226 | 0,3237 | 0,3027 |
| 14 | 0,3164 | 0,3164 | 0,3053 |
| 15 | 0,3248 | 0,3249 | 0,3033 |
| 16 | 0,3246 | 0,3246 | 0,3048 |
| 17 | 0,3218 | 0,3218 | 0,3033 |
| 18 | 0,3342 | 0,3313 | 0,3020 |
| 19 | 0,3170 | 0,3205 | 0,3043 |

Por semente (MAE de teste):

| semente | densa | densa+E | NFH |
|---|---|---|---|
| 10 | 0,2719 | 0,2686 | 0,2447 |
| 11 | 0,2823 | 0,2863 | 0,2432 |
| 12 | 0,2612 | 0,2604 | 0,2464 |
| 13 | 0,2581 | 0,2586 | 0,2434 |
| 14 | 0,2538 | 0,2538 | 0,2453 |
| 15 | 0,2598 | 0,2595 | 0,2441 |
| 16 | 0,2592 | 0,2592 | 0,2451 |
| 17 | 0,2576 | 0,2576 | 0,2445 |
| 18 | 0,2668 | 0,2642 | 0,2429 |
| 19 | 0,2545 | 0,2565 | 0,2450 |

Médias cumprem o critério: True. Vitórias ≥ 8: True.

**Veredito: SUCESSO**
