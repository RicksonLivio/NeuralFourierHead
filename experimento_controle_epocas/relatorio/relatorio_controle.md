# Controle das épocas extras: relatório

Gerado por `codigo/analisar.py` a partir de `resultados/analise.json`; nenhum número foi digitado à mão. Protocolo: `INSTRUCOES.md` (§3). `rel` = RMSE (ou MAE) da NFH dividido pelo do controle, menos 1; negativo favorece a NFH. Nenhuma escolha usou o teste.

Execução completa: 144 tarefas (120 de treino e 24 exata_s0); 24 pares completos.

## 1. Resultado principal: NFH contra a densa com as mesmas épocas

| comparação | pares | vitórias (rel < 0) | média de rel | mediana de rel | mín. a máx. | Wilcoxon bilateral (p) | não inferioridade 1% (p) |
|---|---|---|---|---|---|---|---|
| NFH × densa+E (RMSE) | 24 | 13 | -0,05% | -0,13% | -2,85% a +2,47% | 0,8553 | 5,46e-04 |
| NFH × densa+E (MAE) | 24 | 13 | -0,07% | -0,07% | -3,52% a +1,84% | 0,9441 | 2,47e-04 |
| NFH × densa (RMSE) | 24 | 16 | -0,43% | -0,46% | -4,04% a +2,47% | 0,1780 | 5,38e-05 |
| NFH × densa (MAE) | 24 | 15 | -0,55% | -0,58% | -4,35% a +1,84% | 0,1280 | 3,81e-05 |
| densa+E × densa (RMSE), descritivo | 24 | 13 | -0,38% | -0,07% | -2,05% a +0,56% | 0,0228 | 9,02e-06 |
| densa+E × densa (MAE), descritivo | 24 | 15 | -0,48% | -0,10% | -2,50% a +0,49% | 0,0124 | 9,02e-06 |

As duas últimas linhas mostram o efeito das épocas extras sozinho (densa+E contra densa), sem a NFH; são descritivas e não fazem parte da análise pré-registrada. Nelas, a coluna de vitórias conta os pares em que a densa+E é melhor.

- NFH × densa+E, RMSE: a NFH venceu em 13 de 24 pares; média de rel -0,05%; o teste de não inferioridade com margem de 1% deu p = 5,46e-04 (rejeita a hipótese de a NFH ser pior que o controle em mais de 1%, ao nível de 5%); o Wilcoxon bilateral deu p = 0,8553 (não há diferença detectável ao nível de 5%). Por backbone (vitórias/pares; média de rel): MLP 2/8, +0,79%; LSTM 5/8, -0,06%; Conv1D+LSTM 6/8, -0,86%.
- NFH × densa+E, MAE: a NFH venceu em 13 de 24 pares; média de rel -0,07%; o teste de não inferioridade com margem de 1% deu p = 2,47e-04 (rejeita a hipótese de a NFH ser pior que o controle em mais de 1%, ao nível de 5%); o Wilcoxon bilateral deu p = 0,9441 (não há diferença detectável ao nível de 5%). Por backbone (vitórias/pares; média de rel): MLP 3/8, +0,56%; LSTM 5/8, -0,09%; Conv1D+LSTM 5/8, -0,69%.
- NFH × densa, RMSE: a NFH venceu em 16 de 24 pares; média de rel -0,43%; o teste de não inferioridade com margem de 1% deu p = 5,38e-05 (rejeita a hipótese de a NFH ser pior que o controle em mais de 1%, ao nível de 5%); o Wilcoxon bilateral deu p = 0,1780 (não há diferença detectável ao nível de 5%). Por backbone (vitórias/pares; média de rel): MLP 4/8, +0,12%; LSTM 6/8, -0,40%; Conv1D+LSTM 6/8, -1,00%.
- NFH × densa, MAE: a NFH venceu em 15 de 24 pares; média de rel -0,55%; o teste de não inferioridade com margem de 1% deu p = 3,81e-05 (rejeita a hipótese de a NFH ser pior que o controle em mais de 1%, ao nível de 5%); o Wilcoxon bilateral deu p = 0,1280 (não há diferença detectável ao nível de 5%). Por backbone (vitórias/pares; média de rel): MLP 5/8, -0,25%; LSTM 5/8, -0,42%; Conv1D+LSTM 5/8, -0,99%.

Sensibilidade à margem da não inferioridade (p-valores):

| comparação | 0% | 0,25% | 0,5% | 1% |
|---|---|---|---|---|
| NFH × densa+E (RMSE) | 0,4277 | 0,1514 | 0,0245 | 5,46e-04 |
| NFH × densa+E (MAE) | 0,5391 | 0,2195 | 0,0346 | 2,47e-04 |
| NFH × densa (RMSE) | 0,0890 | 0,0197 | 0,0024 | 5,38e-05 |
| NFH × densa (MAE) | 0,0640 | 0,0097 | 0,0014 | 3,81e-05 |

## 2. Valores por par (RMSE de teste, média das 5 sementes, escala padronizada)

| conjunto | backbone | densa | densa+E | NFH | rel NFH × densa+E | rel NFH × densa |
|---|---|---|---|---|---|---|
| ETTh1-U | MLP | 0,2054 | 0,2032 | 0,2034 | +0,09% | -0,99% |
| ETTh1-U | LSTM | 0,2069 | 0,2073 | 0,2100 | +1,32% | +1,50% |
| ETTh1-U | Conv1D+LSTM | 0,2129 | 0,2141 | 0,2141 | +0,03% | +0,59% |
| ETTh1-M | MLP | 0,2153 | 0,2148 | 0,2117 | -1,47% | -1,69% |
| ETTh1-M | LSTM | 0,2055 | 0,2058 | 0,2051 | -0,34% | -0,19% |
| ETTh1-M | Conv1D+LSTM | 0,2124 | 0,2124 | 0,2088 | -1,70% | -1,70% |
| ETTh2-U | MLP | 0,3254 | 0,3228 | 0,3223 | -0,17% | -0,97% |
| ETTh2-U | LSTM | 0,3170 | 0,3148 | 0,3140 | -0,24% | -0,94% |
| ETTh2-U | Conv1D+LSTM | 0,3211 | 0,3193 | 0,3158 | -1,07% | -1,64% |
| ETTh2-M | MLP | 0,3558 | 0,3574 | 0,3625 | +1,44% | +1,88% |
| ETTh2-M | LSTM | 0,3296 | 0,3293 | 0,3280 | -0,40% | -0,48% |
| ETTh2-M | Conv1D+LSTM | 0,3492 | 0,3492 | 0,3503 | +0,31% | +0,31% |
| Jena-U | MLP | 0,3727 | 0,3724 | 0,3766 | +1,12% | +1,05% |
| Jena-U | LSTM | 0,3769 | 0,3771 | 0,3752 | -0,50% | -0,44% |
| Jena-U | Conv1D+LSTM | 0,3795 | 0,3815 | 0,3777 | -1,01% | -0,47% |
| Jena-M | MLP | 0,3683 | 0,3608 | 0,3689 | +2,25% | +0,15% |
| Jena-M | LSTM | 0,3713 | 0,3668 | 0,3591 | -2,09% | -3,30% |
| Jena-M | Conv1D+LSTM | 0,3686 | 0,3663 | 0,3660 | -0,08% | -0,71% |
| AirQ-U | MLP | 0,5932 | 0,5841 | 0,5875 | +0,60% | -0,96% |
| AirQ-U | LSTM | 0,6532 | 0,6460 | 0,6529 | +1,07% | -0,04% |
| AirQ-U | Conv1D+LSTM | 0,6652 | 0,6570 | 0,6383 | -2,85% | -4,04% |
| AirQ-M | MLP | 0,6822 | 0,6822 | 0,6991 | +2,47% | +2,47% |
| AirQ-M | LSTM | 0,6634 | 0,6634 | 0,6678 | +0,66% | +0,66% |
| AirQ-M | Conv1D+LSTM | 0,6494 | 0,6509 | 0,6474 | -0,54% | -0,31% |

## 3. Épocas escolhidas pela validação

- densa+E: 67 de 120 execuções ficaram na época 0 (a própria densa) e 0 na época 15; média 1,04, mediana 0,0. Na época 0, por backbone: MLP 15/40; LSTM 24/40; Conv1D+LSTM 28/40. Histograma (época: execuções): 0: 67, 1: 28, 2: 10, 3: 3, 4: 5, 5: 3, 6: 1, 7: 2, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 1, 14: 0, 15: 0.
- NFH (ajuste estrutural, seguro): 25 de 120 execuções ficaram na época 0 (a própria densa) e 6 na época 15; média 4,97, mediana 4,0. Na época 0, por backbone: MLP 5/40; LSTM 9/40; Conv1D+LSTM 11/40. Histograma (época: execuções): 0: 25, 1: 12, 2: 16, 3: 5, 4: 7, 5: 12, 6: 7, 7: 3, 8: 4, 9: 6, 10: 3, 11: 3, 12: 0, 13: 6, 14: 5, 15: 6.

## 4. Reprodução da dissertação (retreino com a mesma semente e o mesmo dispositivo)

- densa retreinada (120 linhas): diferença absoluta do RMSE de teste para a dissertação, média 0,0049, máxima 0,0597; MAE, média 0,0038, máxima 0,0432; 67 de 120 com diferença menor que 1e-6 e 78 menor que 1e-3. Por dispositivo (RMSE: média; máxima; n com < 1e-6): GPU 0,0029; 0,0430; 50/79; CPU 0,0086; 0,0597; 17/41.
- NFH retreinada (120 linhas): diferença absoluta do RMSE de teste para a dissertação, média 0,0043, máxima 0,0473; MAE, média 0,0033, máxima 0,0447; 64 de 120 com diferença menor que 1e-6 e 82 menor que 1e-3. Por dispositivo (RMSE: média; máxima; n com < 1e-6): GPU 0,0029; 0,0431; 50/79; CPU 0,0069; 0,0473; 14/41.

Nos mesmos 24 pares, a dissertação (colunas `diss_*`) tem NFH × densa com 17 vitórias e média de rel -0,59%; a repetição acima, NFH × densa (RMSE), tem 16 vitórias e média -0,43%.

Conferência da função estatística: aplicada às linhas da dissertação, reproduz `results/v3/numeros_cap5.json` (NFH × densa, RMSE: 17 vitórias, média de rel -0,59%) com diferença absoluta máxima 2,69e-17 (RMSE) e 8,67e-17 (MAE).

## 5. Comparação exata na semente 0 (densa da dissertação + E épocas × NFH da dissertação)

24 pares gravados. A densa carregada reproduz a linha da dissertação com diferença máxima de RMSE de teste 2,65e-08. Época 0 escolhida: densa+E em 14 pares, NFH da dissertação em 6 pares.

| comparação | pares | vitórias (rel < 0) | média de rel | mediana de rel | mín. a máx. | Wilcoxon bilateral (p) | não inferioridade 1% (p) |
|---|---|---|---|---|---|---|---|
| NFH × densa+E exata (RMSE) | 24 | 15 | -0,55% | -0,13% | -4,37% a +1,99% | 0,1875 | 1,23e-05 |
| NFH × densa+E exata (MAE) | 24 | 11 | -0,07% | +0,00% | -4,71% a +4,36% | 0,9888 | 0,0089 |
| NFH × densa da dissertação (RMSE) | 24 | 18 | -0,63% | -0,88% | -4,37% a +2,48% | 0,0249 | 2,21e-05 |
| NFH × densa da dissertação (MAE) | 24 | 13 | -0,27% | -0,00% | -4,71% a +4,36% | 0,4732 | 0,0036 |
| densa+E exata × densa da dissertação (RMSE), descritivo | 24 | 12 | -0,07% | -0,00% | -1,97% a +6,55% | 0,0468 | 1,46e-04 |

## 6. Verificações e custo

- Erro de conversão máximo nas linhas: 2,92e-06 (limite 1e-5).
- Execuções com val_RMSE(densa+E) acima do da densa (tolerância 1e-6): 0; da NFH: 0.
- Tempo, densa: 120 linhas, média 94,0 s, soma 3,13 h.
- Tempo, densa+E: 120 linhas, média 101,1 s, soma 3,37 h.
- Tempo, nfh: 120 linhas, média 121,4 s, soma 4,05 h.

Linhas: 360 em controle.csv e 48 em exata_s0.csv.
