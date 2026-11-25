# Controle das épocas extras: como rodar

Protocolo pré-registrado: `INSTRUCOES.md`. Esta pasta só importa o código da v3 (`src/v2_common.py`, `src/v3_common.py`,
`src/v3_benchmark.py`); nada de `src/`, `results/` ou do experimento de *backbones* é alterado.

## O que cada tarefa faz

Uma tarefa é (conjunto, backbone, semente), no dispositivo que a dissertação usou (`results/v3/dispositivos.json`):

- `densa`: retreinada do zero como em `v3_benchmark.rodar` (`set_seed`, `construir`, `train_model` com `CFG3`).
- `densa+E`: a partir dessa densa, 15 épocas com Adam (taxa 1e-4, *weight decay*, lote, *clip* e gerador de embaralhamento de
  `ajuste_estrutural`), perda MSE sem penalização; fica a época 0..15 de menor RMSE de validação (a época 0 é a densa).
- `nfh`: `densa_para_nfh(densa, K=10)` + `ajuste_estrutural` com os padrões da v3 (variante "seguro").
- `exata_s0` (24 tarefas, só semente 0): carrega `results/v3/modelos/<conj>__<bb>+densa__s0.pt` e roda o mesmo `densa+E`.

Asserts por variante: erro de conversão < 1e-5; val_RMSE(densa+E) ≤ val_RMSE(densa); val_RMSE(nfh) ≤ val_RMSE(densa)
(tolerância 1e-6; a conversão exata muda o RMSE de validação na 8.ª casa). Se um assert falha, a linha não é gravada, o
traceback vai para `resultados/logs/erro_*.txt`, as outras variantes seguem e a falha é refeita no próximo lançamento.

## Arquivos

| Arquivo | Função |
|---|---|
| `codigo/executar.py` | worker retomável (flock, reivindicação dinâmica, uma linha por variante) |
| `codigo/filas.py` | fila: 120 tarefas + 24 `exata_s0`; semente 0 de todos os pares primeiro |
| `codigo/analisar.py` | análise do §3 → `resultados/analise.json` e `relatorio/relatorio_controle.md` |
| `codigo/lanca.sh`, `parar.sh`, `escalar.sh`, `status.py` | operação |
| `codigo/colunas.py`, `codigo/treino_blocos.py` | colunas dos CSV; treino com validação em blocos de 1024 |
| `codigo/test_retomada.py`, `codigo/test_analisar.py` | testes (não tocam nos resultados reais) |

## Passo a passo

```bash
cd .
unset TESTE_MAX_EPOCAS TESTE_E_EXTRA            # a fase real recusa estas variáveis
.venv/bin/python experimento_controle_epocas/codigo/filas.py real        # cria resultados/filas/controle.json
bash experimento_controle_epocas/codigo/lanca.sh controle experimento_controle_epocas/resultados/filas/controle.json 1 1
setsid nohup bash experimento_controle_epocas/codigo/escalar.sh controle experimento_controle_epocas/resultados/filas/controle.json \
  > experimento_controle_epocas/resultados/logs/escalar.log 2>&1 < /dev/null &
.venv/bin/python experimento_controle_epocas/codigo/status.py controle
.venv/bin/python experimento_controle_epocas/codigo/analisar.py          # quando terminar (ou para ver o parcial)
bash experimento_controle_epocas/codigo/parar.sh controle                # se precisar parar
```

- `lanca.sh <fase> <fila> [n_gpu=1] [n_cpu=1]` só lança o worker que não estiver vivo; repetir não duplica. Máximo 3 GPU e 5 CPU.
- `escalar.sh` espera `experimento_backbones_nfh/resultados/logs/ORQUESTRADOR_FIM` e chama `lanca.sh` com 3 GPU + 5 CPU.
- Worker GPU só pega tarefas GPU; worker CPU (`CUDA_VISIBLE_DEVICES=''`, 2 threads) só pega tarefas CPU (`--qualquer` desfaz
  isso e muda o dispositivo; não usar na fase real). Com `--val-blocos auto` a GPU valida em blocos de 1024 janelas.
- Reiniciar é seguro: o que está no CSV é pulado; a reivindicação de worker morto é devolvida à fila.

## Fases

| Fase | Pasta | Uso |
|---|---|---|
| `controle` | `resultados/` | a real (`controle.csv`, `exata_s0.csv`) |
| `repro` | `resultados/repro/` | reprodução com épocas reais em pares pequenos (teste) |
| `smoke`, `smoke_cpu` | `resultados/smoke/<fase>/` | 2 épocas (`TESTE_MAX_EPOCAS=2 TESTE_E_EXTRA=2`); só aqui as variáveis valem |

## Colunas (`controle.csv`)

`conjunto, backbone, seed, variante, dispositivo, dispositivo_dissertacao, val_RMSE, val_MAE, test_RMSE, test_MAE, epocas,
epoca_escolhida, tempo_s, erro_conversao, K` e a reprodução: `diss_val_RMSE, diss_test_RMSE, diss_test_MAE` (linha da
dissertação: `<bb> + densa` sem variante, ou `<bb> + NFH`, `seguro`, K = 10; se houver duplicata por retomada, vale a última,
como em `v3_tabelas.carregar_runs`) e `dif_val_RMSE, dif_test_RMSE, dif_test_MAE` (retreinada menos dissertação). A
`densa+E` não tem linha na dissertação. Em `exata_s0.csv` as variantes são `densa_diss` (a densa carregada) e `densa+E`.

## Análise (`analisar.py`)

Média por par nas 5 sementes; `rel = NFH / controle − 1` nos 24 pares, controle ∈ {densa+E, densa}, em RMSE e em MAE:
vitórias, média, mediana, mín./máx., `wilcoxon(rel)` bilateral, `wilcoxon(rel − 0,01, alternative="less")` e a
sensibilidade às margens 0; 0,25%; 0,5% (como `v3_tabelas.comparacoes`; a função foi conferida contra
`results/v3/numeros_cap5.json`). Também épocas escolhidas, reprodução e `exata_s0`. Com a execução incompleta, usa só os
pares com as 5 sementes das 3 variantes e o relatório diz quantas tarefas faltam.
