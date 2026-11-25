# Código do experimento de backbones da NFH

Protocolo (pré-registrado): `../INSTRUCOES.md`. Raiz do repositório: `xai_transformer/`. Python: `.venv/bin/python`.
Nada em `src/` nem em `teste_exemplificacao_nfh/` foi alterado; os dois são importados.

## Arquivos

| Arquivo | Função |
|---|---|
| `backbones.py` | Os 10 backbones (os 9 do §4: `Linear`, `DLinear`, `MLP`, `LSTM`, `Conv1D+LSTM`, `TCN`, `PatchTST`, `TSMixer`, `iTransformer`; mais `FreqMLP`, §10 item 1), `REGISTRO` e `construir_modelo` (RevIN + backbone + cabeça densa ou NFH). Acrescentar um backbone = escrever a classe `(d, L, largura)` com atributo `p` e uma linha em `REGISTRO`. |
| `sinteticos.py` | Gera SINT-U48 e SINT-M, grava CSV, `dados/catalogo.json` (junta, sem apagar as entradas da pesquisa) e `dados/sint_u48_oraculo.json`; confere o SINT-M contra `src/v3_sintetico_multivariado.py`. |
| `dados_bb.py` | Carregador que lê os dois catálogos (`teste_exemplificacao_nfh/dados/` e `experimento_backbones_nfh/dados/`). |
| `executar.py` | Executor retomável; cada variante tem o seu `try/except`. |
| `treino_bb.py` | `train_model_blocos`: cópia de `train_model` com a validação em blocos de 1024 janelas (memória da GPU). |
| `filas.py` | Gera `resultados/filas/t1.json`, `t1_B2.json` (`--b2 --opcao-m`, §10) e `smoke.json`. |
| `selecionar.py`, `test_selecionar.py` | Regras T1→T2 e T2→confirmação (só `val_*`) e o teste que falha se o selecionador ler `test_*`. |
| `status.py` | Progresso por fase e por worker, ETA. |
| `lanca.sh`, `parar.sh` | Lançam e param os workers. |
| `medir_tempo.py`, `estimar_t1.py` | Medem o tempo por época e estimam o tempo da T1. |

## Backbones: dimensões

Entrada `x: [B, L = 96, d]` já normalizada pelo `InstanceNorm`; saída `h: [B, p]`. O alvo é a última variável.
`p` com largura 1 / 2: Linear 96·d (sem largura); DLinear 192·d (sem largura); MLP 256 / 512; LSTM 64 / 128;
Conv1D+LSTM 128 / 256; TCN 64 / 128; PatchTST 12·d·64 / 12·d·128 (d = 1: 768 / 1536); TSMixer 64·d / 128·d;
iTransformer 128 / 256. As fontes (artigo e URL do código oficial) e as simplificações estão na docstring de cada classe.
Em modo U o iTransformer tem um só *token* e vira, na prática, um MLP (dito na docstring).
As URLs das docstrings vêm do protocolo e do relatório da pesquisa (`pesquisa/relatorio_backbones.md` §1); o
implementador não reabriu os repositórios oficiais, só implementou as arquiteturas.
`FreqMLP` entrou na rodada 2 (protocolo §10, item 1; ver seção "Rodada 2" abaixo). `CycleMLP` não entra (§10).

## Conjuntos sintéticos

- **SINT-U48** (`sinteticos.gerar_u48`, semente 2026, N = 12 000 h, σ = 0,30): tendência lenta + ciclo de 24 h + ciclo
  de 12 h, com amplitude e fase lentas. A fórmula completa está na docstring de `sinteticos.py`; a pesquisa leu essa
  fórmula e a adotou (`pesquisa/relatorio_backbones.md` §4.1). Oráculo = o sinal sem ruído do futuro. Valores gravados
  em `dados/sint_u48_oraculo.json`: **referência amostral** (RMSE do oráculo sobre as 1657 janelas de teste avaliadas, escala
  padronizada pelo treino) 0,26707, chave `RMSE_oraculo_padronizado_janelas_teste`; sobre o segmento de teste inteiro 0,26584
  (só registro); RMSE de um ridge linear 0,3013. Não é piso rígido: é o RMSE de uma amostra finita de ruído. O sinal sem ruído está em `dados/SINT-U48_com_sinal.csv` (coluna `sinal_sem_ruido`).
- **SINT-M**: `gerar()` de `src/v3_sintetico_multivariado.py` sem alteração. Para caber no formato do catálogo (alvo na
  última coluna), o próprio `y` passado entra como quinta variável (**d = 5**, na v3 era d = 4). A conferência contra
  o gerador deu diferença máxima de 2,2e-4 em x e 1,9e-4 em y (CSV com 9 algarismos e desvio com ddof = 1).
- Para rodar de novo: `python sinteticos.py` (reescreve os CSV dos sintéticos; as entradas da pesquisa no catálogo ficam).

## Variantes (§3 do protocolo) e verificações

`densa`, `densa+E` (E = 15, taxa 1e-4, época 0 a 15 de menor RMSE de validação), `proj_K10`, `conv_L{λ}_K{K}`
(conversão exata + ajuste estrutural de 15 épocas com seleção segura), `zero_L{λ}_K{K}` (NFH do zero com `train_model`).
Fração de treino = **últimas** janelas do treino. A fila é validada ao iniciar: `q + 1 + 2K ≤ H` (K = 10 dá 23 ≤ 48).
Asserts em cada tarefa (falha → `resultados/logs/erro_<fase>_<tarefa>.txt`, o worker segue): erro de conversão < 1e-5
(gravado em `erro_conversao`), RMSE de validação da `densa+E` ≤ o da `densa`, RMSE de validação da `conv` ≤ o da `densa`
(tolerância numérica de 1e-7). Os valores de cada assert ficam em `resultados/logs/<fase>_tarefas.jsonl`.
Colunas do CSV: dataset, modo, backbone, largura, fracao, seed, variante, K, lambda, val_RMSE, val_MAE, test_RMSE, test_MAE,
cobertura_mediana, epocas, epoca_escolhida, tempo_s, params, n_tr, n_val, n_te, d, erro_conversao.
`epoca_escolhida`: densa = época de menor RMSE de validação (1 a `epocas`); densa+E e conv = época 0..15 escolhida;
zero com λ = 0 idem à densa; zero com λ > 0 fica vazio (o treino escolhe pelo objetivo MSE + λρ, não pelo RMSE).
Ingênuas (`ingenua_media`, `ingenua_ultimo`, `ingenua_sazonal24`): uma vez por (dataset, modo), com backbone/largura/fração/semente vazios.
`cobertura_mediana` = mediana de 1 − ε_R no teste, com o K da variante (K = 10 na `densa` e na `densa+E`).

**Pontos a saber:** (1) `n_tr` na linha é o nº de janelas de treino realmente usadas. (2) A `densa` de uma tarefa é guardada
em `resultados/modelos/` enquanto a tarefa não termina e apagada depois (na confirmação os modelos e previsões de teste ficam);
isso permite retomar uma tarefa interrompida sem retreinar. (3) Como a tarefa é a unidade, uma tarefa interrompida refaz
só as variantes que faltam, mas a `densa` é retreinada se o checkpoint não existir; com a mesma semente o resultado é o mesmo na GPU
só até o não determinismo do cuDNN.

## Smoke test (resultado em `resultados/smoke.csv` GPU e `resultados/smoke_cpu.csv` CPU)

```
cd experimento_backbones_nfh/codigo
.venv/bin/python filas.py smoke                       # (a partir da raiz: .venv/bin/python experimento_backbones_nfh/codigo/filas.py smoke)
TESTE_MAX_EPOCAS=2 TESTE_E_EXTRA=2 .venv/bin/python experimento_backbones_nfh/codigo/executar.py --fase smoke --fila experimento_backbones_nfh/resultados/filas/smoke.json --device cuda
TESTE_MAX_EPOCAS=2 TESTE_E_EXTRA=2 CUDA_VISIBLE_DEVICES='' .venv/bin/python experimento_backbones_nfh/codigo/executar.py --fase smoke_cpu --fila experimento_backbones_nfh/resultados/filas/smoke.json --device cpu
```
Os CSV de smoke e seus números **não entram em nenhuma tabela da dissertação** (2 épocas só testam o encanamento).

## Como rodar as etapas (a T1 NÃO foi lançada)

```
# T1 (fila já gerada em resultados/filas/t1.json; ver "Tempo" antes de lançar)
experimento_backbones_nfh/codigo/lanca.sh t1 experimento_backbones_nfh/resultados/filas/t1.json 3 6
.venv/bin/python experimento_backbones_nfh/codigo/status.py t1
experimento_backbones_nfh/codigo/parar.sh t1
.venv/bin/python experimento_backbones_nfh/codigo/selecionar.py t1     # -> resultados/selecao_T1.md e filas/t2.json
experimento_backbones_nfh/codigo/lanca.sh t2 experimento_backbones_nfh/resultados/filas/t2.json 3 6
.venv/bin/python experimento_backbones_nfh/codigo/selecionar.py t2     # -> resultados/selecao_T2.md e filas/confirmacao.json
experimento_backbones_nfh/codigo/lanca.sh confirmacao experimento_backbones_nfh/resultados/filas/confirmacao.json 3 6
```
Para regerar a fila com corte: `.venv/bin/python experimento_backbones_nfh/codigo/filas.py t1 --sem-frac025 ETTh1 ...`
(só se o coordenador registrar o corte no §10 do protocolo). `--reais-m` inclui os conjuntos reais no modo M (não foi
pedido pelo protocolo; só dos que têm covariáveis).
`lanca.sh` recusa lançar se houver workers vivos da fase e apaga as reivindicações antigas (`resultados/claims/<fase>/`);
workers GPU e CPU usam 2 fios. As tarefas são reivindicadas dinamicamente, da mais cara para a mais barata
(ordem da fila). Para reiniciar depois de uma queda: `parar.sh <fase>` e `lanca.sh` de novo; o que está no CSV é pulado.

### Fila T1 gerada

`resultados/filas/t1.json`: 234 tarefas de treino (10 conjuntos reais U × 2 frações × 9 backbones = 180, mais
SINT-U48 e SINT-M × 3 frações × 9 backbones = 54) e 12 de ingênuas. Reais: os 7 de `teste_exemplificacao_nfh/dados/` e os 3 da
pesquisa (`melbourne_pedestrian_southbank`, `electricity_mt156`, `opsd_de_solar`). Variantes: `densa`, `densa+E`,
`proj_K10`, `conv_L1000_K10`, `zero_L100_K10`, `zero_L1000_K10` e, nos sintéticos, `zero_L1000_K4`.
Largura 1, semente 0, modo U nos reais, SINT-M em M (reais em M: `--reais-m`, 378 tarefas).

## Tempo

Medido por `medir_tempo.py` (1 e 3 épocas de `train_model`, fração 0,25, SINT-U48 d = 1 e SINT-M d = 5): segundos por época
**sozinho**, em `resultados/tempo_epoca.json`; **sob carga** (3 processos GPU + 6 CPU de medição ao mesmo tempo, 2 fios cada),
nos nove arquivos `resultados/tempo_epoca_carga_*.json`. Tabela sozinho (s por época, ~2060 janelas de treino; w = largura, d = nº de variáveis; fonte `tempo_epoca.json`; os tempos de GPU foram medidos com os fios padrão do PyTorch e saem de uma diferença entre 3 e 1 época, então têm ruído: o DLinear w2 d1 deu 0,30 s e o w1 d1 deu 0,14 s):

| backbone | GPU w1 d1 | GPU w1 d5 | GPU w2 d1 | GPU w2 d5 | CPU w1 d1 | CPU w1 d5 | CPU w2 d1 | CPU w2 d5 |
|---|---|---|---|---|---|---|---|---|
| Linear | 0.11 | 0.14 | 0.19 | 0.12 | 0.06 | 0.07 | 0.05 | 0.07 |
| DLinear | 0.14 | 0.15 | 0.30 | 0.15 | 0.07 | 0.11 | 0.07 | 0.12 |
| MLP | 0.17 | 0.22 | 0.24 | 0.16 | 0.12 | 0.20 | 0.27 | 0.46 |
| LSTM | 0.21 | 0.34 | 0.28 | 0.23 | 1.27 | 1.68 | 3.81 | 3.51 |
| Conv1D+LSTM | 0.32 | 0.40 | 0.32 | 0.38 | 1.98 | 2.00 | 6.09 | 5.36 |
| TCN | 0.69 | 0.51 | 0.56 | 0.62 | 4.22 | 4.32 | 10.46 | 10.45 |
| PatchTST | 0.58 | 0.53 | 0.55 | 0.77 | 0.92 | 2.91 | 1.50 | 7.12 |
| TSMixer | 0.54 | 0.67 | 0.54 | 0.40 | 0.64 | 0.75 | 0.99 | 2.17 |
| iTransformer | 0.65 | 0.53 | 0.54 | 0.56 | 0.34 | 0.89 | 1.16 | 2.21 |

 Observação: em Linear, DLinear e MLP a CPU é tão rápida quanto a GPU
ou mais; o custo domina em LSTM, Conv1D+LSTM e TCN na CPU (4 a 10 s/época).

Estimativa da T1 (`estimar_t1.py`, simulação com 3 GPU + 6 CPU, fatores de carga medidos, 234 tarefas):

| Cenário | épocas por treino do zero | tempo |
|---|---|---|
| típico (mediana de épocas da densa do MLP na triagem da v3: 12) | 12 | **6,3 h** |
| pior caso (60 épocas sem parada antecipada) | 60 | 20,3 h |

Isso **passa de ~4 h**. Cortes avaliados (mesma simulação, cenário típico / pior caso):

| Corte | tarefas | típico | pior caso |
|---|---|---|---|
| nenhum | 234 | 6,29 h | 20,25 h |
| §7 literal, só em Jena e Household (os com mais janelas) sem fração 0,25 | 224 | 5,99 h | 19,33 h |
| §7 literal em todos os 10 reais | 144 | 5,35 h | 16,94 h |
| **proposta B**: reais só com fração 0,25; sintéticos com as três frações | 144 | **1,85 h** | 6,41 h |
| B sem Conv1D+LSTM e TCN (os mais lentos na CPU) | 112 | 1,09 h | 3,70 h |

O corte do §7 (retirar a fração 0,25 dos reais) **não basta**: o custo está nas frações 1,0 dos reais, que têm 6 000 a 18 000
janelas de treino. **Recomendação:** aplicar a proposta B (nos reais, só a fração 0,25; a fração 1,0 dos reais fica para a
confirmação/T2 dos que passarem), porque a hipótese do protocolo é que a NFH ganha com poucos dados, e porque ela cabe em
~2 h no cenário típico. Isso muda o §6 e precisa ser registrado no §10 pelo coordenador antes de lançar:
(`filas.py t1 --so-frac025-reais`; a fila do corte B já está gerada em `resultados/filas/t1_corteB.json`, e a completa continua em
`t1.json`). Incertezas da estimativa: o tempo por janela vem de treinos de
fração 0,25 (a validação, fixa, entra no custo por janela: superestima os treinos grandes e subestima os de fração 0,1);
a mediana de épocas vem do MLP; os fatores de carga variam de 2,4 a 8,3 entre backbones; o escalonador simulado manda
tarefas para GPU e CPU sem considerar qual é mais rápida para o backbone.


## Rodada 2 (21/11/2026, tarde): mudanças depois da revisão

- **Validação em blocos** (`treino_bb.py`): `executar.py` e `medir_tempo.py` usam `train_model_blocos`. `medir_memoria.py` e
  `verificar_blocos.py` repetem a medição de memória e a comparação do `val_RMSE`.
- **Erro por variante**: uma variante que falha (assert, memória) grava `resultados/logs/erro_<fase>_<tarefa>__<variante>.txt`
  (com `erro_conversao`) e as demais seguem; a lista de erros vai para `logs/<fase>_tarefas.jsonl` (chave `registro.erros`, `ok = false`).
  O que falhou não entra no CSV e é refeito ao relançar; o checkpoint da densa fica enquanto houver erro. Se a densa já está no CSV e o
  checkpoint sumiu, a densa é retreinada e o aviso vai para `registro.avisos` e para o terminal. `test_falha_variante.py` confere isso.
- **Variáveis de teste**: `lanca.sh` e `executar.py` recusam `TESTE_MAX_EPOCAS` e `TESTE_E_EXTRA` (mesmo vazias) nas fases t1, t2 e
  confirmacao; o cabeçalho do log mostra `E_EXTRA` e `MAX_EPOCAS` efetivos.
- **FreqMLP**: `rfft` da janela, bins k = 1..12 (parte real e imaginária, 24·d entradas), `Linear(24d, w)`, GELU, `Linear(w, w)`, GELU,
  p = w = 64·largura (FITS, Xu et al., arXiv 2307.03756; docstring em `backbones.py`).
- **Filas**: `filas.py t1 --b2 --opcao-m --saida resultados/filas/t1_B2.json` (B2 e opção M do §10). `t1.json` e `t1_corteB.json` foram
  regerados com os 10 backbones. Estimativa de tempo (3 GPU + 5 CPU): `resultados/estimativa_t1_B2.json`.
  Lançar (só depois da decisão do coordenador): `codigo/lanca.sh t1 experimento_backbones_nfh/resultados/filas/t1_B2.json 3 5`.
- **T2**: `selecionar.VAR_T2` ganhou `conv_L0_K10` (controle da parametrização, §10 item 4).
- **status.py**: mostra o que falta em relação à fila (sem nenhuma linha, parciais, por backbone e por conjunto);
  `status.py t1 --fila resultados/filas/t1_B2.json` compara com a fila B2.
- Smoke da rodada 2: fases `smoke2` (GPU) e `smoke2_cpu`; `conferir_smoke.py <fase>` confere contagens e asserts.
- Os tempos por época da tabela da seção "Tempo" acima são os da rodada 1 (9 backbones, `train_model`); `resultados/tempo_epoca.json`
  foi remedido na rodada 2 com os 10 backbones e `train_model_blocos`.
