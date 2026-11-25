# Experimento de backbones para a NFH: protocolo e regras

Raiz: `.` (caminhos relativos a ela). Pasta: `experimento_backbones_nfh/`.
Regras gerais da rodada: `.kiro/specs/rodada-20261121/INSTRUCOES_RODADA.md`. Este protocolo foi escrito em 21/11/2026,
antes de qualquer execução desta pasta. Mudanças posteriores entram no §10 com data e motivo.

## 1. Pergunta

Em que *backbones* e conjuntos de dados a NFH, no lugar da camada densa final, tem RMSE **e** MAE de teste iguais ou
menores que os da densa, e com que tamanho de treino? O foco principal é um conjunto **sintético**, em que a verdade é
conhecida e a explicação pode ser conferida à mão. O relatório deixa claro que os casos de vitória foram procurados
(exemplificação) e que isso não é regra geral. Todas as combinações testadas e todas as derrotas são relatadas.

Hipótese de partida (de `teste_exemplificacao_nfh/exemplo_livro/resultados_treino_B.csv`): a NFH ganha quando o futuro está
quase todo em T ⊕ S e há poucos dados de treino, porque a penalização do resíduo reduz a variância da cabeça. Com muitos
dados, empata. Nos dados reais, a NFH do zero perdeu com o MLP (`teste_exemplificacao_nfh/resultados/zero.csv`).

## 2. Estrutura

```
experimento_backbones_nfh/
  INSTRUCOES.md     este arquivo
  pesquisa/         relatorio_backbones.md (web, URLs conferidas), fontes.md
  dados/            geradores sintéticos e catálogo próprio (pode reutilizar ../teste_exemplificacao_nfh/dados/)
  codigo/           backbones.py, sinteticos.py, executar.py, selecionar.py, status.py, lanca.sh, LEIAME.md
  resultados/       t1.csv, t2.csv, confirmacao.csv, filas/, logs/, selecao_*.md
  exemplo_sintetico/  gerador, treino, passo_a_passo.json, exemplo_passo_a_passo.tex/.pdf, VALIDACAO.md
  relatorio/        relatorio.tex/.pdf (ou .md) com todas as tabelas
```

O código pode importar `src/` e `teste_exemplificacao_nfh/codigo/` (só leitura). Reutilizar `NFHHead`, `DenseHead`,
`InstanceNorm`, `BackboneHeadModel`, `train_model`, `residual_penalty` (`src/v2_common.py`), `densa_para_nfh`,
`ajuste_estrutural`, `Projetor`, `MascWindows`, `segmentos`, `ingenuas` (`src/v3_common.py`), e `carregar` do catálogo
(`teste_exemplificacao_nfh/codigo/dados.py`), em vez de reescrever.

## 3. Comparação justa (obrigatória)

- Modelo = `InstanceNorm(BackboneHeadModel(Φ, cabeça))`, como na v3. Densa e NFH usam o **mesmo** Φ, a mesma largura, os
  mesmos dados, a mesma divisão (70/15/15 cronológica, padronização pelo treino), L = 96, H = 48, q = 2, a mesma semente,
  o mesmo otimizador (Adam, taxa 1e-3, *weight decay* 1e-5, lote 32, *clip* 1,0), até 60 épocas e paciência 10.
  A cabeça NFH tem os mesmos H(p + 1) parâmetros da densa.
- Variantes por tarefa (conjunto, modo, *backbone*, largura, fração de treino, semente):
  1. `densa`: treinada do zero, λ = 0.
  2. `densa+E`: a densa treinada por mais E = 15 épocas, Adam com taxa 1e-4, sem penalização, escolhendo a época
     (0 a 15) de menor RMSE de validação. É o controle das épocas extras pedido pela banca.
  3. `proj_K10`: projeção da previsão da densa sobre T ⊕ S (sem treino; decomposição pura com o resíduo descartado).
  4. `conv_L{λ}_K{K}`: conversão exata da densa (erro máximo < 1e-5 na previsão, registrado) + ajuste estrutural de E = 15
     épocas com seleção segura (`ajuste_estrutural` de `src/v3_common.py`).
  5. `zero_L{λ}_K{K}`: NFH treinada do zero, com o mesmo protocolo da densa e perda MSE + λ‖R‖²/H.
- Comparador de todas as variantes NFH: a **melhor densa**, que é a `densa+E` na validação (por construção, RMSE de
  validação da `densa+E` ≤ o da `densa`). No teste, a NFH tem de ser comparada com as duas.
- Fração de treino: usar as **últimas** janelas do segmento de treino (as mais próximas da validação). Validação e teste
  não mudam com a fração.
- Métricas gravadas por variante: `val_RMSE`, `val_MAE`, `test_RMSE`, `test_MAE`, cobertura mediana (1 − ε_R) no teste,
  épocas, época escolhida, tempo, parâmetros, número de janelas de treino/validação/teste, erro de conversão. As três
  ingênuas (média de treino, último valor, sazonal 24 h) entram uma vez por conjunto.

## 4. Backbones (mínimo; a pesquisa pode acrescentar até dois)

Todos devolvem **h ∈ ℝ^p** e não têm camada de saída própria. Cada classe documenta, na *docstring*, de onde veio
(artigo e URL) e o que foi simplificado.

| Nome | Φ (resumo) | h |
|---|---|---|
| `Linear` | identidade sobre a janela achatada | p = L·d (densa = modelo linear com RevIN) |
| `DLinear` | decomposição por média móvel (25) em tendência e sazonal da janela | concatenação das duas, p = 2·L·d |
| `MLP`, `LSTM`, `Conv1D+LSTM` | os da v3 (`teste_exemplificacao_nfh/codigo/busca.py`, `construir`) | como na v3 |
| `TCN` | convoluções causais dilatadas com resíduo (campo receptivo ≥ L) | definido pelo implementador |
| `PatchTST` | *patches* (16, passo 8) da janela, codificador Transformer | *flatten* dos *tokens* (como a *FlattenHead*) |
| `TSMixer` | blocos de mistura no tempo e entre variáveis | definido pelo implementador |
| `iTransformer` | uma ficha por variável (janela inteira embutida), codificador sobre as variáveis | ficha do alvo |

Larguras: 1 (base, tamanho pequeno comparável à v3) e 2 (dobro das dimensões internas, igual nas duas cabeças).

## 5. Conjuntos

- **Sintéticos (foco):**
  - `SINT-U48`: univariado, horário, H = 48, ciclos de 24 h e 12 h com amplitude e fase lentamente variáveis, tendência
    suave e ruído. O gerador grava o **oráculo** (a melhor previsão possível dada a janela, ou uma aproximação declarada)
    e seu RMSE, que é o piso. O futuro tem de ser previsível a partir da janela (lição de
    `teste_exemplificacao_nfh/exemplo_sintetico/SUBSTITUIDO.md`).
  - `SINT-M`: o de `src/v3_sintetico_multivariado.py` (alvo univariado, entrada com quatro variáveis, atraso 48).
  - Tamanhos de treino: frações {1,0; 0,25; 0,10} (e, se a pesquisa indicar, um ruído maior).
- **Reais (U; M só se a pesquisa indicar):** ETTh1, ETTh2, Jena, Air Quality, Bike Sharing DC, Seoul Bike, Household Power
  (já em `teste_exemplificacao_nfh/dados/`), mais até três novos da pesquisa (URL pública, licença, agregação horária,
  ciclo diário medido só no treino). Frações {1,0; 0,25}.

## 6. Etapas (seleção só pela validação)

- **T1 (triagem ampla):** todos os *backbones*, largura 1, todos os conjuntos, frações do §5, semente 0. Variantes:
  `densa`, `densa+E`, `proj_K10`, `conv_L1000_K10`, `zero_L100_K10`, `zero_L1000_K10` (e `zero_L1000_K4` nos sintéticos).
- **Seleção T1 → T2 (`selecionar.py`, que só lê colunas `val_*`):** para cada combinação (conjunto, modo, *backbone*,
  fração), a variante NFH candidata é a de menor RMSE de validação relativo à `densa+E`. A combinação passa se esse
  relativo for < 0 e o MAE de validação relativo for ≤ 0. Vão para T2 as 16 melhores pelo RMSE relativo, com no máximo 3
  por conjunto e 4 por *backbone*, e com ao menos 2 combinações sintéticas se houver qualificadas.
- **T2 (aprofundamento):** as combinações escolhidas, sementes 0, 1 e 2, larguras 1 e 2, todas as variantes:
  `conv_L{100,1000}_K{10,4}`, `zero_L{0,100,1000}_K10`, `zero_L1000_K4`.
- **Seleção T2 → confirmação:** por combinação e largura, a variante candidata é a de menor média (sobre as sementes) do
  RMSE de validação relativo à `densa+E`. Qualifica-se se essa média for < 0, a média do MAE relativo ≤ 0 e a NFH for
  melhor no RMSE de validação nas três sementes. Até 8 combinações, no máximo 2 por conjunto e 3 por *backbone*,
  incluindo a melhor sintética.
- **Confirmação (teste, sementes novas 10 a 14):** `densa`, `densa+E`, `proj_K10` e a variante escolhida. **Sucesso:**
  média do RMSE de teste e média do MAE de teste da NFH ≤ as da `densa` e as da `densa+E`, e NFH com RMSE de teste menor
  que o da `densa+E` em pelo menos 4 das 5 sementes. Só as combinações com sucesso são citadas como exemplos em que a NFH
  iguala ou supera a densa. As demais são relatadas como empates ou derrotas.
- **Explicabilidade:** para cada confirmada, na semente 10, a decomposição (T, S, R, ε), os Integrated Gradients de g_T e
  g_S e, no SINT-M e no `SINT-U48`, a comparação com a verdade conhecida.

## 7. Recursos

GPU RTX 3070 Laptop (8 GB, cerca de 2,3 GB já ocupados por outro processo), 16 núcleos, 15 GB de RAM. Até 3 processos na
GPU e até 6 em CPU (`CUDA_VISIBLE_DEVICES=""`, `torch.set_num_threads(2)`). Executores retomáveis: uma linha por variante
concluída no CSV; uma tarefa já feita é pulada. Meta de tempo: T1 em até cerca de 4 h de relógio; se a estimativa passar
disso, cortar primeiro a fração 0,25 dos conjuntos reais com mais janelas e registrar no §10.

## 8. Subagente varredor

Um agente exclusivo lê os CSVs, roda `selecionar.py`, compara *backbones* e conjuntos, e grava `resultados/selecao_T1.md`,
`selecao_T2.md` e `selecao_confirmacao.md`, com a fila da etapa seguinte. Ele pode propor uma rodada extra de T2 (mais λ,
larguras ou frações) para as combinações mais promissoras, sempre só pela validação, e registra cada proposta no §10.

## 9. Exemplo sintético passo a passo

Com a combinação vencedora no sintético, um exemplo pequeno (H = 8, q = 1, K = 2, como em
`teste_exemplificacao_nfh/exemplo_livro/` e na Aula 4 do CoffeeAI) em que **todas** as contas aparecem: bases P, F, F̃, Q,
M e D; o cálculo de h e de c = Wh + b linha a linha; θ_T, θ_S e u; T, S, R e a soma; o balanço de energia; os harmônicos;
a comparação com a verdade; a conversão densa ↔ NFH; os Integrated Gradients de g_T e g_S; e uma explicação abdutiva.
O modelo didático (p ≤ 16, para que W caiba numa tabela) tem de ter, ele mesmo, RMSE e MAE de teste ≤ os da densa
na confirmação com sementes novas; se não tiver, não é citado como vitória. Cada número do documento sai de
`passo_a_passo.json`, e um validador independente recalcula ao menos 40 números com código próprio.

## 10. Registro de mudanças do protocolo

- 21/11/2026: versão inicial (coordenador).
- 21/11/2026, 16h10 (coordenador), antes de qualquer execução de T1, com base no relatório do pesquisador
  (`pesquisa/relatorio_backbones.md`) e na revisão do código (`.kiro/specs/rodada-20261121/logs/revisao_codigo.md`):
  1. **Backbone acrescentado:** `FreqMLP` (rfft da janela, bins k = 1..12 em parte real e imaginária, duas camadas com
     GELU, p = 64·largura), como sugerido pelo revisor. `CycleMLP` não entra (soma uma tabela fora da cabeça e quebra a
     decomposição única).
  2. **Corte de T1 (B2):** a estimativa da T1 completa foi 6,3 h (típico) e 20,3 h (pior caso). Em vez do corte do §7
     (insuficiente, 5,35 h) e do corte B (que apagaria a fração 1,0 de todos os reais), usa-se o B2: sintéticos com as
     três frações; reais com fração 0,25 e, com fração 1,0, só os que têm até 12.100 janelas de treino.
  3. **Opção M:** `ETTh1-M` e `opsd_de_solar-M` com fração 0,25, só para iTransformer, TSMixer e PatchTST (o
     iTransformer mistura variáveis e só faz sentido com d > 1).
  4. **T2:** acrescenta `conv_L0_K10` (conversão + ajuste sem penalização) como controle da parametrização.
  5. **Oráculo do SINT-U48:** o texto usa o RMSE do oráculo sobre as janelas de teste avaliadas e o chama de
     "referência amostral", não de piso.
- 21/11/2026, 17h20 (coordenador). **Exemplo passo a passo (§9), pré-registro.** Para que as contas caibam à mão, o
  exemplo não usa os *backbones* grandes da T1: usa a série de livro de `teste_exemplificacao_nfh/exemplo_livro/gerar.py`
  (H = 8, q = 1, K = 2, L = 16, d = 4, alvo y_t = x1_{t−8} + x2_{t−8} + ruído, y fora da entrada, oráculo linear
  conhecido), com ruído σ = 0,3, e um *backbone* de uma camada, h = tanh(W_φ vec(X) + b_φ). Grade de seleção (só
  validação, sementes 0 a 4): p ∈ {8, 16}, janelas de treino n ∈ {300, 1000}; variantes `densa`, `densa+E`,
  `conv_L1000` (conversão + ajuste com seleção segura), `zero_L0`, `zero_L1000`. Escolha: a combinação (p, n, variante
  NFH) de menor média do RMSE de validação relativo à `densa+E`, exigindo média < 0, MAE de validação relativo ≤ 0 e NFH
  melhor na validação nas 5 sementes; empate decidido pelo menor p. Confirmação: sementes 10 a 19, no teste. Sucesso:
  média do RMSE e do MAE de teste ≤ as da `densa` e as da `densa+E`, e NFH melhor que a `densa+E` em ≥ 8 das 10
  sementes. O documento passo a passo usa a semente 10 da combinação confirmada. Se nada se qualificar ou a confirmação
  falhar, o documento é feito assim mesmo e diz, na primeira página, que nesse exemplo a NFH não superou a densa.
  Motivo da escolha da série: a de livro já foi validada (futuro linear na janela e quase todo em T ⊕ S; verdade de
  referência para as atribuições) e, no regime B do teste de 16/11, a NFH do zero com λ = 1000 teve RMSE de teste menor
  que a densa com 300 e 1000 janelas (`teste_exemplificacao_nfh/exemplo_livro/resultados_treino_B.csv`). Decisão tomada
  sem usar a T1 desta pasta. Registro honesto: às 17h17 o coordenador imprimiu, para conferir o formato, as diferenças de
  teste das 36 primeiras linhas NFH da T1 (nenhuma do SINT-M, uma combinação do SINT-U48); daqui em diante, até a
  confirmação, só lê colunas `val_*`. As regras de seleção da T1 e da T2 (§6) não mudam.
- 21/11/2026, 18h15 (coordenador), **por tempo de máquina**, antes de rodar qualquer seleção sobre a T1 e sem ler
  colunas de teste: as tarefas da T1 duram de 12 a 92 min (TCN em CPU), e a ETA da T1 passou a ~9 h. Para que T2 e
  confirmação terminem na manhã de 22/11: (1) um 9.º worker (GPU, `--worker 8`) entrou na T1 às 18h10, com reivindicação
  dinâmica (não muda nenhuma tarefa); (2) **T2 reduzida:** até 12 combinações (não 16), no máximo 3 por conjunto e 3
  por *backbone*, ao menos 2 sintéticas; largura 2 só para as 4 primeiras pelo RMSE de validação relativo (as demais só
  largura 1); variantes sem `conv_L100_K4` (ficam `conv_L{0,100,1000}_K10`, `conv_L1000_K4`, `zero_L{0,100,1000}_K10`,
  `zero_L1000_K4`). A regra de seleção T2 → confirmação e a confirmação (§6) não mudam. (3) A sequência T1 → seleção →
  T2 → seleção → confirmação roda sozinha (`codigo/orquestrar.sh`), sem intervenção humana entre as etapas.
- 21/11/2026, 21h25 (coordenador), **depois da confirmação do exemplo passo a passo**, a pedido da auditoria externa
  (`.kiro/specs/rodada-20261121/avaliadores_externos/20261121b_exemplo_auditor.md`, item 2): controle relatado da
  densa projetada em T ⊕ S, sem treino (`exemplo_sintetico/projetar_confirmacao.py`; ŷ − QQᵀŷ nas sementes 10 a 19;
  confere que a densa retreinada reproduz `confirmacao.csv` com tolerância 1e-5). Saída: `projecao_confirmacao.csv`,
  `protocolo_exemplo.json`; o documento passo a passo foi regenerado às 21h33. Não participa da seleção e não muda o
  veredito pré-registrado. Registrado em 22/11 18h35, porque o desligamento da máquina às 21h33 interrompeu o registro;
  validação independente em `.kiro/specs/rodada-20261121/logs/exemplo_projecao_validacao.md`.
- 22/11/2026, 18h28 (coordenador), **retomada depois do desligamento da máquina (21/11 21h33)**. Nada muda no protocolo:
  mesmas filas (`filas/t2.json`, md5 22ac3027…, gerada pela seleção automática de 21/11 21h14), mesmas regras de seleção e
  de confirmação, mesmo código de treino. Só muda a alocação: o controle das épocas
  (`experimento_controle_epocas/`) tem prioridade na GPU (3 workers GPU + 4 CPU); a T2 recomeça com 1 worker GPU e
  ganha workers quando o controle libera a máquina (`.kiro/specs/rodada-20261121/operacao/orquestrar_retomada.sh`, que
  depois roda `selecionar.py t2`, a confirmação e `pos_confirmacao.sh`, como o `orquestrar.sh`). Um vigia
  (`operacao/vigia_recursos.py`) pausa workers com SIGSTOP quando a GPU passa de 83 °C, a memória da GPU passa de
  7400 MiB ou a RAM disponível cai abaixo de 1536 MiB, e os retoma depois; a pausa não muda nenhum resultado, só o
  `tempo_s` gravado. As 9 tarefas parciais da T2 são refeitas a partir das variantes que faltam, com o checkpoint da
  densa salvo (`executar.py` só apaga o checkpoint quando nada falhou). md5 antes da retomada:
  `.kiro/specs/rodada-20261121/checksums/antes_retomada_20261122.md5`.
- 22/11/2026, 23h07 (coordenador), **registro que faltava**, a pedido da auditoria externa
  (`.kiro/specs/rodada-20261121/avaliadores_externos/20261122_backbones_auditor.md`, §6 e §9 item 6). Nada muda no
  protocolo nem em nenhum resultado. O exemplo passo a passo (§9 e o registro de 21/11 17h20) **não usa** os
  hiperparâmetros do §3. Valores de `exemplo_sintetico/protocolo_exemplo.json` (vêm de
  `teste_exemplificacao_nfh/exemplo_livro/treinar.py`, reusado por `exemplo_sintetico/comum.py`): até 400 épocas
  (`EPOCAS`), paciência 40 (`PACIENCIA`), taxa de aprendizado 3e-3 (`LR`), E = 50 épocas extras (`E_EXTRA`) com taxa de
  ajuste 3e-4 (`LR_AJ`) e lote 64 (`LOTE`); o §3 manda até 60 épocas, paciência 10, taxa 1e-3, E = 15 e lote 32. O
  arquivo `protocolo_exemplo.json` foi gravado em 21/11 às 21h29, depois da confirmação do exemplo
  (`exemplo_sintetico/confirmacao.csv`, 21/11 às 17h29), e por isso este registro não é pré-registro: os valores
  vêm do código reusado; a auditoria não achou sinal de ajuste com o teste, mas essa conclusão depende de `comum.py`, que ela não leu linha a linha. Além disso, o exemplo
  treina com as **primeiras** n janelas do segmento de treino (como no `treinar.py` do livro, dito em
  `exemplo_sintetico/LEIAME.md`), e o §3 manda usar as **últimas** para a fração de treino da T1, da T2 e da confirmação.
  Consequência: o exemplo não é comparável em protocolo com as combinações de *backbones*; o relatório já o trata à parte.
- 23/11/2026, 09h10 (coordenador), **depois de todos os resultados**, ao preparar o repositório público: o rótulo de
  licença do OPSD em `dados/baixar_dados.py` (l. 212) e `dados/catalogo.json` dizia "CC BY 4.0", o que a página do pacote
  não sustenta (a licença dos dados não é declarada; ver `DATA.md` do repositório público, seção OPSD). O rótulo foi
  trocado por "licença dos dados não declarada pelo OPSD". É só metadado: nenhum dado, seleção ou número muda. Cópias
  anteriores: `.kiro/specs/rodada-20261121/backup_publicacao_20261123/`.
