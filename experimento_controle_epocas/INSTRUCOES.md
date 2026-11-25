# Controle das épocas extras nos 24 pares da dissertação: protocolo pré-registrado

Raiz: `.`. Pasta: `experimento_controle_epocas/`. Escrito em
21/11/2026, 20h25, antes de qualquer execução. Mudanças posteriores entram no §8 com data e motivo.

## 1. Por que

A dissertação (v7, §4.3.9, §5.5, §6.4) e o artigo da ESANN comparam a NFH (densa treinada + conversão + até E = 15
épocas de ajuste estrutural) com a densa sem as épocas extras. A banca, a avaliação externa (`.kiro/specs/rodada-20261121/
avaliadores_externos/`) e a própria v7 apontam que o efeito do ajuste não foi separado do efeito das épocas. O §6.4 da v7
promete, como primeiro passo até a defesa, treinar um controle denso pelas mesmas 15 épocas. Este experimento é esse passo.

## 2. Desenho

- Os mesmos 24 pares da dissertação: 8 conjuntos (`ETTh1`, `ETTh2`, `Jena`, `AirQ`, em U e M) × 3 *backbones* (MLP, LSTM,
  Conv1D+LSTM), sementes 0 a 4, com o código da v3 (`src/v3_common.py`, `src/v3_benchmark.py`, `src/v2_common.py`),
  importado e nunca editado: `carregar`, `segmentos`, `construir`, `train_model`, `densa_para_nfh`, `ajuste_estrutural`
  (λ = 1000, taxa 1e-4, E = 15, seleção segura), `CFG3`, K = 10.
- Por (par, semente), no **mesmo dispositivo** que a dissertação usou (`results/v3/dispositivos.json`, `detalhe`):
  1. `densa`: retreinada do zero, como em `v3_benchmark.rodar`. Grava RMSE/MAE de validação e teste e a diferença para a
     linha da dissertação (`results/v3/runs/<conjunto>.csv`, `modelo == "<bb> + densa"`, `variante` vazia, mesma
     semente): é a medida de reprodução.
  2. `densa+E`: a partir dessa densa, E = 15 épocas com Adam, taxa 1e-4, o *weight decay*, o lote, o *clip* e o gerador
     de embaralhamento de `ajuste_estrutural`, perda MSE sem penalização; escolhe a época 0..15 de menor RMSE de
     validação (a época 0 é a própria densa).
  3. `nfh`: `densa_para_nfh(densa, K=10)` + `ajuste_estrutural` com os padrões da v3 (a variante "seguro" da
     dissertação). Grava o erro de conversão, a época escolhida e a diferença para a linha da dissertação
     (`modelo == "<bb> + NFH"`, `variante == "seguro"`, `K == 10`).
- Semente 0, comparação exata adicional (`exata_s0`): carrega `results/v3/modelos/<conj>__<bb>+densa__s0.pt` (a densa
  da dissertação) e roda o mesmo `densa+E`; compara com a NFH da dissertação na semente 0 (linha do CSV). Não retreina nada.
- Nada é escolhido pelo teste. As únicas escolhas (época da densa+E, época do ajuste) usam a validação.

## 3. Análise (fixada agora, a mesma de `src/v3_tabelas.py`)

Para cada par, a média do RMSE de teste nas 5 sementes. `rel = RMSE_NFH / RMSE_controle − 1` nos 24 pares, com controle ∈
{densa+E, densa}. Relatar: vitórias da NFH (rel < 0), média e mediana de rel, Wilcoxon bilateral e o teste de não
inferioridade com margem de 1% (`wilcoxon(rel − 0.01, alternative="less")`), como em `v3_tabelas.py`. O mesmo para o MAE.
Relatar também a época escolhida pela densa+E (quantas ficaram na época 0) e pela NFH, a reprodução (diferença da densa e
da NFH retreinadas para a dissertação) e a comparação `exata_s0`. Todos os resultados vão para o relatório, qualquer que
seja a direção. A frase da dissertação e do artigo será a que os números sustentarem.

## 4. Recursos e ordem

Começa com 1 *worker* GPU e 1 CPU enquanto o experimento de *backbones* roda; quando ele terminar
(`experimento_backbones_nfh/resultados/logs/ORQUESTRADOR_FIM`), sobe para até 3 GPU + 5 CPU. Retomável (uma linha por
variante; pula o que existe), reivindicação dinâmica, validação em blocos de 1024 se precisar de memória (sem mudar o
resultado além da 6.ª casa).

## 5. Saídas

`resultados/controle.csv` (uma linha por par, semente e variante), `resultados/exata_s0.csv`, `resultados/analise.json`,
`relatorio/relatorio_controle.md` (e `.pdf`), `codigo/`, `LEIAME.md`.

## 6. Proibições

Não editar `src/`, `results/`, as versões da tese, `artigo_esann2027/paper.tex` nem o experimento de *backbones*. Não
rodar `git`. Nenhum número digitado à mão no relatório.

## 7. Uso dos resultados

Os números entram (a) na nova versão do artigo da ESANN e (b) numa proposta de texto para a dissertação, ambos com
rastreabilidade para `resultados/analise.json`.

## 8. Registro de mudanças

- 21/11/2026, 20h25: versão inicial (coordenador).
- 22/11/2026, 18h28 (coordenador): retomada depois do desligamento da máquina (21/11 21h33; 4 de 144 tarefas prontas).
  Nada muda na análise, nas variantes nem no código de treino. Só a ordem de recursos do §4: o controle passa a ter
  prioridade (3 GPU + 4 CPU desde a retomada, sem esperar o fim dos *backbones*), porque é o resultado que o artigo da
  ESANN precisa até 25/11. Um vigia (`.kiro/specs/rodada-20261121/operacao/vigia_recursos.py`) pausa workers com SIGSTOP
  acima de 83 °C na GPU, 7400 MiB de memória da GPU ou abaixo de 1536 MiB de RAM livre, e os retoma depois; a pausa não
  muda nenhum resultado, só o `tempo_s`. As reivindicações de workers mortos voltaram à fila pelo próprio `executar.py`.
  md5 antes da retomada: `.kiro/specs/rodada-20261121/checksums/antes_retomada_20261122.md5`.
