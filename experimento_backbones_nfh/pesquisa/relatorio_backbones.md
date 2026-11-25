# Relatório de pesquisa: backbones e conjuntos de dados para a NFH

Agente: pesquisa_backbones. Data: 21/11/2026. Todas as páginas e arquivos abertos estão em `fontes.md`.
Regra seguida: número vem de arquivo aberto (indicado), fato externo vem de URL aberta. Quando só li o resumo de um
artigo ou só o código oficial, o texto diz isso. Nada aqui afirma que a NFH vença em algum backbone: isso é o que o
experimento deve medir.

## 0. Resumo

1. Os backbones em que a NFH substitui uma camada final `Linear(p, H)` sem mexer no resto são: Linear, DLinear,
   TSMixer, iTransformer, PatchTST e TCN. Em TimesNet e TimeMixer a saída é multi-escala ou por passo de tempo, e não há
   um único `Linear(p, H)` para trocar.
2. Evidência direta de que restringir a saída ajuda: (a) N-HiTS, comparando o mesmo modelo com e sem interpolação na
   saída, melhora de 0,548 para 0,498 de MSE médio em H = 720, e empata em H = 96 (§2.2); (b) CycleNet, em que o
   termo periódico explícito melhora o Linear em 17,1% a 28,6% de MSE em Electricity e em 25,6% em Traffic H = 96 (§2.3).
   Evidência contrária: o N-BEATS interpretável (tendência polinomial + senoides, sem resíduo) ficou pior que o genérico
   em horizonte longo (MSE médio 0,794 contra 0,548 em H = 720 em cinco conjuntos). A NFH mantém o resíduo R com
   penalização λ, não o proíbe, então essa derrota não se transfere automaticamente.
3. Acrescentar ao §4 do protocolo no máximo dois backbones: `FreqMLP` (estilo FITS: entrada = espectro de baixa
   frequência da janela) e `CycleMLP` (estilo CycleNet: tabela de ciclo de 24 h + MLP). Desenho exato em §3.
4. SINT-U48: fórmula em §4. Já implementada em `codigo/sinteticos.py` por outro agente; conferi por script próprio que o
   sinal futuro está 99,9995% em T ⊕ S com K ≥ 4 (§4.3).
5. Três conjuntos reais novos baixados e padronizados (§5): `melbourne_pedestrian_southbank`, `electricity_mt156`,
   `opsd_de_solar`. O ciclo diário do `electricity_mt156` é o mais fraco dos três em R² do perfil (0,273).

## 1. A camada final de cada backbone no código oficial e como obter h ∈ ℝ^p

Notação: L = 96, H = 48. "Cabeça" é a camada que a NFH substitui. Em todos os casos a NFH tem os mesmos H(p+1)
parâmetros da camada que troca. Leitura: código oficial aberto em 21/11/2026 (URLs em `fontes.md`), salvo onde dito.

| Backbone | Cabeça no código oficial | h para a NFH | p | Custo da cabeça (H(p+1)) |
|---|---|---|---|---|
| Linear (Zeng et al. 2023) | `nn.Linear(seq_len, pred_len)` (`DLinear.py`, `individual=False`; não abri o arquivo do modelo `Linear` do mesmo repositório) | janela achatada (já normalizada) | 96 | 48·97 = 4 656 |
| DLinear | `Linear_Seasonal` e `Linear_Trend`, cada um `Linear(seq_len, pred_len)`, saídas somadas; decomposição por média móvel de 25 (`series_decomp(kernel_size=25)`) | concatenar [sazonal, tendência] (2L). A soma de duas `Linear(L,H)` é uma `Linear(2L,H)` sobre a concatenação, só com bias somados; isso é aritmética minha, não está no artigo | 192 | 48·193 = 9 264 |
| TSMixer (Chen et al. 2023) | `Dense(pred_len)` no eixo do tempo, após os blocos de mistura (`tsmixer.py`: transpõe para `[B, C, L]`, `Dense(pred_len)`, transpõe de volta) | saída dos blocos, canal do alvo, eixo do tempo: vetor de L | 96 | 4 656 |
| iTransformer (Liu et al. 2024) | `self.projector = nn.Linear(d_model, pred_len)` aplicado a cada ficha de variável | ficha do alvo após o codificador. Com entrada univariada há uma só ficha: a atenção age sobre um único token (só aplica a projeção de valor) e o modelo é, na prática, um MLP (observação minha a partir do código) | d_model (ex. 64) | 48·65 = 3 120 |
| PatchTST (Nie et al. 2023) | `Flatten_Head`: `Flatten(start_dim=-2)` + `Linear(nf, target_window)`, `nf = d_model * patch_num` (`PatchTST_backbone.py`) | *flatten* dos tokens `[d_model × patch_num]` por canal. Com patch 16, passo 8, L = 96: `patch_num = (96−16)/8 + 1 = 11`, ou 12 com `padding_patch='end'` (o código soma 1) | d_model·12 (ex. 16·12 = 192) | 48·193 = 9 264 |
| TCN (Bai et al. 2018) | `self.linear = nn.Linear(num_channels[-1], output_size)` sobre `y1[:, :, -1]` (último instante) (`adding_problem/model.py`) | saída da TCN no último instante. Campo receptivo: cada bloco tem duas convoluções causais de kernel k e dilatação 2^i, somando 2(k−1)2^i; com k = 3 e 5 blocos o campo receptivo é 1 + 2·2·31 = 125 ≥ 96 (conta minha) | num_channels[-1] (ex. 64) | 3 120 |
| TimesNet | `predict_linear = Linear(seq_len, pred_len+seq_len)` no eixo do tempo antes dos blocos e `projection = Linear(d_model, c_out)` por instante | sem `Linear(p, H)` único: a saída é por passo de tempo. Não recomendado | n/d | n/d |
| TimeMixer | uma `predict_layers[i]` (`Linear(L_i, H)`) por escala, mais `projection_layer(d_model → c_out)`, somando as escalas | uma NFH por escala mudaria o modelo. Não recomendado | n/d | n/d |
| N-BEATS / N-HiTS | cada bloco: pilha FC → `Linear(layer_size, theta_size)` → função de base (`GenericBasis`, `TrendBasis`, `SeasonalityBasis`; no N-HiTS, `_IdentityBasis` com `F.interpolate`) | a NFH é o análogo da base do bloco interpretável; usar como backbone exigiria uma NFH por bloco | n/d | n/d |
| FITS (Xu et al. 2023) | `nn.Linear(cut_freq, int(cut_freq·η)).to(torch.cfloat)` sobre o espectro `rfft` truncado, depois `irfft` | espectro de baixa frequência (parte real e imaginária) de uma camada anterior: ver `FreqMLP` em §3 | 2·COF | depende |
| SparseTSF | `Conv1d` + reshape por período `w` + `Linear(seg_num_x, seg_num_y)` compartilhada entre fases | a camada final atua no eixo dos segmentos, não num vetor: não há `Linear(p, H)` a trocar sem reescrever o modelo | n/d | n/d |
| CycleNet | `RecurrentCycle` (tabela `cycle_len × canais`) subtraída da entrada e somada à saída, mais `Linear(seq_len, pred_len)` ou MLP de duas camadas | saída do MLP (largura `d_model`); a tabela de ciclo fica de fora da NFH: ver `CycleMLP` em §3 | d_model | 48·(d_model+1) |

Custo de cálculo. A NFH multiplica `c = Wh + b` e depois aplica uma base fixa `H × (q+1+2K)`; o custo extra sobre a
densa é uma multiplicação `H × (q+1+2K)` por janela, desprezível. O custo dominante é o do backbone: PatchTST tem
mais de 1 milhão de parâmetros e o DLinear, 139,7 mil, segundo a tabela de parâmetros do artigo do FITS (lido); o
CycleNet/MLP tem 472,9 mil e o CycleNet/Linear 123,7 mil, e o iTransformer 5,15 milhões, segundo a Tabela 3 do
CycleNet (lido). Os números dependem da configuração de cada artigo e não são da nossa.

## 2. Evidência sobre restringir a saída

Só entram aqui trechos que li na fonte. "Condição" é o que o próprio texto diz ou o que a tabela mostra; onde não há
condição estudada, escrevi isso.

### 2.1 N-BEATS interpretável contra genérico (Oreshkin et al., arXiv 1905.10437)
- Tabela 1 (M4, 100 000 séries; sMAPE, OWA): N-BEATS-G 11,168 e 0,797; N-BEATS-I 11,174 e 0,798; ensemble 11,135 e
  0,795. No M3 (sMAPE): G 12,47, I 12,43. No turismo (MAPE): G 18,47, I 18,97. A base polinomial + harmônicos não
  piorou a acurácia média no M4 e no M3, e piorou no turismo (+0,50).
- Código oficial (`ServiceNow/N-BEATS`): `TrendBasis` usa potências de t/H; `SeasonalityBasis` usa cos e sen de
  uma grade de frequências. Não há resíduo ortogonal nem penalização: a base é obrigatória.
- Texto do artigo, seção de ensemble: "ensembling is a much more powerful regularization technique than the popular
  alternatives, e.g. dropout or L2-norm penalty"; dropout e L2 melhoraram os modelos individuais mas pioraram o
  ensemble. É sobre ensemble, não sobre a base. A penalidade λ‖R‖² da NFH é de outra natureza (penaliza a parte fora
  da base); o experimento compara λ = 0, 100 e 1000.

### 2.2 N-HiTS (Challu et al., arXiv 2201.12886, AAAI 2023)
- O artigo diz que a interpolação hierárquica "enforce[s] smoothness of the multi-step predictions by reducing the
  dimensionality of neural network's prediction". Isso é restringir a saída a um subespaço de baixa dimensão, como a NFH.
- Tabela 2 (média de oito execuções e cinco conjuntos: ETTm2, Exchange, ECL, TrafficL, Weather, segundo o Apêndice G),
  MSE médio por horizonte (96, 192, 336, 720):
  N-HiTS completo 0,195 / 0,250 / 0,315 / 0,484; só interpolação (N-HiTS₂) 0,196 / 0,261 / 0,315 / 0,498;
  sem interpolação nem multi-taxa (N-HiTS₄, equivale ao N-BEATS genérico) 0,196 / 0,263 / 0,346 / 0,548;
  N-BEATS interpretável (N-BEATSᵢ) 0,209 / 0,266 / 0,408 / 0,794.
  Leitura: restringir a saída por interpolação reduz o MSE em horizonte longo (720: 0,548 → 0,498) e empata em H = 96.
  O N-BEATS interpretável, com base polinomial + senoidal fixa, foi o pior, e o texto do artigo o chama de "detrimental
  inductive bias for long-horizon forecasting". Isso vale para esses cinco conjuntos, que não têm ciclo diário estável
  em todos; não vale como regra.
- Lição para a NFH: base fixa sem folga (N-BEATSᵢ) pode custar caro; base de baixa dimensão com folga (interpolação em
  nós, ou T + S + R com λ) pode ajudar. A NFH está no segundo caso, mas isso é hipótese a testar.

### 2.3 CycleNet (Lin et al., arXiv 2409.18479)
- Termo periódico explícito (tabela aprendível de `cycle_len` posições, o "RCF") subtraído da entrada e somado à
  saída. Tabela 4 (ablação, Electricity e Traffic, L = 96): Linear 0,197 / 0,197 / 0,212 / 0,253 de MSE em Electricity,
  com RCF 0,141 / 0,155 / 0,172 / 0,210 (melhora de 28,6% a 17,1%); em Traffic, de 0,645 para 0,480 em H = 96 (25,6%).
  Melhoria de MSE no Electricity: 28,6%, 21,4%, 18,8% e 17,1% (H = 96…720); de MAE: 14,6%, 10,8%, 9,5% e 8,7%. A linha do MLP não foi transcrita.
- Condição, nas palavras do artigo: o comprimento do ciclo "should be set to the maximum stable cycle within the
  dataset" e os conjuntos têm "prominent, explicit cycles". Ou seja, sazonalidade estável.
- Custo: a tabela de ciclo tem W × D parâmetros e nenhuma operação de multiplicação adicional (texto do artigo).

### 2.4 FITS (Xu et al., arXiv 2307.03756) e a crítica de Eefsen et al. (arXiv 2410.18318)
- FITS: filtro passa-baixa; "the high-frequency components filtered out by the LPF typically comprise noise" (afirmação dos
  autores, ilustrada com um trecho do OT do ETTh1). Parâmetros: 4,5 mil a 16 mil, contra 139,7 mil do DLinear (tabela do
  artigo). A frequência de corte é escolhida por um harmônico do período dominante.
- Eefsen et al.: FITS "struggles with datasets that exhibit random or non-periodic behavior such as stock prices, where
  simpler models or naive baselines perform better", e quando o período base é longo em relação à janela o modelo aprende
  frequências mais altas. Os autores construíram um gerador sintético com tendência e ruído ajustáveis, usado para
  interpretar o FITS, não para o benchmark oficial. É o paralelo mais próximo do SINT-U48 que encontrei em previsão.

### 2.5 SparseTSF (Lin et al., arXiv 2405.00946)
- A técnica "Sparse" (reshape por período e `Linear` entre segmentos) é combinada com Linear, Transformer e GRU: melhoras
  médias de 4,7% (Linear), 21,4% (Transformer) e 12,4% (GRU), segundo o texto da seção de ablação.
- Limitações declaradas pelos autores: pior em períodos muito longos (> 100) e com vários períodos entrelaçados, embora
  funcione em "daily periods superimposed with weekly periods". Eles afirmam que a técnica é útil em "small samples and
  low-quality data" (afirmação do texto da seção de generalização; li o parágrafo, não reproduzi os números).

### 2.6 Outros
- DLinear/LTSF-Linear (Zeng et al., arXiv 2205.13504): li o resumo. Mostra que uma camada linear supera Transformers
  em nove conjuntos; não estuda restrição de base nem poucos dados. É o ponto de partida natural: a NFH sobre Linear é a
  comparação mais limpa (densa = modelo linear com RevIN).
- TSMixer (arXiv 2303.06053), iTransformer (2310.06625), PatchTST (2211.14730), TCN (1803.01271), TimesNet,
  TimeMixer (2405.14616): li resumo e/ou código da cabeça. Não encontrei neles comparação com saída restrita a uma base.
  Não afirmo nada sobre ganho com base fixa nesses modelos.
- Low-rank: a busca devolveu "Low Rank Forecasting" (Barratt, Angeris, Boyd, arXiv 2101.12414), cujo resumo descreve
  previsores de posto baixo: um estado latente estimado do passado e a previsão a partir dele. É o mesmo desenho
  "h → c → saída" com `c` de dimensão menor que H, mas não é com base fixa nem tem acurácia comparada aqui. Li só o
  resumo.
- Não encontrei um artigo que compare, com tudo igual, "camada densa final contra camada com base fixa mais resíduo
  penalizado". A NFH é, portanto, o teste dessa hipótese, e as evidências acima são indiretas.

### 2.7 O que a evidência sugere sobre as condições (hipóteses, não resultados)
- Sazonalidade estável e período curto relativo à janela: CycleNet e SparseTSF dizem isso explicitamente; Eefsen e
  SparseTSF dizem que a técnica falha com aleatoriedade e períodos longos. O SINT-U48 (24 h e 12 h, L = 96) está na
  região favorável; ETTh1 e Household Power, não.
- Horizonte longo: o ganho da interpolação do N-HiTS cresce com o horizonte (0% em 96, 9,1% em 720, de 0,548 para 0,498).
  Nosso H = 48 é intermediário.
- Poucos dados e ruído: SparseTSF afirma; FITS afirma sobre ruído de alta frequência; nenhum dos dois, na parte que
  li, mostra uma curva de acurácia contra o tamanho do treino. O experimento (frações 1,0, 0,25 e 0,10) fará isso.

## 3. Ranking e recomendação

### 3.1 Ranking de compatibilidade com uma cabeça de base fixa (apenas os que têm `Linear(p, H)` trocável)

A ordem combina três critérios: a troca é exata (a camada final é um `Linear(p, H)`), p é pequeno (menos
parâmetros para estimar com poucos dados) e há alguma evidência de que a restrição espectral ou periódica ajuda.

1. **Linear/DLinear.** Troca exata, p = 96 ou 192, e é a comparação mais limpa. Evidência indireta: o ganho do CycleNet
   sobre o Linear (§2.3) sugere que há estrutura periódica não usada pela camada linear.
2. **Backbone espectral (FITS-like, `FreqMLP`).** p pequeno por construção, entrada já organizada em frequências que
   coincidem com as da base da NFH (24 h e 12 h são k = 2 e 4 de H = 48). Evidência: parâmetros baixos e desempenho
   comparável (artigo do FITS), com ressalva de Eefsen para séries não periódicas.
3. **TSMixer.** Troca exata sobre o eixo do tempo, p = L. O próprio modelo já é "linear no tempo + mistura".
4. **iTransformer.** Troca exata (`projector`), p = d_model. Em modo U vira MLP; só tem sentido nos conjuntos com
   covariáveis (modo M).
5. **TCN.** Troca exata, p = canais finais; p pequeno. Sem evidência específica.
6. **PatchTST.** Troca exata (`Flatten_Head`), mas p = d_model·patch_num é grande (192 com d_model = 16), então a NFH
   troca 9 264 parâmetros; é o caso em que a restrição mais reduz o espaço de saída em relação ao tamanho de W.
7. **MLP, LSTM, Conv1D+LSTM** do §4: já no protocolo.
8. TimesNet, TimeMixer, SparseTSF, N-BEATS/N-HiTS: sem `Linear(p, H)` único; não usar como backbone. Servem de evidência.

### 3.2 Recomendação: acrescentar no máximo 2 backbones ao §4

**(A) `FreqMLP`** (inspirado no FITS; `forward` devolve h, sem camada de saída).
- Entrada: janela já normalizada pelo `InstanceNorm` (L = 96 por dimensão).
- `S = torch.fft.rfft(x, dim=1)` → bins k = 0…48, o bin k corresponde ao período 96/k h. Manter k = 1…12 (períodos
  96, 48, 32, 24, 19,2, 16, 13,7, 12, 10,7, 9,6, 8,7, 8 h): inclui 24 h (k = 4) e 12 h (k = 8). Descartar k = 0 porque
  a média já foi removida pelo InstanceNorm (conferir no `InstanceNorm` de `src/v2_common.py` antes de implementar; se
  não removeu, manter o bin 0).
- Vetor de entrada: `[Re(S_1..12), Im(S_1..12)]`, 24 números por dimensão (24·d no modo M).
- Φ: `Linear(24d, w)` → GELU → `Linear(w, w)` → GELU; h = saída, **p = w** com w = 64 (largura 1) e 128 (largura 2).
- Custo de Φ: ~24·64 + 64·64 = 5 632 parâmetros (largura 1, d = 1) mais a FFT; a cabeça tem H(p+1) = 48·65 = 3 120.
- Por que: reduz a entrada às frequências que a base da NFH pode escrever, então a NFH não precisa "descobrir" o
  que descartar. Risco conhecido (Eefsen): sem periodicidade o modelo perde para os ingênuos.

**(B) `CycleMLP`** (inspirado no CycleNet/MLP; Lin et al., 2409.18479).
- Parâmetro extra: tabela `C ∈ ℝ^{24×d}` inicializada em zero. Índice de fase da janela: `φ = hora_do_dia(início da janela)`,
  disponível em `datas` do carregador.
- `x' = x − C[(φ + arange(L)) mod 24]`; h = `MLP(x')`: `Linear(L·d, w)` → GELU → `Linear(w, w)` → GELU; **p = w** (64 e 128).
- Previsão: `ŷ = cabeça(h) + C[(φ + L + arange(H)) mod 24]` no canal do alvo. Como `L = 96` e `H = 48` são
  múltiplos de 24, `(φ + L + j) mod 24 = (φ + j) mod 24`.
- A NFH explica `cabeça(h)` (T, S, R); a tabela C entra como um quarto termo nomeado no relatório de explicabilidade.
  Isso quebra a decomposição única da previsão inteira, e deve ser dito. Por isso é a segunda escolha e a densa
  correspondente usa a mesma tabela C (comparação justa).
- Custo: 24·d parâmetros extras, sem multiplicações (artigo do CycleNet).
- Por que: é o backbone com a melhor evidência de ganho em séries de ciclo estável (§2.3). Se a NFH ganhar aqui, o
  ganho vem da cabeça sobre o resíduo do ciclo, e isso é testável.

Alternativas descartadas: CycleNet/Linear (já coberto pelo `Linear` + C), SparseTSF (sem cabeça trocável), N-HiTS
(precisaria de uma NFH por bloco).

## 4. SINT-U48

### 4.1 Fórmula (a implementada em `experimento_backbones_nfh/codigo/sinteticos.py`, lida em 21/11/2026)

Para t = 0…N−1 (N = 12 000 h, semente 2026):

s_t = tend_t + A1_t · sin(2π t/24 + φ1_t) + A2_t · sin(2π t/12 + φ2_t),  y_t = s_t + e_t,  e_t ~ N(0, σ²), σ = 0,30

- tend_t = 0,8 · sin(2π t/2600 + ψ0) + 0,4 · u_t, com u_t = AR(1) com φ = 0,999 e inovação 0,05, suavizado por média móvel de 24 h
  e padronizado a desvio 1;
- A1_t = 1,0 + 0,25 · sin(2π t/1300 + ψ1);  A2_t = 0,5 + 0,15 · sin(2π t/1700 + ψ2);
- φ1_t = φ10 + 0,4 · sin(2π t/2100 + ψ3);  φ2_t = φ20 + 0,4 · sin(2π t/1900 + ψ4);
- ψ0…ψ4, φ10, φ20 ~ U(0, 2π).

Os ciclos de 24 h e 12 h são os harmônicos k = 2 e k = 4 de H = 48. Janela L = 96 (4 ciclos de 24 h), divisão 70/15/15.
Adotei esta fórmula, sem propor um segundo gerador, e conferi suas propriedades por script próprio (§4.3).

### 4.2 Oráculo e limite inferior (fonte: `dados/sint_u48_oraculo.json` e `pesquisa/verificacao_sint_u48.txt`)
- Oráculo: prever s_{t+h} (sinal sem ruído do futuro). Erro = ruído. RMSE de teste padronizado = 0,2658
  (`RMSE_oraculo_padronizado_teste` = 0,26583575925326725 no JSON; σ/sd_treino = 0,2675 pelo meu script).
- É piso: nenhum previsor que só vê a janela o alcança, porque a janela é ruidosa e a deriva futura é desconhecida.
- Limite superior de referência (ridge linear sobre a janela, ajustado no treino, do JSON): 0,3013.

### 4.3 O futuro está em T ⊕ S (conferência minha, `pesquisa/verificacao_sint_u48.py`, saída em `.txt`)
- 1 − ε_R do sinal futuro de 48 h na base q = 2: K = 2 → 0,834332; **K = 4 → 0,999995**; K = 10 → 1,000000.
  Com K ≥ 4 (inclui 24 h e 12 h) o sinal cabe praticamente todo em T ⊕ S; com K = 2 (só 48 h e 24 h) ficam de fora os
  12 h (16,6% da energia).
- RMSE padronizado de teste de um previsor perfeito do sinal projetado em T ⊕ S (R = 0), contra o y futuro ruidoso:
  K = 2 → 0,3982; K = 4 → 0,2663; K = 10 → 0,2663 (piso 0,2658). Projetar custa 0,0005 com K ≥ 4.
  Portanto K = 4 é o menor K que não paga acurácia; K = 2 paga muito. Sugiro rodar `zero_L1000_K4` e `K10`
  (o protocolo já prevê) e esperar que K = 2 perca.
- Observação: o ruído branco do futuro não pertence a T ⊕ S, então R ≠ 0 nos dados. Em λ grande a NFH empurra R
  para zero, o que é o desejado: R vira estimativa do ruído.

### 4.4 Por que a NFH deveria ganhar com poucos dados (hipótese + checagem linear)
- Argumento: a cabeça densa estima H(p+1) = 48·97 = 4 656 parâmetros por janela de 96; a saída verdadeira vive num
  subespaço de dimensão 11 (q = 2, K = 4). Na regressão linear, o excesso de variância cai com o número de dimensões
  da saída que precisam ser aprendidas: 11 de 48. A penalização de R faz isso de forma macia na NFH.
  É um argumento heurístico para o modelo linear; não vale como prova para a rede com não linearidade.
- Checagem numérica **com mínimos quadrados lineares, não com a NFH** (fonte: `verificacao_sint_u48.txt`, RMSE padronizado de
  teste, piso 0,2658): K = 4: 100% do treino (8 257 janelas) denso 0,3022 e saída em T ⊕ S 0,3003; 25% (2 064)
  0,3097 e 0,3034; 10% (825) 0,3193 e 0,3068. A diferença cresce quando o treino diminui (−0,6%, −2,0%, −3,9%).
  Com K = 10 a diferença é menor (−0,5%, −1,3%, −2,0%). Isto motiva a hipótese; não a confirma, porque a NFH
  usa λ macio, rede e Adam. Concorda com a observação anterior do exemplo de livro (NFH com λ = 1000 ganha com
  300 janelas e empata com 3 000; `exemplo_livro/resultados_treino_B.csv`, citado em `INSTRUCOES_RODADA.md` §2).
- Ruído maior: o protocolo permite uma variante. Sugestão: `SINT-U48-R` com σ = 0,60 (mesma semente e mesmo sinal);
  o oráculo e o RMSE de referência têm de ser recalculados com o gerador (o desvio do treino também muda). Ainda não gerado.

### 4.5 Geradores sintéticos na literatura de interpretabilidade (o que li)
- xaitimesynth (arXiv 2603.06781, resumo): gera séries "background signal" + "localized, class-discriminating feature" com
  máscara de verdade; é para classificação, não previsão.
- "Why Do Class-Dependent Evaluation Effects Occur with Time Series Feature Attributions?" (arXiv 2506.11790, resumo):
  séries sintéticas com localização conhecida das características, classificação.
- Mamalakis et al. (arXiv 2103.10005, resumo): benchmark sintético com atribuição teórica conhecida para geociências
  (campos, não séries horárias).
- N-HiTS (2201.12886): sinais harmônicos latentes sintéticos para mostrar a decomposição por bloco (Fig. 1, citada no texto
  da seção de discussão; não abri a figura).
- Eefsen et al. (2410.18318): gerador com tendência, ruído e outros fatores ajustáveis para estudar o FITS.
- Nenhum deles fornece uma verdade espectral (T, S, R) de previsão multi-step; o SINT-U48 fornece, pois o sinal sem
  ruído é conhecido e cai em T ⊕ S.

## 5. Conjuntos reais novos

Script: `experimento_backbones_nfh/dados/baixar_dados.py` (reprodutível; mescla no `catalogo.json` sem apagar entradas
existentes). Formato: coluna `date` ISO 8601, alvo na última coluna, média horária, faltantes vazios. Entradas no
`catalogo.json` com os campos de `teste_exemplificacao_nfh/dados/catalogo.json`, mais `ciclo_diario`, `origem`, `periodo`.
Métricas medidas só nos primeiros 70% da série (treino). Método do R² de perfil, ACF(12 h) e ACF(24 h): o de
`teste_exemplificacao_nfh/dados/baixar_dados.py::compute_daily_cycle_strength` (copiado). Acrescentei três medidas:
`r2_perfil_bruto` e `acf_24h_bruto` (sem remover a média móvel de 24 h, que apaga o ciclo diário em séries como a solar)
e `r2_TS_K10` (R² de mínimos quadrados de janelas de 48 h do treino padronizado em T ⊕ S com q = 2 e K = 10; é o teto de
cobertura 1 − ε_R de uma cabeça NFH, não de previsão).

| Conjunto | n_obs (h) | Faltantes | R² perfil | ACF 12 h | ACF 24 h | R² perfil bruto | ACF 24 h bruto | R² T⊕S (K=10, média / mediana) | Períodos dominantes (h) | Arquivo |
|---|---|---|---|---|---|---|---|---|---|---|
| `melbourne_pedestrian_southbank` | 17 016 | 0,0002 | 0,7515 | −0,7732 | 0,8855 | 0,6757 | 0,8523 | 0,9594 / 0,9637 | 24, 8, 21 | 0,41 MB |
| `electricity_mt156` | 35 065 | 0 | 0,2733 | −0,7421 | 0,7833 | 0,0876 | 0,7457 | 0,5709 / 0,4937 | 24, 168, 84 | 0,98 MB |
| `opsd_de_solar` | 43 824 | 0,0024 | 0,7382 | −0,7078 | 0,9576 | 0,6064 | 0,9423 | 0,9991 / 0,9996 | 24, 12, 8 | 1,54 MB |

Fonte: `experimento_backbones_nfh/dados/catalogo.json` (chave `ciclo_diario`), gerada por `baixar_dados.py` em 21/11/2026.
Comparação com os existentes (`relatorio_pesquisa.md` §3.2): ETTh2 0,661, Bike DC 0,630, Jena 0,574, Air Quality 0,519,
Seoul 0,496, ETTh1 0,318, Household 0,303. Dois dos novos (Melbourne e solar) ficam acima de todos em R² de perfil.

1. **`melbourne_pedestrian_southbank`**. Contagem de pedestres do sensor SouthB_T (`location_id` 35), Pedestrian Counting
   System, City of Melbourne Open Data. URL:
   `https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/pedestrian-counting-system-monthly-counts-per-hour/exports/csv?where=location_id%3D35`.
   Licença: o campo `license` da API devolveu "CC BY"; gravei "CC BY 4.0". A versão exata (4.0) não está no campo, está
   na convenção do portal: **conferir antes de citar numa publicação**. Sem login. Janela fixa 2024-11-22 a 2026-10-31,
   porque a API publica só os últimos ~2 anos (as 8 sensores com mais registros tinham 17 490 a 17 498 linhas, de
   2024-11-22 a 2026-11-21); o CSV gravado fixa o resultado. Hora local de Melbourne (a mudança de horário de verão cria
   horas ausentes, que ficam vazias). Sem covariáveis (as colunas `direction_1`/`direction_2` somam o alvo e vazariam).
   Atenção: série de eventos (feriados, jogos); não é "limpa".
2. **`electricity_mt156`**. Cliente MT_156 do ElectricityLoadDiagrams20112014 (UCI, dataset 321), kW médio por hora
   (15 min agregados por média), 2011-01-01 a 2015-01-01. URL:
   `https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip`. Licença: a página do UCI diz
   Creative Commons Attribution 4.0 International (CC BY 4.0). Sem login. **Regra de escolha declarada antes de medir o
   ciclo:** primeiro cliente, na ordem do arquivo, sem nenhuma hora zero no período inteiro (90 clientes atendem; o
   primeiro é MT_156). Meu primeiro rascunho usou "< 0,1% de zeros" e apontava MT_124; o script final usa zero exato e o
   arquivo `electricity_mt124.csv` foi apagado. O ciclo de 24 h é claro na ACF (0,78), mas o R² de perfil é o mais baixo
   dos três, porque este cliente tem forte padrão semanal (168 h é o 2º período). É um bom caso "ciclo diário
   moderado".
3. **`opsd_de_solar`**. Geração solar real da Alemanha (`DE_solar_generation_actual`, MW), com carga
   (`DE_load_actual_entsoe_transparency`) e geração eólica (`DE_wind_generation_actual`) como covariáveis, 2015-01-01 a
   2019-12-31, horário em UTC. Pacote Open Power System Data, `time_series`, versão 2020-10-06. URL:
   `https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv` (124 MB; recortei
   para 1,54 MB). Licença: a página do pacote diz "Text is licensed under CC-BY license" (a frase está sobre o texto da
   documentação; **a licença exata dos dados deve ser conferida na ficha do pacote** antes de publicar). Fonte primária
   ENTSO-E. Sem login. Zero à noite, amplitude sazonal e crescimento da capacidade instalada: o ciclo de 24 h é um perfil
   de sino, de fácil leitura por poucos harmônicos (R² de T ⊕ S = 0,999 em janelas de 48 h).

Ressalvas de fidelidade ao pedido:
- Não incluí PEMS/Traffic nem Solar-Energy do LSTNet: os arquivos estão no GitHub (`laiguokun/multivariate-time-series-data`)
  sem licença declarada (a API do GitHub devolveu `license: None`), e `nrel.gov` não resolveu desta máquina. O Monash
  (Zenodo) respondeu 403 ("unusual traffic") no momento da consulta.
- Os três têm n_obs ≥ 8 000 h.
- Dois dos três (Melbourne e `opsd_de_solar`) têm ciclo muito forte e janelas de 48 h quase inteiras em T ⊕ S
  (R² T⊕S ≥ 0,96); são os candidatos naturais para a NFH. O `electricity_mt156` é o controle de ciclo moderado.
- Aviso de operação: `codigo/sinteticos.py` sobrescreve `dados/catalogo.json` (linha 173) apenas com as entradas
  sintéticas. Se for rodado de novo, rodar `dados/baixar_dados.py` depois, que reinsere as três entradas reais.

## 6. O que não pude confirmar
- Sem artigo lido que meça acurácia de uma cabeça de base fixa + resíduo penalizado contra a densa com tudo igual.
- Os números do N-BEATS, N-HiTS e CycleNet vêm do HTML do arXiv; não reproduzi nenhum.
- Licença Melbourne (versão da CC BY) e licença dos dados do OPSD: conferir.
- `InstanceNorm` de `src/v2_common.py` não foi aberto por mim; a decisão sobre o bin 0 do `FreqMLP` depende dele.
- Conferência do `electricity_mt156`: o horário de 15 min do UCI marca o fim do intervalo; a agregação horária por média
  usa o rótulo do início da hora do `resample`. Um deslocamento de 15 min não afeta o ciclo de 24 h.
