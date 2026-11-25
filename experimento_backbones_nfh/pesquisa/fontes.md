# Fontes abertas pelo agente pesquisa_backbones (acesso em 21/11/2026)

"Lido" diz o que de fato abri. Cópias de trabalho ficaram em `/tmp/bb` (não versionadas).

## Artigos (arXiv)
| URL | O que li |
|---|---|
| https://arxiv.org/abs/1803.01271 | Resumo do TCN (Bai, Kolter, Koltun). Link do código: github.com/locuslab/TCN |
| https://arxiv.org/abs/2205.13504 e https://arxiv.org/html/2205.13504 | Resumo do LTSF-Linear/DLinear (Zeng et al.); HTML pesquisado só por trechos sobre janela de entrada |
| https://arxiv.org/abs/2211.14730 e /html/2211.14730 | Resumo do PatchTST (Nie et al.) |
| https://arxiv.org/abs/2303.06053 e /html/2303.06053 | Resumo do TSMixer (Chen et al.) |
| https://arxiv.org/abs/2310.06625 e /html/2310.06625 | Resumo do iTransformer (Liu et al.) |
| https://arxiv.org/html/1905.10437 | N-BEATS: Tabela 1 (M4, M3, turismo), seção de ensemble |
| https://arxiv.org/abs/2201.12886 e /html/2201.12886 | N-HiTS: contribuições, Tabela 2, descrição das variantes N-HiTS₂/₃/₄, Apêndice G (interpolação), discussão |
| https://arxiv.org/html/2307.03756 | FITS: resumo, filtro passa-baixa, escolha da frequência de corte, tabela de parâmetros |
| https://arxiv.org/html/2410.18318 | Eefsen et al., crítica ao FITS: conclusões sobre o que o FITS não captura, gerador sintético (seção 4.2) |
| https://arxiv.org/html/2405.00946 | SparseTSF: ablação da técnica Sparse, generalização, limitações (seção 5.1), comparação com N-HiTS (só o título da subseção) |
| https://arxiv.org/html/2409.18479 | CycleNet: RCF, Tabela 3 (parâmetros), Tabela 4 (ablação Electricity e Traffic), escolha do comprimento do ciclo |
| https://arxiv.org/html/2405.14616 | TimeMixer: só o título (HTML baixado; li o código, não o texto) |
| https://arxiv.org/html/2311.06184 e /html/2402.05956 | FreTS e Pathformer: baixados, **não usados** no relatório |
| https://arxiv.org/abs/2101.12414 | Low Rank Forecasting (Barratt et al.): resumo |
| https://arxiv.org/abs/2603.06781 | xaitimesynth: resumo |
| https://arxiv.org/abs/2506.11790 | Class-dependent evaluation effects (séries sintéticas, atribuição): resumo |
| https://arxiv.org/abs/2103.10005 | Benchmark sintético de atribuição para geociências (Mamalakis et al.): resumo |
| Busca web ("synthetic time series benchmark interpretability…", "low-rank output layer time series…") | só os trechos devolvidos pela busca; DoTime (arXiv 2607.27263), 2609.30397 e 2603.29756 aparecem na busca e **não foram abertos nem citados no relatório** |

## Código oficial (raw.githubusercontent.com)
| URL | O que li |
|---|---|
| https://raw.githubusercontent.com/yuqinie98/PatchTST/main/PatchTST_supervised/layers/PatchTST_backbone.py | `Flatten_Head`, `head_nf = d_model * patch_num`, `patch_num` e `padding_patch` |
| https://raw.githubusercontent.com/cure-lab/LTSF-Linear/main/models/DLinear.py | `series_decomp(25)`, `Linear_Seasonal`, `Linear_Trend` |
| https://raw.githubusercontent.com/thuml/iTransformer/main/model/iTransformer.py | `projector = nn.Linear(d_model, pred_len)` |
| https://raw.githubusercontent.com/google-research/google-research/master/tsmixer/tsmixer_basic/models/tsmixer.py | blocos e `Dense(pred_len)` |
| https://raw.githubusercontent.com/locuslab/TCN/master/TCN/adding_problem/model.py e .../TCN/tcn.py | `Linear(num_channels[-1], output_size)` sobre o último instante; dilatação 2**i |
| https://raw.githubusercontent.com/ServiceNow/N-BEATS/master/models/nbeats.py | `GenericBasis`, `TrendBasis`, `SeasonalityBasis`; licença CC BY-NC 4.0 no cabeçalho |
| https://raw.githubusercontent.com/Nixtla/neuralforecast/main/neuralforecast/models/nhits.py | `_IdentityBasis` com `F.interpolate`, `n_theta` |
| https://raw.githubusercontent.com/VEWOXIC/FITS/main/models/FITS.py | `freq_upsampler` complexo, LPF, `irfft` |
| https://raw.githubusercontent.com/lss-1138/SparseTSF/main/models/SparseTSF.py | `conv1d`, reshape por período, `Linear(seg_num_x, seg_num_y)` |
| https://raw.githubusercontent.com/ACAT-SCUT/CycleNet/main/models/CycleNet.py | `RecurrentCycle`, subtração/soma do ciclo, `Linear`/MLP |
| https://raw.githubusercontent.com/kwuking/TimeMixer/main/models/TimeMixer.py | `predict_layers`, `projection_layer` por escala |
| https://raw.githubusercontent.com/thuml/Time-Series-Library/main/models/TimesNet.py | `predict_linear`, `projection` |

## Dados
| URL | O que li ou baixei |
|---|---|
| https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/pedestrian-counting-system-monthly-counts-per-hour | metadados: licença "CC BY", título, 1 631 158 registros, descrição |
| .../pedestrian-counting-system-monthly-counts-per-hour/records?group_by=... e .../exports/csv?where=location_id%3D35 | contagem por sensor (8 maiores com 17 490 a 17 498 linhas); CSV do sensor 35 |
| https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip (261 MB) e https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 | `LD2011_2014.txt` (370 colunas, 15 min); página do dataset: licença CC BY 4.0 |
| https://data.open-power-system-data.org/time_series/ e .../time_series/2020-10-06/time_series_60min_singleindex.csv (130 MB) | página do pacote (frase "Text is licensed under CC-BY license", versão 2020-10-06, citação) e colunas `DE_*` |
| https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/README.md e https://api.github.com/repos/laiguokun/multivariate-time-series-data | descrição de Electricity/Traffic/Solar do LSTNet; API do GitHub: `license: None` (por isso não usei) |
| https://zenodo.org/api/records/4656626 (e 4656140, 4656132, 4656144) | **HTTP 403** ("unusual traffic"); nada lido |
| https://www.nrel.gov/grid/solar-power-data.html | **não resolveu** o host (`Could not resolve host`); nada lido |
| https://data.openei.org/files/4/nrel_solar.zip | HTTP 404 |
| https://re.jrc.ec.europa.eu/api/v5_3/seriescalc?... (PVGIS) | resposta JSON aberta; **descartado** (série modelada, não medida) |

## Arquivos locais lidos
`.kiro/specs/rodada-20261121/INSTRUCOES_RODADA.md`, `experimento_backbones_nfh/INSTRUCOES.md`,
`teste_exemplificacao_nfh/pesquisa/relatorio_pesquisa.md`, `teste_exemplificacao_nfh/dados/catalogo.json`,
`teste_exemplificacao_nfh/dados/baixar_dados.py` (função do ciclo diário), `teste_exemplificacao_nfh/codigo/dados.py`,
`experimento_backbones_nfh/codigo/sinteticos.py`, `experimento_backbones_nfh/dados/sint_u48_oraculo.json`.
