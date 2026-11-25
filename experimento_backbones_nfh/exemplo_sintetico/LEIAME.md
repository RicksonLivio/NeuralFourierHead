# Exemplo sintético passo a passo da NFH

Pré-registro: `experimento_backbones_nfh/INSTRUCOES.md` §10 (21/11/2026, 17h20). Série de livro (`teste_exemplificacao_nfh/exemplo_livro/gerar.py`,
sigma = 0,3), backbone `tanh(W vec(X) + b)`. Só CPU, 1 thread. Rode a partir desta pasta (`.venv` da raiz); o tempo foi medido com a CPU
compartilhada com a T1 (carga ~15).

| # | Comando | O que faz | Tempo medido |
|---|---------|-----------|--------------|
| 1 | `CUDA_VISIBLE_DEVICES='' ../../.venv/bin/python treinar_selecao.py` | grade p∈{8,16}, n∈{300,1000}, sementes 0–4, 5 variantes → `selecao.csv` | ~6,6 min |
| 2 | `../../.venv/bin/python selecionar.py` | regra do pré-registro, lê só `val_*` → `selecao.md`, `selecao.json` | < 1 s |
| 3 | `CUDA_VISIBLE_DEVICES='' ../../.venv/bin/python confirmar.py` | sementes 10–19 → `confirmacao.csv/.md/.json` e veredito | ~2,6 min |
| 4 | `CUDA_VISIBLE_DEVICES='' ../../.venv/bin/python passo_a_passo.py` | retreina a semente 10, confere o RMSE/MAE de teste, grava `passo_a_passo.json` | ~30 s |
| 6 | `CUDA_VISIBLE_DEVICES='' ../../.venv/bin/python projetar_confirmacao.py` | controle relatado (21/11 21h25, depois da confirmação): densa projetada em T ⊕ S, sem treino, sementes 10–19 → `projecao_confirmacao.csv` (rodar antes do passo 5) | não medido |
| 5 | `../../.venv/bin/python gerar_documento.py` | escreve `exemplo_passo_a_passo.tex`, figuras e compila com tectonic (precisa de rede na 1ª vez) | ~8 s |

Arquivos de apoio: `comum.py` (importa, sem editar, `exemplo_livro/treinar.py` e `gerar.py`; define o treino das variantes).
Resultado (ver `confirmacao.md`): combinação escolhida p = 16, n = 1000, zero_L1000; veredito SUCESSO. Isso vale para esta série favorável, não é regra geral.
Treino usa as primeiras n janelas de treino (como no `treinar.py` do livro).
