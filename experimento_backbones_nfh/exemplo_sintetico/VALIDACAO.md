# Validação independente do exemplo sintético passo a passo (21/11/2026)

Validador distinto do construtor. CPU, 1 thread. Nenhum arquivo do construtor foi alterado (md5 de `selecao.csv` e `confirmacao.csv` iguais antes e depois). Scripts do validador: `/tmp/val1.py`, `/tmp/val3.py`, `/tmp/val4.py`, `/tmp/cmp.py` (código numpy próprio, sem importar scripts do projeto).

## Veredito: APROVADO (com observações não bloqueantes)

## 1. Pré-registro (INSTRUCOES.md §10, 17h20)
- Grade: p ∈ {8,16}, n ∈ {300,1000}, variantes densa, densa+E, conv_L1000, zero_L0, zero_L1000, sementes 0 a 4: `selecao.csv` tem 100 linhas, 5 por (p,n,variante). Confere.
- `selecionar.py` lê com `usecols` só `val_*` e tem `assert` contra colunas `test`. Confere.
- Regra recalculada por mim a partir de `selecao.csv`: 3 combinações qualificam (p=16/n=1000/zero_L1000 −3,95%; p=16/n=1000/conv_L1000 −0,995%; p=8/n=1000/zero_L1000 −0,529%). Escolha p=16, n=1000, zero_L1000, sem empate. Confere com `selecao.json`. (p=16/n=300/zero_L0 tem −1,19% mas não qualifica: 3/5 sementes.)
- Confirmação: sementes 10 a 19, só p=16, n=1000. Recalculei de `confirmacao.csv`: RMSE NFH 0,303809, densa 0,328177, densa+E 0,328490; MAE 0,244464, 0,262513, 0,262475; desvios (ddof=1) 0,001509/0,011647/0,012428 e 0,001084/0,008819/0,009316. Vitórias da NFH: 10/10 em RMSE e 10/10 em MAE contra densa e densa+E. Médias ≤ e vitórias ≥ 8: veredito SUCESSO correto.

## 2. Reprodutibilidade
`passo_a_passo.py` rodado de novo (25 s). Comparei os 2373 valores numéricos do JSON com o original: diferença máxima 0 (tolerância 1e-6). Os asserts de reprodução do teste da semente 10 passaram.

## 3. Recálculo próprio (1293 números, todos dentro da tolerância)
τ, P, F, PPᵀ e inversa, F̃ (e F̃Pᵀ=0), B, Q (QᵀQ=I, QᵀB=0, projetor QQᵀ contra o QR), M, cond(M)=4,2086, normas e D; 8 passos de Gram-Schmidt (normas e q, ≤1e-9); gerador de x1 e x2 na janela (i0=8088); 64 produtos w·x do neurônio 1, soma, z, h1=tanh(z)=0,5454; as 8 linhas de c′ (128 produtos e somas) e c = Dc′; θ_T, θ_S, u; T, S, R; ŷ=Mc; T+S+R=ŷ (5,6e-17); média de T = média de ŷ; ε_T,ε_S,ε_R e soma = 1; norma centrada; amplitudes, fases, períodos e e_k dos 2 harmônicos; fórmulas de identificabilidade; verdade (θ_T*, θ_S*, energia); conversão W′=MDW, b′ e erro 2,2e-16; ida e volta de W; N-BEATS-I (lstsq, T_a e soma); IG: soma = g(x)−g(0) nas 4 funções (erros 5e-7, 9e-7, 2e-16, 9e-16), IG do oráculo por fórmula analítica (x/H e x·Π_S y); abdutiva: θ_T1, ξ, w, Γ∅, ganhos, ordem gulosa, acumulado, k*=13, pior caso com k* (0,1619 ≥ ξ) e sem o último (−0,01970 < ξ), e força bruta sobre todos os C(16,12)=1820 conjuntos de 12 neurônios: o melhor dá −0,01970 < ξ, logo 13 é mínimo. Amostragem independente (semente minha, 200 000 completamentos): 0 violações com A*, 53 146 sem o último.

## 4. Rastreabilidade
Extraí 2013 números do .tex e procurei cada um (com arredondamento ao nº de algarismos escritos) em passo_a_passo.json, confirmacao.json/csv, selecao.json/csv e derivados simples. 11 sem correspondência direta; todos são derivados que conferi à mão: somas por variável dos 64 produtos (0,6235; −0,5284; 0,1494; 0,8243), S+R da Tabela N-BEATS, ψ₂ mod 2π = 5,783, e as constantes estruturais 300, 1000. Nenhum número sem origem.
O texto cita `teste_exemplificacao_nfh/resultados/zero.csv` (médias: densa 0,677, nfh_zero_L1000 0,684 de RMSE; contagens desiguais entre variantes). Direção correta, mas é média simples, não pareada.

## 5. Clareza e correção (leitura do PDF, 18 páginas)
A primeira página traz as médias ± dp, as vitórias 10/10, o critério e o veredito, mais a ressalva "caso favorável, não é regra geral; em dados reais a NFH perdeu". A seção final separa "o que mostra" e "o que não mostra", e registra a espiada do coordenador. Sem passo pulado nem conta que não fecha.
Observações não bloqueantes (não corrigidas, pois exigem regenerar o documento):
1. Símbolo E duplicado: E = 50 épocas extras (§2) e E = ‖ŷ−ȳ‖² energia (§ Passo 4).
2. Na primeira página, "±" não diz que é desvio-padrão entre sementes; e o número de algarismos varia (0,3038 ± 0,001509).
3. "graças à penalização" e "a penalização é o que empurra a NFH" (uma janela, resíduo da densa 0,053% contra 1,6e-4%) são afirmações causais que vão além de uma janela. Sugestão: "consistente com o efeito da penalização".
4. "o IG diz o que pesa na média, não o que sustenta uma garantia" generaliza a partir de um caso (k_IG=16 contra k*=13) e a ordem testada é "estilo IG", não o IG. Sugestão: restringir a "neste exemplo".
5. θ_{T,0} aparece em §3.1 antes de ser definido em §Passo 2; ε_R*, ε_T* idem na primeira leitura.
6. O treino usa as primeiras n janelas (o protocolo da T1 usa as últimas); está registrado no log do construtor, mas não no documento. Sugestão: uma frase na seção "o que não mostra".

## 6. Figuras e compilação
As 4 figuras só têm tons de cinza (0 pixels coloridos após rasterizar); linhas distinguíveis por marcador e traço; a série x3 em cinza claro é de baixo contraste, mas legível. `tectonic` compila sem erro: só 1 aviso Underfull (linha 56), nenhum Overfull. 18 páginas.

## Correções feitas
Nenhuma (nada pequeno e objetivo; todos os números estão corretos).

## Polimento de 21/11

Outro agente (não o construtor) tratou as observações 1 a 7 só em `gerar_documento.py` (backup em `/tmp/gerar_documento.py.bak_polimento`). Nenhum número foi digitado: tudo sai de `passo_a_passo.json`, `confirmacao.json/csv`, `selecao.json/csv` (só `val_*`). CPU, 1 thread. `passo_a_passo.json`, `selecao.csv` e `confirmacao.csv` ficaram idênticos (md5 antes e depois).

1. **Símbolo E.** A energia centrada agora é $\mathcal E$ (macro `\En`); $E$ ficou só para as 50 épocas extras, com aviso explícito no Passo 4.
2. **Primeira página.** Diz que $\pm$ é o desvio-padrão amostral (denominador $n-1$) entre as 10 sementes; todas as médias e desvios têm quatro casas decimais (0,3038 ± 0,0015 etc., iguais a `confirmacao.md`). Também define RMSE/MAE, "pré-registrado", "camada densa" e "backbone".
3. **Causalidade.** Saíram "graças à penalização" e "a penalização é o que empurra". Entrou, na Seção 2, a comparação zero_L0 × zero_L1000 calculada de `selecao.csv` (só `val_*`): com p=16, n=1000, RMSE de validação 0,3342 contra 0,3148, menor em 5 de 5 sementes; com n=300 a ordem se inverte (0,4147 contra 0,4206). Passo 4 e Seção 11 passam a dizer que uma janela não demonstra a causa e remetem a essa comparação.
4. **IG.** A frase virou "neste exemplo"; o texto diz que a única ordem testada foi $|w_k(h_k-\bar h_k)|$ (estilo IG, linear, não o IG com integral), com $k_{IG}=16$ contra $k^*=13$.
5. **Definições antes do uso.** $\theta_{T,0},\theta_{T,1},\boldsymbol\theta_S,\bu$ definidos na Seção 1; o asterisco ($\varepsilon^\star$ etc.) definido na Seção 10 junto com $\varepsilon^\star_{T,S,R}$.
6. **Primeiras × últimas janelas.** Frase na seção do modelo e na seção "o que não mostra" (T1 usa as últimas, `INSTRUCOES.md`).
7. **zero.csv.** Frase diz que são médias simples, sem pareamento por semente, com número desigual de linhas por variante (conferido: 5 a 7 linhas por variante).

Extras de clareza: corrigida a indexação $D_{22}\to D_{11}$ na abdutiva (índices de $\bc'$ e $\bD$ começam em 0; antes o texto dizia "segunda coordenada" com $W_{1,k}$); explicado $\oplus$ e $\bS_k$ na definição de $e_k$.

Reverificação: `/tmp/val1.py` (seleção/confirmação: mesmos valores), `/tmp/val3.py` (1293 números, falhas: []), `/tmp/cmp.py` (2373 valores do JSON, dif. máx. 0), `/tmp/val4.py` (rastreabilidade: 2054 tokens, 11 sem correspondência, os mesmos derivados simples da seção 4 acima; os novos 0,3342, 0,3148, 0,4147, 0,4206 têm origem). `tectonic`: 18 páginas, 0 Overfull, 1 Underfull (linha 56, já existia).
