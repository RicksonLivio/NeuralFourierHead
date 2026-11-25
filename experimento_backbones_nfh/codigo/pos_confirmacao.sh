#!/usr/bin/env bash
# pos_confirmacao.sh: espera o orquestrador terminar e roda, em sequência, as explicações das combinações com SUCESSO,
# o relatório e uma sessão separada do avaliador externo (papel auditor experimental). Tudo em CPU.
# Uso: setsid nohup bash experimento_backbones_nfh/codigo/pos_confirmacao.sh > .../resultados/logs/pos_confirmacao.log 2>&1 &
set -u
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
B=$RAIZ/experimento_backbones_nfh; R=$B/resultados; PY=$RAIZ/.venv/bin/python
log() { echo "[$(date '+%F %T')] $*"; }
cd "$RAIZ" || exit 1

log "aguardando ORQUESTRADOR_FIM"
while [ ! -e "$R/logs/ORQUESTRADOR_FIM" ]; do
  pgrep -f "bash .*experimento_backbones_nfh/codigo/orquestrar.sh" > /dev/null || { log "orquestrador não está vivo e não terminou"; break; }
  sleep 300
done

log "explicações (CPU, 2 threads)"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" "$B/codigo/explicar_bb.py" || log "AVISO: explicar_bb.py terminou com erro"

log "relatório"
CUDA_VISIBLE_DEVICES='' "$PY" "$B/codigo/relatorio.py" || log "AVISO: relatorio.py terminou com erro"
ls -la "$B/relatorio/" || true

log "avaliador externo (auditor experimental, sessão separada do Kiro CLI)"
kiro-cli chat --no-interactive --agent nfh-avaliador-externo --trust-all-tools "Papel AUDITOR EXPERIMENTAL sobre experimento_backbones_nfh/. Leia INSTRUCOES.md (inclusive o §10 com todas as mudanças e o registro honesto das 17h17), os CSV de resultados/ (t1, t2, confirmacao), as filas, os selecao_*.md, relatorio/relatorio.pdf e numeros.json, resultados/explicacoes/ e exemplo_sintetico/ (confirmacao.md, VALIDACAO.md). Recalcule com código seu as tabelas do relatório a partir dos CSV; confira que nada foi escolhido pelo teste (refaça as seleções T1 e T2 lendo só val_* e compare com as filas); que a confirmação usou as sementes 10 a 14 (e 10 a 19 no exemplo); que só combinações com RMSE e MAE de teste iguais ou melhores que a densa e a densa+E são citadas como exemplos; e que as derrotas estão relatadas. Grave .kiro/specs/rodada-20261121/avaliadores_externos/20261122_backbones_auditor.md e termine com APROVADO, APROVADO COM RESSALVAS ou NEEDS_CHANGES." || log "AVISO: avaliador externo terminou com erro"

log "POS_CONFIRMACAO_FIM"
touch "$R/logs/POS_CONFIRMACAO_FIM"
