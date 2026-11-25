#!/usr/bin/env bash
# lanca.sh: lança workers do experimento de backbones (GPU e CPU) em segundo plano, sobrevivendo ao terminal.
#
# Uso:  codigo/lanca.sh <fase> <fila.json> [n_gpu=3] [n_cpu=6]
# Ex.:  codigo/lanca.sh t1 experimento_backbones_nfh/resultados/filas/t1.json 3 6
#
# - recusa as variáveis TESTE_MAX_EPOCAS e TESTE_E_EXTRA nas fases t1, t2 e confirmacao; grava no cabeçalho do log
#   os valores efetivos de E_EXTRA e MAX_EPOCAS;
# - até 3 workers na GPU e até 6 em CPU (CUDA_VISIBLE_DEVICES='' e 2 threads cada);
# - cada worker: setsid nohup, log em resultados/logs/<fase>_w<i>.log, PID em resultados/pids/<fase>_w<i>.pid;
# - reivindicação dinâmica de tarefas; reiniciar é seguro (o que já está no CSV é pulado). Antes de lançar,
#   apaga as reivindicações antigas da fase (sobras de workers mortos); recusa se já há workers vivos da fase.
set -euo pipefail
FASE="${1:?fase (t1|t2|confirmacao|smoke)}"
FILA="${2:?arquivo da fila}"
NG="${3:-3}"
NC="${4:-6}"
case "$FASE" in
  t1|t2|confirmacao)
    for v in TESTE_MAX_EPOCAS TESTE_E_EXTRA; do
      if [ -n "${!v+x}" ]; then
        echo "RECUSADO: $v está definida (valor '${!v}'); só vale nas fases smoke*. Rode: unset TESTE_MAX_EPOCAS TESTE_E_EXTRA"
        exit 1
      fi
    done ;;
esac
[ "$NG" -le 3 ] || { echo "no máximo 3 workers GPU"; exit 1; }
[ "$NC" -le 6 ] || { echo "no máximo 6 workers CPU"; exit 1; }
RAIZ="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$RAIZ/.venv/bin/python"
EXE="$RAIZ/experimento_backbones_nfh/codigo/executar.py"
RES="$RAIZ/experimento_backbones_nfh/resultados"
mkdir -p "$RES/logs" "$RES/pids"
EFET_E="${TESTE_E_EXTRA:-15}"; EFET_M="${TESTE_MAX_EPOCAS:-60}"
echo "fase=$FASE fila=$FILA E_EXTRA efetivo=$EFET_E MAX_EPOCAS efetivo=$EFET_M"
[[ "$FILA" = /* ]] || FILA="$RAIZ/$FILA"

for p in "$RES"/pids/"${FASE}"_w*.pid; do
  [ -e "$p" ] || continue
  if kill -0 "$(cat "$p")" 2>/dev/null; then echo "worker vivo ($p): rode codigo/parar.sh $FASE antes"; exit 1; fi
  rm -f "$p"
done
rm -rf "$RES/claims/$FASE"

N=$((NG + NC))
for ((i = 0; i < N; i++)); do
  LOG="$RES/logs/${FASE}_w${i}.log"
  echo "### $(date '+%F %T') lanca.sh: fase=$FASE fila=$FILA E_EXTRA efetivo=$EFET_E MAX_EPOCAS efetivo=$EFET_M" >> "$LOG"
  if [ "$i" -lt "$NG" ]; then
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      setsid nohup "$PY" "$EXE" --fase "$FASE" --fila "$FILA" --worker "$i" --n-workers "$N" --device cuda \
      >> "$LOG" 2>&1 < /dev/null &
  else
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      setsid nohup "$PY" "$EXE" --fase "$FASE" --fila "$FILA" --worker "$i" --n-workers "$N" --device cpu \
      >> "$LOG" 2>&1 < /dev/null &
  fi
  echo $! > "$RES/pids/${FASE}_w${i}.pid"
done
echo "lançados $NG worker(s) GPU e $NC worker(s) CPU para a fase $FASE. Acompanhe: python experimento_backbones_nfh/codigo/status.py $FASE"
