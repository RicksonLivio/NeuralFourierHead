#!/usr/bin/env bash
# lanca.sh: lança (ou completa) os workers do controle das épocas extras, em segundo plano (setsid nohup).
#
# Uso:  codigo/lanca.sh <fase> <fila.json> [n_gpu=1] [n_cpu=1]
#   fase: controle (real) | repro | smoke | smoke_cpu
# Ex.:  codigo/lanca.sh controle experimento_controle_epocas/resultados/filas/controle.json 1 1
#
# - Idempotente: o worker i (g<i> na GPU, c<i> na CPU) só é lançado se não houver um vivo com esse PID file;
#   rodar de novo (ou o escalar.sh) completa até n_gpu + n_cpu sem duplicar. Reivindicação dinâmica (um arquivo por
#   tarefa, retirado se o PID do dono morreu); NÃO apaga reivindicações.
# - Workers GPU só pegam tarefas GPU; workers CPU (CUDA_VISIBLE_DEVICES='', 2 threads) só pegam tarefas CPU.
# - Nas fases controle e repro recusa TESTE_MAX_EPOCAS e TESTE_E_EXTRA (só valem nas fases smoke*).
# - Máximos: 3 GPU e 5 CPU (o plano do protocolo, §4).
set -euo pipefail
FASE="${1:?fase (controle|repro|smoke|smoke_cpu)}"
FILA="${2:?arquivo da fila}"
NG="${3:-1}"
NC="${4:-1}"
case "$FASE" in
  controle|repro)
    for v in TESTE_MAX_EPOCAS TESTE_E_EXTRA; do
      if [ -n "${!v+x}" ]; then
        echo "RECUSADO: $v está definida (valor '${!v}'); só vale nas fases smoke*. Rode: unset TESTE_MAX_EPOCAS TESTE_E_EXTRA"
        exit 1
      fi
    done ;;
  smoke|smoke_cpu) ;;
  *) echo "fase desconhecida: $FASE"; exit 1 ;;
esac
[ "$NG" -le 3 ] || { echo "no máximo 3 workers GPU"; exit 1; }
[ "$NC" -le 5 ] || { echo "no máximo 5 workers CPU"; exit 1; }
RAIZ="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$RAIZ/.venv/bin/python"
EXE="$RAIZ/experimento_controle_epocas/codigo/executar.py"
RES="$RAIZ/experimento_controle_epocas/resultados"
[[ "$FILA" = /* ]] || FILA="$RAIZ/$FILA"
[ -s "$FILA" ] || { echo "fila não encontrada: $FILA (rode codigo/filas.py)"; exit 1; }
mkdir -p "$RES/logs" "$RES/pids"
EFET_E="${TESTE_E_EXTRA:-15}"; EFET_M="${TESTE_MAX_EPOCAS:-60}"
echo "fase=$FASE fila=$FILA E_EXTRA efetivo=$EFET_E MAX_EPOCAS efetivo=$EFET_M"

lancar() {  # $1 = g|c, $2 = índice
  local tipo="$1" i="$2" pidf="$RES/pids/${FASE}_${1}${2}.pid" log="$RES/logs/${FASE}_${1}${2}.log"
  if [ -e "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "  ${tipo}${i}: já vivo (pid $(cat "$pidf"))"; return
  fi
  echo "### $(date '+%F %T') lanca.sh: fase=$FASE worker=${tipo}${i} E_EXTRA=$EFET_E MAX_EPOCAS=$EFET_M" >> "$log"
  if [ "$tipo" = g ]; then
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 setsid nohup "$PY" "$EXE" --fase "$FASE" --fila "$FILA" \
      --worker "g$i" --device cuda >> "$log" 2>&1 < /dev/null &
  else
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 setsid nohup "$PY" "$EXE" --fase "$FASE" --fila "$FILA" \
      --worker "c$i" --device cpu >> "$log" 2>&1 < /dev/null &
  fi
  echo $! > "$pidf"
  echo "  ${tipo}${i}: lançado (pid $!)"
}

for ((i = 0; i < NG; i++)); do lancar g "$i"; done
for ((i = 0; i < NC; i++)); do lancar c "$i"; done
echo "alvo: $NG worker(s) GPU e $NC worker(s) CPU na fase $FASE. Acompanhe: .venv/bin/python experimento_controle_epocas/codigo/status.py $FASE"
