#!/usr/bin/env bash
# orquestrar.sh: encadeia T1 -> seleção -> T2 -> seleção -> confirmação sem intervenção (protocolo §6 e §10).
# A seleção é sempre a de selecionar.py, que só lê colunas val_*. Nenhum número de teste é lido aqui.
# Uso: setsid nohup bash experimento_backbones_nfh/codigo/orquestrar.sh > .../resultados/logs/orquestrador.log 2>&1 &
set -u
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
B=$RAIZ/experimento_backbones_nfh
C=$B/codigo; R=$B/resultados
PY=$RAIZ/.venv/bin/python
log() { echo "[$(date '+%F %T')] $*"; }

vivos() {  # número de workers vivos da fase $1
  local n=0
  for p in "$R"/pids/"$1"_w*.pid; do
    [ -e "$p" ] || continue
    kill -0 "$(cat "$p")" 2>/dev/null && n=$((n + 1))
  done
  echo $n
}

extra_gpu() {  # 9.º worker (GPU, id 8), §10 de 21/11 18h15
  local fase=$1 fila=$2
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 setsid nohup "$PY" "$C/executar.py" --fase "$fase" --fila "$fila" \
    --worker 8 --n-workers 9 --device cuda >> "$R/logs/${fase}_w8.log" 2>&1 < /dev/null &
  echo $! > "$R/pids/${fase}_w8.pid"
}

esperar() {  # espera a fase $1 acabar; se faltar algo, relança uma vez
  local fase=$1 fila=$2
  while [ "$(vivos "$fase")" -gt 0 ]; do sleep 120; done
  if ! "$PY" "$C/completo.py" "$fase" "$fila"; then
    log "$fase incompleta depois que os workers terminaram; relançando uma vez"
    bash "$C/lanca.sh" "$fase" "$fila" 3 5 && extra_gpu "$fase" "$fila"
    sleep 60
    while [ "$(vivos "$fase")" -gt 0 ]; do sleep 120; done
    "$PY" "$C/completo.py" "$fase" "$fila" || log "AVISO: $fase continua incompleta; segue com o que existe"
  fi
  log "$fase terminada"
}

unset TESTE_MAX_EPOCAS TESTE_E_EXTRA
log "orquestrador iniciado"
esperar t1 "$R/filas/t1_B2.json"

log "seleção T1 -> T2"
"$PY" "$C/selecionar.py" t1 || { log "ERRO na seleção T1"; exit 1; }
[ -s "$R/filas/t2.json" ] || { log "ERRO: fila t2 vazia ou ausente"; exit 1; }
bash "$C/lanca.sh" t2 "$R/filas/t2.json" 3 5 && extra_gpu t2 "$R/filas/t2.json"
sleep 60
esperar t2 "$R/filas/t2.json"

log "seleção T2 -> confirmação"
"$PY" "$C/selecionar.py" t2 || { log "ERRO na seleção T2"; exit 1; }
[ -s "$R/filas/confirmacao.json" ] || { log "ERRO: fila de confirmação vazia ou ausente"; exit 1; }
bash "$C/lanca.sh" confirmacao "$R/filas/confirmacao.json" 3 5 && extra_gpu confirmacao "$R/filas/confirmacao.json"
sleep 60
esperar confirmacao "$R/filas/confirmacao.json"

log "ORQUESTRADOR_FIM"
touch "$R/logs/ORQUESTRADOR_FIM"
