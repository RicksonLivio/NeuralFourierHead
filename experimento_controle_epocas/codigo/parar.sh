#!/usr/bin/env bash
# parar.sh: para os workers de uma fase (SIGTERM; o que já está no CSV é mantido; a tarefa em curso é refeita depois,
# e a reivindicação dela some quando o PID morre). Uso: codigo/parar.sh <fase|todas>
set -uo pipefail
FASE="${1:?fase (controle|repro|smoke|smoke_cpu|todas)}"
RAIZ="$(cd "$(dirname "$0")/../.." && pwd)"
RES="$RAIZ/experimento_controle_epocas/resultados"
PAT="$FASE"; [ "$FASE" = "todas" ] && PAT="*"
n=0
for p in "$RES"/pids/${PAT}_[gc]*.pid; do
  [ -e "$p" ] || continue
  pid="$(cat "$p")"
  if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" && n=$((n + 1)); fi
  rm -f "$p"
done
# o escalar.sh da fase, se estiver esperando
for p in "$RES"/pids/${PAT}_escalar.pid; do
  [ -e "$p" ] || continue
  kill -TERM "$(cat "$p")" 2>/dev/null; rm -f "$p"
done
echo "workers parados: $n"
