#!/usr/bin/env bash
# parar.sh: para os workers de uma fase (SIGTERM; o CSV já gravado é mantido, a tarefa em curso é refeita depois).
# Uso: codigo/parar.sh <fase>      (ou 'todas')
set -uo pipefail
FASE="${1:?fase (t1|t2|confirmacao|smoke|todas)}"
RAIZ="$(cd "$(dirname "$0")/../.." && pwd)"
RES="$RAIZ/experimento_backbones_nfh/resultados"
PAT="${FASE}"; [ "$FASE" = "todas" ] && PAT="*"
n=0
for p in "$RES"/pids/${PAT}_w*.pid; do
  [ -e "$p" ] || continue
  pid="$(cat "$p")"
  if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" && n=$((n + 1)); fi
  rm -f "$p"
done
[ "$FASE" = "todas" ] || rm -rf "$RES/claims/$FASE"
echo "workers parados: $n"
