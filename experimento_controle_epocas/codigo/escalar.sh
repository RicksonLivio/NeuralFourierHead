#!/usr/bin/env bash
# escalar.sh: espera o fim do experimento de backbones e então sobe os workers do controle para 3 GPU + 5 CPU.
#
# O sinal é o arquivo experimento_backbones_nfh/resultados/logs/ORQUESTRADOR_FIM (criado pelo orquestrar.sh). Quando
# aparece, chama lanca.sh com 3 GPU e 5 CPU; como o lanca.sh só lança o que não está vivo, nada é duplicado e as
# tarefas são reivindicadas dinamicamente. Só LÊ a existência do arquivo; não toca no experimento de backbones.
#
# Uso: setsid nohup codigo/escalar.sh controle experimento_controle_epocas/resultados/filas/controle.json \
#        > experimento_controle_epocas/resultados/logs/escalar.log 2>&1 &
# Variáveis (testes): ESCALAR_FIM = arquivo-sinal; ESCALAR_ESPERA = segundos entre verificações (padrão 120).
set -uo pipefail
FASE="${1:?fase}"
FILA="${2:?fila}"
RAIZ="$(cd "$(dirname "$0")/../.." && pwd)"
RES="$RAIZ/experimento_controle_epocas/resultados"
FIM="${ESCALAR_FIM:-$RAIZ/experimento_backbones_nfh/resultados/logs/ORQUESTRADOR_FIM}"
ESPERA="${ESCALAR_ESPERA:-120}"
mkdir -p "$RES/pids" "$RES/logs"
PIDF="$RES/pids/${FASE}_escalar.pid"
if [ -e "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "escalar.sh já em execução (pid $(cat "$PIDF"))"; exit 1
fi
echo $$ > "$PIDF"
echo "[$(date '+%F %T')] escalar.sh: esperando $FIM"
while [ ! -e "$FIM" ]; do sleep "$ESPERA"; done
echo "[$(date '+%F %T')] sinal encontrado; subindo para 3 GPU + 5 CPU"
bash "$RAIZ/experimento_controle_epocas/codigo/lanca.sh" "$FASE" "$FILA" 3 5
rm -f "$PIDF"
echo "[$(date '+%F %T')] escalar.sh: feito"
