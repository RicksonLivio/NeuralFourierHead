#!/usr/bin/env bash
# Roda o experimento principal da v3 em 4 processos paralelos na GPU (retomável).
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p logs/v3
run() { .venv/bin/python src/v3_benchmark.py "$@" >> "logs/v3/benchmark_$1.log" 2>&1; echo "FIM $(date +%H:%M:%S) exit=$?" >> "logs/v3/benchmark_$1.log"; }
run Jena-U &
run Jena-M &
run ETTh1-U ETTh1-M AirQ-U &
run ETTh2-U ETTh2-M AirQ-M &
wait
echo "TODOS FIM $(date +%H:%M:%S)" >> logs/v3/benchmark_fim.log
