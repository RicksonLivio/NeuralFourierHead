"""Rastreia em que dispositivo (GPU ou CPU) foi produzida a execução que ficou nos resultados de cada
(conjunto, modelo, semente), a partir dos logs do benchmark v3. Como os resultados usam a última linha
gravada de cada configuração (deduplicação keep="last"), vale a última ocorrência no tempo.

Logs: logs/v3/benchmark_cpu_*.log -> CPU; logs/v3/benchmark_gpu_*.log e os logs originais
logs/v3/benchmark_{ETTh1-U,ETTh2-U,Jena-U,Jena-M}.log -> GPU.
Saída: results/v3/dispositivos.json
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
LOGS = RAIZ / "logs" / "v3"
LIN_BB = re.compile(r"^\[(\d\d:\d\d:\d\d)\] (\S+) (MLP|LSTM|Conv1D\+LSTM) s(\d) K=2:")
LIN_NB = re.compile(r"^\[(\d\d:\d\d:\d\d)\] (\S+) N-BEATS-I s(\d):")


def main():
    ultimo = {}
    for f in sorted(LOGS.glob("benchmark_*.log")):
        disp = "CPU" if f.name.startswith("benchmark_cpu_") else "GPU"
        if f.name == "benchmark_fim.log":
            continue
        for linha in f.read_text(errors="ignore").splitlines():
            m = LIN_BB.match(linha)
            if m:
                hora, conj, bb, s = m.groups()
                chave = (conj, bb, int(s))
            else:
                m = LIN_NB.match(linha)
                if not m:
                    continue
                hora, conj, s = m.groups()
                chave = (conj, "N-BEATS-I", int(s))
            if chave not in ultimo or hora >= ultimo[chave][0]:
                ultimo[chave] = (hora, disp, f.name)
    por_conj = {}
    for (conj, mod, s), (hora, disp, arq) in sorted(ultimo.items()):
        por_conj.setdefault(conj, {}).setdefault(mod, {})[s] = disp
    cont = Counter(d for (_, d, _) in ultimo.values())
    saida = {"total_execucoes": len(ultimo), "por_dispositivo": dict(cont), "detalhe": por_conj}
    (RAIZ / "results" / "v3" / "dispositivos.json").write_text(json.dumps(saida, indent=1, ensure_ascii=False))
    print(json.dumps(saida, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
