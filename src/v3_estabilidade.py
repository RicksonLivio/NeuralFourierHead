"""
E3-v3: estabilidade das participações entre sementes e efeito isolado da identificabilidade.

Lê results/v3/partes/*.npz (gravados por v3_benchmark.py) e, por conjunto e modelo, calcula para cada
janela de teste o desvio-padrão entre as cinco sementes da participação da tendência, e resume pela
mediana sobre as janelas:
  * NFH (K = 10, seleção segura): eps_T;
  * camada densa, decomposição pura (K = 10): eps_T;
  * N-BEATS-I: participação NATIVA da pilha de tendência, rho_T = <T_c, y_c>/||y_c||^2, e participação
    PROJETADA eps_T da mesma previsão (Teorema de identificabilidade), nos MESMOS modelos.
No N-BEATS-I, a diferença entre a dispersão nativa e a projetada isola o efeito da não identificabilidade
(Proposição 3.2 e Corolário 3.1): as previsões são as mesmas, só muda a regra de divisão.
Também relata, no N-BEATS-I, a fração de janelas com rho_T fora de [0, 1] e a diferença média
|rho_T - eps_T| na mesma previsão.
Saída: results/v3/estabilidade.json, results/v3/estabilidade.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from v3_common import BACKBONES, CONJUNTOS, RES3, SEEDS  # noqa: E402

DIR_P = RES3 / "partes"


def carregar_seeds(prefixo, chave):
    vals = []
    for s in SEEDS:
        f = DIR_P / f"{prefixo}__s{s}.npz"
        if not f.exists():
            return None
        vals.append(np.load(f)[chave])
    return np.stack(vals)          # sementes x janelas


def resumo_disp(M):
    dp = M.std(0)
    return {"mediana_dp_entre_sementes": float(np.median(dp)), "media_dp": float(dp.mean()),
            "p90_dp": float(np.quantile(dp, 0.9)), "mediana_valor": float(np.median(M))}


def main():
    linhas, saida = [], {}
    for conj in CONJUNTOS:
        saida[conj] = {}
        for bb in BACKBONES:
            for rot, chave in (("NFH", "K10_nfh_eps_T"), ("densa (decomposição pura)", "K10_densa_eps_T")):
                M = carregar_seeds(f"{conj}__{bb}", chave)
                if M is None:
                    continue
                r = resumo_disp(M)
                saida[conj][f"{bb} + {rot}"] = r
                linhas.append({"conjunto": conj, "modelo": f"{bb} + {rot}", "medida": "eps_T", **r})
        nat = carregar_seeds(f"{conj}__N-BEATS-I", "nativo_rho_T")
        prj = carregar_seeds(f"{conj}__N-BEATS-I", "proj_eps_T")
        if nat is not None and prj is not None:
            rn, rp = resumo_disp(nat), resumo_disp(prj)
            fora = float(((nat < 0) | (nat > 1)).mean())
            dif = float(np.abs(nat - prj).mean())
            saida[conj]["N-BEATS-I nativo"] = {**rn, "fracao_fora_0_1": fora, "dif_media_nativo_proj": dif}
            saida[conj]["N-BEATS-I projetado"] = rp
            linhas.append({"conjunto": conj, "modelo": "N-BEATS-I", "medida": "rho_T nativo", **rn,
                           "fracao_fora_0_1": fora, "dif_media_nativo_proj": dif})
            linhas.append({"conjunto": conj, "modelo": "N-BEATS-I", "medida": "eps_T projetado", **rp})
    df = pd.DataFrame(linhas)
    df.to_csv(RES3 / "estabilidade.csv", index=False)
    (RES3 / "estabilidade.json").write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df[["conjunto", "modelo", "medida", "mediana_dp_entre_sementes", "mediana_valor"]
                 + [c for c in ("fracao_fora_0_1", "dif_media_nativo_proj") if c in df]])


if __name__ == "__main__":
    main()
