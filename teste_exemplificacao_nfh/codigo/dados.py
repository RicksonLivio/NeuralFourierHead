"""
Carregador de conjuntos de dados para o teste de exemplificação da NFH.

Lê as entradas de teste_exemplificacao_nfh/dados/catalogo.json e devolve um dicionário
compatível com o de v3_common.carregar, com os mesmos campos:
  - nome, fonte (nome do dataset), modo (U ou M)
  - feats: np.array [N, d] com as features padronizadas (estatísticas do treino)
  - alvo: np.array [N] com o alvo padronizado
  - nomes: lista de nomes das variáveis
  - observado: np.array [N] bool, mascara de alvo originalmente observado
  - cortes: (fim_treino, fim_validação) índices absolutos
  - datas: np.array de datetime64
  - mu_alvo, sd_alvo: média e desvio do alvo no treino (para desnormalizar)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # xai_transformer/
DADOS = ROOT / "teste_exemplificacao_nfh" / "dados"


def _ler_catalogo():
    """Lê o catálogo de conjuntos de dados."""
    cat_path = DADOS / "catalogo.json"
    if not cat_path.exists():
        return []
    with open(cat_path, "r", encoding="utf-8") as f:
        return json.load(f)


def listar_conjuntos():
    """Lista os conjuntos disponíveis no catálogo (nome-modo)."""
    cat = _ler_catalogo()
    conjs = []
    for e in cat:
        nome = e["nome"]
        conjs.append(f"{nome}-U")  # sempre univariado
        if e.get("covariaveis"):  # multivariado se houver covariáveis
            conjs.append(f"{nome}-M")
    return conjs


def carregar(nome: str, L: int = 96, H: int = 48):
    """
    Carrega um conjunto no formato 'Nome-U' ou 'Nome-M'.
    
    Retorna dicionário compatível com v3_common.carregar:
      nome, fonte, modo, feats [N,d], alvo [N], nomes, observado [N], cortes, datas, mu_alvo, sd_alvo
    
    Protocolo:
      - Modo U: feats = alvo.reshape(-1, 1)
      - Modo M: feats = [covariáveis..., alvo] na ordem do catálogo
      - Padronização: com média e desvio do treino
      - Cortes: usando as frações do catálogo ou divisão padrão (70/15/15)
      - Valores ausentes: preenchidos por interpolação (para montar janelas);
        janelas com alvo ausente no horizonte são descartadas por MascWindows
    """
    partes = nome.rsplit("-", 1)
    if len(partes) != 2 or partes[1] not in ("U", "M"):
        raise ValueError(f"Nome inválido: {nome}. Esperado 'Dataset-U' ou 'Dataset-M'.")
    
    fonte, modo = partes
    cat = _ler_catalogo()
    
    # Encontra a entrada no catálogo
    entrada = None
    for e in cat:
        if e["nome"] == fonte:
            entrada = e
            break
    
    if entrada is None:
        raise ValueError(f"Conjunto '{fonte}' não encontrado em catalogo.json")
    
    # Lê o CSV
    csv_path = DADOS / entrada["arquivo"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")
    
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # Determina alvo e covariáveis
    alvo_col = entrada["alvo"]
    covar_cols = entrada.get("covariaveis", [])
    
    if modo == "M" and not covar_cols:
        raise ValueError(f"Modo M solicitado mas '{fonte}' não tem covariáveis no catálogo")
    
    # Máscara de observados (antes de interpolar)
    observado = df[alvo_col].notna().values
    
    # Interpola valores ausentes (só para montar janelas; MascWindows filtra horizontes com NaN)
    df = df.interpolate(limit_direction="both")
    
    # Determina cortes
    n = len(df)
    if "cortes" in entrada and entrada["cortes"]:
        frac_tr, frac_va = entrada["cortes"]
        a = int(frac_tr * n)
        b = int((frac_tr + frac_va) * n)
    else:
        # Padrão: 70/15/15
        a = int(0.70 * n)
        b = int(0.85 * n)
    
    # Padronização com estatísticas do treino
    treino_df = df.iloc[:a]
    
    # Alvo
    mu_alvo = float(treino_df[alvo_col].mean())
    sd_alvo = float(treino_df[alvo_col].std()) + 1e-12
    alvo_z = ((df[alvo_col].values - mu_alvo) / sd_alvo).astype(np.float32)
    
    # Features
    if modo == "U":
        feats = alvo_z.reshape(-1, 1)
        nomes = [alvo_col]
    else:
        # Ordem: covariáveis primeiro, alvo por último
        colunas = covar_cols + [alvo_col]
        raw = df[colunas].values.astype(float)
        mu_feats = treino_df[colunas].mean().values
        sd_feats = treino_df[colunas].std().values + 1e-12
        feats = ((raw - mu_feats) / sd_feats).astype(np.float32)
        nomes = colunas
    
    # Datas
    datas = df["date"].values
    
    return {
        "nome": nome,
        "fonte": fonte,
        "modo": modo,
        "feats": np.ascontiguousarray(feats),
        "alvo": np.ascontiguousarray(alvo_z),
        "nomes": nomes,
        "observado": observado,
        "cortes": (a, b),
        "datas": datas,
        "mu_alvo": mu_alvo,
        "sd_alvo": sd_alvo,
    }


def segmentos(ds: dict, L: int = 96, H: int = 48):
    """
    Cria os datasets de treino, validação e teste (MascWindows).
    
    Importa MascWindows de v3_common para manter compatibilidade.
    """
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from v3_common import MascWindows
    
    a, b = ds["cortes"]
    f, t, o = ds["feats"], ds["alvo"], ds["observado"]
    
    return (
        MascWindows(f[:a], t[:a], o[:a], L, H),
        MascWindows(f[a:b], t[a:b], o[a:b], L, H),
        MascWindows(f[b:], t[b:], o[b:], L, H),
    )


# ---------------------------------------------------------------------------
# Smoke test: cria entrada do catálogo a partir de ETTh1 existente
# ---------------------------------------------------------------------------
def criar_smoke_test_etth1():
    """
    Cria uma entrada de catálogo e cópia reduzida do ETTh1 para smoke test.
    Usa as primeiras 2000 observações (~ 83 dias) para teste rápido.
    """
    etth1_path = ROOT / "data" / "controle" / "ETTh1.csv"
    if not etth1_path.exists():
        raise FileNotFoundError(f"ETTh1.csv não encontrado em {etth1_path}")
    
    # Lê o ETTh1 original
    df = pd.read_csv(etth1_path, parse_dates=["date"])
    
    # Usa as primeiras 2000 observações (suficiente para smoke test)
    df_pequeno = df.iloc[:2000].copy()
    
    # Salva o CSV pequeno
    DADOS.mkdir(parents=True, exist_ok=True)
    csv_path = DADOS / "ETTh1_smoke.csv"
    df_pequeno.to_csv(csv_path, index=False)
    
    # Cria ou atualiza o catálogo
    cat_path = DADOS / "catalogo.json"
    if cat_path.exists():
        with open(cat_path, "r", encoding="utf-8") as f:
            cat = json.load(f)
    else:
        cat = []
    
    # Remove entrada anterior se existir
    cat = [e for e in cat if e["nome"] != "ETTh1_smoke"]
    
    # Adiciona nova entrada
    entrada = {
        "nome": "ETTh1_smoke",
        "arquivo": "ETTh1_smoke.csv",
        "fonte_url": "https://github.com/zhouhaoyi/ETDataset",
        "licenca": "MIT",
        "freq": "H",
        "alvo": "OT",
        "covariaveis": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
        "periodos_dominantes_h": [24, 12],
        "cortes": [0.70, 0.15],
        "n_obs": len(df_pequeno),
        "faltantes_frac": 0.0,
        "notas": "Subconjunto do ETTh1 para smoke test (primeiras 2000 h)."
    }
    cat.append(entrada)
    
    with open(cat_path, "w", encoding="utf-8") as f:
        json.dump(cat, f, indent=2, ensure_ascii=False)
    
    print(f"Smoke test criado: {csv_path} ({len(df_pequeno)} obs)")
    print(f"Catálogo atualizado: {cat_path}")
    return entrada


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--criar-smoke":
        criar_smoke_test_etth1()
    else:
        # Lista conjuntos disponíveis
        conjs = listar_conjuntos()
        if conjs:
            print("Conjuntos disponíveis:")
            for c in conjs:
                print(f"  {c}")
        else:
            print("Nenhum conjunto no catálogo. Use --criar-smoke para criar um de teste.")
