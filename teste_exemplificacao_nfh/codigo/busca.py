#!/usr/bin/env python3
"""
busca.py — Executor de tarefas retomável para o teste de exemplificação da NFH.

Uma tarefa = (dataset, modo, backbone, largura, semente).

Para cada tarefa:
  1. Treina o modelo denso como v3_benchmark (Adam, MSE, early stopping)
  2. Calcula as variantes:
     - 'densa': modelo denso treinado
     - 'densa+E': continua treinando o denso por E épocas extras (controle de épocas)
     - 'projeção' (K=10): descarta resíduo da previsão densa
     - NFH: conversão exata + ajuste estrutural para cada (lambda, K)

CSV de saída: resultados/<fase>.csv
Modelos salvos: resultados/modelos/<dataset>_<backbone>_w<width>_s<seed>_<variante>.pt

Uso:
  python busca.py --fase triagem --conjuntos ETTh1_smoke-U --backbones MLP --larguras 1 --seeds 0 --device cuda
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Caminhos do projeto
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "teste_exemplificacao_nfh" / "codigo"))

from v2_common import (CONFIG, BackboneHeadModel, ConvLSTMBackbone, DenseHead,
                       InstanceNorm, NFHHead, nfh_bases, residual_penalty,
                       set_seed, stack_ds, train_model)
from v3_common import (CFG3, H3, K3, L3, Q3, Projetor, ajuste_estrutural,
                       avaliar, densa_para_nfh, ingenuas, matriz_sintese)
from dados import carregar, segmentos

# Diretórios de saída
RES = ROOT / "teste_exemplificacao_nfh" / "resultados"
RES.mkdir(parents=True, exist_ok=True)

# Configuração padrão (herdada de CFG3, pode ser sobrescrita por variáveis de ambiente)
L, H, K_DEFAULT, Q = L3, H3, K3, Q3
E_EXTRA = 15  # épocas extras para densa+E
LAMBDAS = [int(x) for x in os.environ.get("TESTE_LAMBDAS", "100,1000").split(",")]
KS = [int(x) for x in os.environ.get("TESTE_KS", "10,4,2").split(",")]
LR_AJUSTE = 1e-4

# Colunas do CSV de saída
COLS = [
    "dataset", "modo", "backbone", "largura", "seed", "variante", "K", "lambda",
    "val_RMSE", "test_RMSE", "test_MAE", "cobertura_mediana", "epocas", "tempo_s",
    "params", "n_janelas_teste", "n_janelas_treino", "n_janelas_val", "d",
    "erro_conversao", "epoca_ajuste", "rmse_val_densa"
]


def obter_cfg():
    """Obtém configuração possivelmente modificada por variável de ambiente."""
    max_ep = int(os.environ.get("TESTE_MAX_EPOCAS", CFG3.max_epochs))
    return replace(CFG3, max_epochs=max_ep)


# ---------------------------------------------------------------------------
# Backbones com fator de largura
# ---------------------------------------------------------------------------
class MLPBackbone(nn.Module):
    """MLP: janela achatada -> 2 camadas densas de largura `p` com ReLU."""
    def __init__(self, d, L, p=256):
        super().__init__()
        self.p = p
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(L * d, p),
            nn.ReLU(),
            nn.Linear(p, p),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class LSTMBackbone(nn.Module):
    """LSTM: 2 camadas de `oculto` unidades -> camada densa com ReLU."""
    def __init__(self, d, oculto=64):
        super().__init__()
        self.p = oculto
        self.lstm = nn.LSTM(d, oculto, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(nn.Linear(oculto, oculto), nn.ReLU())

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


def construir(backbone: str, cabeca: str, d: int, largura: int = 1,
              L: int = L3, H: int = H3, K: int = K3, q: int = Q3):
    """
    Constrói modelo = RevIN(backbone -> cabeça).
    
    largura: fator multiplicador de p (latent_dim) e tamanhos ocultos do backbone.
    """
    p_base_mlp = 256 * largura
    oculto_lstm = 64 * largura
    p_conv = 128 * largura
    hidden_conv = 64 * largura
    
    if backbone == "MLP":
        phi = MLPBackbone(d, L, p=p_base_mlp)
    elif backbone == "LSTM":
        phi = LSTMBackbone(d, oculto=oculto_lstm)
    elif backbone == "Conv1D+LSTM":
        phi = ConvLSTMBackbone(d, p_conv, hidden_conv)
        phi.p = p_conv
    else:
        raise ValueError(f"Backbone desconhecido: {backbone}")
    
    if cabeca == "densa":
        head = DenseHead(phi.p, H)
    elif cabeca == "nfh":
        head = NFHHead(phi.p, H, K, q, normalize=True)
    else:
        raise ValueError(f"Cabeça desconhecida: {cabeca}")
    
    return InstanceNorm(BackboneHeadModel(phi, head), target_idx=-1)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Métricas e avaliação
# ---------------------------------------------------------------------------
def metricas(P, Y):
    """RMSE e MAE."""
    e = P - Y
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))


@torch.no_grad()
def prever(modelo, X, dev, bs=1024):
    """Previsões em batch."""
    modelo.eval()
    prevs = []
    for i in range(0, len(X), bs):
        o = modelo(X[i:i + bs].to(dev))
        prevs.append(o["forecast"].cpu())
    return torch.cat(prevs).double().numpy()


def cobertura(P, K=10, q=Q):
    """Cobertura mediana kappa = 1 - eps_R."""
    proj = Projetor(H, K, q)
    partes = proj.partes(P)
    return float(np.median(1 - partes["eps_R"]))


# ---------------------------------------------------------------------------
# Variantes
# ---------------------------------------------------------------------------
def treinar_densa_e(modelo_denso, tr_ds, va_ds, seed, E=E_EXTRA, cfg=None, dev=None):
    """
    densa+E: continua treinando o modelo denso por E épocas com Adam lr=1e-4.
    Devolve o modelo com os pesos da melhor época de validação (0..E), onde 0 = denso original.
    """
    cfg = cfg or obter_cfg()
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    
    modelo = copy.deepcopy(modelo_denso).to(dev)
    Xv, Yv = stack_ds(va_ds)
    
    # Avalia época 0 (modelo original)
    rmse0 = avaliar(modelo, Xv, Yv, dev, H=H)[0]
    melhor_rmse = rmse0
    melhor_estado = copy.deepcopy(modelo.state_dict())
    melhor_epoca = 0
    
    g = torch.Generator()
    g.manual_seed(seed)
    tl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, generator=g)
    opt = torch.optim.Adam(modelo.parameters(), lr=LR_AJUSTE, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()
    
    t0 = time.time()
    for ep in range(1, E + 1):
        modelo.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            o = modelo(x)
            perda = mse(o["forecast"], y)
            perda.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), cfg.grad_clip)
            opt.step()
        
        rmse_val = avaliar(modelo, Xv, Yv, dev, H=H)[0]
        if rmse_val < melhor_rmse:
            melhor_rmse = rmse_val
            melhor_estado = copy.deepcopy(modelo.state_dict())
            melhor_epoca = ep
    
    modelo.load_state_dict(melhor_estado)
    modelo.eval()
    
    return modelo, {
        "rmse_val": melhor_rmse,
        "rmse_val_ep0": rmse0,
        "melhor_epoca": melhor_epoca,
        "tempo_s": time.time() - t0
    }


def projecao(P, K=10, q=Q):
    """Projeção: y_proj = y_densa - R (descarta resíduo)."""
    proj = Projetor(H, K, q)
    _, _, R = proj.decompor(P)
    return P - R


def verificar_conversao(modelo_denso, modelo_nfh, Xv, dev, n=512):
    """Verifica que a conversão densa->NFH é exata."""
    with torch.no_grad():
        b = Xv[:n].to(dev)
        y_d = modelo_denso(b)["forecast"]
        y_n = modelo_nfh(b)["forecast"]
        return float((y_d - y_n).abs().max())


# ---------------------------------------------------------------------------
# Gerenciamento de arquivos de resultado
# ---------------------------------------------------------------------------
def arquivo_csv(fase):
    return RES / f"{fase}.csv"


def feitos(fase):
    """Retorna conjunto de tarefas já concluídas."""
    f = arquivo_csv(fase)
    if not f.exists():
        return set()
    with open(f, "r", newline="") as fh:
        # Identificador único: (dataset, modo, backbone, largura, seed, variante, K, lambda)
        return {
            (r["dataset"], r["modo"], r["backbone"], r["largura"], r["seed"], r["variante"], r.get("K", ""), r.get("lambda", ""))
            for r in csv.DictReader(fh)
        }


def gravar(fase, linha):
    """Grava uma linha no CSV de resultados."""
    f = arquivo_csv(fase)
    novo = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if novo:
            w.writeheader()
        w.writerow({c: linha.get(c, "") for c in COLS})


def salvar_modelo(modelo, fase, dataset, backbone, largura, seed, variante):
    """Salva o estado do modelo."""
    dir_modelos = RES / "modelos"
    dir_modelos.mkdir(parents=True, exist_ok=True)
    nome = f"{dataset}_{backbone}_w{largura}_s{seed}_{variante}.pt"
    torch.save(modelo.state_dict(), dir_modelos / nome)


def salvar_previsoes(P, fase, dataset, backbone, largura, seed, variante):
    """Salva as previsões de teste."""
    dir_modelos = RES / "modelos"
    dir_modelos.mkdir(parents=True, exist_ok=True)
    nome = f"{dataset}_{backbone}_w{largura}_s{seed}_{variante}_pred.npy"
    np.save(dir_modelos / nome, P.astype(np.float32))


# ---------------------------------------------------------------------------
# Executar uma tarefa
# ---------------------------------------------------------------------------
def executar_tarefa(dataset, modo, backbone, largura, seed, fase, dev,
                    do_zero=False, frac_treino=None, verbose=True):
    """
    Executa uma tarefa completa: treina denso, gera todas as variantes.
    """
    cfg = obter_cfg()
    ja = feitos(fase)
    nome_conjunto = f"{dataset}-{modo}"
    
    # Identificador base da tarefa
    base_id = (dataset, modo, backbone, str(largura), str(seed))
    
    # Carrega dados
    try:
        ds = carregar(nome_conjunto)
    except Exception as e:
        print(f"Erro ao carregar {nome_conjunto}: {e}")
        return
    
    tr, va, te = segmentos(ds)
    d = ds["feats"].shape[1]
    
    # Se frac_treino especificada, usa subconjunto do treino
    if frac_treino and frac_treino < 1.0:
        n_tr = max(1, int(len(tr) * frac_treino))
        tr_original = tr
        # Cria um subset (hack simples: recria os índices)
        tr.idx = tr.idx[:n_tr]
    
    Xv, Yv = stack_ds(va)
    Xte, Yte = stack_ds(te)
    Y = Yte.double().numpy()
    
    info_base = {
        "dataset": dataset,
        "modo": modo,
        "backbone": backbone,
        "largura": str(largura),
        "seed": str(seed),
        "d": d,
        "n_janelas_treino": len(tr),
        "n_janelas_val": len(va),
        "n_janelas_teste": len(te),
    }
    
    # Verifica se todas as variantes já foram feitas
    variantes_obrigatorias = ["densa", "densa+E", "projecao"]
    variantes_obrigatorias += [f"nfh_L{lam}_K{K}" for lam in LAMBDAS for K in KS]
    variantes_obrigatorias += [f"nfh_L{lam}_K{K}_completo" for lam in LAMBDAS for K in KS]
    
    todas_feitas = all(
        (*base_id, v, "", "") in ja or (*base_id, v.split("_K")[0], v.split("_K")[1].replace("_completo", "") if "_K" in v else "", v.split("_L")[1].split("_")[0] if "_L" in v else "") in ja
        for v in variantes_obrigatorias
    )
    
    if verbose:
        print(f"\n[{time.strftime('%H:%M:%S')}] Tarefa: {dataset}-{modo} {backbone} largura={largura} seed={seed}")
    
    # === 1. Treina modelo denso ===
    if (*base_id, "densa", "", "") not in ja:
        set_seed(seed)
        modelo_denso = construir(backbone, "densa", d, largura)
        t0 = time.time()
        modelo_denso, info_treino = train_model(modelo_denso, tr, va, seed, lam=0.0, cfg=cfg, dev=dev)
        tempo_treino = time.time() - t0
        
        P_denso = prever(modelo_denso, Xte, dev)
        rmse_d, mae_d = metricas(P_denso, Y)
        cob_d = cobertura(P_denso, K=10)
        rmse_val_d = avaliar(modelo_denso, Xv, Yv, dev, H=H)[0]
        
        linha_d = {
            **info_base,
            "variante": "densa",
            "K": "",
            "lambda": "",
            "val_RMSE": rmse_val_d,
            "test_RMSE": rmse_d,
            "test_MAE": mae_d,
            "cobertura_mediana": cob_d,
            "epocas": info_treino["epochs"],
            "tempo_s": tempo_treino,
            "params": n_params(modelo_denso),
        }
        gravar(fase, linha_d)
        
        # Salva modelo e previsões (seed 0 na triagem, todos na confirmação)
        if fase != "confirmacao" and seed == 0:
            salvar_modelo(modelo_denso, fase, dataset, backbone, largura, seed, "densa")
            salvar_previsoes(P_denso, fase, dataset, backbone, largura, seed, "densa")
        elif fase == "confirmacao":
            salvar_modelo(modelo_denso, fase, dataset, backbone, largura, seed, "densa")
            salvar_previsoes(P_denso, fase, dataset, backbone, largura, seed, "densa")
        
        if verbose:
            print(f"  densa: RMSE={rmse_d:.4f} val={rmse_val_d:.4f} kappa={cob_d:.3f} ep={info_treino['epochs']}")
    else:
        # Carrega modelo denso existente
        modelo_denso = construir(backbone, "densa", d, largura)
        path_modelo = RES / "modelos" / f"{dataset}_{backbone}_w{largura}_s{seed}_densa.pt"
        if path_modelo.exists():
            modelo_denso.load_state_dict(torch.load(path_modelo, map_location=dev))
        else:
            # Retreina
            set_seed(seed)
            modelo_denso, _ = train_model(modelo_denso, tr, va, seed, lam=0.0, cfg=cfg, dev=dev)
        modelo_denso = modelo_denso.to(dev)
        P_denso = prever(modelo_denso, Xte, dev)
        rmse_val_d = avaliar(modelo_denso, Xv, Yv, dev, H=H)[0]
    
    # === 2. densa+E (controle de épocas) ===
    if (*base_id, "densa+E", "", "") not in ja:
        modelo_de, info_de = treinar_densa_e(modelo_denso, tr, va, seed, E=E_EXTRA, cfg=cfg, dev=dev)
        P_de = prever(modelo_de, Xte, dev)
        rmse_de, mae_de = metricas(P_de, Y)
        cob_de = cobertura(P_de, K=10)
        
        linha_de = {
            **info_base,
            "variante": "densa+E",
            "K": "",
            "lambda": "",
            "val_RMSE": info_de["rmse_val"],
            "test_RMSE": rmse_de,
            "test_MAE": mae_de,
            "cobertura_mediana": cob_de,
            "epocas": info_de["melhor_epoca"],
            "tempo_s": info_de["tempo_s"],
            "params": n_params(modelo_de),
            "rmse_val_densa": info_de["rmse_val_ep0"],
        }
        gravar(fase, linha_de)
        
        if verbose:
            print(f"  densa+E: RMSE={rmse_de:.4f} val={info_de['rmse_val']:.4f} ep_melhor={info_de['melhor_epoca']} (ep0={info_de['rmse_val_ep0']:.4f})")
    
    # === 3. Projeção (K=10) ===
    if (*base_id, "projecao", "10", "") not in ja:
        P_proj = projecao(P_denso, K=10)
        rmse_p, mae_p = metricas(P_proj, Y)
        cob_p = cobertura(P_proj, K=10)  # Por construção = 1.0
        
        linha_p = {
            **info_base,
            "variante": "projecao",
            "K": "10",
            "lambda": "",
            "val_RMSE": "",  # Não treina
            "test_RMSE": rmse_p,
            "test_MAE": mae_p,
            "cobertura_mediana": cob_p,
            "epocas": 0,
            "tempo_s": 0,
            "params": n_params(modelo_denso),
        }
        gravar(fase, linha_p)
        
        if verbose:
            print(f"  projecao K=10: RMSE={rmse_p:.4f} kappa={cob_p:.3f}")
    
    # === 4. NFH: conversão + ajuste estrutural ===
    # TESTE_SEM_AJUSTE=1 pula esta etapa (fases que só comparam densa, densa+E e NFH do zero)
    for lam in ([] if os.environ.get("TESTE_SEM_AJUSTE") == "1" else LAMBDAS):
        for K in KS:
            var_seguro = f"nfh_L{lam}_K{K}"
            var_completo = f"nfh_L{lam}_K{K}_completo"
            
            if (*base_id, var_seguro, str(K), str(lam)) in ja and (*base_id, var_completo, str(K), str(lam)) in ja:
                continue
            
            # Conversão exata
            nfh = densa_para_nfh(modelo_denso, K=K, q=Q, H=H)
            erro_conv = verificar_conversao(modelo_denso, nfh, Xv, dev)
            
            # Ajuste estrutural
            nfh_seguro, est_completo, info_aj = ajuste_estrutural(
                nfh, tr, va, seed, lam=lam, lr=LR_AJUSTE, epocas=E_EXTRA, cfg=cfg, dev=dev
            )
            
            # Variante seguro
            P_nfh = prever(nfh_seguro, Xte, dev)
            rmse_n, mae_n = metricas(P_nfh, Y)
            cob_n = cobertura(P_nfh, K=K)
            
            linha_nfh = {
                **info_base,
                "variante": var_seguro,
                "K": str(K),
                "lambda": str(lam),
                "val_RMSE": info_aj["seguro"]["rmse"],
                "test_RMSE": rmse_n,
                "test_MAE": mae_n,
                "cobertura_mediana": cob_n,
                "epocas": info_aj["seguro"]["epoca"],
                "tempo_s": info_aj["tempo_ajuste_s"],
                "params": n_params(nfh_seguro),
                "erro_conversao": erro_conv,
                "epoca_ajuste": info_aj["seguro"]["epoca"],
                "rmse_val_densa": info_aj["rmse_val_densa"],
            }
            gravar(fase, linha_nfh)
            
            # Salva modelo NFH
            if fase != "confirmacao" and seed == 0:
                salvar_modelo(nfh_seguro, fase, dataset, backbone, largura, seed, var_seguro)
                salvar_previsoes(P_nfh, fase, dataset, backbone, largura, seed, var_seguro)
            elif fase == "confirmacao":
                salvar_modelo(nfh_seguro, fase, dataset, backbone, largura, seed, var_seguro)
                salvar_previsoes(P_nfh, fase, dataset, backbone, largura, seed, var_seguro)
            
            # Variante completo
            nfh_seguro.load_state_dict(est_completo)
            P_nfh_c = prever(nfh_seguro, Xte, dev)
            rmse_nc, mae_nc = metricas(P_nfh_c, Y)
            cob_nc = cobertura(P_nfh_c, K=K)
            
            linha_nfh_c = {
                **info_base,
                "variante": var_completo,
                "K": str(K),
                "lambda": str(lam),
                "val_RMSE": info_aj["completo"]["rmse"],
                "test_RMSE": rmse_nc,
                "test_MAE": mae_nc,
                "cobertura_mediana": cob_nc,
                "epocas": info_aj["completo"]["epoca"],
                "tempo_s": info_aj["tempo_ajuste_s"],
                "params": n_params(nfh_seguro),
                "erro_conversao": erro_conv,
                "epoca_ajuste": info_aj["completo"]["epoca"],
                "rmse_val_densa": info_aj["rmse_val_densa"],
            }
            gravar(fase, linha_nfh_c)
            
            if verbose:
                print(f"  NFH λ={lam} K={K}: seguro RMSE={rmse_n:.4f} val={info_aj['seguro']['rmse']:.4f} kappa={cob_n:.3f} ep={info_aj['seguro']['epoca']} | completo RMSE={rmse_nc:.4f} | conv_err={erro_conv:.0e}")
    
    # === 5. NFH do zero (opcional) ===
    if do_zero:
        for lam in [int(x) for x in os.environ.get("TESTE_LAMBDAS_ZERO", "0,100").split(",")]:
            var = f"nfh_zero_L{lam}"
            if (*base_id, var, str(K_DEFAULT), str(lam)) in ja:
                continue
            
            set_seed(seed)
            nfh_zero = construir(backbone, "nfh", d, largura, K=K_DEFAULT)
            t0 = time.time()
            nfh_zero, info_z = train_model(nfh_zero, tr, va, seed, lam=float(lam), cfg=cfg, dev=dev)
            tempo_z = time.time() - t0
            
            P_z = prever(nfh_zero, Xte, dev)
            rmse_z, mae_z = metricas(P_z, Y)
            cob_z = cobertura(P_z, K=K_DEFAULT)
            rmse_val_z = avaliar(nfh_zero, Xv, Yv, dev, H=H)[0]
            
            linha_z = {
                **info_base,
                "variante": var,
                "K": str(K_DEFAULT),
                "lambda": str(lam),
                "val_RMSE": rmse_val_z,
                "test_RMSE": rmse_z,
                "test_MAE": mae_z,
                "cobertura_mediana": cob_z,
                "epocas": info_z["epochs"],
                "tempo_s": tempo_z,
                "params": n_params(nfh_zero),
            }
            gravar(fase, linha_z)
            
            if verbose:
                print(f"  NFH do zero λ={lam}: RMSE={rmse_z:.4f} val={rmse_val_z:.4f} kappa={cob_z:.3f}")
    
    # === 6. Previsões ingênuas (uma vez por dataset) ===
    for nome_ing, P_ing in ingenuas(Xte, H).items():
        var_ing = nome_ing.replace(" ", "_").replace("(", "").replace(")", "")
        if (dataset, modo, "", "", "-1", var_ing, "", "") in ja:
            continue
        rmse_i, mae_i = metricas(P_ing, Y)
        linha_i = {
            "dataset": dataset,
            "modo": modo,
            "backbone": "",
            "largura": "",
            "seed": "-1",
            "variante": var_ing,
            "K": "",
            "lambda": "",
            "val_RMSE": "",
            "test_RMSE": rmse_i,
            "test_MAE": mae_i,
            "cobertura_mediana": "",
            "epocas": 0,
            "tempo_s": 0,
            "params": 0,
            "n_janelas_teste": len(te),
            "d": d,
        }
        gravar(fase, linha_i)
        if verbose:
            print(f"  {nome_ing}: RMSE={rmse_i:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Executor de tarefas para teste NFH")
    parser.add_argument("--fase", required=True, help="nome da fase = nome do CSV em resultados/ (triagem, confirmacao, zero, zero25, ...)")
    parser.add_argument("--conjuntos", nargs="+", required=True,
                        help="Conjuntos no formato Dataset-U ou Dataset-M")
    parser.add_argument("--backbones", nargs="+", default=["MLP", "LSTM", "Conv1D+LSTM"])
    parser.add_argument("--larguras", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--do-zero", action="store_true", help="Também treina NFH do zero")
    parser.add_argument("--frac-treino", type=float, default=None,
                        help="Fração do treino a usar (ex: 0.25)")
    parser.add_argument("--verbose", action="store_true", default=True)
    
    args = parser.parse_args()
    
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = True
    dev = torch.device(args.device)
    
    print(f"Fase: {args.fase}")
    print(f"Conjuntos: {args.conjuntos}")
    print(f"Backbones: {args.backbones}")
    print(f"Larguras: {args.larguras}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {dev}")
    print(f"Max epochs: {os.environ.get('TESTE_MAX_EPOCAS', CFG3.max_epochs)}")
    print()
    
    for conj in args.conjuntos:
        partes = conj.rsplit("-", 1)
        if len(partes) != 2:
            print(f"Formato inválido: {conj} (esperado Dataset-U ou Dataset-M)")
            continue
        dataset, modo = partes
        
        for backbone in args.backbones:
            for largura in args.larguras:
                for seed in args.seeds:
                    try:
                        executar_tarefa(
                            dataset, modo, backbone, largura, seed, args.fase, dev,
                            do_zero=args.do_zero, frac_treino=args.frac_treino,
                            verbose=args.verbose
                        )
                    except Exception as e:
                        print(f"ERRO em {dataset}-{modo} {backbone} w{largura} s{seed}: {e}")
                        import traceback
                        traceback.print_exc()


if __name__ == "__main__":
    main()
