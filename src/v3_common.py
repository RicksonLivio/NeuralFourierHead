"""
Núcleo comum dos experimentos da versão 3 da dissertação (correções da qualificação).

O que muda em relação à v2 (motivado pelos áudios e pelo PDF anotado do orientador):
  * Conjuntos públicos com ciclo diário conhecido, escolhidos por critérios fixados
    antes de qualquer resultado (relatório em .kiro/specs/versao3-correcao2/
    referencias/datasets_pesquisa.md): ETTh1, ETTh2, Jena Climate 2009-2011 (horário)
    e UCI Air Quality, cada um nas configurações U (só o alvo) e M (alvo + covariáveis).
  * Todos os backbones (MLP, LSTM, Conv1D+LSTM) separados em backbone + camada de
    saída, para que cada um seja avaliado com camada densa e com a NFH.
  * NFH obtida por CONVERSÃO EXATA da camada densa treinada (Teorema de equivalência)
    seguida de AJUSTE ESTRUTURAL (fine-tuning com a penalização do resíduo). O passo 0
    do ajuste é, por construção, a decomposição pura da previsão do modelo denso.
    Seleção do passo: maior cobertura estrutural de validação entre os passos cujo RMSE
    de validação não excede o do modelo denso (garantia de não inferioridade na
    validação, Proposição do Cap. 4).

Reaproveita de v2_common: bases (nfh_bases), NFHHead, ConvLSTMBackbone, InstanceNorm,
NBeatsI, train_model, energy_stats, harmonic_shares, Config.
"""
from __future__ import annotations

import copy
import io
import time
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from v2_common import (CONFIG, BackboneHeadModel, ConvLSTMBackbone, DenseHead, InstanceNorm,  # noqa: F401
                       NBeatsI, NFHHead, device, nfh_bases, residual_penalty, save_json, set_seed,
                       stack_ds, train_model)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "v3_bruto"
RES3 = ROOT / "results" / "v3"
RES3.mkdir(parents=True, exist_ok=True)

L3, H3, K3, Q3 = 96, 48, 10, 2          # janela, horizonte, harmônicos, grau da tendência
CFG3 = replace(CONFIG, lookback=L3, horizon=H3, n_harmonics=K3, trend_degree=Q3)
FONTES = ["ETTh1", "ETTh2", "Jena", "AirQ"]
CONJUNTOS = [f"{f}-{s}" for f in FONTES for s in ("U", "M")]
BACKBONES = ["MLP", "LSTM", "Conv1D+LSTM"]
SEEDS = (0, 1, 2, 3, 4)
LAMBDA_AJUSTE = 1000.0                  # o lambda* selecionado na v2
LR_AJUSTE = 1e-4
EPOCAS_AJUSTE = 15


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def _ett(nome: str):
    caminho = DATA / "controle" / "ETTh1.csv" if nome == "ETTh1" else RAW / "ETTh2.csv"
    df = pd.read_csv(caminho, parse_dates=["date"])
    cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    df = df.iloc[:8640 + 2880 + 2880]            # divisão padrão 12/4/4 meses
    return df["date"], df[cols].astype(float), "OT", (8640, 8640 + 2880)


def _jena():
    with zipfile.ZipFile(RAW / "jena_climate_2009_2016.csv.zip") as z:
        with z.open("jena_climate_2009_2016.csv") as f:
            df = pd.read_csv(f)
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.sort_values("Date Time").drop_duplicates(subset="Date Time", keep="first")
    wv = df["wv (m/s)"].where(df["wv (m/s)"] > -9000, 0.0)
    ang = np.deg2rad(df["wd (deg)"])
    df["Wx"], df["Wy"] = wv * np.cos(ang), wv * np.sin(ang)
    df = df.set_index("Date Time")[["T (degC)", "p (mbar)", "rh (%)", "Wx", "Wy"]]
    h = df.resample("1h", closed="right", label="right").mean()
    h = h.loc["2009-01-01 01:00":"2012-01-01 00:00"]
    h.columns = ["T", "p", "rh", "Wx", "Wy"]
    h = h[["p", "rh", "Wx", "Wy", "T"]]          # alvo por último
    n = len(h)
    return pd.Series(h.index), h, "T", (int(0.70 * n), int(0.85 * n))


def _airq():
    with zipfile.ZipFile(RAW / "air+quality.zip") as z:
        with z.open("AirQualityUCI.csv") as f:
            txt = f.read().decode("latin-1")
    df = pd.read_csv(io.StringIO(txt), sep=";", decimal=",")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].dropna(subset=["Date"])
    t = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S")
    cols = ["PT08.S1(CO)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH", "C6H6(GT)"]
    x = df[cols].astype(float).where(lambda v: v > -199, np.nan)
    x.columns = ["S1_CO", "S3_NOx", "S4_NO2", "S5_O3", "T", "RH", "AH", "C6H6"]
    n = len(x)
    return t.reset_index(drop=True), x.reset_index(drop=True), "C6H6", (int(0.70 * n), int(0.85 * n))


def carregar(nome: str):
    """Conjunto 'Fonte-U' ou 'Fonte-M'. Retorna dict com feats [N,d], alvo [N] (padronizados
    com estatísticas do treino), nomes, máscara de alvo observado, cortes e datas."""
    fonte, modo = nome.split("-")
    datas, df, alvo, (a, b) = {"ETTh1": lambda: _ett("ETTh1"), "ETTh2": lambda: _ett("ETTh2"),
                               "Jena": _jena, "AirQ": _airq}[fonte]()
    observado = df[alvo].notna().values
    # Entradas: lacunas preenchidas por interpolação temporal (só para montar as janelas);
    # janelas cujo horizonte contém alvo originalmente ausente são descartadas (MascWindows).
    df = df.interpolate(limit_direction="both")
    mu, sd = df.iloc[:a].mean(), df.iloc[:a].std() + 1e-12
    z = ((df - mu) / sd).astype(np.float32)
    alvo_z = z[alvo].values
    if modo == "U":
        feats, nomes = alvo_z.reshape(-1, 1), [alvo]
    else:
        nomes = list(z.columns)
        feats = z.values
    return {"nome": nome, "fonte": fonte, "modo": modo, "feats": np.ascontiguousarray(feats),
            "alvo": np.ascontiguousarray(alvo_z), "nomes": nomes, "observado": observado,
            "cortes": (a, b), "datas": datas.values, "mu_alvo": float(mu[alvo]), "sd_alvo": float(sd[alvo])}


class MascWindows(Dataset):
    """Janelas (x_{i..i+L-1}, y_{i+L..i+L+H-1}) dentro de um segmento, mantendo apenas as
    janelas cujo horizonte tem o alvo originalmente observado."""

    def __init__(self, feats, alvo, observado, L, H):
        self.f = torch.from_numpy(np.ascontiguousarray(feats))
        self.t = torch.from_numpy(np.ascontiguousarray(alvo))
        self.L, self.H = L, H
        n = len(feats) - L - H + 1
        obs = np.asarray(observado, dtype=bool)
        # Exige também o último passo da entrada observado: se ele estivesse numa lacuna, seu valor
        # interpolado usaria a primeira observação após a lacuna, que é o início do horizonte (vazamento).
        # Nos conjuntos sem faltantes (ETT, Jena) a máscara é toda verdadeira e nada muda.
        self.idx = [i for i in range(max(0, n)) if obs[i + L:i + L + H].all() and obs[i + L - 1]]
        self.n_total = max(0, n)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        i = self.idx[j]
        return self.f[i:i + self.L], self.t[i + self.L:i + self.L + self.H]


def segmentos(ds: dict, L=L3, H=H3):
    a, b = ds["cortes"]
    f, t, o = ds["feats"], ds["alvo"], ds["observado"]
    return (MascWindows(f[:a], t[:a], o[:a], L, H), MascWindows(f[a:b], t[a:b], o[a:b], L, H),
            MascWindows(f[b:], t[b:], o[b:], L, H))


# ---------------------------------------------------------------------------
# Backbones (mesmas arquiteturas da v2, separadas da camada de saída)
# ---------------------------------------------------------------------------
class MLPBackbone(nn.Module):
    """Janela achatada -> 2 camadas densas de largura 256 com ReLU; h in R^256."""

    def __init__(self, d, L, largura=256):
        super().__init__()
        self.p = largura
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(L * d, largura), nn.ReLU(),
                                 nn.Linear(largura, largura), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class LSTMBackbone(nn.Module):
    """LSTM de 2 camadas (64 unidades) -> camada densa com ReLU; h in R^64."""

    def __init__(self, d, oculto=64):
        super().__init__()
        self.p = oculto
        self.lstm = nn.LSTM(d, oculto, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(nn.Linear(oculto, oculto), nn.ReLU())

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


def construir(backbone: str, cabeca: str, d: int, L=L3, H=H3, K=K3, q=Q3):
    """Modelo = RevIN( Phi -> cabeça ), cabeça em {'densa', 'nfh'} (NFH com colunas normalizadas)."""
    if backbone == "MLP":
        phi = MLPBackbone(d, L)
    elif backbone == "LSTM":
        phi = LSTMBackbone(d)
    elif backbone == "Conv1D+LSTM":
        phi = ConvLSTMBackbone(d, 128, 64)
        phi.p = 128
    else:
        raise ValueError(backbone)
    head = DenseHead(phi.p, H) if cabeca == "densa" else NFHHead(phi.p, H, K, q, normalize=True)
    return InstanceNorm(BackboneHeadModel(phi, head), target_idx=-1)


def matriz_sintese(H=H3, K=K3, q=Q3):
    """M = [P^T  Ft^T  Q_perp] (H x H), em float64."""
    P, F, Ft, B, Qp = nfh_bases(H, K, q)
    return np.concatenate([P.T, Ft.T, Qp], 1), P, Ft, Qp


def densa_para_nfh(modelo_denso, K=K3, q=Q3, H=H3):
    """Conversão EXATA (Teorema de equivalência): W_NFH = (M D)^{-1} W', b_NFH = (M D)^{-1} b'.
    O modelo resultante produz a mesma previsão que o denso, agora com coordenadas (theta_T, theta_S, u)."""
    interno = modelo_denso.inner
    lin = interno.head.lin
    head = NFHHead(lin.in_features, H, K, q, normalize=True)
    M, *_ = matriz_sintese(H, K, q)
    MD = M * head.D.cpu().numpy().astype(np.float64)[None, :]
    W = lin.weight.detach().cpu().double().numpy()
    b = lin.bias.detach().cpu().double().numpy()
    head.lin.weight.data = torch.tensor(np.linalg.solve(MD, W), dtype=torch.float32)
    head.lin.bias.data = torch.tensor(np.linalg.solve(MD, b), dtype=torch.float32)
    novo = InstanceNorm(BackboneHeadModel(copy.deepcopy(interno.backbone), head),
                        target_idx=modelo_denso.target_idx)
    return novo.to(next(modelo_denso.parameters()).device)


# ---------------------------------------------------------------------------
# Decomposição de previsões quaisquer (projeção; Teorema de identificabilidade)
# ---------------------------------------------------------------------------
class Projetor:
    """Projeções ortogonais sobre T, S e R para decompor QUALQUER previsão y_hat in R^H."""

    def __init__(self, H=H3, K=K3, q=Q3):
        P, F, Ft, B, Qp = nfh_bases(H, K, q)
        self.PiT = P.T @ np.linalg.solve(P @ P.T, P)
        self.PiR = Qp @ Qp.T
        self.Ft = Ft
        self.thS = np.linalg.solve(Ft @ Ft.T, Ft)     # theta_S = (Ft Ft^T)^{-1} Ft y_hat
        self.K = K

    def decompor(self, Y: np.ndarray):
        Y = np.asarray(Y, dtype=np.float64)
        T = Y @ self.PiT.T
        R = Y @ self.PiR.T
        S = Y - T - R
        return T, S, R

    def partes(self, Y: np.ndarray):
        """Participações de energia da previsão centrada (Teorema do balanço de energia)."""
        T, S, R = self.decompor(Y)
        Yc = Y - Y.mean(1, keepdims=True)
        Tc = T - T.mean(1, keepdims=True)
        E = (Yc ** 2).sum(1) + 1e-30
        return {"eps_T": (Tc ** 2).sum(1) / E, "eps_S": (S ** 2).sum(1) / E, "eps_R": (R ** 2).sum(1) / E,
                "escala": np.sqrt(E / Y.shape[1])}


def partes_nativas(T, S, Y):
    """Partes aditivas por projeção, rho_c = <C_c, y_c>/||y_c||^2 (somam 1 quando T + S = y);
    para a NFH coincidem com eps_T e eps_S; para o N-BEATS-I podem sair de [0, 1]."""
    Y = np.asarray(Y, dtype=np.float64)
    Yc = Y - Y.mean(1, keepdims=True)
    E = (Yc ** 2).sum(1) + 1e-30
    Tc = T - T.mean(1, keepdims=True)
    Sc = S - S.mean(1, keepdims=True)
    return {"rho_T": (Tc * Yc).sum(1) / E, "rho_S": (Sc * Yc).sum(1) / E}


# ---------------------------------------------------------------------------
# Ajuste estrutural com seleção sob não inferioridade na validação
# ---------------------------------------------------------------------------
@torch.no_grad()
def avaliar(modelo, X, Y, dev, bs=1024, H=H3):
    """RMSE, cobertura mediana kappa = 1 - ||R||^2/||y_c||^2, rho = media(||R||^2)/H e a previsão.
    Tudo na escala desnormalizada (saída do RevIN), a mesma da penalização do treino."""
    modelo.eval()
    prev, res = [], []
    for i in range(0, len(X), bs):
        o = modelo(X[i:i + bs].to(dev))
        prev.append(o["forecast"].detach().cpu())
        if "residual" in o:
            res.append(o["residual"].detach().cpu())
    P = torch.cat(prev).double().numpy()
    rmse = float(np.sqrt(np.mean((P - Y.double().numpy()) ** 2)))
    Pc = P - P.mean(1, keepdims=True)
    if res:
        R = torch.cat(res).double().numpy()
        cob = 1 - (R ** 2).sum(1) / ((Pc ** 2).sum(1) + 1e-30)
        rho = float((R ** 2).sum(1).mean() / H)
    else:
        cob, rho = np.full(len(P), np.nan), float("nan")
    return rmse, float(np.median(cob)), P, rho


def ajuste_estrutural(modelo, tr_ds, va_ds, seed, lam=LAMBDA_AJUSTE, lr=LR_AJUSTE, epocas=EPOCAS_AJUSTE,
                      cfg=CFG3, dev=None):
    """Ajuste estrutural a partir da NFH convertida (passo 0 = decomposição pura da previsão densa).

    Treina o modelo inteiro com a perda MSE + lam ||R||^2 / H e, a cada época, mede na validação o
    RMSE, a cobertura kappa e o objetivo J = MSE + lam rho (o mesmo critério de parada de todos os
    treinamentos). Devolve dois modelos:
      * 'seguro': menor J entre as épocas com RMSE_val <= RMSE_val(passo 0)  (o passo 0 é viável,
        logo RMSE_val(seguro) <= RMSE_val(densa) sempre);
      * 'completo': menor J sem restrição (em geral kappa ~ 1), para comparar com a projeção.
    """
    dev = dev or device()
    set_seed(seed)
    modelo = modelo.to(dev)
    Xv, Yv = stack_ds(va_ds)
    r0, k0, _, rho0 = avaliar(modelo, Xv, Yv, dev, H=cfg.horizon)
    j0 = r0 ** 2 + lam * rho0
    seguro = {"epoca": 0, "rmse": r0, "kappa": k0, "J": j0, "estado": copy.deepcopy(modelo.state_dict())}
    completo = dict(seguro)
    traj = [(0, r0, k0, j0)]
    g = torch.Generator()
    g.manual_seed(seed)
    tl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, generator=g)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()
    t0 = time.time()
    for ep in range(1, epocas + 1):
        modelo.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            o = modelo(x)
            perda = mse(o["forecast"], y) + lam * residual_penalty(o, cfg.horizon)
            perda.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), cfg.grad_clip)
            opt.step()
        r, k, _, rho = avaliar(modelo, Xv, Yv, dev, H=cfg.horizon)
        J = r ** 2 + lam * rho
        traj.append((ep, r, k, J))
        if J < completo["J"]:
            completo = {"epoca": ep, "rmse": r, "kappa": k, "J": J, "estado": copy.deepcopy(modelo.state_dict())}
        if r <= r0 and J < seguro["J"]:
            seguro = {"epoca": ep, "rmse": r, "kappa": k, "J": J, "estado": copy.deepcopy(modelo.state_dict())}
    est_completo = completo.pop("estado")
    modelo.load_state_dict(seguro.pop("estado"))
    modelo.eval()
    return modelo, est_completo, {"seguro": seguro, "completo": completo, "rmse_val_densa": r0,
                                  "kappa_val_densa": k0, "trajetoria": traj, "tempo_ajuste_s": time.time() - t0}


# ---------------------------------------------------------------------------
# Previsões ingênuas
# ---------------------------------------------------------------------------
def ingenuas(X: torch.Tensor, H=H3, periodo=24):
    """Média de treino (= 0 nos dados padronizados), último valor e sazonal de 24 h."""
    y = X[:, :, -1].double().numpy()
    n = len(y)
    media = np.zeros((n, H))
    ultimo = np.repeat(y[:, -1:], H, 1)
    ciclo = y[:, -periodo:]
    sazonal = np.concatenate([ciclo] * int(np.ceil(H / periodo)), 1)[:, :H]
    return {"Ingênuo (média de treino)": media, "Ingênuo (último valor)": ultimo,
            "Ingênuo sazonal (24 h)": sazonal}
