"""
Núcleo comum dos experimentos da versão 2 da dissertação (NFH).

Tudo que é número na v2 sai de scripts que importam este módulo, com
configuração única (CONFIG) e sementes registradas.

Principais diferenças em relação à v1 (corrigidas):
  * Configuração L=60, H=30, K=10, q=2 nos conjuntos financeiros e sintéticos
    e L=96, H=48 no ETTh1 (sempre 2K < H e q+1+2K <= H, logo a base
    B = [P^T F^T] tem posto completo de colunas: condição de Nyquist).
  * NFH com resíduo LINEAR e ORTOGONAL ao espaço da base (R = Q_perp u) e base
    sazonal ortogonalizada contra a tendência (Ft = F (I - Pi_T)): a
    decomposição é identificável e a head é uma reparametrização invertível
    de uma camada densa (mesmo número de parâmetros).
  * N-BEATS-I fiel ao artigo original: 2 pilhas (tendência, sazonalidade),
    3 blocos por pilha com pesos compartilhados, 4 camadas FC, backcast
    duplamente residual e base sazonal com harmônicos até floor(H/2-1).
  * Normalização reversível por instância (RevIN) aplicada igualmente a todos
    os modelos; normalização global apenas com estatísticas do treino.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results" / "v2"
RES.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    lookback: int = 60          # L
    horizon: int = 30           # H
    n_harmonics: int = 10       # K  (2K < H)
    trend_degree: int = 2       # q
    latent_dim: int = 128       # p
    hidden: int = 64            # canais conv / unidades LSTM
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 60
    patience: int = 10
    grad_clip: float = 1.0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seeds: tuple = (0, 1, 2, 3, 4)


CONFIG = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
UNI = ["BNB_USD", "BTC_USD", "ETH_USD", "SOL_USD", "XRP_USD"]
MULTI = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
DATASETS = [(n, False) for n in UNI] + [(n, True) for n in MULTI]


def load_series(name: str, multivariate: bool, cfg: Config = CONFIG, columns=None,
                target_kind: str = "return"):
    """Retorna (features [N, d], target [N], nomes) SEM vazamento.

    target_kind = "return":    alvo r_t = ln(P_t / P_{t-1});
    target_kind = "absreturn": alvo |r_t| (proxy de volatilidade realizada).
    O alvo é padronizado com média e desvio estimados apenas no trecho de
    treino. Features multivariadas também padronizadas apenas com estatísticas
    do treino. No caso univariado, a única variável de entrada é a própria
    série-alvo (padronizada).
    """
    sub = "multivariate" if multivariate else "univariate"
    df = pd.read_csv(DATA / sub / f"{name}.csv")
    close = df["close"].values.astype(float)
    ret = np.diff(np.log(close))
    if target_kind == "absreturn":
        ret = np.abs(ret)
    elif target_kind != "return":
        raise ValueError(target_kind)
    n = len(ret)
    tr_end = int(n * cfg.train_ratio)
    mu, sd = ret[:tr_end].mean(), ret[:tr_end].std() + 1e-12
    target = (ret - mu) / sd
    if multivariate:
        cols = columns or [c for c in df.columns if c != "date"]
        raw = df[cols].values.astype(float)[1:]
        fm = np.nanmean(raw[:tr_end], axis=0)
        fs = np.nanstd(raw[:tr_end], axis=0) + 1e-12
        feats = np.nan_to_num((raw - fm) / fs)
        names = cols
    else:
        feats = target.reshape(-1, 1).copy()
        names = ["log_return"]
    return feats.astype(np.float32), target.astype(np.float32), names


def load_etth1(multivariate: bool):
    """ETTh1 (ZHOU et al., 2021): alvo OT (temperatura do óleo), divisão padrão
    12/4/4 meses (8640/2880/2880 horas). Padronização com estatísticas do treino."""
    df = pd.read_csv(DATA / "controle" / "ETTh1.csv")
    cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    raw = df[cols].values.astype(float)[:8640 + 2880 + 2880]
    tr = raw[:8640]
    z = (raw - tr.mean(0)) / (tr.std(0) + 1e-12)
    target = z[:, -1]
    feats = z if multivariate else target.reshape(-1, 1)
    names = cols if multivariate else ["OT"]
    return feats.astype(np.float32), target.astype(np.float32), names


def synthetic_series(seed: int, n: int = 3000):
    """Série sintética com componentes conhecidos (verdade de referência).

    y_t = mu_t + s_t + e_t, com mu_t linear por partes (tendência), s_t soma de
    dois ciclos de períodos 10 e 6 (harmônicos k=3 e k=5 para H=30) com
    amplitudes lentamente variáveis e e_t ruído AR(1).
    Retorna y padronizada e os componentes padronizados (mesma escala).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    # tendência linear por partes, contínua
    knots = np.arange(0, n + 250, 250)
    slopes = rng.normal(0, 0.004, len(knots))
    mu = np.zeros(n)
    for i in range(1, n):
        mu[i] = mu[i - 1] + slopes[int(t[i] // 250)]
    A1 = 1.0 + 0.3 * np.sin(2 * np.pi * t / 700)
    A2 = 0.6 + 0.2 * np.cos(2 * np.pi * t / 900)
    ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
    s = A1 * np.sin(2 * np.pi * t / 10 + ph1) + A2 * np.cos(2 * np.pi * t / 6 + ph2)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.5 * e[i - 1] + rng.normal(0, 0.3)
    y = mu + s + e
    tr_end = int(n * CONFIG.train_ratio)
    m, sd = y[:tr_end].mean(), y[:tr_end].std()
    return {"y": ((y - m) / sd).astype(np.float32), "trend": ((mu - m) / sd).astype(np.float32),
            "seasonal": (s / sd).astype(np.float32), "noise": (e / sd).astype(np.float32)}


def dataset_spec(name: str):
    """Especificação (L, H, tipo) por conjunto de dados."""
    if name.startswith("ETTh1"):
        return {"L": 96, "H": 48, "group": "controle"}
    if name.startswith("SINT"):
        return {"L": 60, "H": 30, "group": "sintetico"}
    return {"L": 60, "H": 30, "group": "multivariado" if name in MULTI else "univariado"}


def load_any(name: str):
    """Carrega (feats, target, nomes, extras) para qualquer conjunto do estudo."""
    if name == "ETTh1-U":
        f, t, n = load_etth1(False)
        return f, t, n, {}
    if name == "ETTh1-M":
        f, t, n = load_etth1(True)
        return f, t, n, {}
    if name.startswith("SINT"):
        seed = 100 + int(name.split("-")[1])
        s = synthetic_series(seed)
        return s["y"].reshape(-1, 1), s["y"], ["y"], s
    f, t, n = load_series(name, name in MULTI)
    return f, t, n, {}


class WindowDataset(Dataset):
    """Janelas (x_{t-L+1..t}, y_{t+1..t+H}) inteiramente dentro de um segmento."""

    def __init__(self, feats, target, L, H):
        self.f = torch.from_numpy(np.ascontiguousarray(feats))
        self.t = torch.from_numpy(np.ascontiguousarray(target))
        self.L, self.H = L, H
        self.n = max(0, len(feats) - L - H + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.f[i:i + self.L], self.t[i + self.L:i + self.L + self.H]


def split_datasets(feats, target, cfg: Config = CONFIG, name: str | None = None):
    n = len(feats)
    if name is not None and name.startswith("ETTh1"):
        a, b = 8640, 8640 + 2880
    else:
        a, b = int(n * cfg.train_ratio), int(n * (cfg.train_ratio + cfg.val_ratio))
    L, H = cfg.lookback, cfg.horizon
    return (WindowDataset(feats[:a], target[:a], L, H),
            WindowDataset(feats[a:b], target[a:b], L, H),
            WindowDataset(feats[b:], target[b:], L, H))


def stack_ds(ds):
    xs, ys = zip(*[ds[i] for i in range(len(ds))])
    return torch.stack(xs), torch.stack(ys)


# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------
def time_grid(H: int) -> np.ndarray:
    """Grade normalizada tau_j = j/H, j = 0..H-1 (a mesma do N-BEATS)."""
    return np.arange(H, dtype=np.float64) / H


def poly_basis(H: int, q: int) -> np.ndarray:
    """P in R^{(q+1) x H}, linhas tau^i."""
    tau = time_grid(H)
    return np.stack([tau ** i for i in range(q + 1)], 0)


def fourier_basis(H: int, K: int) -> np.ndarray:
    """F in R^{2K x H}, linhas intercaladas cos(2 pi k tau), sin(2 pi k tau), k=1..K."""
    tau = time_grid(H)
    rows = []
    for k in range(1, K + 1):
        rows.append(np.cos(2 * np.pi * k * tau))
        rows.append(np.sin(2 * np.pi * k * tau))
    return np.stack(rows, 0)


def nfh_bases(H: int, K: int, q: int):
    """Bases da NFH.

    P  in R^{(q+1) x H}: polinômios tau^i (tendência);
    F  in R^{2K x H}   : senoides cos/sin de frequência k/H, k=1..K;
    Ft in R^{2K x H}   : senoides com a parte polinomial removida,
                         Ft = F (I - Pi_T), Pi_T = P^T (P P^T)^{-1} P;
    B  = [P^T F^T] (H x m), m = q+1+2K, com posto completo (Lema de independência);
    Qp in R^{H x (H-m)}: base ortonormal do complemento ortogonal de col(B).
    """
    if not 2 * K < H:
        raise ValueError(f"Nyquist violado: 2K={2*K} >= H={H}")
    P, F = poly_basis(H, q), fourier_basis(H, K)
    B = np.concatenate([P, F], 0).T
    m = B.shape[1]
    if m > H:
        raise ValueError("q+1+2K > H")
    rank = np.linalg.matrix_rank(B)
    if rank != m:
        raise ValueError(f"B sem posto completo: rank={rank} < m={m}")
    Pi_T = P.T @ np.linalg.solve(P @ P.T, P)
    Ft = F @ (np.eye(H) - Pi_T)
    Q, _ = np.linalg.qr(B, mode="complete")
    Qp = Q[:, m:]
    return P, F, Ft, B, Qp


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------
class NFHHead(nn.Module):
    """Neural Fourier Head (v2).

    c = W h + b in R^H, particionado em  theta_T (q+1), theta_S (2K), u (H-m).
    T = P^T theta_T,  S = Ft^T theta_S,  R = Q_perp u,  y_hat = T + S + R,
    com T, S e R mutuamente ortogonais (Teorema do balanço de energia).
    Número de parâmetros = H (p + 1), idêntico a nn.Linear(p, H).
    """

    def __init__(self, p: int, H: int, K: int, q: int, normalize: bool = False):
        super().__init__()
        P, F, Ft, B, Qp = nfh_bases(H, K, q)
        self.H, self.K, self.q = H, K, q
        self.m = B.shape[1]
        self.register_buffer("P", torch.tensor(P, dtype=torch.float32))
        self.register_buffer("F", torch.tensor(F, dtype=torch.float32))
        self.register_buffer("Ft", torch.tensor(Ft, dtype=torch.float32))
        self.register_buffer("Qp", torch.tensor(Qp, dtype=torch.float32))
        self.lin = nn.Linear(p, H)
        # Variante "NFHn" (ablação de otimização): c = D c', com D diagonal fixa
        # que dá norma unitária às colunas de M = [P^T Ft^T Q_perp]. A função
        # representada e a explicação (coordenadas c) são as mesmas; muda apenas
        # a escala efetiva do passo do Adam em cada direção de saída.
        self.normalize = normalize
        if normalize:
            norms = np.concatenate([np.linalg.norm(P, axis=1), np.linalg.norm(Ft, axis=1),
                                    np.ones(H - self.m)])
            self.register_buffer("D", torch.tensor(1.0 / norms, dtype=torch.float32))

    def forward(self, h):
        c = self.lin(h)
        if self.normalize:
            c = c * self.D
        q1 = self.q + 1
        th_T, th_S, u = c[:, :q1], c[:, q1:self.m], c[:, self.m:]
        T = th_T @ self.P
        S = th_S @ self.Ft
        R = u @ self.Qp.T
        A = th_S[:, 0::2]
        Bc = th_S[:, 1::2]
        return {"forecast": T + S + R, "trend": T, "seasonal": S, "residual": R,
                "theta_T": th_T, "theta_S": th_S, "u": u,
                "amplitude": torch.sqrt(A ** 2 + Bc ** 2), "phase": torch.atan2(Bc, A)}


class DenseHead(nn.Module):
    def __init__(self, p: int, H: int):
        super().__init__()
        self.lin = nn.Linear(p, H)

    def forward(self, h):
        return {"forecast": self.lin(h)}


# ---------------------------------------------------------------------------
# Backbones e modelos
# ---------------------------------------------------------------------------
class ConvLSTMBackbone(nn.Module):
    """Phi: R^{L x d} -> R^p.  Conv1D(k=7) -> Conv1D(k=5) -> LSTM(2) -> Linear+LN+ReLU."""

    def __init__(self, d: int, p: int = 128, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d, hidden, 7, padding=3), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.proj = nn.Sequential(nn.Linear(hidden, p), nn.LayerNorm(p), nn.ReLU())

    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        _, (hn, _) = self.lstm(z)
        return self.proj(hn[-1])


class BackboneHeadModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, x):
        h = self.backbone(x)
        out = self.head(h)
        out["latent"] = h
        return out


class InstanceNorm(nn.Module):
    """Normalização reversível por instância (inspirada em RevIN, KIM et al., 2022).

    Cada janela de entrada é padronizada por variável com média e desvio da
    própria janela; a previsão do modelo interno é desnormalizada com a média
    (mu_y) e o desvio (sigma_y) da variável-alvo na janela:
        y_hat = sigma_y * y_hat_n + mu_y.
    Para a NFH, a transformação preserva a decomposição (Observação no Cap. 4):
        T = sigma_y T_n + mu_y 1,  S = sigma_y S_n,  R = sigma_y R_n.
    """

    def __init__(self, inner: nn.Module, target_idx: int = -1, eps: float = 1e-5):
        super().__init__()
        self.inner, self.target_idx, self.eps = inner, target_idx, eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        sd = x.std(1, keepdim=True) + self.eps
        out = self.inner((x - mu) / sd)
        my = mu[:, 0, self.target_idx].unsqueeze(1)
        sy = sd[:, 0, self.target_idx].unsqueeze(1)
        res = {}
        for k, v in out.items():
            if k in ("forecast", "trend"):
                res[k] = v * sy + my
            elif k in ("seasonal", "residual", "theta_S", "u", "amplitude"):
                res[k] = v * sy
            elif k == "theta_T":
                t = v * sy
                res[k] = torch.cat([t[:, :1] + my, t[:, 1:]], 1)
            else:
                res[k] = v
        res["mu_y"], res["sigma_y"] = my, sy
        return res


def target_index(name: str, feat_names) -> int:
    if "log_return" in feat_names:
        return list(feat_names).index("log_return")
    return len(feat_names) - 1


def make_nfh(d, cfg: Config = CONFIG, normalize: bool = False):
    return BackboneHeadModel(ConvLSTMBackbone(d, cfg.latent_dim, cfg.hidden),
                             NFHHead(cfg.latent_dim, cfg.horizon, cfg.n_harmonics, cfg.trend_degree,
                                     normalize=normalize))


def make_dense(d, cfg: Config = CONFIG):
    return BackboneHeadModel(ConvLSTMBackbone(d, cfg.latent_dim, cfg.hidden),
                             DenseHead(cfg.latent_dim, cfg.horizon))


class LSTMForecaster(nn.Module):
    def __init__(self, d, H, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(d, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, H))

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return {"forecast": self.fc(hn[-1])}


class MLPForecaster(nn.Module):
    def __init__(self, d, L, H, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(L * d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, H))

    def forward(self, x):
        return {"forecast": self.net(x)}


def nbeats_seasonal_basis(n: int, H_ref: int | None = None) -> np.ndarray:
    """Base S do N-BEATS-I: [1, cos(2 pi i t), ..., sin(2 pi i t)], i=1..floor(H/2-1),
    t = [0..n-1]/n (conforme ORESHKIN et al., 2020, eq. 3)."""
    t = np.arange(n) / n
    kmax = int(np.floor(n / 2 - 1))
    rows = [np.ones(n)] + [np.cos(2 * np.pi * i * t) for i in range(1, kmax + 1)] \
        + [np.sin(2 * np.pi * i * t) for i in range(1, kmax + 1)]
    return np.stack(rows, 0)


class NBeatsBlock(nn.Module):
    def __init__(self, in_size, width, back_basis, fore_basis, layers=4):
        super().__init__()
        mods, s = [], in_size
        for _ in range(layers):
            mods += [nn.Linear(s, width), nn.ReLU()]
            s = width
        self.fc = nn.Sequential(*mods)
        self.register_buffer("Vb", torch.tensor(back_basis, dtype=torch.float32))
        self.register_buffer("Vf", torch.tensor(fore_basis, dtype=torch.float32))
        self.tb = nn.Linear(width, self.Vb.shape[0])
        self.tf = nn.Linear(width, self.Vf.shape[0])

    def forward(self, x):
        z = self.fc(x)
        return self.tb(z) @ self.Vb, self.tf(z) @ self.Vf


class NBeatsI(nn.Module):
    """N-BEATS interpretável (ORESHKIN et al., 2020), adaptado para entrada achatada.

    Pilha de tendência seguida da pilha de sazonalidade; 3 blocos por pilha com
    pesos compartilhados; backcast duplamente residual; y_hat = soma dos
    forecasts parciais. Para entradas multivariadas (d>1) a janela é achatada
    (L*d) e a base de backcast é aplicada a cada variável (repetida).
    """

    def __init__(self, d, L, H, degree=2, width=256, blocks=3):
        super().__init__()
        self.d, self.L, self.H, self.blocks = d, L, H, blocks
        n_in = L * d
        tb = poly_basis(L, degree)
        tf = poly_basis(H, degree)
        sb = nbeats_seasonal_basis(L)
        sf = nbeats_seasonal_basis(H)
        # backcast no espaço achatado (L*d): base repetida por variável (bloco-diagonal)
        tb_flat = np.kron(np.eye(d), tb)
        sb_flat = np.kron(np.eye(d), sb)
        self.trend_block = NBeatsBlock(n_in, width, tb_flat, tf)
        self.season_block = NBeatsBlock(n_in, width, sb_flat, sf)

    def forward(self, x):
        # x: (B, L, d) -> achatado por variável: (B, d*L) com ordem [var1 L..., var2 L...]
        r = x.transpose(1, 2).reshape(x.shape[0], -1)
        trend = 0
        for _ in range(self.blocks):
            b, f = self.trend_block(r)
            r = r - b
            trend = trend + f
        season = 0
        for _ in range(self.blocks):
            b, f = self.season_block(r)
            r = r - b
            season = season + f
        return {"forecast": trend + season, "trend": trend, "seasonal": season}


def build_model(name: str, d: int, cfg: Config = CONFIG, target_idx: int = -1, revin: bool = True):
    L, H = cfg.lookback, cfg.horizon
    if name.startswith("NFHn"):
        m = make_nfh(d, cfg, normalize=True)
    elif name.startswith("NFH"):
        m = make_nfh(d, cfg)
    elif name == "ConvLSTM+Densa":
        m = make_dense(d, cfg)
    elif name == "N-BEATS-I":
        m = NBeatsI(d, L, H, degree=cfg.trend_degree)
    elif name == "LSTM":
        m = LSTMForecaster(d, H, cfg.hidden)
    elif name == "MLP":
        m = MLPForecaster(d, L, H)
    else:
        raise ValueError(name)
    return InstanceNorm(m, target_idx) if revin else m


def build_seeded(name: str, d: int, seed: int, cfg: Config = CONFIG, target_idx: int = -1):
    """Fixa a semente ANTES da construção: inicialização reprodutível isoladamente."""
    set_seed(seed)
    return build_model(name, d, cfg, target_idx=target_idx)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Treino / avaliação
# ---------------------------------------------------------------------------
def residual_penalty(out, H):
    if "residual" not in out:
        return 0.0
    return (out["residual"] ** 2).sum(1).mean() / H


def train_model(model, tr_ds, va_ds, seed: int, lam: float = 0.0, cfg: Config = CONFIG,
                dev=None, verbose=False):
    """Adam + early stopping na validação. Perda = MSE + lam * ||R||^2 / H."""
    dev = dev or device()
    set_seed(seed)
    model = model.to(dev)
    g = torch.Generator()
    g.manual_seed(seed)
    tl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, generator=g)
    Xv, Yv = stack_ds(va_ds)
    Xv, Yv = Xv.to(dev), Yv.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()
    best, best_state, bad, hist = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(cfg.max_epochs):
        model.train()
        tr_losses = []
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = model(x)
            loss = mse(out["forecast"], y)
            if lam > 0:
                loss = loss + lam * residual_penalty(out, cfg.horizon)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tr_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            ov = model(Xv)
            v = mse(ov["forecast"], Yv).item()
            if lam > 0:
                v_obj = v + lam * float(residual_penalty(ov, cfg.horizon))
            else:
                v_obj = v
        hist.append((float(np.mean(tr_losses)), v))
        if v_obj < best - 1e-7:
            best, bad = v_obj, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if verbose:
            print(ep, hist[-1])
        if bad >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"epochs": len(hist), "train_time_s": time.time() - t0, "history": hist}


@torch.no_grad()
def predict(model, X, dev=None, bs=512):
    dev = dev or next(model.parameters()).device
    outs = []
    for i in range(0, len(X), bs):
        o = model(X[i:i + bs].to(dev))
        outs.append({k: v.detach().cpu() for k, v in o.items() if torch.is_tensor(v)})
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def forecast_metrics(pred: np.ndarray, y: np.ndarray):
    e = pred - y
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e)))}


def energy_stats(out):
    """Balanço de energia da previsão CENTRADA (Teorema do balanço energético).

    y_c = y - mean(y) = T_c + S + R, com T_c = T - mean(T) (S e R têm média nula).
    Como T_c, S e R são mutuamente ortogonais: 1 = eps_T + eps_S + eps_R (iota = 0,
    calculado apenas como verificação numérica).
    """
    y = out["forecast"].double()
    T, S, R = out["trend"].double(), out["seasonal"].double(), out["residual"].double()
    yc = y - y.mean(1, keepdim=True)
    Tc = T - T.mean(1, keepdim=True)
    E = (yc ** 2).sum(1) + 1e-30
    eT, eS, eR = (Tc ** 2).sum(1) / E, (S ** 2).sum(1) / E, (R ** 2).sum(1) / E
    iota = 2 * (Tc * S).sum(1) / E
    return {"eps_T": eT.numpy(), "eps_S": eS.numpy(), "eps_R": eR.numpy(), "iota": iota.numpy(),
            "coverage": (1 - eR).numpy(), "level": y.mean(1).numpy()}


def harmonic_shares(out, Ft: torch.Tensor):
    """Participação e_k = <S_k, S> / ||y_c||^2 de cada harmônico (soma = eps_S)."""
    y = out["forecast"].double()
    yc = y - y.mean(1, keepdim=True)
    E = (yc ** 2).sum(1) + 1e-30
    th = out["theta_S"].double()
    S = out["seasonal"].double()
    Ftd = Ft.double().cpu()
    K = th.shape[1] // 2
    ek = []
    for k in range(K):
        sk = th[:, 2 * k:2 * k + 1] * Ftd[2 * k] + th[:, 2 * k + 1:2 * k + 2] * Ftd[2 * k + 1]
        ek.append((sk * S).sum(1) / E)
    return torch.stack(ek, 1).numpy()


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def config_dict(cfg: Config = CONFIG):
    d = asdict(cfg)
    d["seeds"] = list(cfg.seeds)
    return d


# ---------------------------------------------------------------------------
# Seleção de lambda (regra fixada a priori, usa SOMENTE a validação)
# ---------------------------------------------------------------------------
FIN = UNI + MULTI


def select_lambda(runs: pd.DataFrame, tol: float = 0.01, grid=(0.0, 0.01, 0.1, 1.0)) -> float:
    """Maior lambda da grade PRÉ-ESPECIFICADA cujo RMSE médio de VALIDAÇÃO
    (15 conjuntos, 5 sementes) não excede em mais de `tol` o de lambda=0."""
    nfh = runs[runs["model"].str.startswith("NFH-l") & runs["lambda"].isin(list(grid))]
    v = nfh.groupby("lambda")["val_RMSE"].mean()
    base = v.loc[0.0]
    ok = [l for l, val in v.items() if val <= base * (1 + tol)]
    return float(max(ok))


def select_config(runs: pd.DataFrame, tol: float = 0.01):
    """Seleção da configuração final da NFH usando SOMENTE a validação.

    Para cada parametrização (NFH padrão, prefixo "NFH-l"; NFH com colunas
    normalizadas, prefixo "NFHn-l"): lambda* = maior lambda cujo RMSE médio de
    validação (15 conjuntos x 5 sementes) não excede em mais de `tol` o de
    lambda = 0. Entre as duas parametrizações, escolhe-se a de menor RMSE médio
    de validação em seu lambda*. Retorna (nome_do_modelo, lambda*, detalhes).
    """
    details = {}
    for prefix in ("NFH-l", "NFHn-l"):
        sub = runs[runs["model"].str.startswith(prefix)]
        if sub.empty or 0.0 not in set(sub["lambda"]):
            continue
        # exige a grade completa em todos os conjuntos e sementes (evita seleção com dados parciais)
        counts = sub.groupby("lambda").size()
        sub = sub[sub["lambda"].isin(counts[counts == counts.max()].index)]
        v = sub.groupby("lambda")["val_RMSE"].mean()
        base = v.loc[0.0]
        lam = max(l for l, val in v.items() if val <= base * (1 + tol))
        details[prefix] = {"lambda_star": float(lam), "val_RMSE": float(v.loc[lam]),
                           "val_RMSE_por_lambda": {f"{l:g}": float(x) for l, x in v.items()}}
    best = min(details, key=lambda p: details[p]["val_RMSE"])
    lam = details[best]["lambda_star"]
    return f"{best}{lam:g}", lam, details


def final_model_name(runs: pd.DataFrame | None = None) -> str:
    runs = runs if runs is not None else pd.read_csv(RES / "forecast_runs.csv")
    return select_config(runs)[0]


def load_trained(ds: str, model_name: str, seed: int = 0, dev=None):
    """Reconstrói e carrega um modelo salvo pelo E1."""
    import dataclasses
    sp = dataset_spec(ds)
    cfg = dataclasses.replace(CONFIG, lookback=sp["L"], horizon=sp["H"])
    f, t, names, extra = load_any(ds)
    m = build_model(model_name, f.shape[1], cfg, target_idx=target_index(ds, names))
    state = torch.load(RES / "models" / f"{ds}__{model_name}__s{seed}.pt", map_location="cpu")
    m.load_state_dict(state)
    m.eval()
    if dev is not None:
        m = m.to(dev)
    return m, cfg, (f, t, names, extra)
