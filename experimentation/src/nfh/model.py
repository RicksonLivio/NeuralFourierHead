"""
Backbones e modelo completo ``RevIN(backbone → head)``.

A NFH é agnóstica ao backbone: qualquer rede que produza h ∈ R^p serve. Aqui
estão os três usados no tutorial:

* ``LSTMBackbone``          LSTM de 2 camadas (último estado oculto)
* ``TransformerBackbone``   encoder com atenção multi-cabeça + pooling temporal
* ``ConvLSTMBackbone``      Conv1D(7) → Conv1D(5) → LSTM(2): backbone da dissertação (p = 128)
* ``MLPBackbone``           janela achatada → 2 × (Linear 256 + ReLU): backbone da dissertação v3 (p = 256)

e ``Forecaster``, que acopla qualquer backbone a uma head de saída:

* ``head="nfh"``    Neural Fourier Head (interpretável)
* ``head="dense"``  camada densa nn.Linear(p, H) (referência caixa-preta); ``to_nfh()`` a converte
                    EXATAMENTE em NFH (Teorema da equivalência), ponto de partida do ajuste estrutural

com normalização reversível por instância (RevIN) opcional: cada janela é
padronizada por variável com média/desvio da própria janela e a previsão é
desnormalizada com (μ_y, σ_y) do alvo. Para a NFH isso preserva a decomposição
(``NFHOutput.denormalize``).
"""
from __future__ import annotations

import math
from typing import Literal, Optional, Sequence, Union

import torch
import torch.nn as nn

from .components import NFHOutput
from .head import NeuralFourierHead

__all__ = ["LSTMBackbone", "TransformerBackbone", "ConvLSTMBackbone", "MLPBackbone", "Forecaster", "BACKBONES"]


class LSTMBackbone(nn.Module):
    """X (B, L, d) → h (B, p): LSTM(2 camadas) → Linear → LayerNorm → ReLU."""

    def __init__(self, n_features: int, latent_dim: int = 128, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.proj = nn.Sequential(nn.Linear(hidden, latent_dim), nn.LayerNorm(latent_dim), nn.ReLU())
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.proj(h_n[-1])


class TransformerBackbone(nn.Module):
    """X (B, L, d) → h (B, p): embedding + posição senoidal → N blocos encoder → média temporal."""

    def __init__(self, n_features: int, latent_dim: int = 128, d_model: int = 64, n_heads: int = 4,
                 layers: int = 2, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
                                           batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.proj = nn.Sequential(nn.Linear(d_model, latent_dim), nn.LayerNorm(latent_dim), nn.ReLU())
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.embed(x) + self.pe[:, :x.shape[1]])
        return self.proj(z.mean(dim=1))


class ConvLSTMBackbone(nn.Module):
    """Backbone da dissertação: Conv1D(k=7) → Conv1D(k=5) → LSTM(2) → Linear+LN+ReLU."""

    def __init__(self, n_features: int, latent_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden, 7, padding=3), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.proj = nn.Sequential(nn.Linear(hidden, latent_dim), nn.LayerNorm(latent_dim), nn.ReLU())
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        _, (h_n, _) = self.lstm(z)
        return self.proj(h_n[-1])


class MLPBackbone(nn.Module):
    """X (B, L, d) → h (B, 256): janela achatada → Linear 256 + ReLU → Linear 256 + ReLU (dissertação v3)."""

    def __init__(self, n_features: int, latent_dim: int = 256, lookback: int = 96):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(lookback * n_features, latent_dim), nn.ReLU(),
                                 nn.Linear(latent_dim, latent_dim), nn.ReLU())
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


BACKBONES = {"lstm": LSTMBackbone, "transformer": TransformerBackbone, "convlstm": ConvLSTMBackbone,
             "mlp": MLPBackbone}


class Forecaster(nn.Module):
    """RevIN( backbone → head ), com head ∈ {"nfh", "dense"}.

    ``forward`` devolve sempre um ``NFHOutput``. Para a head densa, só
    ``forecast`` é significativo; as demais componentes ficam zeradas (use
    ``decompose_dense`` para ler uma previsão densa na base da NFH).

    Args:
        backbone: "lstm", "transformer", "convlstm" ou um ``nn.Module`` com ``latent_dim``.
        n_features: d, número de variáveis de entrada.
        horizon: H.
        target_idx: coluna do alvo entre as variáveis (para desnormalizar com RevIN).
        head: "nfh" ou "dense".
        n_harmonics, degree, normalize_columns: parâmetros da NFH.
        revin: True (todas as variáveis), False, ou lista de índices das variáveis normalizadas por instância.
    """

    def __init__(self, backbone: Union[str, nn.Module], n_features: int, horizon: int, target_idx: int = 0,
                 head: Literal["nfh", "dense"] = "nfh", n_harmonics: int = 10, degree: int = 2,
                 normalize_columns: bool = True, revin: Union[bool, Sequence[int]] = True, latent_dim: int = 128,
                 backbone_kwargs: Optional[dict] = None):
        super().__init__()
        if not isinstance(revin, bool):
            revin = sorted(int(i) for i in revin)
            if target_idx not in revin:
                raise ValueError("Com RevIN parcial, o alvo deve estar entre as variáveis normalizadas.")
        self._init_args = dict(n_features=n_features, horizon=horizon, target_idx=target_idx, n_harmonics=n_harmonics,
                               degree=degree, normalize_columns=normalize_columns, revin=revin, latent_dim=latent_dim,
                               backbone_kwargs=backbone_kwargs)
        if isinstance(backbone, str):
            self.backbone_name = backbone
            if backbone == "mlp":
                backbone = MLPBackbone(n_features, **({"latent_dim": 256} | (backbone_kwargs or {})))
            else:
                backbone = BACKBONES[backbone](n_features, latent_dim=latent_dim, **(backbone_kwargs or {}))
        else:
            self.backbone_name = type(backbone).__name__
        self.backbone = backbone
        p = getattr(backbone, "latent_dim", latent_dim)
        self.head_type = head
        self.horizon, self.target_idx, self.revin = horizon, target_idx, revin
        if head == "nfh":
            self.head = NeuralFourierHead(p, horizon, n_harmonics, degree, normalize_columns)
        elif head == "dense":
            self.head = nn.Linear(p, horizon)
        else:
            raise ValueError("head deve ser 'nfh' ou 'dense'")

    # ------------------------------------------------------------------ #
    def normalize(self, x: torch.Tensor):
        """RevIN: devolve (x_n, μ_y, σ_y) com μ, σ calculados na própria janela.

        Se ``revin`` for uma lista de índices, só essas variáveis são normalizadas
        por instância (útil para deixar flags binárias e codificações seno/cosseno
        intactas: normalizar uma flag quase constante amplifica ruído por 1/σ).
        """
        if self.revin is False:
            ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
            return x, ones * 0, ones
        mu = x.mean(1, keepdim=True)
        sd = x.std(1, keepdim=True) + 1e-5
        if self.revin is not True:
            mask = torch.zeros(x.shape[-1], dtype=torch.bool, device=x.device)
            mask[list(self.revin)] = True
            mu = torch.where(mask, mu, torch.zeros_like(mu))
            sd = torch.where(mask, sd, torch.ones_like(sd))
        return (x - mu) / sd, mu[:, 0, self.target_idx:self.target_idx + 1], sd[:, 0, self.target_idx:self.target_idx + 1]

    def inner(self, xn: torch.Tensor) -> NFHOutput:
        """Backbone + head sobre a janela já normalizada (escala RevIN)."""
        h = self.backbone(xn)
        if self.head_type == "nfh":
            return self.head(h)
        y = self.head(h)
        z = torch.zeros_like(y)
        return NFHOutput(forecast=y, trend=z, seasonal=z, residual=z, theta_T=y[:, :0], theta_S=y[:, :0],
                         u=y[:, :0], amplitude=y[:, :0], phase=y[:, :0], latent=h)

    def forward(self, x: torch.Tensor) -> NFHOutput:
        xn, mu, sd = self.normalize(x)
        out = self.inner(xn)
        if self.head_type == "nfh":
            return out.denormalize(mu, sd)
        out.forecast = out.forecast * sd + mu
        return out

    def residual_penalty(self, out: NFHOutput) -> torch.Tensor:
        if self.head_type != "nfh":
            return out.forecast.new_zeros(())
        return (out.residual ** 2).sum(-1).mean() / self.horizon

    @torch.no_grad()
    def to_nfh(self) -> "Forecaster":
        """Conversão exata densa → NFH (Teorema da equivalência): W_NFH = (M D)⁻¹ W'.

        Devolve um NOVO Forecaster com o mesmo backbone (cópia), NFH no lugar da densa e exatamente a mesma
        previsão (erro de arredondamento ~1e-6). É a época 0 do ajuste estrutural (decomposição pura).
        """
        import copy

        if self.head_type == "nfh":
            return copy.deepcopy(self)
        args = dict(self._init_args)
        new = Forecaster(copy.deepcopy(self.backbone), head="nfh", **args)
        new.backbone_name = self.backbone_name
        new.head = NeuralFourierHead.from_linear(self.head, n_harmonics=args["n_harmonics"], degree=args["degree"],
                                                 normalize_columns=args["normalize_columns"])
        return new.to(self.head.weight.device)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Atalho: só a previsão (B, H)."""
        return self(x).forecast
