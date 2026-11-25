"""
``NeuralFourierHead``: a camada de saída interpretável, como ``nn.Module``.

Uso mínimo::

    head = NeuralFourierHead(in_features=128, horizon=30, n_harmonics=10, degree=2)
    out = head(h)            # h: (B, 128)  ->  NFHOutput com forecast (B, 30), trend, seasonal, ...
    loss = mse(out.forecast, y) + lam * head.residual_penalty(out)

Propriedades (dissertação, Cap. 4):

* **Linear**: c = W h + b (uma única ``nn.Linear(p, H)``); a não linearidade fica no backbone.
* **Mesmo número de parâmetros** de uma camada densa: H(p + 1).
* **Equivalência**: NFH(h) = M (W h + b) com M = [Pᵀ F̃ᵀ Q⊥] invertível, logo qualquer
  camada densa é uma NFH (``from_linear``) e vice-versa (``to_linear``).
* **Exata e identificável**: ŷ = T + S + R com T, S, R ortogonais e únicos.

Variante NFHn (``normalize_columns=True``, padrão, configuração final da
dissertação): c ← D ⊙ c com D_i = 1/‖M_{:,i}‖. A função representada e a
explicação são as mesmas; muda só a escala efetiva do passo do otimizador em
cada direção de saída.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .bases import NFHBases, build_bases
from .components import NFHOutput

__all__ = ["NeuralFourierHead", "attach_nfh"]


class NeuralFourierHead(nn.Module):
    """Neural Fourier Head.

    Args:
        in_features: dimensão p da representação latente.
        horizon: H, número de passos previstos.
        n_harmonics: K harmônicos (período H/k); exige 2K < H e q + 1 + 2K ≤ H.
        degree: q, grau do polinômio de tendência.
        normalize_columns: usa a variante NFHn (colunas de M com norma 1).
        bias: inclui o viés b (padrão True, como na dissertação).
    """

    def __init__(self, in_features: int, horizon: int, n_harmonics: int = 10, degree: int = 2,
                 normalize_columns: bool = True, bias: bool = True):
        super().__init__()
        bases = build_bases(horizon, n_harmonics, degree, dtype=torch.float32)
        self.in_features, self.horizon = in_features, horizon
        self.n_harmonics, self.degree = n_harmonics, degree
        self.normalize_columns = normalize_columns
        self.q1 = degree + 1
        self.m = bases.m
        self.n_resid = bases.n_resid
        # Bases fixas como buffers: acompanham .to(device) e o state_dict, mas não treinam
        self.register_buffer("P", bases.P)
        self.register_buffer("F", bases.F)
        self.register_buffer("Ft", bases.Ft)
        self.register_buffer("Q", bases.Q)
        self.register_buffer("M", bases.M)
        D = bases.D if normalize_columns else torch.ones(horizon)
        self.register_buffer("D", D)
        self.lin = nn.Linear(in_features, horizon, bias=bias)

    # ------------------------------------------------------------------ #
    def _apply(self, fn, recurse=True):
        """Ao converter para float64 (``.double()``), recalcula as bases em precisão
        total em vez de apenas converter os valores float32 já arredondados."""
        super()._apply(fn, recurse)
        if self.P.dtype == torch.float64:
            b = build_bases(self.horizon, self.n_harmonics, self.degree, dtype=torch.float64)
            dev = self.P.device
            for name in ("P", "F", "Ft", "Q", "M"):
                getattr(self, name).copy_(getattr(b, name).to(dev))
            if self.normalize_columns:
                self.D.copy_(b.D.to(dev))
        return self

    @property
    def bases(self) -> NFHBases:
        """Bases em float64, recalculadas (úteis para análise)."""
        return build_bases(self.horizon, self.n_harmonics, self.degree, dtype=torch.float64)

    @property
    def periods(self) -> torch.Tensor:
        return self.horizon / torch.arange(1, self.n_harmonics + 1, dtype=torch.float32)

    def coordinates(self, h: torch.Tensor) -> torch.Tensor:
        """c = D ⊙ (W h + b) ∈ R^H (coordenadas na base M)."""
        return self.lin(h) * self.D

    def split(self, c: torch.Tensor):
        """Particiona c em (θ_T, θ_S, u)."""
        return c[..., :self.q1], c[..., self.q1:self.m], c[..., self.m:]

    def synthesize(self, theta_T: torch.Tensor, theta_S: torch.Tensor, u: torch.Tensor,
                   latent: Optional[torch.Tensor] = None) -> NFHOutput:
        """Coeficientes → componentes (bases fixas)."""
        T = theta_T @ self.P
        S = theta_S @ self.Ft
        R = u @ self.Q.T
        a, b = theta_S[..., 0::2], theta_S[..., 1::2]
        return NFHOutput(forecast=T + S + R, trend=T, seasonal=S, residual=R,
                         theta_T=theta_T, theta_S=theta_S, u=u,
                         amplitude=torch.sqrt(a ** 2 + b ** 2), phase=torch.atan2(b, a), latent=latent)

    def forward(self, h: torch.Tensor) -> NFHOutput:
        """h (B, p) → NFHOutput (todas as componentes, escala da entrada da head)."""
        theta_T, theta_S, u = self.split(self.coordinates(h))
        return self.synthesize(theta_T, theta_S, u, latent=h)

    def residual_penalty(self, out: NFHOutput) -> torch.Tensor:
        """ρ = (1/(B H)) Σ_n ‖R_n‖²  (termo λ ρ da perda J_λ)."""
        return (out.residual ** 2).sum(dim=-1).mean() / self.horizon

    # ------------------------------------------------------------------ #
    # Identificabilidade: decompor QUALQUER previsão (de qualquer modelo)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def decompose(self, y: torch.Tensor) -> NFHOutput:
        """Teorema da identificabilidade: (θ_T, θ_S, u) = M⁻¹ ŷ.

        Funciona para previsões de outros modelos (ex.: camada densa), dando a
        explicação única que a NFH atribuiria àquela previsão.
        """
        coords = torch.linalg.solve(self.M.double(), y.double().T).T.to(y.dtype)  # (B, H)
        return self.synthesize(*self.split(coords))

    # ------------------------------------------------------------------ #
    # Equivalência com a camada densa (Teorema da equivalência)
    # ------------------------------------------------------------------ #
    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear, n_harmonics: int = 10, degree: int = 2,
                    normalize_columns: bool = True) -> "NeuralFourierHead":
        """Converte uma camada densa treinada em NFH SEM mudar nenhuma previsão.

        W_NFH = D⁻¹ M⁻¹ W_densa,  b_NFH = D⁻¹ M⁻¹ b_densa.
        Útil para explicar um modelo já treinado: troque a última camada e leia
        a decomposição (o modelo continua prevendo exatamente o mesmo).
        """
        p, H = linear.in_features, linear.out_features
        head = cls(p, H, n_harmonics, degree, normalize_columns, bias=linear.bias is not None)
        head = head.to(device=linear.weight.device, dtype=linear.weight.dtype)
        Minv = torch.linalg.inv(head.M.double())
        Dinv = (1.0 / head.D.double())[:, None]
        head.lin.weight.copy_((Dinv * (Minv @ linear.weight.double())).to(linear.weight.dtype))
        if linear.bias is not None:
            head.lin.bias.copy_((Dinv[:, 0] * (Minv @ linear.bias.double())).to(linear.bias.dtype))
        return head

    @torch.no_grad()
    def to_linear(self) -> nn.Linear:
        """Camada densa equivalente: W' = M D W, b' = M D b (mesmas previsões)."""
        lin = nn.Linear(self.in_features, self.horizon, bias=self.lin.bias is not None).to(
            device=self.lin.weight.device, dtype=self.lin.weight.dtype)
        MD = self.M.double() * self.D.double()[None, :]
        lin.weight.copy_((MD @ self.lin.weight.double()).to(lin.weight.dtype))
        if self.lin.bias is not None:
            lin.bias.copy_((MD @ self.lin.bias.double()).to(lin.bias.dtype))
        return lin

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, horizon={self.horizon}, n_harmonics={self.n_harmonics}, "
                f"degree={self.degree}, m={self.m}, normalize_columns={self.normalize_columns}")


def attach_nfh(model: nn.Module, attr: str, n_harmonics: int = 10, degree: int = 2,
               normalize_columns: bool = True, keep_function: bool = True) -> NeuralFourierHead:
    """Substitui ``model.<attr>`` (uma ``nn.Linear`` de saída) por uma NFH.

    Args:
        model: qualquer ``nn.Module`` cuja última camada seja ``nn.Linear(p, H)``.
        attr: caminho do atributo, ex. ``"fc"`` ou ``"decoder.out"``.
        keep_function: se True, converte os pesos (``from_linear``) e o modelo
            continua prevendo exatamente o mesmo; se False, cria uma NFH nova.

    Atenção: o ``forward`` do modelo passa a receber um ``NFHOutput`` dessa
    camada. Se o código do modelo espera um tensor, use ``out.forecast`` ou
    envolva a head com ``ForecastOnly`` (``nfh.model``).
    """
    parent = model
    *path, last = attr.split(".")
    for name in path:
        parent = getattr(parent, name)
    linear = getattr(parent, last)
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"model.{attr} deve ser nn.Linear (é {type(linear).__name__})")
    if keep_function:
        head = NeuralFourierHead.from_linear(linear, n_harmonics, degree, normalize_columns)
    else:
        head = NeuralFourierHead(linear.in_features, linear.out_features, n_harmonics, degree,
                                 normalize_columns, bias=linear.bias is not None).to(
            device=linear.weight.device, dtype=linear.weight.dtype)
    setattr(parent, last, head)
    return head
