"""
Contêiner da saída da NFH.

A NFH devolve, além da previsão, todas as peças da decomposição. Agrupá-las em
uma dataclass deixa o código de treino/explicação legível (``out.trend`` em vez
de ``out["trend"]``) e permite desnormalizar tudo de uma vez, preservando a
decomposição (Observação "Normalização por instância" da dissertação):

    ŷ ← σ ŷ + μ 𝟙,   T ← σ T + μ 𝟙,   S ← σ S,   R ← σ R,
    θ_T0 ← σ θ_T0 + μ,  θ_T(i>0) ← σ θ_T,  θ_S ← σ θ_S,  u ← σ u,  A_k ← σ A_k.

O vetor constante 𝟙 pertence ao espaço de tendência, então o nível μ entra
apenas em T (e no coeficiente θ_T0); S e R só mudam de escala.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Dict, Optional

import torch

__all__ = ["NFHOutput"]


@dataclass
class NFHOutput:
    forecast: torch.Tensor   # (B, H)   ŷ = T + S + R
    trend: torch.Tensor      # (B, H)   T = Pᵀ θ_T
    seasonal: torch.Tensor   # (B, H)   S = F̃ᵀ θ_S
    residual: torch.Tensor   # (B, H)   R = Q⊥ u
    theta_T: torch.Tensor    # (B, q+1)
    theta_S: torch.Tensor    # (B, 2K)  (a_1, b_1, ..., a_K, b_K)
    u: torch.Tensor          # (B, H−m)
    amplitude: torch.Tensor  # (B, K)   A_k = sqrt(a_k² + b_k²)
    phase: torch.Tensor      # (B, K)   φ_k = atan2(b_k, a_k)
    latent: Optional[torch.Tensor] = None  # (B, p) representação do backbone

    # ------------------------------------------------------------------ #
    def denormalize(self, mu: torch.Tensor, sigma: torch.Tensor) -> "NFHOutput":
        """Leva a decomposição para a escala original: x_orig = σ x + μ.

        Args:
            mu, sigma: tensores (B, 1) (ou escalares) com média e desvio do alvo.
        """
        mu = torch.as_tensor(mu, dtype=self.forecast.dtype, device=self.forecast.device)
        sigma = torch.as_tensor(sigma, dtype=self.forecast.dtype, device=self.forecast.device)
        mu = mu.reshape(-1, 1) if mu.ndim else mu.reshape(1, 1)
        sigma = sigma.reshape(-1, 1) if sigma.ndim else sigma.reshape(1, 1)
        th_T = self.theta_T * sigma
        th_T = torch.cat([th_T[:, :1] + mu, th_T[:, 1:]], dim=1)
        return replace(
            self,
            forecast=self.forecast * sigma + mu,
            trend=self.trend * sigma + mu,
            seasonal=self.seasonal * sigma,
            residual=self.residual * sigma,
            theta_T=th_T,
            theta_S=self.theta_S * sigma,
            u=self.u * sigma,
            amplitude=self.amplitude * sigma,
        )

    def detach(self) -> "NFHOutput":
        return replace(self, **{f.name: getattr(self, f.name).detach()
                                for f in fields(self) if getattr(self, f.name) is not None})

    def cpu(self) -> "NFHOutput":
        return replace(self, **{f.name: getattr(self, f.name).cpu()
                                for f in fields(self) if getattr(self, f.name) is not None})

    def __getitem__(self, idx) -> "NFHOutput":
        """Fatia o lote: ``out[0]`` mantém a dimensão do lote (B = 1)."""
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        return replace(self, **{f.name: getattr(self, f.name)[idx]
                                for f in fields(self) if getattr(self, f.name) is not None})

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}
