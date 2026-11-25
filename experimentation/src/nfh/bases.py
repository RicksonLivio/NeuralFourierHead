"""
Bases fixas da Neural Fourier Head (implementação vetorizada com ``torch.linalg``).

Notação (dissertação, Cap. 4, Definição "Bases da NFH"):

    H      horizonte de previsão (número de passos)
    q      grau do polinômio de tendência
    K      número de harmônicos de Fourier
    τ_j    grade normalizada τ_j = j / H, j = 0, ..., H-1
    m      q + 1 + 2K (número de coordenadas "estruturais")

    P   ∈ R^{(q+1) × H}   P[i, j]   = τ_j^i                       (tendência)
    F   ∈ R^{2K × H}      F[2k-2,j] = cos(2π k τ_j),
                          F[2k-1,j] = sin(2π k τ_j)                (Fourier, k = 1..K)
    Π_T = Pᵀ (P Pᵀ)⁻¹ P   projetor ortogonal sobre o espaço de tendência 𝒯
    F̃   = F (I − Π_T)     senoides com a parte polinomial removida  (sazonal)
    Q⊥  ∈ R^{H × (H−m)}   base ortonormal de ℛ = Img([Pᵀ Fᵀ])^⊥     (resíduo)
    M   = [Pᵀ  F̃ᵀ  Q⊥]    matriz H × H invertível (Teorema da equivalência)
    D   = diag(1/‖M_{:,i}‖)  normalização de colunas (variante NFHn)

Condição de dimensão (Lema da independência): m = q + 1 + 2K ≤ H, que contém a
condição de Nyquist 2K < H. Sem ela, harmônicos entram em *aliasing* e a
explicação deixa de ser única.

Todas as bases são calculadas em float64 (estabilidade numérica do projetor e
da QR) e convertidas para o ``dtype`` pedido no fim.

Uma versão "à mão", vetor a vetor, das mesmas contas está em ``nfh.manual``;
os testes verificam que as duas coincidem.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

__all__ = ["NFHBases", "check_dimensions", "max_harmonics", "time_grid", "build_bases"]


def max_harmonics(horizon: int, degree: int) -> int:
    """Maior K que satisfaz q + 1 + 2K ≤ H e 2K < H (Nyquist)."""
    k = (horizon - degree - 1) // 2
    if horizon % 2 == 0:
        k = min(k, horizon // 2 - 1)
    return max(0, k)


def check_dimensions(horizon: int, n_harmonics: int, degree: int) -> int:
    """Valida (H, K, q) e devolve m = q + 1 + 2K.

    Raises:
        ValueError: se H < 1, K < 1, q < 0, 2K ≥ H (Nyquist) ou m > H.
    """
    if horizon < 2:
        raise ValueError(f"O horizonte deve ser >= 2 (recebido H={horizon}).")
    if n_harmonics < 1:
        raise ValueError(f"É preciso ao menos um harmônico (recebido K={n_harmonics}).")
    if degree < 0:
        raise ValueError(f"O grau da tendência deve ser >= 0 (recebido q={degree}).")
    if 2 * n_harmonics >= horizon:
        raise ValueError(
            f"Condição de Nyquist violada: 2K = {2 * n_harmonics} >= H = {horizon}. "
            f"Use K <= {max_harmonics(horizon, degree)}."
        )
    m = degree + 1 + 2 * n_harmonics
    if m > horizon:
        raise ValueError(
            f"q + 1 + 2K = {m} > H = {horizon}: a base [Pᵀ Fᵀ] não tem posto completo. "
            f"Use K <= {max_harmonics(horizon, degree)}."
        )
    return m


def time_grid(horizon: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Grade τ_j = j / H, j = 0..H-1 (a mesma do N-BEATS e da dissertação)."""
    return torch.arange(horizon, dtype=dtype) / horizon


@dataclass(frozen=True)
class NFHBases:
    """Conjunto completo de bases fixas da NFH para um par (H, K, q)."""

    horizon: int
    n_harmonics: int
    degree: int
    P: torch.Tensor        # (q+1, H)
    F: torch.Tensor        # (2K, H)
    Ft: torch.Tensor       # (2K, H)   F̃ = F (I − Π_T)
    Q: torch.Tensor        # (H, H−m)  Q⊥
    Pi_T: torch.Tensor     # (H, H)
    M: torch.Tensor        # (H, H)    [Pᵀ F̃ᵀ Q⊥]
    D: torch.Tensor        # (H,)      1 / ‖colunas de M‖

    @property
    def m(self) -> int:
        return self.degree + 1 + 2 * self.n_harmonics

    @property
    def n_resid(self) -> int:
        return self.horizon - self.m

    @property
    def periods(self) -> torch.Tensor:
        """Período (em passos) de cada harmônico: H / k."""
        k = torch.arange(1, self.n_harmonics + 1, dtype=self.P.dtype)
        return self.horizon / k

    def to(self, dtype: Optional[torch.dtype] = None, device=None) -> "NFHBases":
        kw = {"dtype": dtype, "device": device}
        return NFHBases(
            self.horizon, self.n_harmonics, self.degree,
            *(t.to(**kw) for t in (self.P, self.F, self.Ft, self.Q, self.Pi_T, self.M, self.D)),
        )

    def seasonal_condition(self) -> dict:
        """Constantes de estabilidade dos coeficientes sazonais (Prop. da estabilidade).

        σ_min(F̃) e γ_k = ‖bloco (2k-1, 2k) de (F̃ᵀ)⁺‖₂: quanto maior γ_k, mais a
        amplitude do harmônico k pode mudar para uma pequena mudança da previsão.
        Na base sem tendência, γ_k = sqrt(2/H) para todo k (referência de escala).
        """
        Ft = self.Ft.double()
        sv = torch.linalg.svdvals(Ft)
        pinv = torch.linalg.pinv(Ft.T)  # (2K, H)
        gammas = torch.stack([torch.linalg.matrix_norm(pinv[2 * k:2 * k + 2], ord=2) for k in range(self.n_harmonics)])
        return {"sigma_min_Ft": float(sv.min()), "gamma_k": gammas.tolist(),
                "gamma_reference": math.sqrt(2.0 / self.horizon)}


def build_bases(horizon: int, n_harmonics: int, degree: int,
                dtype: torch.dtype = torch.float32) -> NFHBases:
    """Constrói P, F, F̃, Q⊥, Π_T, M e D com álgebra linear vetorizada.

    Args:
        horizon: H, número de passos previstos.
        n_harmonics: K, número de harmônicos (períodos H/1, ..., H/K).
        degree: q, grau do polinômio de tendência.
        dtype: tipo final dos tensores (as contas são feitas em float64).
    """
    m = check_dimensions(horizon, n_harmonics, degree)
    tau = time_grid(horizon)                                         # (H,)

    # P: potências τ^i em uma única operação (broadcast)
    P = tau.unsqueeze(0) ** torch.arange(degree + 1, dtype=tau.dtype).unsqueeze(1)

    # F: linhas intercaladas cos/sin de frequência k/H
    k = torch.arange(1, n_harmonics + 1, dtype=tau.dtype).unsqueeze(1)   # (K, 1)
    angles = 2 * math.pi * k * tau.unsqueeze(0)                           # (K, H)
    F = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1).reshape(2 * n_harmonics, horizon)

    # Π_T = Pᵀ (P Pᵀ)⁻¹ P  (solve é mais estável que inverter explicitamente)
    Pi_T = P.T @ torch.linalg.solve(P @ P.T, P)
    I = torch.eye(horizon, dtype=tau.dtype)
    Ft = F @ (I - Pi_T)

    # Posto completo de B = [Pᵀ Fᵀ] e Q⊥ = complemento ortogonal via QR completa
    B = torch.cat([P, F], dim=0).T                                       # (H, m)
    rank = int(torch.linalg.matrix_rank(B))
    if rank != m:  # pragma: no cover - garantido pelo lema quando check_dimensions passa
        raise ValueError(f"[Pᵀ Fᵀ] sem posto completo: posto {rank} < m = {m}")
    Qfull, _ = torch.linalg.qr(B, mode="complete")
    Q = Qfull[:, m:]

    M = torch.cat([P.T, Ft.T, Q], dim=1)                                 # (H, H)
    D = 1.0 / torch.linalg.vector_norm(M, dim=0)

    return NFHBases(horizon, n_harmonics, degree,
                    *(t.to(dtype) for t in (P, F, Ft, Q, Pi_T, M, D)))
