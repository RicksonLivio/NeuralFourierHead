"""
NFH implementada "à mão", versor a versor.

Este módulo refaz TODAS as contas da Neural Fourier Head sem usar
``torch.linalg``, ``@`` entre matrizes nem broadcasting: só produtos internos
entre vetores 1D (``torch.dot``), somas e laços explícitos. O objetivo é
didático: cada passo corresponde a uma linha da demonstração da dissertação e
pode ser conferido com papel e lápis.

O resultado é idêntico (até arredondamento) ao de ``nfh.bases`` /
``nfh.head`` / ``nfh.explain`` — ver ``tests/test_manual_vs_library.py``.

Ideia central em uma frase: um vetor y ∈ R^H é escrito numa base de H vetores
fixos; os primeiros q+1 são polinômios, os 2K seguintes são senoides sem parte
polinomial e os últimos H−m completam o espaço de forma ortonormal. Como os três
grupos são ortogonais entre si, as "fatias" de y em cada grupo são únicas.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch

Vec = torch.Tensor  # vetor 1D de tamanho H (float64)


# --------------------------------------------------------------------------- #
# 1. Operações vetoriais elementares
# --------------------------------------------------------------------------- #
def dot(a: Vec, b: Vec) -> float:
    """Produto interno ⟨a, b⟩ = Σ_j a_j b_j."""
    return float(torch.dot(a, b))


def norm(a: Vec) -> float:
    """Norma euclidiana ‖a‖ = sqrt(⟨a, a⟩)."""
    return math.sqrt(dot(a, a))


def combine(coefs: List[float], vectors: List[Vec]) -> Vec:
    """Combinação linear Σ_i c_i v_i, somando um vetor por vez."""
    out = torch.zeros_like(vectors[0])
    for c, v in zip(coefs, vectors):
        out = out + c * v
    return out


# --------------------------------------------------------------------------- #
# 2. Vetores-base (linhas de P e F)
# --------------------------------------------------------------------------- #
def time_grid(H: int) -> Vec:
    """τ_j = j/H, construído elemento por elemento."""
    return torch.tensor([j / H for j in range(H)], dtype=torch.float64)


def trend_vectors(H: int, q: int) -> List[Vec]:
    """Linhas de P: p_i = (τ_0^i, ..., τ_{H-1}^i), i = 0..q."""
    tau = time_grid(H)
    return [torch.tensor([float(t) ** i for t in tau], dtype=torch.float64) for i in range(q + 1)]


def fourier_vectors(H: int, K: int) -> List[Vec]:
    """Linhas de F, intercaladas: cos(2πkτ), sin(2πkτ) para k = 1..K."""
    tau = time_grid(H)
    rows: List[Vec] = []
    for k in range(1, K + 1):
        rows.append(torch.tensor([math.cos(2 * math.pi * k * float(t)) for t in tau], dtype=torch.float64))
        rows.append(torch.tensor([math.sin(2 * math.pi * k * float(t)) for t in tau], dtype=torch.float64))
    return rows


# --------------------------------------------------------------------------- #
# 3. Gram-Schmidt: base ortonormal de um conjunto de vetores
# --------------------------------------------------------------------------- #
def gram_schmidt(vectors: List[Vec], basis: List[Vec] | None = None, tol: float = 1e-10) -> List[Vec]:
    """Ortonormaliza ``vectors`` (Gram-Schmidt modificado, com reortogonalização).

    Cada novo vetor tem removidas, uma a uma, suas projeções sobre os versores
    já aceitos; se sobrar algo de norma > tol, ele é normalizado e aceito.
    """
    out = list(basis) if basis else []
    for v in vectors:
        w = v.clone()
        for _ in range(2):  # duas passadas: estabilidade numérica ("twice is enough")
            for e in out:
                w = w - dot(w, e) * e
        n = norm(w)
        if n > tol:
            out.append(w / n)
    return out


def project(y: Vec, orthonormal: List[Vec]) -> Vec:
    """Projeção ortogonal de y no espaço gerado por versores ortonormais: Σ ⟨y, e⟩ e."""
    return combine([dot(y, e) for e in orthonormal], orthonormal)


# --------------------------------------------------------------------------- #
# 4. As três famílias de vetores da NFH
# --------------------------------------------------------------------------- #
def nfh_vectors(H: int, K: int, q: int) -> Dict[str, List[Vec]]:
    """Constrói, versor a versor:

    trend   : p_0..p_q                     (linhas de P, NÃO ortonormais)
    seasonal: f̃_1..f̃_2K = f − Π_T f         (linhas de F̃)
    resid   : r_1..r_{H−m}                 (versores de Q⊥, ortonormais)
    e_trend : versores ortonormais de 𝒯    (auxiliares para Π_T)
    """
    m = q + 1 + 2 * K
    if 2 * K >= H or m > H:
        raise ValueError(f"Exige 2K < H e q+1+2K <= H (recebido H={H}, K={K}, q={q}).")
    P = trend_vectors(H, q)
    F = fourier_vectors(H, K)

    # Versores de 𝒯 = Img(Pᵀ): Π_T y = Σ ⟨y, e_i⟩ e_i
    e_trend = gram_schmidt(P)
    # F̃: remove de cada senoide sua parte polinomial
    Ft = [f - project(f, e_trend) for f in F]

    # Versores de Img([Pᵀ Fᵀ]) e, completando com a base canônica, os de ℛ = complemento
    e_struct = gram_schmidt(F, basis=e_trend)
    canon = [torch.eye(H, dtype=torch.float64)[j] for j in range(H)]
    full = gram_schmidt(canon, basis=e_struct)
    resid = full[len(e_struct):]
    assert len(resid) == H - m, "dimensão do resíduo deveria ser H − m"
    return {"trend": P, "seasonal": Ft, "resid": resid, "e_trend": e_trend}


# --------------------------------------------------------------------------- #
# 5. Síntese (forward da NFH para um único exemplo)
# --------------------------------------------------------------------------- #
def head_forward(h: Vec, W: torch.Tensor, b: Vec, vecs: Dict[str, List[Vec]],
                 D: List[float] | None = None) -> Dict[str, object]:
    """c = W h + b (linha a linha) → (θ_T, θ_S, u) → T, S, R → ŷ.

    Args:
        h: representação latente (p,).
        W: pesos (H, p) e b: vieses (H,) da camada afim.
        vecs: saída de ``nfh_vectors``.
        D: diagonal opcional da variante NFHn (c_i ← D_i c_i).
    """
    H = W.shape[0]
    # c_i = ⟨W_i, h⟩ + b_i, uma coordenada por vez
    c = [dot(W[i].double(), h.double()) + float(b[i]) for i in range(H)]
    if D is not None:
        c = [ci * di for ci, di in zip(c, D)]
    q1, K2 = len(vecs["trend"]), len(vecs["seasonal"])
    th_T, th_S, u = c[:q1], c[q1:q1 + K2], c[q1 + K2:]
    T = combine(th_T, vecs["trend"])
    S = combine(th_S, vecs["seasonal"])
    R = combine(u, vecs["resid"]) if u else torch.zeros(H, dtype=torch.float64)
    amps = [math.hypot(th_S[2 * k], th_S[2 * k + 1]) for k in range(K2 // 2)]
    phases = [math.atan2(th_S[2 * k + 1], th_S[2 * k]) for k in range(K2 // 2)]
    return {"c": c, "theta_T": th_T, "theta_S": th_S, "u": u, "T": T, "S": S, "R": R,
            "forecast": T + S + R, "amplitude": amps, "phase": phases}


def column_norms(vecs: Dict[str, List[Vec]]) -> List[float]:
    """D_i = 1/‖coluna i de M = [Pᵀ F̃ᵀ Q⊥]‖ (NFHn)."""
    cols = vecs["trend"] + vecs["seasonal"] + vecs["resid"]
    return [1.0 / norm(v) for v in cols]


# --------------------------------------------------------------------------- #
# 6. Explicação: balanço de energia e participação dos harmônicos
# --------------------------------------------------------------------------- #
def center(v: Vec) -> Vec:
    """v − média(v)·𝟙."""
    mean = sum(float(x) for x in v) / len(v)
    return v - mean


def energy_balance(T: Vec, S: Vec, R: Vec) -> Dict[str, float]:
    """ε_T, ε_S, ε_R da previsão centrada (Teorema do balanço de energia).

    ŷ_c = T_c + S + R  e, por ortogonalidade, ‖ŷ_c‖² = ‖T_c‖² + ‖S‖² + ‖R‖².
    """
    y = T + S + R
    yc, Tc = center(y), center(T)
    E = dot(yc, yc)
    if E <= 0:
        return {"eps_T": 0.0, "eps_S": 0.0, "eps_R": 0.0, "level": float(y.mean()), "cross": 0.0}
    eT, eS, eR = dot(Tc, Tc) / E, dot(S, S) / E, dot(R, R) / E
    return {"eps_T": eT, "eps_S": eS, "eps_R": eR, "coverage": 1 - eR,
            "level": sum(float(v) for v in y) / len(y), "cross": 2 * dot(Tc, S) / E}


def harmonic_shares(th_S: List[float], vecs: Dict[str, List[Vec]], y: Vec) -> List[float]:
    """e_k = ⟨S_k, S⟩ / ‖ŷ_c‖², com S_k = a_k f̃_{2k-1} + b_k f̃_{2k}; Σ e_k = ε_S."""
    Ft = vecs["seasonal"]
    S_parts = [th_S[2 * k] * Ft[2 * k] + th_S[2 * k + 1] * Ft[2 * k + 1] for k in range(len(Ft) // 2)]
    S = combine([1.0] * len(S_parts), S_parts)
    yc = center(y)
    E = dot(yc, yc)
    return [dot(Sk, S) / E for Sk in S_parts]


# --------------------------------------------------------------------------- #
# 7. Identificabilidade: recuperar (θ_T, θ_S, u) a partir APENAS de ŷ
# --------------------------------------------------------------------------- #
def solve_small(G: List[List[float]], r: List[float]) -> List[float]:
    """Resolve G x = r por eliminação de Gauss com pivoteamento (sistemas pequenos)."""
    n = len(r)
    A = [row[:] + [ri] for row, ri in zip(G, r)]
    for col in range(n):
        piv = max(range(col, n), key=lambda i: abs(A[i][col]))
        A[col], A[piv] = A[piv], A[col]
        for i in range(col + 1, n):
            f = A[i][col] / A[col][col]
            for j in range(col, n + 1):
                A[i][j] -= f * A[col][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (A[i][n] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
    return x


def decompose(y: Vec, vecs: Dict[str, List[Vec]]) -> Dict[str, object]:
    """Teorema da identificabilidade, versor a versor.

        θ_T = (P Pᵀ)⁻¹ P ŷ      (equações normais com a matriz de Gram de P)
        θ_S = (F̃ F̃ᵀ)⁻¹ F̃ ŷ
        u   = Q⊥ᵀ ŷ             (Q⊥ é ortonormal: basta o produto interno)

    Funciona porque 𝒯, 𝒮 e ℛ são ortogonais: ao projetar ŷ em um deles, as
    outras duas parcelas desaparecem.
    """
    out = {}
    for key, name in (("trend", "theta_T"), ("seasonal", "theta_S")):
        V = vecs[key]
        G = [[dot(vi, vj) for vj in V] for vi in V]  # matriz de Gram
        out[name] = solve_small(G, [dot(vi, y) for vi in V])
    out["u"] = [dot(r, y) for r in vecs["resid"]]
    T = combine(out["theta_T"], vecs["trend"])
    S = combine(out["theta_S"], vecs["seasonal"])
    R = combine(out["u"], vecs["resid"]) if out["u"] else torch.zeros_like(y)
    out.update({"T": T, "S": S, "R": R, "reconstruction_error": float((T + S + R - y).abs().max())})
    return out


def oblique_decompose(y: Vec, groups: Dict[str, List[Vec]]) -> Dict[str, Vec]:
    """Decomposição de y numa base qualquer (não necessariamente ortogonal) que gera R^H.

    Resolve o sistema conjunto G c = (⟨v_i, y⟩)_i com G a matriz de Gram de TODOS os
    vetores, e devolve a parcela de cada grupo. É o que se faz com a base crua
    [Pᵀ Fᵀ Q⊥]: as parcelas somam y, mas não são ortogonais entre si.
    """
    names = list(groups)
    allv = [v for n in names for v in groups[n]]
    G = [[dot(a, b) for b in allv] for a in allv]
    c = solve_small(G, [dot(v, y) for v in allv])
    out, i = {}, 0
    for n in names:
        k = len(groups[n])
        out[n] = combine(c[i:i + k], groups[n]) if k else torch.zeros_like(y)
        i += k
    return out


def orthogonality_report(vecs: Dict[str, List[Vec]]) -> Tuple[float, float, float]:
    """max |⟨𝒯, 𝒮⟩|, max |⟨𝒯, ℛ⟩|, max |⟨𝒮, ℛ⟩| (devem ser ~0)."""
    def mx(A: List[Vec], B: List[Vec]) -> float:
        return max((abs(dot(a, b)) for a in A for b in B), default=0.0)
    return mx(vecs["trend"], vecs["seasonal"]), mx(vecs["trend"], vecs["resid"]), mx(vecs["seasonal"], vecs["resid"])
