"""
Explicações abdutivas estruturais sobre a última camada (dissertação v3, Seções 2.6.1 e 4.5).

Pergunta: qual é o MENOR conjunto 𝒜 de neurônios da representação h que, fixados nos valores da janela
explicada h*, GARANTE uma propriedade da previsão para QUALQUER valor dos demais neurônios dentro de uma
caixa [lo_i, hi_i]?

Propriedades estruturais lineares em h (Definição 4.11), g(h) = c_Tᵀθ_T + c_Sᵀθ_S ≥ ξ:
    P_T  sinal da variação líquida da tendência Δ_T = T_{H−1} − T_0, com ao menos metade do valor:
         s*·Δ_T(h) ≥ |Δ_T(h*)| / 2,  s* = sinal de Δ_T(h*)
    P_S  harmônico k (o de 24 h, k = H/24) com ao menos metade da amplitude na fase prevista:
         e*ᵀ(a_k, b_k)(h) ≥ A_k* / 2,  e* = (a_k*, b_k*)/A_k*   (implica A_k ≥ A_k*/2 e |φ_k − φ_k*| < π/2)

Como θ = W_eff h + b_eff é afim, g(h) = wᵀh + w0. Para uma função afim (Marques-Silva et al., 2020):
    pior caso com o neurônio i livre: min(w_i lo_i, w_i hi_i)
    GANHO de fixá-lo: δ_i = w_i h*_i − min(w_i lo_i, w_i hi_i) ≥ 0
    Γ(∅) = w0 + Σ_i min(w_i lo_i, w_i hi_i)    (todos livres)
Ordenando os ganhos e acumulando até atingir ξ obtém-se a explicação mínima em CARDINALIDADE (Prop. 2.4).

A garantia vale para os neurônios de h na caixa, não diretamente para a janela de entrada.
Caixa empírica: mínimo e máximo de cada h_i no treino, estendidos para conter h*.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .head import NeuralFourierHead

__all__ = ["latent_box", "head_coefficients", "structural_properties", "minimal_explanation",
           "abductive_explanations", "manual_minimal_explanation", "brute_force_minimum"]


@torch.no_grad()
def latent_box(model, X: torch.Tensor, batch: int = 1024) -> Dict[str, np.ndarray]:
    """Caixa empírica de h: mínimo, máximo e média de cada neurônio sobre as janelas de treino ``X``."""
    model.eval()
    dev = next(model.parameters()).device
    H = torch.cat([model(X[i:i + batch].to(dev)).latent.double().cpu() for i in range(0, len(X), batch)]).numpy()
    return {"lo": H.min(0), "hi": H.max(0), "mu": H.mean(0)}


def head_coefficients(head: NeuralFourierHead) -> Tuple[np.ndarray, np.ndarray]:
    """(W_eff, b_eff) de h → c = (θ_T, θ_S, u), incluindo a diagonal D da NFHn (escala normalizada)."""
    D = head.D.detach().double().cpu().numpy()
    W = head.lin.weight.detach().double().cpu().numpy() * D[:, None]
    b = head.lin.bias.detach().double().cpu().numpy() * D
    return W, b


def structural_properties(head: NeuralFourierHead, h: np.ndarray, k_cycle: Optional[int] = None):
    """(w, w0, ξ, info) de P_T e P_S para a instância h* (Definição 4.11)."""
    W, b = head_coefficients(head)
    P = head.P.detach().double().cpu().numpy()
    q1 = head.q1
    k_cycle = k_cycle or max(1, min(head.n_harmonics, round(head.horizon / 24)))
    tH = P[:, -1] - P[:, 0]                          # Δ_T = (P_{:,H−1} − P_{:,0})ᵀ θ_T
    dT = float(tH @ (W[:q1] @ h + b[:q1]))
    s = 1.0 if dT >= 0 else -1.0
    props = {"P_T": (s * (W[:q1].T @ tH), s * float(tH @ b[:q1]), abs(dT) / 2, {"delta_T": dT})}
    ia, ib = q1 + 2 * (k_cycle - 1), q1 + 2 * (k_cycle - 1) + 1
    a, bb = float(W[ia] @ h + b[ia]), float(W[ib] @ h + b[ib])
    A = float(np.hypot(a, bb))
    if A > 0:
        e = np.array([a, bb]) / A
        props["P_S"] = (e[0] * W[ia] + e[1] * W[ib], e[0] * b[ia] + e[1] * b[ib], A / 2,
                        {"A_k": A, "phase": float(np.arctan2(bb, a)), "harmonic": k_cycle,
                         "period": head.horizon / k_cycle})
    return props


def minimal_explanation(w: np.ndarray, w0: float, h: np.ndarray, lo: np.ndarray, hi: np.ndarray, xi: float,
                        mu: Optional[np.ndarray] = None) -> Dict:
    """Explicação abdutiva mínima em cardinalidade de wᵀh + w0 ≥ ξ (vetorizada, O(p log p))."""
    worst = np.minimum(w * lo, w * hi)
    gains = w * h - worst
    g0 = w0 + worst.sum()
    order = np.argsort(-gains, kind="stable")
    acc = g0 + np.concatenate([[0.0], np.cumsum(gains[order])])
    ok = acc >= xi - 1e-12
    k = int(np.argmax(ok)) if ok.any() else -1
    out = {"k": k, "neurons": order[:max(k, 0)].tolist(), "gamma_empty": float(g0), "threshold": float(xi),
           "value": float(w @ h + w0), "gains": gains, "order": order}
    if mu is not None:  # ordem dos Integrated Gradients de uma função afim com referência na média: |w_i (h_i − μ_i)|
        order_ig = np.argsort(-np.abs(w * (h - mu)), kind="stable")
        acc_ig = g0 + np.concatenate([[0.0], np.cumsum(gains[order_ig])])
        ok_ig = acc_ig >= xi - 1e-12
        out["k_ig"] = int(np.argmax(ok_ig)) if ok_ig.any() else -1
    return out


def abductive_explanations(model, h: np.ndarray, box: Dict[str, np.ndarray], k_cycle: Optional[int] = None,
                           n_check: int = 10000, seed: int = 0) -> Dict[str, Dict]:
    """P_T e P_S para a instância h*: tamanho mínimo, neurônios, ordem dos IG e verificação por amostragem."""
    head = model.head
    h = np.asarray(h, dtype=np.float64)
    lo, hi = np.minimum(box["lo"], h), np.maximum(box["hi"], h)
    rng = np.random.default_rng(seed)
    out = {}
    for name, (w, w0, xi, info) in structural_properties(head, h, k_cycle).items():
        r = minimal_explanation(w, w0, h, lo, hi, xi, box.get("mu"))
        viol = 0
        if n_check and r["k"] >= 0:
            sample = rng.uniform(lo, hi, size=(n_check, len(h)))
            sample[:, r["neurons"]] = h[r["neurons"]]
            viol = int(((sample @ w + w0) < xi - 1e-9).sum())
        out[name] = {"k": r["k"], "k_ig": r.get("k_ig"), "p": len(h), "neurons": r["neurons"],
                     "gamma_empty": r["gamma_empty"], "threshold": r["threshold"], "value": r["value"],
                     "violations": viol, "n_check": n_check,
                     "outside_train_box": bool((h < box["lo"]).any() or (h > box["hi"]).any()), **info}
    return out


# --------------------------------------------------------------------------- #
# Versão à mão (didática), neurônio a neurônio
# --------------------------------------------------------------------------- #
def manual_minimal_explanation(w: List[float], w0: float, h: List[float], lo: List[float], hi: List[float],
                               xi: float) -> Dict:
    """O mesmo algoritmo com laços explícitos: pior caso, ganhos, ordenação e acúmulo, passo a passo."""
    p = len(w)
    worst = [min(w[i] * lo[i], w[i] * hi[i]) for i in range(p)]
    gains = [w[i] * h[i] - worst[i] for i in range(p)]
    gamma = w0 + sum(worst)
    steps = [{"fixed": [], "margin": gamma}]
    order = sorted(range(p), key=lambda i: -gains[i])
    chosen: List[int] = []
    for i in order:
        if gamma >= xi:
            break
        chosen.append(i)
        gamma += gains[i]
        steps.append({"fixed": list(chosen), "margin": gamma})
    return {"neurons": chosen, "k": len(chosen) if gamma >= xi else -1, "gains": gains, "worst": worst,
            "steps": steps}


def brute_force_minimum(w, w0, h, lo, hi, xi) -> int:
    """Menor tamanho de conjunto suficiente por enumeração dos 2^p subconjuntos (só para p pequeno)."""
    from itertools import combinations

    w, h, lo, hi = map(np.asarray, (w, h, lo, hi))
    p = len(w)
    worst = np.minimum(w * lo, w * hi)
    for k in range(p + 1):
        for S in combinations(range(p), k):
            S = list(S)
            margin = w0 + worst.sum() + (w[S] * h[S] - worst[S]).sum()
            if margin >= xi - 1e-12:
                return k
    return -1
