"""
Explicação da previsão (dissertação, Cap. 4, Seções "Balanço de energia" e
"Extensão multivariada").

Funções principais
------------------
energy_balance(out)        ε_T, ε_S, ε_R, cobertura κ = 1 − ε_R, nível, escala da forma
harmonic_shares(out, Ft)   e_k = ⟨S_k, S⟩ / ‖ŷ_c‖²   (Σ_k e_k = ε_S; pode ser negativo)
explain(model, x)          tudo acima + harmônicos dominantes para um lote
integrated_gradients(...)  atribuição de cada componente às variáveis de entrada
robustness(...)            cosseno entre explicações antes/depois de ruído
narrative(...)             explicação em tópicos (português)

Convenções
----------
* "Forma" da previsão = previsão centrada ŷ_c = ŷ − ȳ𝟙. O nível ȳ é explicado
  integralmente pela tendência (𝟙 ∈ 𝒯), então as participações descrevem a forma.
* As participações só são porcentagens legítimas porque T_c, S e R são
  ortogonais: ‖ŷ_c‖² = ‖T_c‖² + ‖S‖² + ‖R‖² (o termo cruzado ι é ~0, verificado).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .components import NFHOutput

__all__ = ["energy_balance", "harmonic_shares", "Explanation", "explain", "integrated_gradients", "quadrature",
           "robustness", "narrative"]


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().double().cpu().numpy()


def energy_balance(out: NFHOutput) -> Dict[str, np.ndarray]:
    """Balanço de energia da previsão centrada, por amostra do lote.

    Returns:
        dict com arrays (B,): eps_T, eps_S, eps_R, coverage, level, shape_rms e
        iota (termo cruzado 2⟨T_c, S⟩/‖ŷ_c‖², verificação numérica ≈ 0).
    """
    y, T = out.forecast.double(), out.trend.double()
    S, R = out.seasonal.double(), out.residual.double()
    yc = y - y.mean(-1, keepdim=True)
    Tc = T - T.mean(-1, keepdim=True)
    E = (yc ** 2).sum(-1) + 1e-30
    eT, eS, eR = (Tc ** 2).sum(-1) / E, (S ** 2).sum(-1) / E, (R ** 2).sum(-1) / E
    return {"eps_T": _np(eT), "eps_S": _np(eS), "eps_R": _np(eR), "coverage": _np(1 - eR),
            "level": _np(y.mean(-1)), "shape_rms": _np(torch.sqrt(E / y.shape[-1])),
            "iota": _np(2 * (Tc * S).sum(-1) / E)}


def harmonic_shares(out: NFHOutput, Ft: torch.Tensor) -> np.ndarray:
    """e_k (B, K): participação aditiva de cada harmônico na forma (Σ_k e_k = ε_S)."""
    y = out.forecast.double()
    yc = y - y.mean(-1, keepdim=True)
    E = (yc ** 2).sum(-1, keepdim=True) + 1e-30
    th = out.theta_S.double()
    S = out.seasonal.double()
    Ftd = Ft.double().to(th.device)
    K = th.shape[-1] // 2
    # S_k = a_k F̃_{2k-1} + b_k F̃_{2k}, todos os k de uma vez: (B, K, H)
    Sk = th[:, 0::2, None] * Ftd[None, 0::2] + th[:, 1::2, None] * Ftd[None, 1::2]
    return _np((Sk * S[:, None, :]).sum(-1) / E)[:, :K]


@dataclass
class Explanation:
    """Explicação estrutural de um lote de previsões (arrays numpy)."""

    forecast: np.ndarray          # (B, H)
    trend: np.ndarray             # (B, H)
    seasonal: np.ndarray          # (B, H)
    residual: np.ndarray          # (B, H)
    level: np.ndarray             # (B,)
    eps_T: np.ndarray             # (B,)
    eps_S: np.ndarray             # (B,)
    eps_R: np.ndarray             # (B,)
    coverage: np.ndarray          # (B,)
    shape_rms: np.ndarray         # (B,)
    amplitude: np.ndarray         # (B, K)
    phase: np.ndarray             # (B, K)
    shares: np.ndarray            # (B, K)  e_k
    periods: np.ndarray           # (K,)
    theta_T: np.ndarray           # (B, q+1)
    iota: np.ndarray              # (B,)
    attribution: Optional[Dict[str, Dict]] = field(default=None)

    def summary(self, i: int = 0, top: int = 3) -> Dict:
        order = np.argsort(-self.shares[i])[:top]
        return {
            "level": float(self.level[i]), "eps_T": float(self.eps_T[i]), "eps_S": float(self.eps_S[i]),
            "eps_R": float(self.eps_R[i]), "coverage": float(self.coverage[i]),
            "shape_rms": float(self.shape_rms[i]),
            "top_harmonics": [{"k": int(k + 1), "period": float(self.periods[k]), "e_k": float(self.shares[i, k]),
                               "amplitude": float(self.amplitude[i, k])} for k in order],
        }


@torch.no_grad()
def explain(model: torch.nn.Module, x: torch.Tensor) -> Explanation:
    """Previsão + explicação estrutural para um lote de janelas ``x``.

    ``model`` deve devolver um ``NFHOutput`` (ex.: ``nfh.model.NFHForecaster``)
    e ter o atributo ``head`` (``NeuralFourierHead``).
    """
    model.eval()
    out: NFHOutput = model(x)
    e = energy_balance(out)
    head = model.head
    return Explanation(
        forecast=_np(out.forecast), trend=_np(out.trend), seasonal=_np(out.seasonal), residual=_np(out.residual),
        level=e["level"], eps_T=e["eps_T"], eps_S=e["eps_S"], eps_R=e["eps_R"], coverage=e["coverage"],
        shape_rms=e["shape_rms"], amplitude=_np(out.amplitude), phase=_np(out.phase),
        shares=harmonic_shares(out, head.Ft), periods=_np(head.periods), theta_T=_np(out.theta_T), iota=e["iota"],
    )


# --------------------------------------------------------------------------- #
# Extensão multivariada: Integrated Gradients por componente
# --------------------------------------------------------------------------- #
FUNCTIONALS: Dict[str, Callable[[NFHOutput], torch.Tensor]] = {
    "T": lambda o: o.trend.mean(-1),               # g_T: nível médio previsto
    "S": lambda o: (o.seasonal ** 2).sum(-1),      # g_S: energia da sazonalidade
    "R": lambda o: (o.residual ** 2).sum(-1),      # g_R: energia do resíduo
}


def quadrature(steps: int, method: str = "gausslegendre"):
    """Nós alpha em (0, 1) e pesos (somam 1) da integral de 0 a 1.

    * ``gausslegendre`` (padrão; a mesma da Captum e da dissertação v3): exata para polinômios de grau
      até 2·steps − 1, por isso o erro de completude é muito menor que o da regra do ponto médio.
    * ``midpoint``: regra do ponto médio (versão 2).
    """
    if method == "gausslegendre":
        x, w = np.polynomial.legendre.leggauss(steps)
        return 0.5 * (x + 1.0), 0.5 * w
    if method == "midpoint":
        return (np.arange(steps) + 0.5) / steps, np.full(steps, 1.0 / steps)
    raise ValueError("method deve ser 'gausslegendre' ou 'midpoint'")


def integrated_gradients(component_fn: Callable[[torch.Tensor], NFHOutput], x: torch.Tensor,
                         feature_names: Sequence[str], baseline: Optional[torch.Tensor] = None,
                         steps: int = 64, method: str = "gausslegendre") -> Dict[str, Dict]:
    """Atribuição de cada componente c ∈ {T, S, R} às variáveis (Eq. ig-componente).

        I_c(v) = Σ_ℓ (X_ℓv − X'_ℓv) ∫₀¹ ∂g_c(X' + α(X − X'))/∂X_ℓv dα

    A integral usa a quadratura de Gauss-Legendre com ``steps`` pontos (``method``).
    Completude (Proposição): Σ_v I_c(v) = g_c(X) − g_c(X'); o erro é relatado.

    Args:
        component_fn: função janela → ``NFHOutput`` sobre a qual a atribuição é
            feita. Para modelos com RevIN, passe a janela JÁ normalizada e a
            função interna (``model.inner``), como na dissertação.
        x: janela (1, L, d).
        baseline: referência X' (padrão: janela nula). Atenção à semântica: com a janela
            já normalizada por RevIN, "nula" significa a média da PRÓPRIA janela nas variáveis
            normalizadas por instância (e a média de treino nas apenas padronizadas), como na
            dissertação. A completude vale para qualquer referência.
    """
    if x.shape[0] != 1:
        raise ValueError("integrated_gradients explica uma janela por vez (x com shape (1, L, d)).")
    base = torch.zeros_like(x) if baseline is None else baseline
    a_np, w_np = quadrature(steps, method)
    alphas = torch.tensor(a_np, dtype=x.dtype, device=x.device)
    weights = torch.tensor(w_np, dtype=x.dtype, device=x.device)
    result: Dict[str, Dict] = {}
    # cuDNN não faz backward de LSTM em modo eval: usa a implementação nativa
    with torch.backends.cudnn.flags(enabled=False):
        for key, g in FUNCTIONALS.items():
            path = (base + alphas.view(-1, 1, 1) * (x - base)).detach().requires_grad_(True)
            total = g(component_fn(path)).sum()
            grad = (torch.autograd.grad(total, path)[0] * weights.view(-1, 1, 1)).sum(0, keepdim=True)
            attr = ((x - base) * grad)[0]                          # (L, d)
            with torch.no_grad():
                delta = float(g(component_fn(x)) - g(component_fn(base)))
            per_var = _np(attr.sum(0))                             # I_c(v)
            share = np.abs(per_var) / (np.abs(per_var).sum() + 1e-30)
            result[key] = {"I": dict(zip(feature_names, per_var.tolist())),
                           "share": dict(zip(feature_names, share.tolist())),
                           "per_lag": _np(attr.abs().sum(1)).tolist(),
                           "delta": delta, "completeness_error": float(abs(per_var.sum() - delta))}
    return result


def robustness(model: torch.nn.Module, x: torch.Tensor, sigma: float = 0.05, n_draws: int = 3,
               seed: int = 0) -> np.ndarray:
    """Cosseno entre o vetor (ε_T, e_1..e_K, ε_R) da janela original e o de janelas
    com ruído N(0, σ²) (métrica de robustez da dissertação). Devolve (B,)."""
    def vec(xx):
        e = explain(model, xx)
        return np.concatenate([e.eps_T[:, None], e.shares, e.eps_R[:, None]], axis=1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    v0 = vec(x)
    sims = []
    for _ in range(n_draws):
        noise = torch.randn(x.shape, generator=g, dtype=x.dtype).to(x.device) * sigma
        v1 = vec(x + noise)
        sims.append((v0 * v1).sum(1) / (np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=1) + 1e-30))
    return np.mean(sims, axis=0)


# --------------------------------------------------------------------------- #
# Explicação em tópicos
# --------------------------------------------------------------------------- #
def _f(v: float, nd: int = 2) -> str:
    return f"{v:,.{nd}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def _p(v: float, nd: int = 1) -> str:
    return f"{_f(100 * v, nd)}%"


def narrative(exp: Explanation, i: int = 0, unit: str = "passos", target: str = "alvo",
              history_std: Optional[float] = None, attribution: Optional[Dict] = None) -> List[str]:
    """Explicação em tópicos (bullet points) de uma previsão, em português."""
    s = exp.summary(i)
    H = exp.forecast.shape[1]
    tr = exp.trend[i]
    lines = [
        f"Nível médio previsto de {_f(s['level'])} para '{target}' nos próximos {H} {unit}: "
        f"explicado inteiramente pela tendência.",
        f"Tendência {'crescente' if tr[-1] > tr[0] else 'decrescente'} de {_f(tr[0])} para {_f(tr[-1])} no horizonte.",
        f"Forma da previsão (sem o nível): tendência {_p(s['eps_T'])}, sazonalidade {_p(s['eps_S'])}, "
        f"resíduo {_p(s['eps_R'])}; cobertura estrutural κ = {_p(s['coverage'])}.",
    ]
    for h in s["top_harmonics"]:
        if h["e_k"] > 0.01:
            lines.append(f"Ciclo de ≈{_f(h['period'], 1)} {unit} (k={h['k']}): e_k = {_p(h['e_k'])} da forma, "
                         f"amplitude {_f(h['amplitude'])}.")
    if history_std:
        ratio = s["shape_rms"] / history_std
        lines.append(f"Escala da forma: RMS {_f(s['shape_rms'])} ({_p(ratio)} do desvio histórico)"
                     + (" — oscilação pequena; leia as porcentagens com cautela." if ratio < 0.15 else "."))
    if attribution:
        for c, lab in (("T", "nível/tendência"), ("S", "sazonalidade")):
            sh = sorted(attribution[c]["share"].items(), key=lambda kv: -kv[1])[:2]
            lines.append(f"{lab.capitalize()} sustentada principalmente por "
                         + " e ".join(f"'{k}' ({_p(v)})" for k, v in sh) + " (Integrated Gradients, leitura local).")
    return lines
