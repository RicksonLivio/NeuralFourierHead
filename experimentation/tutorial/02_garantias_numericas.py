"""
Tutorial 02 — As garantias da NFH, verificadas numericamente com a biblioteca.

Cada passo corresponde a um resultado da dissertação (Cap. 4). Usa a
implementação vetorizada (``torch.linalg``) com a configuração dos
experimentos da dissertação: H = 30, q = 2, K = 10.

    python tutorial/02_garantias_numericas.py [--graficos]
"""
from __future__ import annotations

import numpy as np
import torch

from _common import check, header, out_dir, parser, pct, step

from nfh import NeuralFourierHead, build_bases, check_dimensions, energy_balance, harmonic_shares, max_harmonics

H, K, q, p = 30, 10, 2, 128


def main() -> None:
    args = parser(__doc__).parse_args()
    torch.manual_seed(0)
    header("TUTORIAL 02 · Garantias da NFH", f"H = {H}, K = {K}, q = {q}, representação latente p = {p}.")
    b = build_bases(H, K, q, dtype=torch.float64)

    step(1, "Condição de dimensão (Lema da independência, contém Nyquist)",
         "m = q + 1 + 2K ≤ H garante que tendência e senoides são linearmente independentes.")
    print(f"   m = {check_dimensions(H, K, q)} ≤ H = {H};  maior K permitido: {max_harmonics(H, q)}")
    for bad in (15, 14):
        try:
            check_dimensions(H, bad, q)
        except ValueError as e:
            print(f"   K = {bad} recusado: {e}")

    step(2, "Decomposição ortogonal: R^H = 𝒯 ⊕ 𝒮 ⊕ ℛ")
    check("max |P F̃ᵀ|", float((b.P @ b.Ft.T).abs().max()), 1e-10)
    check("max |P Q⊥|", float((b.P @ b.Q).abs().max()), 1e-10)
    check("max |F̃ Q⊥|", float((b.Ft @ b.Q).abs().max()), 1e-10)
    print(f"   dim 𝒯 = {q + 1}, dim 𝒮 = {2 * K}, dim ℛ = {H - b.m}, posto de M = {int(torch.linalg.matrix_rank(b.M))}")

    head = NeuralFourierHead(p, H, K, q).double()
    h = torch.randn(256, p, dtype=torch.float64)
    with torch.no_grad():
        out = head(h)

    step(3, "Exatidão: ŷ = T + S + R")
    check("max |ŷ − (T + S + R)|", float((out.forecast - out.trend - out.seasonal - out.residual).abs().max()), 1e-12)

    step(4, "Identificabilidade: a explicação depende só da previsão",
         """Recuperamos (θ_T, θ_S, u) a partir de ŷ. Depois, convertemos a NFH numa camada densa com pesos
         completamente diferentes (to_linear) e decompomos a previsão dela: a explicação é idêntica.""")
    dec = head.decompose(out.forecast)
    check("max |θ_S recuperado − θ_S|", float((dec.theta_S - out.theta_S).abs().max()), 1e-9)
    dense = head.to_linear()
    with torch.no_grad():
        yd = dense(h)
    dd = head.decompose(yd)
    print(f"   ‖W_densa − W_NFH‖ = {float((dense.weight - head.lin.weight).norm()):.2f} (pesos diferentes)")
    check("max |amplitudes(densa) − amplitudes(NFH)|", float((dd.amplitude - out.amplitude).abs().max()), 1e-9)

    step(5, "Contraste: no N-BEATS-I a divisão NÃO é única",
         """No N-BEATS-I o vetor constante está nas duas bases (tendência e sazonalidade). Logo podemos mover
         qualquer parcela de nível de uma para a outra sem mudar a previsão: infinitas explicações.""")
    y0 = out.forecast[0]
    T0, S0 = out.trend[0], out.seasonal[0] + out.residual[0]
    for shift in (0.0, 0.5, -1.0):
        T1, S1 = T0 + shift, S0 - shift  # 𝟙 pertence às duas bases do N-BEATS-I
        E = lambda v: float((v ** 2).sum())  # noqa: E731
        print(f"   deslocamento {shift:+.1f}: soma idêntica ({float((T1 + S1 - y0).abs().max()):.0e}), "
              f"‖T‖² = {E(T1):7.2f}, ‖S‖² = {E(S1):7.2f}")

    step(6, "Equivalência com a camada densa: mesmos parâmetros e mesmas funções")
    n_nfh = sum(t.numel() for t in head.parameters())
    print(f"   parâmetros NFH = {n_nfh} = H(p + 1) = {H * (p + 1)}  (camada densa: {sum(t.numel() for t in dense.parameters())})")
    back = NeuralFourierHead.from_linear(dense, K, q)
    with torch.no_grad():
        check("max |NFH.from_linear(densa)(h) − densa(h)|", float((back(h).forecast - yd).abs().max()), 1e-9)

    step(7, "Balanço de energia: ε_T + ε_S + ε_R = 1 e Σ e_k = ε_S")
    e = energy_balance(out)
    ek = harmonic_shares(out, head.Ft)
    check("max |ε_T + ε_S + ε_R − 1|", float(np.abs(e["eps_T"] + e["eps_S"] + e["eps_R"] - 1).max()), 1e-10)
    check("max |termo cruzado|", float(np.abs(e["iota"]).max()), 1e-10)
    check("max |Σ e_k − ε_S|", float(np.abs(ek.sum(1) - e["eps_S"]).max()), 1e-10)
    check("nível = média de T", float(np.abs(e["level"] - out.trend.mean(1).numpy()).max()), 1e-12)
    print(f"   exemplo (amostra 0): ε_T {pct(e['eps_T'][0])}, ε_S {pct(e['eps_S'][0])}, ε_R {pct(e['eps_R'][0])}")
    print(f"   e_k negativos no lote: {int((ek < 0).sum())} de {ek.size} (cancelamento parcial entre harmônicos)")

    step(8, "Fluxo de gradiente: ∂ℓ/∂c = (2/H) Mᵀ(ŷ − y)")
    head32 = NeuralFourierHead(p, H, K, q).double()
    hh = torch.randn(1, p, dtype=torch.float64)
    y = torch.randn(1, H, dtype=torch.float64)
    c = head32.coordinates(hh).detach().requires_grad_(True)
    o = head32.synthesize(*head32.split(c))
    loss = ((o.forecast - y) ** 2).mean()
    loss.backward()
    expected = (2 / H) * (head32.M.T @ (o.forecast - y).detach().T).T
    check("max |∂ℓ/∂c − (2/H)Mᵀ(ŷ − y)|", float((c.grad - expected).abs().max()), 1e-12)

    step(9, "Estabilidade: harmônicos longos são mais sensíveis (Prop. da estabilidade)",
         """γ_k mede quanto a amplitude do harmônico k pode variar para uma mudança unitária da previsão. Na base
         sem tendência seria √(2/H) para todo k; com a tendência removida, os ciclos de período longo ficam
         mal condicionados (parte deles se confunde com uma tendência suave).""")
    cond = b.seasonal_condition()
    print(f"   σ_min(F̃) = {cond['sigma_min_Ft']:.3f};  referência √(2/H) = {cond['gamma_reference']:.3f}")
    print("   γ_k: " + "  ".join(f"k={k + 1}:{g:.2f}" for k, g in enumerate(cond["gamma_k"])))

    if args.graficos:
        from nfh.plots import plot_bases
        path = out_dir(args, "02_garantias") / "bases_H30.png"
        plot_bases(b.to(torch.float32), path=path, harmonics=(1, 2, 5))
        print(f"\n📈 Gráfico salvo em {path}")
    print("\nPróximo: tutorial/03_acoplando_em_qualquer_modelo.py")


if __name__ == "__main__":
    main()
