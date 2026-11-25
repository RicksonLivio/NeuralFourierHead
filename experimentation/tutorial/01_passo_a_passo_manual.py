"""
Tutorial 01 — A NFH passo a passo, à mão (versor a versor).

Reproduz, conta por conta, o exemplo univariado da dissertação (ETTh1, H = 8,
q = 1, K = 2): constrói as bases, sintetiza uma previsão, decompõe-na e calcula
o balanço de energia usando só produtos internos entre vetores
(``nfh.manual``). No fim, confere que a biblioteca vetorizada dá o mesmo.

    python tutorial/01_passo_a_passo_manual.py [--graficos]
"""
from __future__ import annotations

import math

import torch

from _common import check, header, out_dir, parser, pct, show_matrix, step

from nfh import build_bases, manual  # noqa: E402

H, K, q = 8, 2, 1  # horizonte de 8 h, 2 harmônicos (8 h e 4 h), tendência afim

# Previsão do exemplo univariado da dissertação (Tabela "Exemplo univariado", 4 casas)
Y_HAT = [-0.7738, -0.7720, -0.7722, -0.7627, -0.7532, -0.7656, -0.7622, -0.7693]


def main() -> None:
    args = parser(__doc__).parse_args()
    header("TUTORIAL 01 · Neural Fourier Head à mão",
           f"Configuração: H = {H} passos, q = {q} (tendência afim), K = {K} harmônicos. "
           f"m = q + 1 + 2K = {q + 1 + 2 * K} ≤ H, logo a condição de dimensão (que contém Nyquist) vale.")

    # ------------------------------------------------------------------ 1
    step(1, "Grade do horizonte", "Os H passos futuros são indexados por τ_j = j/H, j = 0..H−1.")
    tau = manual.time_grid(H)
    show_matrix("τ", tau)

    # ------------------------------------------------------------------ 2
    step(2, "Vetores de tendência (linhas de P)",
         "Cada vetor p_i tem coordenadas τ_j^i. Com q = 1: o vetor constante (nível) e a rampa (inclinação).")
    P = manual.trend_vectors(H, q)
    for i, p in enumerate(P):
        show_matrix(f"p_{i} = τ^{i}", p)

    # ------------------------------------------------------------------ 3
    step(3, "Vetores de Fourier (linhas de F)",
         "Para cada harmônico k há um cosseno e um seno que oscilam k vezes no horizonte: período H/k passos.")
    F = manual.fourier_vectors(H, K)
    for k in range(K):
        show_matrix(f"cos k={k + 1} (período {H / (k + 1):.0f})", F[2 * k])
        show_matrix(f"sin k={k + 1}", F[2 * k + 1])

    # ------------------------------------------------------------------ 4
    step(4, "Remover a parte polinomial das senoides: F̃ = F(I − Π_T)",
         """Dentro de uma janela finita, uma senoide lenta se parece com uma rampa: parte dela vive no espaço
         de tendência 𝒯. Para que tendência e sazonalidade não se sobreponham, projetamos cada senoide em 𝒯
         e subtraímos essa projeção. Para projetar, primeiro ortonormalizamos p_0, p_1 (Gram-Schmidt):
         Π_T f = Σ_i ⟨f, e_i⟩ e_i.""")
    e_trend = manual.gram_schmidt(P)
    for i, e in enumerate(e_trend):
        show_matrix(f"e_{i} (versor de 𝒯)", e)
    Ft = [f - manual.project(f, e_trend) for f in F]
    show_matrix("f̃ = cos k=1 − Π_T(cos k=1)", Ft[0])
    print("   Ortogonalidade com a tendência (deve ser 0):")
    check("max |⟨p_i, f̃_j⟩|", max(abs(manual.dot(p, f)) for p in P for f in Ft), 1e-12)

    # ------------------------------------------------------------------ 5
    step(5, "Completar o espaço com o resíduo: Q⊥",
         f"""Tendência e sazonalidade ocupam m = {q + 1 + 2 * K} dimensões de R^{H}. As H − m = {H - q - 1 - 2 * K}
         restantes formam o espaço residual ℛ. Uma base ortonormal dele (Q⊥) sai de Gram-Schmidt aplicado à
         base canônica, depois dos versores de tendência e sazonalidade.""")
    vecs = manual.nfh_vectors(H, K, q)
    for j, r in enumerate(vecs["resid"]):
        show_matrix(f"r_{j + 1} (versor de ℛ)", r)
    tS, tR, sR = manual.orthogonality_report(vecs)
    check("max |⟨𝒯, 𝒮⟩|", tS, 1e-12)
    check("max |⟨𝒯, ℛ⟩|", tR, 1e-12)
    check("max |⟨𝒮, ℛ⟩|", sR, 1e-12)

    # ------------------------------------------------------------------ 6
    step(6, "Explicar uma previsão: identificabilidade",
         """Pegamos a previsão ŷ do exemplo da dissertação e recuperamos, só a partir dela, os coeficientes
         (θ_T, θ_S, u): θ_T = (PPᵀ)⁻¹Pŷ, θ_S = (F̃F̃ᵀ)⁻¹F̃ŷ, u = Q⊥ᵀŷ. A resposta é única porque os três
         espaços são ortogonais.""")
    y = torch.tensor(Y_HAT, dtype=torch.float64)
    show_matrix("ŷ", y, 4)
    d = manual.decompose(y, vecs)
    print(f"   θ_T = {[round(v, 4) for v in d['theta_T']]}   (dissertação: [-0.771, 0.0105])")
    print(f"   θ_S = {[round(v, 4) for v in d['theta_S']]}   (dissertação: [-0.007, -0.0013, 0.0023, -0.0009])")
    for k in range(K):
        a, b = d["theta_S"][2 * k], d["theta_S"][2 * k + 1]
        print(f"   harmônico k={k + 1}: A_k = √(a² + b²) = {math.hypot(a, b):.4f}, fase φ_k = {math.atan2(b, a):+.2f} rad")
    show_matrix("T", d["T"], 4)
    show_matrix("S", d["S"], 4)
    show_matrix("R", d["R"], 4)
    check("exatidão max |T + S + R − ŷ|", d["reconstruction_error"], 1e-12)

    # ------------------------------------------------------------------ 7
    step(7, "Balanço de energia da forma",
         """O nível (média do horizonte) pertence todo à tendência. Tirando o nível, ‖ŷ_c‖² = ‖T_c‖² + ‖S‖² + ‖R‖²
         (Pitágoras, porque as partes são ortogonais), então as frações somam exatamente 100%.""")
    eb = manual.energy_balance(d["T"], d["S"], d["R"])
    ek = manual.harmonic_shares(d["theta_S"], vecs, y)
    print(f"   nível ȳ = {eb['level']:.4f}")
    print(f"   ε_T = {pct(eb['eps_T'])}   ε_S = {pct(eb['eps_S'])}   ε_R = {pct(eb['eps_R'])}   "
          f"(dissertação: 21,6% · 59,0% · 19,5%)")
    print(f"   e_1 (8 h) = {pct(ek[0])}, e_2 (4 h) = {pct(ek[1])}   (dissertação: 50,2% e 8,7%)")
    check("soma ε_T + ε_S + ε_R − 1", eb["eps_T"] + eb["eps_S"] + eb["eps_R"] - 1, 1e-12)
    check("termo cruzado 2⟨T_c, S⟩/‖ŷ_c‖²", eb["cross"], 1e-12)
    print("   (diferenças na 1ª casa decimal vêm do arredondamento de ŷ a 4 casas)")

    # ------------------------------------------------------------------ 8
    step(8, "Por que remover a parte polinomial? (base sem ortogonalizar)",
         """Decompomos a mesma previsão na base crua [Pᵀ Fᵀ Q⊥] (resolvendo o sistema conjunto, pois os vetores
         não são ortogonais): as parcelas ainda somam ŷ, mas tendência e senoides se sobrepõem e as frações de
         energia deixam de somar 100%. A sobra é um termo cruzado sem interpretação.""")
    dr = manual.oblique_decompose(y, {"T": P, "S": F, "R": vecs["resid"]})
    ebr = manual.energy_balance(dr["T"], dr["S"], dr["R"])
    print(f"   base crua: ε_T = {pct(ebr['eps_T'])}, ε_S = {pct(ebr['eps_S'])}, ε_R = {pct(ebr['eps_R'])} → soma "
          f"{pct(ebr['eps_T'] + ebr['eps_S'] + ebr['eps_R'])}, termo cruzado {pct(ebr['cross'])}  "
          f"(dissertação: soma 90,1%)")
    check("as parcelas ainda somam ŷ", float((dr["T"] + dr["S"] + dr["R"] - y).abs().max()), 1e-12)

    # ------------------------------------------------------------------ 9
    step(9, "Conferência com a biblioteca vetorizada (torch.linalg)")
    b = build_bases(H, K, q, dtype=torch.float64)
    check("|P_manual − P_lib|", float((torch.stack(P) - b.P).abs().max()), 1e-12)
    check("|F̃_manual − F̃_lib|", float((torch.stack(Ft) - b.Ft).abs().max()), 1e-12)
    Qm = torch.stack(vecs["resid"], 1)
    check("|Q⊥Q⊥ᵀ manual − lib| (Q⊥ é única a menos de rotação)", float((Qm @ Qm.T - b.Q @ b.Q.T).abs().max()), 1e-12)

    if args.graficos:
        from nfh.plots import plot_bases
        path = out_dir(args, "01_passo_a_passo") / "bases_H8.png"
        plot_bases(b, path=path, harmonics=(1, 2))
        print(f"\n📈 Gráfico salvo em {path}")
    print("\nPróximo: tutorial/02_garantias_numericas.py")


if __name__ == "__main__":
    main()
