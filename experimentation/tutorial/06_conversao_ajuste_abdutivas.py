"""
Tutorial 06: conversão exata, ajuste estrutural e explicações abdutivas, passo a passo (dissertação v3).

  1. A explicação abdutiva à mão: o Exemplo 4.17 da dissertação (p = 4), com o pior caso, os ganhos e o
     acúmulo, e a conferência por enumeração dos 2⁴ subconjuntos.
  2. A rede densa do tutorial 04 é convertida EXATAMENTE em NFH (mesma previsão).
  3. A época 0 é a decomposição pura: quanto da forma a base estrutural já explica.
  4. O ajuste estrutural, época a época, e a seleção segura (RMSE de validação ≤ o da densa).
  5. Explicações abdutivas de uma previsão de teste: P_T (sinal da tendência) e P_S (ciclo de 24 h),
     verificadas por amostragem, e a comparação com a ordem dos Integrated Gradients.
  6. As mesmas explicações em todas as janelas de teste: tamanho |𝒜|/p e violações.

Pré-requisito: tutorial/04_experimento_bike_sharing.py (modelos em results/04_bike_sharing/modelos).

    python tutorial/06_conversao_ajuste_abdutivas.py [--backbone lstm|transformer] [--janela N]
"""
from __future__ import annotations

import json

import numpy as np
import torch

from _common import RESULTS, check, header, out_dir, parser, pct, step

from nfh import AdjustConfig, Forecaster, abductive_explanations, energy_balance, latent_box, structural_adjustment
from nfh.abductive import brute_force_minimum, manual_minimal_explanation
from nfh.data import load_bike_sharing, make_splits

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--backbone", default="lstm", choices=["lstm", "transformer"])
    p.add_argument("--janela", type=int, default=None, help="índice da janela de teste (padrão: a do meio)")
    p.add_argument("--epocas", type=int, default=15, help="épocas do ajuste estrutural (E)")
    args = p.parse_args()
    outd = out_dir(args, "06_conversao_ajuste_abdutivas")
    header("TUTORIAL 06 · Conversão exata, ajuste estrutural e explicações abdutivas",
           "Procedimento da versão 3 da dissertação, aplicado a um modelo denso treinado no tutorial 04.")
    report = {}

    # ------------------------------------------------------------------ 1
    step(1, "Explicação abdutiva à mão (Exemplo 4.17 da dissertação)",
         """A propriedade é afim em h: g(h) = wᵀh + w0 ≥ ξ. Com o neurônio i livre na caixa [lo_i, hi_i], o
         pior caso é min(w_i lo_i, w_i hi_i); fixá-lo em h*_i rende o GANHO δ_i = w_i h*_i − min(...). Parte-se
         de todos livres (Γ(∅)) e fixam-se os neurônios em ordem decrescente de ganho até a margem chegar a ξ.
         Os números são os do exemplo, arredondados a 3 casas, e por isso as margens coincidem até 10⁻³.""")
    w = [-0.628, -0.010, -0.372, -0.123]
    h = [-0.252, 0.394, 0.391, 0.918]
    w0, xi = 0.155, 0.0253
    lo, hi = [-1.0] * 4, [1.0] * 4
    m = manual_minimal_explanation(w, w0, h, lo, hi, xi)
    print("   pior caso por neurônio  min(w_i lo_i, w_i hi_i) = " + ", ".join(f"{v:+.3f}" for v in m["worst"]))
    print("   ganhos δ_i = w_i h*_i − pior caso              = " + ", ".join(f"{v:.3f}" for v in m["gains"]))
    for s in m["steps"]:
        fixed = "{" + ", ".join(f"h{i + 1}" for i in s["fixed"]) + "}"
        print(f"   fixados {fixed:12s} → margem garantida {s['margin']:+.4f}  {'≥' if s['margin'] >= xi else '<'} ξ = {xi}")
    bf = brute_force_minimum(w, w0, h, lo, hi, xi)
    print(f"   → explicação mínima {{{', '.join(f'h{i + 1}' for i in m['neurons'])}}} (k = {m['k']}); "
          f"enumeração dos 2⁴ subconjuntos: menor k = {bf}")
    order0 = sorted(range(4), key=lambda i: -abs(w[i] * h[i]))
    print("   ordem por |w_i h*_i| (IG com referência nula): " + ", ".join(f"h{i + 1}" for i in order0)
          + ". A dissertação mostra que, com referência na média de treino, os IG pediriam 3 neurônios.")
    report["exemplo_4_17"] = {"k": m["k"], "neurons": [i + 1 for i in m["neurons"]], "k_enumeracao": bf}

    # ------------------------------------------------------------------ 2
    path = RESULTS / "04_bike_sharing" / "modelos" / f"{args.backbone}_dense_s0.pt"
    if not path.exists():
        raise SystemExit(f"Modelo {path} não encontrado: rode antes tutorial/04_experimento_bike_sharing.py")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["config"]
    df = load_bike_sharing()
    split = make_splits(df, "cnt", c["features"], c["L"], c["H"], standardize=c["standardize"])
    revin = [c["features"].index(v) for v in c["continuous"]]
    dense = Forecaster(c["backbone"], len(c["features"]), c["H"], target_idx=0, head="dense", n_harmonics=c["K"],
                       degree=c["q"], normalize_columns=True, revin=revin)
    dense.load_state_dict(ck["state_dict"])
    dense = dense.to(DEV).eval()
    Xtr, _ = split.train.tensors()
    Xva, Yva = split.val.tensors()
    Xte, Yte = split.test.tensors()

    step(2, f"Conversão exata da camada densa ({args.backbone.upper()}, K = {c['K']}, q = {c['q']})",
         """A densa ŷ = W'h + b' vira NFH com W = (M D)⁻¹ W' e b = (M D)⁻¹ b' (Teorema da equivalência):
         mesma função, agora com coordenadas (θ_T, θ_S, u) que somam tendência + sazonalidade + resíduo.""")
    nfh0 = dense.to_nfh().eval()
    with torch.no_grad():
        d_out, n_out = dense(Xte.to(DEV)), nfh0(Xte.to(DEV))
    err = float((d_out.forecast - n_out.forecast).abs().max())
    check("máx |ŷ_densa − ŷ_NFH| em todas as janelas de teste", err, 1e-4)
    check("máx |ŷ − (T + S + R)|", float((n_out.forecast - n_out.trend - n_out.seasonal - n_out.residual).abs().max()), 1e-4)
    report["erro_conversao"] = err

    # ------------------------------------------------------------------ 3
    step(3, "Época 0: a decomposição pura da previsão densa",
         """Antes de qualquer ajuste, a NFH convertida é a leitura da densa na base estrutural. A cobertura
         κ = (ε_T + ε_S) diz quanto da forma prevista essa leitura explica; o resto é resíduo R.""")
    e0 = energy_balance(n_out)
    cov0 = e0["coverage"]
    print(f"   cobertura no teste: mediana {pct(np.median(cov0))}, mínima {pct(cov0.min())}, "
          f"janelas com < 95%: {pct((cov0 < 0.95).mean())}")
    report["decomposicao_pura"] = {"cobertura_mediana": float(np.median(cov0)), "cobertura_min": float(cov0.min())}

    # ------------------------------------------------------------------ 4
    step(4, f"Ajuste estrutural ({args.epocas} épocas, λ = 1000, taxa 10⁻⁴) e seleção segura",
         """Treina-se o modelo inteiro com J = MSE + λ‖R‖²/H. A cada época mede-se, na validação, o RMSE r_ν e
         J_ν = r_ν² + λρ_ν. Escolhe-se a época de menor J entre as que têm r_ν ≤ r_0 (o RMSE da densa).
         Como a época 0 é viável, a NFH escolhida nunca é pior que a densa NA VALIDAÇÃO.""")
    torch.manual_seed(0)
    adj = structural_adjustment(dense.to_nfh(), split.train, split.val,
                                AdjustConfig(epochs=args.epocas, lr=1e-4, lam=1000.0, batch_size=64, device=DEV))
    print(f"   {'época':>5s} {'RMSE val':>9s} {'J val':>9s} {'κ val':>7s}  viável")
    for t in adj.trajectory:
        mark = "  ← seleção segura" if t["epoch"] == adj.selected_epoch else ""
        print(f"   {t['epoch']:5d} {t['rmse']:9.4f} {t['J']:9.4f} {pct(t['kappa']):>7s}  {'sim' if t['feasible'] else 'não'}{mark}")
    model = adj.model.eval()
    with torch.no_grad():
        out = model(Xte.to(DEV))
    rm = lambda P: float(torch.sqrt(((P.cpu() - Yte) ** 2).mean()))  # noqa: E731
    cov = energy_balance(out)["coverage"]
    print(f"   teste: RMSE densa {rm(d_out.forecast):.4f} → NFH {rm(out.forecast):.4f}; cobertura mediana "
          f"{pct(np.median(cov0))} → {pct(np.median(cov))}, mínima {pct(cov0.min())} → {pct(cov.min())}")
    print("   (a seleção garante a validação; no teste a NFH pode ficar um pouco melhor ou pior que a densa)")
    report["ajuste"] = {"epoca_selecionada": adj.selected_epoch, "rmse_val_densa": adj.rmse_val_dense,
                        "rmse_val_nfh": adj.selected["rmse"], "rmse_teste_densa": rm(d_out.forecast),
                        "rmse_teste_nfh": rm(out.forecast), "cobertura_teste_mediana": float(np.median(cov))}

    # ------------------------------------------------------------------ 5
    i = args.janela if args.janela is not None else len(Xte) // 2
    step(5, f"Explicações abdutivas da janela de teste {i}",
         """A caixa é empírica: mínimo e máximo de cada neurônio de h no treino (estendida para conter h*).
         P_T: a tendência prevista mantém o sinal com ao menos metade da variação Δ_T. P_S: o ciclo de 24 h
         mantém ao menos metade da amplitude, na fase prevista (diferença de fase < π/2). A explicação é o
         menor conjunto de neurônios que, fixados, garante a propriedade para QUALQUER valor dos demais na
         caixa. A garantia vale no espaço de h, não diretamente para as janelas de entrada.""")
    box = latent_box(model, Xtr)
    h_star = out.latent[i].double().cpu().numpy()
    ab = abductive_explanations(model, h_star, box, k_cycle=1, n_check=10000)
    pT, pS = ab["P_T"], ab["P_S"]
    print(f"   p = {pT['p']} neurônios em h")
    print(f"   P_T: Δ_T = {pT['delta_T']:+.4f} → k = {pT['k']} ({pct(pT['k'] / pT['p'])} de p); ordem dos IG precisaria de "
          f"{pT['k_ig']}; violações em {pT['n_check']} completamentos: {pT['violations']}")
    print(f"   P_S: A_1 = {pS['A_k']:.4f}, fase {pS['phase']:+.2f} rad (período {pS['period']:.0f} h) → k = {pS['k']} "
          f"({pct(pS['k'] / pS['p'])}); IG: {pS['k_ig']}; violações: {pS['violations']}")
    both = sorted(set(pT["neurons"]) & set(pS["neurons"]))
    print(f"   neurônios nas duas explicações: {len(both)}; fora da caixa de treino: {pT['outside_train_box']}")
    report["janela"] = {"indice": i, "P_T": {k: pT[k] for k in ("k", "k_ig", "p", "violations", "delta_T")},
                        "P_S": {k: pS[k] for k in ("k", "k_ig", "p", "violations", "A_k")}}

    # ------------------------------------------------------------------ 6
    step(6, "As explicações em todas as janelas de teste",
         "Tamanho relativo |𝒜|/p, razão k_IG/k e violações por amostragem (1000 completamentos por janela).")
    H = out.latent.double().cpu().numpy()
    stats = {"P_T": [], "P_S": []}
    for j in range(len(H)):
        r = abductive_explanations(model, H[j], box, k_cycle=1, n_check=1000, seed=j)
        for name in stats:
            stats[name].append((r[name]["k"], r[name]["k_ig"], r[name]["violations"], r[name]["p"]))
    summary = {}
    for name, rows in stats.items():
        a = np.array(rows, dtype=float)
        valid = a[:, 0] > 0
        frac, ratio = a[:, 0] / a[:, 3], a[valid, 1] / a[valid, 0]
        summary[name] = {"janelas": len(a), "frac_mediana": float(np.median(frac)), "frac_p10": float(np.quantile(frac, .1)),
                         "frac_p90": float(np.quantile(frac, .9)), "kig_sobre_k_mediana": float(np.median(ratio)),
                         "violacoes": int(a[:, 2].sum()), "k_zero": int((~valid).sum())}
        s = summary[name]
        print(f"   {name}: |𝒜|/p mediana {pct(s['frac_mediana'])} (p10–p90 {pct(s['frac_p10'])}–{pct(s['frac_p90'])}); "
              f"k_IG/k mediana {s['kig_sobre_k_mediana']:.2f}; violações {s['violacoes']}; janelas com k = 0: {s['k_zero']}")
    report["todas_as_janelas"] = summary
    (outd / "resumo.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n   Resumo salvo em {outd / 'resumo.json'}")


if __name__ == "__main__":
    main()
