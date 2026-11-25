"""
Tutorial 03 — Acoplando a NFH a qualquer modelo PyTorch.

Quatro formas de uso:

A. Como camada de saída de um modelo seu (aqui, uma GRU), treinada de ponta a
   ponta com um laço de treino comum do PyTorch.
B. ``attach_nfh``: trocar a ``nn.Linear`` final de um modelo JÁ TREINADO por
   uma NFH equivalente — o modelo continua prevendo exatamente o mesmo e passa
   a explicar cada previsão.
C. ``head.decompose``: explicar a previsão de QUALQUER modelo (até de um
   baseline ingênuo) na base da NFH, pela identificabilidade.
D. ``NFHOutput.denormalize``: levar a decomposição para a escala original.

Os dados são sintéticos com componentes conhecidas (tendência + ciclos de 24 h
e 8 h + ruído), para comparar a explicação com a verdade.

    python tutorial/03_acoplando_em_qualquer_modelo.py [--graficos]
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from _common import header, out_dir, parser, pct, step

from nfh import NeuralFourierHead, attach_nfh, energy_balance, explain
from nfh.data import make_splits

L, H, K, q = 96, 24, 6, 1
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def synthetic(n: int = 6000, seed: int = 0):
    """y_t = tendência lenta + 1,0·sen(2πt/24) + 0,5·cos(2πt/8) + ruído AR(1)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.0004 * t + 0.5 * np.sin(2 * np.pi * t / 2000)
    seas = 1.0 * np.sin(2 * np.pi * t / 24) + 0.5 * np.cos(2 * np.pi * t / 8)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.6 * e[i - 1] + rng.normal(0, 0.15)
    import pandas as pd
    return pd.DataFrame({"y": trend + seas + e})


class GRUForecaster(nn.Module):
    """Um modelo "seu": GRU + NFH como camada de saída."""

    def __init__(self, n_features: int):
        super().__init__()
        self.gru = nn.GRU(n_features, 64, num_layers=2, batch_first=True)
        self.head = NeuralFourierHead(64, H, n_harmonics=K, degree=q)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])            # NFHOutput (forecast, trend, seasonal, ...)


class DenseGRU(nn.Module):
    """Modelo caixa-preta tradicional: GRU + nn.Linear (para o caso B)."""

    def __init__(self, n_features: int):
        super().__init__()
        self.gru = nn.GRU(n_features, 64, num_layers=2, batch_first=True)
        self.out = nn.Linear(64, H)

    def forward(self, x):
        _, h = self.gru(x)
        o = self.out(h[-1])
        return o.forecast if hasattr(o, "forecast") else o   # funciona antes e depois de attach_nfh


def train(model, Xtr, Ytr, epochs=8, lam=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            x, y = Xtr[idx].to(DEV), Ytr[idx].to(DEV)
            out = model(x)
            pred = out.forecast if hasattr(out, "forecast") else out
            loss = nn.functional.mse_loss(pred, y)
            if lam and hasattr(out, "residual"):
                loss = loss + lam * (out.residual ** 2).sum(-1).mean() / H   # penalização do resíduo
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def main() -> None:
    args = parser(__doc__).parse_args()
    torch.manual_seed(0)
    header("TUTORIAL 03 · Acoplando a NFH", f"Dados sintéticos com componentes conhecidas · L={L}, H={H}, K={K}, q={q} · {DEV}")
    split = make_splits(synthetic(), "y", ["y"], L, H)
    Xtr, Ytr = split.train.tensors()
    Xte, Yte = split.test.tensors()

    # ------------------------------------------------------------------ A
    step("A", "NFH como camada de saída de um modelo seu (GRU + NFH)",
         """Basta trocar nn.Linear(64, H) por NeuralFourierHead(64, H). A saída vira um NFHOutput; a perda usa
         out.forecast e, opcionalmente, λ‖R‖²/H para empurrar a forma da previsão para tendência e ciclos.""")
    model = train(GRUForecaster(1).to(DEV), Xtr, Ytr, lam=10.0)
    with torch.no_grad():
        out = model(Xte.to(DEV))
    rmse = float(((out.forecast.cpu() - Yte) ** 2).mean().sqrt())
    e = energy_balance(out)
    print(f"   RMSE de teste (padronizado): {rmse:.3f}")
    print(f"   forma média: ε_T {pct(e['eps_T'].mean())}, ε_S {pct(e['eps_S'].mean())}, ε_R {pct(e['eps_R'].mean())}")
    amp = out.amplitude.mean(0).cpu().numpy()
    print("   amplitude média por harmônico (verdade: k=1 → 24 h e k=3 → 8 h):")
    print("   " + "  ".join(f"k={k + 1} ({H / (k + 1):.0f}h): {a:.2f}" for k, a in enumerate(amp)))
    top = np.argsort(-amp)[:2] + 1
    print(f"   → os dois harmônicos dominantes aprendidos são k = {sorted(top.tolist())} (esperado [1, 3])")

    # ------------------------------------------------------------------ B
    step("B", "attach_nfh: explicar um modelo caixa-preta JÁ TREINADO, sem retreinar",
         """Treinamos um GRU + nn.Linear comum. Depois trocamos a nn.Linear por uma NFH equivalente
         (Teorema da equivalência): as previsões não mudam, mas agora cada uma vem decomposta.""")
    black_box = train(DenseGRU(1).to(DEV), Xtr, Ytr)
    with torch.no_grad():
        before = black_box(Xte.to(DEV)).cpu()
    head = attach_nfh(black_box, "out", n_harmonics=K, degree=q)
    with torch.no_grad():
        after = black_box(Xte.to(DEV)).cpu()
        nfh_out = head(black_box.gru(Xte.to(DEV))[1][-1])
    print(f"   max |previsão antes − depois| = {float((before - after).abs().max()):.2e}")
    eb = energy_balance(nfh_out)
    print(f"   explicação do modelo antigo: ε_T {pct(eb['eps_T'].mean())}, ε_S {pct(eb['eps_S'].mean())}, "
          f"ε_R {pct(eb['eps_R'].mean())} (treinado sem λ; compare com o caso A)")

    # ------------------------------------------------------------------ C
    step("C", "head.decompose: explicar a previsão de QUALQUER modelo",
         """Pela identificabilidade, (θ_T, θ_S, u) = M⁻¹ŷ para qualquer vetor ŷ. Aqui explicamos o baseline
         sazonal ingênuo (repete as últimas 24 h). Parte de um ciclo completo se parece com uma rampa dentro
         da janela e é atribuída à tendência; o restante é sazonal.""")
    naive = Xte[:, -24:, 0]
    dn = model.head.decompose(naive.to(DEV))
    en = energy_balance(dn)
    print(f"   sazonal ingênuo: ε_T {pct(en['eps_T'].mean())}, ε_S {pct(en['eps_S'].mean())}, ε_R {pct(en['eps_R'].mean())}")

    # ------------------------------------------------------------------ D
    step("D", "Desnormalizar preservando a decomposição",
         """Os dados foram padronizados com média e desvio do treino. denormalize(μ, σ) leva tudo à escala
         original: o nível μ entra só na tendência; S e R só mudam de escala.""")
    mu, sd = split.mean[0], split.std[0]
    orig = out[0].denormalize(torch.tensor(mu), torch.tensor(sd))
    print(f"   ŷ original = T + S + R? max erro {float((orig.forecast - orig.trend - orig.seasonal - orig.residual).abs().max()):.1e}")
    print(f"   nível na escala original: {float(orig.trend.mean()):.3f} (média de teste real: "
          f"{float(split.to_original(Yte[0].numpy()).mean()):.3f})")

    if args.graficos:
        from nfh.plots import plot_decomposition, plot_harmonics
        d = out_dir(args, "03_acoplamento")

        class _Wrap(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m, self.head = m, m.head

            def forward(self, x):
                return self.m(x)

        exp = explain(_Wrap(model), Xte[:1].to(DEV))
        plot_decomposition(exp, history=Xte[0, :, 0].numpy(), truth=Yte[0].numpy(),
                           title="GRU + NFH em dados sintéticos", path=d / "decomposicao.png")
        plot_harmonics(exp, unit="h", path=d / "harmonicos.png")
        print(f"\n📈 Gráficos salvos em {d}")
    print("\nPróximo: tutorial/04_experimento_bike_sharing.py")


if __name__ == "__main__":
    main()
