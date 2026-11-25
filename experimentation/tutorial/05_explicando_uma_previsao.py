"""
Tutorial 05 — Explicando uma previsão do Bike Sharing, como um analista faria.

Carrega o modelo treinado no tutorial 04 (padrão: LSTM + NFH, semente 0) e
explica a previsão das 24 horas seguintes para um dia útil do conjunto de teste:

  1. previsão × realidade × baseline ingênuo;
  2. nível, tendência e balanço de energia da forma;
  3. ciclos: harmônicos, participações e_k e o horário do pico previsto;
  4. de onde vem cada componente: Integrated Gradients por variável e por defasagem;
  5. estabilidade: robustez a ruído e variação da leitura entre muitas janelas;
  6. a mesma previsão de uma camada densa, lida na base da NFH (identificabilidade);
  7. explicação em tópicos.

    python tutorial/05_explicando_uma_previsao.py [--graficos] [--backbone lstm|transformer] [--janela N]
"""
from __future__ import annotations

import json

import numpy as np
import torch

from _common import RESULTS, check, header, out_dir, parser, pct, step

from nfh import Forecaster, energy_balance, explain, integrated_gradients, narrative, robustness
from nfh.data import load_bike_sharing, make_splits
from nfh.training import naive_forecasts

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(backbone: str, head: str):
    path = RESULTS / "04_bike_sharing" / "modelos" / f"{backbone}_{head}_s0.pt"
    if not path.exists():
        raise SystemExit(f"Modelo {path} não encontrado: rode antes tutorial/04_experimento_bike_sharing.py")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["config"]
    revin = [c["features"].index(v) for v in c["continuous"]]
    model = Forecaster(c["backbone"], len(c["features"]), c["H"], target_idx=0, head=c["head"],
                       n_harmonics=c["K"], degree=c["q"], revin=revin)
    model.load_state_dict(ck["state_dict"])
    return model.to(DEV).eval(), ck


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--backbone", default="lstm", choices=["lstm", "transformer"])
    p.add_argument("--janela", type=int, default=None, help="índice da janela de teste (padrão: 1º dia útil às 0 h)")
    p.add_argument("--n-janelas", type=int, default=50, help="janelas para a análise de estabilidade")
    args = p.parse_args()
    outd = out_dir(args, "05_explicacao")
    model, ck = load(args.backbone, "nfh")
    c = ck["config"]
    periods = ", ".join(f"{c['H'] / k:.1f}" for k in range(1, c["K"] + 1))
    header(f"TUTORIAL 05 · Explicando uma previsão ({args.backbone.upper()} + NFH, λ = {c['lambda']:g})",
           f"L = {c['L']} h, H = {c['H']} h, K = {c['K']} harmônicos (períodos {periods} h), q = {c['q']}.")

    # ------------------------------------------------------------------ dados
    df = load_bike_sharing()
    split = make_splits(df, "cnt", c["features"], c["L"], c["H"], standardize=c["standardize"])
    check("padronização igual à do treino (integridade)", float(np.abs(split.mean - np.array(ck["split"]["mean"])).max()), 1e-9)
    Xte, Yte = split.test.tensors()
    t0 = split.bounds[1]                                 # início do trecho de teste
    starts = df.index[t0 + c["L"]: t0 + c["L"] + len(Xte)]  # instante do 1º passo previsto de cada janela
    if args.janela is None:
        ok = [i for i, ts in enumerate(starts) if ts.hour == 0 and df["workingday"].iloc[t0 + c["L"] + i] == 1]
        i = ok[0]
    else:
        i = args.janela
    when = starts[i]
    x, y = Xte[i:i + 1].to(DEV), Yte[i]
    to_bikes = lambda v: np.expm1(split.to_original(np.asarray(v))).clip(min=0)  # noqa: E731

    # ------------------------------------------------------------------ 1
    step(1, "Previsão × realidade",
         f"Janela de teste nº {i}: entrada {starts[i] - np.timedelta64(c['L'], 'h'):%Y-%m-%d %H h} → "
         f"previsão de {when:%A, %Y-%m-%d} (0 h a 23 h).")
    exp = explain(model, x)
    pred_b, true_b = to_bikes(exp.forecast[0]), to_bikes(y.numpy())
    naive_b = to_bikes(naive_forecasts(Xte[i:i + 1].numpy(), c["H"], 0, season=24)["sazonal_24"][0])
    mae, mae_n = np.abs(pred_b - true_b).mean(), np.abs(naive_b - true_b).mean()
    print(f"   total previsto no dia: {pred_b.sum():,.0f} aluguéis (real {true_b.sum():,.0f})")
    print(f"   MAE da janela: NFH {mae:.1f} bicicletas/h · repetir ontem {mae_n:.1f} bicicletas/h")
    print(f"   pico previsto às {int(pred_b.argmax())} h ({pred_b.max():.0f}/h); pico real às {int(true_b.argmax())} h "
          f"({true_b.max():.0f}/h)")

    # ------------------------------------------------------------------ 2
    step(2, "Nível, tendência e balanço de energia (escala log padronizada)",
         """O nível é a média prevista no horizonte e pertence todo à tendência. A forma (previsão menos o nível)
         se divide, sem termos cruzados, em tendência, sazonalidade e resíduo.""")
    s = exp.summary(0)
    print(f"   nível {s['level']:+.3f} (≈ {to_bikes(s['level']):.0f} bicicletas/h na média geométrica)")
    print(f"   tendência de {exp.trend[0, 0]:+.3f} a {exp.trend[0, -1]:+.3f} ao longo do dia")
    print(f"   ε_T = {pct(s['eps_T'])} · ε_S = {pct(s['eps_S'])} · ε_R = {pct(s['eps_R'])} · cobertura κ = {pct(s['coverage'])}")
    check("soma ε_T + ε_S + ε_R − 1", s["eps_T"] + s["eps_S"] + s["eps_R"] - 1, 1e-6)

    # ------------------------------------------------------------------ 3
    step(3, "Ciclos: que harmônicos formam o dia previsto",
         "e_k é a participação aditiva do ciclo de período H/k na forma (Σ e_k = ε_S; negativo = cancelamento).")
    for k in np.argsort(-exp.shares[0]):
        print(f"   k={k + 1} período {exp.periods[k]:4.1f} h: e_k = {pct(exp.shares[0, k]):>6s}, amplitude "
              f"{exp.amplitude[0, k]:.3f}, fase {exp.phase[0, k]:+.2f} rad")
    S = exp.seasonal[0]
    print(f"   a sazonalidade S eleva a previsão ao máximo às {int(S.argmax())} h e a reduz ao mínimo às {int(S.argmin())} h")

    # ------------------------------------------------------------------ 4
    step(4, "De onde vem cada componente: Integrated Gradients",
         """Atribuímos o nível (g_T), a energia dos ciclos (g_S) e a do resíduo (g_R) às 12 variáveis de
         entrada. A referência é a janela nula no espaço normalizado: a média da própria janela para as variáveis
         contínuas (RevIN), a média de treino para a condição do tempo e zero para flags e seno/cosseno.
         Completude: as contribuições somam a variação de cada componente, a menos do erro da quadratura
         (Gauss-Legendre). Como o backbone tem ReLU, o integrando tem quinas e o erro não cai de forma
         monotônica com o número de nós; para uma única janela usamos 512 nós. É uma leitura local do
         modelo, não causal.""")
    xn, _, _ = model.normalize(x)
    attr = integrated_gradients(model.inner, xn, c["features"], steps=512)
    for comp, lab in (("T", "tendência/nível"), ("S", "sazonalidade")):
        top = sorted(attr[comp]["share"].items(), key=lambda kv: -kv[1])[:4]
        print(f"   {lab:16s}: " + ", ".join(f"{k} {pct(v)}" for k, v in top))
        check(f"completude {comp} (Δg = {attr[comp]['delta']:+.4f})", attr[comp]["completeness_error"],
              1e-2 * max(1e-3, abs(attr[comp]["delta"])))
    lags = np.array(attr["S"]["per_lag"])
    top_lags = np.argsort(-lags)[:5]
    print("   defasagens que mais sustentam os ciclos (h antes da previsão): "
          + ", ".join(str(c["L"] - int(t)) for t in top_lags))

    # ------------------------------------------------------------------ 5
    step(5, "Estabilidade da explicação",
         """Robustez: cosseno entre as participações (ε_T, e_1..e_K, ε_R) antes e depois de ruído N(0; 0,05²) na
         entrada. Depois, variação da leitura em várias janelas de teste: uma janela não deve ser generalizada.""")
    rob = robustness(model, x, sigma=0.05, n_draws=3)
    print(f"   robustez a ruído nesta janela: {float(rob[0]):.3f}")
    idx = np.linspace(0, len(Xte) - 1, args.n_janelas).astype(int)
    many = explain(model, Xte[idx].to(DEV))
    dom = np.bincount(many.shares.argmax(1), minlength=c["K"])
    print(f"   em {len(idx)} janelas: ε_S médio {pct(many.eps_S.mean())} (±{pct(many.eps_S.std())}), "
          f"cobertura média {pct(many.coverage.mean())}")
    print("   harmônico dominante: " + ", ".join(f"k={k + 1} ({exp.periods[k]:.0f} h) em {n}" for k, n in enumerate(dom) if n))
    shares = {v: [] for v in c["features"]}
    for j in idx[:: max(1, len(idx) // 20)]:
        xj, _, _ = model.normalize(Xte[j:j + 1].to(DEV))
        a = integrated_gradients(model.inner, xj, c["features"], steps=32)
        for v in c["features"]:
            shares[v].append(a["S"]["share"][v])
    print("   participação média na sazonalidade (≈20 janelas): " + ", ".join(
        f"{v} {pct(np.mean(sh))}±{pct(np.std(sh))}" for v, sh in sorted(shares.items(), key=lambda kv: -np.mean(kv[1]))[:4]))

    # ------------------------------------------------------------------ 6
    step(6, "A mesma pergunta para uma camada densa (identificabilidade)",
         """Toda previsão tem uma única decomposição na base da NFH. Lendo a previsão da rede densa nessa base,
         também obtemos uma explicação — mas ela não faz parte do modelo, e nada no treino incentivou a forma
         a ser estrutural (sem λ).""")
    try:
        dense, _ = load(args.backbone, "dense")
        with torch.no_grad():
            yd = dense(x).forecast
        dec = model.head.decompose(yd)
        ed = energy_balance(dec)
        print(f"   densa: ε_T {pct(ed['eps_T'][0])}, ε_S {pct(ed['eps_S'][0])}, ε_R {pct(ed['eps_R'][0])}; "
              f"MAE da janela {np.abs(to_bikes(yd[0].cpu().numpy()) - true_b).mean():.1f} bicicletas/h")
    except SystemExit as e:
        print(f"   {e}")

    # ------------------------------------------------------------------ 7
    step(7, "Explicação em tópicos")
    hist_std = float(Xte[i, :, 0].std())
    lines = narrative(exp, 0, unit="h", target="log(1+aluguéis) padronizado", history_std=hist_std, attribution=attr)
    lines.insert(0, f"Previsão para {when:%d/%m/%Y}: {pred_b.sum():,.0f} aluguéis no dia, pico às "
                    f"{int(pred_b.argmax())} h com ≈{pred_b.max():.0f} bicicletas/h.")
    for ln in lines:
        print(f"   • {ln}")
    (outd / "explicacao.json").write_text(json.dumps({
        "janela": int(i), "inicio_previsao": str(when), "backbone": args.backbone, "resumo": s,
        "atribuicao": {k: {kk: vv for kk, vv in v.items() if kk != "per_lag"} for k, v in attr.items()},
        "robustez": float(rob[0]), "topicos": lines}, indent=2, ensure_ascii=False, default=float))

    if args.graficos:
        from nfh.plots import plot_attribution, plot_decomposition, plot_energy, plot_harmonics
        hist = Xte[i, :, 0].numpy()[-72:]
        plot_decomposition(exp, history=hist, truth=y.numpy(), ylabel="log(1+aluguéis) padronizado",
                           title=f"{args.backbone.upper()} + NFH · previsão para {when:%d/%m/%Y}",
                           path=outd / "decomposicao.png")
        plot_harmonics(exp, unit="h", path=outd / "harmonicos.png")
        plot_attribution(attr, path=outd / "atribuicao_variaveis.png")
        plot_energy(many, idx=list(range(min(20, len(idx)))),
                    labels=[f"{starts[j]:%d/%m %Hh}" for j in idx[:20]], path=outd / "energia_janelas.png")
        print(f"\n📈 Gráficos salvos em {outd}")


if __name__ == "__main__":
    main()
