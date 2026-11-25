"""
Tutorial 04 — Experimento completo no UCI Bike Sharing com o procedimento da dissertação (versão 3).

Pergunta: prever os aluguéis de bicicleta das próximas 24 horas, hora a hora, a partir da última semana
(168 h) de aluguéis, clima e calendário, e explicar cada previsão em nível, tendência e ciclos.

Por que este conjunto? A demanda de bicicletas tem ciclos fortes (autocorrelação de log(1 + aluguéis) de
0,85 em 24 h e 0,88 em 168 h). Na dissertação v3 ele ficou como conjunto de reserva; aqui serve de
aplicação nova do mesmo procedimento.

Procedimento (Seção 4.3.7), para cada backbone (LSTM e Transformer) e semente:
  1. treina a rede com camada DENSA pelo MSE;
  2. CONVERTE a camada densa exatamente em NFH (mesma previsão; época 0 = decomposição pura);
  3. AJUSTE ESTRUTURAL: 15 épocas com J = MSE + λ‖R‖²/H (λ = 1000, taxa 10⁻⁴);
  4. SELEÇÃO SEGURA: menor J de validação entre as épocas com RMSE de validação ≤ o da densa.
Repete 2–4 com K escolhido no treino (cobertura ≥ 98%) e com K = 2 (só os ciclos de 24 h e 12 h),
o caso em que a decomposição pura deixa mais forma sem explicação.

Protocolo sem vazamento: divisão cronológica 70/15/15, padronização só com o treino, K escolhido só com
os alvos de treino, seleção só na validação, comparação final no TESTE contra a mesma rede densa e
contra baselines ingênuos.

    python tutorial/04_experimento_bike_sharing.py [--graficos] [--rapido] [--sementes 0 1 2]

Saídas em results/04_bike_sharing/: runs.csv, resumo.csv, resumo.json, trajetorias.json, modelos/*.pt e figuras.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from _common import header, out_dir, parser, pct, step

from nfh import AdjustConfig, Forecaster, NeuralFourierHead, energy_balance, harmonic_shares, structural_adjustment
from nfh.bases import build_bases
from nfh.data import BIKE_FEATURES, load_bike_sharing, make_splits
from nfh.training import TrainConfig, fit, metrics, naive_forecasts, predict

L, H, Q = 168, 24, 1
CONTINUOUS = ["cnt", "temp", "atemp", "hum", "windspeed"]       # normalizadas por instância (RevIN)
STANDARDIZE = CONTINUOUS + ["weathersit"]                       # padronizadas com estatísticas do treino
BACKBONES = ["lstm", "transformer"]
NAMES = {"lstm": "LSTM", "transformer": "Transformer"}
LAMBDA, ADJ_EPOCHS, ADJ_LR = 1000.0, 15, 1e-4


def build_dense(backbone: str, K: int) -> Forecaster:
    revin = [BIKE_FEATURES.index(c) for c in CONTINUOUS]
    return Forecaster(backbone, n_features=len(BIKE_FEATURES), horizon=H, target_idx=0, head="dense",
                      n_harmonics=K, degree=Q, normalize_columns=True, revin=revin)


def with_K(dense: Forecaster, K: int) -> Forecaster:
    """A mesma rede densa, com K harmônicos na conversão (a densa não depende de K)."""
    dense._init_args["n_harmonics"] = K
    return dense


def evaluate(model, X, Y, split, K: int):
    """Acurácia (escala padronizada e em bicicletas/hora) e estrutura no teste.

    A estrutura é sempre lida na base da NFH com o mesmo K: para a densa, é a decomposição pura da sua
    previsão (identificabilidade); para a NFH, coincide com as componentes do modelo."""
    out = predict(model, X)
    pred = out["forecast"].numpy()
    m = metrics(pred, Y.numpy())
    bikes_pred = np.expm1(split.to_original(pred)).clip(min=0)
    bikes_true = np.expm1(split.to_original(Y.numpy()))
    m["MAE_bicicletas"] = float(np.abs(bikes_pred - bikes_true).mean())
    m["RMSE_bicicletas"] = float(np.sqrt(((bikes_pred - bikes_true) ** 2).mean()))
    reader = NeuralFourierHead(8, H, K, Q)
    dec = reader.decompose(out["forecast"])
    e = energy_balance(dec)
    ek = harmonic_shares(dec, reader.Ft)
    m.update({"cobertura": float(np.median(e["coverage"])), "cobertura_min": float(e["coverage"].min()),
              "eps_T": float(np.median(e["eps_T"])), "eps_S": float(np.median(e["eps_S"])),
              "eps_R": float(np.median(e["eps_R"])), "k1_dominante": float((ek.argmax(1) == 0).mean())})
    return m


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--rapido", action="store_true", help="execução curta para testar o pipeline (3 épocas, 1 semente)")
    p.add_argument("--sementes", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epocas", type=int, default=40, help="máximo de épocas da densa (parada antecipada)")
    args = p.parse_args()
    seeds = [0] if args.rapido else args.sementes
    max_epochs = 3 if args.rapido else args.epocas
    adj_epochs = 2 if args.rapido else ADJ_EPOCHS
    outd = out_dir(args, "04_bike_sharing")
    (outd / "modelos").mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base_cfg = dict(lr=1e-3, weight_decay=1e-5, batch_size=64, max_epochs=max_epochs, patience=6, grad_clip=1.0,
                    device=dev)
    header("TUTORIAL 04 · Bike Sharing com o procedimento da dissertação v3",
           f"Janela L = {L} h → horizonte H = {H} h. Densa → conversão exata → ajuste estrutural ({adj_epochs} épocas, "
           f"λ = {LAMBDA:g}) com seleção segura. Dispositivo {dev}; sementes {seeds}.")
    t_start = time.time()

    # ------------------------------------------------------------------ 1
    step(1, "Dados e análise exploratória",
         """Contagem horária de aluguéis (Capital Bikeshare, Washington DC, 2011–2012) com clima e calendário.
         As 165 horas ausentes do arquivo original são inseridas com contagem 0. O alvo é log(1 + aluguéis).""")
    df = load_bike_sharing()
    s = df["cnt_raw"]
    print(f"   {len(df):,} horas de {df.index.min():%Y-%m-%d} a {df.index.max():%Y-%m-%d}; média {s.mean():.0f} "
          f"aluguéis/h, máximo {s.max():.0f}")
    print("   autocorrelação de log(1+aluguéis): " +
          ", ".join(f"{lag} h = {df['cnt'].autocorr(lag):.2f}" for lag in (1, 12, 24, 168)))

    # ------------------------------------------------------------------ 2
    step(2, "Divisão cronológica e padronização sem vazamento")
    split = make_splits(df, "cnt", BIKE_FEATURES, L, H, standardize=STANDARDIZE)
    a, b = split.bounds
    for name, ds, (i0, i1) in (("treino", split.train, (0, a)), ("validação", split.val, (a, b)),
                               ("teste", split.test, (b, len(df)))):
        print(f"   {name:9s}: {df.index[i0]:%Y-%m-%d} → {df.index[i1 - 1]:%Y-%m-%d}  ({len(ds):,} janelas)")
    Xva, Yva = split.val.tensors()
    Xte, Yte = split.test.tensors()

    # ------------------------------------------------------------------ 3
    step(3, "Escolha de K usando apenas os alvos de treino",
         "Menor K cuja base estrutural [Pᵀ F̃ᵀ] representa ≥ 98% da forma dos perfis reais de 24 h do treino.")
    _, Ytr = split.train.tensors()
    Y = Ytr.double()
    Yc = Y - Y.mean(1, keepdim=True)
    E = (Yc ** 2).sum(1)
    K_main = 6
    for k in range(1, 12):
        bb = build_bases(H, k, Q, dtype=torch.float64)
        B = torch.cat([bb.P, bb.Ft]).T
        proj = (B @ torch.linalg.lstsq(B, Y.T).solution).T
        cov = float((((proj - proj.mean(1, keepdim=True)) ** 2).sum(1) / E).mean())
        print(f"   K = {k:2d} (períodos até {H / k:4.1f} h): cobertura {pct(cov)}")
        if cov >= 0.98:
            K_main = k
            break
    K_list = [K_main, 2] if K_main != 2 else [2]
    print(f"   → K = {K_main} (principal) e K = 2 (base compacta: ciclos de 24 h e 12 h)")

    # ------------------------------------------------------------------ 4
    step(4, "Baselines ingênuos no teste")
    rows = []
    preds_naive = naive_forecasts(Xte.numpy(), H, 0, season=24)
    preds_naive["sazonal_168"] = naive_forecasts(Xte.numpy(), H, 0, season=168)["sazonal_168"]
    for nm, pred in preds_naive.items():
        rows.append({"modelo": f"Ingênuo ({nm})", "backbone": "", "K": None, "semente": -1,
                     **_naive_metrics(pred, Yte.numpy(), split)})
    for r in rows:
        print(f"   {r['modelo']:24s} RMSE {r['RMSE']:.3f} · MAE {r['MAE_bicicletas']:6.1f} bicicletas/h")

    # ------------------------------------------------------------------ 5
    step(5, "Densa → conversão exata → ajuste estrutural com seleção segura",
         """Para cada backbone e semente: treina a densa; converte-a em NFH com cada K (a previsão não muda);
         faz o ajuste estrutural e escolhe a época pela seleção segura. Registram-se também a decomposição
         pura (época 0) e a estrutura completa (menor J sem a restrição de RMSE).""")
    trajectories = {}
    for bb_name in BACKBONES:
        for seed in seeds:
            t0 = time.time()
            r = fit(build_dense(bb_name, K_main), split.train, split.val, TrainConfig(**base_cfg, lam=0.0, seed=seed))
            dense = r.model
            for K in K_list:
                m_d = evaluate(dense, Xte, Yte, split, K)
                rows.append({"modelo": f"{NAMES[bb_name]} + densa", "backbone": bb_name, "K": K, "semente": seed,
                             "variante": "densa (= decomposição pura)", "melhor_epoca": r.best_epoch, **m_d})
                nfh0 = with_K(dense, K).to_nfh()
                with torch.no_grad():
                    xb = Xva[:512].to(dev)
                    conv_err = float((nfh0(xb).forecast - dense(xb).forecast).abs().max())
                adj = structural_adjustment(nfh0, split.train, split.val,
                                            AdjustConfig(epochs=adj_epochs, lr=ADJ_LR, lam=LAMBDA, batch_size=64,
                                                         seed=seed, device=dev))
                m_n = evaluate(adj.model, Xte, Yte, split, K)
                rows.append({"modelo": f"{NAMES[bb_name]} + NFH", "backbone": bb_name, "K": K, "semente": seed,
                             "variante": "seleção segura", "epoca_selecionada": adj.selected_epoch,
                             "erro_conversao": conv_err, "rmse_val_densa": adj.rmse_val_dense,
                             "rmse_val_nfh": adj.selected["rmse"], **m_n})
                full = with_K(dense, K).to_nfh()
                full.load_state_dict(adj.full_structure_state)
                m_f = evaluate(full, Xte, Yte, split, K)
                rows.append({"modelo": f"{NAMES[bb_name]} + NFH", "backbone": bb_name, "K": K, "semente": seed,
                             "variante": "estrutura completa", "epoca_selecionada": adj.full_structure_epoch, **m_f})
                trajectories[f"{bb_name}|K{K}|s{seed}"] = adj.trajectory
                print(f"   {NAMES[bb_name]:11s} s{seed} K={K:2d}: densa RMSE {m_d['RMSE']:.4f} κ {pct(m_d['cobertura'])} → "
                      f"NFH época {adj.selected_epoch:2d} RMSE {m_n['RMSE']:.4f} κ {pct(m_n['cobertura'])} "
                      f"(val {adj.selected['rmse']:.4f} ≤ {adj.rmse_val_dense:.4f}; conversão {conv_err:.1e})")
                if seed == seeds[0]:
                    cfg = {"backbone": bb_name, "K": K, "q": Q, "L": L, "H": H, "lambda": LAMBDA,
                           "features": BIKE_FEATURES, "continuous": CONTINUOUS, "standardize": STANDARDIZE,
                           "seed": seed}
                    split_info = {"mean": split.mean.tolist(), "std": split.std.tolist(), "bounds": list(split.bounds)}
                    suffix = "" if K == K_main else f"_K{K}"
                    torch.save({"state_dict": adj.model.state_dict(),
                                "config": {**cfg, "head": "nfh", "procedure": "conversao_ajuste",
                                           "selected_epoch": adj.selected_epoch},
                                "split": split_info, "trajectory": adj.trajectory},
                               outd / "modelos" / f"{bb_name}_nfh{suffix}_s{seed}.pt")
                    if K == K_main:
                        torch.save({"state_dict": dense.state_dict(), "config": {**cfg, "head": "dense"},
                                    "split": split_info, "history": r.history},
                                   outd / "modelos" / f"{bb_name}_dense_s{seed}.pt")
            print(f"      ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------ 6
    step(6, "Resultados")
    runs = pd.DataFrame(rows)
    runs.to_csv(outd / "runs.csv", index=False)
    (outd / "trajetorias.json").write_text(json.dumps(trajectories, indent=1))
    keys = ["modelo", "variante", "K"]
    cols = ["RMSE", "MAE_bicicletas", "cobertura", "cobertura_min", "eps_R"]
    model_runs = runs[runs.semente >= 0]
    summ = model_runs.groupby(keys, dropna=False)[cols].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(outd / "resumo.csv", index=False)
    print(f"   {'modelo':20s} {'variante':28s} {'K':>3s} {'RMSE':>15s} {'MAE bic/h':>10s} {'κ mediana':>10s} {'κ mínima':>9s}")
    for _, r in summ.iterrows():
        print(f"   {r['modelo']:20s} {r['variante']:28s} {int(r['K']):3d} {r['RMSE_mean']:.4f} ± {r['RMSE_std']:.4f} "
              f"{r['MAE_bicicletas_mean']:9.1f} {pct(r['cobertura_mean']):>10s} {pct(r['cobertura_min_mean']):>9s}")

    best_naive = min(r["RMSE"] for r in rows if r["semente"] == -1)
    report = {"config": {"L": L, "H": H, "K": K_list, "q": Q, "lambda": LAMBDA, "adjust_epochs": adj_epochs,
                         "adjust_lr": ADJ_LR, "seeds": seeds, "max_epochs_dense": max_epochs},
              "best_naive_rmse": best_naive, "comparacao": {}}
    for bb_name in BACKBONES:
        for K in K_list:
            d = model_runs[(model_runs.backbone == bb_name) & (model_runs.K == K)]
            dn = d[d.variante.str.startswith("densa")].sort_values("semente")
            nf = d[d.variante == "seleção segura"].sort_values("semente")
            fu = d[d.variante == "estrutura completa"].sort_values("semente")
            diff = 100 * (nf.RMSE.to_numpy() - dn.RMSE.to_numpy()) / dn.RMSE.to_numpy()
            c = {"rmse_densa": float(dn.RMSE.mean()), "rmse_nfh": float(nf.RMSE.mean()), "rmse_completa": float(fu.RMSE.mean()),
                 "dif_rel_nfh_densa_pct": float(diff.mean()), "nfh_melhor": int((diff < -1e-4).sum()),
                 "nfh_igual": int((np.abs(diff) <= 1e-4).sum()), "nfh_pior": int((diff > 1e-4).sum()),
                 "cobertura_densa": float(dn.cobertura.mean()), "cobertura_nfh": float(nf.cobertura.mean()),
                 "cobertura_min_densa": float(dn.cobertura_min.mean()), "cobertura_min_nfh": float(nf.cobertura_min.mean()),
                 "epocas_selecionadas": nf.epoca_selecionada.astype(int).tolist(),
                 "ganho_vs_ingenuo_pct": float(100 * (1 - nf.RMSE.mean() / best_naive))}
            report["comparacao"][f"{bb_name}_K{K}"] = c
            print(f"\n   {NAMES[bb_name]} · K = {K}: densa {c['rmse_densa']:.4f} → NFH {c['rmse_nfh']:.4f} "
                  f"({c['dif_rel_nfh_densa_pct']:+.2f}%; melhor/igual/pior {c['nfh_melhor']}/{c['nfh_igual']}/{c['nfh_pior']}; "
                  f"épocas {c['epocas_selecionadas']}) · estrutura completa {c['rmse_completa']:.4f}")
            print(f"      cobertura mediana {pct(c['cobertura_densa'])} → {pct(c['cobertura_nfh'])}; "
                  f"mínima {pct(c['cobertura_min_densa'])} → {pct(c['cobertura_min_nfh'])}; "
                  f"RMSE {abs(c['ganho_vs_ingenuo_pct']):.1f}% {'menor' if c['ganho_vs_ingenuo_pct'] > 0 else 'maior'} que o melhor ingênuo")
    report["tempo_total_min"] = round((time.time() - t_start) / 60, 1)
    (outd / "resumo.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n   Resultados salvos em {outd} ({report['tempo_total_min']} min)")

    if args.graficos:
        from nfh.plots import plot_comparison
        sel = summ[summ.variante != "estrutura completa"].copy()
        sel["model"] = sel.apply(lambda r: f"{r['modelo']} (K={int(r['K'])})" if "NFH" in r["modelo"] else f"{r['modelo']}", axis=1)
        sel = sel.drop_duplicates("model")
        nai = runs[runs.semente < 0][["modelo", "RMSE"]].rename(columns={"modelo": "model", "RMSE": "mean"})
        nai["std"] = 0.0
        tab = pd.concat([nai, sel.rename(columns={"RMSE_mean": "mean", "RMSE_std": "std"})[["model", "mean", "std"]]])
        plot_comparison(tab.fillna(0.0), path=outd / "comparacao_rmse.png",
                        title="Bike Sharing, teste: RMSE de log(1+aluguéis) padronizado (média ± desvio)")
        _plot_trajectories(trajectories, outd / "ajuste_estrutural.png")
        print(f"   📈 Gráficos salvos em {outd}")
    print("\nPróximo: tutorial/05_explicando_uma_previsao.py")


def _naive_metrics(pred, Y, split):
    m = metrics(pred, Y)
    bp = np.expm1(split.to_original(pred)).clip(min=0)
    bt = np.expm1(split.to_original(Y))
    m["MAE_bicicletas"] = float(np.abs(bp - bt).mean())
    m["RMSE_bicicletas"] = float(np.sqrt(((bp - bt) ** 2).mean()))
    return m


def _plot_trajectories(trajs, path):
    import matplotlib.pyplot as plt
    from nfh.plots import COLORS
    keys = [k for k in trajs if k.endswith("|s0")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for k in keys:
        t = trajs[k]
        ep = [r["epoch"] for r in t]
        ls = "-" if "transformer" not in k else "--"
        axes[0].plot(ep, [r["rmse"] for r in t], ls, marker="o", ms=3, label=k.replace("|s0", ""))
        axes[1].plot(ep, [100 * r["kappa"] for r in t], ls, marker="o", ms=3, label=k.replace("|s0", ""))
        axes[0].axhline(t[0]["rmse"], color="#bbb", lw=0.6)
    axes[0].set_title("RMSE de validação no ajuste (linha cinza: densa, limite da seleção)")
    axes[1].set_title("Cobertura estrutural mediana na validação (%)")
    for a in axes:
        a.set_xlabel("época do ajuste (0 = densa convertida)")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
