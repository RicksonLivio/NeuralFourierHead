"""
E6-v3: exemplo detalhado de previsão multivariada (itens M58 e M59 do orientador), Jena Climate (M):
alvo T (temperatura do ar) e covariáveis p, rh, Wx, Wy; Conv1D+LSTM+NFH (K = 10, seleção segura,
semente 0). Janela fixada ANTES de ver resultados: a central do teste, i0 = len(teste) // 2.

  * decomposição da previsão (nível, eps_T, eps_S, eps_R, harmônicos dominantes) e figura no estilo da
    Figura 3 (fig_exemplo_jena.pdf), em graus Celsius;
  * atribuição de g_T (nível normalizado) e g_S (energia sazonal normalizada) às 5 variáveis por
    Integrated Gradients sobre a janela normalizada (referência nula = janela igual à própria média), com
    erro de completude; comparação, na mesma janela e com os mesmos funcionais, com KernelSHAP, LIME e
    Feature Ablation (participação |soma nas defasagens| / soma);
  * agregado em 50 janelas de teste igualmente espaçadas (IG): média e desvio das participações e fração
    de janelas em que cada variável é a de maior participação.
Saída: results/v3/exemplo_jena.json, versao3_correcao2_aplicada/figuras/graficos/fig_exemplo_jena.pdf
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from v2_common import stack_ds  # noqa: E402
from v3_common import RES3, ROOT, carregar, construir, segmentos  # noqa: E402

warnings.filterwarnings("ignore")
CONJ, BB = "Jena-M", "Conv1D+LSTM"
STEPS, N_AGG = 128, 50
FIG = ROOT / "versao3_correcao2_aplicada" / "figuras" / "graficos"
PNG = Path.home() / ".cache" / "v2work" / "figs_v3"
DEEP, TEALD, BLUEG, WINE, PLUM, LILAC, SAND = "#270a33", "#3d7b80", "#8daec3", "#451a3e", "#4d3e6b", "#81749c", "#e3cebd"
plt.rcParams.update({"font.size": 10.5, "text.color": DEEP, "axes.edgecolor": DEEP, "axes.labelcolor": DEEP,
                     "xtick.color": DEEP, "ytick.color": DEEP, "mathtext.fontset": "cm"})


def normalizar(x, eps=1e-5):
    mu = x.mean(1, keepdim=True)
    sd = x.std(1, keepdim=True) + eps
    return (x - mu) / sd


def funcionais(out):
    return {"T": out["trend"].mean(1), "S": (out["seasonal"] ** 2).sum(1)}


def ig(inner, xn, chave, steps=STEPS):
    base = torch.zeros_like(xn)
    alf = (torch.arange(steps, dtype=torch.float32, device=xn.device) + 0.5) / steps
    xs = (base + alf.view(-1, 1, 1) * (xn - base)).requires_grad_(True)
    g = funcionais(inner(xs))[chave].sum()
    grad = torch.autograd.grad(g, xs)[0].mean(0, keepdim=True)
    a = ((xn - base) * grad).squeeze(0).detach().cpu().double().numpy()
    with torch.no_grad():
        delta = float(funcionais(inner(xn))[chave] - funcionais(inner(base))[chave])
    return a, delta


def participacao(por_var):
    a = np.abs(por_var)
    return 100 * a / (a.sum() + 1e-30)


def main():
    import shap
    from captum.attr import FeatureAblation
    from lime.lime_tabular import LimeTabularExplainer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = False
    ds = carregar(CONJ)
    nomes = ds["nomes"]
    tr, va, te = segmentos(ds)
    d = len(nomes)
    modelo = construir(BB, "nfh", d)
    modelo.load_state_dict(torch.load(RES3 / "modelos" / f"{CONJ}__{BB}+NFH__s0.pt", map_location="cpu"))
    modelo = modelo.to(dev).eval()
    for p_ in modelo.parameters():
        p_.requires_grad_(False)
    inner, head = modelo.inner, modelo.inner.head
    X, Y = stack_ds(te)
    Xtr, _ = stack_ds(tr)
    L, H = X.shape[1], Y.shape[1]
    i0 = len(X) // 2
    x = X[i0:i0 + 1].to(dev)
    with torch.no_grad():
        o = {k: v.detach().cpu().double().numpy()[0] for k, v in modelo(x).items() if torch.is_tensor(v) and v.dim() == 2}
    yh, T, S, R = o["forecast"], o["trend"], o["seasonal"], o["residual"]
    yc, Tc = yh - yh.mean(), T - T.mean()
    E = (yc ** 2).sum()
    th = o["theta_S"]
    Ft = head.Ft.cpu().double().numpy()
    # theta_S já sai desnormalizado pelo RevIN (multiplicado por sigma_y), logo S = sum_k S_k na mesma escala
    Sk = [th[2 * k] * Ft[2 * k] + th[2 * k + 1] * Ft[2 * k + 1] for k in range(len(th) // 2)]
    ek = np.array([(sk * S).sum() / E for sk in Sk])
    amp = np.hypot(th[0::2], th[1::2])
    top = np.argsort(-ek)[:3]
    mu_a, sd_a = ds["mu_alvo"], ds["sd_alvo"]
    # Atribuição por IG (convenção normalizada)
    xn = normalizar(x)
    comp = {}
    for chave in ("T", "S"):
        a, delta = ig(inner, xn, chave)
        comp[chave] = {"delta_g": delta, "erro_completude": float(abs(a.sum() - delta)),
                       "por_variavel": dict(zip(nomes, a.sum(0).tolist())),
                       "participacao_pct": dict(zip(nomes, participacao(a.sum(0)).tolist())),
                       "ultimas24h_pct_da_variavel_dominante": None}
        vdom = int(np.argmax(np.abs(a.sum(0))))
        comp[chave]["variavel_dominante"] = nomes[vdom]
        comp[chave]["ultimas24h_pct_da_variavel_dominante"] = float(
            100 * np.abs(a[-24:, vdom]).sum() / (np.abs(a[:, vdom]).sum() + 1e-30))
    # Outros métodos, mesmos funcionais e mesma convenção (np.random.seed: o KernelSHAP amostra coalizões
    # com o gerador global do NumPy)
    np.random.seed(0)
    rng = np.random.default_rng(0)
    bg = normalizar(Xtr[rng.choice(len(Xtr), size=200, replace=False)]).reshape(200, -1).numpy()
    bg_sum = shap.kmeans(bg, 10)
    lime_ex = LimeTabularExplainer(bg, mode="regression", discretize_continuous=False, verbose=False, random_state=0,
                                   sample_around_instance=True)
    n = L * d
    xf = xn.reshape(-1).cpu().numpy().astype(np.float64)
    outros = {}
    for chave in ("T", "S"):
        class Fun(torch.nn.Module):
            def forward(self, flat, chave=chave):
                return funcionais(inner(flat.reshape(flat.shape[0], L, d)))[chave].unsqueeze(1)
        fun = Fun().to(dev)

        def g_np(A):
            with torch.no_grad():
                return np.concatenate([fun(torch.tensor(A[i:i + 4096], dtype=torch.float32, device=dev)).cpu().numpy()[:, 0]
                                       for i in range(0, len(A), 4096)])
        sv = np.asarray(shap.KernelExplainer(g_np, bg_sum).shap_values(xf.reshape(1, -1), nsamples=3000, silent=True)).reshape(L, d)
        e = lime_ex.explain_instance(xf, g_np, num_features=n, num_samples=1000)
        lv = np.zeros(n)
        for j, w in e.as_map()[1]:
            lv[j] = w
        fa = FeatureAblation(fun).attribute(torch.tensor(xf.reshape(1, -1), dtype=torch.float32, device=dev), baselines=0.0,
                                            perturbations_per_eval=128).cpu().numpy().reshape(L, d)
        outros[chave] = {"KernelSHAP": dict(zip(nomes, participacao(sv.sum(0)).tolist())),
                         "LIME": dict(zip(nomes, participacao(lv.reshape(L, d).sum(0)).tolist())),
                         "Feature Ablation": dict(zip(nomes, participacao(fa.sum(0)).tolist()))}
    # Agregado em 50 janelas
    idx = np.linspace(0, len(X) - 1, N_AGG).astype(int)
    agg = {"T": [], "S": []}
    for i in idx:
        xni = normalizar(X[i:i + 1].to(dev))
        for chave in ("T", "S"):
            a, _ = ig(inner, xni, chave, steps=64)
            agg[chave].append(participacao(a.sum(0)))
    agg_res = {c: {"media": dict(zip(nomes, np.mean(v, 0).tolist())), "desvio": dict(zip(nomes, np.std(v, 0).tolist())),
                   "fracao_dominante": dict(zip(nomes, np.bincount(np.argmax(np.array(v), 1), minlength=d) / len(v)))}
               for c, v in agg.items()}
    for c in agg_res:
        agg_res[c]["fracao_dominante"] = {k: float(v) for k, v in agg_res[c]["fracao_dominante"].items()}
    datas = ds["datas"]
    a, b = ds["cortes"]
    i_global = b + te.idx[i0]          # índice da primeira hora da janela no conjunto completo
    res = {"conjunto": CONJ, "backbone": BB, "variaveis": nomes, "janela_teste": int(i0), "n_teste": int(len(X)),
           "inicio_janela": str(datas[i_global]), "inicio_horizonte": str(datas[i_global + L]),
           "rmse_janela": float(np.sqrt(np.mean((yh - Y[i0].numpy()) ** 2))),
           "nivel_previsto_C": float(yh.mean() * sd_a + mu_a), "rms_forma_C": float(np.sqrt((yc ** 2).mean()) * sd_a),
           "eps_T": float((Tc ** 2).sum() / E), "eps_S": float((S ** 2).sum() / E), "eps_R": float((R ** 2).sum() / E),
           "harmonicos_top": [{"k": int(k + 1), "periodo_h": H / (k + 1), "e_k": float(ek[k]),
                               "amplitude_C": float(amp[k] * sd_a)} for k in top],
           "atribuicao_ig": comp, "outros_metodos": outros, "agregado_50": agg_res}
    (RES3 / "exemplo_jena.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    # Figura no estilo da Figura 3: painéis com a mesma escala de tempo (horas relativas ao início do horizonte)
    FIG.mkdir(parents=True, exist_ok=True)
    PNG.mkdir(parents=True, exist_ok=True)
    hist = X[i0, :, -1].numpy() * sd_a + mu_a
    real = Y[i0].numpy() * sd_a + mu_a
    th_h = np.arange(-L, 0)
    tf_h = np.arange(H)
    k_dom = int(top[0])
    import matplotlib.ticker as mticker
    virgula = mticker.FuncFormatter(lambda v, pos: f"{v:g}".replace(".", ",").replace("-", "\u2212"))
    res["figura"] = {"hist": hist.tolist(), "real": real.tolist(), "previsao": (yh * sd_a + mu_a).tolist(),
                     "T": (T * sd_a + mu_a).tolist(), "S": (S * sd_a).tolist(), "S_dom": (Sk[k_dom] * sd_a).tolist(),
                     "R": (R * sd_a).tolist(), "k_dom": k_dom + 1}
    (RES3 / "exemplo_jena.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    fig, ax = plt.subplots(4, 1, figsize=(10.5, 8.4), sharex=True, gridspec_kw={"height_ratios": [1.6, 1, 1, 1]})
    ax[0].plot(th_h, hist, color=BLUEG, lw=1.4, label="Janela observada")
    ax[0].plot(tf_h, real, color=DEEP, lw=1.2, ls="--", label="Valor real")
    ax[0].plot(tf_h, yh * sd_a + mu_a, color=TEALD, lw=2.0, label="Previsão da NFH")
    ax[0].set_ylabel("Temperatura (°C)")
    ax[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, ncol=3, fontsize=9.5)
    ax[1].plot(tf_h, T * sd_a + mu_a, color=WINE, lw=1.8)
    ax[1].set_ylabel("Tendência (°C)")
    ax[2].plot(tf_h, S * sd_a, color=TEALD, lw=1.8, label="Sazonalidade $\\mathbf{S}$")
    ax[2].plot(tf_h, Sk[k_dom] * sd_a, color=LILAC, lw=1.2, ls="--",
               label=f"Harmônico $k={k_dom + 1}$ (período de {H / (k_dom + 1):.0f} h)")
    ax[2].legend(loc="upper left", frameon=False, ncol=2, fontsize=9)
    ax[2].set_ylabel("Sazonalidade (°C)")
    ax[3].plot(tf_h, R * sd_a, color=PLUM, lw=1.4)
    ax[3].set_ylabel("Resíduo (°C)")
    ax[3].set_xlabel("Horas em relação ao início do horizonte")
    for a_ in ax:
        a_.axvline(0, color=LILAC, lw=0.8, alpha=0.6)
        a_.grid(alpha=0.25, color=LILAC)
        a_.yaxis.set_major_formatter(virgula)
        a_.xaxis.set_major_formatter(virgula)
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    fig.align_ylabels(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_exemplo_jena.pdf", bbox_inches="tight")
    fig.savefig(PNG / "fig_exemplo_jena.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({k: res[k] for k in ("inicio_horizonte", "rmse_janela", "nivel_previsto_C", "rms_forma_C",
                                          "eps_T", "eps_S", "eps_R", "harmonicos_top")}, indent=1, ensure_ascii=False))
    for c in ("T", "S"):
        print(c, "IG", {k: round(v, 1) for k, v in comp[c]["participacao_pct"].items()}, "erro", comp[c]["erro_completude"],
              "dom", comp[c]["variavel_dominante"], round(comp[c]["ultimas24h_pct_da_variavel_dominante"], 1))
        for mn, v in outros[c].items():
            print("  ", mn, {k: round(x_, 1) for k, x_ in v.items()})
        print("   agg", {k: round(v, 1) for k, v in agg_res[c]["media"].items()}, agg_res[c]["fracao_dominante"])


if __name__ == "__main__":
    main()
