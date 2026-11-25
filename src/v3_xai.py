"""
E2-v3: comparação de explicabilidade nos 8 conjuntos públicos, todos os métodos aplicados ao MESMO
modelo (Conv1D+LSTM+NFH, K = 10, seleção segura, semente 0) e às MESMAS janelas de teste.

Convenção (a mesma do exemplo multivariado da v2): a normalização reversível por instância (RevIN) é
determinística, então explica-se o modelo interno sobre a janela normalizada X_n, com referência
X_n' = 0 (janela igual à própria média). Funcionais explicados:
  g_T(X_n) = média de T_n (nível previsto; pelo Teorema do balanço de energia é também a média de y_hat_n)
  g_S(X_n) = ||S_n||^2   (energia da sazonalidade; só existe com a decomposição)
Métodos de atribuição de entrada (para g_T e para g_S): KernelSHAP, LIME (amostras em torno da
instância), Integrated Gradients (64 passos; para a NFH, é a atribuição por componente da Eq. do Cap. 4),
Saliency e Feature Ablation. Explicação estrutural da NFH: vetor (eps_T, e_1..e_K, eps_R).
Métricas: tempo por explicação (GPU, sincronizado), robustez (cosseno entre explicações da janela e da
janela com ruído N(0, 0,05^2) na escala padronizada, 3 sorteios) e AOPC relativo (deleção cumulativa
das entradas de maior |phi_i| nas frações 1, 2, 5, 10 e 20%, média de |g(X) - g(X_del)|, dividida pela
média de 10 ordens aleatórias).
Saída: results/v3/xai_janelas.csv, results/v3/xai_por_conjunto.csv, results/v3/xai_resumo.csv
Uso: python src/v3_xai.py [N_JANELAS] [CONJ ...]
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from v2_common import stack_ds  # noqa: E402
from v3_common import CONJUNTOS, RES3, carregar, construir, segmentos  # noqa: E402

warnings.filterwarnings("ignore")
SIGMA, TRIALS = 0.05, 3
FRACS = [0.01, 0.02, 0.05, 0.10, 0.20]
BB = "Conv1D+LSTM"


def normalizar(x, eps=1e-5):
    mu = x.mean(1, keepdim=True)
    sd = x.std(1, keepdim=True) + eps
    return (x - mu) / sd


class Funcional(torch.nn.Module):
    """g_T ou g_S do modelo interno, sobre a janela normalizada achatada."""

    def __init__(self, inner, L, d, chave):
        super().__init__()
        self.inner, self.L, self.d, self.chave = inner, L, d, chave

    def forward(self, flat):
        out = self.inner(flat.reshape(flat.shape[0], self.L, self.d))
        if self.chave == "T":
            return out["trend"].mean(1, keepdim=True)
        return (out["seasonal"] ** 2).sum(1, keepdim=True)


def vetor_estrutural(inner, xn):
    out = inner(xn)
    head = inner.head
    y, T, S, R = out["forecast"], out["trend"], out["seasonal"], out["residual"]
    yc = y - y.mean(1, keepdim=True)
    Tc = T - T.mean(1, keepdim=True)
    E = (yc ** 2).sum(1, keepdim=True) + 1e-30
    th = out["theta_S"]
    ek = [((th[:, 2 * k:2 * k + 1] * head.Ft[2 * k] + th[:, 2 * k + 1:2 * k + 2] * head.Ft[2 * k + 1]) * S).sum(1, keepdim=True) / E
          for k in range(th.shape[1] // 2)]
    return torch.cat([(Tc ** 2).sum(1, keepdim=True) / E] + ek + [(R ** 2).sum(1, keepdim=True) / E], 1)


def cos(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main(n_jan=10, conjuntos=None):
    import shap
    from captum.attr import FeatureAblation, IntegratedGradients, Saliency
    from lime.lime_tabular import LimeTabularExplainer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = False      # backward da LSTM em modo de avaliação; mesma condição para todos
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    linhas = []
    for nome in conjuntos or CONJUNTOS:
        ds = carregar(nome)
        tr, va, te = segmentos(ds)
        d = ds["feats"].shape[1]
        modelo = construir(BB, "nfh", d)
        modelo.load_state_dict(torch.load(RES3 / "modelos" / f"{nome}__{BB}+NFH__s0.pt", map_location="cpu"))
        modelo = modelo.to(dev).eval()
        for p in modelo.parameters():
            p.requires_grad_(False)
        inner = modelo.inner
        Xtr, _ = stack_ds(tr)
        Xte, _ = stack_ds(te)
        L = Xte.shape[1]
        n = L * d
        idx = np.linspace(0, len(Xte) - 1, n_jan).astype(int)
        bg_idx = rng.choice(len(Xtr), size=min(200, len(Xtr)), replace=False)
        bg = normalizar(Xtr[bg_idx]).reshape(len(bg_idx), -1).numpy()
        bg_sum = shap.kmeans(bg, 10)
        lime_ex = LimeTabularExplainer(bg, mode="regression", discretize_continuous=False, verbose=False,
                                       random_state=0, sample_around_instance=True)
        ns_shap = int(min(2 * n + 2048, 3000))
        for chave in ("T", "S"):
            fun = Funcional(inner, L, d, chave).to(dev).eval()

            def g_np(A, fun=fun):
                with torch.no_grad():
                    return np.concatenate([fun(torch.tensor(A[i:i + 4096], dtype=torch.float32, device=dev)).cpu().numpy()[:, 0]
                                           for i in range(0, len(A), 4096)])

            def m_shap(xf, g_np=g_np):
                ex = shap.KernelExplainer(g_np, bg_sum)
                return np.asarray(ex.shap_values(xf.reshape(1, -1), nsamples=ns_shap, silent=True)).reshape(-1)

            def m_lime(xf, g_np=g_np):
                e = lime_ex.explain_instance(xf, g_np, num_features=n, num_samples=1000)
                a = np.zeros(n)
                for j, w in e.as_map()[1]:
                    a[j] = w
                return a

            def captum(cls, fun=fun, **kw):
                def fn(xf):
                    inp = torch.tensor(xf.reshape(1, -1), dtype=torch.float32, device=dev)
                    if cls is Saliency:
                        inp.requires_grad_(True)
                    return cls(fun).attribute(inp, **kw).detach().cpu().numpy().reshape(-1)
                return fn

            metodos = {"KernelSHAP": m_shap, "LIME": m_lime,
                       "Integrated Gradients": captum(IntegratedGradients, baselines=0.0, n_steps=64),
                       "Saliency": captum(Saliency, abs=False),
                       "Feature Ablation": captum(FeatureAblation, baselines=0.0, perturbations_per_eval=128)}
            # aquecimento (não cronometrado): a primeira chamada de cada método inclui compilação e alocação na GPU
            x_aq = normalizar(Xte[idx[0]:idx[0] + 1]).reshape(-1).numpy().astype(np.float64)
            for fn in metodos.values():
                fn(x_aq)
            with torch.no_grad():
                vetor_estrutural(inner, normalizar(Xte[:1]).to(dev))
            for si in idx:
                x = Xte[si:si + 1]
                xn = normalizar(x)
                xf = xn.reshape(-1).numpy().astype(np.float64)
                gx = float(g_np(xf[None])[0])
                ruidos = [normalizar(x + torch.tensor(rng.normal(0, SIGMA, size=x.shape), dtype=torch.float32)
                                     ).reshape(-1).numpy().astype(np.float64) for _ in range(TRIALS)]
                ordens = [rng.permutation(n) for _ in range(10)]

                def aopc(ordem, g_np=g_np, gx=gx, xf=xf):
                    xs = []
                    for fr in FRACS:
                        k = max(1, int(round(fr * n)))
                        xd = xf.copy()
                        xd[ordem[:k]] = 0.0
                        xs.append(xd)
                    return float(np.mean(np.abs(gx - g_np(np.stack(xs)))))

                a_ale = float(np.mean([aopc(o) for o in ordens]))
                if chave == "T":    # explicação estrutural: uma vez por janela
                    xt = xn.to(dev)
                    sync()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        e0 = vetor_estrutural(inner, xt).cpu().numpy()[0]
                    sync()
                    t_nfh = time.perf_counter() - t0
                    with torch.no_grad():
                        rob = [cos(e0, vetor_estrutural(inner, torch.tensor(r.reshape(xn.shape), dtype=torch.float32,
                                                                            device=dev)).cpu().numpy()[0]) for r in ruidos]
                    linhas.append({"conjunto": nome, "janela": int(si), "funcional": "-", "metodo": "NFH (estrutural)",
                                   "tempo_s": t_nfh, "robustez": float(np.mean(rob)), "aopc_rel": np.nan, "g": gx})
                for mn, fn in metodos.items():
                    sync()
                    t0 = time.perf_counter()
                    a = fn(xf)
                    sync()
                    tm = time.perf_counter() - t0
                    rob = [cos(a, fn(r)) for r in ruidos]
                    razao = aopc(np.argsort(-np.abs(a))) / (a_ale + 1e-12)
                    linhas.append({"conjunto": nome, "janela": int(si), "funcional": f"g_{chave}", "metodo": mn,
                                   "tempo_s": tm, "robustez": float(np.mean(rob)), "aopc_rel": float(razao), "g": gx})
                print(f"[{time.strftime('%H:%M:%S')}] {nome} g_{chave} janela {si}: " +
                      " | ".join(f"{r['metodo'][:6]} {r['tempo_s'] * 1e3:.0f}ms rob {r['robustez']:.3f} aopc {r['aopc_rel']:.2f}"
                                 for r in linhas[-len(metodos):]), flush=True)
        pd.DataFrame(linhas).to_csv(RES3 / "xai_janelas.csv", index=False)
    df = pd.DataFrame(linhas)
    df.to_csv(RES3 / "xai_janelas.csv", index=False)
    por = df.groupby(["metodo", "funcional", "conjunto"])[["tempo_s", "robustez", "aopc_rel"]].mean().reset_index()
    por.to_csv(RES3 / "xai_por_conjunto.csv", index=False)
    res = por.groupby(["metodo", "funcional"]).agg(tempo_media=("tempo_s", "mean"), tempo_dp=("tempo_s", "std"),
                                                   rob_media=("robustez", "mean"), rob_dp=("robustez", "std"),
                                                   aopc_media=("aopc_rel", "mean"), aopc_dp=("aopc_rel", "std"),
                                                   n_conjuntos=("conjunto", "nunique"))
    res.to_csv(RES3 / "xai_resumo.csv")
    print(res)


if __name__ == "__main__":
    nj = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(nj, sys.argv[2:] or None)
