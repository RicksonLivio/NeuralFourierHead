"""
E4-v3: atribuição multivariada com verdade de referência (resposta ao áudio 1: "no multivariado
não há comparação quantitativa").

Série horária sintética SINT-M com quatro variáveis de entrada e alvo y:
  x1  condutor da tendência: processo lento (AR(1), phi = 0,998, suavizado por média de 24 h);
  x2  condutor do ciclo: a_t sin(2 pi t / 24 + phi2), amplitude a_t = exp(0,4 z_t) com z_t AR(1);
  x3  distrator lento, do mesmo tipo de x1 (independente);
  x4  distrator oscilante, do mesmo tipo de x2 (independente, outra fase);
  y_t = x1_{t-48} + x2_{t-48} + e_t,  e_t AR(1) (phi = 0,5).
Como o atraso (48 h) é igual ao horizonte, o futuro de y é determinado pelas últimas 48 h de x1 e
x2, que estão na janela; o alvo não entra como entrada (d = 4), para que a verdade de referência
seja inequívoca: a tendência prevista vem de x1 e o ciclo diário previsto vem de x2.

Pergunta avaliada em 50 janelas de teste: qual variável sustenta a tendência prevista e qual
sustenta o ciclo? Acerto = a variável de maior |atribuição| (soma sobre defasagens) é a correta.
  * NFH (Conv1D+LSTM, conversão + ajuste estrutural, K = 10): IG sobre g_T (nível da tendência) e
    sobre g_S (energia da sazonalidade); também KernelSHAP e Feature Ablation sobre g_T e g_S;
  * densa + projeção: IG sobre as componentes obtidas projetando a previsão densa;
  * sem decomposição (prática usual): IG, KernelSHAP, LIME, Saliency e Feature Ablation sobre a
    média da previsão densa e sobre a energia da forma ||y_c||^2; a mesma ordenação responde às
    duas perguntas.
Saída: results/v3/sint_m.json, results/v3/sint_m_janelas.csv
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from v2_common import (BackboneHeadModel, ConvLSTMBackbone, DenseHead, NFHHead, set_seed,  # noqa: E402
                       stack_ds, train_model)
from v3_common import CFG3, H3, K3, L3, Q3, RES3, MascWindows, ajuste_estrutural, avaliar, matriz_sintese  # noqa: E402

warnings.filterwarnings("ignore")
N, ATRASO, N_JANELAS = 12000, 48, 50
NOMES = ["x1 (tendência)", "x2 (ciclo)", "x3 (distrator lento)", "x4 (distrator oscilante)"]


def ar1(rng, n, phi, sigma):
    z = np.zeros(n)
    e = rng.normal(0, sigma, n)
    for i in range(1, n):
        z[i] = phi * z[i - 1] + e[i]
    return z


def gerar(seed=2026):
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    suav = np.ones(24) / 24
    x1 = np.convolve(ar1(rng, N, 0.998, 0.06), suav, mode="same")
    x3 = np.convolve(ar1(rng, N, 0.998, 0.06), suav, mode="same")
    x2 = np.exp(0.4 * ar1(rng, N, 0.99, 0.14)) * np.sin(2 * np.pi * t / 24 + 0.3)
    x4 = np.exp(0.4 * ar1(rng, N, 0.99, 0.14)) * np.sin(2 * np.pi * t / 24 + 1.4)
    y = np.full(N, np.nan)
    y[ATRASO:] = x1[:-ATRASO] + x2[:-ATRASO]
    y = y + ar1(rng, N, 0.5, 0.15)
    X = np.stack([x1, x2, x3, x4], 1)[ATRASO:]
    y = y[ATRASO:]
    a, b = int(0.7 * len(y)), int(0.85 * len(y))
    mu, sd = X[:a].mean(0), X[:a].std(0)
    my, sy = y[:a].mean(), y[:a].std()
    Xz = ((X - mu) / sd).astype(np.float32)
    yz = ((y - my) / sy).astype(np.float32)
    obs = np.ones(len(yz), dtype=bool)
    return Xz, yz, obs, (a, b)


def modelo(cabeca, d=4, K=K3):
    phi = ConvLSTMBackbone(d, 128, 64)
    head = DenseHead(128, H3) if cabeca == "densa" else NFHHead(128, H3, K, Q3, normalize=True)
    return BackboneHeadModel(phi, head)


def converter(densa, K=K3):
    nfh = modelo("nfh", K=K)
    nfh.backbone.load_state_dict(densa.backbone.state_dict())
    M, *_ = matriz_sintese(H3, K, Q3)
    MD = M * nfh.head.D.cpu().numpy().astype(np.float64)[None, :]
    W = densa.head.lin.weight.detach().cpu().double().numpy()
    b = densa.head.lin.bias.detach().cpu().double().numpy()
    nfh.head.lin.weight.data = torch.tensor(np.linalg.solve(MD, W), dtype=torch.float32)
    nfh.head.lin.bias.data = torch.tensor(np.linalg.solve(MD, b), dtype=torch.float32)
    return nfh.to(next(densa.parameters()).device)


class Funcional(torch.nn.Module):
    """Saída escalar a explicar, sobre a janela achatada (L*d)."""

    def __init__(self, m, qual, PiT=None, PiR=None):
        super().__init__()
        self.m, self.qual = m, qual
        if PiT is not None:
            self.register_buffer("PiT", torch.tensor(PiT, dtype=torch.float32))
            self.register_buffer("PiR", torch.tensor(PiR, dtype=torch.float32))

    def forward(self, flat):
        o = self.m(flat.reshape(flat.shape[0], L3, -1))
        y = o["forecast"]
        if self.qual == "media":
            return y.mean(1, keepdim=True)
        if self.qual == "forma":
            yc = y - y.mean(1, keepdim=True)
            return (yc ** 2).sum(1, keepdim=True)
        if self.qual in ("T", "S") and "trend" in o:
            T, S = o["trend"], o["seasonal"]
        else:  # densa + projeção
            T = y @ self.PiT.T
            S = y - T - y @ self.PiR.T
        return T.mean(1, keepdim=True) if self.qual.startswith("T") else (S ** 2).sum(1, keepdim=True)


def por_variavel(a, d=4):
    return np.abs(np.asarray(a).reshape(L3, d).sum(0))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    import shap
    from captum.attr import FeatureAblation, IntegratedGradients, Saliency
    from lime.lime_tabular import LimeTabularExplainer

    Xz, yz, obs, (a, b) = gerar()
    tr = MascWindows(Xz[:a], yz[:a], obs[:a], L3, H3)
    va = MascWindows(Xz[a:b], yz[a:b], obs[a:b], L3, H3)
    te = MascWindows(Xz[b:], yz[b:], obs[b:], L3, H3)
    Xte, Yte = stack_ds(te)
    from v2_common import nfh_bases
    P, F, Ft, B, Qp = nfh_bases(H3, K3, Q3)
    PiT = P.T @ np.linalg.solve(P @ P.T, P)
    PiR = Qp @ Qp.T
    resultado = {"config": {"N": N, "atraso": ATRASO, "L": L3, "H": H3, "K": K3, "q": Q3, "janelas_teste": N_JANELAS,
                            "n_treino": len(tr), "n_val": len(va), "n_teste": len(te)}, "sementes": {}}
    idx = np.linspace(0, len(Xte) - 1, N_JANELAS).astype(int)
    linhas = []
    for seed in (0, 1, 2, 3, 4):
        torch.backends.cudnn.enabled = True
        set_seed(seed)
        densa, info = train_model(modelo("densa"), tr, va, seed, lam=0.0, cfg=CFG3, dev=dev)
        nfh, _, aj = ajuste_estrutural(converter(densa), tr, va, seed, dev=dev)
        for p_ in list(densa.parameters()) + list(nfh.parameters()):
            p_.requires_grad_(False)
        # o backward de LSTM via cuDNN exige modo de treino; para as atribuições (modo de avaliação)
        # o cuDNN é desligado, na mesma condição para todos os métodos
        torch.backends.cudnn.enabled = False
        rd, kd, _, _ = avaliar(densa, Xte, Yte, dev)
        rn, kn, _, _ = avaliar(nfh, Xte, Yte, dev)
        alvos = {"NFH | IG sobre g_T e g_S": ("nfh", "IG"),
                 "densa + projeção | IG sobre g_T e g_S": ("proj", "IG"),
                 "densa | IG sobre a média": ("densa_media", "IG"),
                 "densa | IG sobre a energia da forma": ("densa_forma", "IG")}
        if seed == 0:
            alvos.update({"NFH | KernelSHAP sobre g_T e g_S": ("nfh", "SHAP"),
                          "NFH | Feature Ablation sobre g_T e g_S": ("nfh", "FA"),
                          "densa | KernelSHAP sobre a média": ("densa_media", "SHAP"),
                          "densa | LIME sobre a média": ("densa_media", "LIME"),
                          "densa | Saliency sobre a média": ("densa_media", "Saliency"),
                          "densa | Feature Ablation sobre a média": ("densa_media", "FA")})
        fun = {"nfh_T": Funcional(nfh, "T"), "nfh_S": Funcional(nfh, "S"),
               "proj_T": Funcional(densa, "Tp", PiT, PiR), "proj_S": Funcional(densa, "Sp", PiT, PiR),
               "densa_media": Funcional(densa, "media"), "densa_forma": Funcional(densa, "forma")}
        for f_ in fun.values():
            f_.to(dev).eval()
        Xtr, _ = stack_ds(tr)
        bg = Xtr[np.random.default_rng(0).choice(len(Xtr), 200, replace=False)].reshape(200, -1).numpy()
        bg_sum = shap.kmeans(bg, 10)

        def numpy_de(f_):
            def g(A):
                with torch.no_grad():
                    return np.concatenate([f_(torch.tensor(A[i:i + 2048], dtype=torch.float32, device=dev)).cpu().numpy()[:, 0]
                                           for i in range(0, len(A), 2048)])
            return g

        def atribuir(f_, metodo, xf):
            if metodo == "IG":
                inp = torch.tensor(xf[None], dtype=torch.float32, device=dev)
                return IntegratedGradients(f_).attribute(inp, baselines=0.0, n_steps=64).cpu().numpy()[0]
            if metodo == "FA":
                inp = torch.tensor(xf[None], dtype=torch.float32, device=dev)
                return FeatureAblation(f_).attribute(inp, baselines=0.0, perturbations_per_eval=128).cpu().numpy()[0]
            if metodo == "Saliency":
                inp = torch.tensor(xf[None], dtype=torch.float32, device=dev, requires_grad=True)
                return Saliency(f_).attribute(inp, abs=False).detach().cpu().numpy()[0]
            if metodo == "SHAP":
                ex = shap.KernelExplainer(numpy_de(f_), bg_sum)
                return np.asarray(ex.shap_values(xf[None], nsamples=2 * len(xf) + 2048, silent=True)).reshape(-1)
            if metodo == "LIME":
                ex = LimeTabularExplainer(bg, mode="regression", discretize_continuous=False, verbose=False,
                                          random_state=0)
                e = ex.explain_instance(xf, numpy_de(f_), num_features=len(xf), num_samples=1000)
                v = np.zeros(len(xf))
                for j, w in e.as_map()[1]:
                    v[j] = w
                return v
            raise ValueError(metodo)

        acertos = {k: {"T": [], "S": [], "part_T": [], "part_S": [], "tempo": []} for k in alvos}
        for si in idx:
            xf = Xte[si].reshape(-1).numpy().astype(np.float64)
            for rot, (fonte, metodo) in alvos.items():
                t0 = time.perf_counter()
                if fonte in ("nfh", "proj"):
                    vT = por_variavel(atribuir(fun[f"{fonte}_T"], metodo, xf))
                    vS = por_variavel(atribuir(fun[f"{fonte}_S"], metodo, xf))
                else:
                    vT = vS = por_variavel(atribuir(fun[fonte], metodo, xf))
                tm = time.perf_counter() - t0
                acertos[rot]["T"].append(int(np.argmax(vT) == 0))
                acertos[rot]["S"].append(int(np.argmax(vS) == 1))
                acertos[rot]["part_T"].append(float(vT[0] / (vT.sum() + 1e-12)))
                acertos[rot]["part_S"].append(float(vS[1] / (vS.sum() + 1e-12)))
                acertos[rot]["tempo"].append(tm)
                linhas.append({"seed": seed, "janela": int(si), "metodo": rot, "acerto_T": acertos[rot]["T"][-1],
                               "acerto_S": acertos[rot]["S"][-1], "part_x1_T": acertos[rot]["part_T"][-1],
                               "part_x2_S": acertos[rot]["part_S"][-1], "tempo_s": tm})
        resumo = {rot: {"acerto_tendencia": float(np.mean(v["T"])), "acerto_ciclo": float(np.mean(v["S"])),
                        "participacao_x1_na_tendencia": float(np.mean(v["part_T"])),
                        "participacao_x2_no_ciclo": float(np.mean(v["part_S"])),
                        "tempo_medio_s": float(np.mean(v["tempo"]))} for rot, v in acertos.items()}
        resultado["sementes"][seed] = {"RMSE_densa": rd, "RMSE_nfh": rn, "kappa_densa": kd, "kappa_nfh": kn,
                                       "epocas_densa": info["epochs"], "epoca_ajuste": aj["seguro"]["epoca"],
                                       "metodos": resumo}
        print(f"[{time.strftime('%H:%M:%S')}] semente {seed}: RMSE densa {rd:.4f} NFH {rn:.4f} kappa {kd:.3f}->{kn:.3f}",
              flush=True)
        for rot, v in resumo.items():
            print(f"   {rot}: tendência {v['acerto_tendencia']:.2f} ciclo {v['acerto_ciclo']:.2f} "
                  f"(part. {v['participacao_x1_na_tendencia']:.2f}/{v['participacao_x2_no_ciclo']:.2f}) {v['tempo_medio_s']*1000:.0f} ms",
                  flush=True)
        (RES3 / "sint_m.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
        pd.DataFrame(linhas).to_csv(RES3 / "sint_m_janelas.csv", index=False)


if __name__ == "__main__":
    main()
