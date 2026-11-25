"""
sinteticos.py: geradores dos conjuntos sintéticos do experimento de backbones.

SINT-U48 (univariado, horário, H = 48; fórmula própria desta rodada, documentada abaixo; o relatório
do pesquisador não existia quando o código foi escrito; se existir, ver LEIAME):

    s_t = tendência_t + A1_t sin(2π t/24 + φ1_t) + A2_t sin(2π t/12 + φ2_t)
    y_t = s_t + e_t,   e_t ~ N(0, σ²) independente (ruído branco)

  tendência_t = 0,8 sin(2π t/2600 + ψ0) + 0,4 u_t, com u_t um AR(1) (φ = 0,999) suavizado por média móvel
                de 24 h e reescalado a desvio 1 (deriva lenta e suave; em 48 h quase um polinômio);
  A1_t = 1,0 + 0,25 sin(2π t/1300 + ψ1);  A2_t = 0,5 + 0,15 sin(2π t/1700 + ψ2)   (amplitudes lentas);
  φ1_t = φ10 + 0,4 sin(2π t/2100 + ψ3);   φ2_t = φ20 + 0,4 sin(2π t/1900 + ψ4)   (fases lentas;
  a taxa máxima é 0,4·2π/1900 ≈ 1,3e-3 rad/h, ou seja, ≤ 0,07 rad em 48 h).
  σ = 0,30. N = 12000 h. Semente 2026. Ciclos de 24 h e 12 h são os harmônicos k = 2 e k = 4 de H = 48.

Por que o futuro é previsível da janela de 96 h: amplitude, fase e nível variam pouco em 48 h, então a
janela (4 ciclos de 24 h) determina o futuro a menos do ruído e de uma deriva pequena.

Oráculo declarado: a previsão y_hat = s_{t+h} (sinal sem ruído, incluindo a deriva futura). Seu erro é o
ruído e_t; RMSE_oraculo = σ (em unidades originais) e, na escala padronizada pelo treino que o carregador
usa, σ_emp/sd_treino. É uma REFERÊNCIA AMOSTRAL, não um piso rígido: o RMSE é de uma amostra finita de ruído, então
um modelo pode ficar um pouco abaixo dele por acaso. O número que o texto cita é o do oráculo sobre as janelas de
teste que o experimento avalia (`RMSE_oraculo_padronizado_janelas_teste`; alvos de b+L em diante, cada janela
com seus H = 48 alvos); o calculado sobre todo o segmento de teste (`..._padronizado_teste`) fica gravado só como
registro.
O arquivo `dados/sint_u48_oraculo.json` grava também, como referência, o RMSE de um previsor linear de
mínimos quadrados (ridge) ajustado nas janelas de treino.

SINT-M: o de src/v3_sintetico_multivariado.py (função `gerar`, semente 2026), reaproveitado sem mudança.
Diferença de formato: o catálogo exige o alvo entre as variáveis (modo M = covariáveis + alvo, o alvo é
a ÚLTIMA coluna, porque o InstanceNorm desnormaliza a última variável). Portanto a entrada tem d = 5
(x1..x4 e o próprio y passado), enquanto o experimento E4 da v3 usava d = 4 sem o y. O alvo continua
y_t = x1_{t-48} + x2_{t-48} + e_t (e_t AR(1), φ = 0,5).

Uso:  python sinteticos.py            (gera CSVs, catalogo.json e confere SINT-M contra o gerador)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
DADOS = ROOT / "experimento_backbones_nfh" / "dados"
sys.path.insert(0, str(ROOT / "src"))

N_U48, SEED_U48, SIGMA_U48 = 12000, 2026, 0.30


def _ar1(rng, n, phi, sigma):
    z = np.zeros(n)
    e = rng.normal(0, sigma, n)
    for i in range(1, n):
        z[i] = phi * z[i - 1] + e[i]
    return z


def gerar_u48(seed=SEED_U48, n=N_U48, sigma=SIGMA_U48):
    """Devolve (y, sinal_sem_ruido, ruido); unidades originais (antes da padronização)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    psi = rng.uniform(0, 2 * np.pi, 5)
    f10, f20 = rng.uniform(0, 2 * np.pi, 2)
    u = np.convolve(_ar1(rng, n, 0.999, 0.05), np.ones(24) / 24, mode="same")
    u = (u - u.mean()) / (u.std() + 1e-12)
    tend = 0.8 * np.sin(2 * np.pi * t / 2600 + psi[0]) + 0.4 * u
    a1 = 1.0 + 0.25 * np.sin(2 * np.pi * t / 1300 + psi[1])
    a2 = 0.5 + 0.15 * np.sin(2 * np.pi * t / 1700 + psi[2])
    ph1 = f10 + 0.4 * np.sin(2 * np.pi * t / 2100 + psi[3])
    ph2 = f20 + 0.4 * np.sin(2 * np.pi * t / 1900 + psi[4])
    s = tend + a1 * np.sin(2 * np.pi * t / 24 + ph1) + a2 * np.sin(2 * np.pi * t / 12 + ph2)
    e = rng.normal(0, sigma, n)
    return s + e, s, e


def _datas(n):
    return pd.date_range("2020-01-01", periods=n, freq="h")


def _escrever(nome, df, entrada, ext=None):
    DADOS.mkdir(parents=True, exist_ok=True)
    df.to_csv(DADOS / f"{nome}.csv", index=False, float_format="%.9g")


def _referencia_ridge(y_z, cortes, L=96, H=48, lam=1.0):
    """RMSE de teste (escala padronizada) de um ridge linear janela -> horizonte, ajustado no treino."""
    a, b = cortes

    def jan(seg):
        n = len(seg) - L - H + 1
        X = np.stack([seg[i:i + L] for i in range(n)])
        Y = np.stack([seg[i + L:i + L + H] for i in range(n)])
        return X, Y
    Xa, Ya = jan(y_z[:a])
    Xt, Yt = jan(y_z[b:])
    Xa1, Xt1 = np.c_[Xa, np.ones(len(Xa))], np.c_[Xt, np.ones(len(Xt))]
    W = np.linalg.solve(Xa1.T @ Xa1 + lam * np.eye(Xa1.shape[1]), Xa1.T @ Ya)
    return float(np.sqrt(np.mean((Xt1 @ W - Yt) ** 2)))


def _rmse_oraculo_janelas(e, b, sd, L=96, H=48):
    """RMSE (padronizado) do oráculo sobre as janelas de teste (alvos de b+L em diante): erro do oráculo =
    ruído e_t dos H alvos de cada janela. Janelas i = b .. n-L-H, como segmentos_bb."""
    n = len(e)
    nj = n - b - L - H + 1
    erros = np.stack([e[b + i + L:b + i + L + H] for i in range(nj)])
    return float(np.sqrt(np.mean(erros ** 2)) / sd), nj


def gravar_u48():
    y, s, e = gerar_u48()
    n = len(y)
    a, b = int(0.70 * n), int(0.85 * n)
    mu, sd = float(y[:a].mean()), float(y[:a].std(ddof=1)) + 1e-12      # como o carregador (pandas ddof=1)
    y_z = (y - mu) / sd
    df = pd.DataFrame({"date": _datas(n), "y": y, "sinal_sem_ruido": s})
    _escrever("SINT-U48", df[["date", "y"]], None)
    df.to_csv(DADOS / "SINT-U48_com_sinal.csv", index=False, float_format="%.9g")   # y + sinal sem ruído
    rmse_jan, n_jan = _rmse_oraculo_janelas(e, b, sd)
    orac = {
        "formula": "ver docstring de sinteticos.py",
        "N": n, "seed": SEED_U48, "sigma": SIGMA_U48, "mu_treino": mu, "sd_treino": sd,
        "oraculo": "previsão = sinal sem ruído do futuro (inclui a deriva futura); referência amostral de RMSE (não é piso rígido)",
        "RMSE_oraculo_original": {"treino": float(e[:a].std()), "val": float(e[a:b].std()),
                                   "teste": float(np.sqrt(np.mean(e[b:] ** 2)))},
        "RMSE_oraculo_padronizado_janelas_teste": rmse_jan,
        "n_janelas_teste": n_jan,
        "referencia_amostral": "RMSE_oraculo_padronizado_janelas_teste é o número a citar (janelas de teste avaliadas)",
        "RMSE_oraculo_padronizado_teste": float(np.sqrt(np.mean(e[b:] ** 2)) / sd),   # segmento inteiro, só registro
        "RMSE_ridge_linear_padronizado_teste": _referencia_ridge(y_z, (a, b)),
        "arquivo_sinal_sem_ruido": "SINT-U48_com_sinal.csv (coluna sinal_sem_ruido)",
    }
    (DADOS / "sint_u48_oraculo.json").write_text(json.dumps(orac, indent=2, ensure_ascii=False))
    return orac


def gravar_m():
    from v3_sintetico_multivariado import gerar
    Xz, yz, obs, (a, b) = gerar()
    n = len(yz)
    df = pd.DataFrame(Xz.astype(np.float64), columns=["x1_tendencia", "x2_ciclo", "x3_distrator_lento",
                                                      "x4_distrator_oscilante"])
    df.insert(0, "date", _datas(n))
    df["y"] = yz.astype(np.float64)
    _escrever("SINT-M", df, None)
    return {"n": n, "cortes_gerador": [int(a), int(b)]}


def entradas_catalogo(n_u, n_m):
    base = {"freq": "H", "licenca": "sintético (gerado por este repositório)",
            "faltantes_frac": 0.0, "origem": "sintético"}
    return [
        {**base, "nome": "SINT-U48", "arquivo": "SINT-U48.csv", "fonte_url": "sinteticos.py (gerar_u48)",
         "alvo": "y", "covariaveis": [], "modos": ["U"], "periodos_dominantes_h": [24, 12],
         "cortes": [0.7, 0.15], "n_obs": n_u,
         "notas": "Univariado, H=48; ciclos de 24 h e 12 h com amplitude/fase lentas, tendência suave, ruído branco "
                  "sigma=0,3. Oráculo (referência amostral) e RMSE em dados/sint_u48_oraculo.json."},
        {**base, "nome": "SINT-M", "arquivo": "SINT-M.csv", "fonte_url": "src/v3_sintetico_multivariado.py (gerar)",
         "alvo": "y", "covariaveis": ["x1_tendencia", "x2_ciclo", "x3_distrator_lento", "x4_distrator_oscilante"],
         "modos": ["M"], "periodos_dominantes_h": [24],
         "cortes": [0.7, 0.15], "n_obs": n_m,
         "notas": "y_t = x1_{t-48} + x2_{t-48} + AR(1); já padronizado pelo treino; o alvo y entra como última "
                  "variável (d=5)."},
    ]


def conferir_m():
    """Confere que o conjunto lido pelo carregador do catálogo reproduz `gerar` do v3."""
    from v3_sintetico_multivariado import gerar
    from dados_bb import carregar_bb
    Xz, yz, obs, (a, b) = gerar()
    ds = carregar_bb("SINT-M-M")
    assert ds["feats"].shape == (len(yz), 5), ds["feats"].shape
    assert tuple(ds["cortes"]) == (a, b), (ds["cortes"], (a, b))
    erro_x = float(np.abs(ds["feats"][:, :4] - Xz).max())
    erro_y = float(np.abs(ds["feats"][:, 4] - yz).max())
    assert erro_x < 5e-3 and erro_y < 5e-3, (erro_x, erro_y)     # re-padronização com ddof=1 e CSV em %.9g
    return {"erro_max_x": erro_x, "erro_max_y": erro_y}


def main():
    orac = gravar_u48()
    inf_m = gravar_m()
    novas = entradas_catalogo(orac["N"], inf_m["n"])
    cam = DADOS / "catalogo.json"
    # junta, sem apagar entradas de outros agentes (os conjuntos reais novos da pesquisa)
    atual = json.loads(cam.read_text(encoding="utf-8")) if cam.exists() else []
    nomes_novos = {e["nome"] for e in novas}
    cat = novas + [e for e in atual if e["nome"] not in nomes_novos]
    cam.write_text(json.dumps(cat, indent=2, ensure_ascii=False))
    conf = conferir_m()
    print("SINT-U48:", json.dumps({k: orac[k] for k in ("RMSE_oraculo_padronizado_janelas_teste", "n_janelas_teste", "RMSE_oraculo_padronizado_teste",
                                                         "RMSE_ridge_linear_padronizado_teste")}))
    print("SINT-M confere com v3_sintetico_multivariado.gerar:", conf)


if __name__ == "__main__":
    main()
