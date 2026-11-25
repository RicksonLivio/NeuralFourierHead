"""Verificacoes numericas citadas em relatorio_backbones.md (secao SINT-U48). Somente leitura de codigo/sinteticos.py.
Uso: .venv/bin/python experimento_backbones_nfh/pesquisa/verificacao_sint_u48.py > experimento_backbones_nfh/pesquisa/verificacao_sint_u48.txt"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codigo"))
import sinteticos as S

y, s, e = S.gerar_u48(); n = len(y); a, b = int(.7 * n), int(.85 * n); L, H = 96, 48
mu, sd = y[:a].mean(), y[:a].std(ddof=1); z = (y - mu) / sd
t = np.arange(H); tn = (t - (H - 1) / 2) / H


def basis(q, K):
    c = [tn ** j for j in range(q + 1)]
    for k in range(1, K + 1):
        c += [np.cos(2 * np.pi * k * t / H), np.sin(2 * np.pi * k * t / H)]
    return np.stack(c, 1)


print(f"N={n} sigma={S.SIGMA_U48} sd_treino={sd:.4f} sigma/sd={S.SIGMA_U48/sd:.4f}")
print("1) cobertura do SINAL futuro (sem ruido) por T+S: 1-eps_R medio sobre janelas de 48 h (passo 24)")
for K in (2, 4, 10):
    B = basis(2, K); P = B @ np.linalg.pinv(B)
    mse = np.mean([np.mean((s[i:i+H] - P @ s[i:i+H]) ** 2) for i in range(0, n - H, 24)])
    var = np.mean([np.var(s[i:i+H]) for i in range(0, n - H, 24)])
    print(f"   q=2 K={K}: colunas={B.shape[1]}  1-eps_R={1 - mse/var:.6f}")
print("2) RMSE padronizado de teste de um oraculo que preve o sinal futuro projetado em T+S (R=0), contra y futuro ruidoso")
for K in (2, 4, 10):
    B = basis(2, K); P = B @ np.linalg.pinv(B)
    r = [np.mean((P @ s[i:i+H] - y[i:i+H]) ** 2) for i in range(b, n - H)]
    print(f"   K={K}: {np.sqrt(np.mean(r))/sd:.4f}   (piso = oraculo sem projecao: {np.sqrt(np.mean(e[b:]**2))/sd:.4f})")
print("3) Referencia LINEAR (OLS, NAO e a NFH): janela 96 -> 48, instance norm por media, ultimas janelas do treino")


def jan(lo, hi):
    m = hi - lo - L - H + 1
    return (np.stack([z[lo+i:lo+i+L] for i in range(m)]), np.stack([z[lo+i+L:lo+i+L+H] for i in range(m)]))


def norm(X, Y):
    m = X.mean(1, keepdims=True); return X - m, Y - m

Xt, Yt = norm(*jan(b, n)); Xa, Ya = jan(0, a); N0 = len(Xa)
for K in (4, 10):
    B = basis(2, K); P = B @ np.linalg.pinv(B)
    for fr in (1.0, .25, .10):
        m = int(N0 * fr); X, Y = norm(Xa[-m:], Ya[-m:])
        X1 = np.c_[X, np.ones(m)]; Xt1 = np.c_[Xt, np.ones(len(Xt))]
        W = np.linalg.lstsq(X1, Y, rcond=None)[0]; Wp = np.linalg.lstsq(X1, Y @ P.T, rcond=None)[0]
        r = lambda W: np.sqrt(np.mean((Xt1 @ W - Yt) ** 2))
        print(f"   K={K} fracao={fr}: janelas={m}  OLS denso={r(W):.4f}  OLS com saida em T+S (R=0)={r(Wp):.4f}")
