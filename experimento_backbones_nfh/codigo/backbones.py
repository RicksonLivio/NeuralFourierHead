"""
backbones.py: backbones Phi do experimento de backbones da NFH (protocolo, §4).

Contrato de todo backbone:
  * forward(x: [B, L, d]) -> h: [B, p]; atributo `p` (dimensão de h); sem camada de saída própria;
  * a entrada chega JÁ normalizada pelo InstanceNorm (RevIN) que envolve o modelo; nenhum backbone
    normaliza a janela de novo (as LayerNorm internas dos blocos são parte da arquitetura);
  * o alvo é a última variável (a que o InstanceNorm desnormaliza); os backbones tratam todas as
    variáveis da janela como entrada;
  * `largura` em {1, 2} multiplica as dimensões internas (canais, unidades, d_model).
    `Linear` e `DLinear` não têm dimensão interna: a largura não muda nada (USA_LARGURA = False).

Registro: acrescentar um backbone = escrever a classe com (d, L, largura) e uma linha em REGISTRO.
O modelo completo (RevIN + backbone + cabeça densa ou NFH) é montado por `construir_modelo`.

Simplificações são ditas em cada docstring. Os URLs foram os informados pelo protocolo/pesquisa;
o autor desta rodada NÃO abriu todos os repositórios (ver LEIAME.md, seção "Fontes").
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "teste_exemplificacao_nfh" / "codigo"))

from v2_common import (BackboneHeadModel, ConvLSTMBackbone, DenseHead, InstanceNorm,  # noqa: E402
                       NFHHead)
from v3_common import H3, K3, L3, Q3  # noqa: E402
from busca import LSTMBackbone, MLPBackbone  # noqa: E402  (os da v3, com fator de largura)


class Linear(nn.Module):
    """Linear: janela achatada, h = vec(x), p = L*d. Com a cabeça densa, o modelo inteiro é o
    modelo linear com RevIN (a referência 'Linear' de Zeng et al., 2023).
    Fonte: Zeng et al., "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023),
    https://github.com/cure-lab/LTSF-Linear. Simplificação: sem canal-independente (a janela de
    todas as variáveis entra junta na cabeça)."""

    def __init__(self, d, L=L3, largura=1):
        super().__init__()
        self.p = L * d
        self.flat = nn.Flatten()

    def forward(self, x):
        return self.flat(x)


class DLinear(nn.Module):
    """DLinear: decomposição da janela por média móvel (janela 25, preenchimento por replicação
    das bordas) em tendência e resto sazonal; h = [vec(tendência), vec(sazonal)], p = 2*L*d.
    Fonte: Zeng et al. (AAAI 2023), https://github.com/cure-lab/LTSF-Linear (arquivo models/DLinear.py).
    Simplificação: a camada linear que o DLinear original aplica a cada componente é a cabeça
    (densa ou NFH) aplicada sobre a concatenação; os pesos por componente ficam unificados numa
    só matriz W (equivalente, pois a soma das duas camadas lineares é uma camada linear sobre a
    concatenação)."""

    def __init__(self, d, L=L3, largura=1, kernel=25):
        super().__init__()
        self.p = 2 * L * d
        self.kernel = kernel
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=0)

    def forward(self, x):
        pad = (self.kernel - 1) // 2
        xt = x.transpose(1, 2)                                   # [B, d, L]
        xp = torch.cat([xt[:, :, :1].repeat(1, 1, pad), xt, xt[:, :, -1:].repeat(1, 1, pad)], 2)
        tend = self.avg(xp)                                      # [B, d, L]
        sazo = xt - tend
        return torch.cat([tend.flatten(1), sazo.flatten(1)], 1)


class MLP(MLPBackbone):
    """MLP da v3 (teste_exemplificacao_nfh/codigo/busca.py): janela achatada -> 2 camadas densas
    de 256*largura com ReLU; p = 256*largura."""

    def __init__(self, d, L=L3, largura=1):
        super().__init__(d, L, p=256 * largura)


class LSTM(LSTMBackbone):
    """LSTM da v3 (busca.py): 2 camadas de 64*largura unidades, última saída oculta -> Linear+ReLU;
    p = 64*largura."""

    def __init__(self, d, L=L3, largura=1):
        super().__init__(d, oculto=64 * largura)


class ConvLSTM(ConvLSTMBackbone):
    """Conv1D+LSTM da v2/v3 (src/v2_common.py): Conv1D(k=7) -> Conv1D(k=5) -> LSTM(2) ->
    Linear+LayerNorm+ReLU; canais/unidades 64*largura, p = 128*largura."""

    def __init__(self, d, L=L3, largura=1):
        super().__init__(d, 128 * largura, 64 * largura)
        self.p = 128 * largura


class _BlocoTCN(nn.Module):
    def __init__(self, c_in, c_out, k, dil, drop):
        super().__init__()
        self.pad = (k - 1) * dil
        self.c1 = nn.Conv1d(c_in, c_out, k, dilation=dil)
        self.c2 = nn.Conv1d(c_out, c_out, k, dilation=dil)
        self.act, self.drop = nn.ReLU(), nn.Dropout(drop)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):                                        # x: [B, c, L]
        y = self.drop(self.act(self.c1(nn.functional.pad(x, (self.pad, 0)))))
        y = self.drop(self.act(self.c2(nn.functional.pad(y, (self.pad, 0)))))
        return self.act(y + self.res(x))


class TCN(nn.Module):
    """TCN: pilha de blocos residuais de duas convoluções causais dilatadas (k = 3, dilatações
    1, 2, 4, 8, 16, preenchimento só à esquerda); campo receptivo = 1 + 2*(k-1)*(1+2+4+8+16) = 125 >= L = 96.
    h = ativação do ÚLTIMO instante, p = 64*largura canais.
    Fonte: Bai, Kolter, Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent
    Networks for Sequence Modeling" (2018), https://github.com/locuslab/TCN.
    Simplificações: sem weight_norm; dropout 0,1; definido pelo implementador (o protocolo não fixa
    canais nem profundidade)."""

    def __init__(self, d, L=L3, largura=1, k=3, dils=(1, 2, 4, 8, 16), drop=0.1):
        super().__init__()
        c = 64 * largura
        self.p = c
        blocos, c_in = [], d
        for dl in dils:
            blocos.append(_BlocoTCN(c_in, c, k, dl, drop))
            c_in = c
        self.net = nn.Sequential(*blocos)

    def forward(self, x):
        return self.net(x.transpose(1, 2))[:, :, -1]


class PatchTST(nn.Module):
    """PatchTST: cada variável vira uma sequência independente (canal-independente) de patches de
    16 passos com stride 8 (preenchimento por replicação de 8 passos no fim, como no original:
    (96 + 8 - 16)/8 + 1 = 12 patches); embedding linear + posição aprendida; codificador Transformer
    de 2 camadas (d_model = 64*largura, 4 cabeças, feed-forward 128*largura, dropout 0,1, pré-norm).
    h = flatten de todos os tokens de todas as variáveis (como a FlattenHead), p = d*12*d_model.
    Fonte: Nie et al., "A Time Series is Worth 64 Words" (ICLR 2023),
    https://github.com/yuqinie98/PatchTST.
    Simplificações: sem RevIN interno (já existe o InstanceNorm do modelo); LayerNorm pré-norm do
    PyTorch no lugar da BatchNorm do original; flatten de todas as variáveis (no original cada
    canal tem a sua cabeça); camadas/dimensões escolhidas pelo implementador."""

    def __init__(self, d, L=L3, largura=1, patch=16, stride=8, camadas=2, cabecas=4, drop=0.1):
        super().__init__()
        self.patch, self.stride = patch, stride
        self.n_patch = (L + stride - patch) // stride + 1
        dm = 64 * largura
        self.p = d * self.n_patch * dm
        self.emb = nn.Linear(patch, dm)
        self.pos = nn.Parameter(torch.randn(1, self.n_patch, dm) * 0.02)
        lay = nn.TransformerEncoderLayer(dm, cabecas, 128 * largura, drop, activation="gelu",
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(lay, camadas, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dm)

    def forward(self, x):
        B, L, d = x.shape
        z = x.transpose(1, 2)                                    # [B, d, L]
        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, self.stride)], 2)
        z = z.unfold(2, self.patch, self.stride)                 # [B, d, n_patch, patch]
        z = self.emb(z).reshape(B * d, self.n_patch, -1) + self.pos
        z = self.norm(self.enc(z))
        return z.reshape(B, -1)


class _MisturaTSMixer(nn.Module):
    def __init__(self, L, d, ff, drop):
        super().__init__()
        self.n1 = nn.LayerNorm([L, d])
        self.tempo = nn.Linear(L, L)
        self.n2 = nn.LayerNorm([L, d])
        self.feat = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(drop), nn.Linear(ff, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x):                                        # x: [B, L, d]
        y = self.n1(x).transpose(1, 2)                           # [B, d, L]
        x = x + self.drop(torch.relu(self.tempo(y))).transpose(1, 2)
        return x + self.drop(self.feat(self.n2(x)))


class TSMixer(nn.Module):
    """TSMixer: 2 blocos; cada bloco mistura no tempo (Linear L->L por variável, ReLU, residual) e
    entre variáveis (MLP d -> 64*largura -> d, residual), com LayerNorm sobre (L, d) antes de cada
    mistura. Depois, projeção temporal L -> 64*largura aplicada a cada variável; h = flatten,
    p = d*64*largura.
    Fonte: Chen et al., "TSMixer: An All-MLP Architecture for Time Series Forecasting" (TMLR 2023),
    https://github.com/google-research/google-research/tree/master/tsmixer.
    Simplificações: LayerNorm no lugar da normalização por [tempo, variável] do original em TF;
    projeção temporal L -> 64*largura no lugar de L -> H (a saída é a cabeça densa/NFH); sem
    variáveis estáticas/futuras (TSMixerx)."""

    def __init__(self, d, L=L3, largura=1, blocos=2, drop=0.1):
        super().__init__()
        tp = 64 * largura
        self.p = d * tp
        self.blocos = nn.Sequential(*[_MisturaTSMixer(L, d, 64 * largura, drop) for _ in range(blocos)])
        self.proj = nn.Linear(L, tp)

    def forward(self, x):
        z = self.blocos(x).transpose(1, 2)                       # [B, d, L]
        return torch.relu(self.proj(z)).flatten(1)


class iTransformer(nn.Module):
    """iTransformer: cada variável (a série inteira de L passos) vira UM token por Linear(L -> d_model);
    codificador Transformer de 2 camadas sobre os d tokens (atenção entre variáveis, sem posição);
    h = token do alvo (última variável), p = d_model = 128*largura (4 cabeças, feed-forward 256*largura,
    dropout 0,1, pré-norm, LayerNorm final).
    Fonte: Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series
    Forecasting" (ICLR 2024), https://github.com/thuml/iTransformer.
    Simplificações: sem a projeção final para H (é a cabeça), sem normalização interna da janela
    (já feita pelo InstanceNorm). No modo U há um só token e a atenção é trivial: o bloco reduz a
    um MLP sobre a janela (registrar ao interpretar os resultados)."""

    def __init__(self, d, L=L3, largura=1, camadas=2, cabecas=4, drop=0.1):
        super().__init__()
        dm = 128 * largura
        self.p = dm
        self.emb = nn.Linear(L, dm)
        lay = nn.TransformerEncoderLayer(dm, cabecas, 256 * largura, drop, activation="gelu",
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(lay, camadas, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dm)

    def forward(self, x):
        z = self.emb(x.transpose(1, 2))                          # [B, d, dm]
        return self.norm(self.enc(z))[:, -1]


class FreqMLP(nn.Module):
    """FreqMLP: MLP sobre o espectro da janela. rfft da janela (já normalizada pelo InstanceNorm, logo o
    bin 0 é zero e é descartado), bins k = 1..12 em parte real e parte imaginária (24*d números; com L = 96,
    k = 4 é o ciclo de 24 h e k = 8 o de 12 h), Linear(24d, w) -> GELU -> Linear(w, w) -> GELU, com
    w = p = 64*largura.
    Fonte: a ideia de prever no domínio da frequência com poucas componentes de baixa frequência vem de
    Xu et al., "FITS: Modeling Time Series with 10k Parameters" (2023), arXiv 2307.03756.
    Simplificações: o FITS aplica uma única camada linear COMPLEXA a um corte de frequências e reconstrói a série
    com a irfft; aqui a camada complexa é trocada por parte real e parte imaginária empilhadas, duas camadas com
    GELU e nenhuma reconstrução (a saída é a cabeça densa ou NFH). Dimensões definidas pelo implementador
    (decisão do coordenador, protocolo §10, item 1). Sem camada de saída própria."""

    def __init__(self, d, L=L3, largura=1, k_max=12):
        super().__init__()
        assert k_max <= L // 2, "k_max acima de L/2"
        self.k_max = k_max
        w = 64 * largura
        self.p = w
        self.net = nn.Sequential(nn.Linear(2 * k_max * d, w), nn.GELU(), nn.Linear(w, w), nn.GELU())

    def forward(self, x):                                        # x: [B, L, d]
        z = torch.fft.rfft(x.float(), dim=1)[:, 1:self.k_max + 1, :]      # [B, k_max, d] complexo
        z = torch.cat([z.real, z.imag], 1).flatten(1)            # [B, 2*k_max*d]
        return self.net(z)


# nome -> (classe, usa_largura). Acrescentar um backbone = uma linha aqui.
REGISTRO = {
    "Linear": (Linear, False),
    "DLinear": (DLinear, False),
    "MLP": (MLP, True),
    "LSTM": (LSTM, True),
    "Conv1D+LSTM": (ConvLSTM, True),
    "TCN": (TCN, True),
    "PatchTST": (PatchTST, True),
    "TSMixer": (TSMixer, True),
    "iTransformer": (iTransformer, True),
    "FreqMLP": (FreqMLP, True),
}


def usa_largura(nome: str) -> bool:
    return REGISTRO[nome][1]


def construir_modelo(backbone: str, cabeca: str, d: int, largura: int = 1,
                     L: int = L3, H: int = H3, K: int = K3, q: int = Q3):
    """Modelo = InstanceNorm(BackboneHeadModel(Phi, cabeça)); cabeça em {'densa', 'nfh'}.
    A NFH usa colunas normalizadas (normalize=True) e os mesmos H*(p+1) parâmetros da densa."""
    if backbone not in REGISTRO:
        raise ValueError(f"Backbone desconhecido: {backbone}. Registrados: {list(REGISTRO)}")
    phi = REGISTRO[backbone][0](d, L, largura)
    if cabeca == "densa":
        head = DenseHead(phi.p, H)
    elif cabeca == "nfh":
        head = NFHHead(phi.p, H, K, q, normalize=True)
    else:
        raise ValueError(cabeca)
    return InstanceNorm(BackboneHeadModel(phi, head), target_idx=-1)
