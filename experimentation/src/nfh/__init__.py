"""
nfh — Neural Fourier Head para PyTorch.

Camada de saída interpretável para previsão multi-passo: substitui a camada
densa final de qualquer modelo e entrega, junto com cada previsão, sua
decomposição exata e única em tendência, sazonalidade e resíduo.

Versão 3 da dissertação: a NFH é obtida de um modelo denso treinado por conversão
exata + ajuste estrutural com seleção segura (``Forecaster.to_nfh`` e
``structural_adjustment``), e explicações abdutivas garantem formalmente o sinal da
tendência e o ciclo previstos (``abductive_explanations``).

Autor: Rickson Livio de Souza Gaspar (IFCE, PPGCC) — orientação: Dr. Thiago Alves Rocha.
Licença: BSD 3-Clause (ver LICENSE).

Exemplo::

    import torch
    from nfh import NeuralFourierHead, energy_balance

    head = NeuralFourierHead(in_features=128, horizon=24, n_harmonics=6, degree=1)
    out = head(torch.randn(8, 128))
    out.forecast.shape            # (8, 24)
    energy_balance(out)["eps_S"]  # participação da sazonalidade na forma de cada previsão
"""
from .bases import NFHBases, build_bases, check_dimensions, max_harmonics
from .components import NFHOutput
from .explain import Explanation, energy_balance, explain, harmonic_shares, integrated_gradients, narrative, robustness
from .head import NeuralFourierHead, attach_nfh
from .model import BACKBONES, ConvLSTMBackbone, Forecaster, LSTMBackbone, MLPBackbone, TransformerBackbone
from .adjust import AdjustConfig, AdjustResult, structural_adjustment
from .abductive import abductive_explanations, latent_box

__version__ = "2.0.0"   # versão 3 da dissertação: conversão + ajuste estrutural, explicações abdutivas
__author__ = "Rickson Livio de Souza Gaspar"

__all__ = [
    "NeuralFourierHead", "attach_nfh", "NFHOutput", "NFHBases", "build_bases", "check_dimensions",
    "max_harmonics", "Explanation", "explain", "energy_balance", "harmonic_shares", "integrated_gradients",
    "robustness", "narrative", "Forecaster", "LSTMBackbone", "TransformerBackbone", "ConvLSTMBackbone",
    "MLPBackbone", "BACKBONES", "AdjustConfig", "AdjustResult", "structural_adjustment",
    "abductive_explanations", "latent_box",
]
