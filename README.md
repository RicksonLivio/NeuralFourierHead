# Neural Fourier Head (NFH)

**Neural Fourier Head: an identifiable trend–seasonal decomposition layer for interpretable multi-step forecasting**
(submitted to ESANN 2027, under review; not yet accepted).

NFH is an output layer for multi-step forecasting networks. It replaces the last dense layer of any forecaster and
writes each forecast as

$$\hat{\mathbf{y}} = \mathbf{T} + \mathbf{S} + \mathbf{R},$$

a polynomial **trend** $\mathbf{T}$, **seasonal** harmonics $\mathbf{S}$ and a **residual** $\mathbf{R}$, in mutually
orthogonal subspaces. The layer has the same $H(p+1)$ parameters as the dense layer it replaces and is an invertible
linear reparametrization of it, so it can represent the same functions.

This repository holds the `nfh` library (PyTorch), runnable examples, the code, the stored results and the
reproduction scripts of the paper, and of the author's master's dissertation (PPGCC, Instituto Federal do Ceará, IFCE;
in preparation). The README, `REPRODUCE.md`, `DATA.md` and this documentation are in English; most code comments and the
experiment protocols are in Portuguese.

## What it is

Let a forecaster be $f = g \circ \Phi$: a backbone $\Phi$ maps the window to a latent vector $\mathbf{h} \in \mathbb{R}^p$,
and the output layer gives the $H$ forecasts. NFH computes $\mathbf{c} = \mathbf{W}\mathbf{h} + \mathbf{b}$, splits it into
$\theta_T$ (trend coefficients), $\theta_S = (a_1, b_1, \dots, a_K, b_K)$ (harmonics) and $\mathbf{u}$ (residual), and outputs
$\mathbf{T} = \mathbf{P}^\top\theta_T$, $\mathbf{S} = \widetilde{\mathbf{F}}^\top\theta_S$ and $\mathbf{R} = \mathbf{Q}\mathbf{u}$, where
$\mathbf{P}$ is a polynomial basis of degree $q$, $\widetilde{\mathbf{F}}$ the Fourier basis (periods $H/k$,
$k = 1, \dots, K$) with its polynomial part removed, and $\mathbf{Q}$ an orthonormal basis of the rest. The amplitude of
harmonic $k$ is $A_k = (a_k^2 + b_k^2)^{1/2}$. The shares $\varepsilon_T$, $\varepsilon_S$, $\varepsilon_R$ of the energy of
the centered forecast are in $[0, 1]$ and sum to one; $\kappa = 1 - \varepsilon_R$ is the structural coverage.

## Why it is interpretable

- **The decomposition is exact and unique** (Lemma 1 and Theorem 1 of the paper). The spaces are pairwise orthogonal and
  every forecast in $\mathbb{R}^H$ has exactly one representation, so $\mathbf{T}$, $\mathbf{S}$, $\mathbf{R}$ and $A_k$ depend
  only on the forecast. The explanation is part of the computation, and the same fixed linear map also decomposes the
  forecast of an already trained dense model (`NeuralFourierHead.decompose`, "pure decomposition").
- **Contrast with N-BEATS-I** (Proposition 1). Its trend and seasonal bases share the constant vector and, for even $H > p_{\mathrm{nb}} \ge 1$,
  their spans intersect in a subspace of dimension $p_{\mathrm{nb}}$, so the split of a forecast into trend and seasonality is not unique.
  In the stored N-BEATS-I forecasts, the native trend share fell outside $[0, 1]$ in 21% to 34% of the windows
  (`results/v3/numeros_cap5.json`, keys `estabilidade/<dataset>/nativo/fracao_fora_0_1`).
- **Training with the decomposition never worsens the validation error of a trained dense model.** The procedure converts the
  dense layer exactly, adjusts the whole model with a residual penalty (*structural adjustment*) and keeps the epoch with
  the lowest validation objective among those whose validation RMSE is not above the dense model's (*safe selection*).
  This held in all 360 runs (`results/v3/numeros_cap5.json`, keys `variantes/<K>/fracao_val_ok`). On test data it can be
  worse: in 25 to 31 of 120 runs per row it was.
- **No claim of higher accuracy.** Against a dense layer trained for the same 15 extra epochs, NFH had a lower test RMSE in
  13 of 24 dataset–backbone pairs, with a mean relative difference of −0.05% (two-sided Wilcoxon $p = 0.86$) and is non-inferior
  with a 1% margin ($p = 5.5 \times 10^{-4}$), but not with 0.25%
  (`experimento_controle_epocas/resultados/analise.json`, key `nfh_vs_densaE/RMSE`). The value of the layer is the decomposition.
- **How much of the forecast the decomposition covers.** With $K = 10$ harmonics, the median $\kappa$ of the pure decomposition of the
  dense forecasts is 99.51%; with $K = 2$ it is 93.18% (`results/v3/numeros_cap5.json`, keys `variantes/10/kappa_densa_mediana` and
  `variantes/2/kappa_densa_mediana`).
- **Attribution to input variables is a second, optional layer.** Integrated Gradients (or any other method) can be applied
  to a scalar summary of one component: the level $g_T$ (mean of $\mathbf{T}$) or the seasonal energy $g_S$ ($\lVert\mathbf{S}\rVert^2$).
  On a synthetic series with known drivers, it found the driver of the predicted cycle in all 50 test windows and the trend driver in 92%
  (`results/v3/numeros_cap5.json`, keys `sint_m/metodos/NFH | IG sobre g_T e g_S/acerto_ciclo` and `.../acerto_tendencia`). A dense
  model with the same projection did as well, so in this configuration the decomposition, not the parametrization, answers the question.

`docs/PAPER_TO_CODE.md` maps every definition, result and number of the paper to a file and function or to a result key.

## Installation

```bash
git clone https://github.com/RicksonLivio/NeuralFourierHead.git && cd NeuralFourierHead
uv sync                        # Python 3.11; creates .venv and installs `nfh` in editable mode with the pinned versions
# or, with pip:  pip install -e .
```

The dependency versions are pinned to the environment of the experiments (`pyproject.toml`, `uv.lock`), so that the
reproduction is exact. The library alone needs only `torch`, `numpy`, `pandas` and `matplotlib` (`pip install -e experimentation`
installs it with open version ranges, without the study's other dependencies; `captum` is needed only for example 04).

## Use it in 30 seconds

```python
import torch
import torch.nn as nn
from nfh import NeuralFourierHead, energy_balance

L, d, H, p = 96, 4, 48, 128
backbone = nn.Sequential(nn.Flatten(), nn.Linear(L * d, p), nn.ReLU())        # any model that returns h in R^p
head = NeuralFourierHead(in_features=p, horizon=H, n_harmonics=10, degree=2)  # replaces nn.Linear(p, H)

x = torch.randn(8, L, d)                                  # 8 windows
out = head(backbone(x))                                   # NFHOutput
T, S, R = out.trend, out.seasonal, out.residual           # each (8, 48); out.forecast = T + S + R
A_k = out.amplitude                                       # (8, 10): amplitude of each harmonic (period H / k)
eps = energy_balance(out)                                 # eps_T, eps_S, eps_R (sum to one), coverage = kappa
print(out.forecast.shape, A_k.shape, float(eps["eps_T"][0] + eps["eps_S"][0] + eps["eps_R"][0]))

y = torch.randn(8, H)
loss = nn.functional.mse_loss(out.forecast, y) + 1000.0 * head.residual_penalty(out)   # J_lambda of the paper
```

To explain a model that is already trained, swap its last `nn.Linear(p, H)` for an equivalent NFH
(`attach_nfh(model, "out")` or `NeuralFourierHead.from_linear(linear)`): the predictions do not change and every one
comes decomposed.

## Examples

All run on a CPU in well under two minutes (`uv run python examples/<file>`), in the order of the paper:

| Script | What it shows | Paper |
|---|---|---|
| `examples/01_quickstart.py` | Couple the NFH to a GRU backbone; read $\mathbf{T}$, $\mathbf{S}$, $\mathbf{R}$, $A_k$ and the shares; swap the last layer of a dense model | Section 2 |
| `examples/02_numerical_checks.py` | Numerical checks of Lemma 1, Theorem 1 (uniqueness, orthogonality) and Proposition 1 (the N-BEATS-I intersection has dimension $p_{\mathrm{nb}}$) | Section 2 |
| `examples/03_dense_to_nfh_and_adjustment.py` | Exact dense → NFH conversion, structural adjustment and safe selection, with the whole trajectory | Section 2, Training |
| `examples/04_explain_a_forecast.py` | Explain one forecast (components, amplitudes, shares); as a complement, attribution by component with Captum on $g_T$ and $g_S$ | Section 2, Attribution |
| `examples/05_synthetic_known_drivers.py` | A small synthetic series with known drivers (reduced SINT-M); the numbers differ from the paper's | Section 3, Known drivers |

The library also has a longer, step-by-step tutorial in Portuguese in `experimentation/tutorial/` (it includes a run on
UCI Bike Sharing; `experimentation/README.md`).

## Reproduce the paper

`REPRODUCE.md` lists, for each number of the paper, the command that recomputes it from the stored results (Level 1: a
CPU and a few minutes) and the commands that retrain the models (Levels 2 and 3: hours on a GPU).

```bash
uv sync
uv run python data/download_data.py                              # third-party data, checked with SHA-256
uv run pytest                                                    # library tests, paper-claim tests, smoke test of the examples
uv run python src/v3_tabelas.py && git diff --stat results/v3/numeros_cap5.json   # no output = identical
```

## Repository layout

| Path | Contents |
|---|---|
| `experimentation/src/nfh/` | The `nfh` library: bases, head, backbones, structural adjustment, explanations. |
| `examples/` | Five short runnable scripts (above). |
| `tests/`, `experimentation/tests/` | Tests of the library, of the paper's statements, of `docs/PAPER_TO_CODE.md` and a smoke test of the examples. |
| `docs/PAPER_TO_CODE.md` | Paper → code map. |
| `experimentation/` | The library's own `pyproject.toml`, README, Portuguese tutorial and Bike Sharing data. |
| `src/` | Code of the main study (datasets, models, adjustment, attributions, tables: `v2_common.py`, `v3_*.py`). |
| `results/v3/`, `results/v3_piloto/`, `results/v2/config.json` | Stored results of the main study. |
| `experimento_controle_epocas/` | Control with the same extra epochs: protocol (`INSTRUCOES.md`), code, results, report. |
| `experimento_backbones_nfh/` | Backbone sweep: protocol (`INSTRUCOES.md`, with dated amendments), code, results, report, and a hand-calculation example. |
| `teste_exemplificacao_nfh/` | Data loaders and the small synthetic series imported by the sweep. |
| `data/download_data.py` | Downloads the third-party datasets and checks them with SHA-256 (`DATA.md`). |
| `MANIFEST.sha256` | SHA-256 of every file in this release. |

The reproduction folders (`src/`, `results/`, `experimento_*`, `teste_exemplificacao_nfh/`) keep their original names and
paths because the stored results and `REPRODUCE.md` refer to them. Absolute paths of the author's machine inside
documents and JSON results were made relative in this copy. Not included: third-party raw data (downloaded by the
script), trained weights (release archive, see `REPRODUCE.md`), the dissertation and the paper manuscript.

## License

The code and documentation are under the **BSD 3-Clause License** (`LICENSE`), the license used by PyTorch and Captum.
Third-party datasets keep their own licenses (`DATA.md`); the license of the code does not relicense them.

## How to cite

See `CITATION.cff` for the software. If you use the method, please also cite the paper: R. L. de S. Gaspar and
T. A. Rocha, *Neural Fourier Head: an identifiable trend–seasonal decomposition layer for interpretable multi-step
forecasting*, submitted to ESANN 2027 (under review). The full reference will be added only if it is accepted.

## Contributing

See `CONTRIBUTING.md`; changes are listed in `CHANGELOG.md`.

## Contact

Rickson Livio de Souza Gaspar, rickson.gaspar@outlook.com, https://github.com/RicksonLivio, ORCID 0009-0000-1550-6301.
Advisor: Dr. Thiago Alves Rocha (IFCE).
