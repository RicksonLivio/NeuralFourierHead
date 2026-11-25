# Changelog

## v1.0.0 (2026-11-25)

First public version, matching the paper *Neural Fourier Head: an identifiable trend–seasonal decomposition layer for
interpretable multi-step forecasting* (submitted to ESANN 2027, under review).

Added

- Installable `nfh` library at the repository root (`uv sync` or `pip install -e .`), from `experimentation/src/nfh`
  (dependency versions of the study unchanged in `uv.lock`; only the project entry of the lock changed).
- `examples/`: five runnable scripts in the order of the paper (quickstart; numerical checks of Lemma 1, Theorem 1 and
  Proposition 1; dense → NFH conversion with structural adjustment and safe selection; explaining a forecast with
  Captum as a complement; a small synthetic series with known drivers).
- `docs/PAPER_TO_CODE.md`: map from each definition, result and number of the paper to a file and function or result key.
- Tests: `experimentation/tests/test_paper_claims.py`, `tests/test_paper_to_code.py`, `tests/test_readme_snippet.py`,
  `tests/test_examples_smoke.py`.
- GitHub Actions workflow `.github/workflows/tests.yml` (CPU, `uv`), `CONTRIBUTING.md` and this file.

Changed

- `README.md` reorganized in the order of the paper; numbers cite the stored result file and key.
- Project name in `pyproject.toml`: `neural-fourier-head` (it was `neural-fourier-head-reproduction`, a non-installable project).

Unchanged

- Study code (`src/`, `experimento_*`, `teste_exemplificacao_nfh/`), stored results, `LICENSE`, `DATA.md` and the commands of `REPRODUCE.md`.
