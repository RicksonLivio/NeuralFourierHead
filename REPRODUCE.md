# Reproducing the results

There are three levels. Level 1 needs only a CPU and a few minutes; it recomputes every reported number from the
stored per-run results. Levels 2 and 3 retrain models.

Environment used for all experiments: Python 3.11.11, the versions pinned in `pyproject.toml` (`uv.lock`), PyTorch
2.12.1 (CUDA 13.0 build), an NVIDIA GeForce RTX 3070 Laptop GPU (8 GB), an Intel Core i7-11800H, 16 GB of RAM, Linux
under WSL2. CPU and GPU runs with the same seed do not give identical numbers; `results/v3/dispositivos.json` records
which device each run of the main study used (46 CPU and 114 GPU runs), and the control and the sweep record the device
per row.

## Level 0: integrity

```bash
sha256sum -c MANIFEST.sha256 --quiet && echo "files as released"
uv sync
uv run python data/download_data.py
```

`download_data.py` stage 1 must report `OK` for the four raw files of the main study (byte-identical to the files used
in the experiments). Stage 2 rebuilds the processed files of the sweep and compares them with the SHA-256 recorded on
22/11/2026. The City of Melbourne file is shipped with the repository, because the public API keeps only about two
years of data; the script keeps the shipped file and saves the current API version as `*.api_current.csv`.

## Level 1: recompute the reported numbers from stored results (CPU, minutes)

Each command below was run on a clean copy of this repository and gave files identical to the stored ones (apart from
absolute paths recorded inside some JSON files).

| What | Command | Compare with |
|---|---|---|
| Main study: all numbers of the paper's Section 3 and the dissertation's Chapter 5 | `uv run python src/v3_tabelas.py` | `results/v3/numeros_cap5.json` (the command rewrites it; compare with `git diff`). It also writes LaTeX tables under `versao3_correcao2_aplicada/` (not part of this repository; ignore them). |
| Control with the same extra epochs | `mkdir -p /tmp/ctl && cp experimento_controle_epocas/resultados/{controle,exata_s0}.csv /tmp/ctl/ && uv run python experimento_controle_epocas/codigo/analisar.py --dir /tmp/ctl --fila experimento_controle_epocas/resultados/filas/controle.json` | `experimento_controle_epocas/resultados/analise.json` (all keys except `fonte`) and `relatorio/relatorio_controle.md` |
| Sweep, selection T1 → T2 (reads validation columns only) | `uv run python experimento_backbones_nfh/codigo/selecionar.py t1` | `experimento_backbones_nfh/resultados/filas/t2.json`, `selecao_T1.md` (`git diff`) |
| Sweep, selection T2 → confirmation | `uv run python experimento_backbones_nfh/codigo/selecionar.py t2` | `filas/confirmacao.json`, `selecao_T2.md` |
| Sweep report and all its numbers | `uv run python experimento_backbones_nfh/codigo/relatorio.py` | `experimento_backbones_nfh/relatorio/numeros.json` (all keys except absolute paths) and `relatorio.tex` (`git diff`). The PDF is rebuilt from the `.tex` by `tectonic` at `~/.cache/v2work/bin/tectonic` (without it, the `.json` and the `.tex` are still written); its bytes change at every build, so compare the `.tex`. |

Unit tests (CPU):

```bash
uv run pytest                                # library tests, paper-claim tests, docs/PAPER_TO_CODE.md check, README snippet, smoke test of examples/
(cd experimentation && uv run --project .. python -m pytest -q)        # the library's own tests only (the Level 1 command of v1.0.0)
for f in examples/0*.py; do uv run python "$f"; done                   # the five examples, each under two minutes on a CPU
uv run python experimento_controle_epocas/codigo/test_analisar.py
uv run python experimento_controle_epocas/codigo/test_retomada.py
uv run python experimento_backbones_nfh/codigo/test_selecionar.py
uv run python experimento_backbones_nfh/codigo/test_falha_variante.py
uv run python src/v3_teste_rapido.py        # loads the 8 datasets and checks the exact dense -> NFH conversion
```

## Level 2: smoke runs of the training code (CPU, 2 epochs, minutes)

These runs only test that the training pipelines work. Their numbers are not used anywhere.

```bash
export CUDA_VISIBLE_DEVICES=''
uv run python experimento_controle_epocas/codigo/filas.py smoke
TESTE_MAX_EPOCAS=2 TESTE_E_EXTRA=2 uv run python experimento_controle_epocas/codigo/executar.py \
  --fase smoke_cpu --fila experimento_controle_epocas/resultados/smoke/smoke_cpu/fila.json --worker c0 --device cpu
uv run python experimento_backbones_nfh/codigo/filas.py smoke
TESTE_MAX_EPOCAS=2 TESTE_E_EXTRA=2 uv run python experimento_backbones_nfh/codigo/executar.py \
  --fase smoke_cpu --fila experimento_backbones_nfh/resultados/filas/smoke.json --device cpu
```

The control smoke run reports an error in the `exata_s0` variant unless the trained weights of the main study are in
`results/v3/modelos/` (release archive, below). That variant loads the dissertation's dense models.

## Level 3: full retraining (GPU hours)

Recorded worker time (sum over parallel workers; it includes time paused by a temperature watchdog):

| Experiment | Entry point | Recorded time |
|---|---|---|
| Main study, 8 datasets × 3 backbones × 5 seeds + N-BEATS-I | `src/v3_benchmark.py <dataset>...` (see `src/v3_rodar_benchmark.sh`) | about 22 h in `results/v3/runs/*.csv` (`tempo_s`, includes rows superseded by resumption) |
| Control with the same extra epochs | `experimento_controle_epocas/LEIAME.md` (`lanca.sh`, `status.py`, `analisar.py`) | 10.6 h (`analise.json`, key `tempo`) |
| Backbone sweep T1, T2, confirmation | `experimento_backbones_nfh/codigo/LEIAME.md` (`lanca.sh`, `orquestrar.sh`, `selecionar.py`) | about 61 h (task logs of the author's run; the logs are not in this repository) |

The protocols are in Portuguese: `experimento_controle_epocas/INSTRUCOES.md` and `experimento_backbones_nfh/INSTRUCOES.md`.
Every change made after a protocol was written is dated in its change log (§8 of the control, §10 of the sweep),
including a resumption after a machine shutdown on 21/11/2026 that changed only the allocation of workers.

Retrained numbers will differ slightly from the stored ones. In the control, retraining the dense models of the main
study with the same seeds and devices reproduced 67 of 120 test RMSE values to within 10⁻⁶ (`analise.json`, key
`reproducao`).

## Trained weights

Weights are not in git. The archive `nfh_weights.tar.gz` (296 files, 227 MiB) is published as a release asset of this
repository, with its SHA-256 in the release notes. Extract it at the repository root. Contents:

- `results/v3/modelos/` (56 files): the seed-0 models of the main study for the 8 datasets, one per model: MLP, LSTM
  and Conv1D+LSTM with a dense layer and with NFH, and N-BEATS-I. The control's `exata_s0` variant loads the dense ones.
- `experimento_backbones_nfh/resultados/modelos/` (240 files): for each of the 8 confirmed combinations and each of the
  seeds 10 to 14, the state dicts (`.pt`) of the dense model, the dense model with extra epochs and the NFH variant, and
  the test predictions (`*_pred.npy`) of the dense model with extra epochs, of the NFH variant and of the projected
  dense model (there is no prediction file for the plain dense model).

Each model was trained on the device recorded for its run (CPU or GPU); loading on another device works, but
retraining there gives slightly different numbers.
