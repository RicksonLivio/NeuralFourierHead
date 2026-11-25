# Contributing

Issues and pull requests are welcome. Contributions are accepted under the license of the repository (`LICENSE`, BSD 3-Clause).

- **Scope.** The `nfh` library (`experimentation/src/nfh/`) and the examples (`examples/`) can change. The reproduction
  folders (`src/`, `results/`, `experimento_*`, `teste_exemplificacao_nfh/`) are the record of what produced the paper's
  numbers: please do not edit stored results or move these folders; propose fixes in an issue first.
- **Environment.** `uv sync` (Python 3.11, versions pinned in `uv.lock`). Do not change pinned versions in a pull request
  that is about something else.
- **Tests.** `uv run pytest` must pass on a CPU. It runs the library tests, the tests that tie the code to the paper
  (`experimentation/tests/test_paper_claims.py`), a check of `docs/PAPER_TO_CODE.md` and a smoke test that runs every
  script in `examples/`. A new feature needs a test; a new example must finish in under two minutes on a CPU.
- **Paper map.** If you move or rename a function that `docs/PAPER_TO_CODE.md` cites, update the map (the test reports
  the broken line).
- **Numbers.** Any number in a README or document needs the file and key it comes from.
- **Style.** Notation of the paper ($\mathbf{T}$, $\mathbf{S}$, $\mathbf{R}$, $\theta_T$, $\theta_S$, $\mathbf{u}$, $A_k$, $\varepsilon$, $\kappa$).
  New documentation in English; the Portuguese protocols and comments stay as they are.
- **Do not add** third-party data, trained weights, the dissertation or manuscript files, or personal data.

Contact: rickson.gaspar@outlook.com.
