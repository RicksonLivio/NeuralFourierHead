"""Smoke test: every script in examples/ runs on CPU, exits with status 0 and finishes within two minutes."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("0*.py"))


def test_examples_are_present():
    assert [p.name[:2] for p in EXAMPLES] == ["01", "02", "03", "04", "05"]


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(script):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "0"}
    r = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"{script.name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    assert "[FAIL]" not in r.stdout
