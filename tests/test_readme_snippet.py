"""The code block under "Use it in 30 seconds" in README.md runs and prints what it claims."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_snippet_runs():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text.split("## Use it in 30 seconds", 1)[1].split("\n## ", 1)[0]
    code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[:2] == ["torch.Size([8,", "48])"], r.stdout
