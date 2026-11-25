"""Checks docs/PAPER_TO_CODE.md: every file exists, every `file::symbol` appears in the file, every JSON key path resolves.

Key paths are written `file.json#a/b/c`; a bare `#a/b/c` refers to the last JSON file named before it in the document.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "PAPER_TO_CODE.md").read_text(encoding="utf-8")
SPAN = re.compile(r"`([^`]+)`")
PATHLIKE = re.compile(r"^(experimentation|src|results|examples|experimento_[a-z_]+|tests|data|docs)/[\w./+-]+(::[\w]+)?(#.*)?$")


def _walk(obj, path):
    for key in path.split("/"):
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
        else:
            raise KeyError(path)
    return obj


def _tokens():
    last_json = None
    for line in DOC.splitlines():
        for span in SPAN.findall(line):
            if span.startswith("#"):
                yield last_json, None, span[1:]
            elif PATHLIKE.match(span):
                path = span.split("#")[0].split("::")[0]
                sym = span.split("::")[1].split("#")[0] if "::" in span else None
                key = span.split("#", 1)[1] if "#" in span else None
                if path.endswith(".json"):
                    last_json = path
                yield path, sym, key


def test_doc_has_content():
    assert len(list(_tokens())) > 150


def test_every_reference_resolves():
    bad = []
    for path, sym, key in _tokens():
        f = ROOT / path
        if not f.exists():
            bad.append(f"missing file {path}")
            continue
        if sym and sym not in f.read_text(encoding="utf-8"):
            bad.append(f"{path}: symbol {sym} not found")
        if key:
            try:
                _walk(json.loads(f.read_text(encoding="utf-8")), key)
            except KeyError:
                bad.append(f"{path}#{key}: key path not found")
    assert not bad, "\n".join(bad)
