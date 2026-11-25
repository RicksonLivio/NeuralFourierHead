"""
dados_bb.py: carregador que lê os DOIS catálogos (o do teste de exemplificação e o desta pasta).

Reaproveita `carregar` e `segmentos` de teste_exemplificacao_nfh/codigo/dados.py sem alterá-los: troca,
em tempo de execução, a função que lê o catálogo por uma que devolve a união dos dois, com o campo
`arquivo` já absoluto (Path / absoluto = absoluto).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "teste_exemplificacao_nfh" / "codigo"))

import dados as _dados  # noqa: E402  (teste_exemplificacao_nfh/codigo/dados.py)

CATALOGOS = [ROOT / "teste_exemplificacao_nfh" / "dados" / "catalogo.json",
             ROOT / "experimento_backbones_nfh" / "dados" / "catalogo.json"]


def ler_catalogo_unido():
    uniao = []
    for cam in CATALOGOS:
        if not cam.exists():
            continue
        for e in json.loads(cam.read_text(encoding="utf-8")):
            e = dict(e)
            e["arquivo"] = str((cam.parent / e["arquivo"]).resolve())
            e["_catalogo"] = str(cam)
            uniao.append(e)
    nomes = [e["nome"] for e in uniao]
    assert len(nomes) == len(set(nomes)), "nome repetido entre os catálogos"
    return uniao


_dados._ler_catalogo = ler_catalogo_unido


def listar():
    """Conjuntos 'Nome-U' / 'Nome-M' disponíveis (respeita o campo opcional `modos`)."""
    out = []
    for e in ler_catalogo_unido():
        modos = e.get("modos") or (["U", "M"] if e.get("covariaveis") else ["U"])
        out += [f"{e['nome']}-{m}" for m in modos]
    return out


def carregar_bb(nome: str):
    """'Nome-U' ou 'Nome-M' (o nome pode conter hífen: 'SINT-U48-U')."""
    return _dados.carregar(nome)


def segmentos_bb(ds):
    return _dados.segmentos(ds)


if __name__ == "__main__":
    print("\n".join(listar()))
