#!/usr/bin/env python3
"""Download the third-party datasets used by the paper and the dissertation, and check them with SHA-256.

The repository does not redistribute third-party data (see DATA.md for sources and licenses). This script fetches
them from the original sources into the paths the code expects:

  stage 1 (main study, `src/v3_*.py`): raw files, checked byte for byte
      data/controle/ETTh1.csv, data/v3_bruto/ETTh2.csv, data/v3_bruto/jena_climate_2009_2016.csv.zip,
      data/v3_bruto/air+quality.zip
  stage 2 (backbone sweep, `experimento_backbones_nfh/`): runs the two processing scripts of the repository,
      keeping the shipped Melbourne file and the two catalogs (see SHIPPED below),
      teste_exemplificacao_nfh/dados/baixar_dados.py and experimento_backbones_nfh/dados/baixar_dados.py, and compares
      the processed CSV files with the SHA-256 recorded when the experiments were run (22/11/2026).

A mismatch in stage 1 is an error. A mismatch in stage 2 is reported as a warning: some sources (for example the City
of Melbourne API) can change after the experiments; the comparison tells you whether you are using the same data.

Usage:  uv run python data/download_data.py [--stage 1|2|all]
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STAGE1 = [
    ("data/controle/ETTh1.csv", "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
     "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"),
    ("data/v3_bruto/ETTh2.csv", "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
     "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b"),
    ("data/v3_bruto/jena_climate_2009_2016.csv.zip",
     "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
     "63d757501e92284a7de7cdbef0337f03b24e13ead7ac2b5b8c86a18d8e38ba5b"),
    ("data/v3_bruto/air+quality.zip", "https://archive.ics.uci.edu/static/public/360/air+quality.zip",
     "d4a64013fb385288a8a48d9d193ca7079b2e1bbddf6f8d458feb8c08ab2b8a2a"),
]

STAGE2_SCRIPTS = ["teste_exemplificacao_nfh/dados/baixar_dados.py", "experimento_backbones_nfh/dados/baixar_dados.py"]
STAGE2_EXPECTED = {
    "teste_exemplificacao_nfh/dados/ETTh1.csv": "6f2ab243c71a0b700f5c2b7db09155b0bdcde1d680f09fd46f2ef1e270dfd303",
    "teste_exemplificacao_nfh/dados/ETTh2.csv": "ad51d097c538410648fdb03264c74f51c646d08fa9cd67d633f20f082c1c7214",
    "teste_exemplificacao_nfh/dados/air_quality.csv": "461d98cc3bac28d94ab99c241af0e5291ac5de55a2c7d11bd35791be01e6b3f7",
    "teste_exemplificacao_nfh/dados/bike_sharing_dc.csv": "2ce7ec5cc69ad4d6bb48bcb7eec0039f7a7d930e141a0d78b6b986ce0da0e8bf",
    "teste_exemplificacao_nfh/dados/household_power.csv": "b69275a9b3ce75256d277fbcd8d983ca1629543b9f5e2fb878e22f553d87f6fa",
    "teste_exemplificacao_nfh/dados/jena_climate.csv": "f0e2a541b25a6e3fc69a6c392d7a48b1fd34a8b5bc02a71ac26b4ed90b12f702",
    "teste_exemplificacao_nfh/dados/seoul_bike.csv": "1b220c3822cd06e2a07b41f894435e8495909ec0da8b46de73348398f0445ccb",
    "experimento_backbones_nfh/dados/electricity_mt156.csv": "4821f5f0ebd88f123b146b422e5bca0d53bf84812a97694773609fd64f7b9b36",
    "experimento_backbones_nfh/dados/melbourne_pedestrian_southbank.csv":
        "194f38c9b76565ab71febbe5b29ac6cb1e41e3ff9db00f46440ce2e8ce25903d",
    "experimento_backbones_nfh/dados/opsd_de_solar.csv": "47a2c43124e366f58ace06746a8234d58e2a4c595215462f02ac643064cb8be7",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stage1() -> bool:
    ok = True
    for rel, url, expected in STAGE1:
        dest = ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not (dest.exists() and sha256(dest) == expected):
            print(f"downloading {url}", flush=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as fh:  # noqa: S310 (fixed https URLs)
                fh.write(r.read())
            tmp.replace(dest)
        got = sha256(dest)
        status = "OK" if got == expected else "MISMATCH"
        ok &= got == expected
        print(f"[stage 1] {status} {rel} sha256={got}")
    return ok


# Versioned files that the stage-2 scripts rewrite. The Melbourne file (CC BY 4.0, City of Melbourne) is shipped
# because the public API keeps only about two years, so the file used in the experiments cannot be downloaded again;
# the two catalogs record the metadata of the data actually used. Stage 2 restores them and saves the freshly written
# version next to each one (`*.api_current.*`), for comparison only.
SHIPPED = ["experimento_backbones_nfh/dados/melbourne_pedestrian_southbank.csv",
           "experimento_backbones_nfh/dados/catalogo.json",
           "teste_exemplificacao_nfh/dados/catalogo.json"]


def stage2() -> bool:
    keep = {rel: (ROOT / rel).read_bytes() for rel in SHIPPED if (ROOT / rel).exists()}
    try:
        for s in STAGE2_SCRIPTS:
            print(f"[stage 2] running {s}", flush=True)
            subprocess.run([sys.executable, str(ROOT / s)], cwd=(ROOT / s).parent, check=True)
    finally:
        for rel, data in keep.items():
            p = ROOT / rel
            if p.exists() and p.read_bytes() != data:
                fresh = p.with_name(p.stem + ".api_current" + p.suffix)
                p.replace(fresh)
                print(f"[stage 2] kept the shipped {rel}; the new version is in {fresh.relative_to(ROOT)} (not used)")
            p.write_bytes(data)
    same = True
    for rel, expected in STAGE2_EXPECTED.items():
        p = ROOT / rel
        got = sha256(p) if p.exists() else "missing"
        if got != expected:
            same = False
        print(f"[stage 2] {'OK' if got == expected else 'WARNING: differs from the data used in the experiments'} {rel}")
    return same


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["1", "2", "all"], default="all")
    a = ap.parse_args()
    ok1 = stage1() if a.stage in ("1", "all") else True
    if not ok1:
        sys.exit("stage 1 failed: a raw file differs from the one used in the experiments")
    if a.stage in ("2", "all"):
        same = stage2()
        print("stage 2:", "identical to the experiments" if same else "some files differ (see warnings above)")


if __name__ == "__main__":
    main()
