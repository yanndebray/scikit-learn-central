#!/usr/bin/env python3
"""Download NASA C-MAPSS FD001 and convert it to CSV under
`data/use-cases/datasets/`.

C-MAPSS is the turbofan run-to-failure simulator from NASA's Prognostics Center
of Excellence. Embedding it lets the JupyterLite build ship it alongside the
predictive-maintenance notebook so it runs entirely offline.

The script is patterned after `fetch_datasets.py`. Re-run after a refresh:

    pixi run -e datasets fetch-cmapss
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "data" / "use-cases" / "datasets"

CMAPSS_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)

COLUMNS = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i:02d}" for i in range(1, 22)]
)


def _read_cmapss_txt(raw: bytes) -> pd.DataFrame:
    """C-MAPSS files are space-separated with a couple of trailing blanks."""
    return pd.read_csv(
        io.BytesIO(raw),
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        engine="python",
    )


def fetch_cmapss_fd001() -> None:
    DEST.mkdir(parents=True, exist_ok=True)

    print(f"→ downloading {CMAPSS_URL}")
    outer = zipfile.ZipFile(io.BytesIO(urlopen(CMAPSS_URL).read()))
    inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))

    train = _read_cmapss_txt(inner.read("train_FD001.txt"))
    test = _read_cmapss_txt(inner.read("test_FD001.txt"))
    rul = pd.read_csv(
        io.BytesIO(inner.read("RUL_FD001.txt")),
        sep=r"\s+",
        header=None,
        names=["rul"],
        engine="python",
    )
    rul.insert(0, "unit", range(1, len(rul) + 1))

    train_out = DEST / "cmapss_fd001_train.csv"
    test_out = DEST / "cmapss_fd001_test.csv"
    rul_out = DEST / "cmapss_fd001_rul.csv"

    train.to_csv(train_out, index=False)
    test.to_csv(test_out, index=False)
    rul.to_csv(rul_out, index=False)

    for label, df, path in [
        ("train", train, train_out),
        ("test ", test, test_out),
        ("rul  ", rul, rul_out),
    ]:
        rel = path.relative_to(REPO_ROOT)
        print(f"  → {label} {df.shape}  → {rel}")


def main() -> int:
    print(f"→ writing C-MAPSS FD001 CSVs to {DEST.relative_to(REPO_ROOT)}/")
    fetch_cmapss_fd001()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
