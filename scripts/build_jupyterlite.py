#!/usr/bin/env python3
"""Build a JupyterLite distribution that ships every use-case notebook.

Pipeline:

  1. Convert each ``data/use-cases/*.py`` to ``.ipynb`` via jupytext.
  2. Mirror ``data/use-cases/datasets/`` next to the notebooks so relative
     reads like ``pd.read_csv("datasets/foo.csv")`` resolve in Pyodide.
  3. Pin the Pyodide runtime the in-browser kernel loads (see
     ``PYODIDE_VERSION``) via a generated ``jupyter-lite.json``.
  4. Run ``jupyter lite build`` with those notebooks as content into the
     requested output directory.

Notebooks are emitted as-is from the ``.py`` source. Any per-notebook setup
(``%pip install``, kernel patches, etc.) should be authored directly in the
``.py`` file using jupytext cell markers — this script no longer injects a
synthetic setup cell.

Cross-platform replacement for the original bash script. Driven by pixi
under the ``jupyterlite`` environment.

Usage
-----
  pixi run -e jupyterlite build-jupyterlite                 # → dist/jupyterlite/
  pixi run -e jupyterlite build-jupyterlite --output-dir X  # → X/
  pixi run -e jupyterlite build-jupyterlite --pyodide-version 0.28.3
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Pyodide runtime the in-browser kernel loads. jupyterlite-pyodide-kernel 0.4.x
# defaults to Pyodide 0.27.0, which pre-installs packaging==23.2 — and micropip
# refuses to *upgrade* an already-installed package, so every notebook whose
# `%pip install` pulls something needing `packaging>=24.2` (skore does) dies with
#   ValueError: Requested 'packaging>=24.2', but packaging==23.2 is already installed
# Pyodide 0.27.7 ships packaging==24.2 and is the last 0.27.x, i.e. the same
# Python 3.12 / wasm ABI the pinned kernel was built against.
PYODIDE_VERSION = "0.27.7"

# Settings key the pyodide kernel extension reads its runtime URL from.
PYODIDE_KERNEL_PLUGIN_ID = "@jupyterlite/pyodide-kernel-extension:kernel"


def _write_lite_config(lite_dir: Path, pyodide_version: str) -> Path:
    """Write a ``jupyter-lite.json`` that overrides the kernel's Pyodide URL.

    ``jupyter lite build`` merges any ``jupyter-lite.json`` found under
    ``--lite-dir`` into the generated one, so this only overrides `pyodideUrl`
    and leaves everything else (federated extensions, piplite wheel index, …)
    to the build.
    """
    lite_dir.mkdir(parents=True, exist_ok=True)
    config = lite_dir / "jupyter-lite.json"
    url = f"https://cdn.jsdelivr.net/pyodide/v{pyodide_version}/full/pyodide.js"
    config.write_text(
        json.dumps(
            {
                "jupyter-config-data": {
                    "litePluginSettings": {
                        PYODIDE_KERNEL_PLUGIN_ID: {"pyodideUrl": url}
                    }
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  • pyodide {pyodide_version} → {url}")
    return config


def _convert_use_cases(src_dir: Path, dest_dir: Path) -> list[Path]:
    """jupytext-convert every .py in src_dir to .ipynb in dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for py in sorted(src_dir.glob("*.py")):
        ipynb = dest_dir / f"{py.stem}.ipynb"
        print(f"  • {py.name} → {ipynb.relative_to(dest_dir.parent)}")
        subprocess.run(
            ["jupytext", "--to", "ipynb", "--output", str(ipynb), str(py)],
            check=True,
        )
        out.append(ipynb)
    return out


def _copy_datasets(src_dir: Path, dest_dir: Path) -> int:
    """Mirror data/use-cases/datasets/ next to the generated notebooks so
    relative paths like 'datasets/california_housing.csv' resolve in
    JupyterLite. Returns the number of files copied (0 if no datasets/)."""
    src_datasets = src_dir / "datasets"
    if not src_datasets.is_dir():
        return 0
    dest_datasets = dest_dir / "datasets"
    dest_datasets.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src_datasets.iterdir()):
        if f.is_file():
            shutil.copy2(f, dest_datasets / f.name)
            print(f"  • datasets/{f.name}")
            n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--use-cases",
        type=Path,
        default=Path("data/use-cases"),
        help="Directory of .py use-case sources (default: data/use-cases)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/jupyterlite"),
        help="Where to write the built JupyterLite site (default: dist/jupyterlite)",
    )
    parser.add_argument(
        "--pyodide-version",
        default=PYODIDE_VERSION,
        help=(
            "Pyodide runtime the in-browser kernel loads "
            f"(default: {PYODIDE_VERSION}; must ship packaging>=24.2 for skore)"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    src_dir = (repo_root / args.use_cases).resolve()
    out_dir = (repo_root / args.output_dir).resolve()

    if not src_dir.is_dir():
        print(f"✗ Source directory not found: {src_dir}", file=sys.stderr)
        return 1

    print(f"→ Converting use-cases from {src_dir.relative_to(repo_root)}")
    with tempfile.TemporaryDirectory() as tmp:
        contents_root = Path(tmp) / "files"
        notebooks_dir = contents_root / "use-cases"
        notebooks = _convert_use_cases(src_dir, notebooks_dir)
        print(f"  ✓ converted {len(notebooks)} notebook(s)")

        n_datasets = _copy_datasets(src_dir, notebooks_dir)
        if n_datasets:
            print(f"  ✓ embedded {n_datasets} dataset file(s)")

        # Sibling of `files/`, not its parent: anything under --lite-dir named
        # jupyter-lite.json is merged, and `files/` there would be picked up a
        # second time as content.
        lite_dir = Path(tmp) / "lite"
        _write_lite_config(lite_dir, args.pyodide_version)

        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"→ Building JupyterLite into {out_dir.relative_to(repo_root)}")
        subprocess.run(
            [
                "jupyter", "lite", "build",
                "--contents", str(contents_root),
                "--lite-dir", str(lite_dir),
                "--output-dir", str(out_dir),
            ],
            check=True,
        )

    print(f"✓ JupyterLite ready at {out_dir.relative_to(repo_root)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
