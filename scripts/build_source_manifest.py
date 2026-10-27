#!/usr/bin/env python3
"""Create/check the reviewer source inventory without including run artifacts."""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("ablation", "analysis", "baseline_study", "common", "configs", "docs", "dynamic_hybrid",
               "followup_study", "hardware_study", "publication_reporting",
               "hybrid", "last-1", "lstm", "online_study", "qwen", "reference", "reliability_cascade",
               "scripts", "tests", "textcnn", "xgboost")
SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".ini", ".slurm", ".sh", ".ps1"}
MANIFEST = ROOT / "SOURCE_SHA256SUMS.txt"


def sources():
    files = [p for p in ROOT.iterdir() if p.is_file() and p != MANIFEST
             and (p.suffix in SUFFIXES or p.name in (".gitignore", "LICENSE"))
             and not p.name.startswith(("PRIVATE_", "INTERNAL_"))]
    for directory in SOURCE_DIRS:
        for base, children, names in os.walk(ROOT / directory):
            # Prune before traversing inherited Windows scratch ACLs.
            children[:] = [name for name in children if not name.startswith((".", "tmp"))
                           and name not in ("__pycache__", "outputs", "checkpoints", "qa_tmp", "test_tmp")]
            files.extend(Path(base) / name for name in names
                         if Path(name).suffix in SUFFIXES and not name.startswith(("PRIVATE_", "INTERNAL_")))
    return sorted(set(files), key=lambda p: p.relative_to(ROOT).as_posix())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true", help="Explicitly replace the source checksum list")
    p.add_argument("--syntax-check", action="store_true", help="Parse all inventoried Python source")
    p.add_argument("--archive", type=Path, help="After verification, create a source-only ZIP; no outputs or private data")
    a = p.parse_args(argv)
    files = sources()
    if a.syntax_check:
        for path in files:
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        print(f"Python syntax OK: {sum(p.suffix == '.py' for p in files)} files")
    expected = "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT).as_posix()}\n"
                       for path in files)
    if a.write:
        MANIFEST.write_text(expected, encoding="utf-8", newline="\n")
        print(f"Wrote {len(files)} source records: {MANIFEST}")
    elif not MANIFEST.is_file() or MANIFEST.read_text(encoding="utf-8") != expected:
        raise ValueError("Source inventory differs; do not silently continue an existing study with changed code")
    else:
        print(f"Source checksum inventory OK: {len(files)} files")
    if a.archive:
        archive = a.archive.resolve()
        archive.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects an existing review/upload archive.
        with ZipFile(archive, "x", compression=ZIP_DEFLATED) as output:
            for path in [*files, MANIFEST]:
                output.write(path, arcname="20261026/" + path.relative_to(ROOT).as_posix())
        print(f"Source-only archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
