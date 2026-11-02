"""Package online-benchmark sources for use with a ResourcePredict checkout.

This source subset requires the complete project's dependencies and shared
modules. It excludes prediction results, private inputs, and model weights.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(root / "resourcepredict_online_source.zip"))
    args = parser.parse_args()
    files = sorted((root / "online_study").glob("*.py"))
    files.extend(root / name for name in (
        "README.md", "docs/EXPERIMENTS.md", "docs/DATA_AND_PROTOCOL.md",
        "scripts/run_online.slurm", "scripts/slurm_env.sh", "scripts/package_online_study.py",
    ))
    if not files or any(not p.is_file() for p in files):
        raise FileNotFoundError("Online-benchmark source files are incomplete")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        sums = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}\n" for p in files)
        archive.writestr("ONLINE_SOURCE_SHA256SUMS.txt", sums)
    print(f"Created {output} ({output.stat().st_size:,} bytes; {len(files)} source/documentation files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
