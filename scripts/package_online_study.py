"""Build an additive upload archive; never include results or model weights."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(root.parent / "online_study_supplement_20261022_v2.zip"))
    args = parser.parse_args()
    files = sorted((root / "online_study").glob("*.py"))
    files.extend(root / name for name in (
        "README_ONLINE_STUDY.md", "docs/ONLINE_STUDY_20261022.md",
        "scripts/run_online_study.slurm", "scripts/package_online_study.py",
    ))
    if not files or any(not p.is_file() for p in files):
        raise FileNotFoundError("Supplement is incomplete")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        sums = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}\n" for p in files)
        archive.writestr("ONLINE_SUPPLEMENT_SHA256SUMS.txt", sums)
    print(f"Created {output} ({output.stat().st_size:,} bytes; {len(files)} source/documentation files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
