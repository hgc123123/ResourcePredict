#!/usr/bin/env python3
"""Record source-copy/adaptation provenance; historical trees are read-only inputs.

This tool is only needed when assembling a new source release, not when running
experiments. The resulting metadata uses relative source labels, not local paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_source_manifest import sources


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main-source", type=Path, required=True)
    p.add_argument("--followup-source", type=Path, required=True)
    p.add_argument("--hardware-source", type=Path, required=True)
    a = p.parse_args(argv)
    source_roots = {"main": a.main_source, "followup_study": a.followup_source, "hardware_study": a.hardware_source}
    if not all(path.is_dir() for path in source_roots.values()):
        p.error("All three historical source directories must exist")
    records = []
    for path in sources():
        relative = path.relative_to(ROOT)
        if relative.as_posix() == "SOURCE_PROVENANCE.json":
            continue
        section = relative.parts[0] if relative.parts[0] in source_roots else "main"
        local = Path(*relative.parts[1:]) if section != "main" else relative
        original = source_roots[section] / local
        current = digest(path)
        record = {"path": relative.as_posix(), "release_sha256": current, "status": "new_release_file"}
        if original.is_file():
            before = digest(original)
            label = {"main": "20261021/reviewer_experiments",
                     "followup_study": "20261021/reviewer_followup_20261023",
                     "hardware_study": "20261021/reviewer_hardware_20261023"}[section]
            record.update(original_path=label + "/" + local.as_posix(), original_sha256=before,
                          status="unchanged_copy" if before == current else "adapted_copy")
        records.append(record)
    output = ROOT / "SOURCE_PROVENANCE.json"
    output.write_text(json.dumps({"schema": "publication-source-provenance/v1",
        "scope": "Source files only; original directories were not mutated; self and checksum inventory omitted",
        "files": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Recorded {len(records)} source files: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
