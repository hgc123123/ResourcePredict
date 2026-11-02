#!/usr/bin/env python3
"""Record source origins and SHA-256 identities for a source release.

Create an inventory from source collections, or refresh release hashes from a
previous inventory. Source collections remain read-only and their local paths
are not written to the inventory.
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
    p.add_argument("--main-source", type=Path)
    p.add_argument("--followup-source", type=Path)
    p.add_argument("--hardware-source", type=Path)
    p.add_argument("--baseline-provenance", type=Path,
                   help="Refresh release hashes while preserving recorded source identities")
    a = p.parse_args(argv)
    source_roots = {"main": a.main_source, "followup_study": a.followup_source, "hardware_study": a.hardware_source}
    if a.baseline_provenance:
        if any(source_roots.values()):
            p.error("Use either --baseline-provenance or all three source directories")
        previous = json.loads(a.baseline_provenance.read_text(encoding="utf-8"))
        if previous.get("schema") != "source-provenance/v2":
            p.error("The baseline must use source-provenance/v2")
        origins = {item["path"]: item for item in previous["files"]}
    else:
        if not all(path and path.is_dir() for path in source_roots.values()):
            p.error("Supply all three source directories or --baseline-provenance")
        origins = None
    records = []
    for path in sources():
        relative = path.relative_to(ROOT)
        if relative.as_posix() == "SOURCE_PROVENANCE.json":
            continue
        section = relative.parts[0] if relative.parts[0] in source_roots else "main"
        local = Path(*relative.parts[1:]) if section != "main" else relative
        current = digest(path)
        record = {"path": relative.as_posix(), "release_sha256": current, "status": "new_release_file"}
        if origins is not None:
            origin = origins.get(relative.as_posix(), {})
            before = origin.get("original_sha256")
            label = origin.get("source_collection")
            if before is not None and not label:
                p.error(f"Missing source collection: {relative.as_posix()}")
        else:
            original = source_roots[section] / local
            before = digest(original) if original.is_file() else None
            label = {"main": "prediction-experiments",
                     "followup_study": "routing-calibration",
                     "hardware_study": "hardware-analysis"}[section]
        if before is not None:
            record.update(source_collection=label, original_sha256=before,
                          status="unchanged_copy" if before == current else "adapted_copy")
        records.append(record)
    output = ROOT / "SOURCE_PROVENANCE.json"
    output.write_text(json.dumps({"schema": "source-provenance/v2",
        "scope": "Source files with original digests and neutral collection identifiers; self and root checksum inventory omitted",
        "files": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Recorded {len(records)} source files: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
