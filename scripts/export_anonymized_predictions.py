#!/usr/bin/env python3
"""Stage de-identified outputs locally for institutional release review.

Does not upload/publish. HMAC keys and private source assets stay outside the
export directory; identifiers are stable across methods/seeds with one key.
"""
import argparse
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import sys

import pandas as pd

PUBLIC_COLUMNS = ["resource", "split", "target", "valid", "prediction", "gpu_count",
                  "qwen_prediction", "last1_prediction", "hierarchical_reference_prediction",
                  "history_expert_prediction", "history_available", "exact_script_history_available",
                  "same_last1", "resource_applicable", "reliability_probability", "history_bypass",
                  "post_history_weight", "post_blend_prediction", "embedding_cache_hit_job",
                  "encoder_forward_required_job"]


def deidentify(frame: pd.DataFrame, key: bytes, anchor: pd.Timestamp) -> pd.DataFrame:
    if len(key) < 32:
        raise ValueError("Release HMAC key must contain at least 32 bytes")
    def token(kind, value):
        return hmac.new(key, (kind+":"+str(value)).encode(), hashlib.sha256).hexdigest()
    out = frame[[name for name in PUBLIC_COLUMNS if name in frame]].copy()
    for column in ("sample_id", "user_id", "script_sha256"):
        if column in frame:
            if frame[column].isna().any():
                raise ValueError(f"Missing public grouping identity: {column}")
            out[column] = frame[column].map(lambda value: token(column, value))
    for column in ("submit_time", "completion_time"):
        if column in frame:
            values = pd.to_datetime(frame[column], utc=True, errors="raise")
            out[column+"_relative_seconds"] = (values-anchor).dt.total_seconds()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--arm", default="train_tail")
    p.add_argument("--key-file", type=Path, help="Private stable >=32-byte HMAC key; created outside export if omitted")
    a = p.parse_args(argv)
    source, dest = a.study_root.resolve(), a.output_dir.resolve()
    if dest.exists():
        raise FileExistsError(f"Export already exists: {dest}")
    if dest == source or dest in source.parents:
        raise ValueError("Export directory must not contain the private study root")
    key_path = a.key_file.resolve() if a.key_file else source / "PRIVATE_release_hmac.key"
    if key_path == dest or dest in key_path.parents:
        raise ValueError("HMAC key must remain outside the export directory")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if not key_path.exists():
        with key_path.open("xb") as stream:
            stream.write(secrets.token_bytes(32))
    key = key_path.read_bytes()
    population = pd.read_csv(source / "protocol" / "population_index.csv")
    anchor = pd.to_datetime(population.submit_time, utc=True, errors="raise").min().floor("D")
    paths = sorted((source / a.arm / "cascade" / "reliability_cascade").glob("tokens_*/*/seed_*/predictions.csv"))
    if not paths:
        raise FileNotFoundError("No new Cascade predictions to stage")
    dest.mkdir(parents=True)
    manifest = {"schema":"local-anonymized-release/v1", "published":False,
                "institutional_review_required":True, "files":{},
                "time_representation":"seconds relative to a withheld common source anchor",
                "key_exported":False, "raw_scripts_or_ids_exported":False}
    for path in paths:
        data = pd.read_csv(path, dtype={"sample_id":str,"user_id":str,"script_sha256":str})
        # Cascade predictions may omit the canonical hash; join metadata only.
        if "script_sha256" not in data:
            data = data.merge(population[["sample_id","script_sha256"]],on="sample_id",validate="many_to_one")
        result = deidentify(data,key,anchor)
        relative = path.relative_to(source / a.arm / "cascade")
        output = dest / relative
        output.parent.mkdir(parents=True,exist_ok=True)
        result.to_csv(output,index=False)
        manifest["files"][relative.as_posix()] = {"rows":len(result),
            "sha256":hashlib.sha256(output.read_bytes()).hexdigest()}
    (dest / "release_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Local release-review staging complete: {dest}; the private HMAC key was not exported.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
