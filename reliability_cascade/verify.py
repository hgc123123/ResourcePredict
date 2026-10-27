from __future__ import annotations

import argparse
import json
from pathlib import Path

from reliability_cascade.integrity import verify_run_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a reliability-cascade run and every bound artifact"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-variant")
    parser.add_argument("--expected-max-tokens", type=int)
    parser.add_argument("--expected-run-request-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = verify_run_directory(
        Path(args.run_dir),
        expected_run_request_sha256=args.expected_run_request_sha256,
        expected_seed=args.expected_seed,
        expected_model=args.expected_model,
        expected_variant=args.expected_variant,
        expected_max_tokens=args.expected_max_tokens,
    )
    details = manifest.get("details") or {}
    effective = details.get("qwen_effective_parameters") or {}
    print(
        json.dumps(
            {
                "status": "ok",
                "run_dir": str(Path(args.run_dir).expanduser().resolve()),
                "model": manifest.get("model"),
                "seed": manifest.get("seed"),
                "variant": details.get("variant"),
                "max_tokens": effective.get("max_tokens"),
                "run_request_sha256": details.get("run_request_sha256"),
                "artifact_integrity_sha256": details.get(
                    "artifact_integrity_sha256"
                ),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
