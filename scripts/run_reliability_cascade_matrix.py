#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config, parse_overrides  # noqa: E402
from common.data import sha256_file  # noqa: E402
from reliability_cascade.integrity import (  # noqa: E402
    build_run_request,
    implementation_contract,
    validate_variant,
    verify_run_directory,
    write_json,
)
from reliability_cascade.pipeline import PROTOCOL_VERSION  # noqa: E402


PLAN_SCHEMA = "reliability-cascade-matrix-plan/v1"
RESERVED_OVERRIDES = {
    "seed",
    "qwen.max_tokens",
    "reliability_cascade.qwen_variant",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen-Qwen reliability cascade for multiple seeds"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--component-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--variant", default="full")
    parser.add_argument(
        "--experiment-label",
        help="Human-readable matrix identity recorded in the immutable plan",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--qwen-template",
        default=(
            "qwen/tokens_{max_tokens}/{variant}/seed_{seed}/predictions.csv"
        ),
    )
    parser.add_argument(
        "--last1-template",
        default="last-1/completed_before_submission/seed_{seed}/predictions.csv",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args(argv)


def _resolved(path: str) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _format_path(root: Path, template: str, *, seed: int, variant: str, max_tokens: int) -> Path:
    rendered = template.format(
        seed=seed,
        variant=variant,
        max_tokens=max_tokens,
    )
    value = Path(rendered).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _run_output(root: Path, *, seed: int, variant: str, max_tokens: int) -> Path:
    return (
        root
        / "reliability_cascade"
        / f"tokens_{max_tokens}"
        / variant
        / f"seed_{seed}"
    )


def _preflight_output_directories(
    outputs: list[Path],
    *,
    resume: bool,
    overwrite: bool,
) -> None:
    """Reject an accidental restart before launching any seed."""
    if resume or overwrite:
        return
    existing = [path for path in outputs if path.exists()]
    if not existing:
        return
    preview = ", ".join(str(path) for path in existing[:5])
    remainder = len(existing) - min(len(existing), 5)
    suffix = f" (and {remainder} more)" if remainder else ""
    raise FileExistsError(
        "Cascade output directories already exist before matrix execution: "
        f"{preview}{suffix}. Re-run the identical matrix command with --resume "
        "to verify completed seeds and rebuild only incomplete seeds; use "
        "--overwrite only for an intentional full replacement."
    )


def _complete(
    path: Path,
    *,
    seed: int,
    model: str,
    run_request_sha256: str,
    variant: str = "full",
    max_tokens: int = 4096,
) -> bool:
    try:
        verify_run_directory(
            path,
            expected_run_request_sha256=run_request_sha256,
            expected_seed=seed,
            expected_model=model,
            expected_variant=variant,
            expected_max_tokens=max_tokens,
        )
    except (OSError, ValueError, TypeError):
        return False
    return True


def _acquire_matrix_root_lock(output_root: Path) -> Any:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_root.parent / f".{output_root.name}.matrix.lock"
    handle = lock_path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError) as exc:
        handle.close()
        raise RuntimeError(
            "Another reliability-cascade matrix process is using output root: "
            f"{output_root}"
        ) from exc
    return handle


def _release_matrix_root_lock(handle: Any) -> None:
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _read_plan(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Cascade experiment plan must be a JSON object: {path}")
    return value


def _ensure_plan(path: Path, plan: dict[str, object]) -> None:
    if path.is_file():
        existing = _read_plan(path)
        if existing != plan:
            raise ValueError(
                "The output root belongs to a different cascade experiment plan; "
                "use a new --output-root"
            )
        return
    if path.exists():
        raise FileExistsError(f"Cascade experiment plan path is not a file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        write_json(temporary, plan)
        # The output-root lock prevents cooperating writers from reaching this
        # point concurrently. The existence check also fails closed if an
        # external writer ignores that lock.
        if path.exists():
            existing = _read_plan(path)
            if existing != plan:
                raise ValueError(
                    "The output root acquired a different cascade experiment plan"
                )
        else:
            temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()

    if _read_plan(path) != plan:
        raise ValueError("Cascade experiment plan failed post-write verification")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be unique")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    args.variant = validate_variant(args.variant)
    override_keys = set(parse_overrides(args.overrides))
    conflicting = sorted(override_keys & RESERVED_OVERRIDES)
    if conflicting:
        raise ValueError(
            "Use the dedicated seed/context/variant arguments instead of --set for: "
            f"{conflicting}"
        )

    config_path = _resolved(args.config)
    component_root = _resolved(args.component_root)
    output_root = _resolved(args.output_root)
    model_name = f"reliability_cascade_{args.variant}"
    runs: list[dict[str, object]] = []
    commands: list[tuple[int, Path, Path, Path, list[str], str | None]] = []
    for seed in args.seeds:
        qwen = _format_path(
            component_root,
            args.qwen_template,
            seed=seed,
            variant=args.variant,
            max_tokens=args.max_tokens,
        )
        last1 = _format_path(
            component_root,
            args.last1_template,
            seed=seed,
            variant=args.variant,
            max_tokens=args.max_tokens,
        )
        output = _run_output(
            output_root,
            seed=seed,
            variant=args.variant,
            max_tokens=args.max_tokens,
        )
        run_overrides = [
            *args.overrides,
            f"seed={seed}",
            f"qwen.max_tokens={args.max_tokens}",
            f"reliability_cascade.qwen_variant={args.variant}",
        ]
        request_sha256: str | None = None
        if not args.dry_run:
            if not qwen.is_file() or not last1.is_file():
                raise FileNotFoundError(
                    f"Missing seed {seed} components: qwen={qwen}, last1={last1}"
                )
            config = load_config(config_path, run_overrides)
            request = build_run_request(
                config,
                qwen,
                last1,
                protocol=PROTOCOL_VERSION,
            )
            request_sha256 = str(request["run_request_sha256"])
        runs.append(
            {
                "seed": seed,
                "qwen_predictions": str(qwen),
                "last1_predictions": str(last1),
                "output_dir": str(output),
                "run_request_sha256": request_sha256,
            }
        )
        command = [
            sys.executable,
            "-m",
            "reliability_cascade.run",
            "--config",
            str(config_path),
            "--qwen-predictions",
            str(qwen),
            "--last1-predictions",
            str(last1),
            "--output-dir",
            str(output),
        ]
        for override in run_overrides:
            command.extend(["--set", override])
        commands.append((seed, qwen, last1, output, command, request_sha256))

    plan = {
        "schema": PLAN_SCHEMA,
        "protocol": PROTOCOL_VERSION,
        "implementation": implementation_contract(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "component_root": str(component_root),
        "output_root": str(output_root),
        "seeds": list(args.seeds),
        "variant": args.variant,
        "experiment_label": args.experiment_label or f"{args.variant}_default",
        "max_tokens": args.max_tokens,
        "qwen_template": args.qwen_template,
        "last1_template": args.last1_template,
        "overrides": list(args.overrides),
        "runs": runs,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=True))
        for _, _, _, _, command, _ in commands:
            print("[DRY-RUN] " + shlex.join([*command, "--dry-run"]))
        return 0

    matrix_lock = _acquire_matrix_root_lock(output_root)
    try:
        _ensure_plan(output_root / "cascade_experiment_plan.json", plan)
        _preflight_output_directories(
            [output for _, _, _, output, _, _ in commands],
            resume=bool(args.resume),
            overwrite=bool(args.overwrite),
        )
        for seed, _, _, output, command, request_sha256 in commands:
            if request_sha256 is None:
                raise AssertionError("Formal matrix run has no request fingerprint")
            if args.resume and _complete(
                output,
                seed=seed,
                model=model_name,
                run_request_sha256=request_sha256,
                variant=args.variant,
                max_tokens=args.max_tokens,
            ):
                print(f"[SKIP] seed {seed}: verified artifact at {output}")
                continue
            resume_incomplete = bool(args.resume and output.exists())
            if args.overwrite or resume_incomplete:
                command.append("--overwrite")
            print("[RUN] " + shlex.join(command), flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            if not _complete(
                output,
                seed=seed,
                model=model_name,
                run_request_sha256=request_sha256,
                variant=args.variant,
                max_tokens=args.max_tokens,
            ):
                raise RuntimeError(
                    f"Seed {seed} finished without a matching verified artifact: {output}"
                )
    finally:
        _release_matrix_root_lock(matrix_lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
