from __future__ import annotations

import ast
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "run_paper_experiments.py"
SPEC = importlib.util.spec_from_file_location("run_paper_experiments", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class PaperExperimentRunnerTests(unittest.TestCase):
    def _spec(
        self,
        root: Path,
        *,
        seeds=(40, 42),
        variants=("full", "no_code"),
        max_tokens=(2048, 4096),
        num_processes=8,
    ):
        config = root / "paper.local.yaml"
        config.write_text("seed: 42\n", encoding="utf-8")
        return runner.ExperimentSpec(
            config=config.resolve(),
            output_root=(root / "paper_outputs").resolve(),
            seeds=tuple(seeds),
            variants=tuple(variants),
            num_processes=num_processes,
            model_path="/models/Qwen2.5-Coder-1.5B-Instruct",
            max_tokens=tuple(max_tokens),
            batch_size=2,
            eval_batch_size=2,
            gradient_accumulation_steps=4,
        )

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def _complete_last1(self, step) -> None:
        step.output_dir.mkdir(parents=True, exist_ok=True)
        (step.output_dir / "predictions.csv").write_text("x\n1\n", encoding="utf-8")
        (step.output_dir / "metrics.csv").write_text("x\n1\n", encoding="utf-8")
        self._write_json(
            step.output_dir / "run_manifest.json",
            {
                "model": "last-1",
                "seed": step.seed,
                "details": {
                    "protocol": runner.CAUSAL_LAST1_PROTOCOL,
                    "causal_online_protocol": True,
                },
            },
        )

    def _complete_qwen_training(self, step) -> None:
        checkpoint = step.output_dir / "checkpoint"
        self._write_json(checkpoint / "model_config.json", {"variant": step.variant})
        self._write_json(
            checkpoint / "training_manifest.json",
            {
                "seed": step.seed,
                "variant": step.variant,
                "best_epoch": 2,
                "effective_parameters": {"max_tokens": step.max_tokens},
                "experiment_contract": {"sha256": "contract-sha"},
                "training_runtime": {"completed_epochs": 3},
            },
        )
        self._write_json(
            step.output_dir / "training_complete.json",
            {
                "status": "complete",
                "best_epoch": 2,
                "completed_epochs": 3,
                "experiment_contract_sha256": "contract-sha",
            },
        )

    def _complete_qwen_evaluation(self, step) -> None:
        self._complete_qwen_training(step)
        (step.output_dir / "predictions.csv").write_text("x\n1\n", encoding="utf-8")
        (step.output_dir / "metrics.csv").write_text("x\n1\n", encoding="utf-8")
        self._write_json(
            step.output_dir / "run_manifest.json",
            {
                "model": f"qwen_{step.variant}",
                "seed": step.seed,
                "details": {
                    "variant": step.variant,
                    "experiment_contract_sha256": "contract-sha",
                },
            },
        )

    def _complete_hybrid(self, step, *, include_reference=True) -> None:
        step.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("predictions.csv", "metrics.csv"):
            (step.output_dir / name).write_text("x\n1\n", encoding="utf-8")
        if include_reference:
            (step.output_dir / "reference_metrics.csv").write_text(
                "x\n1\n", encoding="utf-8"
            )
        self._write_json(
            step.output_dir / "run_manifest.json",
            {
                "model": f"safe_script_history_hybrid_{step.variant}",
                "seed": step.seed,
                "qwen_variant": step.variant,
                "selection_objective": runner.SCRIPT_HISTORY_OBJECTIVE,
                "selection_split": "validation",
                "test_labels_used_for_selection": False,
            },
        )
        self._write_json(
            step.output_dir / "script_history_policy.json",
            {
                "selection_objective": runner.SCRIPT_HISTORY_OBJECTIVE,
                "selection_split": "validation",
                "test_labels_used_for_selection": False,
            },
        )

    def test_matrix_has_one_last1_per_seed_and_three_steps_per_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._spec(Path(temporary))
            plan = runner.build_plan(spec)

        expected_runs = 2 * 2 * 2
        self.assertEqual(len(plan), 2 + 3 * expected_runs)
        self.assertEqual(sum(step.stage == "last1" for step in plan), 2)
        self.assertEqual(sum(step.stage == "qwen_train" for step in plan), expected_runs)
        self.assertEqual(
            sum(step.stage == "qwen_evaluate" for step in plan), expected_runs
        )
        self.assertEqual(
            sum(step.stage == "safe_script_history_hybrid" for step in plan),
            expected_runs,
        )

    def test_token_variant_seed_paths_cannot_collide(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._spec(Path(temporary))
            plan = runner.build_plan(spec)

        train_outputs = {
            str(step.output_dir) for step in plan if step.stage == "qwen_train"
        }
        hybrid_outputs = {
            str(step.output_dir)
            for step in plan
            if step.stage == "safe_script_history_hybrid"
        }
        self.assertEqual(len(train_outputs), 8)
        self.assertEqual(len(hybrid_outputs), 8)
        self.assertTrue(any("tokens_2048" in value for value in train_outputs))
        self.assertTrue(any("tokens_4096" in value for value in train_outputs))

    def test_training_uses_single_node_accelerate_and_local_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._spec(
                Path(temporary), seeds=(42,), variants=("full",), max_tokens=(4096,)
            )
            plan = runner.build_plan(spec)
        train = next(step for step in plan if step.stage == "qwen_train")
        command = list(train.command)

        self.assertEqual(command[:3], [sys.executable, "-m", "accelerate.commands.launch"])
        self.assertIn("--multi_gpu", command)
        self.assertEqual(command[command.index("--num_processes") + 1], "8")
        self.assertEqual(command[command.index("--main_process_port") + 1], "0")
        self.assertIn("qwen.max_tokens=4096", command)
        self.assertIn(
            "qwen.model_name_or_path=/models/Qwen2.5-Coder-1.5B-Instruct",
            command,
        )
        self.assertIn("qwen.batch_size=2", command)
        self.assertIn("qwen.eval_batch_size=2", command)
        self.assertIn("qwen.gradient_accumulation_steps=4", command)
        self.assertNotIn("--overwrite", command)

    def test_evaluation_is_single_process_and_uses_matching_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._spec(
                Path(temporary), seeds=(42,), variants=("full",), max_tokens=(4096,)
            )
            plan = runner.build_plan(spec)
        train = next(step for step in plan if step.stage == "qwen_train")
        evaluate = next(step for step in plan if step.stage == "qwen_evaluate")
        command = list(evaluate.command)

        self.assertEqual(command[:3], [sys.executable, "-m", "qwen.evaluate"])
        self.assertNotIn("accelerate.commands.launch", command)
        self.assertEqual(
            Path(command[command.index("--checkpoint") + 1]),
            train.output_dir / "checkpoint",
        )
        self.assertEqual(evaluate.depends_on, (train.step_id,))

    def test_last1_and_hybrid_are_seed_matched(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._spec(
                Path(temporary), seeds=(40, 42), variants=("full",), max_tokens=(4096,)
            )
            plan = runner.build_plan(spec)

        for seed in spec.seeds:
            last1 = next(
                step for step in plan if step.stage == "last1" and step.seed == seed
            )
            hybrid = next(
                step
                for step in plan
                if step.stage == "safe_script_history_hybrid" and step.seed == seed
            )
            command = list(hybrid.command)
            self.assertEqual(
                Path(command[command.index("--last1-predictions") + 1]),
                last1.output_dir / "predictions.csv",
            )
            self.assertIn(f"seed={seed}", command)
            self.assertIn("hybrid.selection_objective=script_history_constrained_mae", command)
            self.assertIn("hybrid.max_acc_drop_vs_reference=0.0", command)
            self.assertIn(last1.step_id, hybrid.depends_on)

    def test_plan_declares_validation_only_selection_and_reads_no_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._spec(
                Path(temporary), seeds=(42,), variants=("full",), max_tokens=(4096,)
            )
            plan = runner.build_plan(spec)
            payload = runner.plan_payload(spec, plan)

        policy = payload["selection_policy"]
        self.assertEqual(payload["schema_version"], 3)
        self.assertIn("qwen/train.py", payload["source_sha256"])
        self.assertIn("common/data.py", payload["source_sha256"])
        self.assertTrue(
            all(len(value) == 64 for value in payload["source_sha256"].values())
        )
        self.assertEqual(policy["selection_split"], "validation")
        self.assertEqual(policy["test_split_role"], "report_only")
        self.assertFalse(policy["runner_reads_metric_files"])
        rendered = json.dumps(payload)
        self.assertNotIn("metrics.csv", rendered)

    def test_plan_binds_the_captured_base_model_inventory(self):
        inventory_sha256 = "a" * 64
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._spec(
                Path(temporary), seeds=(42,), variants=("full",), max_tokens=(4096,)
            )
            plan = runner.build_plan(spec)
            with mock.patch.dict(
                os.environ,
                {"BASE_MODEL_INVENTORY_SHA256": inventory_sha256},
            ):
                payload = runner.plan_payload(spec, plan)

        self.assertEqual(payload["base_model_inventory_sha256"], inventory_sha256)

    def test_plan_rejects_an_invalid_base_model_inventory_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._spec(
                Path(temporary), seeds=(42,), variants=("full",), max_tokens=(4096,)
            )
            plan = runner.build_plan(spec)
            with mock.patch.dict(
                os.environ,
                {"BASE_MODEL_INVENTORY_SHA256": "not-a-sha256"},
            ):
                with self.assertRaisesRegex(ValueError, "SHA-256 hex digest"):
                    runner.plan_payload(spec, plan)

    def test_cli_accepts_comma_and_space_separated_matrix_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.yaml"
            config.write_text("seed: 42\n", encoding="utf-8")
            args = runner.parse_args(
                [
                    "--config",
                    str(config),
                    "--output-root",
                    str(root / "out"),
                    "--seeds",
                    "40,41",
                    "42",
                    "--variants",
                    "full,no_code",
                    "no_user",
                    "--max-tokens",
                    "2048,4096",
                    "--num-processes",
                    "8",
                    "--batch-size",
                    "2",
                    "--eval-batch-size",
                    "2",
                    "--gradient-accumulation-steps",
                    "4",
                ]
            )
            spec = runner.make_spec(args)

        self.assertEqual(spec.seeds, (40, 41, 42))
        self.assertEqual(spec.variants, ("full", "no_code", "no_user"))
        self.assertEqual(spec.max_tokens, (2048, 4096))
        self.assertEqual(spec.batch_size, 2)
        self.assertEqual(spec.eval_batch_size, 2)
        self.assertEqual(spec.gradient_accumulation_steps, 4)

    def test_dry_run_launches_nothing_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.yaml"
            config.write_text("seed: 42\n", encoding="utf-8")
            output_root = root / "out"
            with mock.patch.object(subprocess, "run") as run:
                with redirect_stdout(io.StringIO()):
                    result = runner.main(
                        [
                            "--config",
                            str(config),
                            "--output-root",
                            str(output_root),
                            "--model-path",
                            "/models/qwen",
                            "--dry-run",
                        ]
                    )

            self.assertEqual(result, 0)
            run.assert_not_called()
            self.assertFalse(output_root.exists())

    def test_resume_skips_only_complete_artifact_sets(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._spec(
                Path(temporary), seeds=(42,), variants=("full",), max_tokens=(4096,)
            )
            plan = runner.build_plan(spec)
            last1 = next(step for step in plan if step.stage == "last1")
            self._complete_last1(last1)
            with mock.patch.object(subprocess, "run") as run:
                with redirect_stdout(io.StringIO()) as stdout:
                    runner.execute_plan(spec, plan, dry_run=False, resume=True)

            self.assertIn(f"[SKIP] {last1.step_id}", stdout.getvalue())
            self.assertEqual(run.call_count, len(plan) - 1)

    def test_hybrid_completion_requires_reference_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._spec(
                Path(temporary), seeds=(42,), variants=("full",), max_tokens=(4096,)
            )
            hybrid = next(
                step
                for step in runner.build_plan(spec)
                if step.stage == "safe_script_history_hybrid"
            )
            self._complete_hybrid(hybrid, include_reference=False)
            missing = runner.completion_status(hybrid)
            (hybrid.output_dir / "reference_metrics.csv").write_text(
                "x\n1\n", encoding="utf-8"
            )
            complete = runner.completion_status(hybrid)

        self.assertFalse(missing.complete)
        self.assertIn("reference_metrics.csv", missing.reason)
        self.assertTrue(complete.complete)

    def test_resume_skips_a_fully_validated_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._spec(
                Path(temporary), seeds=(42,), variants=("full",), max_tokens=(4096,)
            )
            plan = runner.build_plan(spec)
            last1 = next(step for step in plan if step.stage == "last1")
            evaluate = next(step for step in plan if step.stage == "qwen_evaluate")
            hybrid = next(
                step for step in plan if step.stage == "safe_script_history_hybrid"
            )
            self._complete_last1(last1)
            self._complete_qwen_evaluation(evaluate)
            self._complete_hybrid(hybrid)

            with mock.patch.object(subprocess, "run") as run:
                with redirect_stdout(io.StringIO()) as stdout:
                    runner.execute_plan(spec, plan, dry_run=False, resume=True)

            run.assert_not_called()
            self.assertEqual(stdout.getvalue().count("[SKIP]"), len(plan))

    def test_partial_qwen_resume_resets_only_exact_run_and_dependents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self._spec(
                root, seeds=(42,), variants=("full",), max_tokens=(4096,)
            )
            plan = runner.build_plan(spec)
            last1 = next(step for step in plan if step.stage == "last1")
            train = next(step for step in plan if step.stage == "qwen_train")
            hybrid = next(
                step for step in plan if step.stage == "safe_script_history_hybrid"
            )
            self._complete_last1(last1)

            self._write_json(train.output_dir / "checkpoint" / "model_config.json", {})
            (train.output_dir / "partial-only.txt").write_text("partial", encoding="utf-8")
            self._complete_hybrid(hybrid)
            outside = spec.output_root / "do-not-remove.txt"
            outside.parent.mkdir(parents=True, exist_ok=True)
            outside.write_text("user", encoding="utf-8")

            with mock.patch.object(subprocess, "run") as run:
                with redirect_stdout(io.StringIO()) as stdout:
                    runner.execute_plan(spec, plan, dry_run=False, resume=True)

            rendered = stdout.getvalue()
            self.assertIn(f"[RERUN] {train.step_id}", rendered)
            self.assertIn(f"[RESET] {train.step_id}", rendered)
            self.assertFalse((train.output_dir / "partial-only.txt").exists())
            self.assertFalse(hybrid.output_dir.exists())
            self.assertTrue(outside.is_file())
            train_call = next(
                call for call in run.call_args_list if "qwen.train" in call.args[0]
            )
            self.assertNotIn("--overwrite", train_call.args[0])

    def test_resume_dry_run_reports_but_does_not_remove_partial_qwen(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._spec(
                Path(temporary), seeds=(42,), variants=("full",), max_tokens=(4096,)
            )
            plan = runner.build_plan(spec)
            train = next(step for step in plan if step.stage == "qwen_train")
            partial = train.output_dir / "partial-only.txt"
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text("partial", encoding="utf-8")

            with mock.patch.object(subprocess, "run") as run:
                with redirect_stdout(io.StringIO()) as stdout:
                    runner.execute_plan(spec, plan, dry_run=True, resume=True)

            run.assert_not_called()
            self.assertTrue(partial.is_file())
            self.assertIn(f"[RESET] {train.step_id}", stdout.getvalue())

    def test_corrupt_completion_json_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._spec(
                Path(temporary), seeds=(42,), variants=("full",), max_tokens=(4096,)
            )
            plan = runner.build_plan(spec)
            last1 = next(step for step in plan if step.stage == "last1")
            last1.output_dir.mkdir(parents=True)
            for name in ("predictions.csv", "metrics.csv"):
                (last1.output_dir / name).write_text("x\n1\n", encoding="utf-8")
            (last1.output_dir / "run_manifest.json").write_text("not-json", encoding="utf-8")
            status = runner.completion_status(last1)
            with mock.patch.object(subprocess, "run"):
                with redirect_stdout(io.StringIO()) as stdout:
                    runner.execute_plan(spec, plan, dry_run=False, resume=True)

        self.assertFalse(status.complete)
        self.assertIn(f"[RERUN] {last1.step_id}", stdout.getvalue())

    def test_training_completion_requires_explicit_complete_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._spec(
                Path(temporary), seeds=(42,), variants=("full",), max_tokens=(4096,)
            )
            train = next(
                step for step in runner.build_plan(spec) if step.stage == "qwen_train"
            )
            self._complete_qwen_training(train)
            marker_path = train.output_dir / "training_complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["status"] = "running"
            self._write_json(marker_path, marker)
            incomplete = runner.completion_status(train)
            marker["status"] = "complete"
            self._write_json(marker_path, marker)
            complete = runner.completion_status(train)

        self.assertFalse(incomplete.complete)
        self.assertIn("status=complete", incomplete.reason)
        self.assertTrue(complete.complete)

    def test_qwen_overwrite_invalidator_removes_old_completion_and_evaluation(self):
        source_path = PROJECT_ROOT / "qwen" / "train.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_invalidate_overwrite_artifacts"
        )
        module = ast.Module(body=[function], type_ignores=[])
        namespace = {"Path": Path}
        exec(compile(module, str(source_path), "exec"), namespace)

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            removed = (
                "training_complete.json",
                "predictions.csv",
                "metrics.csv",
                "run_manifest.json",
            )
            for name in removed:
                (run_dir / name).write_text("old", encoding="utf-8")
            retained = run_dir / "checkpoint.bin"
            retained.write_text("checkpoint", encoding="utf-8")
            namespace["_invalidate_overwrite_artifacts"](run_dir)

            self.assertTrue(retained.is_file())
            self.assertFalse(any((run_dir / name).exists() for name in removed))

        source = source_path.read_text(encoding="utf-8")
        self.assertIn('"status": "complete"', source)


if __name__ == "__main__":
    unittest.main()
