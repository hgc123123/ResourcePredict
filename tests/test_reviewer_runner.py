"""Standard-library planner/recovery tests; no model, YAML or GPU required."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
from contextlib import contextmanager
from uuid import uuid4
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.run_reviewer_experiments import (
    ROOT, archive_interrupted, base_steps, locked_json, parse_args,
    train_base,
)


class ReviewerRunnerTests(unittest.TestCase):
    @contextmanager
    def tempdir(self):
        parent = ROOT / "tests" / ".reviewer_tmp"
        parent.mkdir(exist_ok=True)
        # mkdir defaults also work in Windows sandboxes whose ACL handling is
        # incompatible with tempfile's owner-only creation mode.
        folder = parent / ("test_" + uuid4().hex)
        folder.mkdir()
        try:
            yield str(folder)
        finally:
            if folder.resolve().parent != parent.resolve():
                raise ValueError("Unexpected test cleanup target")
            shutil.rmtree(folder)

    def args(self, root, extra=()):
        return parse_args(["--output-root", str(Path(root) / "study"), "--dry-run", *extra])

    def test_direct_entry_uses_local_scripts_when_another_package_has_same_name(self):
        with self.tempdir() as tmp:
            foreign = Path(tmp) / "third_party"
            competing_package = foreign / "scripts"
            competing_package.mkdir(parents=True)
            (competing_package / "__init__.py").write_text("# Unrelated installed package\n", encoding="utf-8")
            env = os.environ.copy()
            # The project is already on PYTHONPATH, but after the unrelated
            # package. A conditional path insertion does not resolve this.
            env["PYTHONPATH"] = os.pathsep.join([str(foreign), str(ROOT)])
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_reviewer_experiments.py"),
                 "--output-root", str(Path(tmp) / "study"), "--stage", "prepare", "--dry-run"],
                cwd=tmp, env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("[DONE] Commands planned.", result.stdout)

    def test_five_seed_full_no_code_matrix_and_matched_batch(self):
        with self.tempdir() as tmp:
            a = self.args(tmp)
            plan = base_steps(a, "train_tail")
            train = [x for x in plan if x.stage == "qwen_train"]
            self.assertEqual(len(train), 10)
            self.assertEqual({x.seed for x in train}, set(range(40, 45)))
            self.assertEqual({x.variant for x in train}, {"full", "no_code"})
            self.assertEqual(a.num_processes * a.batch_size * a.gradient_accumulation_steps, 12)
            self.assertEqual(len({str(x.output_dir) for x in train}), 10)
            self.assertTrue(all("train_tail" in str(x.output_dir) for x in train))

    def test_arms_do_not_collide(self):
        with self.tempdir() as tmp:
            a = self.args(tmp)
            one = {str(x.output_dir) for x in base_steps(a, "train_tail")}
            two = {str(x.output_dir) for x in base_steps(a, "shared_validation_control")}
            self.assertFalse(one & two)

    def test_locked_plan_is_idempotent_and_rejects_change(self):
        with self.tempdir() as tmp:
            path = Path(tmp) / "plan.json"
            locked_json(path, {"seeds": [40, 41]})
            locked_json(path, {"seeds": [40, 41]})
            with self.assertRaises(FileExistsError):
                locked_json(path, {"seeds": [44]})
            self.assertEqual(json.loads(path.read_text()), {"seeds": [40, 41]})

    def test_recovery_preserves_incomplete_checkpoint(self):
        with self.tempdir() as tmp:
            a = self.args(tmp)
            a.resume = True
            out = a.output_root / "train_tail/base/qwen/tokens_4096/full/seed_40"
            out.mkdir(parents=True)
            (out / "checkpoint.bin").write_bytes(b"recoverable")
            archive_interrupted(a, SimpleNamespace(output_dir=out, stage="qwen_train", step_id="train.40"))
            matches = list((a.output_root / "interrupted").rglob("checkpoint.bin"))
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].read_bytes(), b"recoverable")
            self.assertFalse(out.exists())

    def test_recovery_cannot_touch_outside_study(self):
        with self.tempdir() as tmp:
            a = self.args(tmp)
            a.resume = True
            with self.assertRaises(ValueError):
                archive_interrupted(a, SimpleNamespace(output_dir=Path(tmp), stage="qwen_train", step_id="bad"))

    def test_evaluation_recovery_keeps_checkpoint(self):
        with self.tempdir() as tmp:
            a = self.args(tmp)
            a.resume = True
            out = a.output_root / "train_tail/base/qwen/tokens_4096/full/seed_42"
            out.mkdir(parents=True)
            (out / "checkpoint.bin").write_bytes(b"model")
            (out / "predictions.csv").write_text("partial")
            archive_interrupted(a, SimpleNamespace(output_dir=out, stage="qwen_evaluate", step_id="evaluate.42"))
            self.assertEqual((out / "checkpoint.bin").read_bytes(), b"model")
            self.assertFalse((out / "predictions.csv").exists())

    def test_first_evaluation_accepts_a_fresh_training_directory(self):
        with self.tempdir() as tmp:
            a = self.args(tmp)
            a.dry_run = False
            out = a.output_root / "train_tail/base/qwen/tokens_4096/full/seed_42"
            out.mkdir(parents=True)
            (out / "training_complete.json").write_text("complete")
            archive_interrupted(a, SimpleNamespace(output_dir=out, stage="qwen_evaluate", step_id="evaluate.42"))
            self.assertTrue((out / "training_complete.json").exists())
            self.assertFalse((a.output_root / "interrupted").exists())

    def test_upstream_reexecution_invalidates_existing_dependents(self):
        for failed, expected in (("last1", ["last1", "hybrid"]),
                                 ("train", ["train", "evaluate", "hybrid"])):
            with self.subTest(failed=failed), self.tempdir() as tmp:
                a = self.args(tmp)
                a.dry_run, a.resume = False, True
                qwen = a.output_root / "train_tail/base/qwen/tokens_4096/full/seed_40"
                stages = [("last1", "last1", (), a.output_root / "train_tail/base/last-1/seed_40"),
                          ("train", "qwen_train", (), qwen),
                          ("evaluate", "qwen_evaluate", ("train",), qwen),
                          ("hybrid", "safe_script_history_hybrid", ("last1", "evaluate"),
                           a.output_root / "train_tail/base/hybrid/seed_40")]
                plan = [SimpleNamespace(step_id=i, stage=s, depends_on=d, output_dir=o, command=[i])
                        for i, s, d, o in stages]
                completed = {s.step_id: s.step_id != failed for s in plan}
                for step in plan:
                    step.output_dir.mkdir(parents=True, exist_ok=True)
                (qwen / "predictions.csv").write_text("old evaluation")
                called = []
                def fake_run(command):
                    called.append(command[0])
                    completed[command[0]] = True
                    next(x for x in plan if x.step_id == command[0]).output_dir.mkdir(parents=True, exist_ok=True)
                with patch("scripts.run_reviewer_experiments.base_steps", return_value=plan), \
                     patch("scripts.run_reviewer_experiments.completion_status", side_effect=lambda s: SimpleNamespace(complete=completed[s.step_id], reason="fixture")), \
                     patch("scripts.run_reviewer_experiments.run", side_effect=fake_run):
                    train_base(a, "train_tail")
                self.assertEqual(called, expected)


if __name__ == "__main__":
    unittest.main()
