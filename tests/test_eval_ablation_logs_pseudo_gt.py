from pathlib import Path
import tempfile
import unittest

from TriPoseFusion.eval.eval_ablation_logs_pseudo_gt import (
    EvalJob,
    build_eval_command,
    discover_runs,
    default_data_overrides,
    iter_jobs_with_progress,
    select_checkpoint,
)


def _make_run(root: Path, name: str) -> Path:
    run_dir = root / name / "2026-06-11" / "10-00-00"
    (run_dir / ".hydra").mkdir(parents=True)
    (run_dir / ".hydra" / "config.yaml").write_text("experiment: demo\n")
    ckpt_dir = run_dir / "checkpoints" / "fold_0"
    ckpt_dir.mkdir(parents=True)
    return run_dir


class EvalAblationLogsPseudoGtTests(unittest.TestCase):
    def test_discover_runs_finds_hydra_checkpoint_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = _make_run(tmp_path, "trifusion_base")
            (run_dir / "checkpoints" / "fold_0" / "last.ckpt").write_text("x")
            (tmp_path / "not_a_run").mkdir()

            runs = discover_runs(tmp_path, pattern="trifusion_*", fold=0)

            self.assertEqual(runs, [run_dir])

    def test_select_checkpoint_best_uses_lowest_metric_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = _make_run(tmp_path, "trifusion_base")
            ckpt_dir = run_dir / "checkpoints" / "fold_0"
            (ckpt_dir / "last.ckpt").write_text("last")
            (ckpt_dir / "2-0.87.ckpt").write_text("worse")
            best = ckpt_dir / "0-0.86.ckpt"
            best.write_text("best")

            self.assertEqual(select_checkpoint(run_dir, fold=0, policy="best"), best)

    def test_select_checkpoint_last_uses_last_ckpt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = _make_run(tmp_path, "trifusion_base")
            ckpt_dir = run_dir / "checkpoints" / "fold_0"
            (ckpt_dir / "0-0.86.ckpt").write_text("best")
            last = ckpt_dir / "last.ckpt"
            last.write_text("last")

            self.assertEqual(select_checkpoint(run_dir, fold=0, policy="last"), last)

    def test_build_eval_command_uses_run_config_and_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = _make_run(tmp_path, "trifusion_base")
            ckpt = run_dir / "checkpoints" / "fold_0" / "0-0.86.ckpt"
            output_dir = tmp_path / "eval_out" / "trifusion_base"
            gt_root = tmp_path / "gt"
            eval_script = tmp_path / "eval_trifusion_pesudo_gt.py"

            command = build_eval_command(
                python_executable="python",
                eval_script=eval_script,
                run_dir=run_dir,
                ckpt_path=ckpt,
                output_dir=output_dir,
                gt_root=gt_root,
                split="val",
                fold="0",
                extra_overrides=["data.num_workers=0"],
            )

            self.assertEqual(
                command[:5],
                [
                    "python",
                    str(eval_script),
                    "--config-path",
                    str(run_dir / ".hydra"),
                    "--config-name",
                ],
            )
            self.assertIn("config", command)
            self.assertIn(f"eval.ckpt_path={ckpt}", command)
            self.assertIn(f"eval.output_dir={output_dir}", command)
            self.assertIn(f"++eval.triangulated_gt_root={gt_root}", command)
            self.assertIn("eval.split=val", command)
            self.assertIn("eval.fold=0", command)
            self.assertIn("data.num_workers=0", command)

    def test_build_eval_command_normalizes_relative_paths_for_hydra(self) -> None:
        run_dir = Path("logs/logs/train/trifusion_base/2026-06-11/10-00-00")
        ckpt = run_dir / "checkpoints" / "fold_0" / "0-0.86.ckpt"
        output_dir = Path("logs/eval_ablation_pseudo_gt/trifusion_base")
        gt_root = Path("/home/data/xchen/drive/sam3d_body_triangulated_gt")
        eval_script = Path("TriPoseFusion/eval/eval_trifusion_pesudo_gt.py")

        command = build_eval_command(
            python_executable="python",
            eval_script=eval_script,
            run_dir=run_dir,
            ckpt_path=ckpt,
            output_dir=output_dir,
            gt_root=gt_root,
            split="val",
            fold="0",
        )

        self.assertEqual(command[1], str(eval_script.resolve()))
        self.assertEqual(command[3], str((run_dir / ".hydra").resolve()))
        self.assertIn(f"eval.ckpt_path={ckpt.resolve()}", command)
        self.assertIn(f"eval.output_dir={output_dir.resolve()}", command)
        self.assertIn(f"++eval.triangulated_gt_root={gt_root.resolve()}", command)
        self.assertIn(f"hydra.run.dir={(output_dir / 'hydra_run').resolve()}", command)

    def test_default_data_overrides_use_current_data_layout(self) -> None:
        data_root = Path("/data/drive/multi_view_driver_action")
        sam3d_root = Path("/data/drive/sam3d_body_results_right")

        overrides = default_data_overrides(data_root=data_root, sam3d_root=sam3d_root)

        self.assertEqual(
            overrides,
            [
                "paths.root_path=/data/drive/multi_view_driver_action",
                "paths.index_mapping=/data/drive/multi_view_driver_action/index_mapping",
                "paths.sam3d_results_path=/data/drive/sam3d_body_results_right",
            ],
        )

    def test_iter_jobs_with_progress_wraps_jobs_with_tqdm_factory(self) -> None:
        jobs = [
            EvalJob(
                run_dir=Path("/runs/trifusion_base"),
                ckpt_path=Path("/runs/trifusion_base/checkpoints/fold_0/last.ckpt"),
                output_dir=Path("/eval/trifusion_base"),
                command=["python", "eval.py"],
            )
        ]
        calls = []

        def fake_tqdm(iterable, **kwargs):
            calls.append((list(iterable), kwargs))
            return calls[0][0]

        wrapped = list(iter_jobs_with_progress(jobs, tqdm_factory=fake_tqdm))

        self.assertEqual(wrapped, jobs)
        self.assertEqual(calls[0][0], jobs)
        self.assertEqual(calls[0][1]["total"], 1)
        self.assertEqual(calls[0][1]["desc"], "ablation eval")
        self.assertEqual(calls[0][1]["unit"], "job")


if __name__ == "__main__":
    unittest.main()
