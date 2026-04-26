from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

EXPECTED_KEYS = {
    "accepted",
    "backend",
    "scenario",
    "kind",
    "message",
    "sched_trace_index",
    "task_trace_index",
    "log_line_begin",
    "log_line_end",
}

WRAPPER_FAILURE_EXIT = 2
RUNNER_FAILURE_EXIT = 1
ACCEPTED_EXIT = 0


class WorkloadAcceptanceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.awkernel_root = pathlib.Path(__file__).resolve().parents[1]
        cls.repo_root = cls.find_repo_root(cls.awkernel_root)
        cls.wrapper = cls.awkernel_root / "scripts" / "check_workload_acceptance.py"
        cls.runner = cls.awkernel_root / "scripts" / "haskell" / "WorkloadAcceptanceMain.hs"
        cls.checker_dir = cls.repo_root / "scheduling_theory" / "extracted" / "haskell"
        cls.true_cmd = shutil.which("true")
        cls.runhaskell = os.environ.get("WORKLOAD_ACCEPT_RUNHASKELL") or shutil.which("runhaskell")

    @staticmethod
    def find_repo_root(start: pathlib.Path) -> pathlib.Path:
        env_root = os.environ.get("AWKERNEL_REFINEMENT_ROOT")
        search_roots = [start, pathlib.Path.cwd().resolve()]
        if env_root:
            search_roots.append(pathlib.Path(env_root).resolve())
        search_roots.append(pathlib.Path("/home/ytakano/program/rocq/awkernel_refinement"))
        for root in search_roots:
            for candidate in [root, *root.parents]:
                if (
                    (candidate / "scheduling_theory").is_dir()
                    and (candidate / "awkernel_refinemnet_doc").is_dir()
                ):
                    return candidate
        raise RuntimeError(f"failed to locate awkernel_refinement repo root from {search_roots}")

    def make_log(self, contents: str) -> pathlib.Path:
        tmpdir = tempfile.TemporaryDirectory(prefix="workload-accept-test-")
        self.addCleanup(tmpdir.cleanup)
        log_path = pathlib.Path(tmpdir.name) / "serial.log"
        log_path.write_text(contents, encoding="utf-8")
        return log_path

    def make_dummy_checker_dir(self) -> pathlib.Path:
        tmpdir = tempfile.TemporaryDirectory(prefix="workload-accept-checker-")
        self.addCleanup(tmpdir.cleanup)
        checker_dir = pathlib.Path(tmpdir.name)
        (checker_dir / "AwkernelWorkloadAcceptance.hs").write_text("-- dummy\n", encoding="utf-8")
        return checker_dir

    def make_runner_script(self, body: str) -> pathlib.Path:
        tmpdir = tempfile.TemporaryDirectory(prefix="workload-accept-runner-")
        self.addCleanup(tmpdir.cleanup)
        runner_path = pathlib.Path(tmpdir.name) / "fake_runner.py"
        runner_path.write_text("#!/usr/bin/env python3\nimport sys\n" + body + "\n", encoding="utf-8")
        runner_path.chmod(0o755)
        return runner_path

    @staticmethod
    def make_sched_trace_row(
        cpu: int,
        event_tag: str,
        event_a: str,
        event_b: str,
        current: str,
        runnable: str,
        need_resched: str,
        dispatch_target: str,
        worker_current: str | None = None,
        worker_need_resched: str | None = None,
        worker_dispatch_target: str | None = None,
    ) -> str:
        if worker_current is None:
            worker_current = current
        if worker_need_resched is None:
            worker_need_resched = need_resched
        if worker_dispatch_target is None:
            worker_dispatch_target = dispatch_target
        return "\t".join(
            [
                str(cpu),
                event_tag,
                event_a,
                event_b,
                current,
                runnable,
                need_resched,
                dispatch_target,
                worker_current,
                worker_need_resched,
                worker_dispatch_target,
            ]
        )

    def make_python_runhaskell_shim(self) -> pathlib.Path:
        tmpdir = tempfile.TemporaryDirectory(prefix="workload-accept-runhaskell-")
        self.addCleanup(tmpdir.cleanup)
        shim_path = pathlib.Path(tmpdir.name) / "fake_runhaskell.py"
        shim_path.write_text(
            "#!/usr/bin/env python3\n"
            "import os\n"
            "import sys\n"
            "argv = sys.argv[1:]\n"
            "if argv and argv[0].startswith('-i'):\n"
            "    argv = argv[1:]\n"
            "os.execv(sys.executable, [sys.executable] + argv)\n",
            encoding="utf-8",
        )
        shim_path.chmod(0o755)
        return shim_path

    def run_wrapper(
        self,
        *,
        log_text: str,
        backend: str = "test-backend",
        scenario: str = "test-scenario",
        runhaskell: str | None = None,
        runner: pathlib.Path | None = None,
        checker_dir: pathlib.Path | None = None,
        checker_bin: pathlib.Path | None = None,
    ) -> tuple[int, dict[str, object], str, str]:
        log_path = self.make_log(log_text)
        cmd = [
            sys.executable,
            str(self.wrapper),
            "--backend",
            backend,
            "--scenario",
            scenario,
            "--log",
            str(log_path),
            "--runhaskell",
            runhaskell or self.true_cmd or sys.executable,
            "--runner",
            str(runner or self.wrapper),
            "--checker-dir",
            str(checker_dir or self.make_dummy_checker_dir()),
        ]
        if checker_bin is not None:
            cmd.extend(["--checker-bin", str(checker_bin)])
        result = subprocess.run(cmd, text=True, capture_output=True, cwd=self.awkernel_root)
        stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual(len(stdout_lines), 1, msg=f"stdout must contain exactly one JSON payload line: {result.stdout!r}")
        payload = json.loads(stdout_lines[0])
        return result.returncode, payload, result.stdout, result.stderr

    def assert_common_failure(
        self,
        payload: dict[str, object],
        *,
        kind: str,
        backend: str = "test-backend",
        scenario: str = "test-scenario",
    ) -> None:
        self.assertFalse(payload["accepted"])
        self.assertEqual(payload["backend"], backend)
        self.assertEqual(payload["scenario"], scenario)
        self.assertEqual(payload["kind"], kind)
        self.assertIsInstance(payload["message"], str)
        self.assertEqual(set(payload.keys()), EXPECTED_KEYS)

    def assert_single_json_stdout(self, stdout: str) -> None:
        self.assertEqual(len([line for line in stdout.splitlines() if line.strip()]), 1)

    def test_missing_sched_trace_block_reports_wrapper_failure(self) -> None:
        code, payload, stdout, stderr = self.run_wrapper(
            log_text="\n".join(
                [
                    "boot",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "END_TASK_TRACE",
                ]
            )
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="missing-sched-trace-block")
        self.assertIsNone(payload["sched_trace_index"])
        self.assertIsNone(payload["task_trace_index"])
        self.assertIsNone(payload["log_line_begin"])
        self.assertIsNone(payload["log_line_end"])
        self.assertIn("rejected", stderr)

    def test_baseline_trace_overflow_reports_wrapper_failure(self) -> None:
        code, payload, stdout, stderr = self.run_wrapper(
            log_text="\n".join(
                [
                    "boot",
                    "BASELINE_TRACE_OVERFLOW",
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    self.make_sched_trace_row(1, "Complete", "1", "-", "-", "", "true", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "Runnable\t1\t-",
                    "Complete\t1\t-",
                    "END_TASK_TRACE",
                ]
            )
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="baseline-trace-overflow")
        self.assertIsNone(payload["sched_trace_index"])
        self.assertIsNone(payload["task_trace_index"])
        self.assertEqual(payload["log_line_begin"], 2)
        self.assertEqual(payload["log_line_end"], 2)
        self.assertIn("rejected", stderr)

    def test_empty_sched_trace_block_reports_line_span(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "boot",
                    "BEGIN_SCHED_TRACE",
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "END_TASK_TRACE",
                ]
            )
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="empty-sched-trace-block")
        self.assertEqual(payload["log_line_begin"], 2)
        self.assertEqual(payload["log_line_end"], 3)

    def test_missing_task_trace_block_reports_wrapper_failure(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    "END_SCHED_TRACE",
                ]
            )
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="missing-task-trace-block")
        self.assertIsNone(payload["log_line_begin"])
        self.assertIsNone(payload["log_line_end"])

    def test_empty_task_trace_block_reports_line_span(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "END_TASK_TRACE",
                ]
            )
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="empty-task-trace-block")
        self.assertEqual(payload["log_line_begin"], 4)
        self.assertEqual(payload["log_line_end"], 5)

    def test_runhaskell_not_found_is_reported(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="irrelevant\n",
            runhaskell="/definitely/missing/runhaskell",
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="runhaskell-not-found")

    def test_runner_not_found_is_reported(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="irrelevant\n",
            runner=self.awkernel_root / "scripts" / "missing-runner.hs",
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="runner-not-found")

    def test_checker_module_not_found_is_reported(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="irrelevant\n",
            checker_dir=self.awkernel_root / "scripts" / "missing-checker-dir",
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="checker-module-not-found")

    def test_duplicate_sched_trace_markers_report_wrapper_failure(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "END_TASK_TRACE",
                ]
            )
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="missing-sched-trace-block")
        self.assertEqual(payload["log_line_begin"], 1)
        self.assertEqual(payload["log_line_end"], 4)

    def test_out_of_order_sched_trace_markers_report_wrapper_failure(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "END_SCHED_TRACE",
                    "BEGIN_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "END_TASK_TRACE",
                ]
            )
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="missing-sched-trace-block")
        self.assertEqual(payload["log_line_begin"], 2)
        self.assertEqual(payload["log_line_end"], 1)

    def test_native_checker_bin_success_is_accepted(self) -> None:
        fake_checker = self.make_runner_script(
            "assert sys.argv[1:3] == ['test-backend', 'test-scenario']\n"
            "assert len(sys.argv) == 5\n"
            "print('{\"accepted\": true, \"backend\": \"test-backend\", \"scenario\": \"test-scenario\", "
            "\\\"kind\\\": \\\"accepted\\\", \\\"message\\\": \\\"ok\\\", \\\"sched_trace_index\\\": null, "
            "\\\"task_trace_index\\\": null, \\\"log_line_begin\\\": null, \\\"log_line_end\\\": null}')\n"
        )
        fake_checker.chmod(0o755)
        code, payload, stdout, stderr = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    self.make_sched_trace_row(1, "Complete", "1", "-", "-", "", "true", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "Runnable\t1\t-",
                    "Complete\t1\t-",
                    "END_TASK_TRACE",
                ]
            ),
            checker_bin=fake_checker,
        )
        self.assertEqual(code, ACCEPTED_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assertTrue(payload["accepted"])
        self.assertEqual(payload["kind"], "accepted")
        self.assertIn("accepted", stderr)

    def test_missing_native_checker_bin_reports_wrapper_failure(self) -> None:
        missing_checker = pathlib.Path("/tmp/awkernel-workload-accept-missing-checker")
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "END_TASK_TRACE",
                ]
            ),
            checker_bin=missing_checker,
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="checker-bin-not-found")

    def test_runner_extra_stdout_before_json_is_rejected(self) -> None:
        fake_runner = self.make_runner_script(
            "print('debug banner')\n"
            "print('{\"accepted\": true, \"backend\": \"test-backend\", \"scenario\": \"test-scenario\", "
            "\\\"kind\\\": \\\"accepted\\\", \\\"message\\\": \\\"ok\\\", \\\"sched_trace_index\\\": null, "
            "\\\"task_trace_index\\\": null, \\\"log_line_begin\\\": null, \\\"log_line_end\\\": null}')\n"
        )
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "END_TASK_TRACE",
                ]
            ),
            runhaskell=str(self.make_python_runhaskell_shim()),
            runner=fake_runner,
            checker_dir=self.make_dummy_checker_dir(),
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="internal-checker-error")

    def test_runner_multiple_json_lines_are_rejected(self) -> None:
        fake_runner = self.make_runner_script(
            "print('{\"accepted\": false, \"backend\": \"test-backend\", \"scenario\": \"test-scenario\", "
            "\\\"kind\\\": \\\"workload-family-rejection\\\", \\\"message\\\": \\\"bad\\\", \\\"sched_trace_index\\\": null, "
            "\\\"task_trace_index\\\": null, \\\"log_line_begin\\\": null, \\\"log_line_end\\\": null}')\n"
            "print('{\"accepted\": false, \"backend\": \"test-backend\", \"scenario\": \"test-scenario\", "
            "\\\"kind\\\": \\\"workload-family-rejection\\\", \\\"message\\\": \\\"bad\\\", \\\"sched_trace_index\\\": null, "
            "\\\"task_trace_index\\\": null, \\\"log_line_begin\\\": null, \\\"log_line_end\\\": null}')\n"
            "sys.exit(1)\n"
        )
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "END_TASK_TRACE",
                ]
            ),
            runhaskell=str(self.make_python_runhaskell_shim()),
            runner=fake_runner,
            checker_dir=self.make_dummy_checker_dir(),
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="internal-checker-error")

    def test_runner_malformed_json_is_rejected(self) -> None:
        fake_runner = self.make_runner_script("print('{not json}')\nsys.exit(1)\n")
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "END_TASK_TRACE",
                ]
            ),
            runhaskell=sys.executable,
            runner=fake_runner,
            checker_dir=self.make_dummy_checker_dir(),
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="internal-checker-error")

    @unittest.skipUnless(
        (os.environ.get("WORKLOAD_ACCEPT_RUNHASKELL") or shutil.which("runhaskell")) is not None,
        "runhaskell not available",
    )
    def test_sched_trace_parse_failure_reports_sched_trace_index(self) -> None:
        code, payload, stdout, stderr = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    "not-a-valid-row",
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "END_TASK_TRACE",
                ]
            ),
            runhaskell=self.runhaskell,
            runner=self.runner,
            checker_dir=self.checker_dir,
        )
        self.assertEqual(code, RUNNER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="sched-trace-parse-failure")
        self.assertEqual(payload["sched_trace_index"], 0)
        self.assertIsNone(payload["task_trace_index"])
        self.assertEqual(payload["log_line_begin"], 2)
        self.assertEqual(payload["log_line_end"], 2)
        self.assertIn("rejected", stderr)

    @unittest.skipUnless(
        (os.environ.get("WORKLOAD_ACCEPT_RUNHASKELL") or shutil.which("runhaskell")) is not None,
        "runhaskell not available",
    )
    def test_task_trace_parse_failure_reports_task_trace_index(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Broken\t1\t-",
                    "END_TASK_TRACE",
                ]
            ),
            runhaskell=self.runhaskell,
            runner=self.runner,
            checker_dir=self.checker_dir,
        )
        self.assertEqual(code, RUNNER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="task-trace-parse-failure")
        self.assertIsNone(payload["sched_trace_index"])
        self.assertEqual(payload["task_trace_index"], 0)
        self.assertEqual(payload["log_line_begin"], 5)
        self.assertEqual(payload["log_line_end"], 5)

    @unittest.skipUnless(
        (os.environ.get("WORKLOAD_ACCEPT_RUNHASKELL") or shutil.which("runhaskell")) is not None,
        "runhaskell not available",
    )
    def test_unsupported_event_tag_stays_a_sched_trace_parse_failure(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(1, "Preempt", "1", "2", "-", "1", "false", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "END_TASK_TRACE",
                ]
            ),
            runhaskell=self.runhaskell,
            runner=self.runner,
            checker_dir=self.checker_dir,
        )
        self.assertEqual(code, RUNNER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="sched-trace-parse-failure")
        self.assertEqual(payload["sched_trace_index"], 0)
        self.assertEqual(payload["log_line_begin"], 2)
        self.assertEqual(payload["log_line_end"], 2)

    @unittest.skipUnless(
        (os.environ.get("WORKLOAD_ACCEPT_RUNHASKELL") or shutil.which("runhaskell")) is not None,
        "runhaskell not available",
    )
    def test_malformed_candidate_prefix_stays_a_sched_trace_parse_failure(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    "1\tChoose\t1\t1\t-\t1\tfalse\t1\tbogus\tfalse\t1",
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "END_TASK_TRACE",
                ]
            ),
            runhaskell=self.runhaskell,
            runner=self.runner,
            checker_dir=self.checker_dir,
        )
        self.assertEqual(code, RUNNER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="sched-trace-parse-failure")
        self.assertEqual(payload["sched_trace_index"], 0)
        self.assertEqual(payload["log_line_begin"], 2)
        self.assertEqual(payload["log_line_end"], 2)

    @unittest.skipUnless(
        (os.environ.get("WORKLOAD_ACCEPT_RUNHASKELL") or shutil.which("runhaskell")) is not None,
        "runhaskell not available",
    )
    def test_unsupported_task_trace_kind_stays_a_task_trace_parse_failure(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Wake\t1\t-",
                    "END_TASK_TRACE",
                ]
            ),
            runhaskell=self.runhaskell,
            runner=self.runner,
            checker_dir=self.checker_dir,
        )
        self.assertEqual(code, RUNNER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="task-trace-parse-failure")
        self.assertEqual(payload["task_trace_index"], 0)
        self.assertEqual(payload["log_line_begin"], 5)
        self.assertEqual(payload["log_line_end"], 5)

    @unittest.skipUnless(
        (os.environ.get("WORKLOAD_ACCEPT_RUNHASKELL") or shutil.which("runhaskell")) is not None,
        "runhaskell not available",
    )
    def test_semantic_rejection_reports_family_rejection(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    self.make_sched_trace_row(1, "Complete", "1", "-", "-", "", "true", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "Runnable\t1\t-",
                    "Choose\t1\t-",
                    "Dispatch\t1\t-",
                    "Complete\t1\t-",
                    "END_TASK_TRACE",
                ]
            ),
            runhaskell=self.runhaskell,
            runner=self.runner,
            checker_dir=self.checker_dir,
        )
        self.assertEqual(code, RUNNER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="workload-family-rejection")
        self.assertIsNone(payload["sched_trace_index"])
        self.assertIsNone(payload["task_trace_index"])
        self.assertIsNone(payload["log_line_begin"])
        self.assertIsNone(payload["log_line_end"])

    @unittest.skipUnless(
        (os.environ.get("WORKLOAD_ACCEPT_RUNHASKELL") or shutil.which("runhaskell")) is not None,
        "runhaskell not available",
    )
    def test_global_fifo_rejection_reports_sched_trace_location(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    self.make_sched_trace_row(1, "Choose", "1", "1", "-", "2,1", "false", "1"),
                    self.make_sched_trace_row(1, "Dispatch", "1", "1", "1", "", "false", "-"),
                    self.make_sched_trace_row(1, "Complete", "1", "-", "-", "", "true", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "Runnable\t1\t-",
                    "Choose\t1\t-",
                    "Dispatch\t1\t-",
                    "Complete\t1\t-",
                    "END_TASK_TRACE",
                ]
            ),
            runhaskell=self.runhaskell,
            runner=self.runner,
            checker_dir=self.checker_dir,
        )
        self.assertEqual(code, RUNNER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="global-fifo-rejection")
        self.assertEqual(payload["sched_trace_index"], 1)
        self.assertIsNone(payload["task_trace_index"])
        self.assertEqual(payload["log_line_begin"], 3)
        self.assertEqual(payload["log_line_end"], 3)

    @unittest.skipUnless(
        (os.environ.get("WORKLOAD_ACCEPT_RUNHASKELL") or shutil.which("runhaskell")) is not None,
        "runhaskell not available",
    )
    def test_scheduler_relation_rejection_reports_sched_trace_location(self) -> None:
        fake_runner = self.make_runner_script(
            "import json\n"
            "print(json.dumps({"
            "\"accepted\": False, "
            "\"backend\": \"test-backend\", "
            "\"scenario\": \"test-scenario\", "
            "\"kind\": \"scheduler-relation-rejection\", "
            "\"message\": \"relation mismatch\", "
            "\"sched_trace_index\": 1, "
            "\"task_trace_index\": None, "
            "\"log_line_begin\": None, "
            "\"log_line_end\": None"
            "}))\n"
            "sys.exit(1)\n"
        )
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    self.make_sched_trace_row(1, "Choose", "1", "1", "-", "1,2", "false", "1"),
                    self.make_sched_trace_row(1, "Dispatch", "1", "1", "1", "2", "false", "-"),
                    self.make_sched_trace_row(1, "Complete", "1", "-", "-", "2", "true", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "Spawn\t2\t1",
                    "Runnable\t1\t-",
                    "Runnable\t2\t-",
                    "Choose\t1\t-",
                    "Dispatch\t1\t-",
                    "Complete\t1\t-",
                    "END_TASK_TRACE",
                ]
            ),
            runhaskell=str(self.make_python_runhaskell_shim()),
            runner=fake_runner,
            checker_dir=self.make_dummy_checker_dir(),
        )
        self.assertEqual(code, RUNNER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="scheduler-relation-rejection")
        self.assertEqual(payload["sched_trace_index"], 1)
        self.assertIsNone(payload["task_trace_index"])
        self.assertEqual(payload["log_line_begin"], 3)
        self.assertEqual(payload["log_line_end"], 3)

    @unittest.skipUnless(
        (os.environ.get("WORKLOAD_ACCEPT_RUNHASKELL") or shutil.which("runhaskell")) is not None,
        "runhaskell not available",
    )
    def test_minimal_accepted_trace_returns_fixed_success_schema(self) -> None:
        code, payload, stdout, stderr = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    self.make_sched_trace_row(1, "Choose", "1", "1", "-", "1", "false", "1"),
                    self.make_sched_trace_row(1, "Dispatch", "1", "1", "1", "", "false", "-"),
                    self.make_sched_trace_row(1, "Complete", "1", "-", "-", "", "true", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "Runnable\t1\t-",
                    "Choose\t1\t-",
                    "Dispatch\t1\t-",
                    "Complete\t1\t-",
                    "END_TASK_TRACE",
                ]
            ),
            runhaskell=self.runhaskell,
            runner=self.runner,
            checker_dir=self.checker_dir,
        )
        self.assertEqual(code, ACCEPTED_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assertEqual(set(payload.keys()), EXPECTED_KEYS)
        self.assertTrue(payload["accepted"])
        self.assertEqual(payload["kind"], "accepted")
        self.assertEqual(payload["backend"], "test-backend")
        self.assertEqual(payload["scenario"], "test-scenario")
        self.assertIsNone(payload["sched_trace_index"])
        self.assertIsNone(payload["task_trace_index"])
        self.assertIsNone(payload["log_line_begin"])
        self.assertIsNone(payload["log_line_end"])
        self.assertIn("accepted", stderr)

    @unittest.skipUnless(
        (os.environ.get("WORKLOAD_ACCEPT_RUNHASKELL") or shutil.which("runhaskell")) is not None,
        "runhaskell not available",
    )
    def test_join_target_ready_trace_kind_is_accepted(self) -> None:
        code, payload, stdout, stderr = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    self.make_sched_trace_row(1, "JoinTargetReady", "1", "-", "-", "1", "false", "-"),
                    self.make_sched_trace_row(1, "Choose", "1", "1", "-", "1", "false", "1"),
                    self.make_sched_trace_row(1, "Dispatch", "1", "1", "1", "", "false", "-"),
                    self.make_sched_trace_row(1, "Complete", "1", "-", "-", "", "true", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "Runnable\t1\t-",
                    "JoinTargetReady\t1\t-",
                    "Choose\t1\t-",
                    "Dispatch\t1\t-",
                    "Complete\t1\t-",
                    "END_TASK_TRACE",
                ]
            ),
            runhaskell=self.runhaskell,
            runner=self.runner,
            checker_dir=self.checker_dir,
        )
        self.assertEqual(code, ACCEPTED_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assertEqual(set(payload.keys()), EXPECTED_KEYS)
        self.assertTrue(payload["accepted"])
        self.assertEqual(payload["kind"], "accepted")
        self.assertIn("accepted", stderr)

    @unittest.skipUnless(
        (os.environ.get("WORKLOAD_ACCEPT_RUNHASKELL") or shutil.which("runhaskell")) is not None,
        "runhaskell not available",
    )
    def test_join_wait_completion_requires_join_target_ready(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    self.make_sched_trace_row(0, "Wakeup", "2", "-", "-", "1,2", "false", "-"),
                    self.make_sched_trace_row(1, "Choose", "1", "1", "-", "1,2", "false", "1"),
                    self.make_sched_trace_row(1, "Dispatch", "1", "1", "1", "2", "false", "-"),
                    self.make_sched_trace_row(1, "Complete", "1", "-", "-", "2", "true", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "Spawn\t2\t1",
                    "Runnable\t1\t-",
                    "Runnable\t2\t-",
                    "Choose\t1\t-",
                    "Dispatch\t1\t-",
                    "JoinWait\t1\t2",
                    "Complete\t1\t-",
                    "END_TASK_TRACE",
                ]
            ),
            runhaskell=self.runhaskell,
            runner=self.runner,
            checker_dir=self.checker_dir,
        )
        self.assertEqual(code, RUNNER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="workload-family-rejection")
        self.assertIsNone(payload["sched_trace_index"])
        self.assertIsNone(payload["task_trace_index"])

    @unittest.skipUnless(
        (os.environ.get("WORKLOAD_ACCEPT_RUNHASKELL") or shutil.which("runhaskell")) is not None,
        "runhaskell not available",
    )
    def test_stutter_with_need_resched_true_is_accepted(self) -> None:
        code, payload, stdout, stderr = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_SCHED_TRACE",
                    self.make_sched_trace_row(0, "Wakeup", "1", "-", "-", "1", "false", "-"),
                    self.make_sched_trace_row(1, "Stutter", "-", "-", "-", "1", "true", "-"),
                    self.make_sched_trace_row(1, "Choose", "1", "1", "-", "1", "true", "1"),
                    self.make_sched_trace_row(1, "Dispatch", "1", "1", "1", "", "false", "-"),
                    self.make_sched_trace_row(1, "Complete", "1", "-", "-", "", "true", "-"),
                    "END_SCHED_TRACE",
                    "BEGIN_TASK_TRACE",
                    "Spawn\t1\t-",
                    "Runnable\t1\t-",
                    "Choose\t1\t-",
                    "Dispatch\t1\t-",
                    "Complete\t1\t-",
                    "END_TASK_TRACE",
                ]
            ),
            runhaskell=self.runhaskell,
            runner=self.runner,
            checker_dir=self.checker_dir,
        )
        self.assertEqual(code, ACCEPTED_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assertEqual(set(payload.keys()), EXPECTED_KEYS)
        self.assertTrue(payload["accepted"])
        self.assertEqual(payload["kind"], "accepted")
        self.assertIsNone(payload["sched_trace_index"])
        self.assertIsNone(payload["task_trace_index"])
        self.assertIn("accepted", stderr)


if __name__ == "__main__":
    unittest.main()
