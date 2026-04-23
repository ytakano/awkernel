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
    "row_index",
    "lifecycle_index",
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
                if (candidate / "scheduling_theory").is_dir() and (candidate / "documents").is_dir():
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
        runner_path.write_text(
            "import sys\n"
            + body
            + "\n",
            encoding="utf-8",
        )
        return runner_path

    def run_wrapper(
        self,
        *,
        log_text: str,
        backend: str = "test-backend",
        scenario: str = "test-scenario",
        runhaskell: str | None = None,
        runner: pathlib.Path | None = None,
        checker_dir: pathlib.Path | None = None,
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

    def test_missing_rows_block_reports_wrapper_failure(self) -> None:
        code, payload, stdout, stderr = self.run_wrapper(
            log_text="\n".join(
                [
                    "boot",
                    "BEGIN_TASK_LIFECYCLE",
                    "Spawn\t1\t-",
                    "END_TASK_LIFECYCLE",
                ]
            )
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="missing-rows-block")
        self.assertIsNone(payload["row_index"])
        self.assertIsNone(payload["lifecycle_index"])
        self.assertIsNone(payload["log_line_begin"])
        self.assertIsNone(payload["log_line_end"])
        self.assertIn("rejected", stderr)

    def test_empty_rows_block_reports_line_span(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "boot",
                    "BEGIN_TRACE_ROWS",
                    "END_TRACE_ROWS",
                    "BEGIN_TASK_LIFECYCLE",
                    "Spawn\t1\t-",
                    "END_TASK_LIFECYCLE",
                ]
            )
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="empty-rows-block")
        self.assertEqual(payload["log_line_begin"], 2)
        self.assertEqual(payload["log_line_end"], 3)

    def test_missing_lifecycle_block_reports_wrapper_failure(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_TRACE_ROWS",
                    "0\tWakeup\t1\t-\t-\t1\tfalse\t-",
                    "END_TRACE_ROWS",
                ]
            )
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="missing-lifecycle-block")
        self.assertIsNone(payload["log_line_begin"])
        self.assertIsNone(payload["log_line_end"])

    def test_empty_lifecycle_block_reports_line_span(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_TRACE_ROWS",
                    "0\tWakeup\t1\t-\t-\t1\tfalse\t-",
                    "END_TRACE_ROWS",
                    "BEGIN_TASK_LIFECYCLE",
                    "END_TASK_LIFECYCLE",
                ]
            )
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="empty-lifecycle-block")
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

    def test_duplicate_rows_markers_report_wrapper_failure(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_TRACE_ROWS",
                    "0\tWakeup\t1\t-\t-\t1\tfalse\t-",
                    "END_TRACE_ROWS",
                    "BEGIN_TRACE_ROWS",
                    "0\tWakeup\t1\t-\t-\t1\tfalse\t-",
                    "END_TRACE_ROWS",
                    "BEGIN_TASK_LIFECYCLE",
                    "Spawn\t1\t-",
                    "END_TASK_LIFECYCLE",
                ]
            )
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="missing-rows-block")
        self.assertEqual(payload["log_line_begin"], 1)
        self.assertEqual(payload["log_line_end"], 4)

    def test_out_of_order_rows_markers_report_wrapper_failure(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "END_TRACE_ROWS",
                    "BEGIN_TRACE_ROWS",
                    "BEGIN_TASK_LIFECYCLE",
                    "Spawn\t1\t-",
                    "END_TASK_LIFECYCLE",
                ]
            )
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="missing-rows-block")
        self.assertEqual(payload["log_line_begin"], 2)
        self.assertEqual(payload["log_line_end"], 1)

    def test_runner_extra_stdout_before_json_is_rejected(self) -> None:
        fake_runner = self.make_runner_script(
            "print('debug banner')\n"
            "print('{\"accepted\": true, \"backend\": \"test-backend\", \"scenario\": \"test-scenario\", "
            "\\\"kind\\\": \\\"accepted\\\", \\\"message\\\": \\\"ok\\\", \\\"row_index\\\": null, "
            "\\\"lifecycle_index\\\": null, \\\"log_line_begin\\\": null, \\\"log_line_end\\\": null}')\n"
        )
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_TRACE_ROWS",
                    "0\tWakeup\t1\t-\t-\t1\tfalse\t-",
                    "END_TRACE_ROWS",
                    "BEGIN_TASK_LIFECYCLE",
                    "Spawn\t1\t-",
                    "END_TASK_LIFECYCLE",
                ]
            ),
            runhaskell=sys.executable,
            runner=fake_runner,
            checker_dir=self.make_dummy_checker_dir(),
        )
        self.assertEqual(code, WRAPPER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="internal-checker-error")

    def test_runner_multiple_json_lines_are_rejected(self) -> None:
        fake_runner = self.make_runner_script(
            "print('{\"accepted\": false, \"backend\": \"test-backend\", \"scenario\": \"test-scenario\", "
            "\\\"kind\\\": \\\"workload-family-rejection\\\", \\\"message\\\": \\\"bad\\\", \\\"row_index\\\": null, "
            "\\\"lifecycle_index\\\": null, \\\"log_line_begin\\\": null, \\\"log_line_end\\\": null}')\n"
            "print('{\"accepted\": false, \"backend\": \"test-backend\", \"scenario\": \"test-scenario\", "
            "\\\"kind\\\": \\\"workload-family-rejection\\\", \\\"message\\\": \\\"bad\\\", \\\"row_index\\\": null, "
            "\\\"lifecycle_index\\\": null, \\\"log_line_begin\\\": null, \\\"log_line_end\\\": null}')\n"
            "sys.exit(1)\n"
        )
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_TRACE_ROWS",
                    "0\tWakeup\t1\t-\t-\t1\tfalse\t-",
                    "END_TRACE_ROWS",
                    "BEGIN_TASK_LIFECYCLE",
                    "Spawn\t1\t-",
                    "END_TASK_LIFECYCLE",
                ]
            ),
            runhaskell=sys.executable,
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
                    "BEGIN_TRACE_ROWS",
                    "0\tWakeup\t1\t-\t-\t1\tfalse\t-",
                    "END_TRACE_ROWS",
                    "BEGIN_TASK_LIFECYCLE",
                    "Spawn\t1\t-",
                    "END_TASK_LIFECYCLE",
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
    def test_rows_parse_failure_reports_row_index(self) -> None:
        code, payload, stdout, stderr = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_TRACE_ROWS",
                    "not-a-valid-row",
                    "END_TRACE_ROWS",
                    "BEGIN_TASK_LIFECYCLE",
                    "Spawn\t1\t-",
                    "END_TASK_LIFECYCLE",
                ]
            ),
            runhaskell=self.runhaskell,
            runner=self.runner,
            checker_dir=self.checker_dir,
        )
        self.assertEqual(code, RUNNER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="rows-parse-failure")
        self.assertEqual(payload["row_index"], 0)
        self.assertIsNone(payload["lifecycle_index"])
        self.assertEqual(payload["log_line_begin"], 2)
        self.assertEqual(payload["log_line_end"], 2)
        self.assertIn("rejected", stderr)

    @unittest.skipUnless(
        (os.environ.get("WORKLOAD_ACCEPT_RUNHASKELL") or shutil.which("runhaskell")) is not None,
        "runhaskell not available",
    )
    def test_lifecycle_parse_failure_reports_lifecycle_index(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_TRACE_ROWS",
                    "0\tWakeup\t1\t-\t-\t1\tfalse\t-",
                    "END_TRACE_ROWS",
                    "BEGIN_TASK_LIFECYCLE",
                    "Broken\t1\t-",
                    "END_TASK_LIFECYCLE",
                ]
            ),
            runhaskell=self.runhaskell,
            runner=self.runner,
            checker_dir=self.checker_dir,
        )
        self.assertEqual(code, RUNNER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="lifecycle-parse-failure")
        self.assertIsNone(payload["row_index"])
        self.assertEqual(payload["lifecycle_index"], 0)
        self.assertEqual(payload["log_line_begin"], 5)
        self.assertEqual(payload["log_line_end"], 5)

    @unittest.skipUnless(
        (os.environ.get("WORKLOAD_ACCEPT_RUNHASKELL") or shutil.which("runhaskell")) is not None,
        "runhaskell not available",
    )
    def test_unsupported_event_tag_stays_a_rows_parse_failure(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_TRACE_ROWS",
                    "1\tPreempt\t1\t2\t-\t1\tfalse\t-",
                    "END_TRACE_ROWS",
                    "BEGIN_TASK_LIFECYCLE",
                    "Spawn\t1\t-",
                    "END_TASK_LIFECYCLE",
                ]
            ),
            runhaskell=self.runhaskell,
            runner=self.runner,
            checker_dir=self.checker_dir,
        )
        self.assertEqual(code, RUNNER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="rows-parse-failure")
        self.assertEqual(payload["row_index"], 0)
        self.assertEqual(payload["log_line_begin"], 2)
        self.assertEqual(payload["log_line_end"], 2)

    @unittest.skipUnless(
        (os.environ.get("WORKLOAD_ACCEPT_RUNHASKELL") or shutil.which("runhaskell")) is not None,
        "runhaskell not available",
    )
    def test_unsupported_lifecycle_kind_stays_a_lifecycle_parse_failure(self) -> None:
        code, payload, stdout, _ = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_TRACE_ROWS",
                    "0\tWakeup\t1\t-\t-\t1\tfalse\t-",
                    "END_TRACE_ROWS",
                    "BEGIN_TASK_LIFECYCLE",
                    "Wake\t1\t-",
                    "END_TASK_LIFECYCLE",
                ]
            ),
            runhaskell=self.runhaskell,
            runner=self.runner,
            checker_dir=self.checker_dir,
        )
        self.assertEqual(code, RUNNER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="lifecycle-parse-failure")
        self.assertEqual(payload["lifecycle_index"], 0)
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
                    "BEGIN_TRACE_ROWS",
                    "0\tWakeup\t1\t-\t-\t1\tfalse\t-",
                    "1\tComplete\t1\t-\t-\t\ttrue\t-",
                    "END_TRACE_ROWS",
                    "BEGIN_TASK_LIFECYCLE",
                    "Spawn\t1\t-",
                    "Runnable\t1\t-",
                    "Choose\t1\t-",
                    "Dispatch\t1\t-",
                    "Complete\t1\t-",
                    "END_TASK_LIFECYCLE",
                ]
            ),
            runhaskell=self.runhaskell,
            runner=self.runner,
            checker_dir=self.checker_dir,
        )
        self.assertEqual(code, RUNNER_FAILURE_EXIT)
        self.assert_single_json_stdout(stdout)
        self.assert_common_failure(payload, kind="workload-family-rejection")
        self.assertIsNone(payload["row_index"])
        self.assertIsNone(payload["lifecycle_index"])
        self.assertIsNone(payload["log_line_begin"])
        self.assertIsNone(payload["log_line_end"])

    @unittest.skipUnless(
        (os.environ.get("WORKLOAD_ACCEPT_RUNHASKELL") or shutil.which("runhaskell")) is not None,
        "runhaskell not available",
    )
    def test_minimal_accepted_trace_returns_fixed_success_schema(self) -> None:
        code, payload, stdout, stderr = self.run_wrapper(
            log_text="\n".join(
                [
                    "BEGIN_TRACE_ROWS",
                    "0\tWakeup\t1\t-\t-\t1\tfalse\t-",
                    "1\tChoose\t1\t1\t-\t1\tfalse\t1",
                    "1\tDispatch\t1\t1\t1\t\tfalse\t-",
                    "1\tComplete\t1\t-\t-\t\ttrue\t-",
                    "END_TRACE_ROWS",
                    "BEGIN_TASK_LIFECYCLE",
                    "Spawn\t1\t-",
                    "Runnable\t1\t-",
                    "Choose\t1\t-",
                    "Dispatch\t1\t-",
                    "Complete\t1\t-",
                    "END_TASK_LIFECYCLE",
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
        self.assertIsNone(payload["row_index"])
        self.assertIsNone(payload["lifecycle_index"])
        self.assertIsNone(payload["log_line_begin"])
        self.assertIsNone(payload["log_line_end"])
        self.assertIn("accepted", stderr)


if __name__ == "__main__":
    unittest.main()
