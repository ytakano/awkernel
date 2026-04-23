#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Any


SCHED_TRACE_BEGIN = "BEGIN_SCHED_TRACE"
SCHED_TRACE_END = "END_SCHED_TRACE"
TASK_TRACE_BEGIN = "BEGIN_TASK_TRACE"
TASK_TRACE_END = "END_TASK_TRACE"

WRAPPER_FAILURE_EXIT = 2
RUNNER_FAILURE_EXIT = 1
ACCEPTED_EXIT = 0

EXPECTED_DIAGNOSTIC_KEYS = {
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


class AcceptanceError(RuntimeError):
    def __init__(
        self,
        kind: str,
        message: str,
        *,
        log_line_begin: int | None = None,
        log_line_end: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.log_line_begin = log_line_begin
        self.log_line_end = log_line_end


def load_lines(path: pathlib.Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AcceptanceError(
            "log-read-failure",
            f"failed to read serial log {path}: {exc}",
        ) from exc


def extract_block(
    lines: list[str],
    begin: str,
    end: str,
    *,
    missing_kind: str,
    empty_kind: str,
    empty_message: str,
) -> tuple[list[str], int, int]:
    begin_indices = [i for i, line in enumerate(lines) if line.strip() == begin]
    end_indices = [i for i, line in enumerate(lines) if line.strip() == end]

    if len(begin_indices) != 1:
        raise AcceptanceError(
            missing_kind,
            f"expected exactly one {begin} marker, found {len(begin_indices)}",
            log_line_begin=(begin_indices[0] + 1) if begin_indices else None,
            log_line_end=(begin_indices[-1] + 1) if begin_indices else None,
        )
    if len(end_indices) != 1:
        raise AcceptanceError(
            missing_kind,
            f"expected exactly one {end} marker, found {len(end_indices)}",
            log_line_begin=(end_indices[0] + 1) if end_indices else None,
            log_line_end=(end_indices[-1] + 1) if end_indices else None,
        )

    begin_idx = begin_indices[0]
    end_idx = end_indices[0]
    if not begin_idx < end_idx:
        raise AcceptanceError(
            missing_kind,
            f"{begin} and {end} markers are out of order",
            log_line_begin=begin_idx + 1,
            log_line_end=end_idx + 1,
        )

    block = [line.rstrip() for line in lines[begin_idx + 1 : end_idx]]
    if not block:
        raise AcceptanceError(
            empty_kind,
            empty_message,
            log_line_begin=begin_idx + 1,
            log_line_end=end_idx + 1,
        )
    return block, begin_idx + 2, end_idx


def resolve_runhaskell(command: str) -> str:
    if "/" in command:
        path = pathlib.Path(command)
        if not path.is_file():
            raise AcceptanceError("runhaskell-not-found", f"runhaskell not found: {path}")
        return str(path)

    resolved = shutil.which(command)
    if resolved is None:
        raise AcceptanceError("runhaskell-not-found", f"runhaskell not found in PATH: {command}")
    return resolved


def candidate_checker_dirs() -> list[pathlib.Path]:
    env_candidates = [
        os.environ.get("WORKLOAD_ACCEPT_CHECKER_DIR"),
        os.environ.get("AWKERNEL_WORKLOAD_CHECKER_DIR"),
        os.environ.get("SCHEDULING_THEORY_EXTRACTED_HASKELL_DIR"),
    ]
    script_path = pathlib.Path(__file__).resolve()
    discovered: list[pathlib.Path] = []

    for value in env_candidates:
        if value:
            discovered.append(pathlib.Path(value))

    for base in [script_path.parent, *script_path.parents]:
        discovered.append(base / "scheduling_theory" / "extracted" / "haskell")
        discovered.append(base / "rocq" / "scheduling_theory" / "extracted" / "haskell")

    unique: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for candidate in discovered:
        resolved = candidate.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def resolve_checker_dir(explicit: pathlib.Path | None) -> pathlib.Path:
    candidates = [explicit] if explicit is not None else candidate_checker_dirs()

    for candidate in candidates:
        if candidate is None:
            continue
        module_path = candidate / "AwkernelWorkloadAcceptance.hs"
        if module_path.is_file():
            return candidate

    searched = "\n".join(str(c) for c in candidate_checker_dirs())
    raise AcceptanceError(
        "checker-module-not-found",
        "extracted Haskell workload checker module not found. "
        "Pass --checker-dir or set WORKLOAD_ACCEPT_CHECKER_DIR.\n"
        f"Searched:\n{searched}",
    )


def emit_diagnostic(
    *,
    accepted: bool,
    backend: str,
    scenario: str | None,
    kind: str,
    message: str,
    sched_trace_index: int | None = None,
    task_trace_index: int | None = None,
    log_line_begin: int | None = None,
    log_line_end: int | None = None,
) -> None:
    payload = {
        "accepted": accepted,
        "backend": backend,
        "scenario": scenario,
        "kind": kind,
        "message": message,
        "sched_trace_index": sched_trace_index,
        "task_trace_index": task_trace_index,
        "log_line_begin": log_line_begin,
        "log_line_end": log_line_end,
    }
    print(json.dumps(payload, ensure_ascii=True))
    stream = sys.stderr
    status = "accepted" if accepted else "rejected"
    print(f"{backend}{'' if scenario is None else f'-{scenario}'}: {status}: {message}", file=stream)


def make_internal_checker_error(message: str) -> AcceptanceError:
    return AcceptanceError("internal-checker-error", message)


def parse_runner_payload(stdout: str) -> dict[str, Any]:
    nonempty_lines = [line for line in stdout.splitlines() if line.strip()]
    if len(nonempty_lines) != 1:
        raise make_internal_checker_error(
            f"runner must emit exactly one non-empty JSON line on stdout, found {len(nonempty_lines)}"
        )

    try:
        payload = json.loads(nonempty_lines[0])
    except json.JSONDecodeError as exc:
        raise make_internal_checker_error(f"runner emitted malformed JSON diagnostics: {exc}") from exc

    if not isinstance(payload, dict):
        raise make_internal_checker_error("runner diagnostics payload must be a JSON object")

    actual_keys = set(payload.keys())
    if actual_keys != EXPECTED_DIAGNOSTIC_KEYS:
        missing = sorted(EXPECTED_DIAGNOSTIC_KEYS - actual_keys)
        extra = sorted(actual_keys - EXPECTED_DIAGNOSTIC_KEYS)
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if extra:
            details.append(f"extra keys: {extra}")
        raise make_internal_checker_error(
            "runner diagnostics payload has the wrong key set"
            + ("" if not details else f" ({'; '.join(details)})")
        )

    if not isinstance(payload["accepted"], bool):
        raise make_internal_checker_error("runner diagnostics field 'accepted' must be boolean")
    for key in ["backend", "kind", "message"]:
        if not isinstance(payload[key], str):
            raise make_internal_checker_error(f"runner diagnostics field '{key}' must be a string")
    if payload["scenario"] is not None and not isinstance(payload["scenario"], str):
        raise make_internal_checker_error("runner diagnostics field 'scenario' must be null or a string")
    for key in ["sched_trace_index", "task_trace_index", "log_line_begin", "log_line_end"]:
        if payload[key] is not None and not isinstance(payload[key], int):
            raise make_internal_checker_error(f"runner diagnostics field '{key}' must be null or an integer")

    return payload


def normalized_log_line(
    block_start: int,
    index: int | None,
) -> tuple[int | None, int | None]:
    if index is None:
        return None, None
    line = block_start + index
    return line, line


def normalize_runner_payload(
    *,
    payload: dict[str, Any],
    backend: str,
    scenario: str | None,
    sched_trace_start_line: int,
    task_trace_start_line: int,
    returncode: int,
) -> dict[str, Any]:
    accepted = payload["accepted"]
    if returncode not in (ACCEPTED_EXIT, RUNNER_FAILURE_EXIT):
        raise make_internal_checker_error(f"runner returned unexpected exit code: {returncode}")
    if accepted and returncode != ACCEPTED_EXIT:
        raise make_internal_checker_error("runner reported accepted=true but returned a failure exit code")
    if (not accepted) and returncode != RUNNER_FAILURE_EXIT:
        raise make_internal_checker_error("runner reported accepted=false but did not return the runner failure exit code")

    kind = payload["kind"]
    sched_trace_index = payload["sched_trace_index"]
    task_trace_index = payload["task_trace_index"]
    log_line_begin = payload["log_line_begin"]
    log_line_end = payload["log_line_end"]

    if accepted:
        if kind != "accepted":
            raise make_internal_checker_error("runner success payload must use kind='accepted'")
        if any(value is not None for value in [sched_trace_index, task_trace_index, log_line_begin, log_line_end]):
            raise make_internal_checker_error("runner success payload must leave all location fields null")
    elif kind in {"sched-trace-parse-failure"}:
        if sched_trace_index is None or task_trace_index is not None:
            raise make_internal_checker_error("sched-trace-parse-failure must carry only sched_trace_index")
        log_line_begin, log_line_end = normalized_log_line(sched_trace_start_line, sched_trace_index)
    elif kind in {"task-trace-parse-failure"}:
        if task_trace_index is None or sched_trace_index is not None:
            raise make_internal_checker_error("task-trace-parse-failure must carry only task_trace_index")
        log_line_begin, log_line_end = normalized_log_line(task_trace_start_line, task_trace_index)
    elif kind == "workload-family-rejection":
        if any(value is not None for value in [sched_trace_index, task_trace_index, log_line_begin, log_line_end]):
            raise make_internal_checker_error("workload-family-rejection must leave all location fields null")
    elif kind == "internal-checker-error":
        if any(value is not None for value in [sched_trace_index, task_trace_index, log_line_begin, log_line_end]):
            raise make_internal_checker_error("internal-checker-error must leave all location fields null")
    else:
        raise make_internal_checker_error(f"runner emitted unsupported diagnostics kind: {kind}")

    return {
        "accepted": accepted,
        "backend": backend,
        "scenario": scenario,
        "kind": kind,
        "message": payload["message"],
        "sched_trace_index": sched_trace_index,
        "task_trace_index": task_trace_index,
        "log_line_begin": log_line_begin,
        "log_line_end": log_line_end,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the workload task_trace+sched_trace acceptance gate on a captured serial log."
    )
    parser.add_argument("--log", type=pathlib.Path, required=True, help="Path to the captured serial log.")
    parser.add_argument("--backend", default="backend", help="Backend label for diagnostics.")
    parser.add_argument("--scenario", help="Optional runtime workload label for diagnostics.")
    parser.add_argument("--runhaskell", default="runhaskell", help="Path or command name for runhaskell.")
    parser.add_argument("--runner", type=pathlib.Path, required=True, help="Path to the Haskell workload acceptance runner.")
    parser.add_argument("--checker-dir", type=pathlib.Path, help="Directory containing the extracted AwkernelWorkloadAcceptance module.")
    args = parser.parse_args()

    try:
        runhaskell = resolve_runhaskell(args.runhaskell)
        if not args.runner.is_file():
            raise AcceptanceError("runner-not-found", f"Haskell runner not found: {args.runner}")
        checker_dir = resolve_checker_dir(args.checker_dir)
        lines = load_lines(args.log)
        sched_trace, sched_trace_start_line, _ = extract_block(
            lines,
            SCHED_TRACE_BEGIN,
            SCHED_TRACE_END,
            missing_kind="missing-sched-trace-block",
            empty_kind="empty-sched-trace-block",
            empty_message="sched_trace block is empty",
        )
        task_trace, task_trace_start_line, _ = extract_block(
            lines,
            TASK_TRACE_BEGIN,
            TASK_TRACE_END,
            missing_kind="missing-task-trace-block",
            empty_kind="empty-task-trace-block",
            empty_message="task_trace block is empty",
        )
    except AcceptanceError as exc:
        emit_diagnostic(
            accepted=False,
            backend=args.backend,
            scenario=args.scenario,
            kind=exc.kind,
            message=exc.message,
            log_line_begin=exc.log_line_begin,
            log_line_end=exc.log_line_end,
        )
        return WRAPPER_FAILURE_EXIT

    with tempfile.TemporaryDirectory(prefix="awkernel-workload-accept-") as tmpdir:
        tmpdir_path = pathlib.Path(tmpdir)
        sched_trace_path = tmpdir_path / "sched_trace.tsv"
        task_trace_path = tmpdir_path / "task_trace.tsv"
        sched_trace_path.write_text("\n".join(sched_trace) + "\n", encoding="utf-8")
        task_trace_path.write_text("\n".join(task_trace) + "\n", encoding="utf-8")

        cmd = [
            runhaskell,
            f"-i{checker_dir}",
            str(args.runner),
            args.backend,
            args.scenario or "-",
            str(sched_trace_path),
            str(task_trace_path),
        ]
        result = subprocess.run(cmd, text=True, capture_output=True)

        try:
            payload = parse_runner_payload(result.stdout)
            normalized = normalize_runner_payload(
                payload=payload,
                backend=args.backend,
                scenario=args.scenario,
                sched_trace_start_line=sched_trace_start_line,
                task_trace_start_line=task_trace_start_line,
                returncode=result.returncode,
            )
        except AcceptanceError as exc:
            emit_diagnostic(
                accepted=False,
                backend=args.backend,
                scenario=args.scenario,
                kind=exc.kind,
                message=exc.message,
                log_line_begin=exc.log_line_begin,
                log_line_end=exc.log_line_end,
            )
            return WRAPPER_FAILURE_EXIT

    emit_diagnostic(**normalized)
    return ACCEPTED_EXIT if normalized["accepted"] else RUNNER_FAILURE_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
