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


TRACE_ROWS_BEGIN = "BEGIN_TRACE_ROWS"
TRACE_ROWS_END = "END_TRACE_ROWS"
TASK_LIFECYCLE_BEGIN = "BEGIN_TASK_LIFECYCLE"
TASK_LIFECYCLE_END = "END_TASK_LIFECYCLE"


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
    row_index: int | None = None,
    lifecycle_index: int | None = None,
    log_line_begin: int | None = None,
    log_line_end: int | None = None,
) -> None:
    payload = {
        "accepted": accepted,
        "backend": backend,
        "scenario": scenario,
        "kind": kind,
        "message": message,
        "row_index": row_index,
        "lifecycle_index": lifecycle_index,
        "log_line_begin": log_line_begin,
        "log_line_end": log_line_end,
    }
    print(json.dumps(payload, ensure_ascii=True))
    stream = sys.stderr
    status = "accepted" if accepted else "rejected"
    print(f"{backend}{'' if scenario is None else f'-{scenario}'}: {status}: {message}", file=stream)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the workload lifecycle+rows acceptance gate on a captured serial log."
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
        rows, _, _ = extract_block(
            lines,
            TRACE_ROWS_BEGIN,
            TRACE_ROWS_END,
            missing_kind="missing-rows-block",
            empty_kind="empty-rows-block",
            empty_message="trace rows block is empty",
        )
        lifecycle, _, _ = extract_block(
            lines,
            TASK_LIFECYCLE_BEGIN,
            TASK_LIFECYCLE_END,
            missing_kind="missing-lifecycle-block",
            empty_kind="empty-lifecycle-block",
            empty_message="task lifecycle block is empty",
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
        return 2

    with tempfile.TemporaryDirectory(prefix="awkernel-workload-accept-") as tmpdir:
        tmpdir_path = pathlib.Path(tmpdir)
        rows_path = tmpdir_path / "rows.tsv"
        lifecycle_path = tmpdir_path / "lifecycle.tsv"
        rows_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        lifecycle_path.write_text("\n".join(lifecycle) + "\n", encoding="utf-8")

        cmd = [
            runhaskell,
            f"-i{checker_dir}",
            str(args.runner),
            args.backend,
            args.scenario or "-",
            str(rows_path),
            str(lifecycle_path),
        ]
        result = subprocess.run(cmd, text=True, capture_output=True)

        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
