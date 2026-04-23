#!/usr/bin/env python3

from __future__ import annotations

import argparse
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
    pass


def load_lines(path: pathlib.Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AcceptanceError(f"failed to read serial log {path}: {exc}") from exc


def extract_block(lines: list[str], begin: str, end: str, empty_message: str) -> list[str]:
    begin_indices = [i for i, line in enumerate(lines) if line.strip() == begin]
    end_indices = [i for i, line in enumerate(lines) if line.strip() == end]

    if len(begin_indices) != 1:
        raise AcceptanceError(f"expected exactly one {begin} marker, found {len(begin_indices)}")
    if len(end_indices) != 1:
        raise AcceptanceError(f"expected exactly one {end} marker, found {len(end_indices)}")

    begin_idx = begin_indices[0]
    end_idx = end_indices[0]
    if not begin_idx < end_idx:
        raise AcceptanceError(f"{begin} and {end} markers are out of order")

    block = [line.rstrip() for line in lines[begin_idx + 1 : end_idx]]
    if not block:
        raise AcceptanceError(empty_message)
    return block


def resolve_runhaskell(command: str) -> str:
    if "/" in command:
        path = pathlib.Path(command)
        if not path.is_file():
            raise AcceptanceError(f"runhaskell not found: {path}")
        return str(path)

    resolved = shutil.which(command)
    if resolved is None:
        raise AcceptanceError(f"runhaskell not found in PATH: {command}")
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
        "extracted Haskell workload checker module not found. "
        "Pass --checker-dir or set WORKLOAD_ACCEPT_CHECKER_DIR.\n"
        f"Searched:\n{searched}"
    )


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
    parser.add_argument("--candidate-runner", type=pathlib.Path, help="Optional Haskell runner that generates and validates candidate tables from accepted rows.")
    parser.add_argument("--candidate-output", type=pathlib.Path, help="Optional output path for the generated candidate_table.v. If omitted, a temporary file is used.")
    args = parser.parse_args()

    label = args.backend if not args.scenario else f"{args.backend}-{args.scenario}"

    try:
        runhaskell = resolve_runhaskell(args.runhaskell)
        if not args.runner.is_file():
            raise AcceptanceError(f"Haskell runner not found: {args.runner}")
        checker_dir = resolve_checker_dir(args.checker_dir)
        lines = load_lines(args.log)
        rows = extract_block(lines, TRACE_ROWS_BEGIN, TRACE_ROWS_END, "trace rows block is empty")
        lifecycle = extract_block(
            lines,
            TASK_LIFECYCLE_BEGIN,
            TASK_LIFECYCLE_END,
            "task lifecycle block is empty",
        )
    except AcceptanceError as exc:
        raise SystemExit(f"{label}: {exc}") from exc

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
            label,
            str(rows_path),
            str(lifecycle_path),
        ]
        result = subprocess.run(cmd, text=True, capture_output=True)

        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)

        if result.returncode != 0:
            stderr = result.stderr
            if "failed to parse trace rows" in stderr:
                raise SystemExit(f"{label}: failed to parse extracted trace rows")
            if "failed to parse task lifecycle" in stderr:
                raise SystemExit(f"{label}: failed to parse extracted task lifecycle")
            if "acceptance checker rejected workload trace" in stderr:
                raise SystemExit(f"{label}: workload acceptance rejected emitted lifecycle/rows trace")
            raise SystemExit(f"{label}: workload acceptance checker exited with status {result.returncode}")

        if args.candidate_runner is not None:
            candidate_output = args.candidate_output or (tmpdir_path / "candidate_table.v")
            candidate_cmd = [
                runhaskell,
                f"-i{checker_dir}",
                str(args.candidate_runner),
                label,
                str(rows_path),
                str(candidate_output),
            ]
            candidate_result = subprocess.run(candidate_cmd, text=True, capture_output=True)

            if candidate_result.stdout:
                print(candidate_result.stdout, end="")
            if candidate_result.stderr:
                print(candidate_result.stderr, end="", file=sys.stderr)

            if candidate_result.returncode != 0:
                stderr = candidate_result.stderr
                if "failed to parse trace rows" in stderr:
                    raise SystemExit(f"{label}: failed to parse extracted trace rows for candidate-table generation")
                if "candidate-table sanity check failed" in stderr:
                    raise SystemExit(f"{label}: candidate-table check rejected the accepted trace rows")
                raise SystemExit(f"{label}: candidate-table generator exited with status {candidate_result.returncode}")
            if not candidate_output.is_file():
                raise SystemExit(f"{label}: candidate-table generator did not create {candidate_output}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
