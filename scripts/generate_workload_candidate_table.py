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


class CandidateTableError(RuntimeError):
    pass


def load_lines(path: pathlib.Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CandidateTableError(f"failed to read serial log {path}: {exc}") from exc


def extract_block(lines: list[str], begin: str, end: str, empty_message: str) -> list[str]:
    begin_indices = [i for i, line in enumerate(lines) if line.strip() == begin]
    end_indices = [i for i, line in enumerate(lines) if line.strip() == end]

    if len(begin_indices) != 1:
        raise CandidateTableError(f"expected exactly one {begin} marker, found {len(begin_indices)}")
    if len(end_indices) != 1:
        raise CandidateTableError(f"expected exactly one {end} marker, found {len(end_indices)}")

    begin_idx = begin_indices[0]
    end_idx = end_indices[0]
    if not begin_idx < end_idx:
        raise CandidateTableError(f"{begin} and {end} markers are out of order")

    block = [line.rstrip() for line in lines[begin_idx + 1 : end_idx]]
    if not block:
        raise CandidateTableError(empty_message)
    return block


def resolve_runhaskell(command: str) -> str:
    if "/" in command:
        path = pathlib.Path(command)
        if not path.is_file():
            raise CandidateTableError(f"runhaskell not found: {path}")
        return str(path)

    resolved = shutil.which(command)
    if resolved is None:
        raise CandidateTableError(f"runhaskell not found in PATH: {command}")
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
    raise CandidateTableError(
        "extracted Haskell workload checker module not found. "
        "Pass --checker-dir or set WORKLOAD_ACCEPT_CHECKER_DIR.\n"
        f"Searched:\n{searched}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a candidate_table.v artifact from the trace rows block of a captured workload serial log."
    )
    parser.add_argument("--log", type=pathlib.Path, required=True, help="Path to the captured serial log.")
    parser.add_argument("--backend", default="backend", help="Backend label for diagnostics.")
    parser.add_argument("--scenario", help="Optional runtime workload label for diagnostics.")
    parser.add_argument("--runhaskell", default="runhaskell", help="Path or command name for runhaskell.")
    parser.add_argument("--runner", type=pathlib.Path, required=True, help="Path to the Haskell candidate-table runner.")
    parser.add_argument("--checker-dir", type=pathlib.Path, help="Directory containing the extracted AwkernelWorkloadAcceptance module.")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Path to the generated candidate_table.v.")
    args = parser.parse_args()

    label = args.backend if not args.scenario else f"{args.backend}-{args.scenario}"

    try:
        runhaskell = resolve_runhaskell(args.runhaskell)
        if not args.runner.is_file():
            raise CandidateTableError(f"Haskell runner not found: {args.runner}")
        checker_dir = resolve_checker_dir(args.checker_dir)
        lines = load_lines(args.log)
        rows = extract_block(lines, TRACE_ROWS_BEGIN, TRACE_ROWS_END, "trace rows block is empty")
    except CandidateTableError as exc:
        raise SystemExit(f"{label}: {exc}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="awkernel-workload-candidate-table-") as tmpdir:
        tmpdir_path = pathlib.Path(tmpdir)
        rows_path = tmpdir_path / "rows.tsv"
        rows_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        cmd = [
            runhaskell,
            f"-i{checker_dir}",
            str(args.runner),
            label,
            str(rows_path),
            str(args.output),
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
        if "candidate-table sanity check failed" in stderr:
            raise SystemExit(f"{label}: generated candidate table failed the local sanity check")
        raise SystemExit(f"{label}: candidate-table generator exited with status {result.returncode}")

    if not args.output.is_file():
        raise SystemExit(f"{label}: candidate-table generator did not create {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
