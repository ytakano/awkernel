#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys


BEGIN_MARKER = "BEGIN_TRACE_ROWS"
END_MARKER = "END_TRACE_ROWS"


class AcceptanceError(RuntimeError):
    pass


def load_lines(path: pathlib.Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AcceptanceError(f"failed to read serial log {path}: {exc}") from exc


def extract_trace_rows_block(lines: list[str]) -> list[str]:
    begin_indices = [i for i, line in enumerate(lines) if line.strip() == BEGIN_MARKER]
    end_indices = [i for i, line in enumerate(lines) if line.strip() == END_MARKER]

    if len(begin_indices) != 1:
        raise AcceptanceError(f"expected exactly one {BEGIN_MARKER} marker, found {len(begin_indices)}")
    if len(end_indices) != 1:
        raise AcceptanceError(f"expected exactly one {END_MARKER} marker, found {len(end_indices)}")

    begin = begin_indices[0]
    end = end_indices[0]
    if not begin < end:
        raise AcceptanceError("trace rows markers are out of order")

    rows = [line.rstrip() for line in lines[begin + 1 : end]]
    if not rows:
        raise AcceptanceError("trace rows block is empty")
    return rows


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


def resolve_checker_bin(path: pathlib.Path) -> str:
    if not path.is_file():
        raise AcceptanceError(f"Haskell checker binary not found: {path}")
    if not os.access(path, os.X_OK):
        raise AcceptanceError(f"Haskell checker binary is not executable: {path}")
    return str(path)


def candidate_checker_dirs() -> list[pathlib.Path]:
    env_candidates = [
        os.environ.get("HANDOFF_ACCEPT_CHECKER_DIR"),
        os.environ.get("AWKERNEL_HANDOFF_CHECKER_DIR"),
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
        module_path = candidate / "AwkernelHandoffAcceptance.hs"
        if module_path.is_file():
            return candidate

    searched = "\n".join(str(c) for c in candidate_checker_dirs())
    raise AcceptanceError(
        "extracted Haskell checker module not found. "
        "Pass --checker-dir or set HANDOFF_ACCEPT_CHECKER_DIR.\n"
        f"Searched:\n{searched}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the phase-2 neutral-rows acceptance gate on a captured neutral trace-rows block."
    )
    parser.add_argument("--log", type=pathlib.Path, required=True, help="Path to the captured serial log.")
    parser.add_argument("--backend", default="backend", help="Backend label for diagnostics.")
    parser.add_argument(
        "--runhaskell",
        default="runhaskell",
        help="Path or command name for runhaskell.",
    )
    parser.add_argument(
        "--runner",
        type=pathlib.Path,
        default=pathlib.Path("scripts/haskell/HandoffAcceptanceMain.hs"),
        help="Path to the Haskell phase-2 acceptance runner used by the runhaskell fallback.",
    )
    parser.add_argument(
        "--checker-dir",
        type=pathlib.Path,
        help="Directory containing the extracted AwkernelHandoffAcceptance module for the phase-2 gate.",
    )
    parser.add_argument("--checker-bin", type=pathlib.Path, help="Native handoff acceptance checker binary.")
    args = parser.parse_args()

    try:
        checker_bin = resolve_checker_bin(args.checker_bin) if args.checker_bin is not None else None
        runhaskell = None
        checker_dir = None
        if checker_bin is None:
            runhaskell = resolve_runhaskell(args.runhaskell)
            if not args.runner.is_file():
                raise AcceptanceError(f"Haskell runner not found: {args.runner}")
            checker_dir = resolve_checker_dir(args.checker_dir)
        rows = extract_trace_rows_block(load_lines(args.log))
    except AcceptanceError as exc:
        raise SystemExit(f"{args.backend}: {exc}") from exc

    payload = "\n".join(rows) + "\n"
    if checker_bin is not None:
        cmd = [
            checker_bin,
            args.backend,
        ]
    else:
        cmd = [
            runhaskell,
            f"-i{checker_dir}",
            str(args.runner),
            args.backend,
        ]
    result = subprocess.run(cmd, input=payload, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        stderr = result.stderr
        if "failed to parse trace rows" in stderr:
            raise SystemExit(f"{args.backend}: failed to parse extracted trace rows")
        if "acceptance checker rejected trace rows" in stderr:
            raise SystemExit(f"{args.backend}: phase-2 acceptance rejected trace rows")
        raise SystemExit(f"{args.backend}: phase-2 acceptance checker exited with status {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
