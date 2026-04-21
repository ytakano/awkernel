#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


BEGIN_MARKER = "BEGIN_TRACE_ROWS"
END_MARKER = "END_TRACE_ROWS"


def load_lines(path: pathlib.Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def extract_trace_rows_block(lines: list[str]) -> list[str]:
    begin_indices = [i for i, line in enumerate(lines) if line.strip() == BEGIN_MARKER]
    end_indices = [i for i, line in enumerate(lines) if line.strip() == END_MARKER]

    if len(begin_indices) != 1:
        raise SystemExit(f"expected exactly one {BEGIN_MARKER} marker, found {len(begin_indices)}")
    if len(end_indices) != 1:
        raise SystemExit(f"expected exactly one {END_MARKER} marker, found {len(end_indices)}")

    begin = begin_indices[0]
    end = end_indices[0]
    if not begin < end:
        raise SystemExit("trace rows markers are out of order")

    rows = [line.rstrip() for line in lines[begin + 1 : end]]
    if not rows:
        raise SystemExit("trace rows block is empty")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the extracted Haskell handoff acceptance checker on a captured neutral trace-rows block."
    )
    parser.add_argument("--log", type=pathlib.Path, required=True, help="Path to the captured serial log.")
    parser.add_argument("--backend", default="backend", help="Backend label for diagnostics.")
    parser.add_argument(
        "--runhaskell",
        type=pathlib.Path,
        default=pathlib.Path("/home/ytakano/.ghcup/bin/runhaskell"),
        help="Path to runhaskell.",
    )
    parser.add_argument(
        "--runner",
        type=pathlib.Path,
        required=True,
        help="Path to the Haskell acceptance runner.",
    )
    parser.add_argument(
        "--checker-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing the extracted AwkernelHandoffAcceptance module.",
    )
    args = parser.parse_args()

    if not args.runhaskell.is_file():
        raise SystemExit(f"runhaskell not found: {args.runhaskell}")
    if not args.runner.is_file():
        raise SystemExit(f"Haskell runner not found: {args.runner}")
    if not (args.checker_dir / "AwkernelHandoffAcceptance.hs").is_file():
        raise SystemExit(
            f"extracted Haskell checker module not found: {args.checker_dir / 'AwkernelHandoffAcceptance.hs'}"
        )

    rows = extract_trace_rows_block(load_lines(args.log))
    payload = "\n".join(rows) + "\n"
    cmd = [
        str(args.runhaskell),
        f"-i{args.checker_dir}",
        str(args.runner),
        args.backend,
    ]
    result = subprocess.run(cmd, input=payload, text=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
