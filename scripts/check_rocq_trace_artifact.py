#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import sys


BEGIN_MARKER = "BEGIN_ROCQ_TRACE"
END_MARKER = "END_ROCQ_TRACE"


def load_lines(path: pathlib.Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def extract_rocq_block(lines: list[str]) -> list[str]:
    begin_indices = [i for i, line in enumerate(lines) if line.strip() == BEGIN_MARKER]
    end_indices = [i for i, line in enumerate(lines) if line.strip() == END_MARKER]

    if len(begin_indices) != 1:
        raise SystemExit(f"expected exactly one {BEGIN_MARKER} marker, found {len(begin_indices)}")
    if len(end_indices) != 1:
        raise SystemExit(f"expected exactly one {END_MARKER} marker, found {len(end_indices)}")

    begin = begin_indices[0]
    end = end_indices[0]
    if not begin < end:
        raise SystemExit("ROCQ trace markers are out of order")

    return [line.rstrip() for line in lines[begin + 1 : end]]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check a Rocq-encoded Awkernel trace artifact against the canonical generated module."
    )
    parser.add_argument("--expected", type=pathlib.Path, required=True, help="Path to the canonical Rocq artifact.")
    parser.add_argument("--log", type=pathlib.Path, required=True, help="Path to the captured serial log.")
    parser.add_argument("--backend", default="backend", help="Backend label for diagnostics.")
    args = parser.parse_args()

    expected = load_lines(args.expected)
    actual = extract_rocq_block(load_lines(args.log))

    if actual != expected:
        print(f"{args.backend}: Rocq trace artifact mismatch", file=sys.stderr)
        print("--- expected ---", file=sys.stderr)
        print("\n".join(expected), file=sys.stderr)
        print("--- actual ---", file=sys.stderr)
        print("\n".join(actual), file=sys.stderr)
        return 1

    print(f"{args.backend}: Rocq trace artifact matches canonical module")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
