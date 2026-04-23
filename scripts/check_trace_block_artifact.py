#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import sys


def load_lines(path: pathlib.Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def extract_block(lines: list[str], begin: str, end: str) -> list[str]:
    begin_indices = [i for i, line in enumerate(lines) if line.strip() == begin]
    end_indices = [i for i, line in enumerate(lines) if line.strip() == end]

    if len(begin_indices) != 1:
        raise SystemExit(f"expected exactly one {begin} marker, found {len(begin_indices)}")
    if len(end_indices) != 1:
        raise SystemExit(f"expected exactly one {end} marker, found {len(end_indices)}")

    begin_idx = begin_indices[0]
    end_idx = end_indices[0]
    if not begin_idx < end_idx:
        raise SystemExit("artifact markers are out of order")

    return [line.rstrip() for line in lines[begin_idx + 1 : end_idx]]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check a marked serial-log block against a canonical fixture."
    )
    parser.add_argument("--expected", type=pathlib.Path, required=True)
    parser.add_argument("--log", type=pathlib.Path, required=True)
    parser.add_argument("--backend", default="backend")
    parser.add_argument("--begin", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--label", default="trace block")
    args = parser.parse_args()

    expected = load_lines(args.expected)
    actual = extract_block(load_lines(args.log), args.begin, args.end)

    if actual != expected:
        print(f"{args.backend}: {args.label} artifact mismatch", file=sys.stderr)
        print("--- expected ---", file=sys.stderr)
        print("\n".join(expected), file=sys.stderr)
        print("--- actual ---", file=sys.stderr)
        print("\n".join(actual), file=sys.stderr)
        return 1

    print(f"{args.backend}: {args.label} artifact matches canonical fixture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
