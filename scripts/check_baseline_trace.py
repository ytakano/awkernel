#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import sys


def load_lines(path: pathlib.Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def extract_trace(lines: list[str]) -> list[str]:
    extracted = [line.rstrip() for line in lines if line.startswith("BASELINE_TRACE:")]
    done_markers = [line.rstrip() for line in lines if line.strip() == "BASELINE_TRACE_DONE"]

    if not extracted:
        raise SystemExit("no BASELINE_TRACE lines found in log")
    if len(done_markers) != 1:
        raise SystemExit(f"expected exactly one BASELINE_TRACE_DONE marker, found {len(done_markers)}")

    extracted.append("BASELINE_TRACE_DONE")
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a captured Awkernel baseline trace against the canonical fixture.")
    parser.add_argument("--expected", type=pathlib.Path, required=True, help="Path to the canonical baseline-trace fixture.")
    parser.add_argument("--log", type=pathlib.Path, required=True, help="Path to the captured serial log.")
    parser.add_argument("--backend", default="backend", help="Backend label for diagnostics.")
    args = parser.parse_args()

    expected = load_lines(args.expected)
    actual = extract_trace(load_lines(args.log))

    if actual != expected:
        print(f"{args.backend}: baseline trace mismatch", file=sys.stderr)
        print("--- expected ---", file=sys.stderr)
        print("\n".join(expected), file=sys.stderr)
        print("--- actual ---", file=sys.stderr)
        print("\n".join(actual), file=sys.stderr)
        return 1

    print(f"{args.backend}: baseline trace matches canonical fixture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
