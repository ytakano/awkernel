#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib


def load_lines(path: pathlib.Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def extract_baseline(lines: list[str]) -> list[str]:
    extracted = [line.rstrip() for line in lines if line.startswith("BASELINE_TRACE:")]
    done_markers = [line.rstrip() for line in lines if line.strip() == "BASELINE_TRACE_DONE"]

    if not extracted:
        raise SystemExit("no BASELINE_TRACE lines found in log")
    if len(done_markers) != 1:
        raise SystemExit(f"expected exactly one BASELINE_TRACE_DONE marker, found {len(done_markers)}")

    extracted.append("BASELINE_TRACE_DONE")
    return extracted


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
        description="Extract a canonical Awkernel trace artifact from a captured serial log."
    )
    parser.add_argument("--mode", choices=("baseline", "block"), required=True)
    parser.add_argument("--log", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--begin")
    parser.add_argument("--end")
    args = parser.parse_args()

    lines = load_lines(args.log)
    if args.mode == "baseline":
        artifact = extract_baseline(lines)
    else:
        if args.begin is None or args.end is None:
            raise SystemExit("--begin and --end are required in block mode")
        artifact = extract_block(lines, args.begin, args.end)

    text = "\n".join(artifact) + "\n"
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
