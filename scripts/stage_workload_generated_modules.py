#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib


def load_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage workload-generated Rocq artifacts into the stable scheduling_theory import path."
    )
    parser.add_argument("--rows-rocq", type=pathlib.Path, required=True, help="Path to the generated workload rows Rocq artifact.")
    parser.add_argument("--candidate-table", type=pathlib.Path, required=True, help="Path to the generated workload candidate table Rocq artifact.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Directory for stable generated Rocq modules.")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "WorkloadTraceArtifact.v").write_text(
        load_text(args.rows_rocq),
        encoding="utf-8",
    )
    (output_dir / "WorkloadCandidateTable.v").write_text(
        load_text(args.candidate_table),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
