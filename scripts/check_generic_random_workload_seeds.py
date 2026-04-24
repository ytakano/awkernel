#!/usr/bin/env python3
"""Run generic_random workload acceptance repeatedly with different seeds."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


MASK64 = (1 << 64) - 1
DEFAULT_START_SEED = 0x4D59_5DF4_D0F3_3173
TRACE_DONE_MARKER = "END_ROCQ_TRACE"


@dataclass(frozen=True)
class RunResult:
    code: int
    log_path: Path | None = None


def splitmix64(value: int) -> int:
    value = (value + 0x9E37_79B9_7F4A_7C15) & MASK64
    mixed = value
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58_476D_1CE4_E5B9) & MASK64
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D0_49BB_1331_11EB) & MASK64
    return (mixed ^ (mixed >> 31)) & MASK64


def parse_seed(raw: str) -> int:
    try:
        return int(raw, 0) & MASK64
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid seed: {raw}") from exc


def positive_int(raw: str) -> int:
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid run count: {raw}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("run count must be at least 1")
    return value


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and check generic_random workload traces with a deterministic "
            "sequence of different GENERIC_TRACE_SEED values."
        )
    )
    parser.add_argument("max_runs", type=positive_int, help="maximum number of seeds to test")
    parser.add_argument(
        "--start-seed",
        type=parse_seed,
        default=DEFAULT_START_SEED,
        help=f"base seed for the deterministic sequence, default 0x{DEFAULT_START_SEED:016x}",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=parse_seed,
        default=[],
        help="exact seed to test; may be passed multiple times and is capped by max_runs",
    )
    parser.add_argument(
        "--ovmf-path",
        default="target/ovmf/x64",
        help="OVMF_PATH value passed to make, default target/ovmf/x64",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="seconds to wait for one QEMU run, default 120",
    )
    parser.add_argument(
        "--make",
        default="make",
        help="make command to execute, default make",
    )
    parser.add_argument(
        "--qemu",
        default="qemu-system-x86_64",
        help="QEMU command to execute, default qemu-system-x86_64",
    )
    parser.add_argument(
        "--runhaskell",
        default="runhaskell",
        help="runhaskell command passed to the workload checker, default runhaskell",
    )
    parser.add_argument(
        "--runner",
        default="scripts/haskell/WorkloadAcceptanceMain.hs",
        help="Haskell workload checker runner, default scripts/haskell/WorkloadAcceptanceMain.hs",
    )
    return parser


def wait_for_trace_dump(log_path: Path, process: subprocess.Popen[object], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    offset = 0

    while time.monotonic() < deadline:
        if log_path.exists():
            with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
                log_file.seek(offset)
                chunk = log_file.read()
                offset = log_file.tell()
            if TRACE_DONE_MARKER in chunk:
                return True

        if process.poll() is not None:
            if log_path.exists() and TRACE_DONE_MARKER in log_path.read_text(
                encoding="utf-8", errors="replace"
            ):
                return True
            return False

        time.sleep(0.1)

    return False


def stop_qemu(process: subprocess.Popen[object]) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def build_image(args: argparse.Namespace, seed_arg: str) -> int:
    command = [
        args.make,
        "build-workload-trace-x86_64",
        "WORKLOAD_SCENARIO=generic_random",
        f"GENERIC_TRACE_SEED={seed_arg}",
    ]
    return subprocess.run(command, cwd=repo_root()).returncode


def capture_qemu_log(args: argparse.Namespace, seed_arg: str, index: int) -> tuple[int, Path]:
    root = repo_root()
    ovmf_path = root / args.ovmf_path
    log_path = Path(f"/tmp/awkernel_qemu_2cpu_generic_random_{index}_{seed_arg[2:]}.log")

    shutil.copyfile(ovmf_path / "vars.fd", ovmf_path / "vars_qemu.fd")
    log_path.unlink(missing_ok=True)

    command = [
        args.qemu,
        "-drive",
        f"if=pflash,format=raw,readonly=on,file={ovmf_path / 'code.fd'}",
        "-drive",
        f"if=pflash,format=raw,file={ovmf_path / 'vars_qemu.fd'}",
        "-drive",
        "format=raw,file=x86_64_uefi.img",
        "-machine",
        "q35",
        "-chardev",
        f"stdio,id=workload_serial,signal=off,logfile={log_path},logappend=off",
        "-serial",
        "chardev:workload_serial",
        "-monitor",
        "none",
        "-m",
        "2G",
        "-smp",
        "2",
        "-nographic",
    ]

    process = subprocess.Popen(
        command,
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    dump_completed = wait_for_trace_dump(log_path, process, args.timeout)
    stop_qemu(process)

    if not dump_completed:
        return 124, log_path
    return 0, log_path


def check_log(args: argparse.Namespace, log_path: Path) -> int:
    command = [
        "python3",
        "scripts/check_workload_acceptance.py",
        "--backend",
        "qemu-workload",
        "--scenario",
        "generic_random",
        "--log",
        str(log_path),
        "--runhaskell",
        args.runhaskell,
        "--runner",
        args.runner,
    ]
    return subprocess.run(command, cwd=repo_root()).returncode


def print_trace_log(log_path: Path | None) -> None:
    if log_path is None:
        return

    print(f"generic_random trace log: {log_path}", file=sys.stderr)
    if not log_path.exists():
        print(f"trace log not found: {log_path}", file=sys.stderr)
        return

    print("----- BEGIN GENERIC_RANDOM TRACE LOG -----", file=sys.stderr)
    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            print(line, end="", file=sys.stderr)
    print("----- END GENERIC_RANDOM TRACE LOG -----", file=sys.stderr)


def run_one(args: argparse.Namespace, seed: int, index: int) -> RunResult:
    seed_arg = f"0x{seed:016x}"
    print(f"[{index + 1}/{args.max_runs}] GENERIC_TRACE_SEED={seed_arg}", flush=True)

    code = build_image(args, seed_arg)
    if code != 0:
        return RunResult(code)

    code, log_path = capture_qemu_log(args, seed_arg, index)
    if code != 0:
        print(
            f"QEMU run did not emit {TRACE_DONE_MARKER} within {args.timeout}s; log={log_path}",
            file=sys.stderr,
        )
        return RunResult(code, log_path)

    return RunResult(check_log(args, log_path), log_path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    failures: list[tuple[int, int, RunResult]] = []
    seeds = args.seed[: args.max_runs]

    for index in range(len(seeds), args.max_runs):
        seeds.append(splitmix64(args.start_seed + index))

    for index, seed in enumerate(seeds):
        result = run_one(args, seed, index)
        if result.code != 0:
            failures.append((index, seed, result))
            break

    if failures:
        index, seed, result = failures[0]
        print(
            f"generic_random seed run failed at index {index} "
            f"with seed 0x{seed:016x}, exit code {result.code}",
            file=sys.stderr,
        )
        print_trace_log(result.log_path)
        return result.code

    print(f"generic_random seed runs accepted: {args.max_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
