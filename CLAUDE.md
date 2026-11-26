# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Awkernel is a safe, real-time operating system written in Rust that executes async/await applications in kernel space. It targets x86_64, AArch64 (Raspberry Pi 3/4/5, QEMU virt), RISC-V 32/64, and can also run as a normal process on Linux/macOS (`std` feature).

Run all commands from the repository root (the directory containing this file and the `Makefile`).

## Toolchain

A pinned Rust nightly is required. The pin (currently `nightly-2026-11-09`) appears in `Makefile` (`RUSTV`), `scripts/ci.sh`, `.pre-commit-config.yaml`, `.github/workflows/rust.yml`, `README.md`, and `userland_extern_template/rust-toolchain.toml` — update all of them together. Cross builds use `-Zbuild-std` with the custom target JSONs in `targets/`. Prefer `make` targets over raw cargo so the right toolchain and flags are used.

`Cargo.lock` is not tracked; CI resolves the latest compatible dependencies. If a check fails only locally, try `cargo update`.

## Commands

```sh
make x86_64 RELEASE=1                     # x86_64 UEFI image (x86_64_uefi.img)
make aarch64 BSP=aarch64_virt RELEASE=1   # BSP: raspi3 | raspi4 | raspi5 | aarch64_virt -> kernel8.img
make riscv64 / make riscv32
make std                                  # build as a host process (Linux/macOS)
make run-std                              # run the std build

make qemu-x86_64                          # boot in QEMU (OVMF directory path in ~/.ovmfpath); also qemu-x86_64-net, -nographic
make qemu-aarch64-virt / qemu-raspi3 / qemu-riscv64 / qemu-riscv32
make debug-x86_64 + make gdb-x86_64       # QEMU with gdbstub (-s -S), then attach gdb-multiarch
make monitor                              # telnet to the QEMU monitor (port 5556)

make test                                 # host unit tests
make check                                # cargo check for every target
make clippy                               # clippy for every target
make fmt
make docs                                 # build mdbook/ into docs/
scripts/ci.sh                             # exactly what CI runs (needs cargo-udeps)
```

- Without `RELEASE=1`, builds are debug and add `--features debug`. Real hardware (Raspberry Pi) requires `RELEASE=1`.
- `make check_aarch64` (and `make aarch64`) need `BSP=...`; CI checks all four BSPs.
- CI is strict: `cargo fmt --check`, clippy with `--deny warnings` for every target, `cargo udeps`, and tests with `RUSTFLAGS="-D warnings"`. A change touching arch-gated code should be checked on the other targets too (`make check`, or `scripts/ci.sh` before pushing).
- On a non-x86_64 host, the x86_64 build needs an `x86_64-elf-` GNU toolchain for `kernel/asm/x86/mpboot.S`; pass `CROSS_COMPILE=x86_64-elf-` (a prefix, keep the trailing `-`).
- `README.md` mentions `make loom`, but no such target exists; loom tests live in the external `awkernel_sync` repo.

### Running a single test

Tests run on the host through cargo aliases in `.cargo/config.toml` (they select `--no-default-features --features std`):

```sh
cargo test_awkernel_lib <test_name>
cargo test_awkernel_async_lib <test_name> -- --nocapture
cargo test_awkernel_drivers
cargo test_smoltcp
cargo test_rd_gen_to_dags
```

### Enabling in-kernel applications / tests

Applications are selected by cargo features on the `awkernel_userland` package. The Makefile has no dedicated variable for extra features, so override `OPT` (keep `RELEASE=1` so output paths match):

```sh
make x86_64 RELEASE=1 OPT="--release --features test_pubsub"
make aarch64 BSP=aarch64_virt RELEASE=1 OPT="--release --features test_network"
```

### Cargo aliases

`.cargo/config.toml` defines per-target aliases used by the Makefile: build (`x86`, `aarch64`, `rv32`, `rv64`, `std`), `check_*`, `clippy_*`, `doc_*`, `udeps_*`. Use them with the pinned toolchain, e.g. `cargo +nightly-2026-11-09 check_x86`.

## Architecture

### Crate layout

The final binary is produced by **`userland`** (package `awkernel_userland`, bin `awkernel`), not by `kernel`:

- `userland` — `src/main.rs` marks its async entry point with `#[awkernel::main]`; `src/lib.rs::run()` awaits `awkernel_services::run()` and then each feature-gated application.
- `kernel` (package `awkernel`, a library) — boot assembly (`kernel/asm/`), linker scripts (`kernel/ld/`), arch-specific init (`kernel/src/arch/`), and the per-CPU `main()` in `kernel/src/lib.rs`. `#[awkernel::main]` (from `awkernel_macros`) defines the `__awkernel_user_main` symbol the kernel calls. `kernel/build.rs` assembles startup code and publishes the linker script as `awkernel.ld`.
- `awkernel_lib` — low-level library shared by kernel and async runtime: paging, heap, CPU, interrupts, DMA, device tree, net, console, context switch. Arch code lives in `arch/` and `context/`, selected by features.
- `awkernel_async_lib` — no_std async runtime: task management (`task.rs`, `task/`), schedulers (`scheduler/`), channels, pubsub, DAG execution, preemption.
- `awkernel_lib_verified` — pure-Rust data structures with Kani proof harnesses (delta list, ring queue), re-exported as `awkernel_lib::verified`.
- `awkernel_drivers` — PCIe, UART, interrupt controllers, NICs, Raspberry Pi HAL, etc.
- `applications/` — `awkernel_services`, `awkernel_shell`, `awkernel_display`, `autoware`, `rd_gen_to_dags`, and in-kernel test apps under `applications/tests/*` (each exposes `pub async fn run()`).
- `smoltcp` — vendored TCP/IP stack. `x86bootdisk` — host tool that builds the x86_64 boot image.
- `userland_extern_template/` — template for a userland crate living *outside* this workspace that depends on the `awkernel` crate (not a workspace member; has its own Makefile/toolchain file).

MCS lock and RW lock live in the external `awkernel_sync` repo (github.com/tier4/awkernel_sync).

### Boot and execution model

Arch boot code calls `kernel::main()` once per CPU. CPU 0 (primary) initializes interrupts, spawns the user main as the first task (`SchedulerType::PrioritizedFIFO(31)`), then loops forever handling IRQs, calling `wake_task()` for timed wakeups, and waking workers. All other CPUs are workers that poll tasks. Consequently **tasks never run on CPU 0**; scheduler CPU sets are normalized to exclude it.

Schedulers are chosen per task via `SchedulerType` (`awkernel_async_lib/src/scheduler.rs`): `EDF(deadline, CpuSet)`, `LLFT(deadline, CpuSet)`, `PrioritizedFIFO(prio)`, `PrioritizedRR(prio)`. EDF and LLFT share the clustered scheduling logic in `scheduler/clustered.rs`.

### Feature flags

Platform features (`x86`, `aarch64` + `raspi`/`raspi5`/`aarch64_virt`, `rv32`, `rv64`, `std`) are declared on `awkernel_userland`, forwarded to `awkernel`, and from there fan out to `awkernel_lib`, `awkernel_async_lib`, and `awkernel_drivers`. Exactly one platform must be enabled; nearly all arch-specific code is `#[cfg(feature = ...)]`-gated rather than `target_arch`-gated, which is why host `std` tests cannot catch breakage in other targets.

Adding an application: create the crate under `applications/` (or `applications/tests/`, auto-included in the workspace), add it as an optional dependency plus a feature in `userland/Cargo.toml`, and call its `run()` behind `#[cfg(feature = ...)]` in `userland/src/lib.rs`.

### Formal verification

`specification/` holds TLA+ and SPIN/Promela models mirroring the source tree paths; `SPEC_TEST.md` maps each verified component to its model and tool (loom, Kani, TLA+, SPIN). When changing verified components (e.g. `awkernel_async_lib/src/task.rs`, `awkernel_lib/src/context/`, `kernel/asm/aarch64/exception.S`, `awkernel_lib/src/net/if_net.rs`, `awkernel_async_lib/src/sync/barrier.rs`), check whether the model needs updating.

### Documentation

`mdbook/src/` is the source for the design docs (`internal/` covers boot, context switch, scheduler, page tables, allocators, etc.) — useful background before changing those subsystems. `docs/` is generated output from `make docs`; edit `mdbook/` instead.
