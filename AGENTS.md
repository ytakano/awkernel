# Rust specialist guidance

You are in awkernel/.

Primary responsibility:
- implementation feasibility
- low-disruption integration
- trace hooks
- test strategy
- runtime risk
- API deltas

What to optimize for:
- minimal runtime disruption
- cheap observables
- concrete hook locations
- realistic verification hooks

Required output structure:
1. Implementation plan
2. Observable events available now
3. Interface delta
4. Test and trace plan
5. Risks for the Rocq design

Rules:
- Preserve runtime structure unless proof obligations force interface changes.
- Classify observables as:
  - already available
  - easy to add
  - expensive to add
  - better replaced by a cheaper proxy
- Do not expand proof-facing abstraction casually.

# Repository Guidelines

## Build, Test, and Development Commands
Use Rust `nightly-2025-11-16`; CI and the Makefile assume it.

- `make x86_64 RELEASE=1`: build the x86_64 UEFI image.
- `make aarch64 BSP=aarch64_virt RELEASE=1`: build the AArch64 QEMU target.
- `make check`: run cross-target `cargo check` for x86_64, AArch64, RISC-V, and `std`.
- `make test`: run unit tests for `awkernel_lib`, `awkernel_async_lib`, `awkernel_drivers`, `smoltcp`, and `rd_gen_to_dags`.
- `make clippy`: lint all supported targets; CI treats warnings as errors.
- `make fmt`: format the workspace with `cargo fmt`.
- `make qemu-x86_64` or `make qemu-aarch64-virt`: boot local images in QEMU.

## Agent Runbook
For x86_64 power-control work, always build with `make x86_64 RELEASE=1`. In this environment, graphical QEMU is not the reference path; use `make qemu-x86_64_nographic` and verify from the serial shell with `(shutdown)` and `(reboot)`.

Before checking non-x86 architectures after x86_64 work, always run `make clean`. Then run `make check_aarch64 BSP=aarch64_virt`, `make check_riscv32`, `make check_riscv64`, and `make check_std` in that order.

## Coding Style & Naming Conventions
Follow standard Rust formatting with `cargo fmt`; use 4-space indentation and idiomatic Rust naming: `snake_case` for functions/modules, `CamelCase` for types, `SCREAMING_SNAKE_CASE` for constants. Keep architecture- or BSP-specific code in the matching arch subtree instead of mixing target conditionals into unrelated modules. Prefer small, focused modules that match existing crate boundaries.

## Testing Guidelines
Run `make test` before opening a PR, and run the narrowest target-specific check for the area you changed when possible, such as `make check_x86_64` or `cargo test_awkernel_lib`. Changes touching verified or specified components should update the corresponding assets in `specification/` or `awkernel_async_lib_verified/`. Add tests near the affected crate; existing test app crates under `applications/tests/` are the pattern for integration-style coverage.

## Commit & Pull Request Guidelines
Recent history uses short Conventional Commit-style subjects such as `fix: ...`, `feat(error): ...`, and `fix(ixgbe): ...`. Keep subjects imperative and scoped when helpful. PRs should describe the affected architecture or crate, list the commands you ran, and link the relevant issue or PR. Include boot logs or screenshots only when a UI, QEMU run, or hardware-visible behavior changed.
