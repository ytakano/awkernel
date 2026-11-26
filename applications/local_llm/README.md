# local_llm

Chat with **Qwen3-0.6B** running inside Awkernel (x86_64, CPU only), using
the weight format of [llm.asm](https://github.com/alesha-pro/llm.asm).

```text
> (llm "What is the capital of Japan? Answer in one sentence.")
The capital of Japan is Tokyo.
[llm] prompt 24 tok 0.30 s (80.1 tok/s), answer 7 tok 0.12 s (58.4 tok/s), ctx 31/2048
```

- Engine: [`awkernel_llm`](../../awkernel_llm/) (no_std, AVX2/FMA/F16C or scalar).
- The model is put into the boot image as the bootloader's ramdisk and used
  without copying.
- The service task runs one forward pass per scheduling step; helper tasks on
  other CPUs run the matrix and attention loops in parallel.

## Requirements

- x86_64 host with AVX2, FMA and F16C, and KVM (`/dev/kvm`).
- QEMU 8.x or later, the Awkernel toolchain (see the top-level README).
- Python 3 with `numpy` (only to convert the model once).
- About 9 GiB of RAM for the VM (614 MiB model, 448 MiB KV cache, rest heap).

## 1. Prepare the model

```sh
git clone https://github.com/alesha-pro/llm.asm.git
mkdir -p models/qwen3-0.6b-bf16
for f in config.json tokenizer.json model.safetensors; do
  curl -fL -o models/qwen3-0.6b-bf16/$f \
    https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/$f
done

python3 -m venv venv && venv/bin/pip install numpy
venv/bin/python llm.asm/scripts/export_weights.py \
  --model-dir models/qwen3-0.6b-bf16 \
  --out models/qwen3-0.6b-q8_0.asmbin --scale-dtype fp16
```

Only the `.asmbin` v2 format (`--scale-dtype fp16`) is supported.

## 2. Build

From the repository root:

```sh
make x86_64 RELEASE=1 RAMDISK=/path/to/qwen3-0.6b-q8_0.asmbin \
  OPT="--release --features local_llm"
```

`local_llm_selftest` instead of `local_llm` additionally runs a reference
prompt at boot and compares the tokens with llm.asm
(`local_llm selftest: ... PASS`).

## 3. Run under KVM

```sh
make qemu-x86_64-kvm            # KVM_SMP=5 KVM_MEM=8G by default
```

Wait for `[local_llm] service: ready`, then use the shell:

| Command | |
|---|---|
| `(llm "text")` | Ask a question; the conversation continues across calls |
| `(llm_clear)` | Forget the conversation |
| `(llm_think true)` | Enable Qwen3 reasoning (`<think>…`); `false` disables it |
| Ctrl-C | Stop the current answer (twice: stop waiting for the service) |

Exit QEMU with `Ctrl-A x`.

`KVM_SMP` sets the number of vCPUs. CPU 0 runs no tasks, so with `KVM_SMP=5`
the service uses 4 CPUs (itself and 3 helpers). Measured on a Ryzen 9 5900HX:

| vCPUs | prompt | answer |
|---:|---:|---:|
| 2 | 38 tok/s | 28 tok/s |
| 3 | 67 tok/s | 49 tok/s |
| 5 | 76 tok/s | 59 tok/s |

The answers are identical for any number of vCPUs.

## Host-side checks

```sh
# Engine unit tests and comparison with the llm.asm reference tokens.
AWKERNEL_LLM_MODEL=/path/to/qwen3-0.6b-q8_0.asmbin make test

# Run the engine as a host program.
cargo run --release -p awkernel_llm --features std --example llm_cli -- \
  -m /path/to/qwen3-0.6b-q8_0.asmbin -p "Hello" -t 4
```

## Notes

- The model is small (0.6B) and quantized; answers, especially in languages
  other than English, are often wrong.
- AVX is enabled by the kernel at boot. The preemption interrupt handler
  saves the YMM state of the interrupted task (`test_avx_preempt` checks it:
  `make x86_64 RELEASE=1 OPT="--release --features test_avx_preempt"`).
