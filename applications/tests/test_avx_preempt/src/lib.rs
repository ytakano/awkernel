//! Checks that the upper halves of the YMM registers survive preemption by
//! an interrupt (IRQ 255), including resumption on another CPU.
//!
//! Low-priority "victim" tasks, one per worker CPU, load a distinct pattern
//! into YMM0-15, spin without making any call (so the compiler cannot save
//! anything), and compare the registers with the pattern. High-priority
//! "aggressor" tasks wake up periodically, which preempts the victims, and
//! overwrite all YMM registers. Without saving the AVX state in the
//! preemption handler, victims observe corrupted registers.
//!
//! The result is printed as `test_avx_preempt: ... PASS` or `FAIL`.

#![no_std]

extern crate alloc;

/// AVX state handling is specific to the bare-metal x86_64 kernel.
#[cfg(not(all(target_arch = "x86_64", target_os = "none")))]
pub async fn run() {
    log::warn!("test_avx_preempt: only for the x86_64 kernel; skipped");
}

#[cfg(all(target_arch = "x86_64", target_os = "none"))]
pub use imp::run;

#[cfg(all(target_arch = "x86_64", target_os = "none"))]
mod imp {

    use alloc::sync::Arc;
    use awkernel_async_lib::{scheduler::SchedulerType, spawn};
    use core::{
        arch::{
            asm,
            x86_64::{__cpuid_count, _rdtsc},
        },
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    /// Rounds per victim task.
    const ROUNDS: usize = 300;
    /// How long a victim holds its pattern per round.
    const HOLD_US: u64 = 2000;
    /// A gap between two TSC reads inside the hold loop longer than this means
    /// the victim was switched out (preempted) while holding the pattern.
    const GAP_US: u64 = 20;

    #[derive(Default)]
    struct Counters {
        corrupted: AtomicUsize,
        /// Rounds in which the victim was switched out while holding.
        interrupted: AtomicUsize,
        /// Rounds that started and ended on different CPUs.
        migrated: AtomicUsize,
        finished: AtomicUsize,
    }

    pub async fn run() {
        if !awkernel_lib::arch::x86_64::xsave::is_avx_enabled() {
            log::warn!("test_avx_preempt: AVX is not enabled; skipped");
            return;
        }
        // The victims use AVX2 integer instructions.
        if __cpuid_count(7, 0).ebx & (1 << 5) == 0 {
            log::warn!("test_avx_preempt: AVX2 is not supported; skipped");
            return;
        }

        let ticks_per_us = tsc_per_us();
        let workers = awkernel_lib::cpu::num_cpu().saturating_sub(1).max(1);
        let c = Arc::new(Counters::default());
        let preempt_before = awkernel_async_lib::task::get_num_preemption();

        for i in 0..workers {
            let c = c.clone();
            spawn(
                alloc::format!("avx victim {i}").into(),
                async move {
                    for round in 0..ROUNDS {
                        let seed = ((i as u32) << 24) ^ (round as u32).wrapping_mul(0x9e37_79b9);
                        let cpu = awkernel_lib::cpu::cpu_id();
                        // SAFETY: AVX/AVX2 are available on every CPU.
                        let (ok, max_gap) = unsafe { hold_pattern(seed, HOLD_US * ticks_per_us) };
                        if !ok {
                            c.corrupted.fetch_add(1, Ordering::Relaxed);
                        }
                        if max_gap > GAP_US * ticks_per_us {
                            c.interrupted.fetch_add(1, Ordering::Relaxed);
                        }
                        if awkernel_lib::cpu::cpu_id() != cpu {
                            c.migrated.fetch_add(1, Ordering::Relaxed);
                        }
                        awkernel_async_lib::r#yield().await;
                    }
                    c.finished.fetch_add(1, Ordering::Release);
                },
                SchedulerType::PrioritizedFIFO(1),
            )
            .await;
        }

        for i in 0..workers {
            let c = c.clone();
            spawn(
                alloc::format!("avx aggressor {i}").into(),
                async move {
                    while c.finished.load(Ordering::Acquire) < workers {
                        awkernel_async_lib::sleep(Duration::from_micros(700)).await;
                        // SAFETY: AVX is enabled on every CPU.
                        unsafe { clobber_ymm() };
                    }
                },
                SchedulerType::PrioritizedFIFO(20),
            )
            .await;
        }

        while c.finished.load(Ordering::Acquire) < workers {
            awkernel_async_lib::sleep(Duration::from_millis(100)).await;
        }

        let preemptions = awkernel_async_lib::task::get_num_preemption() - preempt_before;
        let corrupted = c.corrupted.load(Ordering::Relaxed);
        let interrupted = c.interrupted.load(Ordering::Relaxed);
        let migrated = c.migrated.load(Ordering::Relaxed);

        // The test is only meaningful if victims were really switched out
        // while holding their pattern, and, with several workers, resumed on
        // another CPU.
        let covered = preemptions > 0 && interrupted > 0 && (workers < 2 || migrated > 0);
        let verdict = match (corrupted == 0, covered) {
            (true, true) => "PASS",
            (true, false) => "INCONCLUSIVE (preemption/migration not exercised)",
            (false, _) => "FAIL",
        };
        log::info!(
            "test_avx_preempt: victims={workers} rounds={} preemptions={preemptions} \
             interrupted={interrupted} migrated={migrated} corrupted={corrupted} {verdict}",
            workers * ROUNDS
        );
    }

    /// TSC ticks per microsecond.
    fn tsc_per_us() -> u64 {
        let t0 = unsafe { _rdtsc() };
        awkernel_lib::delay::wait_microsec(10_000);
        let t1 = unsafe { _rdtsc() };
        ((t1 - t0) / 10_000).max(1)
    }

    /// Loads a pattern derived from `seed` into YMM0-15 (different in every
    /// lane and register), spins for `ticks` TSC ticks, and checks that all 512
    /// bytes are unchanged. Also returns the longest gap between two TSC reads
    /// in the spin loop.
    #[target_feature(enable = "avx2")]
    unsafe fn hold_pattern(seed: u32, ticks: u64) -> (bool, u64) {
        let mut pattern = [0u32; 16 * 8];
        for (j, p) in pattern.iter_mut().enumerate() {
            *p = seed.wrapping_add((j as u32).wrapping_mul(0x0101_0101)) | 1;
        }
        let deadline = unsafe { _rdtsc() } + ticks;
        let mask: u32;
        let max_gap: u64;
        unsafe {
            asm!(
                "vmovdqu ymm0, [{p}]",
                "vmovdqu ymm1, [{p} + 32]",
                "vmovdqu ymm2, [{p} + 64]",
                "vmovdqu ymm3, [{p} + 96]",
                "vmovdqu ymm4, [{p} + 128]",
                "vmovdqu ymm5, [{p} + 160]",
                "vmovdqu ymm6, [{p} + 192]",
                "vmovdqu ymm7, [{p} + 224]",
                "vmovdqu ymm8, [{p} + 256]",
                "vmovdqu ymm9, [{p} + 288]",
                "vmovdqu ymm10, [{p} + 320]",
                "vmovdqu ymm11, [{p} + 352]",
                "vmovdqu ymm12, [{p} + 384]",
                "vmovdqu ymm13, [{p} + 416]",
                "vmovdqu ymm14, [{p} + 448]",
                "vmovdqu ymm15, [{p} + 480]",
                "rdtsc",
                "shl rdx, 32",
                "or rax, rdx",
                "mov {prev}, rax",
                "xor {gap:e}, {gap:e}",
                "2:",
                "pause",
                "rdtsc",
                "shl rdx, 32",
                "or rax, rdx",
                "mov {t}, rax",
                "sub {t}, {prev}",
                "cmp {t}, {gap}",
                "jbe 3f",
                "mov {gap}, {t}",
                "3:",
                "mov {prev}, rax",
                "cmp rax, {deadline}",
                "jb 2b",
                "vpcmpeqd ymm0, ymm0, [{p}]",
                "vpcmpeqd ymm1, ymm1, [{p} + 32]",
                "vpcmpeqd ymm2, ymm2, [{p} + 64]",
                "vpcmpeqd ymm3, ymm3, [{p} + 96]",
                "vpcmpeqd ymm4, ymm4, [{p} + 128]",
                "vpcmpeqd ymm5, ymm5, [{p} + 160]",
                "vpcmpeqd ymm6, ymm6, [{p} + 192]",
                "vpcmpeqd ymm7, ymm7, [{p} + 224]",
                "vpcmpeqd ymm8, ymm8, [{p} + 256]",
                "vpcmpeqd ymm9, ymm9, [{p} + 288]",
                "vpcmpeqd ymm10, ymm10, [{p} + 320]",
                "vpcmpeqd ymm11, ymm11, [{p} + 352]",
                "vpcmpeqd ymm12, ymm12, [{p} + 384]",
                "vpcmpeqd ymm13, ymm13, [{p} + 416]",
                "vpcmpeqd ymm14, ymm14, [{p} + 448]",
                "vpcmpeqd ymm15, ymm15, [{p} + 480]",
                "vpand ymm0, ymm0, ymm1",
                "vpand ymm2, ymm2, ymm3",
                "vpand ymm4, ymm4, ymm5",
                "vpand ymm6, ymm6, ymm7",
                "vpand ymm8, ymm8, ymm9",
                "vpand ymm10, ymm10, ymm11",
                "vpand ymm12, ymm12, ymm13",
                "vpand ymm14, ymm14, ymm15",
                "vpand ymm0, ymm0, ymm2",
                "vpand ymm4, ymm4, ymm6",
                "vpand ymm8, ymm8, ymm10",
                "vpand ymm12, ymm12, ymm14",
                "vpand ymm0, ymm0, ymm4",
                "vpand ymm8, ymm8, ymm12",
                "vpand ymm0, ymm0, ymm8",
                "vpmovmskb {m:e}, ymm0",
                "vzeroupper",
                p = in(reg) pattern.as_ptr(),
                deadline = in(reg) deadline,
                prev = out(reg) _,
                t = out(reg) _,
                gap = out(reg) max_gap,
                m = out(reg) mask,
                out("rax") _, out("rdx") _,
                out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
                out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
                out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
                out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
                options(nostack, readonly),
            );
        }
        (mask == u32::MAX, max_gap)
    }

    /// Overwrites all YMM registers, including the upper halves.
    #[target_feature(enable = "avx2")]
    unsafe fn clobber_ymm() {
        unsafe {
            asm!(
                "vpcmpeqd ymm0, ymm0, ymm0",
                "vmovdqa ymm1, ymm0",
                "vmovdqa ymm2, ymm0",
                "vmovdqa ymm3, ymm0",
                "vmovdqa ymm4, ymm0",
                "vmovdqa ymm5, ymm0",
                "vmovdqa ymm6, ymm0",
                "vmovdqa ymm7, ymm0",
                "vmovdqa ymm8, ymm0",
                "vmovdqa ymm9, ymm0",
                "vmovdqa ymm10, ymm0",
                "vmovdqa ymm11, ymm0",
                "vmovdqa ymm12, ymm0",
                "vmovdqa ymm13, ymm0",
                "vmovdqa ymm14, ymm0",
                "vmovdqa ymm15, ymm0",
                out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
                out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
                out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
                out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
                options(nomem, nostack),
            );
        }
    }
}
