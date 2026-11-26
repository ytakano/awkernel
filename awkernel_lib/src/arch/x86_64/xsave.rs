//! AVX state management.
//!
//! The kernel is compiled for SSE only. Code that uses AVX (selected at run
//! time with `#[target_feature]`, e.g. the LLM engine) needs
//!
//! 1. `CR4.OSXSAVE` and the AVX bit of XCR0 set on every CPU, which
//!    [`enable_avx`] does during boot, and
//! 2. the upper halves of the YMM registers preserved when a task is preempted
//!    by an interrupt. The `x86-interrupt` ABI saves only XMM0-15 (the kernel
//!    has no AVX), so the preemption handler wraps the context switch in
//!    [`with_avx_saved`], which saves the AVX state on the preempted task's
//!    own stack. The state therefore follows the task even if it resumes on
//!    another CPU.
//!
//! Voluntary context switches are ordinary function calls; the System V ABI
//! makes all vector registers caller-saved there, so nothing has to be saved.
//!
//! Other interrupt and exception handlers do not save the YMM upper halves,
//! so AVX code must only run in tasks, never in interrupt context.

use core::{
    arch::{
        asm,
        x86_64::{__cpuid, __cpuid_count},
    },
    sync::atomic::{AtomicBool, Ordering},
};

const XCR0_X87: u64 = 1 << 0;
const XCR0_SSE: u64 = 1 << 1;
const XCR0_AVX: u64 = 1 << 2;
const CR4_OSXSAVE: u64 = 1 << 18;

/// Bytes of the standard-format XSAVE area up to and including the AVX
/// component (legacy 512 + header 64 + YMM_Hi128 256).
const XSAVE_AVX_SIZE: usize = 832;

/// AVX is enabled on every CPU that has booted so far.
static AVX_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enables XSAVE and the AVX state on the calling CPU, if supported.
///
/// Call it on the primary CPU (`primary = true`) before any other CPU, and
/// then on every other CPU during boot. If any CPU cannot enable AVX, AVX is
/// treated as unavailable (the CPUs are assumed to be identical).
///
/// # Safety
///
/// Must be called at boot, before tasks run, with interrupts disabled.
pub unsafe fn enable_avx(primary: bool) -> bool {
    let ok = unsafe { try_enable_avx() };
    if primary {
        AVX_ENABLED.store(ok, Ordering::Release);
    } else if !ok {
        AVX_ENABLED.store(false, Ordering::Release);
    }
    ok
}

unsafe fn try_enable_avx() -> bool {
    let l1 = __cpuid(1);
    let has_xsave = l1.ecx & (1 << 26) != 0;
    let has_avx = l1.ecx & (1 << 28) != 0;
    if !(has_xsave && has_avx) || __cpuid(0).eax < 0xd {
        return false;
    }

    let mut cr4: u64;
    unsafe {
        asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags));
        asm!("mov cr4, {}", in(reg) cr4 | CR4_OSXSAVE, options(nostack, preserves_flags));
    }

    // Enable only components the CPU supports (CPUID.(0Dh,0):EDX:EAX).
    let l = __cpuid_count(0xd, 0);
    let supported = ((l.edx as u64) << 32) | l.eax as u64;
    let want = XCR0_X87 | XCR0_SSE | XCR0_AVX;
    if supported & want != want {
        return false;
    }

    // Use exactly these components on every CPU, so that all CPUs agree and
    // no state the preemption handler does not save (e.g. AVX-512) is on.
    let xcr0 = want;
    unsafe {
        asm!(
            "xsetbv",
            in("ecx") 0u32,
            in("eax") xcr0 as u32,
            in("edx") (xcr0 >> 32) as u32,
            options(nomem, nostack, preserves_flags)
        );
    }
    true
}

/// True if AVX has been enabled on every CPU booted so far.
#[inline]
pub fn is_avx_enabled() -> bool {
    AVX_ENABLED.load(Ordering::Acquire)
}

#[repr(C, align(64))]
struct XSaveArea(core::mem::MaybeUninit<[u8; XSAVE_AVX_SIZE]>);

/// Offset of the XSAVE header (XSTATE_BV, XCOMP_BV, reserved).
const XSAVE_HEADER: usize = 512;

/// Runs `f`, which may switch to another task (and later resume this one,
/// possibly on another CPU), preserving the upper halves of YMM0-15.
///
/// Only the AVX component is requested: the XMM halves are saved by the
/// `x86-interrupt` prologue. (XSAVE also stores MXCSR, which XRSTOR reloads.)
#[inline(always)]
pub fn with_avx_saved<R>(f: impl FnOnce() -> R) -> R {
    if !is_avx_enabled() {
        return f();
    }

    let mut area = XSaveArea(core::mem::MaybeUninit::uninit());
    let ptr = area.0.as_mut_ptr() as *mut u8;
    // SAFETY: AVX is enabled on every CPU and `area` is 64-byte aligned and
    // large enough. XRSTOR requires zero XCOMP_BV and reserved header bytes,
    // which XSAVE does not write; it writes everything else it later reads.
    unsafe {
        core::ptr::write_bytes(ptr.add(XSAVE_HEADER), 0, 64);
        asm!(
            "xsave64 [{0}]",
            // The scheduler and the next task run legacy SSE code; clean
            // upper halves avoid SSE/AVX transition penalties. XRSTOR
            // restores them below.
            "vzeroupper",
            in(reg) ptr,
            in("eax") XCR0_AVX as u32,
            in("edx") 0u32,
            options(nostack, preserves_flags)
        );
    }

    let r = f();

    unsafe {
        asm!(
            "xrstor64 [{0}]",
            in(reg) ptr,
            in("eax") XCR0_AVX as u32,
            in("edx") 0u32,
            options(nostack, preserves_flags)
        );
    }
    r
}
