//! Selection of the numeric kernel implementation.

/// Which implementation of the numeric kernels to use.
///
/// The AVX2 variant can only be obtained from [`Backend::detect`], so code
/// holding a `Backend` may call the AVX2 kernels without re-checking the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Backend(Impl);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Impl {
    /// Portable Rust. Used on non-x86_64 targets and as the reference.
    Scalar,
    /// AVX2 + FMA + F16C (x86_64 only).
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    Avx2,
}

impl Backend {
    /// Portable scalar kernels.
    pub const SCALAR: Self = Backend(Impl::Scalar);

    /// Returns the fastest backend the CPU *and* the OS (XCR0) support.
    ///
    /// In a kernel, the result is only valid on every CPU if AVX is enabled
    /// on all of them, since a task may migrate.
    pub fn detect() -> Self {
        if has_avx2_fma_f16c() {
            Backend(Impl::Avx2)
        } else {
            Backend::SCALAR
        }
    }

    pub fn is_avx2(&self) -> bool {
        self.0 == Impl::Avx2
    }

    pub(crate) fn imp(&self) -> Impl {
        self.0
    }

    pub fn name(&self) -> &'static str {
        match self.0 {
            Impl::Scalar => "scalar",
            Impl::Avx2 => "avx2",
        }
    }
}

/// True if AVX2, FMA and F16C can be executed: the CPU implements them and
/// the OS has enabled the AVX state in XCR0 (`CR4.OSXSAVE` and `XCR0[2:1]`).
///
/// Without the XCR0 check, a kernel that has not enabled AVX would take `#UD`
/// on the first VEX instruction.
#[cfg(target_arch = "x86_64")]
pub fn has_avx2_fma_f16c() -> bool {
    use core::arch::x86_64::{__cpuid, __cpuid_count};

    let leaf0 = __cpuid(0);
    if leaf0.eax < 7 {
        return false;
    }

    let l1 = __cpuid(1);
    let fma = l1.ecx & (1 << 12) != 0;
    let osxsave = l1.ecx & (1 << 27) != 0;
    let avx = l1.ecx & (1 << 28) != 0;
    let f16c = l1.ecx & (1 << 29) != 0;
    if !(fma && osxsave && avx && f16c) {
        return false;
    }

    let l7 = __cpuid_count(7, 0);
    let avx2 = l7.ebx & (1 << 5) != 0;
    if !avx2 {
        return false;
    }

    // SSE (bit 1) and AVX (bit 2) state must be enabled by the OS.
    let xcr0 = unsafe { xgetbv0() };
    xcr0 & 0b110 == 0b110
}

#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx2_fma_f16c() -> bool {
    false
}

/// Reads XCR0. Only valid when `CPUID.1:ECX.OSXSAVE` is set.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "xsave")]
unsafe fn xgetbv0() -> u64 {
    core::arch::x86_64::_xgetbv(0)
}
