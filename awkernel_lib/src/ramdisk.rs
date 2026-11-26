//! Read-only access to the ramdisk loaded by the bootloader.
//!
//! On x86_64 the UEFI bootloader loads the file `ramdisk` of the boot image
//! into memory (reported as not usable, so the heap never reuses it), maps it
//! and passes its address to the kernel, which registers it here during boot.
//! Applications get it as a `&'static [u8]`, e.g. to use model weights
//! without copying them.

use core::sync::atomic::{AtomicUsize, Ordering};

static ADDR: AtomicUsize = AtomicUsize::new(0);
static LEN: AtomicUsize = AtomicUsize::new(0);

/// Registers the ramdisk.
///
/// # Safety
///
/// `addr..addr + len` must be mapped readable for the rest of the kernel's
/// lifetime and must never be written or reused for anything else.
pub unsafe fn set(addr: usize, len: usize) {
    LEN.store(len, Ordering::Relaxed);
    ADDR.store(addr, Ordering::Release);
}

/// The ramdisk contents, if the bootloader loaded one.
pub fn get() -> Option<&'static [u8]> {
    let addr = ADDR.load(Ordering::Acquire);
    let len = LEN.load(Ordering::Relaxed);
    if addr == 0 || len == 0 {
        return None;
    }
    // SAFETY: guaranteed by the caller of `set`.
    Some(unsafe { core::slice::from_raw_parts(addr as *const u8, len) })
}
