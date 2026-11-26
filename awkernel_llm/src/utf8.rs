//! Incremental UTF-8 decoding of a token stream.
//!
//! A token may end in the middle of a multi-byte character (CJK, emoji), so
//! token bytes cannot be printed one token at a time as `&str`.

use alloc::{string::String, vec::Vec};

/// Buffers an incomplete trailing UTF-8 sequence between calls.
#[derive(Debug, Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub const fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Appends the decodable prefix of `pending + bytes` to `out`. Invalid
    /// sequences become U+FFFD; an incomplete tail is kept for the next call.
    pub fn push(&mut self, bytes: &[u8], out: &mut String) {
        self.pending.extend_from_slice(bytes);
        let mut rest: &[u8] = &self.pending;
        loop {
            match core::str::from_utf8(rest) {
                Ok(s) => {
                    out.push_str(s);
                    rest = &[];
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    // SAFETY: `from_utf8` validated `rest[..valid]`.
                    out.push_str(unsafe { core::str::from_utf8_unchecked(&rest[..valid]) });
                    match e.error_len() {
                        Some(n) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            rest = &rest[valid + n..];
                        }
                        None => {
                            rest = &rest[valid..];
                            break;
                        }
                    }
                }
            }
        }
        let keep = rest.len();
        let start = self.pending.len() - keep;
        self.pending.drain(..start);
    }

    /// Flushes an incomplete tail as U+FFFD.
    pub fn finish(&mut self, out: &mut String) {
        if !self.pending.is_empty() {
            out.push(char::REPLACEMENT_CHARACTER);
            self.pending.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_multibyte() {
        let s = "aあ😀b".as_bytes();
        let mut d = Utf8Stream::new();
        let mut out = String::new();
        for b in s {
            d.push(&[*b], &mut out);
        }
        d.finish(&mut out);
        assert_eq!(out, "aあ😀b");
    }

    #[test]
    fn invalid_bytes() {
        let mut d = Utf8Stream::new();
        let mut out = String::new();
        d.push(&[b'a', 0xff, b'b', 0xe3, 0x81], &mut out);
        assert_eq!(out, "a\u{fffd}b");
        d.finish(&mut out);
        assert_eq!(out, "a\u{fffd}b\u{fffd}");
    }
}
