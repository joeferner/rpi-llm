//! What the reference tests share: reading model files the way the board
//! does.

use std::path::PathBuf;

/// The repository's `models/` directory, where `make models` puts them.
pub fn models() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

/// A file's bytes in memory aligned for `f32`s, as the board's loader puts
/// them.
pub struct Aligned {
    words: Vec<u64>,
    len: usize,
}

impl Aligned {
    /// The file's bytes.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: `words` holds at least `len` bytes, all initialised.
        unsafe { std::slice::from_raw_parts(self.words.as_ptr().cast(), self.len) }
    }
}

/// Reads `name` from `models/`; `None` if it is not there.
pub fn read_model_file(name: &str) -> Option<Aligned> {
    let bytes = std::fs::read(models().join(name)).ok()?;
    let mut words = vec![0u64; bytes.len().div_ceil(8)];
    // SAFETY: `words` has at least `bytes.len()` bytes, and `u64` has no
    // invalid bit patterns.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), words.as_mut_ptr().cast(), bytes.len());
    }
    Some(Aligned {
        words,
        len: bytes.len(),
    })
}
