//! What the reference tests share: reading model files the way the board
//! does.

use std::path::PathBuf;

use rpi_llm_engine::parallel::Workers;

/// [`Workers`] as OS threads, a fresh set per job: slow, but the same
/// split the board's cores run, which is what the tests need.
#[allow(dead_code, reason = "not every test file runs in parallel")]
pub struct Threads(pub usize);

impl Workers for Threads {
    fn count(&self) -> usize {
        self.0
    }

    fn run(&self, job: &(dyn Fn(usize) + Sync)) {
        std::thread::scope(|scope| {
            for i in 1..self.0 {
                scope.spawn(move || job(i));
            }
            job(0);
        });
    }
}

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

    /// The file's bytes, writable: what repacking a GGUF file needs.
    #[allow(dead_code, reason = "not every test file repacks")]
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as in `bytes`, and `&mut self` makes it the only view.
        unsafe { std::slice::from_raw_parts_mut(self.words.as_mut_ptr().cast(), self.len) }
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
