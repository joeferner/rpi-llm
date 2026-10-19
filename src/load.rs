//! Reading a file off the card into RAM, with a progress line.

use alloc::vec::Vec;

use resident_fat::BlockDevice;
use rpi_hal::timer::Timer;
use rpi_kickstart::storage::{Error, Volume};

use crate::term::Progress;

/// How much each read asks the card for.
///
/// Big enough that the per-command cost disappears — `read_at` turns a
/// contiguous run of clusters into one multi-block read, and the card
/// charges per command, not per block — and small enough that the
/// progress line moves several times a second.
const CHUNK: usize = 1024 * 1024;

/// One cache line of a [`Blob`], the unit its buffer is allocated in.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([u8; 64]);

/// A file's contents in RAM, starting on a cache-line boundary.
///
/// Aligned because what is in it is read as `f32`s (and, later, by NEON
/// loads that want 16-byte alignment): a `Vec<u8>` promises only byte
/// alignment, and reinterpreting one would be undefined behaviour the
/// first time the allocator handed back an odd address.
pub struct Blob {
    lines: Vec<Line>,
    len: usize,
}

impl Blob {
    /// The file's bytes.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: `lines` is `len.div_ceil(64)` initialised `Line`s, each 64
        // plain bytes, so its first `len` bytes are initialised and in
        // bounds.
        unsafe { core::slice::from_raw_parts(self.lines.as_ptr().cast(), self.len) }
    }

    /// The file's bytes, writable: what [`load`] reads into, and what
    /// repacking a model in place needs.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as in `bytes`, and `&mut self` makes it the only view.
        unsafe { core::slice::from_raw_parts_mut(self.lines.as_mut_ptr().cast(), self.len) }
    }
}

/// Why [`load`] could not bring a file into RAM.
#[derive(Debug)]
#[allow(
    dead_code,
    reason = "the fields are read through `Debug`, in the failure message"
)]
pub enum LoadError<E> {
    /// The card or the filesystem refused; includes a missing file.
    Fs(Error<E>),
    /// The heap could not hold the file.
    OutOfMemory {
        /// The file's size, in bytes.
        bytes: usize,
    },
}

/// Reads all of `path` into RAM, drawing a progress line as it goes.
/// Returns the contents and how long the read took, in microseconds.
pub fn load<D: BlockDevice>(
    volume: &mut Volume<D>,
    path: &str,
    timer: &Timer,
) -> Result<(Blob, u64), LoadError<D::Error>> {
    let file = volume.open(path).map_err(LoadError::Fs)?;
    let len = file.len() as usize;

    let mut lines = Vec::new();
    lines
        .try_reserve_exact(len.div_ceil(64))
        .map_err(|_| LoadError::OutOfMemory { bytes: len })?;
    lines.resize(len.div_ceil(64), Line([0; 64]));
    let mut blob = Blob { lines, len };

    let mut progress = Progress::new(path, len as u64, timer.now_micros());
    let mut done = 0;
    while done < len {
        let end = (done + CHUNK).min(len);
        let read = volume
            .read_at(&file, done as u64, &mut blob.bytes_mut()[done..end])
            .map_err(LoadError::Fs)?;
        // `read_at` is short only at the end of the file, and the file's
        // length is what `end` is bounded by, so this never fires on a
        // healthy card; a zero here would otherwise spin forever.
        if read == 0 {
            break;
        }
        done += read;
        progress.update(done as u64, timer.now_micros());
    }
    let elapsed = progress.finish(timer.now_micros());

    Ok((blob, elapsed))
}

/// FNV-1a, 64-bit, over `bytes`.
///
/// What the loaded files are checked against: one number the host can
/// compute over the same file with a few lines of Python, so a match says
/// every byte in RAM is the byte on the card. Not a cryptographic hash, and
/// it does not need to be — the question is corruption, not tampering.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
