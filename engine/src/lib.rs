//! The model side of rpi-llm: everything that is arithmetic over bytes
//! already in memory, and nothing that touches hardware.
//!
//! Its own crate so that it runs on the host too. The board program reads
//! files off the card and owns the terminal; this crate is handed the
//! bytes and does the rest, which means `cargo test` can check it against
//! a reference implementation on a desktop in seconds rather than through
//! a card swap and a serial console.
//!
//! `no_std` (with `alloc`) everywhere except under `cargo test`, where the
//! test harness needs `std`.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod bpe;
pub mod generate;
pub mod gguf;
pub mod llama2;
pub mod model;
pub mod parallel;
pub mod quant;
pub mod sample;
pub mod tokenizer;
