//! rpi-llm's engine as a Linux program: the same model, the same chat
//! request and the same measurements as the bare-metal board program, so
//! the two (and llama.cpp) can be compared on the same Pi.
//!
//! ```text
//! rpi-llm-bench <model.gguf> [--threads N]
//! ```
//!
//! It reads the whole model into RAM (timed, as the board's load is),
//! repacks it, answers the request greedily, and checks the answer against
//! what the engine produced on the development host. The last line it
//! prints is machine-readable, for `linux/run.sh`.

use std::fs::File;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::time::Instant;

use rpi_llm_engine::bpe::Bpe;
use rpi_llm_engine::generate::Generation;
use rpi_llm_engine::gguf::Gguf;
use rpi_llm_engine::model::{Transformer, Weights};
use rpi_llm_engine::parallel::{Serial, Workers};

/// The chat request: the model's ChatML template around one request, the
/// one the board program and the engine's tests use.
const PROMPT: &str = include_str!("../../engine/tests/data/smollm2_story.prompt");

/// What the engine generated from [`PROMPT`] on the development host.
const REFERENCE: &str = include_str!("../../engine/tests/data/smollm2_story.engine.ids");

/// How many tokens the answer may run to.
const ANSWER_TOKENS: usize = 200;

/// Positions the key/value cache holds, as on the board.
const CONTEXT: usize = 2048;

/// One cache line: the model is read into memory aligned to these, as the
/// board's loader does.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([u8; 64]);

fn main() {
    let mut args = std::env::args().skip(1);
    let mut path = None;
    let mut threads = 4usize;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--threads" => {
                threads = args
                    .next()
                    .and_then(|n| n.parse().ok())
                    .expect("--threads takes a number");
            }
            _ => path = Some(arg),
        }
    }
    let path = path.expect("usage: rpi-llm-bench <model.gguf> [--threads N]");

    // The load: the whole file into aligned memory, timed.
    let started = Instant::now();
    let mut file = File::open(&path).expect("opening the model");
    let len = file.metadata().expect("the model's size").len() as usize;
    let mut lines = vec![Line([0; 64]); len.div_ceil(64)];
    // SAFETY: `lines` holds at least `len` bytes, all initialised.
    let bytes = unsafe { std::slice::from_raw_parts_mut(lines.as_mut_ptr().cast::<u8>(), len) };
    file.read_exact(bytes).expect("reading the model");
    let load = started.elapsed();
    eprintln!(
        "load: {len} bytes in {:.2} s, {:.1} MB/s",
        load.as_secs_f64(),
        len as f64 / load.as_secs_f64() / 1e6
    );

    let started = Instant::now();
    let gguf = Gguf::parse_repacked(bytes).expect("parsing the model");
    let repack = started.elapsed();
    eprintln!("repack: {} ms", repack.as_millis());

    let bpe = Bpe::from_gguf(&gguf).expect("the tokenizer");
    let (config, weights) = Weights::from_gguf(&gguf).expect("the weights");
    let mut transformer = Transformer::new(config, weights, CONTEXT);
    let pool;
    if threads > 1 {
        pool = Pool::start(threads);
        transformer.set_workers(pool);
    } else {
        transformer.set_workers(&Serial);
    }
    eprintln!("threads: {threads}");

    // The chat turn, timed as the board times it: the prompt until the
    // first answer token, the answer from there on.
    let prompt = bpe.encode(PROMPT);
    let prompt_len = prompt.len();
    let steps = prompt_len + ANSWER_TOKENS - 1;
    let mut generation = Generation::new(prompt, steps, &[bpe.eos()]);
    let mut ids = Vec::new();
    let mut piece = Vec::new();
    let mut stdout = std::io::stdout().lock();
    let started = Instant::now();
    // The prompt, all but its last token, through the model together.
    generation.prefill(&mut transformer);
    let mut first = None;
    while let Some(step) = generation.step(&mut transformer) {
        if !step.generated {
            continue;
        }
        first.get_or_insert_with(Instant::now);
        ids.push(step.next);
        piece.clear();
        bpe.decode(step.next, &mut piece);
        let _ = stdout.write_all(&piece);
        let _ = stdout.flush();
    }
    let ended = Instant::now();
    let _ = writeln!(stdout);
    let first = first.expect("an answer");
    let prompt_time = first - started;
    let answer_time = ended - first;
    eprintln!(
        "prompt: {prompt_len} tokens in {:.2} s, {:.1} tok/s",
        prompt_time.as_secs_f64(),
        prompt_len as f64 / prompt_time.as_secs_f64()
    );
    eprintln!(
        "answer: {} tokens in {:.2} s, {:.1} tok/s",
        ids.len(),
        answer_time.as_secs_f64(),
        (ids.len() - 1) as f64 / answer_time.as_secs_f64()
    );

    let expected: Vec<u32> = REFERENCE.lines().map(|l| l.parse().unwrap()).collect();
    let matches = ids == expected;
    eprintln!(
        "reference: {}",
        if matches {
            "all tokens match the host engine"
        } else {
            "DIFFERS from the host engine"
        }
    );

    eprintln!(
        "BENCH load_us={} bytes={len} repack_us={} prompt_tokens={prompt_len} prompt_us={} answer_tokens={} answer_us={} match={}",
        load.as_micros(),
        repack.as_micros(),
        prompt_time.as_micros(),
        ids.len(),
        answer_time.as_micros(),
        if matches { "yes" } else { "no" }
    );
}

/// The engine's [`Workers`] as OS threads that spin between jobs.
///
/// Spinning rather than sleeping because a token runs about 256 jobs:
/// waking a sleeping thread through the kernel costs more than many of
/// them take. The board's cores wait the same way (in `wfe`), and
/// llama.cpp's threads spin too.
struct Pool {
    threads: usize,
    /// The job being run, as a fat pointer split into two words, its
    /// lifetime erased for the span [`Workers::run`] waits out.
    job: [AtomicUsize; 2],
    /// Bumped to start a job.
    go: AtomicU32,
    /// Workers finished with the current job.
    done: AtomicU32,
}

impl Pool {
    /// Starts `threads - 1` spinning workers; the caller is worker 0.
    fn start(threads: usize) -> &'static Pool {
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            threads,
            job: [AtomicUsize::new(0), AtomicUsize::new(0)],
            go: AtomicU32::new(0),
            done: AtomicU32::new(0),
        }));
        for id in 1..threads {
            std::thread::spawn(move || pool.worker(id));
        }
        pool
    }

    fn worker(&self, id: usize) {
        let mut seen = 0;
        loop {
            let go = self.go.load(Ordering::Acquire);
            if go == seen {
                std::hint::spin_loop();
                continue;
            }
            seen = go;
            let words = [
                self.job[0].load(Ordering::Relaxed),
                self.job[1].load(Ordering::Relaxed),
            ];
            // SAFETY: `run` stored this pointer before publishing `go` and
            // waits for this worker's done count, so the job is alive.
            let job: &(dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(words) };
            job(id);
            self.done.fetch_add(1, Ordering::Release);
        }
    }
}

impl Workers for Pool {
    fn count(&self) -> usize {
        self.threads
    }

    fn run(&self, job: &(dyn Fn(usize) + Sync)) {
        // SAFETY: a fat reference is two words; the workers turn it back
        // into a reference only while this call is waiting for them.
        let words: [usize; 2] = unsafe { std::mem::transmute(job) };
        self.job[0].store(words[0], Ordering::Relaxed);
        self.job[1].store(words[1], Ordering::Relaxed);
        self.done.store(0, Ordering::Relaxed);
        self.go.fetch_add(1, Ordering::Release);
        job(0);
        let others = (self.threads - 1) as u32;
        while self.done.load(Ordering::Acquire) < others {
            std::hint::spin_loop();
        }
    }
}
