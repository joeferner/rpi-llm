# rpi-llm plan

A small chat LLM running bare metal on a Raspberry Pi 5 on top of
`rpi-hal` (`bcm2712` feature), with no Linux. You talk to it from a serial
terminal. It's meant to show off the Pi 5's speed, and to drive the Pi 5
features `rpi-hal` doesn't have yet.

## Model

Generating tokens with a small LLM is limited by memory bandwidth, not
compute: every weight is read once per token. So the model's size sets the
speed.

| Model | Q8_0 size | Chat quality | Rough speed (Pi 5) |
|---|---|---|---|
| llama2.c `stories15M` / `110M` | 15–110 MB (fp32) | None: it only continues children's stories | Hundreds of tok/s |
| **SmolLM2-360M-Instruct** | ~380 MB | Usable small talk and simple Q&A | ~15–25 tok/s on 4 cores |
| Qwen2.5-0.5B-Instruct | ~530 MB | Noticeably smarter | ~10–15 tok/s on 4 cores |
| Llama-3.2-1B-Instruct | ~1.3 GB | Good | ~4–7 tok/s, too slow to impress |

The speeds are estimates. The real baseline is llama.cpp on Raspberry Pi OS
on the same board.

Target model: **SmolLM2-360M-Instruct, GGUF Q8_0**. It's a plain Llama
architecture (RMSNorm, RoPE, grouped-query attention, SwiGLU). Qwen2.5 adds
attention biases and Qwen3 adds an extra normalization step; SmolLM2 needs
neither, so its engine is about 1,000 lines. `stories15M` is the first
bring-up target, because its output is easy to check against llama2.c.

### What the demo is really showing

Because generation is bandwidth-bound, bare metal won't be dramatically
faster per token than llama.cpp on Linux; parity is the realistic goal. The
wins worth showing are:

- **Fast boot.** Our code starts in about 2 seconds, against roughly 20
  seconds for Linux to boot. After that, time to the first prompt is just
  the model load, and storage bandwidth sets that (see section 1).
- **Steady token timing**, with no scheduler jitter.
- **The whole stack is ours**, with no OS involved.

Headline: "Linux-class tok/s, no OS."

## What's involved

### 1. Getting the weights into RAM

- Uploading with rpi-loader over UART is out: at 921,600 baud, 380 MB takes
  over an hour.
- The app reads the model from SD itself, with `rpi-hal`'s SD driver and
  `resident-fat`, rather than having the firmware load it before boot. That
  way the load is visible: a progress line on the terminal, updated as each
  chunk arrives, showing the file name, MB read out of the total, a percent
  bar, MB/s and time remaining. Reads are large multi-block ones straight
  into the weights buffer, since the card charges per command, not per
  block.
- Storage bandwidth sets the load time for a 380 MB model:
  - SD in high-speed mode (25 MB/s at most): about 15 s
  - SD in SDR104 mode (~100 MB/s at most): about 4 s
  - NVMe on the PCIe connector (stretch goal below): about 1 s
- Read GGUF files directly. The format is easy to parse, it holds the
  tokenizer vocabulary and merge rules in the same file, and Q8_0 GGUF files
  for these models are already published on Hugging Face.

### 2. Runtime basics

- A global allocator; `rpi-hal`'s `heap_alloc` example shows the pattern.
  The KV cache for SmolLM2 at 2,048 tokens of context is about 166 MB in
  f32, which fits easily.
- The `libm` crate for `expf`/`sqrtf`, since `core` doesn't provide them.
- NEON is already enabled on `aarch64-unknown-none`. The Cortex-A76 also has
  `sdot` (int8 dot product), the instruction for the Q8_0 matrix multiply.
  If its Rust intrinsic still needs nightly, a few lines of `asm!` work on
  stable.
- `set_clock_rate_hz(max)` at boot. The Pi 3 firmware ran the ARM below its
  maximum until told otherwise; the Pi 5 hasn't been checked, so measure
  the actual clock.

### 3. The engine

A no_std port of llama2.c's int8 path (`runq.c`), adapted for GGUF Q8_0
blocks:

- matrix multiply, RMSNorm, RoPE, softmax and the attention loop
- a byte-level BPE tokenizer
- SmolLM2's ChatML template (`<|im_start|>user\n…<|im_end|>`)

### 4. Multicore (the biggest speed lever)

One A76 core can't use all the memory bandwidth, so single-core will run
roughly 2–3× slower than llama.cpp with 4 threads. Start cores 1–3 with
`rpi-hal`'s `multicore` feature and split each matrix multiply's output
rows across the 4 cores.

### 5. Cooling

With nothing driving it, the Pi 5's fan runs at full speed, so the cores
won't throttle; the cost is noise. Drive it from the SoC temperature with
`rpi-hal`'s `pwm` (RP1 PWM1, `Pwm1Pin::Gpio45`, inverted), quiet at idle
and full speed under inference. Start each speed change at 50% or more:
the fan sits at 0 RPM for seconds when asked for 25% from a standstill.

### 6. Chat interface

A serial terminal over the RP1 UART0 (GPIO14/15):

- the model-load progress line from section 1, then the prompt
- line input with basic editing
- tokens streamed as they are generated
- a `[42 tok, 18.3 tok/s, prompt 95 tok/s]` footer after each reply. That
  footer is the showcase number.

## Baseline: Linux on the same Pi

`bench/linux/run.sh` runs llama.cpp (commit `0ce8cd1`, its defaults, mmap
off so the load is real) and this engine as a Linux program
(`rpi-llm-bench`) on the same Pi 5, model and chat request, with the page
cache dropped before each run. Raspberry Pi OS (Debian 13, kernel 6.18),
`performance` governor, 2.4 GHz, not throttled; 2026-10-20:

| | Load | Prompt | Answer |
|---|---|---|---|
| llama.cpp, Linux, 4 threads | 5.05 s (77 MB/s) | 261.3 tok/s | 25.2 tok/s |
| llama.cpp, Linux, 1 thread | 5.01 s | 91.4 | 28.8 |
| rpi-llm, Linux, 4 threads | 4.37 s (88 MB/s) | 27.6 | 25.9 |
| rpi-llm, Linux, 1 thread | 4.42 s | 26.0 | 22.3 |
| rpi-llm, bare metal, 4 cores | 32.9 s (11.7 MB/s) | 28.5 | 25.5 |

- Generation is at parity: everything meets the same memory wall.
- The prompt is 9x behind llama.cpp, which batches it.
- The load is 7x behind Linux, whose SD driver runs the card in a fast
  UHS mode; rpi-hal's stays at default speed.
- llama.cpp's one-thread kernel (28.8) beats this engine's (22.3 on
  Linux), and its four threads lose to its one.
- This engine's one thread is faster on Linux (22.3) than it was on bare
  metal (19.4): the Linux target is not `strict-align`, so code outside
  the NEON kernel may still be paying for alignment on the board.

## Build order

1. Close the rest of the prompt gap: 241.3 tok/s against llama.cpp's 261.
   The prompt's 141.6 ms is now matmul 107.0, other 19.3, quantize 9.2,
   attention 6.1. The 4x8 kernel takes about 38 cycles per 4x4 block
   against a floor of about 25 from its instruction count; its `sdot`
   latency chains were only part of that (114.9 -> 107.0 ms). Benchmark
   it alone on the board (L1-hot, real shapes, four cores) to find the
   rest: the 2560-column matrix's tokens (127 KB) don't fit in L1.
2. Fan control, then rerun the baseline above for the final side-by-side.

## Stretch goal: models on an NVMe SSD on the PCIe connector

Keep the models on the SSD attached through the Pi 5's external PCIe
connector, and load them from there. Faster loads are what you'd see (about
1 s instead of about 15 s from SD), but the larger gain is in `rpi-hal`: a
PCIe root complex driver opens the bus that every add-on board for that
connector uses.

What it takes:

- **Enable the external PCIe port** in config.txt (`dtparam=pciex1`, plus
  `dtparam=pciex1_gen=3` for Gen 3). Still to check: whether the firmware
  leaves the link trained for a bare-metal kernel, or only describes it in
  the device tree for Linux to bring up.
- **Map its address window in `rpi-hal`'s MMU.** It needs the PCIe1
  outbound window added to the Device regions; today only the SoC block
  and RP1's window (PCIe2) are mapped.
- **PCIe root complex bring-up** for the Broadcom controller (what Linux's
  `pcie-brcmstb` driver does), if the firmware doesn't leave it trained:
  reset, link training, outbound window for config space and memory BARs,
  and an inbound window so the SSD can DMA into RAM.
- **Bus enumeration**: find the SSD's function in config space, assign its
  BAR, enable bus mastering.
- **A minimal NVMe driver**: admin queue, Identify, one I/O queue pair,
  polled Read commands. No interrupts needed: polling the completion queue
  is fine for one bulk load, and it avoids MSI work.
- **A filesystem on it**: a `resident-fat` `BlockDevice` over the NVMe
  driver, so the SSD holds the models as ordinary files the same way the
  SD card does.

## rpi-hal work this pulls in

Each item lands in `rpi-hal` as its own PR with its own example, tested on
the board before this app uses it. This app uses `rpi-hal` by crate name,
through a local `[patch.crates-io]` path while the work is in progress.

- **Mapping RAM above 4 GiB**, for bigger models on 8 GB and 16 GB boards.
- **Faster SD modes** (needs the switch to 1.8 V signalling): Linux reads
  the same card at 88 MB/s against rpi-hal's 11.7, so SmolLM2 loads in
  4.4 s there and 32.9 s on bare metal — the largest gap in the baseline.
- **Stretch: PCIe root complex + NVMe driver**, for loading models from the
  SSD (see the stretch goal above).
