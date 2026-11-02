# rpi-llm plan

A small chat LLM running bare metal on a Raspberry Pi 5 on top of
`rpi-hal` (`bcm2712` feature), with no Linux. You talk to it from a serial
terminal. It's meant to show off the Pi 5's speed, and to drive the Pi 5
features `rpi-hal` doesn't have yet.

The model is **SmolLM2-360M-Instruct, GGUF Q8_0**, read off the SD card
with a progress line, run on all four cores with NEON `sdot` kernels, and
bit-identical to the engine's host build (checked on every run against a
recorded 200-token answer).

### What the demo is really showing

- **Linux-class speed, no OS.** The prompt and the answer both now run
  faster than llama.cpp on Raspberry Pi OS on the same board.
- **Fast boot.** Our code starts in about 2 seconds, against roughly 20
  for Linux. After that, time to the first prompt is the model load, which
  storage bandwidth sets: 5.4 s at rpi-hal's UHS-I SDR104.
- **Steady token timing**, with no scheduler jitter.
- **The whole stack is ours**, with no OS involved.

## Where it stands

SmolLM2-360M-Instruct Q8_0, the same chat request (42 prompt tokens, 200
answer tokens, greedy), one Pi 5 at 2.4 GHz:

| | Load | Prompt | Answer |
|---|---|---|---|
| rpi-llm, bare metal, 4 cores (2026-10-21) | 4.24 s (91.0 MB/s) | 284.4 tok/s | 30.1 tok/s |
| llama.cpp, Linux, 4 threads (2026-10-20) | 5.05 s (77 MB/s) | 261.3 | 25.2 |
| llama.cpp, Linux, 1 thread | 5.01 s | 91.4 | 28.8 |
| rpi-llm, Linux, 4 threads (before batching) | 4.37 s (88 MB/s) | 27.6 | 25.9 |

The Linux rows come from `bench/linux/run.sh` (Raspberry Pi OS, Debian 13,
kernel 6.18, `performance` governor, page cache dropped, llama.cpp
`0ce8cd1` with mmap off). Every row read the same card (bare metal from
its FAT boot partition, in UHS-I SDR104). Bare metal is ahead on all
three; from power-on the board is ready for the prompt in 6.75 s. On the
slower card the board usually boots from, the load is 5.37 s (71.9 MB/s,
that card's own limit).

## Build order

1. **Rerun the Linux baseline** for the final side-by-side: the `rpi-llm`
   Linux rows predate batching, NEON attention and the 4x8 kernel, and the
   card should be the same one the board reads.
2. **Faster startup: overlap the hash and the repack with the read.**
   After the 4.24 s load, the FNV-1a check of the whole file takes 1.4 s
   on one core and repacking the weights 0.5 s, both after the read, so
   the prompt is ready 6.75 s after power-on. Done on each chunk while
   the next one is still coming off the card, they'd end with the read:
   about 4.9 s.
   - **In `rpi-hal`: a non-blocking SD read on ADMA2.** ADMA2 is in
     (`Sd::set_adma`, off by default) and verified both ways on the
     board, but `read_blocks` still blocks until the transfer ends, so it
     frees nothing yet. Add a pair of calls: one that issues the command
     and returns once the controller is moving the data, and one that
     waits for the end and does the post-transfer cache invalidate —
     polled, since the Pi 5 has no interrupt controller support. The
     buffer is borrowed until the wait, so the API has to keep it from
     being touched or freed in between. Example: `sd_speed` timing how
     much work a core gets done while a read is in flight.
   - **In `rpi-llm`'s loader:** double-buffer the read. Start chunk *n+1*,
     then hash and repack chunk *n* on core 0 while the controller fills
     the next; the progress line stays as it is. The FNV-1a hash runs at
     about 270 MB/s and the repack faster, against 91 MB/s off the card,
     so one core keeps up. This needs `resident-fat` to expose the file's
     runs of blocks (or a read call that doesn't block) rather than only
     a blocking `read_at`.
   - **The last few MB/s:** the load reaches 91.0 MB/s against
     `sd_speed`'s 95.1 on this card; requests larger than 1 MiB should
     close most of that. The demo needs a card as fast as this one (it is
     the Linux install's).
3. **Smaller prompt work**, if it's worth it after the above:
   - `matmul_batch` spends about 20 ms of the prompt's 111.7 ms of matrix
     multiplies outside the kernel (`make kernel` puts the kernel at about
     30.7 cycles per 4x4 block, 86.5 ms of it).
   - One run's prompt phases all came in about 40% slower together, and
     the next didn't; the `arm:` lines (clock, temperature, throttling)
     before and after the prompt are there to catch it if it happens
     again. Not heat: the next run ran at 30–33 °C, never throttled.
4. **Quieter fan**, optional polish. Undriven, the fan runs at full speed,
   and under this load the SoC stays near 33 °C, so it isn't holding
   anything back. Driving it from the SoC temperature with `rpi-hal`'s
   `pwm` (RP1 PWM1, `Pwm1Pin::Gpio45`, inverted) would make it quiet at
   idle. Start each speed change at 50% or more: the fan sits at 0 RPM for
   seconds when asked for 25% from a standstill.

## Stretch goal: models on an NVMe SSD on the PCIe connector

Keep the models on the SSD attached through the Pi 5's external PCIe
connector, and load them from there. Faster loads are what you'd see (about
1 s), but the larger gain is in `rpi-hal`: a PCIe root complex driver opens
the bus that every add-on board for that connector uses.

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

- **A non-blocking SD read over ADMA2** (build order step 2). ADMA2
  itself is done (`Sd::set_adma`, off by default, since blocking
  transfers gain nothing from it: the FIFO keeps up with the bus and the
  cache maintenance costs about 1%); what's left is a read that starts
  and returns, so the CPU it frees can be used.
- **Stretch: PCIe root complex + NVMe driver**, for loading models from the
  SSD (see the stretch goal above).
