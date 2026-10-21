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
| rpi-llm, bare metal, 4 cores (2026-10-21) | 5.37 s (71.9 MB/s) | 287.7 tok/s | 30.0 tok/s |
| llama.cpp, Linux, 4 threads (2026-10-20) | 5.05 s (77 MB/s) | 261.3 | 25.2 |
| llama.cpp, Linux, 1 thread | 5.01 s | 91.4 | 28.8 |
| rpi-llm, Linux, 4 threads (before batching) | 4.37 s (88 MB/s) | 27.6 | 25.9 |

The Linux rows come from `bench/linux/run.sh` (Raspberry Pi OS, Debian 13,
kernel 6.18, `performance` governor, page cache dropped, llama.cpp
`0ce8cd1` with mmap off). The bare-metal load is from a different card
than Linux's; rpi-hal now runs it in UHS-I SDR104, as Linux does. From
power-on the board is ready for the prompt in 7.8 s.

## Build order

1. **The chat interface** on the RP1 UART0 (next): line input with basic
   editing, the answer streamed as it's generated, and a `[200 tok, 30.0
   tok/s, prompt 288 tok/s]` footer after each reply — the showcase
   number. Today the program answers one fixed request.
2. **Rerun the Linux baseline** for the final side-by-side: the `rpi-llm`
   Linux rows predate batching, NEON attention and the 4x8 kernel, and the
   card should be the same one the board reads.
3. **Faster startup**, after the 5.4 s load: the FNV-1a check of the
   whole file takes 1.4 s on one core (drop it, or spread it over four),
   and repacking the weights 0.5 s. SDR104 reads at 73% of its bus;
   ADMA2 in `rpi-hal` would show whether the word-at-a-time FIFO reads or
   the card set that.
4. **Smaller prompt work**, if it's worth it after the above:
   - `matmul_batch` spends about 20 ms of the prompt's 111.7 ms of matrix
     multiplies outside the kernel (`make kernel` puts the kernel at about
     30.7 cycles per 4x4 block, 86.5 ms of it).
   - One run's prompt phases all came in about 40% slower together, and
     the next didn't; the `arm:` lines (clock, temperature, throttling)
     before and after the prompt are there to catch it if it happens
     again. Not heat: the next run ran at 30–33 °C, never throttled.
5. **Quieter fan**, optional polish. Undriven, the fan runs at full speed,
   and under this load the SoC stays near 33 °C, so it isn't holding
   anything back. Driving it from the SoC temperature with `rpi-hal`'s
   `pwm` (RP1 PWM1, `Pwm1Pin::Gpio45`, inverted) would make it quiet at
   idle. Start each speed change at 50% or more: the fan sits at 0 RPM for
   seconds when asked for 25% from a standstill.
6. **Drop the temporary `[patch.crates-io]` path to rpi-hal** once its next
   release (with `mem` on bcm2712 and PSCI multicore) is out.

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

- **ADMA2 for the SD driver**, if SDR104's 73% of its bus turns out to
  be the FIFO reads rather than the card (build order step 3).
- **Mapping RAM above 4 GiB**, for bigger models on 8 GB and 16 GB boards.
- **Stretch: PCIe root complex + NVMe driver**, for loading models from the
  SSD (see the stretch goal above).
