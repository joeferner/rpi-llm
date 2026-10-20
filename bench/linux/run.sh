#!/usr/bin/env bash
# Benchmarks llama.cpp and rpi-llm's engine on a Raspberry Pi 5 running
# Linux (Raspberry Pi OS, 64-bit), on the same model and the same chat
# request the bare-metal program answers, and prints one table.
#
#   bench/linux/run.sh [work-dir]      (default: ~/rpi-llm-bench)
#
# Each program is run with 4 threads and with 1. Before every run the page
# cache is dropped, so the model is read from the card each time and the
# load time is a real one. For each run it records:
#
#   load    the whole model read into RAM (llama.cpp with mmap off, which
#           a one-line patch to its `simple` example does; its time also
#           includes llama.cpp's own repacking of the weights)
#   prompt  the prompt's tokens until the first answer token, in tok/s
#   answer  the 200 answer tokens after that, in tok/s
#
# llama.cpp is built at the commit rpi-llm's reference outputs came from,
# with bench/linux/llama-simple.patch, which adds the timing, the thread
# count (RPI_LLM_THREADS) and mmap off; everything else is llama.cpp's
# default (f16 KV cache, flash attention on auto, OpenMP, -mcpu=native).
#
# Needs: git, cmake, g++, curl (`sudo apt install git cmake g++ curl`),
# Rust (https://rustup.rs), and sudo, for dropping the page cache and
# setting the CPU governor to `performance`.

set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${1:-$HOME/rpi-llm-bench}
MODEL_NAME=smollm2-360m-instruct-q8_0.gguf
MODEL_URL=https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/$MODEL_NAME
LLAMA_REPO=https://github.com/ggml-org/llama.cpp
LLAMA_COMMIT=0ce8cd11416c8840498e6701bcbe39ad4b858c2b
PROMPT_FILE=$REPO/engine/tests/data/smollm2_story.prompt
ANSWER_TOKENS=200
THREADS=(4 1)

for tool in git cmake g++ curl sha256sum cargo rustc; do
    command -v "$tool" > /dev/null || {
        echo "missing: $tool (see the top of $0)" >&2
        exit 1
    }
done

mkdir -p "$WORK"
STAMP=$(date +%Y%m%d-%H%M%S)
RESULTS=$WORK/results-$STAMP
mkdir -p "$RESULTS"
echo "results go to $RESULTS"

# --- The system, as it stands before anything runs ------------------------

{
    echo "date:      $(date -Iseconds)"
    echo "board:     $(tr -d '\0' < /proc/device-tree/model)"
    echo "kernel:    $(uname -srm)"
    echo "os:        $(. /etc/os-release && echo "$PRETTY_NAME")"
    echo "memory:    $(free -m | awk '/^Mem:/ { print $2 " MiB" }')"
    echo "storage:   $(findmnt -no SOURCE,FSTYPE -T "$WORK")"
    if command -v vcgencmd > /dev/null; then
        echo "arm clock: $(vcgencmd measure_clock arm)"
        echo "temp:      $(vcgencmd measure_temp)"
        echo "throttled: $(vcgencmd get_throttled)"
    fi
    echo "governor:  $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    echo "rpi-llm:   $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet || echo ' (modified)')"
    echo "llama.cpp: ${LLAMA_COMMIT:0:7} + bench/linux/llama-simple.patch"
} | tee "$RESULTS/system.txt"

# The governor decides the clock: Raspberry Pi OS defaults to `ondemand`,
# which can leave a core below full speed while it ramps. `performance`
# holds the maximum, as the bare-metal program does.
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee "$gov" > /dev/null
done
echo "governor set to: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)" | tee -a "$RESULTS/system.txt"

# --- The model ------------------------------------------------------------

MODEL=$WORK/models/$MODEL_NAME
if [ ! -f "$MODEL" ]; then
    mkdir -p "$WORK/models"
    echo "fetching $MODEL_NAME"
    curl -fsSL --proto '=https' -o "$MODEL.part" "$MODEL_URL"
    mv "$MODEL.part" "$MODEL"
fi
expected=$(awk -v name="$MODEL_NAME" '$2 == name { print $1 }' "$REPO/models.sha256")
actual=$(sha256sum < "$MODEL" | cut -d' ' -f1)
if [ "$expected" != "$actual" ]; then
    echo "$MODEL_NAME: checksum mismatch (expected $expected, got $actual)" >&2
    exit 1
fi
echo "model: $MODEL (sha256 ok)"

# --- llama.cpp ------------------------------------------------------------

LLAMA=$WORK/llama.cpp
if [ ! -x "$LLAMA/build/bin/llama-simple" ]; then
    rm -rf "$LLAMA"
    git init -q "$LLAMA"
    git -C "$LLAMA" fetch -q --depth 1 "$LLAMA_REPO" "$LLAMA_COMMIT"
    git -C "$LLAMA" checkout -q FETCH_HEAD
    git -C "$LLAMA" apply "$REPO/bench/linux/llama-simple.patch"
    cmake -S "$LLAMA" -B "$LLAMA/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > "$RESULTS/llama-cmake.log"
    cmake --build "$LLAMA/build" --target llama-simple -j "$(nproc)" > "$RESULTS/llama-build.log"
fi
echo "llama.cpp: $LLAMA/build/bin/llama-simple"

# --- rpi-llm --------------------------------------------------------------

# The repository's .cargo/config.toml points cargo at the bare-metal
# target, so the host target is named explicitly.
HOST=$(rustc -vV | sed -n 's/^host: //p')
cargo build -q --release --manifest-path "$REPO/bench/Cargo.toml" --target "$HOST"
OURS=$REPO/bench/target/$HOST/release/rpi-llm-bench
echo "rpi-llm: $OURS"

# --- The runs -------------------------------------------------------------

# The prompt as an argument, its final newline kept: `$(cat ...)` alone
# would strip it, and change the prompt by one token.
PROMPT=$(cat "$PROMPT_FILE"; printf x)
PROMPT=${PROMPT%x}

drop_caches() {
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
}

# run <name> <threads> <command...>: runs it cold, keeps its output, and
# appends one row of numbers to rows.txt.
run() {
    local name=$1 threads=$2
    shift 2
    local log=$RESULTS/$name-$threads.log
    drop_caches
    echo "running $name with $threads thread(s)"
    "$@" > "$RESULTS/$name-$threads.out" 2> "$log"
    local bench
    bench=$(grep '^BENCH ' "$log" | tail -1)
    local temp=""
    if command -v vcgencmd > /dev/null; then
        temp=$(vcgencmd measure_temp | sed 's/temp=//')
    fi
    echo "$name $threads $temp $bench" >> "$RESULTS/rows.txt"
}

: > "$RESULTS/rows.txt"
for threads in "${THREADS[@]}"; do
    run llama.cpp "$threads" env RPI_LLM_THREADS="$threads" \
        "$LLAMA/build/bin/llama-simple" -m "$MODEL" -n "$ANSWER_TOKENS" "$PROMPT"
    run rpi-llm "$threads" "$OURS" "$MODEL" --threads "$threads"
done

if command -v vcgencmd > /dev/null; then
    echo "throttled after: $(vcgencmd get_throttled)" | tee -a "$RESULTS/system.txt"
fi

# --- The table ------------------------------------------------------------

awk -v size="$(stat -c %s "$MODEL")" '
function field(name,    i, kv) {
    for (i = 5; i <= NF; i++) {
        split($i, kv, "=")
        if (kv[1] == name) return kv[2]
    }
    return ""
}
BEGIN {
    printf "%-10s %7s %9s %9s %12s %12s %7s %6s\n",
        "program", "threads", "load s", "load MB/s", "prompt tok/s", "answer tok/s", "match", "temp"
}
{
    load = field("load_us") / 1e6
    pt = field("prompt_tokens"); pu = field("prompt_us") / 1e6
    at = field("answer_tokens"); au = field("answer_us") / 1e6
    match_ = field("match"); if (match_ == "") match_ = "-"
    printf "%-10s %7s %9.2f %9.1f %12.1f %12.1f %7s %6s\n",
        $1, $2, load, size / load / 1e6, pt / pu, (at - 1) / au, match_, $3
}' "$RESULTS/rows.txt" | tee "$RESULTS/summary.txt"
