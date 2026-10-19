# rpi-llm, for the Raspberry Pi 5 (preliminary) (aarch64-unknown-none).
#
#   make                          build target/kernel8.img
#   make run DEVICE=/dev/ttyUSB0  build it, upload it over the UART with
#                                 rpi-loader, and boot it
#   make test                     the engine's tests, on this machine
#   make models                   fetch the model files into models/, for
#                                 the tests and for the card's models/
#   make sd-card                  fetch what goes on the SD card into
#                                 sd-card/ (firmware + the rpi-loader image)
#
# `cargo objcopy` needs cargo-binutils: `cargo install cargo-binutils`.
# `make run` needs the loader's host CLI: `cargo install rpi-loader`.

KERNEL := kernel8.img
# Where rpi-hal's linker script links this target, and so where the
# loader has to put the image.
LOAD_ADDR := 0x80000
DEVICE ?= /dev/ttyUSB0

# What `make sd-card` fetches. Each file is checked against
# sd-card.sha256 before it is kept.
FIRMWARE_URL := https://raw.githubusercontent.com/raspberrypi/firmware/1.20260824/boot
FIRMWARE_FILES := 
LOADER_URL := https://github.com/joeferner/rpi-hal/releases/download/loader-v0.6.0
LOADER_IMAGE := rpi-loader-0.6.0-bcm2712-kernel8.img

# What `make models` fetches, checked against models.sha256.
MODEL_FILES := stories15M.bin tokenizer.bin smollm2-360m-instruct-q8_0.gguf
MODEL_URL_stories15M.bin := https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
MODEL_URL_tokenizer.bin := https://raw.githubusercontent.com/karpathy/llama2.c/350e04f/tokenizer.bin
MODEL_URL_smollm2-360m-instruct-q8_0.gguf := https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf

.PHONY: all build run test models sd-card clean

# The host's own target, which `make test` builds the engine for:
# `.cargo/config.toml` points plain `cargo` at the board.
HOST := $(shell rustc -vV | sed -n 's/^host: //p')

all: build

build:
	cargo build --release
	cargo objcopy --release -- -O binary target/$(KERNEL)
	@echo "built target/$(KERNEL)"

run: build
	rpi-loader --device $(DEVICE) boot --load-addr $(LOAD_ADDR) target/$(KERNEL)

# The engine's tests, on this machine rather than the board. The one
# that compares against llama2.c needs `make models` first, and skips
# without it.
test:
	cargo test --release -p rpi-llm-engine --target $(HOST)

# The model files, for the tests and for copying onto the card's models/.
models: $(addprefix models/,$(MODEL_FILES))

models/%: models.sha256
	@mkdir -p models
	@echo "fetching $*"
	@curl -fsSL --proto '=https' -o models/$*.part $(MODEL_URL_$*)
	@expected=$$(awk -v name="$*" '$$2 == name { print $$1 }' models.sha256); \
	 actual=$$(sha256sum < models/$*.part | cut -d' ' -f1); \
	 if [ "$$expected" != "$$actual" ]; then \
	   echo "$*: checksum mismatch (expected $$expected, got $$actual)" >&2; \
	   rm -f models/$*.part; exit 1; \
	 fi
	@mv models/$*.part models/$*

# Everything the card's FAT32 boot partition needs: the Raspberry Pi boot
# firmware, config.txt, and the rpi-loader image renamed to $(KERNEL) --
# the only filename the firmware will load on this board. Copy the whole
# of sd-card/ onto the card.
sd-card: $(addprefix sd-card/,$(FIRMWARE_FILES)) sd-card/$(KERNEL)
	cd sd-card && sha256sum -c ../sd-card.sha256
	@echo "copy everything in sd-card/ onto the card's boot partition"

# $(call fetch,<url>,<name on the card>). Downloads to a temporary name and
# only moves the file into place once its checksum matches, so a truncated
# transfer leaves nothing that a later `make` would mistake for a good copy.
define fetch
	@mkdir -p sd-card
	@echo "fetching $(2)"
	@curl -fsSL --proto '=https' -o sd-card/$(2).part $(1)
	@expected=$$(awk -v name="$(2)" '$$2 == name { print $$1 }' sd-card.sha256); \
	 actual=$$(sha256sum < sd-card/$(2).part | cut -d' ' -f1); \
	 if [ "$$expected" != "$$actual" ]; then \
	   echo "$(2): checksum mismatch (expected $$expected, got $$actual)" >&2; \
	   rm -f sd-card/$(2).part; exit 1; \
	 fi
	@mv sd-card/$(2).part sd-card/$(2)
endef

sd-card/$(KERNEL): sd-card.sha256
	$(call fetch,$(LOADER_URL)/$(LOADER_IMAGE),$(KERNEL))

sd-card/%: sd-card.sha256
	$(call fetch,$(FIRMWARE_URL)/$*,$*)

clean:
	cargo clean
