# rpi-llm, for the Raspberry Pi 5 (preliminary) (aarch64-unknown-none-softfloat).
#
#   make                          build target/kernel8.img
#   make run DEVICE=/dev/ttyUSB0  build it, upload it over the UART with
#                                 rpi-loader, and boot it
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

.PHONY: all build run sd-card clean

all: build

build:
	cargo build --release
	cargo objcopy --release -- -O binary target/$(KERNEL)
	@echo "built target/$(KERNEL)"

run: build
	rpi-loader --device $(DEVICE) boot --load-addr $(LOAD_ADDR) target/$(KERNEL)

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
