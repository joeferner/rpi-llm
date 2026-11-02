//! The NVMe SSD on the Pi 5's external PCIe connector, as a second place
//! to read the model from.
//!
//! [`mount`] brings the connector's link up, finds the NVMe controller
//! behind it, starts it, and mounts the first FAT volume on the drive, the
//! same way the card's is mounted. Everything here is optional: with no
//! SSD, or one that will not come up, it says why on the console and the
//! program carries on with the card alone.

use rpi_hal::nvme::{self, Nvme, NvmeBlockDevice};
use rpi_hal::pac::PCIE1;
use rpi_hal::pcie::{LinkSpeed, Pcie};
use rpi_hal::timer::Timer;
use rpi_kickstart::{logln, storage};
use static_cell::ConstStaticCell;

/// The controller's queues and scratch page, which it reads and writes
/// directly, so they have to stay put for as long as the drive is up.
static MEMORY: ConstStaticCell<nvme::Memory> = ConstStaticCell::new(nvme::Memory::new());

/// A mounted volume on the SSD.
pub type Volume<'t> = storage::Volume<NvmeBlockDevice<'static, 't>>;

/// Brings up the SSD and mounts its first FAT volume, or logs why not and
/// returns `None`.
///
/// Gen 2 is the most the Pi 5's connector is rated for.
pub fn mount<'t>(pcie1: PCIE1, timer: &'t Timer) -> Option<Volume<'t>> {
    let started = timer.now_micros();
    let mut pcie = Pcie::init(pcie1, LinkSpeed::Gen2, timer)
        .inspect_err(|e| logln!("ssd: no PCIe link: {e:?}"))
        .ok()?;
    let link = pcie.link();
    logln!(
        "ssd: PCIe link up in {} ms: Gen {} x{}",
        (timer.now_micros() - started) / 1000,
        link.generation,
        link.width,
    );

    let Some(info) = pcie.functions().find(|f| f.class == nvme::CLASS) else {
        logln!("ssd: no NVMe controller on the bus");
        return None;
    };
    let function = pcie
        .enable(info)
        .inspect_err(|e| logln!("ssd: enabling the controller failed: {e:?}"))
        .ok()?;

    let nvme = Nvme::new(&function, MEMORY.take(), timer)
        .inspect_err(|e| logln!("ssd: controller bring-up failed: {e:?}"))
        .ok()?;
    logln!(
        "ssd: {} ({} GB, {}-byte blocks), up to {} KiB per command",
        nvme.model(),
        nvme.block_count() * nvme.block_size() as u64 / 1_000_000_000,
        nvme.block_size(),
        nvme.max_transfer() / 1024,
    );

    let device = NvmeBlockDevice::new(nvme, timer)
        .inspect_err(|e| logln!("ssd: {e:?}"))
        .ok()?;
    storage::mount(device)
        .inspect_err(|e| logln!("ssd: storage: {e:?}"))
        .ok()
}

/// Unmounts `volume` and shuts the drive down, the way NVMe asks before
/// power goes: nothing reads the SSD again once the model is in RAM.
pub fn shut_down(volume: Volume, timer: &Timer) {
    let device = match volume.unmount() {
        Ok(device) => device,
        Err(e) => {
            logln!("ssd: unmount: {e:?}");
            return;
        }
    };
    match device.into_inner().shutdown(timer) {
        Ok(()) => logln!("ssd: shut down"),
        Err(e) => logln!("ssd: shutdown: {e:?}"),
    }
}
