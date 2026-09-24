//! What the machine can take.
//!
//! One engine resident is gigabytes, and two on a 16 GB machine drive it into swap — which
//! presents as the models getting slower rather than as a mistake, so it is worth refusing
//! or at least warning about. But "refuse a second engine" is wrong on a 64 GB machine,
//! where running two books at once is a reasonable thing to want.
//!
//! So this reports rather than decides: how much memory the machine has, what an engine
//! costs, and therefore how many fit. The caller says what it thinks and the user chooses.

/// Bytes of physical memory, or `None` where it cannot be asked.
///
/// Two bodies rather than one with `cfg` branches inside: sharing a signature across
/// platforms that answer the question in unrelated ways buys nothing and needs a `return`
/// that reads as a mistake.
#[cfg(target_os = "macos")]
pub fn total_memory() -> Option<u64> {
    let mut value: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    // SAFETY: `hw.memsize` is a `uint64_t`, and `size` describes the buffer exactly.
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&raw mut value).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0).then_some(value)
}

/// This process's physical footprint in bytes — what Activity Monitor calls Memory, and what
/// decides whether a render swaps. `None` off macOS.
#[cfg(target_os = "macos")]
pub fn footprint() -> Option<u64> {
    let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
    // SAFETY: `info` is a `rusage_info_v2`, which is what flavor 2 writes.
    let rc = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as i32,
            libc::RUSAGE_INFO_V2,
            (&raw mut info).cast(),
        )
    };
    (rc == 0).then_some(info.ri_phys_footprint)
}

#[cfg(not(target_os = "macos"))]
pub fn footprint() -> Option<u64> {
    None
}

/// `MemTotal: N kB`, the first line of `/proc/meminfo`.
#[cfg(not(target_os = "macos"))]
pub fn total_memory() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: u64 = text
        .lines()
        .find_map(|l| l.strip_prefix("MemTotal:"))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    Some(kb * 1024)
}

/// Roughly what one resident engine costs, in bytes.
///
/// A single figure rather than per-engine: the three are within a factor of two of each
/// other at their default weight formats, and the decision this informs — one more engine or
/// not — does not turn on the difference. Measured, not derived from parameter counts, which
/// undercount the KV cache and the activation peak.
pub const ENGINE_FOOTPRINT: u64 = 4 * 1024 * 1024 * 1024;

/// How many engines this machine can hold, keeping a quarter of memory for everything else.
///
/// `None` when the memory could not be read: no advice beats invented advice.
pub fn engines_that_fit() -> Option<usize> {
    let total = total_memory()?;
    let usable = total - total / 4;
    Some((usable / ENGINE_FOOTPRINT).max(1) as usize)
}

/// `16 GB`, for a message.
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit <= 1 {
        format!("{bytes} B")
    } else {
        format!("{value:.0} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn this_machine_reports_its_memory() {
        let total = total_memory().expect("a supported platform");
        // A sanity band rather than a value: this runs on whatever the developer has.
        assert!(
            total > 1 << 30,
            "less than a gigabyte is not a machine that runs this"
        );
        assert!(total < 1 << 42, "four terabytes is a misread, not a laptop");
    }

    #[test]
    fn at_least_one_engine_always_fits() {
        // Otherwise the advice would be "do not run this at all", which is not useful on a
        // machine that is already running it.
        assert!(engines_that_fit().is_some_and(|n| n >= 1));
    }

    #[test]
    fn bytes_read_as_a_person_would_say_them() {
        assert_eq!(human_bytes(16 * 1024 * 1024 * 1024), "16 GB");
        assert_eq!(human_bytes(512), "512 B");
    }
}
