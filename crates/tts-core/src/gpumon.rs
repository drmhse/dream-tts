//! Live GPU telemetry without root.
//!
//! `powermetrics` needs the superuser, but the AGX accelerator node publishes
//! `PerformanceStatistics` — device/renderer/tiler utilization and memory
//! footprints — to any reader through IOKit. Sampled live on this M4 at 5% idle
//! against 91% under a Kokoro render, so the numbers below are checked, not
//! assumed.
//!
//! What this deliberately does not cover: system memory bandwidth. macOS exposes
//! no unprivileged byte counters — only residency and utilization — so a live
//! GB/s number needs root (`powermetrics --samplers bandwidth`) and is not
//! attempted here. Per-kernel GB/s stays where it belongs, in `tts-probe`
//! microbenchmarks, and `tts_nn::stats` counts the dense-GEMM traffic of a
//! render for the achieved-rate line the engines print.

#[cfg(target_os = "macos")]
mod imp {
    use std::ffi::CString;
    use std::os::raw::{c_char, c_void};

    type CFTypeRef = *const c_void;
    type CFDictionaryRef = *const c_void;
    type CFMutableDictionaryRef = *mut c_void;
    type CFStringRef = *const c_void;

    #[link(name = "IOKit", kind = "framework")]
    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn IOServiceMatching(name: *const c_char) -> CFMutableDictionaryRef;
        fn IOServiceGetMatchingServices(
            master: u32,
            matching: CFDictionaryRef,
            existing: *mut u32,
        ) -> i32;
        fn IOIteratorNext(iterator: u32) -> u32;
        fn IOObjectRelease(object: u32) -> i32;
        fn IORegistryEntryCreateCFProperties(
            entry: u32,
            properties: *mut CFTypeRef,
            allocator: CFTypeRef,
            options: u32,
        ) -> i32;
        fn CFDictionaryGetValue(dict: CFDictionaryRef, key: CFStringRef) -> CFTypeRef;
        fn CFStringCreateWithCString(
            alloc: CFTypeRef,
            cstr: *const c_char,
            encoding: u32,
        ) -> CFStringRef;
        fn CFNumberGetValue(number: CFTypeRef, the_type: i32, value_ptr: *mut c_void) -> u8;
        fn CFRelease(obj: CFTypeRef);
    }

    const UTF8: u32 = 0x0800_0100;
    const SINT64: i32 = 4;

    /// One poll of the accelerator node. `None` is never an error worth
    /// propagating: a monitor must not crash a render, on this machine or on one
    /// without an AGX node at all.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Sample {
        pub device_pct: u64,
        pub renderer_pct: u64,
        pub tiler_pct: u64,
        pub alloc_bytes: u64,
        pub in_use_bytes: u64,
    }

    pub struct Sampler {
        service: u32,
        stats_key: CFStringRef,
        device_key: CFStringRef,
        renderer_key: CFStringRef,
        tiler_key: CFStringRef,
        alloc_key: CFStringRef,
        in_use_key: CFStringRef,
    }

    // IOKit and CoreFoundation objects are thread-safe handles; the sampler is
    // `Send` so a sidecar thread can own it.
    unsafe impl Send for Sampler {}

    impl Drop for Sampler {
        fn drop(&mut self) {
            // SAFETY: every handle below came from the create/copy/match call
            // named beside it, each of which returns one owned reference.
            unsafe {
                IOObjectRelease(self.service); // IOServiceGetMatchingServices
                for k in [
                    self.stats_key,
                    self.device_key,
                    self.renderer_key,
                    self.tiler_key,
                    self.alloc_key,
                    self.in_use_key,
                ] {
                    CFRelease(k); // CFStringCreateWithCString
                }
            }
        }
    }

    fn cfstr(s: &str) -> CFStringRef {
        let c = CString::new(s).expect("telemetry keys have no interior nul");
        // SAFETY: null allocator is the default, UTF-8 is always available.
        unsafe { CFStringCreateWithCString(std::ptr::null(), c.as_ptr(), UTF8) }
    }

    fn get_int(dict: CFDictionaryRef, key: CFStringRef) -> Option<u64> {
        // SAFETY: dict and key are live CF objects from the registry read or the
        // sampler's own cache; a missing key or a non-number value is `None`.
        unsafe {
            let v = CFDictionaryGetValue(dict, key);
            if v.is_null() {
                return None;
            }
            let mut out: i64 = 0;
            (CFNumberGetValue(v, SINT64, (&mut out as *mut i64).cast()) != 0).then_some(out as u64)
        }
    }

    impl Sampler {
        /// The first `IOAccelerator` whose statistics name a device utilization.
        /// `None` on machines without one — VMs, non-Apple GPUs — which is an
        /// answer, not a failure.
        pub fn open() -> Option<Self> {
            // The keys live in locals until the service is known, so no half-built
            // `Sampler` — and its `Drop` — ever exists.
            let stats_key = cfstr("PerformanceStatistics");
            let svc = Self::find_service(stats_key);
            // SAFETY: released exactly once per `cfstr` call, on every path.
            let release = |k: CFStringRef| unsafe { CFRelease(k) };
            let svc = match svc {
                Some(s) => s,
                None => {
                    release(stats_key);
                    return None;
                }
            };
            Some(Self {
                service: svc,
                stats_key,
                device_key: cfstr("Device Utilization %"),
                renderer_key: cfstr("Renderer Utilization %"),
                tiler_key: cfstr("Tiler Utilization %"),
                alloc_key: cfstr("Alloc system memory"),
                in_use_key: cfstr("In use system memory"),
            })
        }

        fn find_service(stats_key: CFStringRef) -> Option<u32> {
            // SAFETY: class names are static C strings; a null return is checked.
            // Master port 0 is the default port, which is what `ioreg` queries.
            // The matching dictionary is consumed by a successful call and must
            // only be released when the call fails (verified: releasing after
            // success traps on the double free).
            let iter = unsafe {
                let matching = IOServiceMatching(c"IOAccelerator".as_ptr());
                if matching.is_null() {
                    return None;
                }
                let mut iter = 0u32;
                let rc = IOServiceGetMatchingServices(0, matching, &mut iter);
                if rc != 0 {
                    CFRelease(matching);
                    return None;
                }
                if iter == 0 {
                    return None;
                }
                iter
            };
            // SAFETY: the iterator yields owned service handles, one per call;
            // the first handle with counters is returned, the rest released.
            let mut kept = None;
            loop {
                let next = unsafe { IOIteratorNext(iter) };
                if next == 0 {
                    break;
                }
                if kept.is_none() && unsafe { Self::has_stats(next, stats_key) } {
                    kept = Some(next);
                } else {
                    unsafe { IOObjectRelease(next) };
                }
            }
            unsafe { IOObjectRelease(iter) };
            kept
        }

        unsafe fn has_stats(service: u32, stats_key: CFStringRef) -> bool {
            let mut props: CFTypeRef = std::ptr::null();
            if IORegistryEntryCreateCFProperties(service, &mut props, std::ptr::null(), 0) != 0
                || props.is_null()
            {
                return false;
            }
            let stats = CFDictionaryGetValue(props, stats_key);
            let ok = !stats.is_null();
            CFRelease(props);
            ok
        }

        pub fn sample(&self) -> Option<Sample> {
            // SAFETY: `service` is owned by this sampler; the property dict is
            // released before return on every path.
            unsafe {
                let mut props: CFTypeRef = std::ptr::null();
                if IORegistryEntryCreateCFProperties(self.service, &mut props, std::ptr::null(), 0)
                    != 0
                    || props.is_null()
                {
                    return None;
                }
                let stats = CFDictionaryGetValue(props, self.stats_key);
                let out = if stats.is_null() {
                    None
                } else {
                    Some(Sample {
                        device_pct: get_int(stats, self.device_key)?,
                        renderer_pct: get_int(stats, self.renderer_key)?,
                        tiler_pct: get_int(stats, self.tiler_key)?,
                        alloc_bytes: get_int(stats, self.alloc_key)?,
                        in_use_bytes: get_int(stats, self.in_use_key)?,
                    })
                };
                CFRelease(props);
                out
            }
        }
    }
}

#[cfg(target_os = "macos")]
pub use imp::{Sample, Sampler};

/// Off macOS there is no AGX node to read. The API is the same so callers do not
/// branch: `open` reports the absence.
#[cfg(not(target_os = "macos"))]
mod imp {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Sample {
        pub device_pct: u64,
        pub renderer_pct: u64,
        pub tiler_pct: u64,
        pub alloc_bytes: u64,
        pub in_use_bytes: u64,
    }

    pub struct Sampler;

    impl Sampler {
        pub fn open() -> Option<Self> {
            None
        }

        pub fn sample(&self) -> Option<Sample> {
            None
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub use imp::{Sample, Sampler};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absence_is_an_answer() {
        // Must hold everywhere: CI VMs have no accelerator node, and a monitor
        // that panics there is worse than no monitor.
        if let Some(s) = Sampler::open() {
            let sample = s
                .sample()
                .expect("a matched accelerator answers its own counters");
            assert!(sample.device_pct <= 100, "not a percentage: {sample:?}");
            assert!(sample.renderer_pct <= 100, "not a percentage: {sample:?}");
            assert!(sample.tiler_pct <= 100, "not a percentage: {sample:?}");
            assert!(sample.in_use_bytes <= sample.alloc_bytes, "{sample:?}");
        }
    }
}
