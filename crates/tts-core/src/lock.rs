//! One engine on the GPU at a time.
//!
//! A single render peaks well above what a 16 GB machine can spare alongside a second
//! engine, and two resident engines drive it into swap — which looks like the models
//! getting slower rather than like a mistake. The narration scripts guarded this with
//! `pgrep` against `target/release/tts`, a heuristic that broke the moment binaries could
//! also live in `bin/`, and that never saw a second process started any other way.
//!
//! So the guard lives here instead, as an advisory `flock` on a file in the data
//! directory. Advisory is the right strength: it is per-weights rather than per-machine,
//! two independent installs do not contend, and anything that declines to take it is
//! simply not covered rather than blocked.
//!
//! Held for the lifetime of the value. Dropping it, including on a panic or a signal that
//! unwinds, releases it; a process killed outright releases it too, because the kernel
//! closes the descriptor. That is the property a pidfile does not have.

use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct GpuLock {
    // Held to keep the descriptor open: the lock is the descriptor.
    file: File,
    path: PathBuf,
}

impl GpuLock {
    /// Take the lock, or fail immediately naming whoever holds it.
    ///
    /// Never blocks. A caller that waits silently is indistinguishable from one that hung,
    /// and these operations run for minutes.
    pub fn acquire(path: &Path, what: &str) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("creating {} for the GPU lock", dir.display()))?;
        }
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("opening the GPU lock at {}", path.display()))?;

        // SAFETY: `file` owns the descriptor for the whole call.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
                anyhow::bail!(
                    "another tts process holds the GPU{}.\n  \
                     Two resident engines do not fit in 16 GB and the machine swaps rather \
                     than failing, so this refuses instead.\n  \
                     Wait for it, stop it, or pass --no-gpu-lock to opt out.\n  \
                     Lock file: {}",
                    describe_holder(&mut file),
                    path.display()
                );
            }
            return Err(err).with_context(|| format!("locking {}", path.display()));
        }

        // Only now that it is ours: record who, for the next process's error message.
        file.set_len(0).ok();
        file.seek(SeekFrom::Start(0)).ok();
        let _ = writeln!(file, "pid {}\n{}", std::process::id(), what);
        let _ = file.flush();

        Ok(Self {
            file,
            path: path.to_path_buf(),
        })
    }

    /// Take it only if `enabled`. Keeps the call site free of a conditional whose two
    /// branches have different types.
    pub fn maybe(enabled: bool, path: &Path, what: &str) -> Result<Option<Self>> {
        if enabled {
            Ok(Some(Self::acquire(path, what)?))
        } else {
            Ok(None)
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for GpuLock {
    fn drop(&mut self) {
        // Unlock explicitly rather than relying on close, so the ordering is visible. The
        // file is left in place: unlinking a locked file races with the next acquirer,
        // which would open the unlinked inode and think it succeeded.
        // SAFETY: the descriptor is still open here.
        unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
    }
}

/// The holder's own description, if it left one. Best effort: a truncated or empty file
/// means a process that died between creating and writing, which is not worth reporting as
/// an error of its own.
fn describe_holder(file: &mut File) -> String {
    let mut s = String::new();
    if file.seek(SeekFrom::Start(0)).is_err() || file.read_to_string(&mut s).is_err() {
        return String::new();
    }
    let held = s.trim();
    if held.is_empty() {
        String::new()
    } else {
        format!(" ({})", held.replace('\n', ", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("tts-lock-test-{}-{}", std::process::id(), name));
        p
    }

    #[test]
    fn a_second_acquire_in_process_reports_the_first() {
        // flock is per open file description, so a second `open` in this same process
        // contends exactly as another process would.
        let path = tmp("contend");
        let _held = GpuLock::acquire(&path, "tts speak --engine qwen3tts").unwrap();
        let err = GpuLock::acquire(&path, "second").expect_err("must not double-acquire");
        let msg = err.to_string();
        assert!(msg.contains("holds the GPU"), "{msg}");
        assert!(msg.contains("tts speak --engine qwen3tts"), "{msg}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn releasing_lets_the_next_one_in() {
        let path = tmp("release");
        drop(GpuLock::acquire(&path, "first").unwrap());
        let _second = GpuLock::acquire(&path, "second").expect("released");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn maybe_disabled_takes_nothing() {
        let path = tmp("disabled");
        let none = GpuLock::maybe(false, &path, "x").unwrap();
        assert!(none.is_none());
        let _held = GpuLock::acquire(&path, "still free").unwrap();
        std::fs::remove_file(&path).ok();
    }
}
