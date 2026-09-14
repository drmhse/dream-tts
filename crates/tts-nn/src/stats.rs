//! Counted dense-GEMM traffic for the achieved-rate line.
//!
//! A render's RTF says how fast it went; this says what the GEMMs did while it
//! went there. Every dense matmul in this crate reports its exact FLOP count
//! (`2*m*k*n`) and its exact element traffic (inputs read, output written, in
//! the dtypes actually moved) into a thread-local accumulator. The engine takes
//! the totals around synthesis and prints achieved TFLOP/s and GB/s.
//!
//! Coverage is dense GEMMs only: candle's direct convolutions, attention
//! scores, and all elementwise passes are not counted, so the GB/s reads low
//! against system DRAM bandwidth by construction. What it answers is whether
//! the GEMM shapes are near the 2.4 TFLOP/s this backend reaches elsewhere —
//! and the recording itself is integer adds, never a dispatch.

use std::cell::Cell;

thread_local! {
    static FLOPS: Cell<u64> = const { Cell::new(0) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
}

/// Add one matmul: `[m, k] x [k, n]`, with the bytes its three tensors move.
pub fn record(m: usize, k: usize, n: usize, bytes: u64) {
    let flops = 2 * m as u64 * k as u64 * n as u64;
    FLOPS.with(|f| f.set(f.get().saturating_add(flops)));
    BYTES.with(|b| b.set(b.get().saturating_add(bytes)));
}

/// Take the totals and reset. Renders reset before synthesis and take after.
pub fn take() -> (u64, u64) {
    let out = (FLOPS.with(|f| f.get()), BYTES.with(|b| b.get()));
    FLOPS.with(|f| f.set(0));
    BYTES.with(|b| b.set(0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_and_reset() {
        take();
        record(48, 2048, 4096, 100);
        record(48, 2048, 4096, 100);
        let (flops, bytes) = take();
        assert_eq!(flops, 2 * 2 * 48 * 2048 * 4096);
        assert_eq!(bytes, 200);
        assert_eq!(take(), (0, 0));
    }
}
