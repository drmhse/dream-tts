//! A channels-last causal conv that gathers its taps inside the GEMM.
//!
//! Both conv-as-GEMM routes in this crate lose to *assembling* the matrix rather than to
//! multiplying it. `convgemm` at the qwen3tts codec's widest stage — 96 channels over 131072
//! samples, k=7 d=9 — measured **85.8 ms to build the `[672, 131072]` im2col against 8.2 ms
//! for the GEMM that consumes it**, and the channels-last route pays the same toll in a
//! different currency: `cat` over `k` narrowed views writes `k * L * C_in` and reads it back.
//!
//! The tap index is arithmetic, not data. Doing it where the tile already sits in threadgroup
//! memory costs nothing and removes the whole matrix from device memory.

use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};

/// `x [L, cin]`, `w [k * cin, cout]` tap-major, `bias [cout]` -> `y [L, cout]`.
pub(crate) struct NlcConv {
    pub len: usize,
    pub cin: usize,
    pub cout: usize,
    pub k: usize,
    pub dilation: usize,
    pub has_bias: bool,
    /// SnakeBeta `(alpha, 1/beta)` applied to `x` as it loads.
    pub snake: Option<(Tensor, Tensor)>,
    /// `[L, cout]` added to the output.
    pub res: Option<Tensor>,
}

impl CustomOp3 for NlcConv {
    fn name(&self) -> &'static str {
        "nlc_conv"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if self.snake.is_some() || self.res.is_some() {
            candle_core::bail!("nlc_conv: snake and residual are Metal only");
        }
        let (x, w, b) = match (s1, s2, s3) {
            (CpuStorage::F32(x), CpuStorage::F32(w), CpuStorage::F32(b)) => (x, w, b),
            _ => candle_core::bail!("nlc_conv: only f32"),
        };
        let (ox, ow, ob) = (l1.start_offset(), l2.start_offset(), l3.start_offset());
        let pad = ((self.k - 1) * self.dilation) as isize;
        let mut dst = vec![0f32; self.len * self.cout];
        for l in 0..self.len {
            let row = &mut dst[l * self.cout..(l + 1) * self.cout];
            for (co, r) in row.iter_mut().enumerate() {
                *r = if self.has_bias { b[ob + co] } else { 0.0 };
            }
            for t in 0..self.k {
                let s = l as isize + (t * self.dilation) as isize - pad;
                if s < 0 || s >= self.len as isize {
                    continue;
                }
                let xr = ox + s as usize * self.cin;
                for ci in 0..self.cin {
                    let xv = x[xr + ci];
                    let wr = ow + (t * self.cin + ci) * self.cout;
                    for (co, r) in row.iter_mut().enumerate() {
                        *r += xv * w[wr + co];
                    }
                }
            }
        }
        Ok((CpuStorage::F32(dst), (1, self.len, self.cout).into()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
        s3: &candle_core::MetalStorage,
        l3: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use crate::mtl;
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        if !l1.is_contiguous() || !l2.is_contiguous() || !l3.is_contiguous() {
            candle_core::bail!("nlc_conv: inputs must be contiguous");
        }
        if s1.dtype() != DType::F32 || s2.dtype() != DType::F32 || s3.dtype() != DType::F32 {
            candle_core::bail!("nlc_conv: only f32");
        }
        let device = s1.device();
        let p = mtl::pipeline(device, "nlc_conv_f32")?;
        let n = self.len * self.cout;
        let dst = device.new_buffer(n, DType::F32, "nlc_conv")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::nlc_conv");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(s1.buffer()), l1.start_offset() * 4);
        encoder.set_buffer(1, Some(s2.buffer()), l2.start_offset() * 4);
        encoder.set_buffer(2, Some(s3.buffer()), l3.start_offset() * 4);
        encoder.set_buffer(3, Some(dst.as_ref()), 0);
        encoder.set_bytes(4, &(self.len as u32));
        encoder.set_bytes(5, &(self.cin as u32));
        encoder.set_bytes(6, &(self.cout as u32));
        encoder.set_bytes(7, &(self.k as u32));
        encoder.set_bytes(8, &(self.dilation as u32));
        encoder.set_bytes(9, &(u32::from(self.has_bias)));
        let buf = |t: &Tensor| -> Result<_> {
            let (st, l) = t.storage_and_layout();
            match &*st {
                candle_core::Storage::Metal(m) => Ok((m.buffer().clone(), l.start_offset() * 4)),
                _ => candle_core::bail!("nlc_conv: operand must be on the device"),
            }
        };
        let snake = match &self.snake {
            Some((a, b)) => Some((buf(a)?, buf(b)?)),
            None => None,
        };
        let res = self.res.as_ref().map(buf).transpose()?;
        // Unused operands still need a binding; the input serves.
        let fallback = (s1.buffer().clone(), 0usize);
        let (alpha, brecip) = snake
            .clone()
            .unwrap_or((fallback.clone(), fallback.clone()));
        let r = res.clone().unwrap_or(fallback);
        encoder.set_buffer(10, Some(&alpha.0), alpha.1);
        encoder.set_buffer(11, Some(&brecip.0), brecip.1);
        encoder.set_buffer(12, Some(&r.0), r.1);
        encoder.set_bytes(13, &(u32::from(snake.is_some())));
        encoder.set_bytes(14, &(u32::from(res.is_some())));
        for b in [&alpha.0, &brecip.0, &r.0] {
            encoder.use_resource(b, MTLResourceUsage::Read);
        }
        encoder.use_resource(s1.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(s3.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        // One 64x32 output tile per threadgroup, channel tiles on the fast axis. The grid
        // is rounded up to whole tiles so
        // every threadgroup is full — the tile cooperates through `threadgroup_barrier`, and
        // a short trailing group would deadlock on it.
        let tiles_l = self.len.div_ceil(64);
        let tiles_n = self.cout / 32;
        encoder.dispatch_threads(
            MTLSize {
                width: tiles_n * 16,
                height: tiles_l * 8,
                depth: 1,
            },
            MTLSize {
                width: 16,
                height: 8,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), n, DType::F32),
            (1, self.len, self.cout).into(),
        ))
    }
}

/// Shortest signal the fused kernel is used on.
///
/// It wins by keeping a long signal out of device memory, and loses when there is not enough
/// of one to fill the machine: the codec's `head_conv` — 1024 to 1536 channels over 1200
/// positions — measured 14.1 ms fused against 12.4 through `cat` and MPS, while the decoder
/// blocks it exists for run 9600 to 576000 positions.
const MIN_LEN: usize = 4096;

/// Whether [`crate::nlc::causal_conv1d`] can take the fused path for these shapes.
///
/// `cin % 32` is what lets the kernel resolve the tap with one division per reduction chunk
/// instead of one per element, and `cout % 32` is its output tile. Every conv in the qwen3tts
/// decoder's waveform stack satisfies both; the 1-channel output conv does not, and takes the
/// `cat` route.
pub(crate) fn eligible(b: usize, cin: usize, cout: usize, len: usize, x: &Tensor) -> bool {
    // A/B switch: this kernel is only worth its shapes, and the two routes have to be
    // comparable in one thermal state to know that.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !ON.get_or_init(|| std::env::var("TTS_NN_FUSED_CONV").as_deref() != Ok("0")) {
        return false;
    }
    b == 1
        && len >= MIN_LEN
        && cin.is_multiple_of(32)
        && cout.is_multiple_of(32)
        && x.dtype() == candle_core::DType::F32
        && x.device().is_metal()
}
