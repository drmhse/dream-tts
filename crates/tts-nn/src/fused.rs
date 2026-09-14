//! Fused elementwise activations.
//!
//! `docs/reference.md#performance` Finding 2 measured snake at 11.5 ms for `[1, 96, 131072]` and
//! showed its five constituent ops sum to 13.7 ms — they match, which is the proof that
//! candle fuses nothing. A single pass costs what `affine` costs, 1.33 ms.
//!
//! Alpha folding (Finding 3) already removed both broadcasts from most call sites, taking
//! snake from five ops to three. This removes the remaining two round-trips.
//!
//! Both entry points fall back to the composed candle form off Metal, and the unit tests
//! check the kernels against exactly that form.

#[cfg(feature = "metal")]
use crate::mtl;
use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};

/// `x + sin^2(x)` in one pass — the folded snake, where `alpha` already lives in the
/// preceding conv's output weights.
struct SnakeFolded;

impl CustomOp1 for SnakeFolded {
    fn name(&self) -> &'static str {
        "snake_folded"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let src = match storage {
            CpuStorage::F32(s) => s,
            _ => candle_core::bail!("snake_folded: only f32"),
        };
        let n = layout.shape().elem_count();
        let start = layout.start_offset();
        if !layout.is_contiguous() {
            candle_core::bail!("snake_folded: input must be contiguous");
        }
        let dst = src[start..start + n]
            .iter()
            .map(|&x| x + x.sin().powi(2))
            .collect();
        Ok((CpuStorage::F32(dst), layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle_core::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        if !layout.is_contiguous() {
            candle_core::bail!("snake_folded: input must be contiguous");
        }
        if storage.dtype() != DType::F32 {
            candle_core::bail!("snake_folded: only f32, got {:?}", storage.dtype());
        }
        let n = layout.shape().elem_count();
        let device = storage.device();
        let p = mtl::pipeline(device, "snake_folded_f32")?;
        let dst = device.new_buffer(n, DType::F32, "snake_folded")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::snake_folded");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(storage.buffer()), layout.start_offset() * 4);
        encoder.set_buffer(1, Some(dst.as_ref()), 0);
        encoder.set_bytes(2, &(n as u32));
        encoder.use_resource(storage.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        let w = mtl::group_width(&p, n);
        encoder.dispatch_threads(
            MTLSize {
                width: n,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: w,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), n, DType::F32),
            layout.shape().clone(),
        ))
    }
}

/// `u + sin^2(u)` with `u = alpha[c] * x`, for `[1, C, L]` inputs.
struct SnakeAlpha {
    channels: usize,
    len: usize,
}

impl candle_core::CustomOp2 for SnakeAlpha {
    fn name(&self) -> &'static str {
        "snake_alpha"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (x, a) = match (s1, s2) {
            (CpuStorage::F32(x), CpuStorage::F32(a)) => (x, a),
            _ => candle_core::bail!("snake_alpha: only f32"),
        };
        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("snake_alpha: inputs must be contiguous");
        }
        let (o1, o2) = (l1.start_offset(), l2.start_offset());
        let mut dst = vec![0f32; self.channels * self.len];
        for c in 0..self.channels {
            let alpha = a[o2 + c];
            for l in 0..self.len {
                let u = alpha * x[o1 + c * self.len + l];
                dst[c * self.len + l] = u + u.sin().powi(2);
            }
        }
        Ok((CpuStorage::F32(dst), (1, self.channels, self.len).into()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("snake_alpha: inputs must be contiguous");
        }
        if s1.dtype() != DType::F32 || s2.dtype() != DType::F32 {
            candle_core::bail!("snake_alpha: only f32");
        }
        let n = self.channels * self.len;
        let device = s1.device();
        let p = mtl::pipeline(device, "snake_alpha_f32")?;
        let dst = device.new_buffer(n, DType::F32, "snake_alpha")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::snake_alpha");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(s1.buffer()), l1.start_offset() * 4);
        encoder.set_buffer(1, Some(s2.buffer()), l2.start_offset() * 4);
        encoder.set_buffer(2, Some(dst.as_ref()), 0);
        encoder.set_bytes(3, &(self.len as u32));
        encoder.use_resource(s1.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        let w = mtl::group_width(&p, self.len);
        encoder.dispatch_threads(
            MTLSize {
                width: self.len,
                height: self.channels,
                depth: 1,
            },
            MTLSize {
                width: w,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), n, DType::F32),
            (1, self.channels, self.len).into(),
        ))
    }
}

/// `x + beta_recip[c] * sin^2(alpha[c] * x)` for `[1, C, L]` inputs.
struct SnakeBeta {
    channels: usize,
    len: usize,
}

impl candle_core::CustomOp3 for SnakeBeta {
    fn name(&self) -> &'static str {
        "snake_beta"
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
        let (x, a, br) = match (s1, s2, s3) {
            (CpuStorage::F32(x), CpuStorage::F32(a), CpuStorage::F32(br)) => (x, a, br),
            _ => candle_core::bail!("snake_beta: only f32"),
        };
        for l in [l1, l2, l3] {
            if !l.is_contiguous() {
                candle_core::bail!("snake_beta: inputs must be contiguous");
            }
        }
        let (o1, o2, o3) = (l1.start_offset(), l2.start_offset(), l3.start_offset());
        let mut dst = vec![0f32; self.channels * self.len];
        for c in 0..self.channels {
            let (alpha, brecip) = (a[o2 + c], br[o3 + c]);
            for l in 0..self.len {
                let x = x[o1 + c * self.len + l];
                dst[c * self.len + l] = x + brecip * (alpha * x).sin().powi(2);
            }
        }
        Ok((CpuStorage::F32(dst), (1, self.channels, self.len).into()))
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
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        for l in [l1, l2, l3] {
            if !l.is_contiguous() {
                candle_core::bail!("snake_beta: inputs must be contiguous");
            }
        }
        for s in [s1, s2, s3] {
            if s.dtype() != DType::F32 {
                candle_core::bail!("snake_beta: only f32");
            }
        }
        let n = self.channels * self.len;
        let device = s1.device();
        let p = mtl::pipeline(device, "snake_beta_f32")?;
        let dst = device.new_buffer(n, DType::F32, "snake_beta")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::snake_beta");
        encoder.set_compute_pipeline_state(&p);
        for (i, (s, l)) in [(s1, l1), (s2, l2), (s3, l3)].iter().enumerate() {
            encoder.set_buffer(i, Some(s.buffer()), l.start_offset() * 4);
            encoder.use_resource(s.buffer(), MTLResourceUsage::Read);
        }
        encoder.set_buffer(3, Some(dst.as_ref()), 0);
        encoder.set_bytes(4, &(self.len as u32));
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        let w = mtl::group_width(&p, self.len);
        encoder.dispatch_threads(
            MTLSize {
                width: self.len,
                height: self.channels,
                depth: 1,
            },
            MTLSize {
                width: w,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), n, DType::F32),
            (1, self.channels, self.len).into(),
        ))
    }
}

/// `silu(gate) * up`, elementwise over identically shaped inputs.
struct SwigluMul {
    n: usize,
}

impl candle_core::CustomOp2 for SwigluMul {
    fn name(&self) -> &'static str {
        "swiglu_mul"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (g, u) = match (s1, s2) {
            (CpuStorage::F32(g), CpuStorage::F32(u)) => (g, u),
            _ => candle_core::bail!("swiglu_mul: only f32"),
        };
        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("swiglu_mul: inputs must be contiguous");
        }
        let (o1, o2) = (l1.start_offset(), l2.start_offset());
        let dst = (0..self.n)
            .map(|i| {
                let x = g[o1 + i];
                (x / (1.0 + (-x).exp())) * u[o2 + i]
            })
            .collect();
        Ok((CpuStorage::F32(dst), l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("swiglu_mul: inputs must be contiguous");
        }
        if s1.dtype() != DType::F32 || s2.dtype() != DType::F32 {
            candle_core::bail!("swiglu_mul: only f32");
        }
        let device = s1.device();
        let p = mtl::pipeline(device, "swiglu_mul_f32")?;
        let dst = device.new_buffer(self.n, DType::F32, "swiglu_mul")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::swiglu_mul");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(s1.buffer()), l1.start_offset() * 4);
        encoder.set_buffer(1, Some(s2.buffer()), l2.start_offset() * 4);
        encoder.set_buffer(2, Some(dst.as_ref()), 0);
        encoder.set_bytes(3, &(self.n as u32));
        encoder.use_resource(s1.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        let w = mtl::group_width(&p, self.n);
        encoder.dispatch_threads(
            MTLSize {
                width: self.n,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: w,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), self.n, DType::F32),
            l1.shape().clone(),
        ))
    }
}

/// `silu(gate) * up` in one pass — the tail of a SwiGLU feed-forward.
pub fn swiglu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.device().is_metal() && gate.shape() == up.shape() {
        let op = SwigluMul {
            n: gate.elem_count(),
        };
        return gate.contiguous()?.apply_op2_no_bwd(&up.contiguous()?, &op);
    }
    candle_nn::ops::silu(gate)? * up
}

/// `[b, n, h, d] -> [b, h, n, d]` as one coalesced pass.
struct HeadTranspose {
    n: usize,
    heads: usize,
    dim: usize,
}

impl CustomOp1 for HeadTranspose {
    fn name(&self) -> &'static str {
        "head_transpose"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let src = match storage {
            CpuStorage::F32(s) => s,
            _ => candle_core::bail!("head_transpose: only f32"),
        };
        if !layout.is_contiguous() {
            candle_core::bail!("head_transpose: input must be contiguous");
        }
        let (b, n, hd, dim) = (layout.shape().dims()[0], self.n, self.heads, self.dim);
        let o = layout.start_offset();
        let mut dst = vec![0f32; b * hd * n * dim];
        for bi in 0..b {
            for h in 0..hd {
                for p in 0..n {
                    let s = o + ((bi * n + p) * hd + h) * dim;
                    let t = ((bi * hd + h) * n + p) * dim;
                    dst[t..t + dim].copy_from_slice(&src[s..s + dim]);
                }
            }
        }
        Ok((CpuStorage::F32(dst), (b, hd, n, dim).into()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle_core::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        if !layout.is_contiguous() {
            candle_core::bail!("head_transpose: input must be contiguous");
        }
        if storage.dtype() != DType::F32 {
            candle_core::bail!("head_transpose: only f32");
        }
        let b = layout.shape().dims()[0];
        let n = self.n * b * self.heads * self.dim;
        let device = storage.device();
        let p = mtl::pipeline(device, "head_transpose_f32")?;
        let dst = device.new_buffer(n, DType::F32, "head_transpose")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::head_transpose");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(storage.buffer()), layout.start_offset() * 4);
        encoder.set_buffer(1, Some(dst.as_ref()), 0);
        encoder.set_bytes(2, &(self.n as u32));
        encoder.set_bytes(3, &(self.heads as u32));
        encoder.set_bytes(4, &(self.dim as u32));
        encoder.use_resource(storage.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        // `dim` is 64 here, so pair it with several positions to fill a threadgroup.
        let w = mtl::group_width(&p, self.dim);
        encoder.dispatch_threads(
            MTLSize {
                width: self.dim,
                height: self.n,
                depth: b * self.heads,
            },
            MTLSize {
                width: w,
                height: (256 / w).max(1),
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), n, DType::F32),
            (b, self.heads, self.n, self.dim).into(),
        ))
    }
}

/// `[b, n, h*d] -> [b, h, n, d]`, the reshape-and-transpose multi-head attention needs.
///
/// **Measured 4.0x-7.4x faster than `transpose(1,2).contiguous()`** (63 GB/s against 8.5 at
/// `[2, 3192, 1024]`) and bit-identical — but it does *not* speed up attention, and
/// `flow.rs` does not use it. `sdpa` accepts strides, so the DiT feeds it lazy transposed
/// views and never materialises them; making them contiguous first, even this cheaply,
/// measured **0.98x** at the engine's real sequence length. The strided-view decision in
/// `DiTBlock::attention` was checked against a fast transpose and survives.
///
/// Kept because the negative result is worth being able to re-run (`tts-probe --bin
/// attnlayout`), and because any future path that needs a genuinely contiguous head layout
/// should not pay candle's 8.5 GB/s for it.
pub fn head_transpose(x: &Tensor, heads: usize, dim: usize) -> Result<Tensor> {
    let (b, n, hd) = x.dims3()?;
    if hd != heads * dim {
        candle_core::bail!("head_transpose: {hd} != {heads} * {dim}");
    }
    if x.device().is_metal() {
        let op = HeadTranspose { n, heads, dim };
        return x.contiguous()?.apply_op1_no_bwd(&op);
    }
    x.reshape((b, n, heads, dim))?.transpose(1, 2)?.contiguous()
}

/// `x + sin^2(x)`, one pass on Metal.
pub fn snake_folded(x: &Tensor) -> Result<Tensor> {
    if x.device().is_metal() {
        x.contiguous()?.apply_op1_no_bwd(&SnakeFolded)
    } else {
        x + x.sin()?.sqr()?
    }
}

/// `u + sin^2(u)` with `u = alpha * x` broadcast over `[1, C, L]`'s channel axis.
///
/// `alpha` may be `[C]`, `[1, C, 1]`, or any shape with `C` elements.
pub fn snake_alpha(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let (b, c, len) = x.dims3()?;
    if b == 1 && x.device().is_metal() && alpha.elem_count() == c {
        let op = SnakeAlpha { channels: c, len };
        return x
            .contiguous()?
            .apply_op2_no_bwd(&alpha.flatten_all()?.contiguous()?, &op);
    }
    let u = x.broadcast_mul(&alpha.reshape((1, c, 1))?)?.contiguous()?;
    &u + u.sin()?.sqr()?
}

/// [`snake_beta`] for channels-last `[b, L, C]`.
pub fn snake_beta_nlc(x: &Tensor, alpha: &Tensor, beta_recip: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let c = dims[dims.len() - 1];
    let rows: usize = dims[..dims.len() - 1].iter().product();
    // Not gated on Metal, unlike the channel-major sibling: `cpu_fwd` handles f16 activations
    // against f32 parameters and the composed fallback cannot, since candle has no mixed-dtype
    // multiply. The gate runs its numerics on CPU, so the fallback is a real path, not a
    // formality.
    if alpha.elem_count() == c && beta_recip.elem_count() == c {
        let op = SnakeBetaNlc {
            rows,
            chan: c,
            half: x.dtype() == candle_core::DType::F16,
        };
        return x
            .contiguous()?
            .apply_op3_no_bwd(
                &alpha.flatten_all()?.contiguous()?,
                &beta_recip.flatten_all()?.contiguous()?,
                &op,
            )?
            .reshape(dims);
    }
    let u = x.broadcast_mul(alpha)?.contiguous()?;
    x.broadcast_add(&u.sin()?.sqr()?.broadcast_mul(beta_recip)?)
}

struct SnakeBetaNlc {
    rows: usize,
    chan: usize,
    half: bool,
}

impl candle_core::CustomOp3 for SnakeBetaNlc {
    fn name(&self) -> &'static str {
        "snake_beta_nlc"
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
        let xf: Vec<f32>;
        let (x, a, br) = match (s1, s2, s3) {
            (CpuStorage::F32(x), CpuStorage::F32(a), CpuStorage::F32(br)) => (&x[..], a, br),
            (CpuStorage::F16(x), CpuStorage::F32(a), CpuStorage::F32(br)) => {
                xf = x.iter().map(|v| v.to_f32()).collect();
                (&xf[..], a, br)
            }
            _ => candle_core::bail!("snake_beta_nlc: activations f32 or f16, params f32"),
        };
        let (o1, o2, o3) = (l1.start_offset(), l2.start_offset(), l3.start_offset());
        let n = self.rows * self.chan;
        let mut dst = vec![0f32; n];
        for r in 0..self.rows {
            for c in 0..self.chan {
                let i = r * self.chan + c;
                let v = x[o1 + i];
                dst[i] = v + br[o3 + c] * (a[o2 + c] * v).sin().powi(2);
            }
        }
        let shape: candle_core::Shape = (self.rows, self.chan).into();
        if self.half {
            let h = dst.into_iter().map(half::f16::from_f32).collect();
            Ok((CpuStorage::F16(h), shape))
        } else {
            Ok((CpuStorage::F32(dst), shape))
        }
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
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        if s2.dtype() != DType::F32 || s3.dtype() != DType::F32 {
            candle_core::bail!("snake_beta_nlc: parameters must be f32");
        }
        let (kernel, dt, esz) = match s1.dtype() {
            DType::F32 => ("snake_beta_nlc_f32", DType::F32, 4),
            DType::F16 => ("snake_beta_nlc_f16", DType::F16, 2),
            d => candle_core::bail!("snake_beta_nlc: activations f32 or f16, got {d:?}"),
        };
        let n = self.rows * self.chan;
        let device = s1.device();
        let p = mtl::pipeline(device, kernel)?;
        let dst = device.new_buffer(n, dt, "snake_beta_nlc")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::snake_beta_nlc");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(s1.buffer()), l1.start_offset() * esz);
        encoder.set_buffer(1, Some(s2.buffer()), l2.start_offset() * 4);
        encoder.set_buffer(2, Some(s3.buffer()), l3.start_offset() * 4);
        for s in [s1, s2, s3] {
            encoder.use_resource(s.buffer(), MTLResourceUsage::Read);
        }
        encoder.set_buffer(3, Some(dst.as_ref()), 0);
        encoder.set_bytes(4, &(self.chan as u32));
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        encoder.dispatch_threads(
            MTLSize {
                width: self.chan,
                height: self.rows,
                depth: 1,
            },
            MTLSize {
                width: mtl::group_width(&p, self.chan),
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), n, dt),
            (self.rows, self.chan).into(),
        ))
    }
}

/// `x + beta_recip * sin^2(alpha * x)`, both parameters per-channel over `[1, C, L]`.
///
/// The unfoldable snake — see [`crate::snake_full`], which is the composed fallback and
/// what the tests compare against.
pub fn snake_beta(x: &Tensor, alpha: &Tensor, beta_recip: &Tensor) -> Result<Tensor> {
    let (b, c, len) = x.dims3()?;
    if b == 1 && x.device().is_metal() && alpha.elem_count() == c && beta_recip.elem_count() == c {
        let op = SnakeBeta { channels: c, len };
        return x.contiguous()?.apply_op3_no_bwd(
            &alpha.flatten_all()?.contiguous()?,
            &beta_recip.flatten_all()?.contiguous()?,
            &op,
        );
    }
    let u = x.broadcast_mul(&alpha.reshape((1, c, 1))?)?.contiguous()?;
    x.broadcast_add(
        &u.sin()?
            .sqr()?
            .broadcast_mul(&beta_recip.reshape((1, c, 1))?)?,
    )
}

// ------------------------------------------------ DiT block elementwise

/// Shared shape for the two `[b, n, d]`-against-`[b, 1, d]` kernels.
struct Bcast3 {
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    kernel: &'static str,
    label: &'static str,
    n: usize,
    dim: usize,
    /// `true` for `x * (1 + v1) + v2`, `false` for `x + y * v`.
    affine: bool,
}

impl candle_core::CustomOp3 for Bcast3 {
    fn name(&self) -> &'static str {
        "bcast3"
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
        let (a, b, c) = match (s1, s2, s3) {
            (CpuStorage::F32(a), CpuStorage::F32(b), CpuStorage::F32(c)) => (a, b, c),
            _ => candle_core::bail!("{}: only f32", self.label),
        };
        for l in [l1, l2, l3] {
            if !l.is_contiguous() {
                candle_core::bail!("{}: inputs must be contiguous", self.label);
            }
        }
        let batch = l1.shape().dims()[0];
        let (o1, o2, o3) = (l1.start_offset(), l2.start_offset(), l3.start_offset());
        let mut dst = vec![0f32; batch * self.n * self.dim];
        for bi in 0..batch {
            for p in 0..self.n {
                for d in 0..self.dim {
                    let i = (bi * self.n + p) * self.dim + d;
                    let j = bi * self.dim + d;
                    dst[i] = if self.affine {
                        a[o1 + i] * (1.0 + b[o2 + j]) + c[o3 + j]
                    } else {
                        a[o1 + i] + b[o2 + i] * c[o3 + j]
                    };
                }
            }
        }
        Ok((CpuStorage::F32(dst), (batch, self.n, self.dim).into()))
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
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        for l in [l1, l2, l3] {
            if !l.is_contiguous() {
                candle_core::bail!("{}: inputs must be contiguous", self.label);
            }
        }
        for s in [s1, s2, s3] {
            if s.dtype() != DType::F32 {
                candle_core::bail!("{}: only f32", self.label);
            }
        }
        let batch = l1.shape().dims()[0];
        let count = batch * self.n * self.dim;
        let device = s1.device();
        let p = mtl::pipeline(device, self.kernel)?;
        let dst = device.new_buffer(count, DType::F32, self.label)?;

        let encoder = device.command_encoder()?;
        encoder.set_label(self.label);
        encoder.set_compute_pipeline_state(&p);
        for (i, (s, l)) in [(s1, l1), (s2, l2), (s3, l3)].iter().enumerate() {
            encoder.set_buffer(i, Some(s.buffer()), l.start_offset() * 4);
            encoder.use_resource(s.buffer(), MTLResourceUsage::Read);
        }
        encoder.set_buffer(3, Some(dst.as_ref()), 0);
        encoder.set_bytes(4, &(self.n as u32));
        encoder.set_bytes(5, &(self.dim as u32));
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        let w = mtl::group_width(&p, self.dim);
        encoder.dispatch_threads(
            MTLSize {
                width: self.dim,
                height: self.n,
                depth: batch,
            },
            MTLSize {
                width: w,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), count, DType::F32),
            (batch, self.n, self.dim).into(),
        ))
    }
}

/// `x * (1 + scale) + shift` with `scale`/`shift` broadcast over the sequence axis.
pub fn modulate_affine(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let (_, n, dim) = x.dims3()?;
    if x.device().is_metal() {
        let op = Bcast3 {
            kernel: "modulate_affine_f32",
            label: "tts_nn::modulate_affine",
            n,
            dim,
            affine: true,
        };
        return x
            .contiguous()?
            .apply_op3_no_bwd(&scale.contiguous()?, &shift.contiguous()?, &op);
    }
    x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

/// `residual + y * gate`, with `gate` broadcast over the sequence axis.
pub fn gate_residual(residual: &Tensor, y: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let (_, n, dim) = residual.dims3()?;
    if residual.device().is_metal() {
        let op = Bcast3 {
            kernel: "gate_residual_f32",
            label: "tts_nn::gate_residual",
            n,
            dim,
            affine: false,
        };
        return residual
            .contiguous()?
            .apply_op3_no_bwd(&y.contiguous()?, &gate.contiguous()?, &op);
    }
    residual + y.broadcast_mul(gate)?
}

// ------------------------------------------------ AdaIN halves

/// `(x - m)^2` with `m` per-channel over `[1, C, L]`.
///
/// The centred square the variance needs, without a broadcast: `m` is indexed
/// by the grid's y axis. Bit-exact against the composed form.
struct SubSqr {
    channels: usize,
    len: usize,
}

impl candle_core::CustomOp2 for SubSqr {
    fn name(&self) -> &'static str {
        "sub_sqr"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (x, m) = match (s1, s2) {
            (CpuStorage::F32(x), CpuStorage::F32(m)) => (x, m),
            _ => candle_core::bail!("sub_sqr: only f32"),
        };
        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("sub_sqr: inputs must be contiguous");
        }
        let (o1, o2) = (l1.start_offset(), l2.start_offset());
        let mut dst = vec![0f32; self.channels * self.len];
        for c in 0..self.channels {
            let m = m[o2 + c];
            for l in 0..self.len {
                let d = x[o1 + c * self.len + l] - m;
                dst[c * self.len + l] = d * d;
            }
        }
        Ok((CpuStorage::F32(dst), (1, self.channels, self.len).into()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        for l in [l1, l2] {
            if !l.is_contiguous() {
                candle_core::bail!("sub_sqr: inputs must be contiguous");
            }
        }
        for s in [s1, s2] {
            if s.dtype() != DType::F32 {
                candle_core::bail!("sub_sqr: only f32");
            }
        }
        let n = self.channels * self.len;
        let device = s1.device();
        let p = mtl::pipeline(device, "sub_sqr_f32")?;
        let dst = device.new_buffer(n, DType::F32, "sub_sqr")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::sub_sqr");
        encoder.set_compute_pipeline_state(&p);
        for (i, (s, l)) in [(s1, l1), (s2, l2)].iter().enumerate() {
            encoder.set_buffer(i, Some(s.buffer()), l.start_offset() * 4);
            encoder.use_resource(s.buffer(), MTLResourceUsage::Read);
        }
        encoder.set_buffer(2, Some(dst.as_ref()), 0);
        encoder.set_bytes(3, &(self.len as u32));
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        let w = mtl::group_width(&p, self.len);
        encoder.dispatch_threads(
            MTLSize {
                width: self.len,
                height: self.channels,
                depth: 1,
            },
            MTLSize {
                width: w,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), n, DType::F32),
            (1, self.channels, self.len).into(),
        ))
    }
}

/// `(x - m)^2` with `m` holding one value per channel.
///
/// `m` may be `[C]` or `[1, C, 1]`; the fallback is the two composed ops.
pub fn sub_sqr(x: &Tensor, m: &Tensor) -> Result<Tensor> {
    let (b, c, len) = x.dims3()?;
    if b == 1 && x.device().is_metal() && m.elem_count() == c {
        let op = SubSqr { channels: c, len };
        return x
            .contiguous()?
            .apply_op2_no_bwd(&m.flatten_all()?.contiguous()?, &op);
    }
    Ok(x.broadcast_sub(&m.reshape((1, c, 1))?)?.sqr()?)
}

/// `out = (x - mean) * rsqrt(var + eps) * (gamma + 1) + beta`, per channel.
///
/// The tail of an AdaIN in one pass: normalise, scale and shift with direct
/// per-channel indexing instead of four broadcasts and a division. `rsqrt`
/// against the composed `div(sqrt)` differs ~1 ulp; the tests bound it.
struct AdainApply {
    channels: usize,
    len: usize,
    eps: f32,
}

impl candle_core::CustomOp2 for AdainApply {
    fn name(&self) -> &'static str {
        "adain_apply"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (x, p) = match (s1, s2) {
            (CpuStorage::F32(x), CpuStorage::F32(p)) => (x, p),
            _ => candle_core::bail!("adain_apply: only f32"),
        };
        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("adain_apply: inputs must be contiguous");
        }
        let (o1, o2) = (l1.start_offset(), l2.start_offset());
        let c = self.channels;
        let mut dst = vec![0f32; c * self.len];
        for ch in 0..c {
            let (mean, var, gamma, beta) = (
                p[o2 + ch],
                p[o2 + c + ch],
                p[o2 + 2 * c + ch],
                p[o2 + 3 * c + ch],
            );
            let scale = 1.0 / (var + self.eps).sqrt() * (gamma + 1.0);
            for l in 0..self.len {
                dst[ch * self.len + l] = (x[o1 + ch * self.len + l] - mean) * scale + beta;
            }
        }
        Ok((CpuStorage::F32(dst), (1, c, self.len).into()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        for l in [l1, l2] {
            if !l.is_contiguous() {
                candle_core::bail!("adain_apply: inputs must be contiguous");
            }
        }
        for s in [s1, s2] {
            if s.dtype() != DType::F32 {
                candle_core::bail!("adain_apply: only f32");
            }
        }
        let n = self.channels * self.len;
        let device = s1.device();
        let p = mtl::pipeline(device, "adain_apply_f32")?;
        let dst = device.new_buffer(n, DType::F32, "adain_apply")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::adain_apply");
        encoder.set_compute_pipeline_state(&p);
        for (i, (s, l)) in [(s1, l1), (s2, l2)].iter().enumerate() {
            encoder.set_buffer(i, Some(s.buffer()), l.start_offset() * 4);
            encoder.use_resource(s.buffer(), MTLResourceUsage::Read);
        }
        encoder.set_buffer(2, Some(dst.as_ref()), 0);
        encoder.set_bytes(3, &(self.len as u32));
        encoder.set_bytes(4, &(self.channels as u32));
        encoder.set_bytes(5, &self.eps);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        let w = mtl::group_width(&p, self.len);
        encoder.dispatch_threads(
            MTLSize {
                width: self.len,
                height: self.channels,
                depth: 1,
            },
            MTLSize {
                width: w,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), n, DType::F32),
            (1, self.channels, self.len).into(),
        ))
    }
}

/// Normalise `x` per channel and apply a style scale and shift, one pass.
///
/// All four parameters hold one value per channel, each `[C]` or `[1, C, 1]`.
/// The composed fallback is exactly the operations this replaces, in order.
pub fn adain_apply(
    x: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let (b, c, len) = x.dims3()?;
    let flat = |t: &Tensor| -> Result<Tensor> {
        if t.elem_count() != c {
            candle_core::bail!("adain_apply: parameter has wrong size");
        }
        Ok(t.flatten_all()?.contiguous()?)
    };
    if b == 1
        && x.device().is_metal()
        && [mean, var, gamma, beta].iter().all(|t| t.elem_count() == c)
    {
        let op = AdainApply {
            channels: c,
            len,
            eps: eps as f32,
        };
        let packed = Tensor::stack(&[flat(mean)?, flat(var)?, flat(gamma)?, flat(beta)?], 0)?;
        return x.contiguous()?.apply_op2_no_bwd(&packed, &op);
    }
    let m = |t: &Tensor| t.reshape((1, c, 1));
    let centred = x.broadcast_sub(&m(mean)?)?;
    let normed = centred.broadcast_div(&(m(var)? + eps)?.sqrt()?)?;
    normed
        .broadcast_mul(&(m(gamma)? + 1.0)?)?
        .broadcast_add(&m(beta)?)
}

/// Per-channel mean and variance of a `[1, C, L]` signal, as `[2, C]`.
struct Moments {
    channels: usize,
    len: usize,
}

impl CustomOp1 for Moments {
    fn name(&self) -> &'static str {
        "channel_moments"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let x = match s {
            CpuStorage::F32(x) => x,
            _ => candle_core::bail!("channel_moments: only f32"),
        };
        if !l.is_contiguous() {
            candle_core::bail!("channel_moments: input must be contiguous");
        }
        let o = l.start_offset();
        let mut dst = vec![0f32; 2 * self.channels];
        for c in 0..self.channels {
            let row = &x[o + c * self.len..o + (c + 1) * self.len];
            let (mut sum, mut sq) = (0f32, 0f32);
            for v in row {
                sum += v;
                sq += v * v;
            }
            let mean = sum / self.len as f32;
            dst[c] = mean;
            dst[self.channels + c] = (sq / self.len as f32 - mean * mean).max(0.0);
        }
        Ok((CpuStorage::F32(dst), (2, self.channels).into()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s: &candle_core::MetalStorage,
        l: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        if !l.is_contiguous() {
            candle_core::bail!("channel_moments: input must be contiguous");
        }
        if s.dtype() != DType::F32 {
            candle_core::bail!("channel_moments: only f32");
        }
        let device = s.device();
        let p = mtl::pipeline(device, "channel_moments_f32")?;
        let dst = device.new_buffer(2 * self.channels, DType::F32, "channel_moments")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::channel_moments");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(s.buffer()), l.start_offset() * 4);
        encoder.set_buffer(1, Some(dst.as_ref()), 0);
        encoder.set_bytes(2, &(self.len as u32));
        encoder.set_bytes(3, &(self.channels as u32));
        encoder.use_resource(s.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        // One threadgroup per channel: grid width equals the group width, so the
        // dispatch stays uniform and `threadgroup_position_in_grid.y` is the channel.
        let w = mtl::group_width(&p, self.len);
        encoder.dispatch_threads(
            MTLSize { width: w, height: self.channels, depth: 1 },
            MTLSize { width: w, height: 1, depth: 1 },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), 2 * self.channels, DType::F32),
            (2, self.channels).into(),
        ))
    }
}

/// Mean and variance per channel in one read, returned as `[2, C]`.
///
/// Replaces `mean_keepdim` + [`sub_sqr`] + `mean_keepdim`, which is three passes over the
/// signal and one full-size intermediate.
pub fn moments(x: &Tensor) -> Result<Tensor> {
    let (b, c, len) = x.dims3()?;
    if b != 1 {
        candle_core::bail!("channel_moments: batch must be 1, got {b}");
    }
    x.contiguous()?.apply_op1_no_bwd(&Moments { channels: c, len })
}

/// [`adain_apply`] with SnakeBeta folded into its epilogue.
struct AdainSnake {
    channels: usize,
    len: usize,
    eps: f32,
}

impl candle_core::CustomOp2 for AdainSnake {
    fn name(&self) -> &'static str {
        "adain_snake"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (x, p) = match (s1, s2) {
            (CpuStorage::F32(x), CpuStorage::F32(p)) => (x, p),
            _ => candle_core::bail!("adain_snake: only f32"),
        };
        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("adain_snake: inputs must be contiguous");
        }
        let (o1, o2, c) = (l1.start_offset(), l2.start_offset(), self.channels);
        let mut dst = vec![0f32; c * self.len];
        for ch in 0..c {
            let at = |row: usize| p[o2 + row * c + ch];
            let (mean, var, gamma, beta, alpha, brecip) =
                (at(0), at(1), at(2), at(3), at(4), at(5));
            let scale = 1.0 / (var + self.eps).sqrt() * (gamma + 1.0);
            for l in 0..self.len {
                let y = (x[o1 + ch * self.len + l] - mean) * scale + beta;
                let sn = (alpha * y).sin();
                dst[ch * self.len + l] = y + brecip * sn * sn;
            }
        }
        Ok((CpuStorage::F32(dst), (1, c, self.len).into()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        for l in [l1, l2] {
            if !l.is_contiguous() {
                candle_core::bail!("adain_snake: inputs must be contiguous");
            }
        }
        for s in [s1, s2] {
            if s.dtype() != DType::F32 {
                candle_core::bail!("adain_snake: only f32");
            }
        }
        let n = self.channels * self.len;
        let device = s1.device();
        let p = mtl::pipeline(device, "adain_snake_f32")?;
        let dst = device.new_buffer(n, DType::F32, "adain_snake")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::adain_snake");
        encoder.set_compute_pipeline_state(&p);
        for (i, (s, l)) in [(s1, l1), (s2, l2)].iter().enumerate() {
            encoder.set_buffer(i, Some(s.buffer()), l.start_offset() * 4);
            encoder.use_resource(s.buffer(), MTLResourceUsage::Read);
        }
        encoder.set_buffer(2, Some(dst.as_ref()), 0);
        encoder.set_bytes(3, &(self.len as u32));
        encoder.set_bytes(4, &(self.channels as u32));
        encoder.set_bytes(5, &self.eps);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        let w = mtl::group_width(&p, self.len);
        encoder.dispatch_threads(
            MTLSize { width: self.len, height: self.channels, depth: 1 },
            MTLSize { width: w, height: 1, depth: 1 },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), n, DType::F32),
            (1, self.channels, self.len).into(),
        ))
    }
}

/// [`adain_apply`] followed by [`snake_beta`], in one pass.
///
/// The generator's residual blocks never do one without the other, and separately they
/// read and write the whole signal twice.
#[allow(clippy::too_many_arguments)]
pub fn adain_snake(
    x: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    alpha: &Tensor,
    beta_recip: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let (b, c, len) = x.dims3()?;
    let parts = [mean, var, gamma, beta, alpha, beta_recip];
    let flat = |t: &Tensor| -> Result<Tensor> {
        if t.elem_count() != c {
            candle_core::bail!("adain_snake: parameter has wrong size");
        }
        t.flatten_all()?.contiguous()
    };
    if b == 1 && x.device().is_metal() && parts.iter().all(|t| t.elem_count() == c) {
        let packed = Tensor::stack(
            &parts.iter().map(|t| flat(t)).collect::<Result<Vec<_>>>()?,
            0,
        )?;
        let op = AdainSnake { channels: c, len, eps: eps as f32 };
        return x
            .contiguous()?
            .apply_op2_no_bwd(&packed, &op);
    }
    let y = adain_apply(x, mean, var, gamma, beta, eps)?;
    snake_beta(&y, &alpha.reshape((1, c, 1))?, &beta_recip.reshape((1, c, 1))?)
}

/// An LSTM step's gates, cell update and output, in one pass.
struct LstmGates {
    hidden: usize,
}

impl candle_core::CustomOp3 for LstmGates {
    fn name(&self) -> &'static str {
        "lstm_gates"
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
        let (g, p, c) = match (s1, s2, s3) {
            (CpuStorage::F32(g), CpuStorage::F32(p), CpuStorage::F32(c)) => (g, p, c),
            _ => candle_core::bail!("lstm_gates: only f32"),
        };
        let (o1, o2, o3, h) = (
            l1.start_offset(),
            l2.start_offset(),
            l3.start_offset(),
            self.hidden,
        );
        let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
        let mut dst = vec![0f32; 2 * h];
        for j in 0..h {
            let gate = |n: usize| g[o1 + n * h + j] + p[o2 + n * h + j];
            let ct = sig(gate(1)) * c[o3 + j] + sig(gate(0)) * gate(2).tanh();
            dst[h + j] = ct;
            dst[j] = sig(gate(3)) * ct.tanh();
        }
        Ok((CpuStorage::F32(dst), (2, h).into()))
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
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        for l in [l1, l2, l3] {
            if !l.is_contiguous() {
                candle_core::bail!("lstm_gates: inputs must be contiguous");
            }
        }
        for s in [s1, s2, s3] {
            if s.dtype() != DType::F32 {
                candle_core::bail!("lstm_gates: only f32");
            }
        }
        let device = s1.device();
        let p = mtl::pipeline(device, "lstm_gates_f32")?;
        let dst = device.new_buffer(2 * self.hidden, DType::F32, "lstm_gates")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::lstm_gates");
        encoder.set_compute_pipeline_state(&p);
        for (i, (s, l)) in [(s1, l1), (s2, l2), (s3, l3)].iter().enumerate() {
            encoder.set_buffer(i, Some(s.buffer()), l.start_offset() * 4);
            encoder.use_resource(s.buffer(), MTLResourceUsage::Read);
        }
        encoder.set_buffer(3, Some(dst.as_ref()), 0);
        encoder.set_bytes(4, &(self.hidden as u32));
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        let w = mtl::group_width(&p, self.hidden);
        encoder.dispatch_threads(
            MTLSize { width: self.hidden, height: 1, depth: 1 },
            MTLSize { width: w, height: 1, depth: 1 },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), 2 * self.hidden, DType::F32),
            (2, self.hidden).into(),
        ))
    }
}

/// One LSTM timestep after its two matmuls: `gates` is `h @ w_hh`, `pre` the input
/// projection's row with both biases folded in, `c` the cell state. Returns `[2, hidden]`
/// — the new h on row 0, the new c on row 1 — so the next step's matmul reads a row of it
/// with no copy.
pub fn lstm_gates(gates: &Tensor, pre: &Tensor, c: &Tensor) -> Result<Tensor> {
    let hidden = c.elem_count();
    if gates.elem_count() != 4 * hidden || pre.elem_count() != 4 * hidden {
        candle_core::bail!("lstm_gates: gate vectors must be 4 x hidden");
    }
    gates
        .contiguous()?
        .apply_op3_no_bwd(&pre.contiguous()?, &c.contiguous()?, &LstmGates { hidden })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// The fused step against the composed one, on every device. A `[2, hidden]` result
    /// whose rows are read back as the next step's inputs is easy to get subtly wrong.
    #[test]
    fn lstm_gates_matches_composed() -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        let _gpu = crate::gpu_guard();
        #[cfg_attr(not(feature = "metal"), allow(unused_mut))]
        let mut devices = vec![Device::Cpu];
        if let Some(m) = crate::usable_metal() {
            devices.push(m);
        }
        for d in devices {
            for h in [3usize, 64, 256] {
                let g = Tensor::randn(0f32, 1., (1, 4 * h), &d)?;
                let p = Tensor::randn(0f32, 1., (1, 4 * h), &d)?;
                let c = Tensor::randn(0f32, 1., (1, h), &d)?;
                let z = (&g + &p)?;
                let it = candle_nn::ops::sigmoid(&z.narrow(1, 0, h)?)?;
                let ft = candle_nn::ops::sigmoid(&z.narrow(1, h, h)?)?;
                let gt = z.narrow(1, 2 * h, h)?.tanh()?;
                let ot = candle_nn::ops::sigmoid(&z.narrow(1, 3 * h, h)?)?;
                let cw = ((ft * &c)? + (it * gt)?)?;
                let hw = (ot * cw.tanh()?)?;
                let got = lstm_gates(&g, &p, &c)?;
                for (i, want) in [hw, cw].iter().enumerate() {
                    let (abs, rel) =
                        crate::abs_and_rel(&got.narrow(0, i, 1)?.reshape((1, h))?, want)?;
                    assert!(rel < 1e-6, "row {i} {d:?} h={h}: abs {abs:.3e} rel {rel:.3e}");
                }
                // Offset inputs: every step after the first feeds narrowed rows in.
                let stacked = Tensor::cat(&[&c, &c.affine(2.0, 1.0)?], 0)?;
                let got = lstm_gates(&g, &p, &stacked.narrow(0, 1, 1)?)?;
                let want = lstm_gates(&g, &p, &stacked.narrow(0, 1, 1)?.contiguous()?)?;
                let (abs, rel) = crate::abs_and_rel(&got, &want)?;
                assert!(rel < 1e-6, "offset {d:?} h={h}: abs {abs:.3e} rel {rel:.3e}");
            }
        }
        Ok(())
    }

    /// The step kernel against a double-precision host reference, across the whole input
    /// range, to a couple of ulp.
    ///
    /// This is the one that matters, and it is deliberately not a comparison against
    /// candle's composed form: that agrees to 1e-7 whatever the kernel does, because the
    /// two are equally wrong. Metal compiles with fast math by default, and a recurrence
    /// multiplies its own rounding — with the fast `exp` and `tanh` this kernel matched
    /// composed to 1e-7 for one step, had diverged to 4e-1 after sixty, moved Kokoro's
    /// predicted durations and more than halved the length of the audio. Every fixture row
    /// still passed, because the fixtures are 52 timesteps and saturate the gates.
    #[test]
    fn lstm_gates_is_accurate_to_double_precision() -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        let _gpu = crate::gpu_guard();
        let Some(d) = crate::usable_metal() else {
            return Ok(());
        };
        // A sweep, not noise: the gates have to be checked where they are neither
        // saturated nor centred, which is where a fast transcendental costs the most.
        let h = 256usize;
        let span = |lo: f64, hi: f64| -> Vec<f32> {
            (0..h).map(|j| (lo + (hi - lo) * j as f64 / (h - 1) as f64) as f32).collect()
        };
        let mut g = Vec::new();
        for (lo, hi) in [(-12.0, 12.0), (-6.0, 6.0), (-3.0, 3.0), (-1.0, 1.0)] {
            g.extend(span(lo, hi));
        }
        let p = vec![0f32; 4 * h];
        let c: Vec<f32> = span(-2.0, 2.0);
        let gt = Tensor::from_vec(g.clone(), (1, 4 * h), &d)?;
        let pt = Tensor::from_vec(p, (1, 4 * h), &d)?;
        let ct = Tensor::from_vec(c.clone(), (1, h), &d)?;
        let got = lstm_gates(&gt, &pt, &ct)?.flatten_all()?.to_vec1::<f32>()?;

        let sig = |v: f64| 1.0 / (1.0 + (-v).exp());
        let (mut worst, mut at) = (0f64, 0usize);
        for j in 0..h {
            let gate = |n: usize| g[n * h + j] as f64;
            let cw = sig(gate(1)) * c[j] as f64 + sig(gate(0)) * gate(2).tanh();
            let hw = sig(gate(3)) * cw.tanh();
            for (want, have) in [(hw, got[j] as f64), (cw, got[h + j] as f64)] {
                let rel = (have - want).abs() / want.abs().max(1e-3);
                if rel > worst {
                    worst = rel;
                    at = j;
                }
            }
        }
        assert!(worst < 1e-6, "worst relative error {worst:.3e} at channel {at}");
        Ok(())
    }

    /// Every other kernel on the three older engines' paths that calls a transcendental,
    /// against a double-precision host reference.
    ///
    /// Same audit as the SnakeBeta one below, for the same reason: `lstm_gates` proved that
    /// Metal's fast math can be wrong enough to matter, and nothing else here had been
    /// checked against anything but candle — which would be wrong identically. `swiglu_mul`
    /// is the one to care about: it is the SiLU in qwen3tts's talker, and the talker is
    /// autoregressive, so it has the property that turned the LSTM's ulp into 4e-1.
    ///
    /// Ranges are generous on purpose. Vocoder activations reach tens, and `sin`'s range
    /// reduction and `exp`'s overflow are exactly what a narrow sweep would miss.
    #[test]
    fn transcendental_kernels_are_accurate_to_double_precision() -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        let _gpu = crate::gpu_guard();
        let Some(d) = crate::usable_metal() else {
            return Ok(());
        };
        let (c, len) = (8usize, 4096usize);
        let sweep = |lo: f64, hi: f64, n: usize| -> Vec<f32> {
            (0..n).map(|i| (lo + (hi - lo) * (i % len) as f64 / (len - 1) as f64) as f32).collect()
        };
        let xs = sweep(-30.0, 30.0, c * len);
        let al: Vec<f32> = (0..c).map(|j| 0.05 + 2.5 * j as f32 / (c - 1) as f32).collect();
        let br: Vec<f32> = (0..c).map(|j| 0.5 + j as f32).collect();
        let x = Tensor::from_vec(xs.clone(), (1, c, len), &d)?;
        let a3 = Tensor::from_vec(al.clone(), (1, c, 1), &d)?;
        let b3 = Tensor::from_vec(br.clone(), (1, c, 1), &d)?;

        // Scaled by the operands: every one of these can cancel near zero, and dividing by
        // the result there reports the cancellation as kernel error.
        let mut worst: Vec<(&str, f64)> = Vec::new();
        let mut check = |name: &'static str, got: Vec<f32>, want: &dyn Fn(usize, usize) -> (f64, f64)| {
            let mut w = 0f64;
            for j in 0..c {
                for i in 0..len {
                    let (v, scale) = want(j, i);
                    w = w.max((got[j * len + i] as f64 - v).abs() / scale.max(1.0));
                }
            }
            worst.push((name, w));
        };

        check("snake_folded", crate::fused::snake_folded(&x)?.flatten_all()?.to_vec1()?, &|j, i| {
            let xv = xs[j * len + i] as f64;
            (xv + xv.sin().powi(2), xv.abs())
        });
        check("snake (alpha)", crate::snake(&x, &a3)?.flatten_all()?.to_vec1()?, &|j, i| {
            let u = al[j] as f64 * xs[j * len + i] as f64;
            (u + u.sin().powi(2), u.abs())
        });
        check("snake_full", crate::snake_full(&x, &a3, &b3)?.flatten_all()?.to_vec1()?, &|j, i| {
            let xv = xs[j * len + i] as f64;
            let u = al[j] as f64 * xv;
            (xv + br[j] as f64 * u.sin().powi(2), xv.abs().max(br[j] as f64))
        });

        // The channels-last sibling, f32 and f16, as qwen3tts's codec calls it.
        let xn = Tensor::from_vec(xs.clone(), (1, len, c), &d)?;
        let a1 = Tensor::from_vec(al.clone(), c, &d)?;
        let b1 = Tensor::from_vec(br.clone(), c, &d)?;
        let nlc = snake_beta_nlc(&xn, &a1, &b1)?.flatten_all()?.to_vec1::<f32>()?;
        let mut w = 0f64;
        for i in 0..len {
            for j in 0..c {
                let xv = xs[i * c + j] as f64;
                let v = xv + br[j] as f64 * (al[j] as f64 * xv).sin().powi(2);
                let scale = xv.abs().max(br[j] as f64).max(1.0);
                w = w.max((nlc[i * c + j] as f64 - v).abs() / scale);
            }
        }
        worst.push(("snake_beta_nlc", w));

        // SiLU * up, the talker's FFN.
        let gs = sweep(-30.0, 30.0, c * len);
        let us = sweep(-4.0, 4.0, c * len);
        let g = Tensor::from_vec(gs.clone(), (1, c * len), &d)?;
        let u = Tensor::from_vec(us.clone(), (1, c * len), &d)?;
        let sw = swiglu_mul(&g, &u)?.flatten_all()?.to_vec1::<f32>()?;
        let mut w = 0f64;
        for i in 0..c * len {
            let (gv, uv) = (gs[i] as f64, us[i] as f64);
            let v = gv / (1.0 + (-gv).exp()) * uv;
            w = w.max((sw[i] as f64 - v).abs() / (gv * uv).abs().max(1.0));
        }
        worst.push(("swiglu_mul", w));

        // 2e-6, not one: at the top of the sweep `alpha * x` is ~75, and f32 cannot hold
        // that argument to better than a few microradians, which lands in `sin` before any
        // kernel runs. The two that reach 1.03e-6 do so identically under `precise::sin`,
        // and one of them — `snake_full` — is candle's composed path rather than a kernel
        // here, which is the clincher.
        let mut bad = Vec::new();
        for (name, e) in &worst {
            eprintln!("    {name:<16} {e:.3e}");
            if *e >= 2e-6 {
                bad.push(format!("{name} {e:.3e}"));
            }
        }
        assert!(bad.is_empty(), "off double precision: {}", bad.join(", "));
        Ok(())
    }

    /// SnakeBeta against a double-precision host reference, over the range the generator
    /// actually drives it through.
    ///
    /// Checked because `lstm_gates` was caught by Metal's fast math and this kernel calls
    /// `sin`: the checkpoint's alpha reaches 2.33, so the argument runs past +-20, far
    /// enough for a weak range reduction to show. It does not — fast and `precise::sin`
    /// agree to 8.4e-7 here, and swapping every `sin`, `rsqrt`, `sqrt` and `sincos` on
    /// Kokoro's path for its `precise::` form moved no fixture number at all. So this
    /// stands as a correctness check on the kernel, not as a guard against fast math, and
    /// the composed-form test beside it cannot serve that purpose: it compares against
    /// candle ops that would be wrong in exactly the same way.
    ///
    /// Scale the error by the operands, never by the result. `x + b*sin^2` cancels near
    /// zero, and dividing by the result there reports the cancellation as kernel error —
    /// 1.65e-4 of it, which is what sent this audit down a false trail to begin with.
    #[test]
    fn snake_beta_is_accurate_to_double_precision() -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        let _gpu = crate::gpu_guard();
        let Some(d) = crate::usable_metal() else {
            return Ok(());
        };
        let (c, len) = (8usize, 2048usize);
        // x sweeps +-12, alpha spans the checkpoint's own range, so the product reaches ~28.
        let xs: Vec<f32> = (0..c * len)
            .map(|i| (-12.0 + 24.0 * (i % len) as f64 / (len - 1) as f64) as f32)
            .collect();
        let al: Vec<f32> = (0..c).map(|j| 0.03 + 2.3 * j as f32 / (c - 1) as f32).collect();
        let br: Vec<f32> = (0..c).map(|j| 0.5 + j as f32).collect();
        let x = Tensor::from_vec(xs.clone(), (1, c, len), &d)?;
        let a = Tensor::from_vec(al.clone(), (1, c, 1), &d)?;
        let b = Tensor::from_vec(br.clone(), (1, c, 1), &d)?;
        let got = snake_beta(&x, &a, &b)?.flatten_all()?.to_vec1::<f32>()?;

        let mut worst = 0f64;
        for j in 0..c {
            for i in 0..len {
                let xv = xs[j * len + i] as f64;
                let sn = (al[j] as f64 * xv).sin();
                let want = xv + br[j] as f64 * sn * sn;
                // Scaled by the operands, not by the result: `x + b*sin^2` cancels near
                // zero, and dividing by the result there measures the cancellation rather
                // than the kernel.
                let scale = xv.abs().max(br[j] as f64).max(1.0);
                let rel = (got[j * len + i] as f64 - want).abs() / scale;
                worst = worst.max(rel);
            }
        }
        assert!(worst < 1e-6, "worst relative error {worst:.3e}");
        Ok(())
    }

    /// Sum-of-squares against candle's two-pass variance, at the generator's own shapes.
    /// The bound is looser than the elementwise kernels': a raw second moment loses
    /// digits the composed form keeps, and this records how many.
    #[test]
    fn moments_match_composed() -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        let _gpu = crate::gpu_guard();
        #[cfg_attr(not(feature = "metal"), allow(unused_mut))]
        let mut devices = vec![Device::Cpu];
        if let Some(m) = crate::usable_metal() {
            devices.push(m);
        }
        for d in devices {
            for (c, len) in [(7usize, 33usize), (256, 8040), (128, 48240)] {
                let x = Tensor::randn(0f32, 2., (1, c, len), &d)?;
                let mean = x.mean_keepdim(2)?;
                let var = crate::fused::sub_sqr(&x, &mean)?.mean_keepdim(2)?;
                let got = moments(&x)?;
                for (i, want) in [mean, var].iter().enumerate() {
                    let (abs, rel) = crate::abs_and_rel(
                        &got.narrow(0, i, 1)?.reshape((1, c, 1))?,
                        want,
                    )?;
                    assert!(
                        rel < 1e-4,
                        "moments[{i}] {d:?} at {c}x{len}: abs {abs:.3e} rel {rel:.3e}"
                    );
                }
            }
        }
        Ok(())
    }

    /// Against the composed form the kernels replace. Fused arithmetic is not required to
    /// be bit-identical — it skips intermediate rounding through memory — so this checks a
    /// tight relative bound rather than equality.
    #[test]
    fn snake_forms_agree() -> anyhow::Result<()> {
        let d = Device::Cpu;
        let x = Tensor::randn(0f32, 2., (1, 7, 33), &d)?;
        let alpha = Tensor::randn(0f32, 1., 7, &d)?;

        let want = (&x + x.sin()?.sqr()?)?;
        let got = snake_folded(&x)?;
        assert!(crate::max_abs_diff(&want, &got)? < 1e-5);

        let want = crate::snake(&x, &alpha.reshape((1, 7, 1))?)?;
        let got = snake_alpha(&x, &alpha)?;
        assert_eq!(want.dims(), got.dims());
        assert!(crate::max_abs_diff(&want, &got)? < 1e-5);
        Ok(())
    }

    /// `snake_beta` against `snake_full`, **on every device that is actually available**.
    ///
    /// The CPU arm alone would not be worth much: it exercises `cpu_fwd`, which is the
    /// fallback, not the kernel that runs in production. Reverting device sampling cost a
    /// day to a unit test that passed on a shape the real model never uses — the shapes
    /// here are the codec's own (1536 channels, a chunk's worth of samples).
    #[test]
    fn snake_beta_matches_composed() -> anyhow::Result<()> {
        #[cfg_attr(not(feature = "metal"), allow(unused_mut))]
        let mut devices = vec![Device::Cpu];
        #[cfg(feature = "metal")]
        #[cfg(feature = "metal")]
        let _gpu = crate::gpu_guard();
        if let Some(m) = crate::usable_metal() {
            devices.push(m);
        }
        for d in devices {
            for (c, len) in [(7, 33), (1536, 601), (24, 4801)] {
                let x = Tensor::randn(0f32, 2., (1, c, len), &d)?;
                let alpha = Tensor::randn(0f32, 1., c, &d)?.exp()?;
                let beta_recip = Tensor::randn(0f32, 1., c, &d)?.exp()?;

                let want = crate::snake_full(
                    &x,
                    &alpha.reshape((1, c, 1))?,
                    &beta_recip.reshape((1, c, 1))?,
                )?;
                let got = snake_beta(&x, &alpha, &beta_recip)?;
                assert_eq!(want.dims(), got.dims(), "{d:?} at {c}x{len}");
                let (abs, rel) = crate::abs_and_rel(&want, &got)?;
                assert!(
                    rel < 1e-5,
                    "{d:?} at {c}x{len}: abs {abs:.3e} rel {rel:.3e}"
                );
            }
        }
        Ok(())
    }

    /// The AdaIN halves against their composed forms, at the generator's shapes.
    /// `adain_apply` uses `rsqrt` where the composed form divides by a root, so
    /// the bound is 1e-6 rather than bit equality; `sub_sqr` is exact.
    #[test]
    fn adain_forms_agree() -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        let _gpu = crate::gpu_guard();
        #[cfg_attr(not(feature = "metal"), allow(unused_mut))]
        let mut devices = vec![Device::Cpu];
        if let Some(m) = crate::usable_metal() {
            devices.push(m);
        }
        for d in devices {
            for (c, len) in [(7, 33), (256, 444), (128, 4440)] {
                let x = Tensor::randn(0f32, 2., (1, c, len), &d)?;
                let m = Tensor::randn(0f32, 1., c, &d)?;
                let want = x.broadcast_sub(&m.reshape((1, c, 1))?)?.sqr()?;
                let got = sub_sqr(&x, &m)?;
                let (abs, rel) = crate::abs_and_rel(&want, &got)?;
                assert!(
                    rel < 1e-6,
                    "sub_sqr {d:?} at {c}x{len}: abs {abs:.3e} rel {rel:.3e}"
                );

                let (mean, var, gamma, beta) = (
                    Tensor::randn(0f32, 1., c, &d)?,
                    Tensor::randn(0f32, 1., c, &d)?.sqr()?,
                    Tensor::randn(0f32, 0.3, c, &d)?,
                    Tensor::randn(0f32, 1., c, &d)?,
                );
                let r = |t: &Tensor| t.reshape((1, c, 1));
                let want = x
                    .broadcast_sub(&r(&mean)?)?
                    .broadcast_div(&(r(&var)? + 1e-5)?.sqrt()?)?
                    .broadcast_mul(&(r(&gamma)? + 1.0)?)?
                    .broadcast_add(&r(&beta)?)?;
                let got = adain_apply(&x, &mean, &var, &gamma, &beta, 1e-5)?;
                let (abs, rel) = crate::abs_and_rel(&want, &got)?;
                assert!(
                    rel < 1e-6,
                    "adain_apply {d:?} at {c}x{len}: abs {abs:.3e} rel {rel:.3e}"
                );

                let (alpha, brecip) = (
                    Tensor::randn(0f32, 1., c, &d)?.abs()?,
                    Tensor::randn(0f32, 1., c, &d)?.abs()?,
                );
                let want = snake_beta(&want, &r(&alpha)?, &r(&brecip)?)?;
                let got =
                    adain_snake(&x, &mean, &var, &gamma, &beta, &alpha, &brecip, 1e-5)?;
                let (abs, rel) = crate::abs_and_rel(&want, &got)?;
                assert!(
                    rel < 1e-5,
                    "adain_snake {d:?} at {c}x{len}: abs {abs:.3e} rel {rel:.3e}"
                );
            }
        }
        Ok(())
    }
}
