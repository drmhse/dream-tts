//! Kokoro's padded SnakeResBlock as one MPSGraph: every AdaIN, snake, conv and residual of a
//! block in one executable, where the composed route is a moments pass, an AdaIN pass and an
//! MPSGraph conv — with its commit-and-wait — per conv.
//!
//! Per pair `i`: `t = conv1(snake(adain(x)))`, `x = conv2(snake(adain(t))) + x`. Moments are
//! over the first `valid` samples and snake outputs are zero past them, exactly as
//! `adain_snake_masked` does, so a bucket-padded signal stays exact.

use anyhow::{Context, Result};
use candle_core::backend::BackendStorage;
use candle_core::{
    CpuStorage, CustomOp1, DType, Layout, MetalDevice, Result as CResult, Shape, Tensor,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_foundation::{NSArray, NSDictionary, NSNumber};
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLDevice};
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphConvolution2DOpDescriptor, MPSGraphDevice, MPSGraphExecutable,
    MPSGraphPaddingStyle, MPSGraphShapedType, MPSGraphTensor, MPSGraphTensorData,
    MPSGraphTensorNamedDataLayout,
};
use std::sync::{Arc, Mutex, OnceLock};

const DYNAMIC: isize = -1;

fn shape(dims: &[isize]) -> Retained<NSArray<NSNumber>> {
    let v: Vec<Retained<NSNumber>> = dims.iter().map(|d| NSNumber::new_isize(*d)).collect();
    NSArray::from_retained_slice(&v)
}

/// One conv of a block: kernel size and dilation. Padding is centred.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ConvShape {
    pub k: usize,
    pub dilation: usize,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct BlockKey {
    device: candle_core::metal_backend::DeviceId,
    pub channels: usize,
    /// `(conv1, conv2)` per pair.
    pub pairs: Vec<(ConvShape, ConvShape)>,
    pub eps_bits: u32,
}

struct Graph {
    graph: Retained<MPSGraph>,
    /// Placeholders in feed order: x, params, mask, inv_n, then weight and bias per conv.
    inputs: Vec<Retained<MPSGraphTensor>>,
    out: Retained<MPSGraphTensor>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    device: Retained<MPSGraphDevice>,
    compiled: Mutex<Option<Arc<Exec>>>,
}

/// Graphs kept across every block and length. A graph holds its runs' intermediates, ~90 MB at
/// 128 channels and 48k samples, until it is dropped; one kokoro segment needs six, and three
/// recompiled every segment (RTF 0.053).
const KEEP: usize = 6;

struct Exec {
    exe: Retained<MPSGraphExecutable>,
    /// For each executable input, its index into `Graph::inputs`.
    order: Vec<usize>,
}

unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}
unsafe impl Send for Exec {}
unsafe impl Sync for Exec {}

impl Graph {
    fn build(dev: &MetalDevice, key: &BlockKey) -> Result<Self> {
        let c = key.channels as isize;
        let p = key.pairs.len() as isize;
        let eps = f32::from_bits(key.eps_bits) as f64;
        unsafe {
            let g = MPSGraph::new();
            let ph = |dims: &[isize]| {
                g.placeholderWithShape_dataType_name(Some(&shape(dims)), MPSDataType::Float32, None)
            };
            let x_in = ph(&[1, c, 1, DYNAMIC]);
            // [2P adains, (gamma, beta, alpha, 1/beta), C]
            let params = ph(&[2 * p, 4, c]);
            let mask = ph(&[1, 1, 1, DYNAMIC]);
            let inv_n = ph(&[1, 1, 1, 1]);
            let mut inputs = vec![x_in.clone(), params.clone(), mask.clone(), inv_n.clone()];

            let mul = |a: &MPSGraphTensor, b: &MPSGraphTensor| {
                g.multiplicationWithPrimaryTensor_secondaryTensor_name(a, b, None)
            };
            let add = |a: &MPSGraphTensor, b: &MPSGraphTensor| {
                g.additionWithPrimaryTensor_secondaryTensor_name(a, b, None)
            };
            let sub = |a: &MPSGraphTensor, b: &MPSGraphTensor| {
                g.subtractionWithPrimaryTensor_secondaryTensor_name(a, b, None)
            };
            let param = |k: isize, j: isize| {
                let t = g.sliceTensor_dimension_start_length_name(&params, 0, k, 1, None);
                let t = g.sliceTensor_dimension_start_length_name(&t, 1, j, 1, None);
                g.reshapeTensor_withShape_name(&t, &shape(&[1, c, 1, 1]), None)
            };
            let one = g.constantWithScalar_shape_dataType(1.0, &shape(&[1]), MPSDataType::Float32);
            let eps_t =
                g.constantWithScalar_shape_dataType(eps, &shape(&[1]), MPSDataType::Float32);
            let adain_snake = |x: &MPSGraphTensor, k: isize| {
                let mean = mul(
                    &g.reductionSumWithTensor_axis_name(&mul(x, &mask), 3, None),
                    &inv_n,
                );
                let d = mul(&sub(x, &mean), &mask);
                let var = mul(
                    &g.reductionSumWithTensor_axis_name(&mul(&d, &d), 3, None),
                    &inv_n,
                );
                let scale = mul(
                    &g.reciprocalSquareRootWithTensor_name(&add(&var, &eps_t), None),
                    &add(&param(k, 0), &one),
                );
                let y = add(&mul(&d, &scale), &param(k, 1));
                let s = g.sinWithTensor_name(&mul(&y, &param(k, 2)), None);
                mul(&add(&y, &mul(&param(k, 3), &mul(&s, &s))), &mask)
            };
            let mut conv = |x: &MPSGraphTensor, s: ConvShape| -> Result<Retained<MPSGraphTensor>> {
                let w = ph(&[1, s.k as isize, c, c]);
                let b = ph(&[1, c, 1, 1]);
                inputs.push(w.clone());
                inputs.push(b.clone());
                let pad = (s.k - 1) * s.dilation / 2;
                let desc = MPSGraphConvolution2DOpDescriptor::
                    descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout(
                        1, 1, s.dilation, 1, 1, pad, (s.k - 1) * s.dilation - pad, 0, 0,
                        MPSGraphPaddingStyle::Explicit,
                        MPSGraphTensorNamedDataLayout::NCHW,
                        MPSGraphTensorNamedDataLayout::HWIO,
                    )
                    .context("MPSGraph convolution descriptor")?;
                let y = g.convolution2DWithSourceTensor_weightsTensor_descriptor_name(
                    x, &w, &desc, None,
                );
                Ok(add(&y, &b))
            };
            let mut x = x_in.clone();
            for (i, (c1, c2)) in key.pairs.iter().enumerate() {
                let i = i as isize;
                let t = conv(&adain_snake(&x, 2 * i), *c1)?;
                let y = conv(&adain_snake(&t, 2 * i + 1), *c2)?;
                x = add(&y, &x);
            }
            let queue = dev
                .device()
                .as_ref()
                .newCommandQueue()
                .context("MPSGraph command queue")?;
            Ok(Self {
                graph: g,
                inputs,
                out: x,
                queue,
                device: MPSGraphDevice::deviceWithMTLDevice(dev.device().as_ref()),
                compiled: Mutex::new(None),
            })
        }
    }

    fn exec(&self, key: &BlockKey, len: usize) -> Result<Arc<Exec>> {
        let mut compiled = self
            .compiled
            .lock()
            .map_err(|e| anyhow::anyhow!("executable cache poisoned: {e}"))?;
        if let Some(e) = compiled.as_ref() {
            return Ok(e.clone());
        }
        let c = key.channels as isize;
        let l = len as isize;
        let typed = |dims: &[isize]| unsafe {
            MPSGraphShapedType::initWithShape_dataType(
                MPSGraphShapedType::alloc(),
                Some(&shape(dims)),
                MPSDataType::Float32,
            )
        };
        let mut types = vec![
            typed(&[1, c, 1, l]),
            typed(&[2 * key.pairs.len() as isize, 4, c]),
            typed(&[1, 1, 1, l]),
            typed(&[1, 1, 1, 1]),
        ];
        for (c1, c2) in &key.pairs {
            for s in [c1, c2] {
                types.push(typed(&[1, s.k as isize, c, c]));
                types.push(typed(&[1, c, 1, 1]));
            }
        }
        let tensors: Vec<&MPSGraphTensor> = self.inputs.iter().map(|t| &**t).collect();
        let feeds = NSDictionary::from_retained_objects(&tensors, &types);
        let exe = unsafe {
            self.graph
                .compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
                    Some(&self.device),
                    &feeds,
                    &NSArray::from_slice(&[&*self.out]),
                    None,
                    None,
                )
        };
        let fed = unsafe { exe.feedTensors() }.context("executable lists no inputs")?;
        let order = fed
            .iter()
            .map(|t| {
                self.inputs
                    .iter()
                    .position(|o| std::ptr::eq(&*t, &**o))
                    .context("executable input is none of the graph's placeholders")
            })
            .collect::<Result<Vec<_>>>()?;
        let e = Arc::new(Exec { exe, order });
        *compiled = Some(e.clone());
        Ok(e)
    }
}

/// The graph for `(key, len)`, most recently used last.
fn cached(dev: &MetalDevice, key: &BlockKey, len: usize) -> Result<Arc<Graph>> {
    type Lru = Vec<((BlockKey, usize), Arc<Graph>)>;
    static CACHE: OnceLock<Mutex<Lru>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = cache
        .lock()
        .map_err(|e| anyhow::anyhow!("graph cache poisoned: {e}"))?;
    if let Some(i) = guard.iter().position(|(k, _)| k.0 == *key && k.1 == len) {
        let hit = guard.remove(i);
        let g = hit.1.clone();
        guard.push(hit);
        return Ok(g);
    }
    let g = Arc::new(Graph::build(dev, key)?);
    while guard.len() >= KEEP {
        guard.remove(0);
    }
    guard.push(((key.clone(), len), g.clone()));
    Ok(g)
}

/// The block's key, for [`prewarm`] and [`apply`].
pub fn key(
    device: &candle_core::Device,
    channels: usize,
    pairs: Vec<(ConvShape, ConvShape)>,
    eps: f32,
) -> Option<BlockKey> {
    let candle_core::Device::Metal(dev) = device else {
        return None;
    };
    Some(BlockKey {
        device: dev.id(),
        channels,
        pairs,
        eps_bits: eps.to_bits(),
    })
}

/// Compile the executables for `(key, len)` on background threads.
pub fn prewarm(device: &candle_core::Device, jobs: Vec<(BlockKey, usize)>) {
    let candle_core::Device::Metal(dev) = device else {
        return;
    };
    for (key, len) in jobs {
        let dev = dev.clone();
        std::thread::spawn(move || {
            if let Ok(g) = cached(&dev, &key, len) {
                let _ = g.exec(&key, len);
            }
        });
    }
}

/// `x` `[1, C, len]`; `params` `[2P, 4, C]` as (gamma, beta, alpha, 1/beta) per AdaIN; `convs`
/// the `2P` `(kio [k, C, C], bias [C])` in order.
pub fn apply(
    key: &BlockKey,
    x: &Tensor,
    params: &Tensor,
    convs: &[(&Tensor, &Tensor)],
    valid: usize,
) -> Result<Tensor> {
    let len = x.dim(2)?;
    anyhow::ensure!(valid > 0 && valid <= len, "{valid} valid of {len}");
    anyhow::ensure!(convs.len() == 2 * key.pairs.len(), "two convs per pair");
    let mut m = vec![0f32; len];
    m[..valid].fill(1.0);
    let mask = Tensor::from_vec(m, (1, 1, 1, len), x.device())?;
    let inv_n = Tensor::from_vec(vec![1.0f32 / valid as f32], (1, 1, 1, 1), x.device())?;
    let mut extra = vec![params.contiguous()?, mask, inv_n];
    for (w, b) in convs {
        extra.push(w.contiguous()?);
        extra.push(b.contiguous()?);
    }
    let op = Block {
        key: key.clone(),
        extra,
    };
    Ok(x.contiguous()?.apply_op1_no_bwd(&op)?)
}

struct Block {
    key: BlockKey,
    /// Everything but `x`, in placeholder order.
    extra: Vec<Tensor>,
}

impl CustomOp1 for Block {
    fn name(&self) -> &'static str {
        "mps_snake_block"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> CResult<(CpuStorage, Shape)> {
        candle_core::bail!("mps_snake_block: Metal only")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
    ) -> CResult<(candle_core::MetalStorage, Shape)> {
        use candle_core::MetalStorage;
        if !l1.is_contiguous() || l1.start_offset() != 0 {
            candle_core::bail!("mps_snake_block: input must be contiguous and unoffset");
        }
        let (_, c, len) = l1.shape().dims3()?;
        let dev = s1.device().clone();
        let g = cached(&dev, &self.key, len).map_err(candle_core::Error::wrap)?;
        let exe = g.exec(&self.key, len).map_err(candle_core::Error::wrap)?;
        let mut buffers = Vec::with_capacity(self.extra.len());
        for t in &self.extra {
            let (s, l) = t.storage_and_layout();
            if l.start_offset() != 0 || !l.is_contiguous() {
                candle_core::bail!("mps_snake_block: operands must be contiguous and unoffset");
            }
            match &*s {
                candle_core::Storage::Metal(m) => {
                    buffers.push((m.buffer().clone(), t.dims().to_vec()))
                }
                _ => candle_core::bail!("mps_snake_block: operands must be on the device"),
            }
        }
        dev.wait_until_completed()?;
        let out = dev.new_buffer(c * len, DType::F32, "mps_snake_block")?;
        let pairs = self.key.pairs.clone();
        unsafe {
            let td = |buf: &ProtocolObject<dyn MTLBuffer>, dims: &[isize]| {
                MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                    MPSGraphTensorData::alloc(),
                    buf,
                    &shape(dims),
                    MPSDataType::Float32,
                )
            };
            let (ci, li) = (c as isize, len as isize);
            let dims_of = |i: usize| -> Vec<isize> {
                match i {
                    0 => vec![1, ci, 1, li],
                    1 => vec![2 * pairs.len() as isize, 4, ci],
                    2 => vec![1, 1, 1, li],
                    3 => vec![1, 1, 1, 1],
                    _ if (i - 4).is_multiple_of(2) => {
                        let conv = (i - 4) / 2;
                        let (c1, c2) = pairs[conv / 2];
                        let k = if conv.is_multiple_of(2) { c1.k } else { c2.k };
                        vec![1, k as isize, ci, ci]
                    }
                    _ => vec![1, ci, 1, 1],
                }
            };
            let mut inputs = Vec::with_capacity(exe.order.len());
            for &i in &exe.order {
                let buf = if i == 0 {
                    s1.buffer().clone()
                } else {
                    buffers[i - 1].0.clone()
                };
                inputs.push(td(buf.as_ref(), &dims_of(i)));
            }
            let results = [td(
                AsRef::<ProtocolObject<dyn MTLBuffer>>::as_ref(out.as_ref()),
                &[1, ci, 1, li],
            )];
            // Drained here: the run's intermediates are autoreleased, and a synthesis thread
            // has no pool of its own, so they were kept for the life of the process.
            objc2::rc::autoreleasepool(|_| {
                exe.exe
                    .runWithMTLCommandQueue_inputsArray_resultsArray_executionDescriptor(
                        &g.queue,
                        &NSArray::from_retained_slice(&inputs),
                        Some(&NSArray::from_retained_slice(&results)),
                        None,
                    );
            });
        }
        Ok((
            MetalStorage::new(out, dev, c * len, DType::F32),
            (1, c, len).into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Against the composed route: masked moments, `adain_snake_masked`, and the gather conv.
    #[test]
    fn matches_composed_route() -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        let _gpu = crate::gpu_guard();
        let Some(d) = crate::usable_metal() else {
            return Ok(());
        };
        let (c, len, valid, eps) = (128usize, 8192usize, 7000usize, 1e-5f32);
        for k in [3usize, 7] {
            let dils = [1usize, 3, 5];
            let pairs: Vec<(ConvShape, ConvShape)> = dils
                .iter()
                .map(|&dl| (ConvShape { k, dilation: dl }, ConvShape { k, dilation: 1 }))
                .collect();
            let x = Tensor::randn(0f32, 1., (1, c, len), &d)?;
            let gb = Tensor::randn(0f32, 0.3, (6, 2, c), &d)?;
            let ab = Tensor::rand(0.5f32, 1.5, (6, 2, c), &d)?;
            let params = Tensor::cat(&[&gb, &ab], 1)?;
            let mut convs = Vec::new();
            for _ in 0..6 {
                convs.push((
                    Tensor::randn(0f32, 0.02, (k, c, c), &d)?,
                    Tensor::randn(0f32, 0.1, c, &d)?,
                ));
            }
            let mut want = x.clone();
            for i in 0..3 {
                let conv = |x: &Tensor, j: usize, dl: usize| -> anyhow::Result<Tensor> {
                    let (w, b) = &convs[j];
                    Ok(crate::centered_conv1d_gemm(
                        x,
                        &w.reshape((k * c, c))?.t()?,
                        Some(b),
                        k,
                        dl,
                        (k - 1) * dl / 2,
                    )?)
                };
                let a = crate::fused::adain_snake_masked(
                    &want,
                    &gb.get(2 * i)?,
                    &ab.get(2 * i)?,
                    eps as f64,
                    valid,
                )?;
                let t = conv(&a, 2 * i, dils[i])?;
                let a = crate::fused::adain_snake_masked(
                    &t,
                    &gb.get(2 * i + 1)?,
                    &ab.get(2 * i + 1)?,
                    eps as f64,
                    valid,
                )?;
                want = (conv(&a, 2 * i + 1, 1)? + &want)?;
            }
            let key = key(&d, c, pairs, eps).unwrap();
            let refs: Vec<(&Tensor, &Tensor)> = convs.iter().map(|(w, b)| (w, b)).collect();
            let got = apply(&key, &x, &params, &refs, valid)?;
            let (abs, rel) = crate::abs_and_rel(
                &want.narrow(2, 0, valid)?.contiguous()?,
                &got.narrow(2, 0, valid)?.contiguous()?,
            )?;
            assert!(rel < 1e-4, "k={k}: abs {abs:.3e} rel {rel:.3e}");
        }
        Ok(())
    }
}
