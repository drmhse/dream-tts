//! Kokoro's generator convolutions through MPSGraph, which does not build an im2col matrix.
//!
//! `tapconv` records the failed attempt to remove that gather by hand: two kernels, both
//! several times slower than MPS's GEMM. The conclusion there — that Apple's kernels are
//! better than the ones this codebase can write — points at MPSGraph, which candle does
//! not use and which has a tuned `convolution2D`. It wins on every shape in the generator,
//! by 1.7x on the one that dominates:
//!
//! | 128ch @ 48240 | gather + candle matmul | MPSGraph |
//! |---|---|---|
//! | k=3 | 2.90 ms | 2.24 |
//! | k=7 | 6.23 ms | 4.23 |
//! | k=11 | 10.31 ms | 6.04 |
//!
//! A 1-D convolution is a 2-D one with a height of 1. The weight is `[k, cin, cout]`, MPSGraph's
//! `HWIO` with the height axis inserted, because transposed it is also the gather GEMM's
//! `[cout, k * cin]`: a caller that needs both routes holds one copy.
//!
//! Two things make this shippable rather than a benchmark. The length axis is declared
//! dynamic, which costs nothing measurable and means one compiled graph serves every
//! utterance instead of one per length. And the bias is a node in the graph, so it does
//! not become a second pass over the output.
//!
//! The cost is a synchronisation: MPSGraph runs on its own queue, so candle's pending work
//! has to be committed and waited on first or the graph reads a buffer whose producing
//! kernel has not been dispatched. The generator is a serial chain, so there is little
//! overlap to lose, and `kokorogen` measures what it costs.

use anyhow::{Context, Result};
use candle_core::backend::BackendStorage;
use candle_core::{
    CpuStorage, CustomOp2, DType, Layout, MetalDevice, Result as CResult, Shape, Tensor,
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
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// `-1` on the length axis: one graph, every utterance.
const DYNAMIC: isize = -1;

fn shape(dims: &[isize]) -> Retained<NSArray<NSNumber>> {
    let v: Vec<Retained<NSNumber>> = dims.iter().map(|d| NSNumber::new_isize(*d)).collect();
    NSArray::from_retained_slice(&v)
}

/// One compiled graph, keyed by everything about the convolution except its length.
struct Graph {
    graph: Retained<MPSGraph>,
    src: Retained<MPSGraphTensor>,
    wts: Retained<MPSGraphTensor>,
    bias: Option<Retained<MPSGraphTensor>>,
    residual: Option<Retained<MPSGraphTensor>>,
    out: Retained<MPSGraphTensor>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    device: Retained<MPSGraphDevice>,
    /// One executable per length. The lock is held while one compiles, so a run that wants
    /// the length [`prewarm`] is compiling waits for it instead of compiling it again.
    compiled: Mutex<HashMap<usize, Arc<Exec>>>,
}

/// The graph compiled for one length, and which of our tensors each of its inputs is.
struct Exec {
    exe: Retained<MPSGraphExecutable>,
    order: Vec<Feed>,
}

#[derive(Clone, Copy, PartialEq)]
enum Feed {
    Src,
    Wts,
    Bias,
    Residual,
}

// The Metal objects are used behind the cache's mutex and never handed across a suspend
// point; objc2 does not know that, so the marker is asserted here rather than inferred.
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}
unsafe impl Send for Exec {}
unsafe impl Sync for Exec {}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct Key {
    device: candle_core::metal_backend::DeviceId,
    cin: usize,
    cout: usize,
    k: usize,
    dilation: usize,
    pad_left: usize,
    bias: bool,
    residual: bool,
}

impl Graph {
    fn build(dev: &MetalDevice, key: &Key) -> Result<Self> {
        let pad_right = (key.k - 1) * key.dilation - key.pad_left;
        unsafe {
            let graph = MPSGraph::new();
            let src = graph.placeholderWithShape_dataType_name(
                Some(&shape(&[1, key.cin as isize, 1, DYNAMIC])),
                MPSDataType::Float32,
                None,
            );
            let wts = graph.placeholderWithShape_dataType_name(
                Some(&shape(&[
                    1,
                    key.k as isize,
                    key.cin as isize,
                    key.cout as isize,
                ])),
                MPSDataType::Float32,
                None,
            );
            let desc = MPSGraphConvolution2DOpDescriptor::
                descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout(
                    1, 1, key.dilation, 1, 1, key.pad_left, pad_right, 0, 0,
                    MPSGraphPaddingStyle::Explicit,
                    MPSGraphTensorNamedDataLayout::NCHW,
                    MPSGraphTensorNamedDataLayout::HWIO,
                )
                .context("MPSGraph convolution descriptor")?;
            let mut out = graph.convolution2DWithSourceTensor_weightsTensor_descriptor_name(
                &src, &wts, &desc, None,
            );
            let bias = key.bias.then(|| {
                let b = graph.placeholderWithShape_dataType_name(
                    Some(&shape(&[1, key.cout as isize, 1, 1])),
                    MPSDataType::Float32,
                    None,
                );
                out = graph.additionWithPrimaryTensor_secondaryTensor_name(&out, &b, None);
                b
            });
            let residual = key.residual.then(|| {
                let r = graph.placeholderWithShape_dataType_name(
                    Some(&shape(&[1, key.cout as isize, 1, DYNAMIC])),
                    MPSDataType::Float32,
                    None,
                );
                out = graph.additionWithPrimaryTensor_secondaryTensor_name(&out, &r, None);
                r
            });
            let queue = dev
                .device()
                .as_ref()
                .newCommandQueue()
                .context("MPSGraph command queue")?;
            Ok(Self {
                graph,
                src,
                wts,
                bias,
                residual,
                out,
                queue,
                device: MPSGraphDevice::deviceWithMTLDevice(dev.device().as_ref()),
                compiled: Mutex::new(HashMap::new()),
            })
        }
    }

    /// The executable for `len`, compiled on first use.
    ///
    /// A graph run with a length it has not met specialises itself for it, ~4 ms a graph on
    /// the caller's thread with the GPU idle. An executable compiled ahead of time on another
    /// thread, by [`prewarm`], costs the run nothing.
    fn exec(&self, key: &Key, len: usize) -> Result<Arc<Exec>> {
        let mut compiled = self
            .compiled
            .lock()
            .map_err(|e| anyhow::anyhow!("executable cache poisoned: {e}"))?;
        if let Some(e) = compiled.get(&len) {
            return Ok(e.clone());
        }
        let typed = |dims: &[isize]| unsafe {
            MPSGraphShapedType::initWithShape_dataType(
                MPSGraphShapedType::alloc(),
                Some(&shape(dims)),
                MPSDataType::Float32,
            )
        };
        let (cin, cout, k, len_i) = (
            key.cin as isize,
            key.cout as isize,
            key.k as isize,
            len as isize,
        );
        let mut tensors: Vec<&MPSGraphTensor> = vec![&self.src, &self.wts];
        let mut types = vec![typed(&[1, cin, 1, len_i]), typed(&[1, k, cin, cout])];
        if let Some(b) = &self.bias {
            tensors.push(b);
            types.push(typed(&[1, cout, 1, 1]));
        }
        if let Some(r) = &self.residual {
            tensors.push(r);
            types.push(typed(&[1, cout, 1, len_i]));
        }
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
                let is = |o: &MPSGraphTensor| std::ptr::eq(&*t, o);
                if is(&self.src) {
                    Ok(Feed::Src)
                } else if is(&self.wts) {
                    Ok(Feed::Wts)
                } else if self.bias.as_deref().is_some_and(is) {
                    Ok(Feed::Bias)
                } else if self.residual.as_deref().is_some_and(is) {
                    Ok(Feed::Residual)
                } else {
                    anyhow::bail!("executable input is none of the graph's placeholders")
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let e = Arc::new(Exec { exe, order });
        compiled.insert(len, e.clone());
        Ok(e)
    }
}

/// A convolution [`centered_conv1d`] or [`centered_conv1d_residual`] will be asked for.
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    pub cin: usize,
    pub cout: usize,
    pub k: usize,
    pub dilation: usize,
    pub pad_left: usize,
    pub bias: bool,
    pub residual: bool,
    pub len: usize,
}

/// Compile the executables `specs` will need on background threads, and return at once.
///
/// Called when the lengths are known and the GPU has work queued ahead of the first of them,
/// which hides the compiles: they are CPU work, and the synthesis thread would otherwise do
/// them one at a time with the GPU waiting.
pub fn prewarm(device: &candle_core::Device, specs: Vec<Spec>) {
    let candle_core::Device::Metal(dev) = device else {
        return;
    };
    const THREADS: usize = 4;
    let mut lanes: Vec<Vec<Spec>> = vec![Vec::new(); THREADS];
    for (i, s) in specs.into_iter().enumerate() {
        lanes[i % THREADS].push(s);
    }
    for lane in lanes.into_iter().filter(|l| !l.is_empty()) {
        let dev = dev.clone();
        std::thread::spawn(move || {
            for s in lane {
                let key = Key {
                    device: dev.id(),
                    cin: s.cin,
                    cout: s.cout,
                    k: s.k,
                    dilation: s.dilation,
                    pad_left: s.pad_left,
                    bias: s.bias,
                    residual: s.residual,
                };
                // A failure here is the run's to report, when it compiles the same thing.
                if let Ok(g) = cached(&dev, key) {
                    let _ = g.exec(&key, s.len);
                }
            }
        });
    }
}

fn cached(dev: &MetalDevice, key: Key) -> Result<Arc<Graph>> {
    static CACHE: OnceLock<Mutex<HashMap<Key, Arc<Graph>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|e| anyhow::anyhow!("graph cache poisoned: {e}"))?;
    if let Some(g) = guard.get(&key) {
        return Ok(g.clone());
    }
    let g = Arc::new(Graph::build(dev, &key)?);
    guard.insert(key, g.clone());
    Ok(g)
}

/// Length below which the gather route wins anyway.
///
/// MPSGraph runs on its own queue, so each call costs a commit and a wait. That is fixed
/// where the saving scales with the convolution, and below a few thousand frames the fixed
/// part is the larger of the two — the prosody predictor's blocks, at 804 and 1608 frames,
/// measured 6 ms *slower* through MPSGraph. The generator's two stages are 8040 and 48240.
fn min_len() -> usize {
    static M: OnceLock<usize> = OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("TTS_MPS_MIN_LEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096)
    })
}

/// Whether a centred conv at `len` would take this route on `device`.
pub fn eligible_len(device: &candle_core::Device, len: usize) -> bool {
    len >= min_len() && device.is_metal()
}

/// Whether this convolution can go through MPSGraph at all.
pub fn eligible(x: &Tensor, w: &Tensor) -> bool {
    x.dims().last().copied().unwrap_or(0) >= min_len()
        && x.device().is_metal()
        && x.dtype() == DType::F32
        && w.dtype() == DType::F32
        && x.dims().len() == 3
        && x.dims()[0] == 1
        && w.dims().len() == 3
}

/// The op candle sees: `x` and `w` in, the convolution out.
struct MpsConv {
    dilation: usize,
    pad_left: usize,
    bias: Option<Tensor>,
    residual: Option<Tensor>,
}

impl CustomOp2 for MpsConv {
    fn name(&self) -> &'static str {
        "mps_conv1d"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> CResult<(CpuStorage, Shape)> {
        candle_core::bail!("mps_conv1d: Metal only")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> CResult<(candle_core::MetalStorage, Shape)> {
        use candle_core::MetalStorage;

        for l in [l1, l2] {
            if !l.is_contiguous() || l.start_offset() != 0 {
                candle_core::bail!("mps_conv1d: inputs must be contiguous and unoffset");
            }
        }
        let (_, cin, len) = l1.shape().dims3()?;
        let (k, _, cout) = l2.shape().dims3()?;
        let dev = s1.device().clone();
        let key = Key {
            device: dev.id(),
            cin,
            cout,
            k,
            dilation: self.dilation,
            pad_left: self.pad_left,
            bias: self.bias.is_some(),
            residual: self.residual.is_some(),
        };
        let g = cached(&dev, key).map_err(candle_core::Error::wrap)?;
        let exe = g.exec(&key, len).map_err(candle_core::Error::wrap)?;

        // MPSGraph runs on its own queue, so everything candle has queued must have landed.
        dev.wait_until_completed()?;
        let out = dev.new_buffer(cout * len, DType::F32, "mps_conv1d")?;

        unsafe {
            let td = |buf: &ProtocolObject<dyn MTLBuffer>, dims: &[isize]| {
                MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                    MPSGraphTensorData::alloc(),
                    buf,
                    &shape(dims),
                    MPSDataType::Float32,
                )
            };
            let buffer = |t: &Tensor| -> CResult<_> {
                let (bs, bl) = t.storage_and_layout();
                if bl.start_offset() != 0 || !bl.is_contiguous() {
                    candle_core::bail!(
                        "mps_conv1d: bias and residual must be contiguous and unoffset"
                    );
                }
                match &*bs {
                    candle_core::Storage::Metal(m) => Ok(m.buffer().clone()),
                    _ => candle_core::bail!("mps_conv1d: bias and residual must be on the device"),
                }
            };
            let bias = self.bias.as_ref().map(buffer).transpose()?;
            let residual = self.residual.as_ref().map(buffer).transpose()?;
            let mut inputs = Vec::with_capacity(exe.order.len());
            for feed in &exe.order {
                inputs.push(match feed {
                    Feed::Src => td(s1.buffer().as_ref(), &[1, cin as isize, 1, len as isize]),
                    Feed::Wts => td(
                        s2.buffer().as_ref(),
                        &[1, k as isize, cin as isize, cout as isize],
                    ),
                    Feed::Bias => td(
                        bias.as_ref()
                            .context("bias")
                            .map_err(candle_core::Error::wrap)?
                            .as_ref(),
                        &[1, cout as isize, 1, 1],
                    ),
                    Feed::Residual => td(
                        residual
                            .as_ref()
                            .context("residual")
                            .map_err(candle_core::Error::wrap)?
                            .as_ref(),
                        &[1, cout as isize, 1, len as isize],
                    ),
                });
            }
            let results = [td(
                AsRef::<ProtocolObject<dyn MTLBuffer>>::as_ref(out.as_ref()),
                &[1, cout as isize, 1, len as isize],
            )];
            exe.exe
                .runWithMTLCommandQueue_inputsArray_resultsArray_executionDescriptor(
                    &g.queue,
                    &NSArray::from_retained_slice(&inputs),
                    Some(&NSArray::from_retained_slice(&results)),
                    None,
                );
        }

        crate::stats::record(
            cout,
            k * cin,
            len,
            (cout * k * cin + cin * len + cout * len) as u64 * 4,
        );
        Ok((
            MetalStorage::new(out, dev, cout * len, DType::F32),
            (1, cout, len).into(),
        ))
    }
}

/// A centred 1-D convolution, `[1, cin, len] -> [1, cout, len]`.
///
/// `w` is `[k, cin, cout]`. `pad_left + pad_right` must be `(k - 1) * dilation`, which is
/// what "centred" means here.
pub fn centered_conv1d(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    dilation: usize,
    pad_left: usize,
) -> Result<Tensor> {
    centered_conv1d_residual(x, w, b, dilation, pad_left, None)
}

/// [`centered_conv1d`] plus `residual` `[1, cout, len]`, added inside the graph rather than as
/// another pass over the output.
pub fn centered_conv1d_residual(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    dilation: usize,
    pad_left: usize,
    residual: Option<&Tensor>,
) -> Result<Tensor> {
    let op = MpsConv {
        dilation,
        pad_left,
        bias: b.map(|t| t.flatten_all()?.contiguous()).transpose()?,
        residual: residual.map(|t| t.contiguous()).transpose()?,
    };
    Ok(x.contiguous()?.apply_op2_no_bwd(&w.contiguous()?, &op)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Against the gather route, at the generator's shapes and every dilation it uses.
    #[test]
    fn matches_gather_route() -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        let _gpu = crate::gpu_guard();
        let Some(d) = crate::usable_metal() else {
            return Ok(());
        };
        for (cin, cout, len) in [
            (128usize, 128usize, 4441usize),
            (256, 256, 8040),
            (22, 128, 511),
        ] {
            for (k, dil) in [(1usize, 1usize), (3, 1), (3, 5), (7, 3), (11, 1), (11, 5)] {
                let x = Tensor::randn(0f32, 1., (1, cin, len), &d)?;
                let w = Tensor::randn(0f32, 0.02, (cout, cin, k), &d)?;
                let bias = Tensor::randn(0f32, 1., cout, &d)?;
                let w_tap = crate::tap_major_weight(&w)?;
                let kio = w.permute((2, 1, 0))?.contiguous()?;
                let pad = (k - 1) * dil / 2;
                for b in [Some(&bias), None] {
                    let want = crate::centered_conv1d_gemm(&x, &w_tap, b, k, dil, pad)?;
                    let got = centered_conv1d(&x, &kio, b, dil, pad)?;
                    assert_eq!(want.dims(), got.dims());
                    let res = Tensor::randn(0f32, 1., (1, cout, len), &d)?;
                    let fused = centered_conv1d_residual(&x, &kio, b, dil, pad, Some(&res))?;
                    let (_, rel_res) = crate::abs_and_rel(&(&want + &res)?, &fused)?;
                    assert!(
                        rel_res < 1e-5,
                        "residual {cin}->{cout} k={k}: rel {rel_res:.3e}"
                    );
                    let (abs, rel) = crate::abs_and_rel(&want, &got)?;
                    assert!(
                        rel < 1e-5,
                        "{cin}->{cout} @ {len} k={k} d={dil} bias={}: abs {abs:.3e} rel {rel:.3e}",
                        b.is_some()
                    );
                }
            }
        }
        Ok(())
    }
}
