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
//! A 1-D convolution is a 2-D one with a height of 1, and the weight candle already holds
//! — `[cout, cin, k]` — is MPSGraph's `OIHW` with the height axis inserted.
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
    MPSGraph, MPSGraphConvolution2DOpDescriptor, MPSGraphPaddingStyle, MPSGraphTensor,
    MPSGraphTensorData, MPSGraphTensorNamedDataLayout,
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
    out: Retained<MPSGraphTensor>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

// The Metal objects are used behind the cache's mutex and never handed across a suspend
// point; objc2 does not know that, so the marker is asserted here rather than inferred.
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

#[derive(PartialEq, Eq, Hash)]
struct Key {
    device: candle_core::metal_backend::DeviceId,
    cin: usize,
    cout: usize,
    k: usize,
    dilation: usize,
    pad_left: usize,
    bias: bool,
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
                    key.cout as isize,
                    key.cin as isize,
                    1,
                    key.k as isize,
                ])),
                MPSDataType::Float32,
                None,
            );
            let desc = MPSGraphConvolution2DOpDescriptor::
                descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout(
                    1, 1, key.dilation, 1, 1, key.pad_left, pad_right, 0, 0,
                    MPSGraphPaddingStyle::Explicit,
                    MPSGraphTensorNamedDataLayout::NCHW,
                    MPSGraphTensorNamedDataLayout::OIHW,
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
                out,
                queue,
            })
        }
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
        let (cout, _, k) = l2.shape().dims3()?;
        let dev = s1.device().clone();
        let g = cached(
            &dev,
            Key {
                device: dev.id(),
                cin,
                cout,
                k,
                dilation: self.dilation,
                pad_left: self.pad_left,
                bias: self.bias.is_some(),
            },
        )
        .map_err(candle_core::Error::wrap)?;

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
            let mut keys: Vec<&MPSGraphTensor> = vec![&g.src, &g.wts];
            let mut vals = vec![
                td(s1.buffer().as_ref(), &[1, cin as isize, 1, len as isize]),
                td(
                    s2.buffer().as_ref(),
                    &[cout as isize, cin as isize, 1, k as isize],
                ),
            ];
            let held;
            if let (Some(bt), Some(bten)) = (&self.bias, &g.bias) {
                let (bs, bl) = bt.storage_and_layout();
                let bb = match &*bs {
                    candle_core::Storage::Metal(m) => m.buffer().clone(),
                    _ => candle_core::bail!("mps_conv1d: bias must be on the device"),
                };
                if bl.start_offset() != 0 {
                    candle_core::bail!("mps_conv1d: bias must be unoffset");
                }
                held = bb;
                keys.push(bten);
                vals.push(td(
                    AsRef::<ProtocolObject<dyn MTLBuffer>>::as_ref(held.as_ref()),
                    &[1, cout as isize, 1, 1],
                ));
            }
            let feeds = NSDictionary::from_retained_objects(&keys, &vals);
            let results = NSDictionary::from_retained_objects(
                &[&*g.out],
                &[td(
                    AsRef::<ProtocolObject<dyn MTLBuffer>>::as_ref(out.as_ref()),
                    &[1, cout as isize, 1, len as isize],
                )],
            );
            g.graph
                .runWithMTLCommandQueue_feeds_targetOperations_resultsDictionary(
                    &g.queue, &feeds, None, &results,
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
/// `w` is candle's own conv weight layout, `[cout, cin, k]`. `pad_left + pad_right` must
/// be `(k - 1) * dilation`, which is what "centred" means here.
pub fn centered_conv1d(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    dilation: usize,
    pad_left: usize,
) -> Result<Tensor> {
    let op = MpsConv {
        dilation,
        pad_left,
        bias: b.map(|t| t.flatten_all()?.contiguous()).transpose()?,
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
                let pad = (k - 1) * dil / 2;
                for b in [Some(&bias), None] {
                    let want = crate::centered_conv1d_gemm(&x, &w_tap, b, k, dil, pad)?;
                    let got = centered_conv1d(&x, &w, b, dil, pad)?;
                    assert_eq!(want.dims(), got.dims());
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
