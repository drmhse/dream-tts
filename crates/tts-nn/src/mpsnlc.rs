//! A channels-last causal conv through MPSGraph: [`crate::mpsconv`] for [`crate::nlc`]'s layout.
//!
//! `[b, len, cin]` is NHWC at height 1 and the tap weight `[k, cin, cout]` is HWIO, so neither
//! side needs a copy. The bias is a graph node, not a second pass.

use anyhow::{Context, Result};
use candle_core::backend::BackendStorage;
use candle_core::{
    CpuStorage, CustomOp3, DType, Layout, MetalDevice, Result as CResult, Shape, Tensor,
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

const DYNAMIC: isize = -1;

fn shape(dims: &[isize]) -> Retained<NSArray<NSNumber>> {
    let v: Vec<Retained<NSNumber>> = dims.iter().map(|d| NSNumber::new_isize(*d)).collect();
    NSArray::from_retained_slice(&v)
}

fn mps_type(dt: DType) -> Result<MPSDataType> {
    Ok(match dt {
        DType::F32 => MPSDataType::Float32,
        DType::F16 => MPSDataType::Float16,
        other => anyhow::bail!("mps_nlc_conv: {other:?} unsupported"),
    })
}

struct Graph {
    graph: Retained<MPSGraph>,
    src: Retained<MPSGraphTensor>,
    wts: Retained<MPSGraphTensor>,
    bias: Retained<MPSGraphTensor>,
    out: Retained<MPSGraphTensor>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

// Used only behind the cache's mutex; objc2 cannot infer that.
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct Key {
    device: candle_core::metal_backend::DeviceId,
    cin: usize,
    cout: usize,
    k: usize,
    dilation: usize,
    dtype: DType,
}

impl Graph {
    fn build(dev: &MetalDevice, key: &Key) -> Result<Self> {
        let dt = mps_type(key.dtype)?;
        unsafe {
            let graph = MPSGraph::new();
            let src = graph.placeholderWithShape_dataType_name(
                Some(&shape(&[1, 1, DYNAMIC, key.cin as isize])),
                dt,
                None,
            );
            let wts = graph.placeholderWithShape_dataType_name(
                Some(&shape(&[
                    1,
                    key.k as isize,
                    key.cin as isize,
                    key.cout as isize,
                ])),
                dt,
                None,
            );
            let desc = MPSGraphConvolution2DOpDescriptor::
                descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout(
                    1, 1, key.dilation, 1, 1, (key.k - 1) * key.dilation, 0, 0, 0,
                    MPSGraphPaddingStyle::Explicit,
                    MPSGraphTensorNamedDataLayout::NHWC,
                    MPSGraphTensorNamedDataLayout::HWIO,
                )
                .context("MPSGraph convolution descriptor")?;
            let conv = graph.convolution2DWithSourceTensor_weightsTensor_descriptor_name(
                &src, &wts, &desc, None,
            );
            let bias = graph.placeholderWithShape_dataType_name(
                Some(&shape(&[1, 1, 1, key.cout as isize])),
                dt,
                None,
            );
            let out = graph.additionWithPrimaryTensor_secondaryTensor_name(&conv, &bias, None);
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

struct MpsNlcConv {
    dilation: usize,
}

impl CustomOp3 for MpsNlcConv {
    fn name(&self) -> &'static str {
        "mps_nlc_conv"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CResult<(CpuStorage, Shape)> {
        candle_core::bail!("mps_nlc_conv: Metal only")
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
    ) -> CResult<(candle_core::MetalStorage, Shape)> {
        use candle_core::MetalStorage;

        for l in [l1, l2, l3] {
            if !l.is_contiguous() || l.start_offset() != 0 {
                candle_core::bail!("mps_nlc_conv: inputs must be contiguous and unoffset");
            }
        }
        let dtype = s1.dtype();
        if s2.dtype() != dtype || s3.dtype() != dtype {
            candle_core::bail!("mps_nlc_conv: mixed dtypes");
        }
        let (_, len, cin) = l1.shape().dims3()?;
        let (k, _, cout) = l2.shape().dims3()?;
        let dev = s1.device().clone();
        let key = Key {
            device: dev.id(),
            cin,
            cout,
            k,
            dilation: self.dilation,
            dtype,
        };
        let g = cached(&dev, key).map_err(candle_core::Error::wrap)?;
        let dt = mps_type(dtype).map_err(candle_core::Error::wrap)?;

        // MPSGraph runs on its own queue: candle's pending work has to land first.
        dev.wait_until_completed()?;
        let out = dev.new_buffer(len * cout, dtype, "mps_nlc_conv")?;

        unsafe {
            let td = |buf: &ProtocolObject<dyn MTLBuffer>, dims: &[isize]| {
                MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                    MPSGraphTensorData::alloc(),
                    buf,
                    &shape(dims),
                    dt,
                )
            };
            let keys: Vec<&MPSGraphTensor> = vec![&g.src, &g.wts, &g.bias];
            let vals = vec![
                td(s1.buffer().as_ref(), &[1, 1, len as isize, cin as isize]),
                td(
                    s2.buffer().as_ref(),
                    &[1, k as isize, cin as isize, cout as isize],
                ),
                td(s3.buffer().as_ref(), &[1, 1, 1, cout as isize]),
            ];
            let feeds = NSDictionary::from_retained_objects(&keys, &vals);
            let results = NSDictionary::from_retained_objects(
                &[&*g.out],
                &[td(
                    AsRef::<ProtocolObject<dyn MTLBuffer>>::as_ref(out.as_ref()),
                    &[1, 1, len as isize, cout as isize],
                )],
            );
            g.graph
                .runWithMTLCommandQueue_feeds_targetOperations_resultsDictionary(
                    &g.queue, &feeds, None, &results,
                );
        }

        Ok((
            MetalStorage::new(out, dev, len * cout, dtype),
            (1, len, cout).into(),
        ))
    }
}

/// Whether this conv is one MPSGraph wins, as well as one it can run.
///
/// Measured against `nlcconv` on qwen3tts's codec, f32, where the two agree bit for bit.
/// MPSGraph wins while the reduction is short and loses once it is long:
///
/// | cin -> cout, k | len | nlcconv | MPSGraph |
/// |---|---|---|---|
/// | 96 -> 1, 7 | 624000 | 60.6 ms | 22.3 |
/// | 96 -> 96, 7 | 624000 | 30.3 | 24.1 |
/// | 192 -> 288, 2 | 208000 | 18.4 | 14.2 |
/// | 288 -> 288, 7 | 100000 | 43.8 | 34.4 |
/// | 384 -> 384, 7 | 52000 | 40.5 | 41.7 |
/// | 768 -> 1920, 2 | 10400 | 23.9 | 28.6 |
/// | 768 -> 768, 1 | 10400 | 5.3 | 4.0, but 0.50x at 96 channels |
///
/// Each call also waits for candle's queue, so short inputs stay on the fused kernel.
pub fn eligible(x: &Tensor, w: &Tensor) -> bool {
    let (Ok((b, len, cin)), Ok((k, _, _))) = (x.dims3(), w.dims3()) else {
        return false;
    };
    x.device().is_metal()
        && matches!(x.dtype(), DType::F32 | DType::F16)
        && w.dtype() == x.dtype()
        && b == 1
        && k > 1
        && len >= 4096
        && cin <= 384
        && cin * k <= 2048
}

/// `[1, len, cin] -> [1, len, cout]`, left-padded by `(k - 1) * dilation`. `w` is the tap
/// weight `[k, cin, cout]`.
pub fn causal_conv1d(x: &Tensor, w: &Tensor, b: &Tensor, dilation: usize) -> Result<Tensor> {
    Ok(x.contiguous()?.apply_op3_no_bwd(
        &w.contiguous()?,
        &b.flatten_all()?.contiguous()?,
        &MpsNlcConv { dilation },
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Against `nlc`'s fused kernel, at the codec's shapes this takes and both dtypes.
    #[test]
    fn matches_nlc() -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        let _gpu = crate::gpu_guard();
        let Some(d) = crate::usable_metal() else {
            return Ok(());
        };
        for (cin, cout, len, k, dil) in [
            (96usize, 1usize, 9000usize, 7usize, 1usize),
            (96, 96, 9000, 7, 9),
            (192, 288, 5000, 2, 1),
            (288, 288, 4096, 7, 3),
        ] {
            for dt in [DType::F32, DType::F16] {
                let x = Tensor::randn(0f32, 1., (1, len, cin), &d)?.to_dtype(dt)?;
                let w = (Tensor::randn(0f32, 1., (k, cin, cout), &d)?
                    * (1.0 / ((k * cin) as f64).sqrt()))?
                .to_dtype(dt)?;
                let b = Tensor::randn(0f32, 0.1, cout, &d)?.to_dtype(dt)?;
                assert!(eligible(&x, &w), "{cin}->{cout} k={k} should take MPSGraph");
                let want =
                    crate::nlc::causal_conv1d(&x, &w, Some(&b), dil)?.to_dtype(DType::F32)?;
                let got = causal_conv1d(&x, &w, &b, dil)?.to_dtype(DType::F32)?;
                assert_eq!(want.dims(), got.dims());
                let (abs, rel) = crate::abs_and_rel(&want, &got)?;
                let tol = if dt == DType::F32 { 1e-5 } else { 2e-3 };
                assert!(
                    rel < tol,
                    "{cin}->{cout} @ {len} k={k} d={dil} {dt:?}: abs {abs:.3e} rel {rel:.3e}"
                );
            }
        }
        Ok(())
    }
}
