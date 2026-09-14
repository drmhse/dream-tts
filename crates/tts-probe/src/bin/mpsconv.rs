//! Apple's own convolution against this crate's gather-plus-matmul route.
//!
//! `tapconv` established that a hand-written GEMM loses to MPS by 4x, which is an argument
//! for using more of MPS, not less. candle never touches MPSGraph, and MPSGraph has a
//! tuned `convolution2D` that does not materialise an im2col matrix at all — the thing
//! `tapconv` was trying and failing to do by hand.
//!
//! A 1-D convolution is a 2-D one with a height of 1. Shapes are Kokoro's generator.
//!
//! The f16 row settles a question the earlier f16 measurement could not: that one timed
//! candle's matmul, and MPSGraph is a different engine. It is not: 1.86x against f32's
//! 1.71x at `128ch @ 48240, k=11`, so half the bytes buy about 8% and the rest of the
//! arithmetic runs at the same rate. Not worth a dtype boundary through the generator.
//!
//! Run: `cargo run -p tts-probe --release --bin mpsconv`

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
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
use tts_bench::Harness;

fn shape(dims: &[usize]) -> Retained<NSArray<NSNumber>> {
    let v: Vec<Retained<NSNumber>> = dims.iter().map(|d| NSNumber::new_usize(*d)).collect();
    NSArray::from_retained_slice(&v)
}

/// The same shape with the length left unspecified, so one compiled graph serves every
/// utterance. Whether that costs anything is the question this probe exists to answer.
fn dyn_shape(dims: &[usize], dynamic: bool) -> Retained<NSArray<NSNumber>> {
    if !dynamic {
        return shape(dims);
    }
    let v: Vec<Retained<NSNumber>> = dims
        .iter()
        .enumerate()
        .map(|(i, d)| if i == 3 { NSNumber::new_isize(-1) } else { NSNumber::new_usize(*d) })
        .collect();
    NSArray::from_retained_slice(&v)
}

/// One conv shape compiled once and re-run: what a real integration would keep cached.
struct GraphConv {
    graph: Retained<MPSGraph>,
    src: Retained<MPSGraphTensor>,
    wts: Retained<MPSGraphTensor>,
    out: Retained<MPSGraphTensor>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    cin: usize,
    cout: usize,
    len: usize,
    k: usize,
    dt: MPSDataType,
}

impl GraphConv {
    fn new(
        dev: &ProtocolObject<dyn MTLDevice>,
        cin: usize,
        cout: usize,
        len: usize,
        k: usize,
        dil: usize,
        dynamic: bool,
        dt: MPSDataType,
    ) -> Result<Self> {
        let pad = (k - 1) * dil / 2;
        unsafe {
            let graph = MPSGraph::new();
            let src = graph.placeholderWithShape_dataType_name(
                Some(&dyn_shape(&[1, cin, 1, len], dynamic)),
                dt,
                None,
            );
            let wts = graph.placeholderWithShape_dataType_name(
                Some(&shape(&[cout, cin, 1, k])),
                dt,
                None,
            );
            let desc = MPSGraphConvolution2DOpDescriptor::
                descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout(
                    1, 1, dil, 1, 1, pad, (k - 1) * dil - pad, 0, 0,
                    MPSGraphPaddingStyle::Explicit,
                    MPSGraphTensorNamedDataLayout::NCHW,
                    MPSGraphTensorNamedDataLayout::OIHW,
                )
                .context("conv descriptor")?;
            let out = graph.convolution2DWithSourceTensor_weightsTensor_descriptor_name(
                &src, &wts, &desc, None,
            );
            let queue = dev.newCommandQueue().context("command queue")?;
            Ok(Self { graph, src, wts, out, queue, cin, cout, len, k, dt })
        }
    }

    fn run(
        &self,
        x: &ProtocolObject<dyn MTLBuffer>,
        w: &ProtocolObject<dyn MTLBuffer>,
    ) -> Result<()> {
        unsafe {
            let xd = MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                MPSGraphTensorData::alloc(),
                x,
                &shape(&[1, self.cin, 1, self.len]),
                self.dt,
            );
            let wd = MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                MPSGraphTensorData::alloc(),
                w,
                &shape(&[self.cout, self.cin, 1, self.k]),
                self.dt,
            );
            let feeds = NSDictionary::from_retained_objects(&[&*self.src, &*self.wts], &[xd, wd]);
            let targets = NSArray::from_retained_slice(&[self.out.clone()]);
            let _ = self.graph.runWithMTLCommandQueue_feeds_targetTensors_targetOperations(
                &self.queue,
                &feeds,
                &targets,
                None,
            );
        }
        Ok(())
    }
}

/// The MTLBuffer behind a candle tensor, so both routes read the same bytes.
fn buffer(t: &Tensor) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let (storage, _) = t.storage_and_layout();
    match &*storage {
        candle_core::Storage::Metal(m) => {
            use candle_core::backend::BackendStorage;
            let _ = m.dtype();
            Ok(Retained::from(m.buffer().as_ref()))
        }
        _ => anyhow::bail!("not a metal tensor"),
    }
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let raw: Retained<ProtocolObject<dyn MTLDevice>> = match &dev {
        Device::Metal(m) => Retained::from(m.device().as_ref()),
        _ => anyhow::bail!("metal only"),
    };
    let mut h = Harness::new(&dev, 7)?;

    for (label, c, len) in [("stage0 256ch@8040", 256usize, 8040usize), ("stage1 128ch@48240", 128, 48240)] {
        for k in [3usize, 7, 11] {
            let x = Tensor::randn(0f32, 1.0, (1, c, len), &dev)?.contiguous()?;
            let w = Tensor::randn(0f32, 0.02, (c, c, k), &dev)?.contiguous()?;
            let w_tap = tts_nn::tap_major_weight(&w)?;
            let xb = buffer(&x)?;
            let wb = buffer(&w)?;
            let t0 = std::time::Instant::now();
            let g = GraphConv::new(&raw, c, c, len, k, 1, false, MPSDataType::Float32)?;
            let built = t0.elapsed().as_secs_f64() * 1000.0;
            let t1 = std::time::Instant::now();
            g.run(&xb, &wb)?;
            let first = t1.elapsed().as_secs_f64() * 1000.0;
            let xh = x.to_dtype(candle_core::DType::F16)?.contiguous()?;
            let wh = w.to_dtype(candle_core::DType::F16)?.contiguous()?;
            let xhb = buffer(&xh)?;
            let whb = buffer(&wh)?;
            let gh = GraphConv::new(&raw, c, c, len, k, 1, true, MPSDataType::Float16)?;
            let mut bh = || { gh.run(&xhb, &whb).unwrap(); Ok(()) };
            let gd = GraphConv::new(&raw, c, c, len, k, 1, true, MPSDataType::Float32)?;
            let t2 = std::time::Instant::now();
            gd.run(&xb, &wb)?;
            let first_dyn = t2.elapsed().as_secs_f64() * 1000.0;
            println!(
                "\n  build {built:.2} ms, first run {first:.1} ms (static) / {first_dyn:.1} ms (dynamic length)"
            );
            let mut bd = || { gd.run(&xb, &wb).unwrap(); Ok(()) };

            let mut a = || {
                tts_nn::centered_conv1d_gemm(&x, &w_tap, None, k, 1, (k - 1) / 2).unwrap();
                Ok(())
            };
            let mut b = || {
                g.run(&xb, &wb).unwrap();
                Ok(())
            };
            let mut variants: Vec<(&str, &mut dyn FnMut() -> candle_core::Result<()>)> =
                vec![
                    ("gather + candle matmul", &mut a),
                    ("MPSGraph convolution2D", &mut b),
                    ("MPSGraph, dynamic length", &mut bd),
                    ("MPSGraph f16", &mut bh),
                ];
            h.ab(&format!("{label} k={k}"), &mut variants)?;
        }
    }
    h.report_drift()?;
    Ok(())
}
