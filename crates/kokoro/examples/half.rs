//! `cargo run -p kokoro --release --example half -- in.safetensors out.safetensors`
//!
//! The checkpoint stored at half precision, for shipping. `Weights::get` widens every tensor to
//! f32 on read, so the only change is each weight's rounding, which the bundle gate measures.
use candle_core::{safetensors, DType, Device};
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let (src, dst) = (args.next().expect("in"), args.next().expect("out"));
    let tensors = safetensors::load(&src, &Device::Cpu)?;
    let mut out = HashMap::new();
    for (name, t) in tensors {
        let t = if t.dtype() == DType::F32 {
            t.to_dtype(DType::F16)?
        } else {
            t
        };
        out.insert(name, t);
    }
    safetensors::save(&out, &dst)?;
    Ok(())
}
