//! The MPSGraph modules' surface without Metal: nothing is eligible and nothing compiles, which
//! is what every caller already does with a device that is not Metal.

pub mod mpsblock {
    use anyhow::{bail, Result};
    use candle_core::Tensor;

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct ConvShape {
        pub k: usize,
        pub dilation: usize,
    }

    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    pub struct BlockKey {
        pub channels: usize,
        pub pairs: Vec<(ConvShape, ConvShape)>,
        pub eps_bits: u32,
    }

    pub fn key(
        _device: &candle_core::Device,
        _channels: usize,
        _pairs: Vec<(ConvShape, ConvShape)>,
        _eps: f32,
    ) -> Option<BlockKey> {
        None
    }

    pub fn prewarm(_device: &candle_core::Device, _jobs: Vec<(BlockKey, usize)>) {}

    pub fn apply(
        _key: &BlockKey,
        _x: &Tensor,
        _params: &Tensor,
        _convs: &[(&Tensor, &Tensor)],
        _valid: usize,
    ) -> Result<Tensor> {
        bail!("built without Metal")
    }
}

pub mod mpsconv {
    use anyhow::{bail, Result};
    use candle_core::Tensor;

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

    pub fn prewarm(_device: &candle_core::Device, _specs: Vec<Spec>) {}

    pub fn eligible_len(_device: &candle_core::Device, _len: usize) -> bool {
        false
    }

    pub fn eligible(_x: &Tensor, _w: &Tensor) -> bool {
        false
    }

    pub fn centered_conv1d(
        _x: &Tensor,
        _w: &Tensor,
        _b: Option<&Tensor>,
        _dilation: usize,
        _pad_left: usize,
    ) -> Result<Tensor> {
        bail!("built without Metal")
    }

    pub fn centered_conv1d_residual(
        _x: &Tensor,
        _w: &Tensor,
        _b: Option<&Tensor>,
        _dilation: usize,
        _pad_left: usize,
        _residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        bail!("built without Metal")
    }
}

pub mod mpsnlc {
    use anyhow::{bail, Result};
    use candle_core::Tensor;

    pub fn eligible(_x: &Tensor, _w: &Tensor) -> bool {
        false
    }

    pub fn causal_conv1d(
        _x: &Tensor,
        _w: &Tensor,
        _b: &Tensor,
        _dilation: usize,
    ) -> Result<Tensor> {
        bail!("built without Metal")
    }
}
