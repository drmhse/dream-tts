//! The custom Metal kernels this crate ships, and the pipeline cache behind them.
//!
//! Everything here exists because of two measurements in `docs/reference.md#performance`:
//!
//! - **Finding 2**: candle performs no fusion whatsoever. `snake` as five composed ops
//!   costs 11.5 ms at `[1, 96, 131072]`, and the five ops measured individually sum to
//!   13.7 ms — so every elementwise expression pays one full round-trip to device memory
//!   *per operator*. A single fused pass costs what `affine` costs, 1.33 ms: **8.7x**.
//! - **The im2col gather**: `cat(dim=0)` builds the conv matrix at ~24 GB/s where the
//!   hardware manages ~81-130 GB/s.
//!
//! Both are the same shape of problem — candle composes correct ops that each re-read
//! memory — and both are fixed the same way, by doing the whole expression in one pass.
//!
//! The kernels live in one source string compiled once per device, because compiling a
//! library costs milliseconds and these run hundreds of times per utterance.

#[cfg(feature = "metal")]
use candle_core::Result;

#[cfg(feature = "metal")]
pub(crate) const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---- im2col ------------------------------------------------------------------
//
// dst[(t * cin + c) * l_in + l] = src[c * l_in + l + t * dilation - pad], or 0 when the
// tap reaches past either edge of the signal. `pad` is the left context; causal
// callers pass (k-1)*dilation, for which the upper bound never fires.
//
// The indices arrive as grid coordinates, so there is not a single division in the body.
// candle's own im2col recovers four indices from a linear thread id with three size_t
// divisions each, and at 88 M elements that arithmetic — not the traffic — is what makes
// it 0.66x slower than the `cat` route it was meant to replace.
kernel void im2col_tap_major_f32(
    device const float *src   [[buffer(0)]],
    device float       *dst   [[buffer(1)]],
    constant uint      &l_in  [[buffer(2)]],
    constant uint      &cin   [[buffer(3)]],
    constant uint      &dil   [[buffer(4)]],
    constant uint      &pad   [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint l = gid.x;
    if (l >= l_in) { return; }
    const uint c = gid.y;
    const uint t = gid.z;

    const uint s = l + t * dil;
    dst[(t * cin + c) * l_in + l] =
        (s < pad || s >= pad + l_in) ? 0.0f : src[c * l_in + (s - pad)];
}

// ---- head transpose ------------------------------------------------------------
//
// [b, n, h, d] -> [b, h, n, d], the layout `sdpa` wants for multi-head attention.
//
// `d` is the fast axis and is unit-stride on *both* sides, so every thread's read and
// write coalesce. candle's `transpose(1,2).contiguous()` manages ~4.6 GB/s on this shape;
// there is nothing about the movement that requires that.
kernel void head_transpose_f32(
    device const float *src [[buffer(0)]],
    device float       *dst [[buffer(1)]],
    constant uint      &n   [[buffer(2)]],
    constant uint      &hd  [[buffer(3)]],
    constant uint      &dim [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint d  = gid.x;
    if (d >= dim) { return; }
    const uint pos = gid.y;          // n
    const uint bh  = gid.z;          // b * hd + h
    const uint b   = bh / hd;
    const uint hh  = bh - b * hd;

    dst[(bh * n + pos) * dim + d] = src[((b * n + pos) * hd + hh) * dim + d];
}

// ---- modulate tail and gated residual -------------------------------------------
//
// Both are `[b, n, d]` against a `[b, 1, d]` vector, which candle does with
// `broadcast_mul` at 22 GB/s — 3.6x slower than a plain unary op, for what is only index
// arithmetic (Finding 2). A DiT block runs two of each, 660 times per utterance.
//
// `d` is the fast axis and unit-stride on both sides; the broadcast vector is indexed
// directly rather than expanded.

// out = x * (1 + scale[b,d]) + shift[b,d]  — the affine half of `modulate`, after the norm.
kernel void modulate_affine_f32(
    device const float *x     [[buffer(0)]],
    device const float *scale [[buffer(1)]],
    device const float *shift [[buffer(2)]],
    device float       *dst   [[buffer(3)]],
    constant uint      &n     [[buffer(4)]],
    constant uint      &dim   [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    if (d >= dim) { return; }
    const uint pos = gid.y;
    const uint b   = gid.z;

    const uint i = (b * n + pos) * dim + d;
    const uint j = b * dim + d;
    dst[i] = x[i] * (1.0f + scale[j]) + shift[j];
}

// out = r + y * gate[b,d]  — the residual add and its gate, in one pass instead of two.
kernel void gate_residual_f32(
    device const float *r    [[buffer(0)]],
    device const float *y    [[buffer(1)]],
    device const float *gate [[buffer(2)]],
    device float       *dst  [[buffer(3)]],
    constant uint      &n    [[buffer(4)]],
    constant uint      &dim  [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    if (d >= dim) { return; }
    const uint pos = gid.y;
    const uint b   = gid.z;

    const uint i = (b * n + pos) * dim + d;
    dst[i] = r[i] + y[i] * gate[b * dim + d];
}

// ---- snake -------------------------------------------------------------------
//
// y = x + sin^2(x), for inputs whose alpha has already been folded into the preceding
// conv's output weights. Three composed candle ops (sin, sqr, add) become one pass.
kernel void snake_folded_f32(
    device const float *src [[buffer(0)]],
    device float       *dst [[buffer(1)]],
    constant uint      &n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    const float x = src[gid];
    const float s = sin(x);
    dst[gid] = x + s * s;
}

// ---- snake with a per-channel alpha -------------------------------------------
//
// y = u + sin^2(u) where u = alpha[c] * x, for the leading snake of a residual group —
// there the block input also feeds the skip, so alpha cannot be folded away.
//
// alpha is indexed by the grid's y axis rather than broadcast, which matters: a candle
// `broadcast_mul` of [1,C,1] against [1,C,L] runs at 22 GB/s, 3.6x slower than a plain
// unary op, and snake as written contained two of them.
kernel void snake_alpha_f32(
    device const float *src   [[buffer(0)]],
    device const float *alpha [[buffer(1)]],
    device float       *dst   [[buffer(2)]],
    constant uint      &len   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint l = gid.x;
    if (l >= len) { return; }
    const uint c = gid.y;

    const float u = alpha[c] * src[c * len + l];
    const float s = sin(u);
    dst[c * len + l] = u + s * s;
}

// ---- snake beta ---------------------------------------------------------------
//
// y = x + beta_recip[c] * sin^2(alpha[c] * x) — SnakeBeta, with a per-channel amplitude
// as well as a per-channel frequency. Neither can be folded away: the input also feeds a
// skip, and beta is independent of alpha. `snake_full` is six composed ops, so it pays six
// round trips where this pays one.
kernel void snake_beta_f32(
    device const float *src    [[buffer(0)]],
    device const float *alpha  [[buffer(1)]],
    device const float *brecip [[buffer(2)]],
    device float       *dst    [[buffer(3)]],
    constant uint      &len    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint l = gid.x;
    if (l >= len) { return; }
    const uint c = gid.y;

    const uint i = c * len + l;
    const float x = src[i];
    const float s = sin(alpha[c] * x);
    dst[i] = x + brecip[c] * s * s;
}

// ---- adain halves --------------------------------------------------------------
//
// Instance norm plus style scale and shift, in two passes instead of ten. The
// disease is the same one `snake_alpha` names: every `broadcast_*` against a
// `[1, C, 1]` parameter runs ~5x slower than a plain op, and one AdaIN contains
// four of them plus a division. `sub_sqr` fuses the centred square the variance
// needs; `adain_apply` fuses normalise, scale and shift with direct per-channel
// indexing. The two reductions stay candle's — its `mean` is already at full
// bandwidth, and reimplementing a reduction risks its numerics for no traffic
// saved.
kernel void sub_sqr_f32(
    device const float *src [[buffer(0)]],
    device const float *m   [[buffer(1)]],
    device float       *dst [[buffer(2)]],
    constant uint      &len [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint l = gid.x;
    if (l >= len) { return; }
    const uint c = gid.y;

    const float d = src[c * len + l] - m[c];
    dst[c * len + l] = d * d;
}

// out = (x - mean) * rsqrt(var + eps) * (gamma + 1) + beta, per channel.
// `prm` packs the four `[C]` vectors as mean, var, gamma, beta.
kernel void adain_apply_f32(
    device const float *src [[buffer(0)]],
    device const float *prm [[buffer(1)]],
    device float       *dst [[buffer(2)]],
    constant uint      &len [[buffer(3)]],
    constant uint      &chn [[buffer(4)]],
    constant float     &eps [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint l = gid.x;
    if (l >= len) { return; }
    const uint c = gid.y;

    const float mean = prm[c];
    const float var = prm[chn + c];
    const float gamma = prm[2u * chn + c];
    const float beta = prm[3u * chn + c];
    const uint i = c * len + l;
    dst[i] = (src[i] - mean) * rsqrt(var + eps) * (gamma + 1.0f) + beta;
}

// Per-channel mean and variance in one read of the signal, written as [2, C].
//
// candle's `mean` is one dispatch per reduction and AdaIN needs two of them with a
// materialised `(x-m)^2` in between: three passes over 24 MB where the data only has to
// be read once. One threadgroup per channel, sum and sum-of-squares together.
//
// Squares are accumulated raw rather than by Welford. Every input here is a convolution
// output whose mean is small against its spread, so the cancellation Welford defends
// against does not arise; `moments_match_composed` bounds what it costs.
//
// Only the first `valid` samples of each row count: a signal padded to a length bucket keeps
// its statistics.
kernel void channel_moments_f32(
    device const float *src  [[buffer(0)]],
    device float       *dst  [[buffer(1)]],
    constant uint      &len  [[buffer(2)]],
    constant uint      &chn  [[buffer(3)]],
    constant uint      &valid [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 ntid3 [[threads_per_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  slid [[thread_index_in_simdgroup]],
    uint  nsg  [[simdgroups_per_threadgroup]])
{
    threadgroup float psum[32];
    threadgroup float psq[32];

    const uint tid = tid3.x;
    const uint ntid = ntid3.x;
    const uint c = tgid.y;
    device const float *row = src + (ulong)c * len;
    float s = 0.0f;
    float q = 0.0f;
    for (uint i = tid; i < valid; i += ntid) {
        const float v = row[i];
        s += v;
        q += v * v;
    }
    s = simd_sum(s);
    q = simd_sum(q);
    if (slid == 0) { psum[sgid] = s; psq[sgid] = q; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid != 0) { return; }
    s = simd_sum(slid < nsg ? psum[slid] : 0.0f);
    q = simd_sum(slid < nsg ? psq[slid] : 0.0f);
    if (slid != 0) { return; }
    const float m = s / (float)valid;
    dst[c] = m;
    dst[chn + c] = max(q / (float)valid - m * m, 0.0f);
}

// AdaIN's tail and SnakeBeta in one pass: the two always appear together in the
// generator's residual blocks, and separately they read and write the signal twice.
// `prm` packs six [C] vectors: mean, var, gamma, beta, alpha, beta_recip.
kernel void adain_snake_f32(
    device const float *src [[buffer(0)]],
    device const float *prm [[buffer(1)]],
    device float       *dst [[buffer(2)]],
    constant uint      &len [[buffer(3)]],
    constant uint      &chn [[buffer(4)]],
    constant float     &eps [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint l = gid.x;
    if (l >= len) { return; }
    const uint c = gid.y;

    const float mean = prm[c];
    const float var = prm[chn + c];
    const float gamma = prm[2u * chn + c];
    const float beta = prm[3u * chn + c];
    const float alpha = prm[4u * chn + c];
    const float brecip = prm[5u * chn + c];
    const uint i = c * len + l;
    const float y = (src[i] - mean) * rsqrt(var + eps) * (gamma + 1.0f) + beta;
    const float sn = sin(alpha * y);
    dst[i] = y + brecip * sn * sn;
}

// ---- convolution without the im2col matrix -------------------------------------
//
// The gather is a third of the generator's convolution time and every byte of it is
// written only to be read straight back: 271 MB for one `k=11, 128ch @ 48240` tap-major
// matrix. This computes the same GEMM with the tap window read from `x` into threadgroup
// memory instead, so the matrix never exists.
//
// The tile is the whole of M. That is the point: with 128 output channels one threadgroup
// row covers them all, so the input is read once rather than once per M-tile, and the
// weight — 720 KB at the widest — is small enough to stay in cache across the N tiles.
//
// A K step of 8 never straddles two taps, because `cin` is a multiple of 8 on every conv
// routed here. So the tap index is computed once per step, not once per element, and the
// gather is a plain strided read.
//
// 8 simdgroups as 4 (M) x 2 (N), each holding a 32x32 accumulator block: a 128x64 tile
// per threadgroup.
kernel void conv1d_tap_gemm_f32(
    device const float *x    [[buffer(0)]],
    device const float *w    [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float       *dst  [[buffer(3)]],
    constant uint      &len  [[buffer(4)]],
    constant uint      &cin  [[buffer(5)]],
    constant uint      &cout [[buffer(6)]],
    constant uint      &kk   [[buffer(7)]],
    constant uint      &dil  [[buffer(8)]],
    constant uint      &pad  [[buffer(9)]],
    constant uint      &usebias [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint NT = 64;
    threadgroup float Bs[8][NT];
    threadgroup float Tail[8][8][8];

    const uint tid = tid3.x;
    const uint K = kk * cin;
    const uint n_tile = tgid.x * NT;
    const uint m_base = tgid.y * 128 + (sg >> 1) * 32;
    const uint n_base = n_tile + (sg & 1) * 32;

    // The bias enters as the accumulator's starting value: a stride of 0 with the
    // transpose flag makes every column of the fragment the same [8] slice of `bias`.
    // The alternative is a scratch tile per simdgroup, and that much threadgroup memory
    // costs more occupancy than the epilogue is worth.
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; ++i) {
        simdgroup_float8x8 seed = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        if (usebias) { simdgroup_load(seed, bias + m_base + i * 8, 0, ulong2(0, 0), true); }
        for (uint j = 0; j < 4; ++j) { acc[i][j] = seed; }
    }

    for (uint k0 = 0; k0 < K; k0 += 8) {
        const uint t = k0 / cin;
        const uint c0 = k0 - t * cin;
        const int shift = int(t * dil) - int(pad);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < 8 * NT; idx += 256) {
            const uint r = idx >> 6;
            const uint n = idx & 63;
            const int sp = int(n_tile + n) + shift;
            Bs[r][n] = (sp >= 0 && sp < int(len)) ? x[(c0 + r) * len + uint(sp)] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 A[4], B[4];
        for (uint i = 0; i < 4; ++i) { simdgroup_load(A[i], w + (m_base + i * 8) * K + k0, K); }
        for (uint j = 0; j < 4; ++j) { simdgroup_load(B[j], &Bs[0][(sg & 1) * 32 + j * 8], NT); }
        for (uint i = 0; i < 4; ++i) {
            for (uint j = 0; j < 4; ++j) {
                simdgroup_multiply_accumulate(acc[i][j], A[i], B[j], acc[i][j]);
            }
        }
    }

    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            const uint n0 = n_base + j * 8;
            if (n0 + 8 <= len) {
                simdgroup_store(acc[i][j], dst + (m_base + i * 8) * len + n0, len);
            } else if (n0 < len) {
                // The last tile of a length that is not a multiple of 8, through scratch
                // so the store stays inside the row.
                simdgroup_store(acc[i][j], &Tail[sg][0][0], 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = lane; e < 64; e += 32) {
                    const uint r = e >> 3;
                    const uint col = e & 7;
                    if (n0 + col < len) {
                        dst[(m_base + i * 8 + r) * len + n0 + col] = Tail[sg][r][col];
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

// The same convolution with a register tile instead of simdgroup matrices.
//
// M3/M4 have no matrix unit in the GPU: `simdgroup_multiply_accumulate` is a scheduled
// ALU sequence, and the first version of this kernel reached 0.73 TFLOP/s against MPS's
// 3.0 on the same GEMM. This is the classical shape instead — 128x128 tile, 8x8 outputs
// per thread, A staged transposed so each thread's eight rows are contiguous.
kernel void conv1d_tap_reg_f32(
    device const float *x    [[buffer(0)]],
    device const float *w    [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float       *dst  [[buffer(3)]],
    constant uint      &len  [[buffer(4)]],
    constant uint      &cin  [[buffer(5)]],
    constant uint      &cout [[buffer(6)]],
    constant uint      &kk   [[buffer(7)]],
    constant uint      &dil  [[buffer(8)]],
    constant uint      &pad  [[buffer(9)]],
    constant uint      &usebias [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    const uint BM = 128, BN = 128, BK = 8;
    threadgroup float As[BK][BM];
    threadgroup float Bs[BK][BN];

    const uint tid = tid3.x;
    const uint K = kk * cin;
    const uint n_tile = tgid.x * BN;
    const uint m_tile = tgid.y * BM;
    const uint ty = tid >> 4;          // 16 x 16 threads, 8 x 8 outputs each
    const uint tx = tid & 15;
    const uint m0 = m_tile + ty * 8;
    const uint n0 = n_tile + tx * 8;

    float acc[8][8];
    for (uint i = 0; i < 8; ++i) {
        const float b = usebias ? bias[m0 + i] : 0.0f;
        for (uint j = 0; j < 8; ++j) { acc[i][j] = b; }
    }

    // Staging assignments, fixed for the whole loop: A by (m, k) with k fast so each
    // group of eight lanes reads one 32-byte run of a weight row; B by (k, n) with n
    // fast, which is the axis `x` is contiguous in.
    const uint am = tid >> 3, ak = tid & 7;
    const uint bk = tid >> 7, bn = tid & 127;

    for (uint k0 = 0; k0 < K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < 4; ++r) {
            const uint m = am + r * 32;
            As[ak][m] = w[(m_tile + m) * K + k0 + ak];
        }
        for (uint r = 0; r < 4; ++r) {
            const uint kr = bk + r * 2;
            const uint row = k0 + kr;
            const uint t = row / cin;
            const int sp = int(n_tile + bn) + int(t * dil) - int(pad);
            Bs[kr][bn] = (sp >= 0 && sp < int(len)) ? x[(row - t * cin) * len + uint(sp)] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kb = 0; kb < BK; ++kb) {
            float a[8], b[8];
            for (uint i = 0; i < 8; ++i) { a[i] = As[kb][ty * 8 + i]; }
            for (uint j = 0; j < 8; ++j) { b[j] = Bs[kb][tx * 8 + j]; }
            for (uint i = 0; i < 8; ++i) {
                for (uint j = 0; j < 8; ++j) { acc[i][j] = fma(a[i], b[j], acc[i][j]); }
            }
        }
    }

    for (uint i = 0; i < 8; ++i) {
        device float *row = dst + (m0 + i) * len;
        for (uint j = 0; j < 8; ++j) {
            if (n0 + j < len) { row[n0 + j] = acc[i][j]; }
        }
    }
}

// ---- padded-length variants ------------------------------------------------------
//
// A signal padded to a length bucket, so MPSGraph meets a length it has already specialised
// for, stays exact if every convolution reads zeros past the true length: these write them.

// adain_snake_f32 with its parameters unpacked — moments, the style's (gamma, beta) and the
// learned (alpha, 1/beta) are three buffers, so nothing is stacked per call — and zeros past
// `valid`.
kernel void adain_snake_masked_f32(
    device const float *src   [[buffer(0)]],
    device const float *mom   [[buffer(1)]],
    device const float *gb    [[buffer(2)]],
    device const float *ab    [[buffer(3)]],
    device float       *dst   [[buffer(4)]],
    constant uint      &len   [[buffer(5)]],
    constant uint      &chn   [[buffer(6)]],
    constant uint      &valid [[buffer(7)]],
    constant float     &eps   [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint l = gid.x;
    if (l >= len) { return; }
    const uint c = gid.y;
    const uint i = c * len + l;
    if (l >= valid) { dst[i] = 0.0f; return; }
    const float y = (src[i] - mom[c]) * rsqrt(mom[chn + c] + eps) * (gb[c] + 1.0f) + gb[chn + c];
    const float sn = sin(ab[c] * y);
    dst[i] = y + ab[chn + c] * sn * sn;
}

// max(x, slope * x), and zeros past `valid`.
kernel void leaky_masked_f32(
    device const float *src   [[buffer(0)]],
    device float       *dst   [[buffer(1)]],
    constant uint      &len   [[buffer(2)]],
    constant uint      &valid [[buffer(3)]],
    constant float     &slope [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint l = gid.x;
    if (l >= len) { return; }
    const uint i = gid.y * len + l;
    const float v = src[i];
    dst[i] = l < valid ? (v > 0.0f ? v : v * slope) : 0.0f;
}

// ---- LSTM gates ----------------------------------------------------------------
//
// The whole of an LSTM step except its two matmuls. candle spends eleven dispatches per
// timestep on this — four sigmoids and a tanh, each narrowed out of the gate vector, then
// the cell update — and a prosody LSTM runs ~3000 timesteps, so the dispatches cost more
// than the arithmetic by a wide margin.
//
// `gates` is the recurrent term `h @ w_hh`, `pre` the row of the input projection with
// both biases already folded in. Output is `[2, hidden]`: the new h, then the new c.
kernel void lstm_gates_f32(
    device const float *gates [[buffer(0)]],
    device const float *pre   [[buffer(1)]],
    device const float *c_in  [[buffer(2)]],
    device float       *dst   [[buffer(3)]],
    constant uint      &hidden [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= hidden) { return; }
    const uint h = hidden;
    // torch's gate order is input, forget, cell, output.
    const float gi = gates[gid] + pre[gid];
    const float gf = gates[h + gid] + pre[h + gid];
    const float gg = gates[2u * h + gid] + pre[2u * h + gid];
    const float go = gates[3u * h + gid] + pre[3u * h + gid];
    // `precise::`, not the default. Metal compiles with fast math on, and a recurrent
    // net multiplies its own rounding: with the fast `exp` and `tanh` this kernel matched
    // the composed form to 1e-7 for one step and to 4e-1 after sixty, which moved the
    // predicted durations and changed the length of the audio.
    const float it = 1.0f / (1.0f + metal::precise::exp(-gi));
    const float ft = 1.0f / (1.0f + metal::precise::exp(-gf));
    const float ot = 1.0f / (1.0f + metal::precise::exp(-go));
    const float ct = ft * c_in[gid] + it * metal::precise::tanh(gg);
    dst[h + gid] = ct;
    dst[gid] = ot * metal::precise::tanh(ct);
}

// ---- whole-sequence LSTM ---------------------------------------------------------
//
// One timestep of both directions per dispatch, all dispatches in one encoder: the dispatch
// boundary is the grid-wide barrier the recurrence needs, where the step form spent a gemv,
// a gates kernel and copies per direction. One simdgroup per hidden unit: each lane sums
// H/32 terms of all four gate rows, so its loads are few, coalesced and independent — a
// thread per row looping over H was latency-bound at ~33 us a step.
//
// `w` is torch's own `[4H, H]`. h is read back from the previous step's output row and c
// lives in `cell`.
kernel void lstm_step_f32(
    device const float *pre_f [[buffer(0)]],
    device const float *pre_b [[buffer(1)]],
    device const float *w_f   [[buffer(2)]],
    device const float *w_b   [[buffer(3)]],
    device float       *dst   [[buffer(4)]],
    device float       *cell  [[buffer(5)]],
    constant uint      &steps [[buffer(6)]],
    constant uint      &hidden [[buffer(7)]],
    constant uint      &at    [[buffer(8)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sgs  [[simdgroups_per_threadgroup]])
{
    const uint H = hidden, dir = tg.y, u = tg.x * sgs + sg;
    if (u >= H) { return; }
    const uint t = dir == 0 ? at : steps - 1u - at;
    device const float *pre = (dir == 0 ? pre_f : pre_b) + t * 4u * H;
    device const float *w = dir == 0 ? w_f : w_b;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (at != 0) {
        device const float *hp = dst + (dir == 0 ? t - 1u : t + 1u) * 2u * H + dir * H;
        for (uint k = lane; k < H; k += 32u) {
            const float hk = hp[k];
            for (uint q = 0; q < 4u; ++q) { acc[q] = fma(hk, w[(q * H + u) * H + k], acc[q]); }
        }
    }
    float gate[4];
    for (uint q = 0; q < 4u; ++q) { gate[q] = simd_sum(acc[q]) + pre[q * H + u]; }
    if (lane == 0) {
        // precise: see lstm_gates_f32.
        const float it = 1.0f / (1.0f + metal::precise::exp(-gate[0]));
        const float ft = 1.0f / (1.0f + metal::precise::exp(-gate[1]));
        const float ot = 1.0f / (1.0f + metal::precise::exp(-gate[3]));
        const float c_prev = at == 0 ? 0.0f : cell[dir * H + u];
        const float ct = ft * c_prev + it * metal::precise::tanh(gate[2]);
        cell[dir * H + u] = ct;
        dst[t * 2u * H + dir * H + u] = ot * metal::precise::tanh(ct);
    }
}

// ---- decode attention ----------------------------------------------------------
//
// Both read the KV cache in place, indexing with `capacity` as the row stride, which is what
// candle cannot do: `narrow(2, 0, span)` of the cache is non-contiguous, so it copies the span
// twice per layer per step.

// scores[bh, g, p] = dot(q[bh, g, :], k[bh, p, :]), masked outside [window_start, span).
kernel void decode_scores_f32(
    device const float *q     [[buffer(0)]],
    device const float *k     [[buffer(1)]],
    device float       *dst   [[buffer(2)]],
    constant uint      &span  [[buffer(3)]],
    constant uint      &cap   [[buffer(4)]],
    constant uint      &hd    [[buffer(5)]],
    constant uint      &gqa   [[buffer(6)]],
    constant uint      &wstart [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint p = gid.x;
    if (p >= span) { return; }
    const uint g  = gid.y;
    const uint bh = gid.z;

    const uint o = (bh * gqa + g) * span + p;
    if (p < wstart) { dst[o] = -INFINITY; return; }

    const device float *qr = q + (bh * gqa + g) * hd;
    const device float *kr = k + (bh * cap + p) * hd;
    float acc = 0.0f;
    for (uint d = 0; d < hd; ++d) { acc += qr[d] * kr[d]; }
    dst[o] = acc;
}

// out[bh, g, d] = sum_p probs[bh, g, p] * v[bh, p, d].
//
// `d` is the grid's fast axis, so consecutive threads read consecutive `v` — coalesced.
kernel void decode_weighted_f32(
    device const float *probs [[buffer(0)]],
    device const float *v     [[buffer(1)]],
    device float       *dst   [[buffer(2)]],
    constant uint      &span  [[buffer(3)]],
    constant uint      &cap   [[buffer(4)]],
    constant uint      &hd    [[buffer(5)]],
    constant uint      &gqa   [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    if (d >= hd) { return; }
    const uint g  = gid.y;
    const uint bh = gid.z;

    const device float *pr = probs + (bh * gqa + g) * span;
    const device float *vc = v + bh * cap * hd + d;
    float acc = 0.0f;
    for (uint p = 0; p < span; ++p) { acc += pr[p] * vc[p * hd]; }
    dst[(bh * gqa + g) * hd + d] = acc;
}

// Fused decode attention, f16 cache, head_dim 128, gqa <= 2: one threadgroup per (lane, kv
// head), so each K and V row is read once for both query heads. A simdgroup covers a row with
// one half4 per thread — 256 coalesced bytes — and keeps an online softmax over its share of
// positions; the simdgroups merge at the end. The two-kernel form read each row once per query
// head, one thread per row or per output dimension, and ran at ~17 GB/s.
constant uint FA_SG = 4;

kernel void decode_attn_f16(
    device const float *q      [[buffer(0)]],
    device const half  *k      [[buffer(1)]],
    device const half  *v      [[buffer(2)]],
    device float       *dst    [[buffer(3)]],
    constant uint      &span   [[buffer(4)]],
    constant uint      &cap    [[buffer(5)]],
    constant uint      &gqa    [[buffer(6)]],
    constant uint      &wstart [[buffer(7)]],
    uint bh   [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float part_m[FA_SG][2];
    threadgroup float part_l[FA_SG][2];
    threadgroup float4 part_o[FA_SG][2][32];

    const device float4 *q4 = (const device float4 *)(q + bh * gqa * 128);
    const float4 q0 = q4[lane];
    const float4 q1 = gqa > 1 ? q4[32 + lane] : float4(0.0f);

    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    float4 o0 = float4(0.0f), o1 = float4(0.0f);

    const device half4 *k4 = (const device half4 *)(k + bh * cap * 128);
    const device half4 *v4 = (const device half4 *)(v + bh * cap * 128);
    for (uint p = wstart + sg; p < span; p += FA_SG) {
        const float4 kr = float4(k4[p * 32 + lane]);
        const float4 vr = float4(v4[p * 32 + lane]);
        const float s0 = simd_sum(dot(q0, kr));
        const float n0 = max(m0, s0);
        const float c0 = precise::exp(m0 - n0), e0 = precise::exp(s0 - n0);
        l0 = l0 * c0 + e0;
        o0 = o0 * c0 + e0 * vr;
        m0 = n0;
        if (gqa > 1) {
            const float s1 = simd_sum(dot(q1, kr));
            const float n1 = max(m1, s1);
            const float c1 = precise::exp(m1 - n1), e1 = precise::exp(s1 - n1);
            l1 = l1 * c1 + e1;
            o1 = o1 * c1 + e1 * vr;
            m1 = n1;
        }
    }

    if (lane == 0) {
        part_m[sg][0] = m0; part_l[sg][0] = l0;
        part_m[sg][1] = m1; part_l[sg][1] = l1;
    }
    part_o[sg][0][lane] = o0;
    part_o[sg][1][lane] = o1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Simdgroup g merges query head g; an empty share has m = -inf and weighs nothing.
    if (sg < gqa) {
        float mx = -INFINITY;
        for (uint i = 0; i < FA_SG; ++i) { mx = max(mx, part_m[i][sg]); }
        float l = 0.0f;
        float4 o = float4(0.0f);
        for (uint i = 0; i < FA_SG; ++i) {
            const float w = part_m[i][sg] == -INFINITY ? 0.0f : precise::exp(part_m[i][sg] - mx);
            l += part_l[i][sg] * w;
            o += part_o[i][sg][lane] * w;
        }
        ((device float4 *)(dst + (bh * gqa + sg) * 128))[lane] = o / l;
    }
}

// QK-norm, half-split rope and the cache write in one pass, head_dim 128. One threadgroup per
// (lane, position); a simdgroup per head, query heads first. A lane holds dims `l`, `l+32` and
// their rope partners `l+64`, `l+96`. Eight dispatches per layer became this one.
//
// `qkv` is `[b, t, heads*128 + 2*n_kv*128]`; `norms` is `[2, 128]`, q's weight then k's;
// `rope` is `[positions, 64]` cos then the same for sin at `rope + rope_len * 64`.
kernel void qk_rope_f32(
    device const float *qkv    [[buffer(0)]],
    device const float *norms  [[buffer(1)]],
    device const float *cosb   [[buffer(2)]],
    device const float *sinb   [[buffer(3)]],
    device float       *q_out  [[buffer(4)]],
    device half        *kc     [[buffer(5)]],
    device half        *vc     [[buffer(6)]],
    device float       *kf     [[buffer(7)]],
    device float       *vf     [[buffer(8)]],
    constant uint      &t      [[buffer(9)]],
    constant uint      &heads  [[buffer(10)]],
    constant uint      &n_kv   [[buffer(11)]],
    constant uint      &start  [[buffer(12)]],
    constant uint      &cap    [[buffer(13)]],
    constant uint      &has_f  [[buffer(14)]],
    constant float     &eps    [[buffer(15)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint hh   [[simdgroup_index_in_threadgroup]])
{
    const uint bi = row / t, ti = row % t, pos = start + ti;
    const uint width = (heads + 2 * n_kv) * 128;
    device const float *x = qkv + row * width;
    const bool is_q = hh < heads;
    const uint kh = hh - heads;
    device const float *src = x + hh * 128;
    device const float *w = norms + (is_q ? 0 : 128);

    float y[4];
    float ss = 0.0f;
    for (uint j = 0; j < 4; ++j) { y[j] = src[lane + 32 * j]; ss += y[j] * y[j]; }
    const float r = rsqrt(simd_sum(ss) / 128.0f + eps);
    for (uint j = 0; j < 4; ++j) { y[j] = y[j] * r * w[lane + 32 * j]; }

    float o[4];
    for (uint j = 0; j < 2; ++j) {
        const float c = cosb[pos * 64 + lane + 32 * j];
        const float s = sinb[pos * 64 + lane + 32 * j];
        o[j] = y[j] * c - y[j + 2] * s;
        o[j + 2] = y[j + 2] * c + y[j] * s;
    }

    if (is_q) {
        device float *dq = q_out + ((bi * heads + hh) * t + ti) * 128;
        for (uint j = 0; j < 4; ++j) { dq[lane + 32 * j] = o[j]; }
        return;
    }
    device const float *vs = x + (heads + n_kv + kh) * 128;
    device half *dk = kc + ((bi * n_kv + kh) * cap + pos) * 128;
    device half *dv = vc + ((bi * n_kv + kh) * cap + pos) * 128;
    for (uint j = 0; j < 4; ++j) {
        const uint d = lane + 32 * j;
        dk[d] = half(o[j]);
        dv[d] = half(vs[d]);
    }
    if (has_f) {
        device float *fk = kf + ((bi * n_kv + kh) * t + ti) * 128;
        device float *fv = vf + ((bi * n_kv + kh) * t + ti) * 128;
        for (uint j = 0; j < 4; ++j) {
            const uint d = lane + 32 * j;
            fk[d] = o[j];
            fv[d] = vs[d];
        }
    }
}

// f16 cache variants: half the bytes on the hot read, accumulated in float either way.
kernel void decode_scores_f16(
    device const float *q     [[buffer(0)]],
    device const half  *k     [[buffer(1)]],
    device float       *dst   [[buffer(2)]],
    constant uint      &span  [[buffer(3)]],
    constant uint      &cap   [[buffer(4)]],
    constant uint      &hd    [[buffer(5)]],
    constant uint      &gqa   [[buffer(6)]],
    constant uint      &wstart [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint p = gid.x;
    if (p >= span) { return; }
    const uint g  = gid.y;
    const uint bh = gid.z;

    const uint o = (bh * gqa + g) * span + p;
    if (p < wstart) { dst[o] = -INFINITY; return; }

    const device float *qr = q + (bh * gqa + g) * hd;
    const device half  *kr = k + (bh * cap + p) * hd;
    float acc = 0.0f;
    for (uint d = 0; d < hd; ++d) { acc += qr[d] * (float)kr[d]; }
    dst[o] = acc;
}

kernel void decode_weighted_f16(
    device const float *probs [[buffer(0)]],
    device const half  *v     [[buffer(1)]],
    device float       *dst   [[buffer(2)]],
    constant uint      &span  [[buffer(3)]],
    constant uint      &cap   [[buffer(4)]],
    constant uint      &hd    [[buffer(5)]],
    constant uint      &gqa   [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    if (d >= hd) { return; }
    const uint g  = gid.y;
    const uint bh = gid.z;

    const device float *pr = probs + (bh * gqa + g) * span;
    const device half  *vc = v + bh * cap * hd + d;
    float acc = 0.0f;
    for (uint p = 0; p < span; ++p) { acc += pr[p] * (float)vc[p * hd]; }
    dst[(bh * gqa + g) * hd + d] = acc;
}

// Channels-last SnakeBeta: C is the grid's fast axis, so the parameter lookup is a broadcast
// within a threadgroup and both the read and the write coalesce.
kernel void snake_beta_nlc_f32(
    device const float *src    [[buffer(0)]],
    device const float *alpha  [[buffer(1)]],
    device const float *brecip [[buffer(2)]],
    device float       *dst    [[buffer(3)]],
    constant uint      &chan   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    if (c >= chan) { return; }
    const uint i = gid.y * chan + c;
    const float x = src[i];
    const float s = sin(alpha[c] * x);
    dst[i] = x + brecip[c] * s * s;
}

// f16 activations, f32 parameters and math: the sin and the square want the wider type, the
// tensor crossing memory does not.
kernel void snake_beta_nlc_f16(
    device const half  *src    [[buffer(0)]],
    device const float *alpha  [[buffer(1)]],
    device const float *brecip [[buffer(2)]],
    device half        *dst    [[buffer(3)]],
    constant uint      &chan   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    if (c >= chan) { return; }
    const uint i = gid.y * chan + c;
    const float x = (float)src[i];
    const float s = sin(alpha[c] * x);
    dst[i] = (half)(x + brecip[c] * s * s);
}

// ---- channels-last causal conv -------------------------------------------------
//
// y[l, co] = bias[co] + sum_t sum_ci x[l + t*dil - pad, ci] * w[t*cin + ci, co]
//
// The conv-as-GEMM routes both lose to *building* the matrix, not to multiplying it:
// `convgemm` measured 85.8 ms to assemble the `[672, 131072]` im2col against 8.2 ms for the
// GEMM that consumes it. So the tap gather belongs inside the GEMM, where the tile it needs
// is already in threadgroup memory and never crosses device memory at all.
//
// One 32x32 output tile per threadgroup, reduced 32 deep. `cin` is a multiple of 32 at every
// stage of the codec, so a reduction chunk never straddles two taps and the tap index is one
// division per chunk rather than one per element — the same trap `im2col_tap_major` documents.
kernel void nlc_conv_f32(
    device const float *x     [[buffer(0)]],
    device const float *w     [[buffer(1)]],
    device const float *bias  [[buffer(2)]],
    device float       *y     [[buffer(3)]],
    constant uint      &len   [[buffer(4)]],
    constant uint      &cin   [[buffer(5)]],
    constant uint      &cout  [[buffer(6)]],
    constant uint      &k     [[buffer(7)]],
    constant uint      &dil   [[buffer(8)]],
    constant uint      &has_b [[buffer(9)]],
    device const float *alpha [[buffer(10)]],
    device const float *brecip [[buffer(11)]],
    device const float *res   [[buffer(12)]],
    constant uint      &has_snake [[buffer(13)]],
    constant uint      &has_res [[buffer(14)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]],
    uint  sg  [[simdgroup_index_in_threadgroup]])
{
    // Optionally SnakeBeta on the input as it loads and a residual added on the way out, so a
    // residual unit's second activation and its add are not passes of their own. A padded tap
    // stays zero, since snake(0) = 0.
    //
    // 64 positions x 32 output channels per threadgroup, on the matrix units.
    //
    // **The channel tile is the grid's fast axis.** A threadgroup re-reads its slice of `x`
    // once per channel tile, so the gather's device traffic is `L * k * cin * (cout / 32)` —
    // tens of GB per chunk. Widening the tile to 96 to divide that by three needs 22 KB of
    // threadgroup memory and measured **2313 ms** against 1098, because one resident
    // threadgroup per core hides no latency. Issuing the channel tiles for one position range
    // back to back instead leaves them to the cache, and costs nothing.
    //
    // Two register-tiled versions came first and both lost to the `cat`-then-MPS route they
    // replace: 2x2 per thread measured 1283 ms against 1204 on a 300-frame chunk, 4x4 measured
    // 1168. The multiply has to run on the same hardware MPS uses.
    threadgroup float as_[64 * 40];
    threadgroup float bs_[32 * 32];

    const uint n0 = tgp.x * 32;
    const uint l0 = tgp.y * 64;
    const int pad = int((k - 1) * dil);
    const uint depth = k * cin;

    // Four simdgroups: two down the positions, two across the channels, so each owns
    // 32 positions x 16 channels — 4x2 accumulators of 8x8.
    const uint sm = (sg >> 1) * 32;
    const uint sn = (sg & 1) * 16;

    simdgroup_float8x8 acc[4][2];
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 2; ++j) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (uint kc = 0; kc < depth; kc += 32) {
        const uint t = kc / cin;
        const uint ci0 = kc - t * cin;
        const int shift = int(t * dil) - pad;

        // `kk` on the fast axis so both the device read and the threadgroup write coalesce.
        // Rows are padded to 40 rather than 32 to keep `simdgroup_load`'s strided reads off a
        // single bank.
        // Four channels at a time. `cin` and `cout` are multiples of 32 and `kc` of 32, so
        // every one of these addresses is 16-byte aligned, and the loop that fills a 32-deep
        // tile drops from 24 iterations to 6.
        threadgroup float4 *as4 = (threadgroup float4 *)as_;
        threadgroup float4 *bs4 = (threadgroup float4 *)bs_;
        for (uint idx = tid; idx < 64 * 8; idx += 128) {
            const uint m = idx >> 3;
            const uint q = idx & 7;
            const int s = int(l0 + m) + shift;
            float4 v = (s >= 0 && s < int(len))
                ? *(const device float4 *)(x + uint(s) * cin + ci0 + q * 4)
                : float4(0.0f);
            if (has_snake) {
                const float4 a = *(const device float4 *)(alpha + ci0 + q * 4);
                const float4 br = *(const device float4 *)(brecip + ci0 + q * 4);
                const float4 sv = sin(a * v);
                v = v + br * sv * sv;
            }
            as4[m * 10 + q] = v;
        }
        for (uint idx = tid; idx < 32 * 8; idx += 128) {
            const uint kk = idx >> 3;
            bs4[kk * 8 + (idx & 7)] =
                *(const device float4 *)(w + (kc + kk) * cout + n0 + (idx & 7) * 4);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 a[4], b[2];
            for (uint i = 0; i < 4; ++i) {
                simdgroup_load(a[i], as_ + (sm + i * 8) * 40 + kk, 40);
            }
            for (uint j = 0; j < 2; ++j) {
                simdgroup_load(b[j], bs_ + kk * 32 + sn + j * 8, 32);
            }
            for (uint i = 0; i < 4; ++i) {
                for (uint j = 0; j < 2; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Out through `as_`: `simdgroup_store` wants a fixed row stride, and the trailing tile has
    // to be masked against `len` before it reaches the output.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 2; ++j) {
            simdgroup_store(acc[i][j], as_ + (sm + i * 8) * 40 + sn + j * 8, 40);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < 64 * 8; idx += 128) {
        const uint m = idx >> 3;
        const uint q = idx & 7;
        const uint l = l0 + m;
        if (l >= len) { continue; }
        float4 v = ((threadgroup float4 *)as_)[m * 10 + q];
        if (has_b) { v += *(const device float4 *)(bias + n0 + q * 4); }
        if (has_res) { v += *(const device float4 *)(res + l * cout + n0 + q * 4); }
        *(device float4 *)(y + l * cout + n0 + q * 4) = v;
    }
}

// ---- skinny GEMM ---------------------------------------------------------------
//
// `[m, k] x [k, n] -> [m, n]`, f16 in, f32 accumulated and out, for the shapes an
// autoregressive decode step actually has: `m` is the lane count, at most 48, where candle
// reaches 1.1-2.35 TFLOP/s against the 3.64 it manages on a 2048-cube.
//
// What mattered, measured at m = 48 on an M4: splitting `k` so the narrow projections fill
// the GPU (the predictor's down projection launched 16 threadgroups), and giving each
// simdgroup a 48x16 strip loaded straight from device memory — no threadgroup staging, no
// barriers. Tried and slower: double-buffered staging, a 48x128 tile, 64-deep chunks, and any
// layout past 12 accumulators per simdgroup, which spills (0.4 TFLOP/s at 24).
kernel void gemm_skinny_f16(
    device const half  *a   [[buffer(0)]],
    device const half  *b   [[buffer(1)]],
    device float       *c   [[buffer(2)]],
    constant uint      &m   [[buffer(3)]],
    constant uint      &k   [[buffer(4)]],
    constant uint      &n   [[buffer(5)]],
    constant uint      &kper [[buffer(6)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint  sg  [[simdgroup_index_in_threadgroup]])
{
    // Split-K: grid row `tgp.y` reduces `[tgp.y * kper, +kper)` into its own `[m, n]` slab,
    // summed in split order by `sum_splits_f32`. One split is the whole reduction, in place.
    c += tgp.y * m * n;
    const uint kbeg = tgp.y * kper;
    const uint kend = kbeg + kper;

    // 48 rows because 48 is the batch: a 64-row tile wasted a quarter of the matrix work on
    // padding, and a runtime row count lost the unrolling (0.55-0.80x). Every B element belongs
    // to one simdgroup, so staging it bought only barriers; A is small enough to stay in cache.
    // `m` is a multiple of 8 (see `skinny::eligible`); rows past it read row 0, never stored.
    // Each simdgroup owns a 24x32 patch (rows `sm`, columns `n0`): 7 tile loads per 12 MMAs.
    const uint sm = (sg & 1) * 24;
    const uint n0 = tgp.x * 64 + (sg >> 1) * 32;

    simdgroup_float8x8 acc[3][4];
    for (uint i = 0; i < 3; ++i) {
        for (uint j = 0; j < 4; ++j) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    uint rows[3];
    for (uint i = 0; i < 3; ++i) { rows[i] = (sm + i * 8 < m) ? sm + i * 8 : 0; }

    for (uint kc = kbeg; kc < kend; kc += 8) {
        simdgroup_half8x8 av[3], bv[4];
        for (uint i = 0; i < 3; ++i) {
            simdgroup_load(av[i], a + rows[i] * k + kc, k);
        }
        for (uint j = 0; j < 4; ++j) {
            simdgroup_load(bv[j], b + kc * n + n0 + j * 8, n);
        }
        for (uint i = 0; i < 3; ++i) {
            for (uint j = 0; j < 4; ++j) {
                simdgroup_multiply_accumulate(acc[i][j], av[i], bv[j], acc[i][j]);
            }
        }
    }

    for (uint i = 0; i < 3; ++i) {
        if (sm + i * 8 >= m) { break; }
        for (uint j = 0; j < 4; ++j) {
            simdgroup_store(acc[i][j], c + (sm + i * 8) * n + n0 + j * 8, n);
        }
    }
}

// dst[i] = src[i] + src[count + i] + ..., in split order so a render stays reproducible.
kernel void sum_splits_f32(
    device const float4 *src    [[buffer(0)]],
    device float4       *dst    [[buffer(1)]],
    constant uint       &count4 [[buffer(2)]],
    constant uint       &splits [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= count4) { return; }
    float4 acc = src[i];
    for (uint s = 1; s < splits; ++s) {
        acc += src[s * count4 + i];
    }
    dst[i] = acc;
}

// ---- swiglu tail ---------------------------------------------------------------
//
// out = silu(g) * u, elementwise. candle spends two dispatches and two full round trips on
// this. That is nothing on a long sequence, but a batch-1 decode step is dispatch-bound: the
// qwen3 predictor runs it 70 times per audio frame, where the cost is the launch and not
// the 3072 lanes of arithmetic.
kernel void swiglu_mul_f32(
    device const float *g   [[buffer(0)]],
    device const float *u   [[buffer(1)]],
    device float       *dst [[buffer(2)]],
    constant uint      &n   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    const float x = g[gid];
    dst[gid] = (x / (1.0f + exp(-x))) * u[gid];
}

// ---- short-time Fourier pair ----------------------------------------------------
//
// torch.stft / torch.istft with a periodic Hann window, reflect padding and a small
// hop, as one dispatch each. The CPU pair costs tens of milliseconds per utterance
// in trig calls alone — tens of millions of per-tap sin/cos/atan2 — while the
// arithmetic is a 20-point DFT begging for tables. The tables hold the pure DFT
// exponentials; windowing, the conjugate-pair weight and the 1/n stay runtime
// multiplies in the reference's order, so the only numeric distance is
// libm-vs-Metal trig (~1 ulp each).
//
// Forward layout matches the CPU code: mag rows then phase rows in one
// `[2 * bins, frames]` buffer.
kernel void stft_forward_f32(
    device const float *x   [[buffer(0)]],
    device const float *tre [[buffer(1)]],
    device const float *tim [[buffer(2)]],
    device const float *win [[buffer(3)]],
    device float       *dst [[buffer(4)]],
    constant uint      &len    [[buffer(5)]],
    constant uint      &frames [[buffer(6)]],
    constant uint      &n_fft  [[buffer(7)]],
    constant uint      &hop    [[buffer(8)]],
    constant uint      &pad    [[buffer(9)]],
    constant uint      &bins   [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint f = gid.x;
    if (f >= frames) { return; }
    const uint k = gid.y;
    if (k >= bins) { return; }

    float re = 0.0f;
    float im = 0.0f;
    const uint base = f * hop;
    const uint trow = k * n_fft;
    for (uint nn = 0; nn < n_fft; ++nn) {
        const uint j = base + nn;
        float s;
        if (j < pad) {
            s = x[pad - j];
        } else if (j < pad + len) {
            s = x[j - pad];
        } else {
            s = x[2u * len + pad - 2u - j];
        }
        const float v = s * win[nn];
        re += v * tre[trow + nn];
        im += v * tim[trow + nn];
    }
    dst[k * frames + f] = sqrt(re * re + im * im);
    dst[(bins + k) * frames + f] = atan2(im, re);
}

// Overlap-add back to a waveform: out[t] for t in 0..(frames-1)*hop, one thread
// each, gathering its covering frames. No races, no envelope array — the
// envelope is signal-independent, so each thread rebuilds its own few terms.
// `tc`/`ts` fold cos/sin(2πkn/n), the conjugate-pair weight and 1/n_fft.
kernel void stft_inverse_f32(
    device const float *mag   [[buffer(0)]],
    device const float *phase [[buffer(1)]],
    device const float *tc    [[buffer(2)]],
    device const float *ts    [[buffer(3)]],
    device const float *win   [[buffer(4)]],
    device float       *dst   [[buffer(5)]],
    constant uint      &out_len [[buffer(6)]],
    constant uint      &frames  [[buffer(7)]],
    constant uint      &bins    [[buffer(8)]],
    constant uint      &n_fft   [[buffer(9)]],
    constant uint      &hop     [[buffer(10)]],
    constant uint      &pad     [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    const uint t = gid;
    if (t >= out_len) { return; }
    const uint g = pad + t;
    // Covering frames: f*hop <= g < f*hop + n_fft.
    uint f_lo = (g < n_fft) ? 0u : (g - n_fft + hop) / hop;
    uint f_hi = g / hop;
    if (f_hi >= frames) { f_hi = frames - 1u; }
    float acc = 0.0f;
    float env = 0.0f;
    for (uint f = f_lo; f <= f_hi; ++f) {
        const uint nn = g - f * hop;
        float v = 0.0f;
        for (uint k = 0; k < bins; ++k) {
            const float m = mag[k * frames + f];
            const float p = phase[k * frames + f];
            const uint ti = k * n_fft + nn;
            float cp;
            const float sp = sincos(p, cp);
            v += m * (cp * tc[ti] - sp * ts[ti]);
        }
        const float w = win[nn];
        acc += v * w;
        env += w * w;
    }
    dst[t] = (env > 1e-11f) ? acc / env : 0.0f;
}
// ---- top-k sampling ------------------------------------------------------------
//
// One threadgroup per row. The top `k` are selected by value, ties to the lower index, which
// is the host sampler's order; thread 0 then repeats its arithmetic exactly — sequential sums,
// a division by the temperature, `precise::exp` — so a pick differs from the host's only when
// the draw lands within an ulp of a boundary.
constant uint TOPK_MAX_N = 4096;
constant uint TOPK_MAX_K = 64;

inline bool topk_better(float va, uint ia, float vb, uint ib) {
    return va > vb || (va == vb && ia < ib);
}

kernel void topk_sample_f32(
    device const float* logits [[buffer(0)]],
    device const float* draws  [[buffer(1)]],
    device uint*        out    [[buffer(2)]],
    constant uint&      n      [[buffer(3)]],
    constant uint&      k      [[buffer(4)]],
    constant float&     temp   [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float v[TOPK_MAX_N];
    threadgroup float sg_val[32];
    threadgroup uint  sg_idx[32];
    threadgroup float sel_val[TOPK_MAX_K];
    threadgroup uint  sel_idx[TOPK_MAX_K];

    device const float* x = logits + row * n;
    for (uint i = tid; i < n; i += tpg) {
        v[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint groups = (tpg + 31) / 32;
    for (uint j = 0; j < k; ++j) {
        float bv = -INFINITY;
        uint bi = 0xffffffffu;
        for (uint i = tid; i < n; i += tpg) {
            // A taken slot is NaN, which no comparison selects.
            if (topk_better(v[i], i, bv, bi)) { bv = v[i]; bi = i; }
        }
        for (uint off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_down(bv, off);
            const uint oi = simd_shuffle_down(bi, off);
            if (lane + off < 32 && topk_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { sg_val[sg] = bv; sg_idx[sg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float wv = sg_val[0];
            uint wi = sg_idx[0];
            for (uint g = 1; g < groups; ++g) {
                if (topk_better(sg_val[g], sg_idx[g], wv, wi)) { wv = sg_val[g]; wi = sg_idx[g]; }
            }
            sel_val[j] = wv;
            sel_idx[j] = wi;
            v[wi] = NAN;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float p[TOPK_MAX_K];
        const float mx = sel_val[0];
        float total = 0.0f;
        for (uint j = 0; j < k; ++j) {
            p[j] = precise::exp((sel_val[j] - mx) / temp);
            total += p[j];
        }
        float renorm = 0.0f;
        for (uint j = 0; j < k; ++j) {
            p[j] /= total;
            renorm += p[j];
        }
        float u = draws[row] * renorm;
        uint pick = sel_idx[k - 1];
        for (uint j = 0; j < k; ++j) {
            u -= p[j];
            if (u <= 0.0f) { pick = sel_idx[j]; break; }
        }
        out[row] = pick;
    }
}
"#;

/// Compile the library once per device and hand out cached pipelines by function name.
#[cfg(feature = "metal")]
pub(crate) fn pipeline(
    device: &candle_core::MetalDevice,
    name: &'static str,
) -> Result<candle_metal_kernels::metal::ComputePipeline> {
    use candle_core::metal_backend::DeviceId;
    use candle_metal_kernels::metal::{ComputePipeline, Library};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    #[allow(clippy::type_complexity)]
    static CACHE: OnceLock<
        Mutex<(
            HashMap<DeviceId, Library>,
            HashMap<(DeviceId, &'static str), ComputePipeline>,
        )>,
    > = OnceLock::new();

    let cache = CACHE.get_or_init(|| Mutex::new((HashMap::new(), HashMap::new())));
    let mut guard = cache
        .lock()
        .map_err(|e| candle_core::Error::Metal(format!("kernel cache poisoned: {e}").into()))?;
    let id = device.id();

    if let Some(p) = guard.1.get(&(id, name)) {
        return Ok(p.clone());
    }
    let (libs, pipelines) = &mut *guard;
    let lib = match libs.get(&id) {
        Some(l) => l.clone(),
        None => {
            let l = device
                .metal_device()
                .new_library_with_source(SHADER, None)
                .map_err(candle_core::Error::wrap)?;
            libs.insert(id, l.clone());
            l
        }
    };
    let func = lib
        .get_function(name, None)
        .map_err(candle_core::Error::wrap)?;
    let p = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&func)
        .map_err(candle_core::Error::wrap)?;
    pipelines.insert((id, name), p.clone());
    Ok(p)
}

/// A threadgroup width for a 1-D fast axis, capped by what the pipeline allows.
#[cfg(feature = "metal")]
pub(crate) fn group_width(
    p: &candle_metal_kernels::metal::ComputePipeline,
    fast_axis: usize,
) -> usize {
    // Two independent caps: what the pipeline permits, and how wide the axis actually
    // is. 256 is the practical ceiling — wider threadgroups do not help these kernels.
    let permitted = p.max_total_threads_per_threadgroup().clamp(1, 256);
    permitted.min(fast_axis.max(1))
}
