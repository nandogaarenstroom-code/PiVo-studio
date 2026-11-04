//! Runtime CPU multiversioning for PiVo's f32 kernels.
//! Portable distribution builds MUST NOT use target-cpu=native or global +avx flags.
//! The portable backend retains the original arithmetic. Other backends can change
//! reduction order and use FMA, so model-level equivalence tests are required.
//! AVX-512 is explicit opt-in until representative end-to-end benchmarks justify it.
use std::sync::OnceLock;
const EPS: f32 = 1e-6;

#[derive(Clone, Copy)]
struct Kernels {
    name: &'static str,
    dot: unsafe fn(&[f32], &[f32]) -> f32,
    matvec_into: unsafe fn(&[f32], &[f32], usize, usize, &mut [f32]) ,
    rms_norm_into: unsafe fn(&[f32], &[f32], &mut [f32]) ,
    rms_norm_block: unsafe fn(&mut [f32], &[f32]) ,
    apply_rope: unsafe fn(&mut [f32], usize, usize, usize, &[f32], &[f32]) ,
    conv1d: unsafe fn(&[f32], usize, usize, usize, usize, usize, usize, &[f32], Option<&[f32]>, &mut [f32]) ,
    conv1d_depthwise: unsafe fn(&[f32], usize, usize, usize, usize, usize, &[f32], &[f32], &mut [f32]) ,
    silu_in_place: unsafe fn(&mut [f32]) ,
    softmax_row: unsafe fn(&mut [f32]) ,
    clip_in_place: unsafe fn(&mut [f32], f32, f32) ,
}
static SELECTED: OnceLock<Kernels> = OnceLock::new();
fn table_portable() -> Kernels { Kernels { name: "portable", dot: portable::dot, matvec_into: portable::matvec_into, rms_norm_into: portable::rms_norm_into, rms_norm_block: portable::rms_norm_block, apply_rope: portable::apply_rope, conv1d: portable::conv1d, conv1d_depthwise: portable::conv1d_depthwise, silu_in_place: portable::silu_in_place, softmax_row: portable::softmax_row, clip_in_place: portable::clip_in_place } }
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn table_sse2() -> Kernels { Kernels { name: "sse2", dot: sse2::dot, matvec_into: sse2::matvec_into, rms_norm_into: sse2::rms_norm_into, rms_norm_block: sse2::rms_norm_block, apply_rope: sse2::apply_rope, conv1d: sse2::conv1d, conv1d_depthwise: sse2::conv1d_depthwise, silu_in_place: sse2::silu_in_place, softmax_row: sse2::softmax_row, clip_in_place: sse2::clip_in_place } }
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn table_avx() -> Kernels { Kernels { name: "avx", dot: avx::dot, matvec_into: avx::matvec_into, rms_norm_into: avx::rms_norm_into, rms_norm_block: avx::rms_norm_block, apply_rope: avx::apply_rope, conv1d: avx::conv1d, conv1d_depthwise: avx::conv1d_depthwise, silu_in_place: avx::silu_in_place, softmax_row: avx::softmax_row, clip_in_place: avx::clip_in_place } }
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn table_avx_fma() -> Kernels { Kernels { name: "avx_fma", dot: avx_fma::dot, matvec_into: avx_fma::matvec_into, rms_norm_into: avx_fma::rms_norm_into, rms_norm_block: avx_fma::rms_norm_block, apply_rope: avx_fma::apply_rope, conv1d: avx_fma::conv1d, conv1d_depthwise: avx_fma::conv1d_depthwise, silu_in_place: avx_fma::silu_in_place, softmax_row: avx_fma::softmax_row, clip_in_place: avx_fma::clip_in_place } }
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn table_avx2_fma() -> Kernels { Kernels { name: "avx2_fma", dot: avx2_fma::dot, matvec_into: avx2_fma::matvec_into, rms_norm_into: avx2_fma::rms_norm_into, rms_norm_block: avx2_fma::rms_norm_block, apply_rope: avx2_fma::apply_rope, conv1d: avx2_fma::conv1d, conv1d_depthwise: avx2_fma::conv1d_depthwise, silu_in_place: avx2_fma::silu_in_place, softmax_row: avx2_fma::softmax_row, clip_in_place: avx2_fma::clip_in_place } }
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn table_avx512() -> Kernels { Kernels { name: "avx512", dot: avx512::dot, matvec_into: avx512::matvec_into, rms_norm_into: avx512::rms_norm_into, rms_norm_block: avx512::rms_norm_block, apply_rope: avx512::apply_rope, conv1d: avx512::conv1d, conv1d_depthwise: avx512::conv1d_depthwise, silu_in_place: avx512::silu_in_place, softmax_row: avx512::softmax_row, clip_in_place: avx512::clip_in_place } }
#[cfg(target_arch = "aarch64")]
fn table_neon() -> Kernels { Kernels { name: "neon", dot: neon::dot, matvec_into: neon::matvec_into, rms_norm_into: neon::rms_norm_into, rms_norm_block: neon::rms_norm_block, apply_rope: neon::apply_rope, conv1d: neon::conv1d, conv1d_depthwise: neon::conv1d_depthwise, silu_in_place: neon::silu_in_place, softmax_row: neon::softmax_row, clip_in_place: neon::clip_in_place } }

fn available(name: &str) -> Option<Kernels> {
    if name == "portable" { return Some(table_portable()); }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        match name {
            "sse2" if std::is_x86_feature_detected!("sse2") => return Some(table_sse2()),
            "avx" if std::is_x86_feature_detected!("avx") => return Some(table_avx()),
            "avx_fma" if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") => return Some(table_avx_fma()),
            "avx2_fma" if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") => return Some(table_avx2_fma()),
            "avx512" if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") => return Some(table_avx512()),
            _ => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    if name == "neon" && std::arch::is_aarch64_feature_detected!("neon") { return Some(table_neon()); }
    None
}
fn select() -> Kernels {
    let requested = std::env::var("PIVO_CPU_BACKEND").unwrap_or_else(|_| "auto".into());
    if requested != "auto" {
        if let Some(k) = available(&requested) { return k; }
        eprintln!("[pivo] unsupported CPU backend {:?}; using portable", requested);
        return table_portable();
    }
    // Conservative initial policy, NOT a claim of measured optimality.
    // AVX-512 is available by override, but is not blindly preferred.
    for name in ["avx2_fma", "avx_fma", "avx", "sse2", "neon"] {
        if let Some(k) = available(name) { return k; }
    }
    table_portable()
}
fn kernels() -> &'static Kernels { SELECTED.get_or_init(select) }
/// Initialize dispatch once and return the selected implementation name.
/// Set PIVO_CPU_BACKEND before the first kernel call; changing it later has no effect.
pub fn backend_name() -> &'static str { kernels().name }

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    if a.len() < 16 { return portable::dot(a, b); }
    // SAFETY: the immutable table is selected only after runtime feature checks.
    unsafe { (kernels().dot)(a, b) }
}

pub(crate) fn matvec_into(x: &[f32], w: &[f32], out_dim: usize, in_dim: usize, y: &mut [f32])  {
    assert_eq!(x.len(), in_dim);
    assert_eq!(y.len(), out_dim);
    assert_eq!(Some(w.len()), out_dim.checked_mul(in_dim));
    if in_dim < 16 { return portable::matvec_into(x, w, out_dim, in_dim, y); }
    // SAFETY: the immutable table is selected only after runtime feature checks.
    unsafe { (kernels().matvec_into)(x, w, out_dim, in_dim, y) }
}

pub(crate) fn rms_norm_into(x: &[f32], weight: &[f32], out: &mut [f32])  {
    assert!(!x.is_empty()); assert_eq!(x.len(), weight.len()); assert_eq!(x.len(), out.len());
    // SAFETY: the immutable table is selected only after runtime feature checks.
    unsafe { (kernels().rms_norm_into)(x, weight, out) }
}

pub(crate) fn rms_norm_block(x: &mut [f32], weight: &[f32])  {
    assert!(!weight.is_empty()); assert_eq!(x.len() % weight.len(), 0);
    // SAFETY: the immutable table is selected only after runtime feature checks.
    unsafe { (kernels().rms_norm_block)(x, weight) }
}

pub(crate) fn apply_rope(x: &mut [f32], heads: usize, seq: usize, head_dim: usize, cos: &[f32], sin: &[f32])  {
    
    // SAFETY: the immutable table is selected only after runtime feature checks.
    unsafe { (kernels().apply_rope)(x, heads, seq, head_dim, cos, sin) }
}

pub(crate) fn conv1d(
    x: &[f32],
    t_len: usize,
    c_in: usize,
    c_out: usize,
    k: usize,
    dilation: usize,
    pad: usize,
    w: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
)  {
    
    // SAFETY: the immutable table is selected only after runtime feature checks.
    unsafe { (kernels().conv1d)(x, t_len, c_in, c_out, k, dilation, pad, w, bias, out) }
}

pub(crate) fn conv1d_depthwise(
    x: &[f32],
    t_len: usize,
    channels: usize,
    k: usize,
    dilation: usize,
    pad: usize,
    w: &[f32],
    bias: &[f32],
    out: &mut [f32],
)  {
    
    // SAFETY: the immutable table is selected only after runtime feature checks.
    unsafe { (kernels().conv1d_depthwise)(x, t_len, channels, k, dilation, pad, w, bias, out) }
}

pub(crate) fn silu_in_place(x: &mut [f32])  {
    
    // SAFETY: the immutable table is selected only after runtime feature checks.
    unsafe { (kernels().silu_in_place)(x) }
}

pub(crate) fn softmax_row(row: &mut [f32])  {
    
    // SAFETY: the immutable table is selected only after runtime feature checks.
    unsafe { (kernels().softmax_row)(row) }
}

pub(crate) fn clip_in_place(x: &mut [f32], lo: f32, hi: f32)  {
    
    // SAFETY: the immutable table is selected only after runtime feature checks.
    unsafe { (kernels().clip_in_place)(x, lo, hi) }
}
mod portable {
use super::EPS;
#[inline]
fn sigmoid(v: f32) -> f32 { 1.0 / (1.0 + (-v).exp()) }
pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    const N: usize = 8;
    let mut acc = [0f32; N];
    let mut i = 0;
    while i + N <= a.len() {
        for k in 0..N {
            acc[k] += a[i + k] * b[i + k];
        }
        i += N;
    }
    let mut sum = 0f32;
    for k in 0..N {
        sum += acc[k];
    }
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}
pub(super) fn matvec_into(x: &[f32], w: &[f32], out_dim: usize, in_dim: usize, y: &mut [f32]) {
    debug_assert_eq!(w.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(y.len(), out_dim);
    for (row, yr) in y.iter_mut().enumerate().take(out_dim) {
        let wrow = &w[row * in_dim..(row + 1) * in_dim];
        *yr = dot(wrow, x);
    }
}
pub(super) fn rms_norm_into(x: &[f32], weight: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    let mut ss = 0f32;
    for &v in x {
        ss += v * v;
    }
    let inv = 1.0f32 / (ss / x.len() as f32 + EPS).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * weight[i];
    }
}
pub(super) fn rms_norm_block(x: &mut [f32], weight: &[f32]) {
    let dim = weight.len();
    debug_assert_eq!(x.len() % dim, 0);
    for row in x.chunks_exact_mut(dim) {
        let mut ss = 0f32;
        for &v in row.iter() {
            ss += v * v;
        }
        let inv = 1.0f32 / (ss / dim as f32 + EPS).sqrt();
        for (v, &w) in row.iter_mut().zip(weight.iter()) {
            *v = *v * inv * w;
        }
    }
}
pub(super) fn apply_rope(x: &mut [f32], heads: usize, seq: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    debug_assert_eq!(x.len(), seq * heads * head_dim);
    debug_assert_eq!(cos.len(), seq * half);
    for t in 0..seq {
        let c = &cos[t * half..(t + 1) * half];
        let s = &sin[t * half..(t + 1) * half];
        for h in 0..heads {
            let row = &mut x[(t * heads + h) * head_dim..(t * heads + h) * head_dim + head_dim];
            for i in 0..half {
                let x1 = row[i];
                let x2 = row[i + half];
                row[i] = x1 * c[i] - x2 * s[i];
                row[i + half] = x2 * c[i] + x1 * s[i];
            }
        }
    }
}
pub(super) fn conv1d(
    x: &[f32],
    t_len: usize,
    c_in: usize,
    c_out: usize,
    k: usize,
    dilation: usize,
    pad: usize,
    w: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), t_len * c_in);
    debug_assert_eq!(w.len(), c_out * c_in * k);
    let t_out = (t_len + 2 * pad).saturating_sub((k - 1) * dilation);
    debug_assert_eq!(out.len(), t_out * c_out);
    for t in 0..t_out {
        let orow = &mut out[t * c_out..(t + 1) * c_out];
        match bias {
            Some(b) => orow.copy_from_slice(b),
            None => {
                for v in orow.iter_mut() {
                    *v = 0.0;
                }
            }
        }
        for co in 0..c_out {
            // ONNX Conv weight layout: [C_out, C_in, K]
            let w_co = &w[co * c_in * k..(co + 1) * c_in * k];
            let mut acc = orow[co];
            for kk in 0..k {
                let src = t as isize + kk as isize * dilation as isize - pad as isize;
                if src < 0 || src >= t_len as isize {
                    continue;
                }
                let xrow = &x[src as usize * c_in..src as usize * c_in + c_in];
                for ci in 0..c_in {
                    acc += w_co[ci * k + kk] * xrow[ci];
                }
            }
            orow[co] = acc;
        }
    }
}
pub(super) fn conv1d_depthwise(
    x: &[f32],
    t_len: usize,
    channels: usize,
    k: usize,
    dilation: usize,
    pad: usize,
    w: &[f32],
    bias: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), t_len * channels);
    debug_assert_eq!(w.len(), channels * k);
    let t_out = (t_len + 2 * pad).saturating_sub((k - 1) * dilation);
    debug_assert_eq!(out.len(), t_out * channels);
    for t in 0..t_out {
        let orow = &mut out[t * channels..(t + 1) * channels];
        for c in 0..channels {
            let mut acc = bias[c];
            let wc = &w[c * k..(c + 1) * k];
            for kk in 0..k {
                let src = t as isize + kk as isize * dilation as isize - pad as isize;
                if src < 0 || src >= t_len as isize {
                    continue;
                }
                acc += wc[kk] * x[src as usize * channels + c];
            }
            orow[c] = acc;
        }
    }
}
pub(super) fn silu_in_place(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v * sigmoid(*v);
    }
}
pub(super) fn softmax_row(row: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in row.iter() {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0f32;
    for v in row.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}
pub(super) fn clip_in_place(x: &mut [f32], lo: f32, hi: f32) {
    for v in x.iter_mut() {
        *v = v.clamp(lo, hi);
    }
}
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse2 {
use super::EPS;
#[cfg(target_arch = "x86")] use std::arch::x86::*;
#[cfg(target_arch = "x86_64")] use std::arch::x86_64::*;
#[inline]
fn sigmoid(v: f32) -> f32 { 1.0 / (1.0 + (-v).exp()) }
#[target_feature(enable = "sse2")]
pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: feature support is checked by dispatch; full vector loads are in bounds.
    unsafe {
        let mut acc0 = _mm_setzero_ps();
        let mut acc1 = _mm_setzero_ps();
        let mut acc2 = _mm_setzero_ps();
        let mut acc3 = _mm_setzero_ps();
        let mut i = 0;
        while a.len() - i >= 16 {
            acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(a.as_ptr().add(i + 0)), _mm_loadu_ps(b.as_ptr().add(i + 0))));
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(a.as_ptr().add(i + 4)), _mm_loadu_ps(b.as_ptr().add(i + 4))));
            acc2 = _mm_add_ps(acc2, _mm_mul_ps(_mm_loadu_ps(a.as_ptr().add(i + 8)), _mm_loadu_ps(b.as_ptr().add(i + 8))));
            acc3 = _mm_add_ps(acc3, _mm_mul_ps(_mm_loadu_ps(a.as_ptr().add(i + 12)), _mm_loadu_ps(b.as_ptr().add(i + 12))));
            i += 16;
        }
        while a.len() - i >= 4 {
            acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(a.as_ptr().add(i)), _mm_loadu_ps(b.as_ptr().add(i))));
            i += 4;
        }
        let mut lanes = [0.0f32; 4];
        let mut sum = 0.0f32;
        _mm_storeu_ps(lanes.as_mut_ptr(), acc0);
        for v in lanes { sum += v; }
        _mm_storeu_ps(lanes.as_mut_ptr(), acc1);
        for v in lanes { sum += v; }
        _mm_storeu_ps(lanes.as_mut_ptr(), acc2);
        for v in lanes { sum += v; }
        _mm_storeu_ps(lanes.as_mut_ptr(), acc3);
        for v in lanes { sum += v; }
        while i < a.len() { sum += a[i] * b[i]; i += 1; }
        sum
    }
}

#[target_feature(enable = "sse2")]
pub(super) unsafe fn matvec_into(x: &[f32], w: &[f32], out_dim: usize, in_dim: usize, y: &mut [f32]) {
    assert_eq!(x.len(), in_dim);
    assert_eq!(y.len(), out_dim);
    assert_eq!(Some(w.len()), out_dim.checked_mul(in_dim));
    if in_dim == 0 { y.fill(0.0); return; }
    // SAFETY: shape checks bound every load; dispatch checks the ISA.
    unsafe {
        let mut row = 0;
        while out_dim - row >= 4 {
            let mut a0_0 = _mm_setzero_ps();
            let mut a0_1 = _mm_setzero_ps();
            let mut a1_0 = _mm_setzero_ps();
            let mut a1_1 = _mm_setzero_ps();
            let mut a2_0 = _mm_setzero_ps();
            let mut a2_1 = _mm_setzero_ps();
            let mut a3_0 = _mm_setzero_ps();
            let mut a3_1 = _mm_setzero_ps();
            let mut i = 0;
            while in_dim - i >= 8 {
                let x0 = _mm_loadu_ps(x.as_ptr().add(i + 0));
                let x1 = _mm_loadu_ps(x.as_ptr().add(i + 4));
                a0_0 = _mm_add_ps(a0_0, _mm_mul_ps(x0, _mm_loadu_ps(w.as_ptr().add((row + 0) * in_dim + i + 0))));
                a0_1 = _mm_add_ps(a0_1, _mm_mul_ps(x1, _mm_loadu_ps(w.as_ptr().add((row + 0) * in_dim + i + 4))));
                a1_0 = _mm_add_ps(a1_0, _mm_mul_ps(x0, _mm_loadu_ps(w.as_ptr().add((row + 1) * in_dim + i + 0))));
                a1_1 = _mm_add_ps(a1_1, _mm_mul_ps(x1, _mm_loadu_ps(w.as_ptr().add((row + 1) * in_dim + i + 4))));
                a2_0 = _mm_add_ps(a2_0, _mm_mul_ps(x0, _mm_loadu_ps(w.as_ptr().add((row + 2) * in_dim + i + 0))));
                a2_1 = _mm_add_ps(a2_1, _mm_mul_ps(x1, _mm_loadu_ps(w.as_ptr().add((row + 2) * in_dim + i + 4))));
                a3_0 = _mm_add_ps(a3_0, _mm_mul_ps(x0, _mm_loadu_ps(w.as_ptr().add((row + 3) * in_dim + i + 0))));
                a3_1 = _mm_add_ps(a3_1, _mm_mul_ps(x1, _mm_loadu_ps(w.as_ptr().add((row + 3) * in_dim + i + 4))));
                i += 8;
            }
            while in_dim - i >= 4 {
                let xv = _mm_loadu_ps(x.as_ptr().add(i));
                a0_0 = _mm_add_ps(a0_0, _mm_mul_ps(xv, _mm_loadu_ps(w.as_ptr().add((row + 0) * in_dim + i))));
                a1_0 = _mm_add_ps(a1_0, _mm_mul_ps(xv, _mm_loadu_ps(w.as_ptr().add((row + 1) * in_dim + i))));
                a2_0 = _mm_add_ps(a2_0, _mm_mul_ps(xv, _mm_loadu_ps(w.as_ptr().add((row + 2) * in_dim + i))));
                a3_0 = _mm_add_ps(a3_0, _mm_mul_ps(xv, _mm_loadu_ps(w.as_ptr().add((row + 3) * in_dim + i))));
                i += 4;
            }
            let mut lanes = [0.0f32; 4];
            let mut sum = 0.0f32;
            _mm_storeu_ps(lanes.as_mut_ptr(), a0_0);
            for v in lanes { sum += v; }
            _mm_storeu_ps(lanes.as_mut_ptr(), a0_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 0) * in_dim + k]; }
            y[row + 0] = sum;
            let mut sum = 0.0f32;
            _mm_storeu_ps(lanes.as_mut_ptr(), a1_0);
            for v in lanes { sum += v; }
            _mm_storeu_ps(lanes.as_mut_ptr(), a1_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 1) * in_dim + k]; }
            y[row + 1] = sum;
            let mut sum = 0.0f32;
            _mm_storeu_ps(lanes.as_mut_ptr(), a2_0);
            for v in lanes { sum += v; }
            _mm_storeu_ps(lanes.as_mut_ptr(), a2_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 2) * in_dim + k]; }
            y[row + 2] = sum;
            let mut sum = 0.0f32;
            _mm_storeu_ps(lanes.as_mut_ptr(), a3_0);
            for v in lanes { sum += v; }
            _mm_storeu_ps(lanes.as_mut_ptr(), a3_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 3) * in_dim + k]; }
            y[row + 3] = sum;
            row += 4;
        }
        for r in row..out_dim { y[r] = dot(&w[r * in_dim..(r + 1) * in_dim], x); }
    }
}
#[target_feature(enable = "sse2")]
pub(super) unsafe fn rms_norm_into(x: &[f32], weight: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    let ss = unsafe { dot(x, x) };
    let inv = 1.0f32 / (ss / x.len() as f32 + EPS).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * weight[i];
    }
}
#[target_feature(enable = "sse2")]
pub(super) unsafe fn rms_norm_block(x: &mut [f32], weight: &[f32]) {
    let dim = weight.len();
    debug_assert_eq!(x.len() % dim, 0);
    for row in x.chunks_exact_mut(dim) {
        let ss = unsafe { dot(row, row) };
        let inv = 1.0f32 / (ss / dim as f32 + EPS).sqrt();
        for (v, &w) in row.iter_mut().zip(weight.iter()) {
            *v = *v * inv * w;
        }
    }
}
#[target_feature(enable = "sse2")]
pub(super) unsafe fn apply_rope(x: &mut [f32], heads: usize, seq: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    debug_assert_eq!(x.len(), seq * heads * head_dim);
    debug_assert_eq!(cos.len(), seq * half);
    for t in 0..seq {
        let c = &cos[t * half..(t + 1) * half];
        let s = &sin[t * half..(t + 1) * half];
        for h in 0..heads {
            let row = &mut x[(t * heads + h) * head_dim..(t * heads + h) * head_dim + head_dim];
            for i in 0..half {
                let x1 = row[i];
                let x2 = row[i + half];
                row[i] = x1 * c[i] - x2 * s[i];
                row[i + half] = x2 * c[i] + x1 * s[i];
            }
        }
    }
}
#[target_feature(enable = "sse2")]
pub(super) unsafe fn conv1d(
    x: &[f32],
    t_len: usize,
    c_in: usize,
    c_out: usize,
    k: usize,
    dilation: usize,
    pad: usize,
    w: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), t_len * c_in);
    debug_assert_eq!(w.len(), c_out * c_in * k);
    let t_out = (t_len + 2 * pad).saturating_sub((k - 1) * dilation);
    debug_assert_eq!(out.len(), t_out * c_out);
    for t in 0..t_out {
        let orow = &mut out[t * c_out..(t + 1) * c_out];
        match bias {
            Some(b) => orow.copy_from_slice(b),
            None => {
                for v in orow.iter_mut() {
                    *v = 0.0;
                }
            }
        }
        for co in 0..c_out {
            // ONNX Conv weight layout: [C_out, C_in, K]
            let w_co = &w[co * c_in * k..(co + 1) * c_in * k];
            let mut acc = orow[co];
            for kk in 0..k {
                let src = t as isize + kk as isize * dilation as isize - pad as isize;
                if src < 0 || src >= t_len as isize {
                    continue;
                }
                let xrow = &x[src as usize * c_in..src as usize * c_in + c_in];
                for ci in 0..c_in {
                    acc += w_co[ci * k + kk] * xrow[ci];
                }
            }
            orow[co] = acc;
        }
    }
}
#[target_feature(enable = "sse2")]
pub(super) unsafe fn conv1d_depthwise(
    x: &[f32],
    t_len: usize,
    channels: usize,
    k: usize,
    dilation: usize,
    pad: usize,
    w: &[f32],
    bias: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), t_len * channels);
    debug_assert_eq!(w.len(), channels * k);
    let t_out = (t_len + 2 * pad).saturating_sub((k - 1) * dilation);
    debug_assert_eq!(out.len(), t_out * channels);
    for t in 0..t_out {
        let orow = &mut out[t * channels..(t + 1) * channels];
        for c in 0..channels {
            let mut acc = bias[c];
            let wc = &w[c * k..(c + 1) * k];
            for kk in 0..k {
                let src = t as isize + kk as isize * dilation as isize - pad as isize;
                if src < 0 || src >= t_len as isize {
                    continue;
                }
                acc += wc[kk] * x[src as usize * channels + c];
            }
            orow[c] = acc;
        }
    }
}
#[target_feature(enable = "sse2")]
pub(super) unsafe fn silu_in_place(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v * sigmoid(*v);
    }
}
#[target_feature(enable = "sse2")]
pub(super) unsafe fn softmax_row(row: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in row.iter() {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0f32;
    for v in row.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}
#[target_feature(enable = "sse2")]
pub(super) unsafe fn clip_in_place(x: &mut [f32], lo: f32, hi: f32) {
    for v in x.iter_mut() {
        *v = v.clamp(lo, hi);
    }
}
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx {
use super::EPS;
#[cfg(target_arch = "x86")] use std::arch::x86::*;
#[cfg(target_arch = "x86_64")] use std::arch::x86_64::*;
#[inline]
fn sigmoid(v: f32) -> f32 { 1.0 / (1.0 + (-v).exp()) }
#[target_feature(enable = "avx")]
pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: feature support is checked by dispatch; full vector loads are in bounds.
    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut i = 0;
        while a.len() - i >= 32 {
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(a.as_ptr().add(i + 0)), _mm256_loadu_ps(b.as_ptr().add(i + 0))));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_loadu_ps(a.as_ptr().add(i + 8)), _mm256_loadu_ps(b.as_ptr().add(i + 8))));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_loadu_ps(a.as_ptr().add(i + 16)), _mm256_loadu_ps(b.as_ptr().add(i + 16))));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_loadu_ps(a.as_ptr().add(i + 24)), _mm256_loadu_ps(b.as_ptr().add(i + 24))));
            i += 32;
        }
        while a.len() - i >= 8 {
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i))));
            i += 8;
        }
        let mut lanes = [0.0f32; 8];
        let mut sum = 0.0f32;
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc0);
        for v in lanes { sum += v; }
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc1);
        for v in lanes { sum += v; }
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc2);
        for v in lanes { sum += v; }
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc3);
        for v in lanes { sum += v; }
        while i < a.len() { sum += a[i] * b[i]; i += 1; }
        sum
    }
}

#[target_feature(enable = "avx")]
pub(super) unsafe fn matvec_into(x: &[f32], w: &[f32], out_dim: usize, in_dim: usize, y: &mut [f32]) {
    assert_eq!(x.len(), in_dim);
    assert_eq!(y.len(), out_dim);
    assert_eq!(Some(w.len()), out_dim.checked_mul(in_dim));
    if in_dim == 0 { y.fill(0.0); return; }
    // SAFETY: shape checks bound every load; dispatch checks the ISA.
    unsafe {
        let mut row = 0;
        while out_dim - row >= 4 {
            let mut a0_0 = _mm256_setzero_ps();
            let mut a0_1 = _mm256_setzero_ps();
            let mut a1_0 = _mm256_setzero_ps();
            let mut a1_1 = _mm256_setzero_ps();
            let mut a2_0 = _mm256_setzero_ps();
            let mut a2_1 = _mm256_setzero_ps();
            let mut a3_0 = _mm256_setzero_ps();
            let mut a3_1 = _mm256_setzero_ps();
            let mut i = 0;
            while in_dim - i >= 16 {
                let x0 = _mm256_loadu_ps(x.as_ptr().add(i + 0));
                let x1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
                a0_0 = _mm256_add_ps(a0_0, _mm256_mul_ps(x0, _mm256_loadu_ps(w.as_ptr().add((row + 0) * in_dim + i + 0))));
                a0_1 = _mm256_add_ps(a0_1, _mm256_mul_ps(x1, _mm256_loadu_ps(w.as_ptr().add((row + 0) * in_dim + i + 8))));
                a1_0 = _mm256_add_ps(a1_0, _mm256_mul_ps(x0, _mm256_loadu_ps(w.as_ptr().add((row + 1) * in_dim + i + 0))));
                a1_1 = _mm256_add_ps(a1_1, _mm256_mul_ps(x1, _mm256_loadu_ps(w.as_ptr().add((row + 1) * in_dim + i + 8))));
                a2_0 = _mm256_add_ps(a2_0, _mm256_mul_ps(x0, _mm256_loadu_ps(w.as_ptr().add((row + 2) * in_dim + i + 0))));
                a2_1 = _mm256_add_ps(a2_1, _mm256_mul_ps(x1, _mm256_loadu_ps(w.as_ptr().add((row + 2) * in_dim + i + 8))));
                a3_0 = _mm256_add_ps(a3_0, _mm256_mul_ps(x0, _mm256_loadu_ps(w.as_ptr().add((row + 3) * in_dim + i + 0))));
                a3_1 = _mm256_add_ps(a3_1, _mm256_mul_ps(x1, _mm256_loadu_ps(w.as_ptr().add((row + 3) * in_dim + i + 8))));
                i += 16;
            }
            while in_dim - i >= 8 {
                let xv = _mm256_loadu_ps(x.as_ptr().add(i));
                a0_0 = _mm256_add_ps(a0_0, _mm256_mul_ps(xv, _mm256_loadu_ps(w.as_ptr().add((row + 0) * in_dim + i))));
                a1_0 = _mm256_add_ps(a1_0, _mm256_mul_ps(xv, _mm256_loadu_ps(w.as_ptr().add((row + 1) * in_dim + i))));
                a2_0 = _mm256_add_ps(a2_0, _mm256_mul_ps(xv, _mm256_loadu_ps(w.as_ptr().add((row + 2) * in_dim + i))));
                a3_0 = _mm256_add_ps(a3_0, _mm256_mul_ps(xv, _mm256_loadu_ps(w.as_ptr().add((row + 3) * in_dim + i))));
                i += 8;
            }
            let mut lanes = [0.0f32; 8];
            let mut sum = 0.0f32;
            _mm256_storeu_ps(lanes.as_mut_ptr(), a0_0);
            for v in lanes { sum += v; }
            _mm256_storeu_ps(lanes.as_mut_ptr(), a0_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 0) * in_dim + k]; }
            y[row + 0] = sum;
            let mut sum = 0.0f32;
            _mm256_storeu_ps(lanes.as_mut_ptr(), a1_0);
            for v in lanes { sum += v; }
            _mm256_storeu_ps(lanes.as_mut_ptr(), a1_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 1) * in_dim + k]; }
            y[row + 1] = sum;
            let mut sum = 0.0f32;
            _mm256_storeu_ps(lanes.as_mut_ptr(), a2_0);
            for v in lanes { sum += v; }
            _mm256_storeu_ps(lanes.as_mut_ptr(), a2_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 2) * in_dim + k]; }
            y[row + 2] = sum;
            let mut sum = 0.0f32;
            _mm256_storeu_ps(lanes.as_mut_ptr(), a3_0);
            for v in lanes { sum += v; }
            _mm256_storeu_ps(lanes.as_mut_ptr(), a3_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 3) * in_dim + k]; }
            y[row + 3] = sum;
            row += 4;
        }
        for r in row..out_dim { y[r] = dot(&w[r * in_dim..(r + 1) * in_dim], x); }
    }
}
#[target_feature(enable = "avx")]
pub(super) unsafe fn rms_norm_into(x: &[f32], weight: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    let ss = unsafe { dot(x, x) };
    let inv = 1.0f32 / (ss / x.len() as f32 + EPS).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * weight[i];
    }
}
#[target_feature(enable = "avx")]
pub(super) unsafe fn rms_norm_block(x: &mut [f32], weight: &[f32]) {
    let dim = weight.len();
    debug_assert_eq!(x.len() % dim, 0);
    for row in x.chunks_exact_mut(dim) {
        let ss = unsafe { dot(row, row) };
        let inv = 1.0f32 / (ss / dim as f32 + EPS).sqrt();
        for (v, &w) in row.iter_mut().zip(weight.iter()) {
            *v = *v * inv * w;
        }
    }
}
#[target_feature(enable = "avx")]
pub(super) unsafe fn apply_rope(x: &mut [f32], heads: usize, seq: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    debug_assert_eq!(x.len(), seq * heads * head_dim);
    debug_assert_eq!(cos.len(), seq * half);
    for t in 0..seq {
        let c = &cos[t * half..(t + 1) * half];
        let s = &sin[t * half..(t + 1) * half];
        for h in 0..heads {
            let row = &mut x[(t * heads + h) * head_dim..(t * heads + h) * head_dim + head_dim];
            for i in 0..half {
                let x1 = row[i];
                let x2 = row[i + half];
                row[i] = x1 * c[i] - x2 * s[i];
                row[i + half] = x2 * c[i] + x1 * s[i];
            }
        }
    }
}
#[target_feature(enable = "avx")]
pub(super) unsafe fn conv1d(
    x: &[f32],
    t_len: usize,
    c_in: usize,
    c_out: usize,
    k: usize,
    dilation: usize,
    pad: usize,
    w: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), t_len * c_in);
    debug_assert_eq!(w.len(), c_out * c_in * k);
    let t_out = (t_len + 2 * pad).saturating_sub((k - 1) * dilation);
    debug_assert_eq!(out.len(), t_out * c_out);
    for t in 0..t_out {
        let orow = &mut out[t * c_out..(t + 1) * c_out];
        match bias {
            Some(b) => orow.copy_from_slice(b),
            None => {
                for v in orow.iter_mut() {
                    *v = 0.0;
                }
            }
        }
        for co in 0..c_out {
            // ONNX Conv weight layout: [C_out, C_in, K]
            let w_co = &w[co * c_in * k..(co + 1) * c_in * k];
            let mut acc = orow[co];
            for kk in 0..k {
                let src = t as isize + kk as isize * dilation as isize - pad as isize;
                if src < 0 || src >= t_len as isize {
                    continue;
                }
                let xrow = &x[src as usize * c_in..src as usize * c_in + c_in];
                for ci in 0..c_in {
                    acc += w_co[ci * k + kk] * xrow[ci];
                }
            }
            orow[co] = acc;
        }
    }
}
#[target_feature(enable = "avx")]
pub(super) unsafe fn conv1d_depthwise(
    x: &[f32],
    t_len: usize,
    channels: usize,
    k: usize,
    dilation: usize,
    pad: usize,
    w: &[f32],
    bias: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), t_len * channels);
    debug_assert_eq!(w.len(), channels * k);
    let t_out = (t_len + 2 * pad).saturating_sub((k - 1) * dilation);
    debug_assert_eq!(out.len(), t_out * channels);
    for t in 0..t_out {
        let orow = &mut out[t * channels..(t + 1) * channels];
        for c in 0..channels {
            let mut acc = bias[c];
            let wc = &w[c * k..(c + 1) * k];
            for kk in 0..k {
                let src = t as isize + kk as isize * dilation as isize - pad as isize;
                if src < 0 || src >= t_len as isize {
                    continue;
                }
                acc += wc[kk] * x[src as usize * channels + c];
            }
            orow[c] = acc;
        }
    }
}
#[target_feature(enable = "avx")]
pub(super) unsafe fn silu_in_place(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v * sigmoid(*v);
    }
}
#[target_feature(enable = "avx")]
pub(super) unsafe fn softmax_row(row: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in row.iter() {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0f32;
    for v in row.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}
#[target_feature(enable = "avx")]
pub(super) unsafe fn clip_in_place(x: &mut [f32], lo: f32, hi: f32) {
    for v in x.iter_mut() {
        *v = v.clamp(lo, hi);
    }
}
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx_fma {
use super::EPS;
#[cfg(target_arch = "x86")] use std::arch::x86::*;
#[cfg(target_arch = "x86_64")] use std::arch::x86_64::*;
#[inline]
fn sigmoid(v: f32) -> f32 { 1.0 / (1.0 + (-v).exp()) }
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: feature support is checked by dispatch; full vector loads are in bounds.
    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut i = 0;
        while a.len() - i >= 32 {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 0)), _mm256_loadu_ps(b.as_ptr().add(i + 0)), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 8)), _mm256_loadu_ps(b.as_ptr().add(i + 8)), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 16)), _mm256_loadu_ps(b.as_ptr().add(i + 16)), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 24)), _mm256_loadu_ps(b.as_ptr().add(i + 24)), acc3);
            i += 32;
        }
        while a.len() - i >= 8 {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i)), acc0);
            i += 8;
        }
        let mut lanes = [0.0f32; 8];
        let mut sum = 0.0f32;
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc0);
        for v in lanes { sum += v; }
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc1);
        for v in lanes { sum += v; }
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc2);
        for v in lanes { sum += v; }
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc3);
        for v in lanes { sum += v; }
        while i < a.len() { sum += a[i] * b[i]; i += 1; }
        sum
    }
}

#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn matvec_into(x: &[f32], w: &[f32], out_dim: usize, in_dim: usize, y: &mut [f32]) {
    assert_eq!(x.len(), in_dim);
    assert_eq!(y.len(), out_dim);
    assert_eq!(Some(w.len()), out_dim.checked_mul(in_dim));
    if in_dim == 0 { y.fill(0.0); return; }
    // SAFETY: shape checks bound every load; dispatch checks the ISA.
    unsafe {
        let mut row = 0;
        while out_dim - row >= 4 {
            let mut a0_0 = _mm256_setzero_ps();
            let mut a0_1 = _mm256_setzero_ps();
            let mut a1_0 = _mm256_setzero_ps();
            let mut a1_1 = _mm256_setzero_ps();
            let mut a2_0 = _mm256_setzero_ps();
            let mut a2_1 = _mm256_setzero_ps();
            let mut a3_0 = _mm256_setzero_ps();
            let mut a3_1 = _mm256_setzero_ps();
            let mut i = 0;
            while in_dim - i >= 16 {
                let x0 = _mm256_loadu_ps(x.as_ptr().add(i + 0));
                let x1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
                a0_0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w.as_ptr().add((row + 0) * in_dim + i + 0)), a0_0);
                a0_1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(w.as_ptr().add((row + 0) * in_dim + i + 8)), a0_1);
                a1_0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w.as_ptr().add((row + 1) * in_dim + i + 0)), a1_0);
                a1_1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(w.as_ptr().add((row + 1) * in_dim + i + 8)), a1_1);
                a2_0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w.as_ptr().add((row + 2) * in_dim + i + 0)), a2_0);
                a2_1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(w.as_ptr().add((row + 2) * in_dim + i + 8)), a2_1);
                a3_0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w.as_ptr().add((row + 3) * in_dim + i + 0)), a3_0);
                a3_1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(w.as_ptr().add((row + 3) * in_dim + i + 8)), a3_1);
                i += 16;
            }
            while in_dim - i >= 8 {
                let xv = _mm256_loadu_ps(x.as_ptr().add(i));
                a0_0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w.as_ptr().add((row + 0) * in_dim + i)), a0_0);
                a1_0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w.as_ptr().add((row + 1) * in_dim + i)), a1_0);
                a2_0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w.as_ptr().add((row + 2) * in_dim + i)), a2_0);
                a3_0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w.as_ptr().add((row + 3) * in_dim + i)), a3_0);
                i += 8;
            }
            let mut lanes = [0.0f32; 8];
            let mut sum = 0.0f32;
            _mm256_storeu_ps(lanes.as_mut_ptr(), a0_0);
            for v in lanes { sum += v; }
            _mm256_storeu_ps(lanes.as_mut_ptr(), a0_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 0) * in_dim + k]; }
            y[row + 0] = sum;
            let mut sum = 0.0f32;
            _mm256_storeu_ps(lanes.as_mut_ptr(), a1_0);
            for v in lanes { sum += v; }
            _mm256_storeu_ps(lanes.as_mut_ptr(), a1_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 1) * in_dim + k]; }
            y[row + 1] = sum;
            let mut sum = 0.0f32;
            _mm256_storeu_ps(lanes.as_mut_ptr(), a2_0);
            for v in lanes { sum += v; }
            _mm256_storeu_ps(lanes.as_mut_ptr(), a2_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 2) * in_dim + k]; }
            y[row + 2] = sum;
            let mut sum = 0.0f32;
            _mm256_storeu_ps(lanes.as_mut_ptr(), a3_0);
            for v in lanes { sum += v; }
            _mm256_storeu_ps(lanes.as_mut_ptr(), a3_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 3) * in_dim + k]; }
            y[row + 3] = sum;
            row += 4;
        }
        for r in row..out_dim { y[r] = dot(&w[r * in_dim..(r + 1) * in_dim], x); }
    }
}
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn rms_norm_into(x: &[f32], weight: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    let ss = unsafe { dot(x, x) };
    let inv = 1.0f32 / (ss / x.len() as f32 + EPS).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * weight[i];
    }
}
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn rms_norm_block(x: &mut [f32], weight: &[f32]) {
    let dim = weight.len();
    debug_assert_eq!(x.len() % dim, 0);
    for row in x.chunks_exact_mut(dim) {
        let ss = unsafe { dot(row, row) };
        let inv = 1.0f32 / (ss / dim as f32 + EPS).sqrt();
        for (v, &w) in row.iter_mut().zip(weight.iter()) {
            *v = *v * inv * w;
        }
    }
}
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn apply_rope(x: &mut [f32], heads: usize, seq: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    debug_assert_eq!(x.len(), seq * heads * head_dim);
    debug_assert_eq!(cos.len(), seq * half);
    for t in 0..seq {
        let c = &cos[t * half..(t + 1) * half];
        let s = &sin[t * half..(t + 1) * half];
        for h in 0..heads {
            let row = &mut x[(t * heads + h) * head_dim..(t * heads + h) * head_dim + head_dim];
            for i in 0..half {
                let x1 = row[i];
                let x2 = row[i + half];
                row[i] = x1 * c[i] - x2 * s[i];
                row[i + half] = x2 * c[i] + x1 * s[i];
            }
        }
    }
}
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn conv1d(
    x: &[f32],
    t_len: usize,
    c_in: usize,
    c_out: usize,
    k: usize,
    dilation: usize,
    pad: usize,
    w: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), t_len * c_in);
    debug_assert_eq!(w.len(), c_out * c_in * k);
    let t_out = (t_len + 2 * pad).saturating_sub((k - 1) * dilation);
    debug_assert_eq!(out.len(), t_out * c_out);
    for t in 0..t_out {
        let orow = &mut out[t * c_out..(t + 1) * c_out];
        match bias {
            Some(b) => orow.copy_from_slice(b),
            None => {
                for v in orow.iter_mut() {
                    *v = 0.0;
                }
            }
        }
        for co in 0..c_out {
            // ONNX Conv weight layout: [C_out, C_in, K]
            let w_co = &w[co * c_in * k..(co + 1) * c_in * k];
            let mut acc = orow[co];
            for kk in 0..k {
                let src = t as isize + kk as isize * dilation as isize - pad as isize;
                if src < 0 || src >= t_len as isize {
                    continue;
                }
                let xrow = &x[src as usize * c_in..src as usize * c_in + c_in];
                for ci in 0..c_in {
                    acc += w_co[ci * k + kk] * xrow[ci];
                }
            }
            orow[co] = acc;
        }
    }
}
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn conv1d_depthwise(
    x: &[f32],
    t_len: usize,
    channels: usize,
    k: usize,
    dilation: usize,
    pad: usize,
    w: &[f32],
    bias: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), t_len * channels);
    debug_assert_eq!(w.len(), channels * k);
    let t_out = (t_len + 2 * pad).saturating_sub((k - 1) * dilation);
    debug_assert_eq!(out.len(), t_out * channels);
    for t in 0..t_out {
        let orow = &mut out[t * channels..(t + 1) * channels];
        for c in 0..channels {
            let mut acc = bias[c];
            let wc = &w[c * k..(c + 1) * k];
            for kk in 0..k {
                let src = t as isize + kk as isize * dilation as isize - pad as isize;
                if src < 0 || src >= t_len as isize {
                    continue;
                }
                acc += wc[kk] * x[src as usize * channels + c];
            }
            orow[c] = acc;
        }
    }
}
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn silu_in_place(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v * sigmoid(*v);
    }
}
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn softmax_row(row: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in row.iter() {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0f32;
    for v in row.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn clip_in_place(x: &mut [f32], lo: f32, hi: f32) {
    for v in x.iter_mut() {
        *v = v.clamp(lo, hi);
    }
}
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2_fma {
use super::EPS;
#[cfg(target_arch = "x86")] use std::arch::x86::*;
#[cfg(target_arch = "x86_64")] use std::arch::x86_64::*;
#[inline]
fn sigmoid(v: f32) -> f32 { 1.0 / (1.0 + (-v).exp()) }
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: feature support is checked by dispatch; full vector loads are in bounds.
    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut i = 0;
        while a.len() - i >= 32 {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 0)), _mm256_loadu_ps(b.as_ptr().add(i + 0)), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 8)), _mm256_loadu_ps(b.as_ptr().add(i + 8)), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 16)), _mm256_loadu_ps(b.as_ptr().add(i + 16)), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 24)), _mm256_loadu_ps(b.as_ptr().add(i + 24)), acc3);
            i += 32;
        }
        while a.len() - i >= 8 {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i)), acc0);
            i += 8;
        }
        let mut lanes = [0.0f32; 8];
        let mut sum = 0.0f32;
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc0);
        for v in lanes { sum += v; }
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc1);
        for v in lanes { sum += v; }
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc2);
        for v in lanes { sum += v; }
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc3);
        for v in lanes { sum += v; }
        while i < a.len() { sum += a[i] * b[i]; i += 1; }
        sum
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn matvec_into(x: &[f32], w: &[f32], out_dim: usize, in_dim: usize, y: &mut [f32]) {
    assert_eq!(x.len(), in_dim);
    assert_eq!(y.len(), out_dim);
    assert_eq!(Some(w.len()), out_dim.checked_mul(in_dim));
    if in_dim == 0 { y.fill(0.0); return; }
    // SAFETY: checked shapes bound full-vector loads and tails; dispatch checks AVX2/FMA.
    unsafe {
        let mut row = 0usize;
        while out_dim - row >= 4 {
            let mut a0_0 = _mm256_setzero_ps();
            let mut a0_1 = _mm256_setzero_ps();
            let mut a1_0 = _mm256_setzero_ps();
            let mut a1_1 = _mm256_setzero_ps();
            let mut a2_0 = _mm256_setzero_ps();
            let mut a2_1 = _mm256_setzero_ps();
            let mut a3_0 = _mm256_setzero_ps();
            let mut a3_1 = _mm256_setzero_ps();
            let mut i = 0usize;
            while in_dim - i >= 16 {
                let x0 = _mm256_loadu_ps(x.as_ptr().add(i + 0));
                let x1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
                a0_0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w.as_ptr().add((row + 0) * in_dim + i + 0)), a0_0);
                a0_1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(w.as_ptr().add((row + 0) * in_dim + i + 8)), a0_1);
                a1_0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w.as_ptr().add((row + 1) * in_dim + i + 0)), a1_0);
                a1_1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(w.as_ptr().add((row + 1) * in_dim + i + 8)), a1_1);
                a2_0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w.as_ptr().add((row + 2) * in_dim + i + 0)), a2_0);
                a2_1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(w.as_ptr().add((row + 2) * in_dim + i + 8)), a2_1);
                a3_0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w.as_ptr().add((row + 3) * in_dim + i + 0)), a3_0);
                a3_1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(w.as_ptr().add((row + 3) * in_dim + i + 8)), a3_1);
                i += 16;
            }
            while in_dim - i >= 8 {
                let xv = _mm256_loadu_ps(x.as_ptr().add(i));
                a0_0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w.as_ptr().add((row + 0) * in_dim + i)), a0_0);
                a1_0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w.as_ptr().add((row + 1) * in_dim + i)), a1_0);
                a2_0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w.as_ptr().add((row + 2) * in_dim + i)), a2_0);
                a3_0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w.as_ptr().add((row + 3) * in_dim + i)), a3_0);
                i += 8;
            }
            let v = _mm256_add_ps(a0_0, a0_1);
            let half = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
            let pairs = _mm_hadd_ps(half, half);
            let mut sum = _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
            for k in i..in_dim { sum += x[k] * w[(row + 0) * in_dim + k]; }
            y[row + 0] = sum;
            let v = _mm256_add_ps(a1_0, a1_1);
            let half = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
            let pairs = _mm_hadd_ps(half, half);
            let mut sum = _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
            for k in i..in_dim { sum += x[k] * w[(row + 1) * in_dim + k]; }
            y[row + 1] = sum;
            let v = _mm256_add_ps(a2_0, a2_1);
            let half = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
            let pairs = _mm_hadd_ps(half, half);
            let mut sum = _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
            for k in i..in_dim { sum += x[k] * w[(row + 2) * in_dim + k]; }
            y[row + 2] = sum;
            let v = _mm256_add_ps(a3_0, a3_1);
            let half = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
            let pairs = _mm_hadd_ps(half, half);
            let mut sum = _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
            for k in i..in_dim { sum += x[k] * w[(row + 3) * in_dim + k]; }
            y[row + 3] = sum;
            row += 4;
        }
        for r in row..out_dim { y[r] = dot(&w[r * in_dim..(r + 1) * in_dim], x); }
    }
}
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn rms_norm_into(x: &[f32], weight: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    let ss = unsafe { dot(x, x) };
    let inv = 1.0f32 / (ss / x.len() as f32 + EPS).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * weight[i];
    }
}
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn rms_norm_block(x: &mut [f32], weight: &[f32]) {
    let dim = weight.len();
    debug_assert_eq!(x.len() % dim, 0);
    for row in x.chunks_exact_mut(dim) {
        let ss = unsafe { dot(row, row) };
        let inv = 1.0f32 / (ss / dim as f32 + EPS).sqrt();
        for (v, &w) in row.iter_mut().zip(weight.iter()) {
            *v = *v * inv * w;
        }
    }
}
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn apply_rope(x: &mut [f32], heads: usize, seq: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    debug_assert_eq!(x.len(), seq * heads * head_dim);
    debug_assert_eq!(cos.len(), seq * half);
    for t in 0..seq {
        let c = &cos[t * half..(t + 1) * half];
        let s = &sin[t * half..(t + 1) * half];
        for h in 0..heads {
            let row = &mut x[(t * heads + h) * head_dim..(t * heads + h) * head_dim + head_dim];
            for i in 0..half {
                let x1 = row[i];
                let x2 = row[i + half];
                row[i] = x1 * c[i] - x2 * s[i];
                row[i + half] = x2 * c[i] + x1 * s[i];
            }
        }
    }
}
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn conv1d(
    x: &[f32],
    t_len: usize,
    c_in: usize,
    c_out: usize,
    k: usize,
    dilation: usize,
    pad: usize,
    w: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), t_len * c_in);
    debug_assert_eq!(w.len(), c_out * c_in * k);
    let t_out = (t_len + 2 * pad).saturating_sub((k - 1) * dilation);
    debug_assert_eq!(out.len(), t_out * c_out);
    for t in 0..t_out {
        let orow = &mut out[t * c_out..(t + 1) * c_out];
        match bias {
            Some(b) => orow.copy_from_slice(b),
            None => {
                for v in orow.iter_mut() {
                    *v = 0.0;
                }
            }
        }
        for co in 0..c_out {
            // ONNX Conv weight layout: [C_out, C_in, K]
            let w_co = &w[co * c_in * k..(co + 1) * c_in * k];
            let mut acc = orow[co];
            for kk in 0..k {
                let src = t as isize + kk as isize * dilation as isize - pad as isize;
                if src < 0 || src >= t_len as isize {
                    continue;
                }
                let xrow = &x[src as usize * c_in..src as usize * c_in + c_in];
                for ci in 0..c_in {
                    acc += w_co[ci * k + kk] * xrow[ci];
                }
            }
            orow[co] = acc;
        }
    }
}
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn conv1d_depthwise(
    x: &[f32],
    t_len: usize,
    channels: usize,
    k: usize,
    dilation: usize,
    pad: usize,
    w: &[f32],
    bias: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), t_len * channels);
    debug_assert_eq!(w.len(), channels * k);
    let t_out = (t_len + 2 * pad).saturating_sub((k - 1) * dilation);
    debug_assert_eq!(out.len(), t_out * channels);
    for t in 0..t_out {
        let orow = &mut out[t * channels..(t + 1) * channels];
        for c in 0..channels {
            let mut acc = bias[c];
            let wc = &w[c * k..(c + 1) * k];
            for kk in 0..k {
                let src = t as isize + kk as isize * dilation as isize - pad as isize;
                if src < 0 || src >= t_len as isize {
                    continue;
                }
                acc += wc[kk] * x[src as usize * channels + c];
            }
            orow[c] = acc;
        }
    }
}
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn silu_in_place(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v * sigmoid(*v);
    }
}
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn softmax_row(row: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in row.iter() {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0f32;
    for v in row.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn clip_in_place(x: &mut [f32], lo: f32, hi: f32) {
    for v in x.iter_mut() {
        *v = v.clamp(lo, hi);
    }
}
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512 {
use super::EPS;
#[cfg(target_arch = "x86")] use std::arch::x86::*;
#[cfg(target_arch = "x86_64")] use std::arch::x86_64::*;
#[inline]
fn sigmoid(v: f32) -> f32 { 1.0 / (1.0 + (-v).exp()) }
#[target_feature(enable = "avx512f,avx2,fma")]
pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: feature support is checked by dispatch; full vector loads are in bounds.
    unsafe {
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();
        let mut i = 0;
        while a.len() - i >= 64 {
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(i + 0)), _mm512_loadu_ps(b.as_ptr().add(i + 0)), acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(i + 16)), _mm512_loadu_ps(b.as_ptr().add(i + 16)), acc1);
            acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(i + 32)), _mm512_loadu_ps(b.as_ptr().add(i + 32)), acc2);
            acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(i + 48)), _mm512_loadu_ps(b.as_ptr().add(i + 48)), acc3);
            i += 64;
        }
        while a.len() - i >= 16 {
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(i)), _mm512_loadu_ps(b.as_ptr().add(i)), acc0);
            i += 16;
        }
        let mut lanes = [0.0f32; 16];
        let mut sum = 0.0f32;
        _mm512_storeu_ps(lanes.as_mut_ptr(), acc0);
        for v in lanes { sum += v; }
        _mm512_storeu_ps(lanes.as_mut_ptr(), acc1);
        for v in lanes { sum += v; }
        _mm512_storeu_ps(lanes.as_mut_ptr(), acc2);
        for v in lanes { sum += v; }
        _mm512_storeu_ps(lanes.as_mut_ptr(), acc3);
        for v in lanes { sum += v; }
        while i < a.len() { sum += a[i] * b[i]; i += 1; }
        sum
    }
}

#[target_feature(enable = "avx512f,avx2,fma")]
pub(super) unsafe fn matvec_into(x: &[f32], w: &[f32], out_dim: usize, in_dim: usize, y: &mut [f32]) {
    assert_eq!(x.len(), in_dim);
    assert_eq!(y.len(), out_dim);
    assert_eq!(Some(w.len()), out_dim.checked_mul(in_dim));
    if in_dim == 0 { y.fill(0.0); return; }
    // SAFETY: shape checks bound every load; dispatch checks the ISA.
    unsafe {
        let mut row = 0;
        while out_dim - row >= 4 {
            let mut a0_0 = _mm512_setzero_ps();
            let mut a0_1 = _mm512_setzero_ps();
            let mut a1_0 = _mm512_setzero_ps();
            let mut a1_1 = _mm512_setzero_ps();
            let mut a2_0 = _mm512_setzero_ps();
            let mut a2_1 = _mm512_setzero_ps();
            let mut a3_0 = _mm512_setzero_ps();
            let mut a3_1 = _mm512_setzero_ps();
            let mut i = 0;
            while in_dim - i >= 32 {
                let x0 = _mm512_loadu_ps(x.as_ptr().add(i + 0));
                let x1 = _mm512_loadu_ps(x.as_ptr().add(i + 16));
                a0_0 = _mm512_fmadd_ps(x0, _mm512_loadu_ps(w.as_ptr().add((row + 0) * in_dim + i + 0)), a0_0);
                a0_1 = _mm512_fmadd_ps(x1, _mm512_loadu_ps(w.as_ptr().add((row + 0) * in_dim + i + 16)), a0_1);
                a1_0 = _mm512_fmadd_ps(x0, _mm512_loadu_ps(w.as_ptr().add((row + 1) * in_dim + i + 0)), a1_0);
                a1_1 = _mm512_fmadd_ps(x1, _mm512_loadu_ps(w.as_ptr().add((row + 1) * in_dim + i + 16)), a1_1);
                a2_0 = _mm512_fmadd_ps(x0, _mm512_loadu_ps(w.as_ptr().add((row + 2) * in_dim + i + 0)), a2_0);
                a2_1 = _mm512_fmadd_ps(x1, _mm512_loadu_ps(w.as_ptr().add((row + 2) * in_dim + i + 16)), a2_1);
                a3_0 = _mm512_fmadd_ps(x0, _mm512_loadu_ps(w.as_ptr().add((row + 3) * in_dim + i + 0)), a3_0);
                a3_1 = _mm512_fmadd_ps(x1, _mm512_loadu_ps(w.as_ptr().add((row + 3) * in_dim + i + 16)), a3_1);
                i += 32;
            }
            while in_dim - i >= 16 {
                let xv = _mm512_loadu_ps(x.as_ptr().add(i));
                a0_0 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(w.as_ptr().add((row + 0) * in_dim + i)), a0_0);
                a1_0 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(w.as_ptr().add((row + 1) * in_dim + i)), a1_0);
                a2_0 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(w.as_ptr().add((row + 2) * in_dim + i)), a2_0);
                a3_0 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(w.as_ptr().add((row + 3) * in_dim + i)), a3_0);
                i += 16;
            }
            let mut lanes = [0.0f32; 16];
            let mut sum = 0.0f32;
            _mm512_storeu_ps(lanes.as_mut_ptr(), a0_0);
            for v in lanes { sum += v; }
            _mm512_storeu_ps(lanes.as_mut_ptr(), a0_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 0) * in_dim + k]; }
            y[row + 0] = sum;
            let mut sum = 0.0f32;
            _mm512_storeu_ps(lanes.as_mut_ptr(), a1_0);
            for v in lanes { sum += v; }
            _mm512_storeu_ps(lanes.as_mut_ptr(), a1_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 1) * in_dim + k]; }
            y[row + 1] = sum;
            let mut sum = 0.0f32;
            _mm512_storeu_ps(lanes.as_mut_ptr(), a2_0);
            for v in lanes { sum += v; }
            _mm512_storeu_ps(lanes.as_mut_ptr(), a2_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 2) * in_dim + k]; }
            y[row + 2] = sum;
            let mut sum = 0.0f32;
            _mm512_storeu_ps(lanes.as_mut_ptr(), a3_0);
            for v in lanes { sum += v; }
            _mm512_storeu_ps(lanes.as_mut_ptr(), a3_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 3) * in_dim + k]; }
            y[row + 3] = sum;
            row += 4;
        }
        for r in row..out_dim { y[r] = dot(&w[r * in_dim..(r + 1) * in_dim], x); }
    }
}
#[target_feature(enable = "avx512f,avx2,fma")]
pub(super) unsafe fn rms_norm_into(x: &[f32], weight: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    let ss = unsafe { dot(x, x) };
    let inv = 1.0f32 / (ss / x.len() as f32 + EPS).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * weight[i];
    }
}
#[target_feature(enable = "avx512f,avx2,fma")]
pub(super) unsafe fn rms_norm_block(x: &mut [f32], weight: &[f32]) {
    let dim = weight.len();
    debug_assert_eq!(x.len() % dim, 0);
    for row in x.chunks_exact_mut(dim) {
        let ss = unsafe { dot(row, row) };
        let inv = 1.0f32 / (ss / dim as f32 + EPS).sqrt();
        for (v, &w) in row.iter_mut().zip(weight.iter()) {
            *v = *v * inv * w;
        }
    }
}
#[target_feature(enable = "avx512f,avx2,fma")]
pub(super) unsafe fn apply_rope(x: &mut [f32], heads: usize, seq: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    debug_assert_eq!(x.len(), seq * heads * head_dim);
    debug_assert_eq!(cos.len(), seq * half);
    for t in 0..seq {
        let c = &cos[t * half..(t + 1) * half];
        let s = &sin[t * half..(t + 1) * half];
        for h in 0..heads {
            let row = &mut x[(t * heads + h) * head_dim..(t * heads + h) * head_dim + head_dim];
            for i in 0..half {
                let x1 = row[i];
                let x2 = row[i + half];
                row[i] = x1 * c[i] - x2 * s[i];
                row[i + half] = x2 * c[i] + x1 * s[i];
            }
        }
    }
}
#[target_feature(enable = "avx512f,avx2,fma")]
pub(super) unsafe fn conv1d(
    x: &[f32],
    t_len: usize,
    c_in: usize,
    c_out: usize,
    k: usize,
    dilation: usize,
    pad: usize,
    w: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), t_len * c_in);
    debug_assert_eq!(w.len(), c_out * c_in * k);
    let t_out = (t_len + 2 * pad).saturating_sub((k - 1) * dilation);
    debug_assert_eq!(out.len(), t_out * c_out);
    for t in 0..t_out {
        let orow = &mut out[t * c_out..(t + 1) * c_out];
        match bias {
            Some(b) => orow.copy_from_slice(b),
            None => {
                for v in orow.iter_mut() {
                    *v = 0.0;
                }
            }
        }
        for co in 0..c_out {
            // ONNX Conv weight layout: [C_out, C_in, K]
            let w_co = &w[co * c_in * k..(co + 1) * c_in * k];
            let mut acc = orow[co];
            for kk in 0..k {
                let src = t as isize + kk as isize * dilation as isize - pad as isize;
                if src < 0 || src >= t_len as isize {
                    continue;
                }
                let xrow = &x[src as usize * c_in..src as usize * c_in + c_in];
                for ci in 0..c_in {
                    acc += w_co[ci * k + kk] * xrow[ci];
                }
            }
            orow[co] = acc;
        }
    }
}
#[target_feature(enable = "avx512f,avx2,fma")]
pub(super) unsafe fn conv1d_depthwise(
    x: &[f32],
    t_len: usize,
    channels: usize,
    k: usize,
    dilation: usize,
    pad: usize,
    w: &[f32],
    bias: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), t_len * channels);
    debug_assert_eq!(w.len(), channels * k);
    let t_out = (t_len + 2 * pad).saturating_sub((k - 1) * dilation);
    debug_assert_eq!(out.len(), t_out * channels);
    for t in 0..t_out {
        let orow = &mut out[t * channels..(t + 1) * channels];
        for c in 0..channels {
            let mut acc = bias[c];
            let wc = &w[c * k..(c + 1) * k];
            for kk in 0..k {
                let src = t as isize + kk as isize * dilation as isize - pad as isize;
                if src < 0 || src >= t_len as isize {
                    continue;
                }
                acc += wc[kk] * x[src as usize * channels + c];
            }
            orow[c] = acc;
        }
    }
}
#[target_feature(enable = "avx512f,avx2,fma")]
pub(super) unsafe fn silu_in_place(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v * sigmoid(*v);
    }
}
#[target_feature(enable = "avx512f,avx2,fma")]
pub(super) unsafe fn softmax_row(row: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in row.iter() {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0f32;
    for v in row.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}
#[target_feature(enable = "avx512f,avx2,fma")]
pub(super) unsafe fn clip_in_place(x: &mut [f32], lo: f32, hi: f32) {
    for v in x.iter_mut() {
        *v = v.clamp(lo, hi);
    }
}
}

#[cfg(target_arch = "aarch64")]
mod neon {
use super::EPS;
use std::arch::aarch64::*;
#[inline]
fn sigmoid(v: f32) -> f32 { 1.0 / (1.0 + (-v).exp()) }
#[target_feature(enable = "neon")]
pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: feature support is checked by dispatch; full vector loads are in bounds.
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while a.len() - i >= 16 {
            acc0 = vfmaq_f32(acc0, vld1q_f32(a.as_ptr().add(i + 0)), vld1q_f32(b.as_ptr().add(i + 0)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(a.as_ptr().add(i + 4)), vld1q_f32(b.as_ptr().add(i + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(a.as_ptr().add(i + 8)), vld1q_f32(b.as_ptr().add(i + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(a.as_ptr().add(i + 12)), vld1q_f32(b.as_ptr().add(i + 12)));
            i += 16;
        }
        while a.len() - i >= 4 {
            acc0 = vfmaq_f32(acc0, vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            i += 4;
        }
        let mut lanes = [0.0f32; 4];
        let mut sum = 0.0f32;
        vst1q_f32(lanes.as_mut_ptr(), acc0);
        for v in lanes { sum += v; }
        vst1q_f32(lanes.as_mut_ptr(), acc1);
        for v in lanes { sum += v; }
        vst1q_f32(lanes.as_mut_ptr(), acc2);
        for v in lanes { sum += v; }
        vst1q_f32(lanes.as_mut_ptr(), acc3);
        for v in lanes { sum += v; }
        while i < a.len() { sum += a[i] * b[i]; i += 1; }
        sum
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn matvec_into(x: &[f32], w: &[f32], out_dim: usize, in_dim: usize, y: &mut [f32]) {
    assert_eq!(x.len(), in_dim);
    assert_eq!(y.len(), out_dim);
    assert_eq!(Some(w.len()), out_dim.checked_mul(in_dim));
    if in_dim == 0 { y.fill(0.0); return; }
    // SAFETY: shape checks bound every load; dispatch checks the ISA.
    unsafe {
        let mut row = 0;
        while out_dim - row >= 4 {
            let mut a0_0 = vdupq_n_f32(0.0);
            let mut a0_1 = vdupq_n_f32(0.0);
            let mut a1_0 = vdupq_n_f32(0.0);
            let mut a1_1 = vdupq_n_f32(0.0);
            let mut a2_0 = vdupq_n_f32(0.0);
            let mut a2_1 = vdupq_n_f32(0.0);
            let mut a3_0 = vdupq_n_f32(0.0);
            let mut a3_1 = vdupq_n_f32(0.0);
            let mut i = 0;
            while in_dim - i >= 8 {
                let x0 = vld1q_f32(x.as_ptr().add(i + 0));
                let x1 = vld1q_f32(x.as_ptr().add(i + 4));
                a0_0 = vfmaq_f32(a0_0, x0, vld1q_f32(w.as_ptr().add((row + 0) * in_dim + i + 0)));
                a0_1 = vfmaq_f32(a0_1, x1, vld1q_f32(w.as_ptr().add((row + 0) * in_dim + i + 4)));
                a1_0 = vfmaq_f32(a1_0, x0, vld1q_f32(w.as_ptr().add((row + 1) * in_dim + i + 0)));
                a1_1 = vfmaq_f32(a1_1, x1, vld1q_f32(w.as_ptr().add((row + 1) * in_dim + i + 4)));
                a2_0 = vfmaq_f32(a2_0, x0, vld1q_f32(w.as_ptr().add((row + 2) * in_dim + i + 0)));
                a2_1 = vfmaq_f32(a2_1, x1, vld1q_f32(w.as_ptr().add((row + 2) * in_dim + i + 4)));
                a3_0 = vfmaq_f32(a3_0, x0, vld1q_f32(w.as_ptr().add((row + 3) * in_dim + i + 0)));
                a3_1 = vfmaq_f32(a3_1, x1, vld1q_f32(w.as_ptr().add((row + 3) * in_dim + i + 4)));
                i += 8;
            }
            while in_dim - i >= 4 {
                let xv = vld1q_f32(x.as_ptr().add(i));
                a0_0 = vfmaq_f32(a0_0, xv, vld1q_f32(w.as_ptr().add((row + 0) * in_dim + i)));
                a1_0 = vfmaq_f32(a1_0, xv, vld1q_f32(w.as_ptr().add((row + 1) * in_dim + i)));
                a2_0 = vfmaq_f32(a2_0, xv, vld1q_f32(w.as_ptr().add((row + 2) * in_dim + i)));
                a3_0 = vfmaq_f32(a3_0, xv, vld1q_f32(w.as_ptr().add((row + 3) * in_dim + i)));
                i += 4;
            }
            let mut lanes = [0.0f32; 4];
            let mut sum = 0.0f32;
            vst1q_f32(lanes.as_mut_ptr(), a0_0);
            for v in lanes { sum += v; }
            vst1q_f32(lanes.as_mut_ptr(), a0_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 0) * in_dim + k]; }
            y[row + 0] = sum;
            let mut sum = 0.0f32;
            vst1q_f32(lanes.as_mut_ptr(), a1_0);
            for v in lanes { sum += v; }
            vst1q_f32(lanes.as_mut_ptr(), a1_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 1) * in_dim + k]; }
            y[row + 1] = sum;
            let mut sum = 0.0f32;
            vst1q_f32(lanes.as_mut_ptr(), a2_0);
            for v in lanes { sum += v; }
            vst1q_f32(lanes.as_mut_ptr(), a2_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 2) * in_dim + k]; }
            y[row + 2] = sum;
            let mut sum = 0.0f32;
            vst1q_f32(lanes.as_mut_ptr(), a3_0);
            for v in lanes { sum += v; }
            vst1q_f32(lanes.as_mut_ptr(), a3_1);
            for v in lanes { sum += v; }
            for k in i..in_dim { sum += x[k] * w[(row + 3) * in_dim + k]; }
            y[row + 3] = sum;
            row += 4;
        }
        for r in row..out_dim { y[r] = dot(&w[r * in_dim..(r + 1) * in_dim], x); }
    }
}
#[target_feature(enable = "neon")]
pub(super) unsafe fn rms_norm_into(x: &[f32], weight: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    let ss = unsafe { dot(x, x) };
    let inv = 1.0f32 / (ss / x.len() as f32 + EPS).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * weight[i];
    }
}
#[target_feature(enable = "neon")]
pub(super) unsafe fn rms_norm_block(x: &mut [f32], weight: &[f32]) {
    let dim = weight.len();
    debug_assert_eq!(x.len() % dim, 0);
    for row in x.chunks_exact_mut(dim) {
        let ss = unsafe { dot(row, row) };
        let inv = 1.0f32 / (ss / dim as f32 + EPS).sqrt();
        for (v, &w) in row.iter_mut().zip(weight.iter()) {
            *v = *v * inv * w;
        }
    }
}
#[target_feature(enable = "neon")]
pub(super) unsafe fn apply_rope(x: &mut [f32], heads: usize, seq: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    debug_assert_eq!(x.len(), seq * heads * head_dim);
    debug_assert_eq!(cos.len(), seq * half);
    for t in 0..seq {
        let c = &cos[t * half..(t + 1) * half];
        let s = &sin[t * half..(t + 1) * half];
        for h in 0..heads {
            let row = &mut x[(t * heads + h) * head_dim..(t * heads + h) * head_dim + head_dim];
            for i in 0..half {
                let x1 = row[i];
                let x2 = row[i + half];
                row[i] = x1 * c[i] - x2 * s[i];
                row[i + half] = x2 * c[i] + x1 * s[i];
            }
        }
    }
}
#[target_feature(enable = "neon")]
pub(super) unsafe fn conv1d(
    x: &[f32],
    t_len: usize,
    c_in: usize,
    c_out: usize,
    k: usize,
    dilation: usize,
    pad: usize,
    w: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), t_len * c_in);
    debug_assert_eq!(w.len(), c_out * c_in * k);
    let t_out = (t_len + 2 * pad).saturating_sub((k - 1) * dilation);
    debug_assert_eq!(out.len(), t_out * c_out);
    for t in 0..t_out {
        let orow = &mut out[t * c_out..(t + 1) * c_out];
        match bias {
            Some(b) => orow.copy_from_slice(b),
            None => {
                for v in orow.iter_mut() {
                    *v = 0.0;
                }
            }
        }
        for co in 0..c_out {
            // ONNX Conv weight layout: [C_out, C_in, K]
            let w_co = &w[co * c_in * k..(co + 1) * c_in * k];
            let mut acc = orow[co];
            for kk in 0..k {
                let src = t as isize + kk as isize * dilation as isize - pad as isize;
                if src < 0 || src >= t_len as isize {
                    continue;
                }
                let xrow = &x[src as usize * c_in..src as usize * c_in + c_in];
                for ci in 0..c_in {
                    acc += w_co[ci * k + kk] * xrow[ci];
                }
            }
            orow[co] = acc;
        }
    }
}
#[target_feature(enable = "neon")]
pub(super) unsafe fn conv1d_depthwise(
    x: &[f32],
    t_len: usize,
    channels: usize,
    k: usize,
    dilation: usize,
    pad: usize,
    w: &[f32],
    bias: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), t_len * channels);
    debug_assert_eq!(w.len(), channels * k);
    let t_out = (t_len + 2 * pad).saturating_sub((k - 1) * dilation);
    debug_assert_eq!(out.len(), t_out * channels);
    for t in 0..t_out {
        let orow = &mut out[t * channels..(t + 1) * channels];
        for c in 0..channels {
            let mut acc = bias[c];
            let wc = &w[c * k..(c + 1) * k];
            for kk in 0..k {
                let src = t as isize + kk as isize * dilation as isize - pad as isize;
                if src < 0 || src >= t_len as isize {
                    continue;
                }
                acc += wc[kk] * x[src as usize * channels + c];
            }
            orow[c] = acc;
        }
    }
}
#[target_feature(enable = "neon")]
pub(super) unsafe fn silu_in_place(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v * sigmoid(*v);
    }
}
#[target_feature(enable = "neon")]
pub(super) unsafe fn softmax_row(row: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in row.iter() {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0f32;
    for v in row.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}
#[target_feature(enable = "neon")]
pub(super) unsafe fn clip_in_place(x: &mut [f32], lo: f32, hi: f32) {
    for v in x.iter_mut() {
        *v = v.clamp(lo, hi);
    }
}
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tables() -> Vec<Kernels> {
        ["portable", "sse2", "avx", "avx_fma", "avx2_fma", "avx512", "neon"]
            .iter().filter_map(|n| available(n)).collect()
    }
    fn data(n: usize, seed: usize) -> Vec<f32> {
        (0..n).map(|i| (((i * 37 + seed) % 211) as f32 - 105.0) / 113.0).collect()
    }
    fn close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (&x, &y) in a.iter().zip(b) {
            assert!(x.is_finite() && y.is_finite());
            assert!((x-y).abs() <= tol * (1.0 + y.abs()), "{x} != {y}");
        }
    }
    #[test]
    fn all_available_dot_backends_match_f64() {
        for k in tables() {
            for n in [0,1,3,4,7,8,15,16,17,24,31,32,33,63,64,65,95,96,97,127,128,129,160,255,256,257,1025] {
                let a=data(n+3,11); let b=data(n+5,79);
                let a=&a[1..n+1]; let b=&b[2..n+2];
                let expected: f64=a.iter().zip(b).map(|(&x,&y)| x as f64*y as f64).sum();
                let scale: f64=a.iter().zip(b).map(|(&x,&y)| (x as f64*y as f64).abs()).sum();
                let got=unsafe {(k.dot)(a,b)} as f64;
                assert!(got.is_finite() && (got-expected).abs() <= 2e-6*(1.0+scale), "{} n={n}",k.name);
            }
        }
    }
    #[test]
    fn all_available_matvec_and_norm_backends() {
        for k in tables() {
            for n in [0,1,7,16,17,24,32,63,64,96,128,160,257] {
                for rows in [0,1,3,4,5,8,9] {
                    let x=data(n,3); let w=data(n*rows,13);
                    let mut y=vec![0.0;rows]; let mut reference=y.clone();
                    unsafe {(k.matvec_into)(&x,&w,rows,n,&mut y)};
                    portable::matvec_into(&x,&w,rows,n,&mut reference); close(&y,&reference,2e-5);
                }
                if n>0 {
                    let x=data(n,7); let w=data(n,5); let mut y=vec![0.0;n]; let mut reference=y.clone();
                    unsafe {(k.rms_norm_into)(&x,&w,&mut y)};
                    portable::rms_norm_into(&x,&w,&mut reference); close(&y,&reference,2e-5);
                    let mut x=data(n*3,9); let mut reference=x.clone();
                    unsafe {(k.rms_norm_block)(&mut x,&w)};
                    portable::rms_norm_block(&mut reference,&w); close(&x,&reference,2e-5);
                }
            }
        }
    }
    #[test]
    fn remaining_kernels_match_portable() {
        for k in tables() {
            for n in [1,7,24,96,129] {
                let mut a=data(n,7); let mut b=a.clone();
                unsafe {(k.silu_in_place)(&mut a)}; portable::silu_in_place(&mut b); close(&a,&b,2e-6);
                unsafe {(k.softmax_row)(&mut a)}; portable::softmax_row(&mut b); close(&a,&b,2e-6);
                unsafe {(k.clip_in_place)(&mut a,0.01,0.1)}; portable::clip_in_place(&mut b,0.01,0.1); close(&a,&b,2e-6);
            }
            let mut a=data(2*3*24,5); let mut b=a.clone(); let c=vec![0.75;2*12]; let s=vec![0.25;2*12];
            unsafe {(k.apply_rope)(&mut a,3,2,24,&c,&s)};
            portable::apply_rope(&mut b,3,2,24,&c,&s); close(&a,&b,2e-6);
            for channels in [1,7,24] {
                for dilation in [1,2] {
                    let t=9; let kw=3; let pad=dilation; let co=5;
                    let x=data(t*channels,3); let w=data(co*channels*kw,7); let bias=data(co,13);
                    let mut a=vec![0.0;t*co]; let mut b=a.clone();
                    unsafe {(k.conv1d)(&x,t,channels,co,kw,dilation,pad,&w,Some(&bias),&mut a)};
                    portable::conv1d(&x,t,channels,co,kw,dilation,pad,&w,Some(&bias),&mut b); close(&a,&b,2e-5);
                    let w=data(channels*kw,11); let bias=data(channels,3);
                    let mut a=vec![0.0;t*channels]; let mut b=a.clone();
                    unsafe {(k.conv1d_depthwise)(&x,t,channels,kw,dilation,pad,&w,&bias,&mut a)};
                    portable::conv1d_depthwise(&x,t,channels,kw,dilation,pad,&w,&bias,&mut b); close(&a,&b,2e-5);
                }
            }
        }
    }
    #[test]
    #[should_panic]
    fn dot_rejects_mismatched_lengths() { dot(&[1.0],&[]); }
    #[test]
    #[should_panic]
    fn matvec_rejects_mismatched_shape() { matvec_into(&[1.0],&[],1,1,&mut [0.0]); }
    #[test]
    fn unknown_backend_is_unavailable() { assert!(available("not-a-backend").is_none()); }
    #[test]
    #[ignore = "manual microbenchmark, not an end-to-end performance claim"]
    fn benchmark_available_backends() {
        use std::{hint::black_box, time::Instant};
        for k in tables() {
            for n in [24,96,128,160,256] {
                let x=data(n,7); let w=data(n*128,3); let mut y=vec![0.0;128];
                for _ in 0..100 { unsafe {(k.matvec_into)(black_box(&x),black_box(&w),128,n,black_box(&mut y))}; }
                let start=Instant::now();
                for _ in 0..2000 { unsafe {(k.matvec_into)(black_box(&x),black_box(&w),128,n,black_box(&mut y))}; black_box(&y); }
                eprintln!("{} matvec 128x{}: {:.1} ns/call",k.name,n,start.elapsed().as_nanos() as f64/2000.0);
            }
        }
    }
}
