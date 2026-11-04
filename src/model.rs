//! Pure-Rust numerical core for the PiVo acoustic pipeline: hand-vectorized f32
//! kernels (dot, matvec, RMSNorm, RoPE, 1-D convolutions, softmax, SiLU, GRUs)
//! plus complete reproductions of the four PiVo ONNX inference graphs (encoder,
//! decoder, conditioning, vocoder_step) without ONNX Runtime.
//!
//! ## Future-proofing for PiVo-v2 (same pipeline, new dims)
//!
//! Every dimension and graph parameter is resolved at load time — never
//! hard-coded — from three sources:
//!   1. `metadata.json` `config` (widths, layers, heads, window, hop, rate…)
//!   2. ONNX initializer *shapes* (vocab, voices, feature_dim, vocoder dims…)
//!   3. ONNX *node structure* (block types, RoPE presence via Sin/Cos ops,
//!      conv dilations/pads, weight lookup by stable node name).
//!
//! A v2 export with a bigger/smaller model, more voices, a different hop size,
//! or new initializer names drops in with zero code changes.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::onnx_loader::{ModelLoadError, ModelMetadata, OnnxGraph};
use crate::text;
use crate::cpu::{dot, matvec_into, rms_norm_into, rms_norm_block, apply_rope, conv1d, conv1d_depthwise, silu_in_place, softmax_row, clip_in_place};

const EPS: f32 = 1e-6;
const NEG_MASK: f32 = -10000.0;
const PITCH_GAIN_EPS: f32 = 1e-5;
/// FARGAN pitch memory ring length (structural constant of the vocoder graph).
const PITCH_MEMORY_LEN: usize = 256;

// ---------------------------------------------------------------------------
// Platform-dispatched numerical primitives live in cpu.rs.
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// RoPE freqs: exp(i * -ln(10000) / head_dim * 2) for i = 0..head_dim/2.
/// Matches ONNX: freqs[i] = exp(2i * -ln(10000) / head_dim) — for head_dim=24
/// these are exp(-0.7677·i) = 10000^{-i/12}.
fn rope_freqs(head_dim: usize) -> Vec<f32> {
    const LN_10000: f32 = 9.210_340_499_877_93;
    let half = head_dim / 2;
    (0..half)
        .map(|i| (2.0f32 * i as f32 * (-LN_10000 / head_dim as f32)).exp())
        .collect()
}

fn rope_tables(positions: &[i64], freqs: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut cos = Vec::with_capacity(positions.len() * freqs.len());
    let mut sin = Vec::with_capacity(positions.len() * freqs.len());
    for &p in positions {
        for &f in freqs {
            let angle = p as f32 * f;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

// ---------------------------------------------------------------------------
// Weight helpers (graph-driven)
// ---------------------------------------------------------------------------

fn val_err(msg: impl Into<String>) -> ModelLoadError {
    ModelLoadError::Validation(msg.into())
}

/// Take a named initializer with an expected shape (empty shape = any).
fn take_w(g: &mut OnnxGraph, name: &str, shape: &[usize]) -> Result<Vec<f32>, ModelLoadError> {
    let t = g.take(name)?;
    if !shape.is_empty() && t.dims != shape {
        return Err(val_err(format!(
            "initializer '{}' shape mismatch: expected {:?}, got {:?}",
            name, shape, t.dims
        )));
    }
    Ok(t.data)
}

/// Take a named initializer and return (data, dims).
fn take_w_dims(g: &mut OnnxGraph, name: &str) -> Result<(Vec<f32>, Vec<usize>), ModelLoadError> {
    let t = g.take(name)?;
    Ok((t.data, t.dims))
}

/// Take a MatMul node's weight by node name, transposed to [out,in].
fn w_mat(g: &mut OnnxGraph, node: &str) -> Result<Vec<f32>, ModelLoadError> {
    Ok(g.take_node_weight_t(node)?.0)
}

/// Take a MatMul node's weight by node name, transposed, returning (data, out_dim, in_dim).
fn w_mat_dims(g: &mut OnnxGraph, node: &str) -> Result<(Vec<f32>, usize, usize), ModelLoadError> {
    g.take_node_weight_t(node)
}

// ---------------------------------------------------------------------------
// Attention block weights (shared by encoder + decoder)
// ---------------------------------------------------------------------------

struct AttnWeights {
    q_w: Vec<f32>,    // [q_out, width]
    k_w: Vec<f32>,    // [kv_out, width]
    v_w: Vec<f32>,    // [kv_out, width]
    o_w: Vec<f32>,    // [width, width]
    q_norm: Vec<f32>, // [head_dim]
    k_norm: Vec<f32>, // [head_dim]
    norm1: Vec<f32>,  // [width]
    norm2: Vec<f32>,  // [width]
    gate_w: Vec<f32>, // [ff, width]
    up_w: Vec<f32>,   // [ff, width]
    down_w: Vec<f32>, // [width, ff]
    q_out: usize,
    kv_out: usize,
    ff: usize,
    rope: bool,
}

impl AttnWeights {
    fn load(
        g: &mut OnnxGraph,
        scope: &str,
        width: usize,
    ) -> Result<Self, ModelLoadError> {
        let m = |s: &str| format!("/{}/{}", scope, s);
        let (q_w, q_out, _) = w_mat_dims(g, &m("attention/q/MatMul"))?;
        let (k_w, kv_out, _) = w_mat_dims(g, &m("attention/k/MatMul"))?;
        let (v_w, v_out, _) = w_mat_dims(g, &m("attention/v/MatMul"))?;
        if v_out != kv_out {
            return Err(val_err(format!(
                "{scope}: k/v output dims differ ({kv_out} vs {v_out})"
            )));
        }
        let (o_w, _, _) = w_mat_dims(g, &m("attention/out/MatMul"))?;
        let (gate_w, ff, _) = w_mat_dims(g, &m("gate/MatMul"))?;
        let (up_w, up_out, _) = w_mat_dims(g, &m("up/MatMul"))?;
        if up_out != ff {
            return Err(val_err(format!(
                "{scope}: gate/up output dims differ ({ff} vs {up_out})"
            )));
        }
        let (down_w, _, _) = w_mat_dims(g, &m("down/MatMul"))?;
        // RoPE detection: Sin+Cos trig ops inside this attention's graph scope.
        let rope_scope = format!("/{}/attention/", scope);
        let rope = g.has_op_under(&rope_scope, &["Sin"]) && g.has_op_under(&rope_scope, &["Cos"]);
        Ok(Self {
            q_w,
            k_w,
            v_w,
            o_w,
            q_norm: take_w(g, &format!("acoustic.{}.attention.q_norm.weight", scope), &[])?,
            k_norm: take_w(g, &format!("acoustic.{}.attention.k_norm.weight", scope), &[])?,
            norm1: take_w(g, &format!("acoustic.{}.norm1.weight", scope), &[width])?,
            norm2: take_w(g, &format!("acoustic.{}.norm2.weight", scope), &[width])?,
            gate_w,
            up_w,
            down_w,
            q_out,
            kv_out,
            ff,
            rope,
        })
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

struct ConvBlockWeights {
    norm1: Vec<f32>, // [width]
    norm2: Vec<f32>, // [width]
    conv_w: Vec<f32>, // [width, width, kernel]
    dilation: usize,
    kernel: usize,
    pad: usize,
}

enum EncBlock {
    Attn(AttnWeights),
    Conv(ConvBlockWeights),
}

/// One conv→pointwise layer of the duration predictor.
struct DurLayer {
    dw_w: Vec<f32>, // [dw, 1, k]
    dw_b: Vec<f32>, // [dw]
    pw_w: Vec<f32>, // [dw, dw, 1]
    pw_b: Vec<f32>, // [dw]
    norm: Vec<f32>, // [dw]
    k: usize,
    pad: usize,
}

pub struct Encoder {
    tokens: Vec<f32>,       // [vocab, width]
    voices: Vec<f32>,       // [n_voices, width]
    blocks: Vec<EncBlock>,  // text_layers
    text_norm: Vec<f32>,    // [width]
    width: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    // duration predictor
    dur_width: usize,
    dur_in_w: Vec<f32>,  // [dw, width]
    dur_in_b: Vec<f32>,  // [dw]
    dur_layers: Vec<DurLayer>,
    dur_out_w: Vec<f32>, // [1, dw]
    dur_out_b: Vec<f32>, // [1]
    rope_freqs: Vec<f32>,
}

impl Encoder {
    pub fn from_onnx_bytes(buf: &[u8], cfg: &crate::onnx_loader::ModelConfig) -> Result<Self, ModelLoadError> {
        let mut g = crate::onnx_loader::parse_onnx_graph(buf)?;
        let (tokens, tdims) = take_w_dims(&mut g, "acoustic.tokens.weight")?;
        if tdims.len() != 2 {
            return Err(val_err("acoustic.tokens.weight must be 2-D"));
        }
        let width = tdims[1];
        if cfg.width != width {
            eprintln!(
                "[pivo-studio] note: metadata width={} but token embedding is {} wide — using {}",
                cfg.width, width, width
            );
        }
        let (voices, vdims) = take_w_dims(&mut g, "acoustic.voices.weight")?;
        if vdims.len() != 2 || vdims[1] != width {
            return Err(val_err("acoustic.voices.weight must be [n_voices, width]"));
        }
        let heads = cfg.heads.max(1);
        if width % heads != 0 {
            return Err(val_err(format!(
                "width {width} not divisible by heads {heads}"
            )));
        }
        let head_dim = width / heads;
        let rope_freqs_v = rope_freqs(head_dim);

        // Discover text blocks by scanning for /text_blocks.N/{conv|attention} nodes.
        let mut n_blocks = 0usize;
        while g.node(&format!("/text_blocks.{}/norm1/RMSNorm", n_blocks)).is_some()
            || g.node(&format!("/text_blocks.{}/conv/Conv", n_blocks)).is_some()
            || g.node(&format!("/text_blocks.{}/attention/q/MatMul", n_blocks)).is_some()
        {
            n_blocks += 1;
        }
        if n_blocks == 0 {
            return Err(val_err("encoder graph has no text_blocks.* nodes"));
        }
        if n_blocks != cfg.text_layers {
            eprintln!(
                "[pivo-studio] note: metadata text_layers={} but encoder has {} blocks — using {}",
                cfg.text_layers, n_blocks, n_blocks
            );
        }

        let mut blocks = Vec::with_capacity(n_blocks);
        let mut kv_heads = cfg.kv_heads;
        for i in 0..n_blocks {
            let conv_node = format!("/text_blocks.{}/conv/Conv", i);
            if g.node(&conv_node).is_some() {
                let scope = format!("text_blocks.{}", i);
                let (conv_w, cdims) =
                    take_w_dims(&mut g, &format!("acoustic.{}.conv.weight", scope))?;
                if cdims.len() != 3 {
                    return Err(val_err(format!("{}.conv.weight must be 3-D", scope)));
                }
                let kernel = cdims[2];
                let dilation = g
                    .conv_attr_ints(&conv_node, "dilations")
                    .and_then(|d| d.first().copied())
                    .unwrap_or(1)
                    .max(1) as usize;
                let pads = g
                    .conv_attr_ints(&conv_node, "pads")
                    .cloned()
                    .unwrap_or_else(|| vec![0, 0]);
                let pad = pads[0].max(0) as usize;
                blocks.push(EncBlock::Conv(ConvBlockWeights {
                    norm1: take_w(&mut g, &format!("acoustic.{}.norm1.weight", scope), &[width])?,
                    norm2: take_w(&mut g, &format!("acoustic.{}.norm2.weight", scope), &[width])?,
                    conv_w,
                    dilation,
                    kernel,
                    pad,
                }));
            } else if g.node(&format!("/text_blocks.{}/attention/q/MatMul", i)).is_some() {
                let scope = format!("text_blocks.{}", i);
                let aw = AttnWeights::load(&mut g, &scope, width)?;
                kv_heads = aw.kv_out / head_dim;
                blocks.push(EncBlock::Attn(aw));
            } else {
                return Err(val_err(format!(
                    "encoder text_blocks.{} has neither conv nor attention",
                    i
                )));
            }
        }

        // duration predictor: in_proj MatMul defines dur width; layers.* conv pairs
        let (dur_in_w, dur_width, _) = w_mat_dims(&mut g, "/durations/in_proj/MatMul")?;
        let dur_in_b = take_w(&mut g, "acoustic.durations.in_proj.bias", &[dur_width])?;
        let mut dur_layers = Vec::new();
        let mut li = 0;
        loop {
            let dw_name = format!("acoustic.durations.layers.{}.0.weight", li);
            if !g.tensors.contains_key(&dw_name) {
                break;
            }
            let db_name = format!("acoustic.durations.layers.{}.0.bias", li);
            let pw_name = format!("acoustic.durations.layers.{}.1.weight", li);
            let pb_name = format!("acoustic.durations.layers.{}.1.bias", li);
            let norm_name = format!("acoustic.durations.norms.{}.weight", li);
            let (dw_w, ddims) = take_w_dims(&mut g, &dw_name)?;
            if ddims.len() != 3 {
                return Err(val_err("duration depthwise weight must be [dw,1,k]"));
            }
            let k = ddims[2];
            let conv_node = format!("/durations/layers.{}/layers.{}.0/Conv", li, li);
            let pads = g
                .conv_attr_ints(&conv_node, "pads")
                .cloned()
                .unwrap_or_else(|| vec![0, 0]);
            dur_layers.push(DurLayer {
                dw_w,
                dw_b: take_w(&mut g, &db_name, &[dur_width])?,
                pw_w: take_w(&mut g, &pw_name, &[dur_width, dur_width, 1])?,
                pw_b: take_w(&mut g, &pb_name, &[dur_width])?,
                norm: take_w(&mut g, &norm_name, &[dur_width])?,
                k,
                pad: pads[0].max(0) as usize,
            });
            li += 1;
        }
        if dur_layers.is_empty() {
            return Err(val_err("duration predictor has no conv layers"));
        }
        let dur_out_w = w_mat(&mut g, "/durations/out/MatMul")?;
        let dur_out_b = take_w(&mut g, "acoustic.durations.out.bias", &[1])?;

        Ok(Self {
            tokens,
            voices,
            blocks,
            text_norm: take_w(&mut g, "acoustic.text_norm.weight", &[width])?,
            width,
            heads,
            kv_heads: kv_heads.max(1),
            head_dim,
            dur_width,
            dur_in_w,
            dur_in_b,
            dur_layers,
            dur_out_w,
            dur_out_b,
            rope_freqs: rope_freqs_v,
        })
    }

    /// tokens: [T] token ids, voice index, length: T (single utterance, batch=1).
    /// Returns (encoded [T,width], log_durations [T]).
    pub fn run(&self, tokens: &[i64], voice: i64, length: usize) -> (Vec<f32>, Vec<f32>) {
        let t_len = tokens.len();
        let width = self.width;
        let heads = self.heads;
        let kv_heads = self.kv_heads;
        let head_dim = self.head_dim;
        let kv_dim = kv_heads * head_dim;
        let group = heads / kv_heads.max(1);
        let attn_scale = (head_dim as f32).sqrt();
        let positions: Vec<i64> = (0..t_len as i64).collect();
        let (cos, sin) = rope_tables(&positions, &self.rope_freqs);

        // embeddings + padding mask
        let mut x = vec![0f32; t_len * width];
        for (t, &tok) in tokens.iter().enumerate() {
            let emb = &self.tokens[tok as usize * width..(tok as usize + 1) * width];
            let v = &self.voices[voice as usize * width..(voice as usize + 1) * width];
            let row = &mut x[t * width..(t + 1) * width];
            for i in 0..width {
                row[i] = emb[i] + v[i];
            }
        }
        for t in length..t_len {
            for v in &mut x[t * width..(t + 1) * width] {
                *v = 0.0;
            }
        }

        let mut scratch = vec![0f32; t_len * width];
        for block in &self.blocks {
            match block {
                EncBlock::Attn(aw) => {
                    for t in 0..t_len {
                        rms_norm_into(
                            &x[t * width..(t + 1) * width],
                            &aw.norm1,
                            &mut scratch[t * width..(t + 1) * width],
                        );
                    }
                    let mut q = vec![0f32; t_len * aw.q_out];
                    let mut k = vec![0f32; t_len * aw.kv_out];
                    let mut v = vec![0f32; t_len * aw.kv_out];
                    for t in 0..t_len {
                        let xr = &scratch[t * width..(t + 1) * width];
                        matvec_into(xr, &aw.q_w, aw.q_out, width, &mut q[t * aw.q_out..(t + 1) * aw.q_out]);
                        matvec_into(xr, &aw.k_w, aw.kv_out, width, &mut k[t * kv_dim..(t + 1) * kv_dim]);
                        matvec_into(xr, &aw.v_w, aw.kv_out, width, &mut v[t * kv_dim..(t + 1) * kv_dim]);
                    }
                    rms_norm_block(&mut q, &aw.q_norm);
                    rms_norm_block(&mut k, &aw.k_norm);
                    if aw.rope {
                        apply_rope(&mut q, heads, t_len, head_dim, &cos, &sin);
                        apply_rope(&mut k, kv_heads, t_len, head_dim, &cos, &sin);
                    }
                    // attention: token-major q [T,heads,hd] over k [T,kv,hd] with j < length
                    let mut attn = vec![0f32; t_len * aw.q_out];
                    let mut scores = vec![0f32; t_len];
                    for t in 0..t_len {
                        for h in 0..heads {
                            let kvh = h / group;
                            let qrow = &q[t * aw.q_out + h * head_dim..t * aw.q_out + h * head_dim + head_dim];
                            for j in 0..length {
                                let krow = &k[j * kv_dim + kvh * head_dim..j * kv_dim + kvh * head_dim + head_dim];
                                scores[j] = dot(qrow, krow) / attn_scale;
                            }
                            for j in length..t_len {
                                scores[j] = NEG_MASK;
                            }
                            softmax_row(&mut scores[..t_len]);
                            let orow = &mut attn[t * aw.q_out + h * head_dim..t * aw.q_out + h * head_dim + head_dim];
                            for vi in 0..head_dim {
                                let mut acc = 0f32;
                                for j in 0..t_len {
                                    acc += scores[j] * v[j * kv_dim + kvh * head_dim + vi];
                                }
                                orow[vi] = acc;
                            }
                        }
                    }
                    // out proj + residual
                    for t in 0..t_len {
                        let mut orow = vec![0f32; width];
                        matvec_into(&attn[t * aw.q_out..(t + 1) * aw.q_out], &aw.o_w, width, aw.q_out, &mut orow);
                        for i in 0..width {
                            x[t * width + i] += orow[i];
                        }
                    }
                    // norm2 + gated FFN + residual
                    for t in 0..t_len {
                        let xr = &x[t * width..(t + 1) * width];
                        rms_norm_into(xr, &aw.norm2, &mut scratch[t * width..(t + 1) * width]);
                    }
                    let mut gate = vec![0f32; t_len * aw.ff];
                    let mut up = vec![0f32; t_len * aw.ff];
                    for t in 0..t_len {
                        let xr = &scratch[t * width..(t + 1) * width];
                        matvec_into(xr, &aw.gate_w, aw.ff, width, &mut gate[t * aw.ff..(t + 1) * aw.ff]);
                        matvec_into(xr, &aw.up_w, aw.ff, width, &mut up[t * aw.ff..(t + 1) * aw.ff]);
                    }
                    for i in 0..gate.len() {
                        gate[i] = gate[i] * sigmoid(gate[i]) * up[i];
                    }
                    for t in 0..t_len {
                        let mut down = vec![0f32; width];
                        matvec_into(&gate[t * aw.ff..(t + 1) * aw.ff], &aw.down_w, width, aw.ff, &mut down);
                        for i in 0..width {
                            x[t * width + i] += down[i];
                        }
                    }
                    for t in length..t_len {
                        for v in &mut x[t * width..(t + 1) * width] {
                            *v = 0.0;
                        }
                    }
                }
                EncBlock::Conv(cb) => {
                    for t in 0..t_len {
                        rms_norm_into(
                            &x[t * width..(t + 1) * width],
                            &cb.norm1,
                            &mut scratch[t * width..(t + 1) * width],
                        );
                    }
                    for t in length..t_len {
                        for v in &mut scratch[t * width..(t + 1) * width] {
                            *v = 0.0;
                        }
                    }
                    let mut conv_out = vec![0f32; t_len * width];
                    conv1d(&scratch, t_len, width, width, cb.kernel, cb.dilation, cb.pad, &cb.conv_w, None, &mut conv_out);
                    for t in 0..t_len {
                        rms_norm_into(
                            &conv_out[t * width..(t + 1) * width],
                            &cb.norm2,
                            &mut scratch[t * width..(t + 1) * width],
                        );
                    }
                    silu_in_place(&mut scratch);
                    for i in 0..x.len() {
                        x[i] += scratch[i];
                    }
                    for t in length..t_len {
                        for v in &mut x[t * width..(t + 1) * width] {
                            *v = 0.0;
                        }
                    }
                }
            }
        }

        // text_norm + mask → encoded
        let mut encoded = vec![0f32; t_len * width];
        for t in 0..t_len {
            rms_norm_into(&x[t * width..(t + 1) * width], &self.text_norm, &mut encoded[t * width..(t + 1) * width]);
        }
        for t in length..t_len {
            for v in &mut encoded[t * width..(t + 1) * width] {
                *v = 0.0;
            }
        }

        // duration predictor
        let dw = self.dur_width;
        let mut dur = vec![0f32; t_len * dw];
        for t in 0..t_len {
            let mut out = vec![0f32; dw];
            matvec_into(&encoded[t * width..(t + 1) * width], &self.dur_in_w, dw, width, &mut out);
            for i in 0..dw {
                let v = out[i] + self.dur_in_b[i];
                dur[t * dw + i] = v * sigmoid(v);
            }
        }
        for t in length..t_len {
            for v in &mut dur[t * dw..(t + 1) * dw] {
                *v = 0.0;
            }
        }
        let mut dbuf = vec![0f32; t_len * dw];
        let mut dbuf2 = vec![0f32; t_len * dw];
        for (li, layer) in self.dur_layers.iter().enumerate() {
            let src: &[f32] = if li == 0 { &dur } else { &dbuf2 };
            conv1d_depthwise(src, t_len, dw, layer.k, 1, layer.pad, &layer.dw_w, &layer.dw_b, &mut dbuf);
            conv1d(&dbuf, t_len, dw, dw, 1, 1, 0, &layer.pw_w, Some(&layer.pw_b), &mut dbuf2);
            silu_in_place(&mut dbuf2);
            rms_norm_block(&mut dbuf2, &layer.norm);
            for t in length..t_len {
                for v in &mut dbuf2[t * dw..(t + 1) * dw] {
                    *v = 0.0;
                }
            }
        }
        let mut log_durations = vec![0f32; t_len];
        for t in 0..t_len {
            let mut o = [0f32; 1];
            matvec_into(&dbuf2[t * dw..(t + 1) * dw], &self.dur_out_w, 1, dw, &mut o);
            log_durations[t] = if t < length { o[0] + self.dur_out_b[0] } else { 0.0 };
        }

        (encoded, log_durations)
    }
}

// ---------------------------------------------------------------------------
// Decoder (causal GQA blocks + feature head; KV cache keeps last window-1 keys)
// ---------------------------------------------------------------------------

/// KV cache for one decoder block: post-RoPE keys/values stored token-major
/// [hist][kv_heads][head_dim], capped at `cap` entries (oldest evicted).
pub(crate) struct KvCache {
    k: Vec<f32>, // len = cap * kv_heads * head_dim
    v: Vec<f32>,
    hist: usize,
}

impl KvCache {
    fn new(kv_heads: usize, head_dim: usize, cap: usize) -> Self {
        Self {
            k: vec![0f32; kv_heads * cap * head_dim],
            v: vec![0f32; kv_heads * cap * head_dim],
            hist: 0,
        }
    }
}

pub struct Decoder {
    phase_proj_w: Vec<f32>, // [width, phase_dim] (empty when phoneme_phase=false)
    phase_dim: usize,
    blocks: Vec<AttnWeights>,
    feature_norm: Vec<f32>, // [width]
    head_w: Vec<f32>,       // [feature_dim, width]
    head_b: Vec<f32>,       // [feature_dim]
    feat_mean: Vec<f32>,    // [feature_dim]
    feat_std: Vec<f32>,     // [feature_dim]
    width: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: usize,
    kv_cap: usize,
    /// number of values per output feature frame (schema: cep18 + pitch + corr in v1)
    pub feature_dim: usize,
    rope_freqs: Vec<f32>,
}

impl Decoder {
    pub fn from_onnx_bytes(buf: &[u8], cfg: &crate::onnx_loader::ModelConfig) -> Result<Self, ModelLoadError> {
        let mut g = crate::onnx_loader::parse_onnx_graph(buf)?;

        // input projection: /phase_projection/MatMul is [phase_dim, width]
        let (phase_proj_w, pp_out, pp_in) = if cfg.phoneme_phase {
            let (w, o, i) = w_mat_dims(&mut g, "/phase_projection/MatMul")?;
            (w, o, i)
        } else {
            (Vec::new(), cfg.width, 0)
        };
        let width = pp_out;

        // discover decoder blocks
        let mut n_blocks = 0usize;
        while g.node(&format!("/decoder.{}/attention/q/MatMul", n_blocks)).is_some() {
            n_blocks += 1;
        }
        if n_blocks == 0 {
            return Err(val_err("decoder graph has no decoder.* attention blocks"));
        }
        if n_blocks != cfg.decoder_layers {
            eprintln!(
                "[pivo-studio] note: metadata decoder_layers={} but decoder has {} — using {}",
                cfg.decoder_layers, n_blocks, n_blocks
            );
        }
        let heads = cfg.heads.max(1);
        let head_dim = width / heads;
        let rope_freqs_v = rope_freqs(head_dim);
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut kv_heads = cfg.kv_heads;
        for i in 0..n_blocks {
            let scope = format!("decoder.{}", i);
            let aw = AttnWeights::load(&mut g, &scope, width)?;
            kv_heads = aw.kv_out / head_dim;
            blocks.push(aw);
        }

        let feature_norm = take_w(&mut g, "acoustic.feature_norm.weight", &[width])?;
        let (head_w, feature_dim, _) = w_mat_dims(&mut g, "/feature_head/MatMul")?;
        let head_b = take_w(&mut g, "acoustic.feature_head.bias", &[feature_dim])?;
        let feat_mean = take_w(&mut g, "acoustic.feature_mean", &[feature_dim])?;
        let feat_std = take_w(&mut g, "acoustic.feature_std", &[feature_dim])?;

        let window = cfg.attention_window.max(2);
        Ok(Self {
            phase_proj_w,
            phase_dim: pp_in,
            blocks,
            feature_norm,
            head_w,
            head_b,
            feat_mean,
            feat_std,
            width,
            heads,
            kv_heads: kv_heads.max(1),
            head_dim,
            window,
            kv_cap: window - 1,
            feature_dim,
            rope_freqs: rope_freqs_v,
        })
    }

    /// expanded: [F, expanded_dim] (encoded + optional phase), position: absolute offset.
    /// Returns features [F, feature_dim]; updates the KV caches (last kv_cap entries).
    pub(crate) fn run(
        &self,
        expanded: &[f32],
        expanded_dim: usize,
        frames: usize,
        position: usize,
        caches: &mut [KvCache],
    ) -> Vec<f32> {
        let width = self.width;
        let heads = self.heads;
        let kv_heads = self.kv_heads;
        let head_dim = self.head_dim;
        let kv_dim = kv_heads * head_dim;
        let group = heads / kv_heads.max(1);
        let attn_scale = (head_dim as f32).sqrt();
        let q_positions: Vec<i64> = (0..frames as i64).map(|t| t + position as i64).collect();
        let (cos, sin) = rope_tables(&q_positions, &self.rope_freqs);

        // input = expanded[:, :width] + phase_proj(expanded[:, width:])
        let mut x = vec![0f32; frames * width];
        for t in 0..frames {
            let erow = &expanded[t * expanded_dim..t * expanded_dim + width];
            let row = &mut x[t * width..(t + 1) * width];
            row.copy_from_slice(erow);
            if self.phase_dim > 0 {
                let phase = &expanded[t * expanded_dim + width..t * expanded_dim + width + self.phase_dim];
                let mut proj = vec![0f32; width];
                matvec_into(phase, &self.phase_proj_w, width, self.phase_dim, &mut proj);
                for i in 0..width {
                    row[i] += proj[i];
                }
            }
        }

        let mut scratch = vec![0f32; frames * width];
        let mut scores = vec![0f32; self.kv_cap + frames + 8];

        for (bi, blk) in self.blocks.iter().enumerate() {
            let cache = &mut caches[bi];
            for t in 0..frames {
                rms_norm_into(&x[t * width..(t + 1) * width], &blk.norm1, &mut scratch[t * width..(t + 1) * width]);
            }
            let mut q = vec![0f32; frames * blk.q_out];
            let mut k_new = vec![0f32; frames * blk.kv_out];
            let mut v_new = vec![0f32; frames * blk.kv_out];
            for t in 0..frames {
                let xr = &scratch[t * width..(t + 1) * width];
                matvec_into(xr, &blk.q_w, blk.q_out, width, &mut q[t * blk.q_out..(t + 1) * blk.q_out]);
                matvec_into(xr, &blk.k_w, blk.kv_out, width, &mut k_new[t * kv_dim..(t + 1) * kv_dim]);
                matvec_into(xr, &blk.v_w, blk.kv_out, width, &mut v_new[t * kv_dim..(t + 1) * kv_dim]);
            }
            rms_norm_block(&mut q, &blk.q_norm);
            rms_norm_block(&mut k_new, &blk.k_norm);
            if blk.rope {
                apply_rope(&mut q, heads, frames, head_dim, &cos, &sin);
                apply_rope(&mut k_new, kv_heads, frames, head_dim, &cos, &sin);
            }

            let old_hist = cache.hist;
            let total = old_hist + frames;

            // scores: q[t] over keys j in 0..total; key abs pos = position + j - old_hist
            // valid iff 0 <= q_pos[t] - k_pos[j] < window
            let mut attn = vec![0f32; frames * blk.q_out];
            for t in 0..frames {
                let q_pos = position + t;
                for h in 0..heads {
                    let kvh = h / group;
                    let qrow = &q[t * blk.q_out + h * head_dim..t * blk.q_out + h * head_dim + head_dim];
                    for j in 0..total {
                        let k_pos = position + j - old_hist;
                        let diff = q_pos as i64 - k_pos as i64;
                        if diff >= 0 && diff < self.window as i64 {
                            let krow: &[f32] = if j < old_hist {
                                &cache.k[(j * kv_heads + kvh) * head_dim..(j * kv_heads + kvh) * head_dim + head_dim]
                            } else {
                                let jj = j - old_hist;
                                &k_new[jj * kv_dim + kvh * head_dim..jj * kv_dim + kvh * head_dim + head_dim]
                            };
                            scores[j] = dot(qrow, krow) / attn_scale;
                        } else {
                            scores[j] = NEG_MASK;
                        }
                    }
                    softmax_row(&mut scores[..total]);
                    let orow = &mut attn[t * blk.q_out + h * head_dim..t * blk.q_out + h * head_dim + head_dim];
                    for vi in 0..head_dim {
                        let mut acc = 0f32;
                        for j in 0..total {
                            let s = scores[j];
                            if s == 0.0 {
                                continue;
                            }
                            let vv = if j < old_hist {
                                cache.v[(j * kv_heads + kvh) * head_dim + vi]
                            } else {
                                let jj = j - old_hist;
                                v_new[jj * kv_dim + kvh * head_dim + vi]
                            };
                            acc += s * vv;
                        }
                        orow[vi] = acc;
                    }
                }
            }

            // append to cache; keep last kv_cap entries of (old ++ new)
            let cap = self.kv_cap;
            let keep = total.min(cap);
            let drop = total - keep;
            {
                let entry = kv_heads * head_dim;
                if drop > 0 && old_hist > drop {
                    cache.k.copy_within(drop * entry..old_hist * entry, 0);
                    cache.v.copy_within(drop * entry..old_hist * entry, 0);
                }
                let start_jj = drop.saturating_sub(old_hist);
                for jj in start_jj..frames {
                    let dst = old_hist - drop + jj;
                    for kvh in 0..kv_heads {
                        cache.k[(dst * kv_heads + kvh) * head_dim..(dst * kv_heads + kvh) * head_dim + head_dim]
                            .copy_from_slice(&k_new[jj * kv_dim + kvh * head_dim..jj * kv_dim + kvh * head_dim + head_dim]);
                        cache.v[(dst * kv_heads + kvh) * head_dim..(dst * kv_heads + kvh) * head_dim + head_dim]
                            .copy_from_slice(&v_new[jj * kv_dim + kvh * head_dim..jj * kv_dim + kvh * head_dim + head_dim]);
                    }
                }
            }
            cache.hist = keep;

            for t in 0..frames {
                let mut orow = vec![0f32; width];
                matvec_into(&attn[t * blk.q_out..(t + 1) * blk.q_out], &blk.o_w, width, blk.q_out, &mut orow);
                for i in 0..width {
                    x[t * width + i] += orow[i];
                }
            }
            for t in 0..frames {
                let xr = &x[t * width..(t + 1) * width];
                rms_norm_into(xr, &blk.norm2, &mut scratch[t * width..(t + 1) * width]);
            }
            let mut gate = vec![0f32; frames * blk.ff];
            let mut up = vec![0f32; frames * blk.ff];
            for t in 0..frames {
                let xr = &scratch[t * width..(t + 1) * width];
                matvec_into(xr, &blk.gate_w, blk.ff, width, &mut gate[t * blk.ff..(t + 1) * blk.ff]);
                matvec_into(xr, &blk.up_w, blk.ff, width, &mut up[t * blk.ff..(t + 1) * blk.ff]);
            }
            for i in 0..gate.len() {
                gate[i] = gate[i] * sigmoid(gate[i]) * up[i];
            }
            for t in 0..frames {
                let mut down = vec![0f32; width];
                matvec_into(&gate[t * blk.ff..(t + 1) * blk.ff], &blk.down_w, width, blk.ff, &mut down);
                for i in 0..width {
                    x[t * width + i] += down[i];
                }
            }
        }

        // feature_norm → head → *std + mean
        let fd = self.feature_dim;
        let mut features = vec![0f32; frames * fd];
        for t in 0..frames {
            let mut normed = vec![0f32; width];
            rms_norm_into(&x[t * width..(t + 1) * width], &self.feature_norm, &mut normed);
            let mut head_out = vec![0f32; fd];
            matvec_into(&normed, &self.head_w, fd, width, &mut head_out);
            for i in 0..fd {
                features[t * fd + i] = (head_out[i] + self.head_b[i]) * self.feat_std[i] + self.feat_mean[i];
            }
        }
        features
    }
}

// ---------------------------------------------------------------------------
// Conditioning network: pitch embedding + dense → conv → dense
// ---------------------------------------------------------------------------

pub struct Conditioning {
    pitch_emb: Vec<f32>,  // [n_periods, pitch_dim]
    pitch_rows: usize,
    pitch_dim: usize,
    dense1_w: Vec<f32>,   // [hidden, ctx*(feature_dim+pitch_dim)]
    hidden: usize,
    conv_w: Vec<f32>,     // [c_out, hidden, ctx]
    c_out: usize,
    ctx: usize,
    dense2_w: Vec<f32>,   // [out, c_out]
    out_dim: usize,
    feature_dim: usize,
}

impl Conditioning {
    pub fn from_onnx_bytes(buf: &[u8], feature_dim: usize) -> Result<Self, ModelLoadError> {
        let mut g = crate::onnx_loader::parse_onnx_graph(buf)?;
        let (pitch_emb, pdims) = take_w_dims(&mut g, "pitch_embedding.weight")?;
        if pdims.len() != 2 {
            return Err(val_err("pitch_embedding.weight must be 2-D"));
        }
        let (pitch_rows, pitch_dim) = (pdims[0], pdims[1]);
        let (dense1_w, hidden, dense_in) = w_mat_dims(&mut g, "/dense1/MatMul")?;
        // conv kernel over time = ctx frames (v1: 3)
        let (conv_w, cdims) = take_w_dims(&mut g, "conv.weight")?;
        if cdims.len() != 3 {
            return Err(val_err("conv.weight must be 3-D"));
        }
        let (c_out, conv_in, ctx) = (cdims[0], cdims[1], cdims[2]);
        if conv_in != hidden {
            return Err(val_err(format!(
                "conditioning: dense1 out {} != conv in {}",
                hidden, conv_in
            )));
        }
        let expected_in = feature_dim + pitch_dim; // dense1 is per-context-frame
        if dense_in != expected_in {
            return Err(val_err(format!(
                "conditioning dense1 input {} != feature_dim({})+pitch_dim({}) = {}",
                dense_in, feature_dim, pitch_dim, expected_in
            )));
        }
        let (dense2_w, out_dim, _) = w_mat_dims(&mut g, "/dense2/MatMul")?;
        Ok(Self {
            pitch_emb,
            pitch_rows,
            pitch_dim,
            dense1_w,
            hidden,
            conv_w,
            c_out,
            ctx,
            dense2_w,
            out_dim,
            feature_dim,
        })
    }

    /// features: [ctx, feature_dim] flattened; periods: [ctx] in pitch range.
    /// Returns conditioning [out_dim].
    pub fn run(&self, features: &[f32], periods: &[i64]) -> Vec<f32> {
        let ctx = self.ctx;
        let row = self.feature_dim + self.pitch_dim;
        let mut inp = vec![0f32; ctx * row];
        for t in 0..ctx {
            inp[t * row..t * row + self.feature_dim]
                .copy_from_slice(&features[t * self.feature_dim..(t + 1) * self.feature_dim]);
            let p = (periods[t] - self.pitch_min() as i64).clamp(0, self.pitch_rows as i64 - 1) as usize;
            inp[t * row + self.feature_dim..(t + 1) * row]
                .copy_from_slice(&self.pitch_emb[p * self.pitch_dim..(p + 1) * self.pitch_dim]);
        }
        let mut d1 = vec![0f32; ctx * self.hidden];
        for t in 0..ctx {
            matvec_into(&inp[t * row..(t + 1) * row], &self.dense1_w, self.hidden, row, &mut d1[t * self.hidden..(t + 1) * self.hidden]);
        }
        for v in d1.iter_mut() {
            *v = v.tanh();
        }
        let mut c = vec![0f32; self.c_out];
        conv1d(&d1, ctx, self.hidden, self.c_out, ctx, 1, 0, &self.conv_w, None, &mut c);
        for v in c.iter_mut() {
            *v = v.tanh();
        }
        let mut out = vec![0f32; self.out_dim];
        matvec_into(&c, &self.dense2_w, self.out_dim, self.c_out, &mut out);
        for v in out.iter_mut() {
            *v = v.tanh();
        }
        out
    }

    /// Minimum pitch period indexable by the embedding (v1: 32).
    fn pitch_min(&self) -> usize {
        PITCH_MEMORY_LEN.saturating_sub(self.pitch_rows)
    }
}

// ---------------------------------------------------------------------------
// Vocoder step (FARGAN): N GRUs + GLU skips + pitch memory
// ---------------------------------------------------------------------------

pub struct VocoderStep {
    cond_gain_w: Vec<f32>, // [1, cond_dim]
    cond_gain_b: Vec<f32>, // [1]
    pitch_gain_w: Vec<f32>, // [n_grus+1, s0_dim]
    pitch_gain_b: Vec<f32>,
    frame_conv_w: Vec<f32>, // [s0_dim, 2*fs_len]
    frame_glu_w: Vec<f32>,  // [s0_dim, s0_dim]
    gru_in_w: Vec<Vec<f32>>,// per GRU [3h, in]
    gru_h_w: Vec<Vec<f32>>, // per GRU [3h, h]
    glu_w: Vec<Vec<f32>>,   // per GRU [h, h]
    gru_h: Vec<usize>,
    skip_dense_w: Vec<f32>, // [skip, sum(h_i)+s0+pm+samp]
    skip_glu_w: Vec<f32>,   // [skip, skip]
    signal_out_w: Vec<f32>, // [samp, skip]
    cond_dim: usize,
    s0_dim: usize,
    fs_len: usize,     // frame_state len = cond + pm + samp
    pm_len: usize,     // pitch memory slice len
    samp_len: usize,   // excitation samples per step
    n_grus: usize,
    skip_dim: usize,
}

pub struct VocoderState {
    pub memory: Vec<f32>,      // [PITCH_MEMORY_LEN]
    pub grus: Vec<Vec<f32>>,   // per-GRU hidden state
    pub frame_state: Vec<f32>, // [fs_len]
}

impl VocoderStep {
    pub fn from_onnx_bytes(buf: &[u8]) -> Result<Self, ModelLoadError> {
        let mut g = crate::onnx_loader::parse_onnx_graph(buf)?;
        let (cond_gain_w, cgd) = take_w_dims(&mut g, "cond_gain.weight")?;
        let cond_dim = cgd[1];
        let cond_gain_b = take_w(&mut g, "cond_gain.bias", &[1])?;
        let (pitch_gain_w, pgd) = take_w_dims(&mut g, "pitch_gain.weight")?;
        let n_grus = pgd[0] - 1;
        let s0_dim = pgd[1];
        let pitch_gain_b = take_w(&mut g, "pitch_gain.bias", &[pgd[0]])?;

        let (frame_conv_w, fc_out, fc_in) = w_mat_dims(&mut g, "/frame_conv/MatMul")?;
        if fc_out != s0_dim {
            return Err(val_err("frame_conv out != pitch_gain input"));
        }
        let fs_len = fc_in / 2;
        let frame_glu_w = w_mat(&mut g, "/frame_glu/gate/MatMul")?;

        let mut gru_in_w = Vec::with_capacity(n_grus);
        let mut gru_h_w = Vec::with_capacity(n_grus);
        let mut glu_w = Vec::with_capacity(n_grus);
        let mut gru_h = Vec::with_capacity(n_grus);
        for i in 1..=n_grus {
            let in_name = format!("/gru{}/MatMul", i);
            let h_name = format!("/gru{}/MatMul_1", i);
            let glu_name = format!("/glu{}/gate/MatMul", i);
            let (iw, iout, _iin) = w_mat_dims(&mut g, &in_name)?;
            let (hw, hout, _hin) = w_mat_dims(&mut g, &h_name)?;
            let h = hout / 3;
            if iout != 3 * h {
                return Err(val_err(format!("gru{}: input proj {} != 3*{}", i, iout, h)));
            }
            gru_in_w.push(iw);
            gru_h_w.push(hw);
            glu_w.push(w_mat(&mut g, &glu_name)?);
            gru_h.push(h);
        }

        let (skip_dense_w, skip_dim, skip_in) = w_mat_dims(&mut g, "/skip_dense/MatMul")?;
        let skip_glu_w = w_mat(&mut g, "/skip_glu/gate/MatMul")?;
        let (signal_out_w, samp_len, _) = w_mat_dims(&mut g, "/signal_out/MatMul")?;

        // pitch-mem slice length: fs_len = cond + pm + samp
        // sanity: skip_in = s0 + sum(gru_h) + (pm-4) + samp  (pitch mid = pm[2..pm-2])
        let pm_len = fs_len - cond_dim - samp_len;
        let pm_mid = pm_len.saturating_sub(4);
        let expected_skip = s0_dim + gru_h.iter().sum::<usize>() + pm_mid + samp_len;
        if skip_in != expected_skip {
            return Err(val_err(format!(
                "vocoder skip_dense in {} != expected {} (s0={} grus={:?} pm_mid={} samp={})",
                skip_in, expected_skip, s0_dim, gru_h, pm_mid, samp_len
            )));
        }

        Ok(Self {
            cond_gain_w,
            cond_gain_b,
            pitch_gain_w,
            pitch_gain_b,
            frame_conv_w,
            frame_glu_w,
            gru_in_w,
            gru_h_w,
            glu_w,
            gru_h,
            skip_dense_w,
            skip_glu_w,
            signal_out_w,
            cond_dim,
            s0_dim,
            fs_len,
            pm_len,
            samp_len,
            n_grus,
            skip_dim,
        })
    }

    pub fn new_state(&self) -> VocoderState {
        VocoderState {
            memory: vec![0.0; PITCH_MEMORY_LEN],
            grus: self.gru_h.iter().map(|&h| vec![0f32; h]).collect(),
            frame_state: vec![0f32; self.fs_len],
        }
    }

    /// Samples produced per feature frame (hop_length).
    pub fn frame_samples(&self, subframes: usize) -> usize {
        self.samp_len * subframes
    }

    /// conditioning: [cond_dim], period within pitch range.
    /// Returns samp_len excitation samples; updates state.
    pub fn run(&self, conditioning: &[f32], period: i64, st: &mut VocoderState) -> Vec<f32> {
        // cond_gain = exp(cond @ W^T + b)
        let mut cg = 0f32;
        for i in 0..self.cond_dim {
            cg += conditioning[i] * self.cond_gain_w[i];
        }
        cg += self.cond_gain_b[0];
        let gain = cg.exp();

        // pitch memory slice: idx[i] = 256 - period + i - 2; wrap by period; clamp ≥0
        let mut pitch_mem = vec![0f32; self.pm_len];
        for i in 0..self.pm_len as i64 {
            let mut idx = PITCH_MEMORY_LEN as i64 - period + i - 2;
            if idx >= PITCH_MEMORY_LEN as i64 {
                idx -= period;
            }
            if idx < 0 {
                idx = 0;
            }
            pitch_mem[i as usize] = st.memory[idx as usize] / (gain + PITCH_GAIN_EPS);
        }
        clip_in_place(&mut pitch_mem, -1.0, 1.0);
        // last-samp slice of memory
        let mem_len = st.memory.len();
        let mut last_samp = vec![0f32; self.samp_len];
        for i in 0..self.samp_len {
            last_samp[i] = st.memory[mem_len - self.samp_len + i] / (gain + PITCH_GAIN_EPS);
        }
        clip_in_place(&mut last_samp, -1.0, 1.0);

        // frame_state_out = concat(cond, pitch_mem, last_samp)
        let mut fs_out = vec![0f32; self.fs_len];
        fs_out[..self.cond_dim].copy_from_slice(conditioning);
        fs_out[self.cond_dim..self.cond_dim + self.pm_len].copy_from_slice(&pitch_mem);
        fs_out[self.cond_dim + self.pm_len..].copy_from_slice(&last_samp);

        // frame_conv over concat(frame_state, frame_state_out) → s0 → tanh
        let mut fs_cat = vec![0f32; 2 * self.fs_len];
        fs_cat[..self.fs_len].copy_from_slice(&st.frame_state);
        fs_cat[self.fs_len..].copy_from_slice(&fs_out);
        st.frame_state = fs_out;
        let mut fc = vec![0f32; self.s0_dim];
        matvec_into(&fs_cat, &self.frame_conv_w, self.s0_dim, 2 * self.fs_len, &mut fc);
        for v in fc.iter_mut() {
            *v = v.tanh();
        }
        // frame_glu → clip ±1
        let mut gate = vec![0f32; self.s0_dim];
        matvec_into(&fc, &self.frame_glu_w, self.s0_dim, self.s0_dim, &mut gate);
        let mut s0 = vec![0f32; self.s0_dim];
        for i in 0..self.s0_dim {
            s0[i] = (fc[i] * sigmoid(gate[i])).clamp(-1.0, 1.0);
        }
        // pitch_gain: sigmoid(s0 @ W^T + b) → n_grus+1 gains
        let mut pg = vec![0f32; self.n_grus + 1];
        for j in 0..=self.n_grus {
            let mut acc = self.pitch_gain_b[j];
            let wrow = &self.pitch_gain_w[j * self.s0_dim..(j + 1) * self.s0_dim];
            for i in 0..self.s0_dim {
                acc += s0[i] * wrow[i];
            }
            pg[j] = sigmoid(acc);
        }

        fn gru_step(x_in: &[f32], xw: &[f32], h: &mut [f32], hw: &[f32], h_dim: usize) {
            let mut xp = vec![0f32; 3 * h_dim];
            matvec_into(x_in, xw, 3 * h_dim, x_in.len(), &mut xp);
            let mut hp = vec![0f32; 3 * h_dim];
            matvec_into(h, hw, 3 * h_dim, h_dim, &mut hp);
            for i in 0..h_dim {
                let r = sigmoid(xp[i] + hp[i]);
                let z = sigmoid(xp[h_dim + i] + hp[h_dim + i]);
                let n = (xp[2 * h_dim + i] + r * hp[2 * h_dim + i]).tanh();
                h[i] = z * h[i] + (1.0 - z) * n;
            }
        }

        // GRU chain: g_i input = [s_i-1, pg[i-1]*pitch_mid, last_samp]
        // pitch_mid = pitch_mem[2 .. 2+pm_mid], pm_mid = pm_len - 4 (v1: 44-4=40)
        let pm_mid = self.pm_len.saturating_sub(4);
        let mut s_prev = s0.clone();
        let mut s_list: Vec<Vec<f32>> = Vec::with_capacity(self.n_grus);
        for i in 0..self.n_grus {
            let h = self.gru_h[i];
            let in_dim = s_prev.len() + pm_mid + self.samp_len;
            let mut gin = vec![0f32; in_dim];
            gin[..s_prev.len()].copy_from_slice(&s_prev);
            for j in 0..pm_mid {
                gin[s_prev.len() + j] = pg[i] * pitch_mem[2 + j];
            }
            gin[s_prev.len() + pm_mid..].copy_from_slice(&last_samp);
            gru_step(&gin, &self.gru_in_w[i], &mut st.grus[i], &self.gru_h_w[i], h);
            // glu_i: clip(h) * sigmoid(W·clip(h)) → clip
            let mut gc = vec![0f32; h];
            for j in 0..h {
                gc[j] = st.grus[i][j].clamp(-1.0, 1.0);
            }
            let mut ggate = vec![0f32; h];
            matvec_into(&gc, &self.glu_w[i], h, h, &mut ggate);
            let mut s = vec![0f32; h];
            for j in 0..h {
                s[j] = (gc[j] * sigmoid(ggate[j])).clamp(-1.0, 1.0);
            }
            s_prev = s.clone();
            s_list.push(s);
        }

        // skip_dense input = [s_1..s_n, s0, pg[n]*pitch_mid, last_samp]
        let mut skip_in = Vec::with_capacity(
            s_list.iter().map(|s| s.len()).sum::<usize>() + self.s0_dim + pm_mid + self.samp_len,
        );
        for s in &s_list {
            skip_in.extend_from_slice(s);
        }
        skip_in.extend_from_slice(&s0);
        for j in 0..pm_mid {
            skip_in.push(pg[self.n_grus] * pitch_mem[2 + j]);
        }
        skip_in.extend_from_slice(&last_samp);

        let mut sd = vec![0f32; self.skip_dim];
        matvec_into(&skip_in, &self.skip_dense_w, self.skip_dim, skip_in.len(), &mut sd);
        for v in sd.iter_mut() {
            *v = v.tanh().clamp(-1.0, 1.0);
        }
        let mut sgate = vec![0f32; self.skip_dim];
        matvec_into(&sd, &self.skip_glu_w, self.skip_dim, self.skip_dim, &mut sgate);
        let mut sg = vec![0f32; self.skip_dim];
        for i in 0..self.skip_dim {
            sg[i] = sd[i] * sigmoid(sgate[i]);
        }
        let mut sig = vec![0f32; self.samp_len];
        matvec_into(&sg, &self.signal_out_w, self.samp_len, self.skip_dim, &mut sig);

        let mut excitation = vec![0f32; self.samp_len];
        for i in 0..self.samp_len {
            excitation[i] = sig[i].tanh() * gain;
        }
        // memory_out = memory[samp..] + excitation
        st.memory.copy_within(self.samp_len.., 0);
        let ml = st.memory.len();
        st.memory[ml - self.samp_len..].copy_from_slice(&excitation);

        excitation
    }
}

// ---------------------------------------------------------------------------
// Phase features (phoneme timing) — only used when config.phoneme_phase
// ---------------------------------------------------------------------------

/// Emit 4 phase features per output frame.
/// `durations`: integer frame count per token. Returns [total_frames*4].
pub fn phoneme_phase_features(durations: &[i64], max_token_frames: usize) -> Vec<f32> {
    let total: usize = durations.iter().map(|&d| d.max(0) as usize).sum();
    let mut out = vec![0f32; total * 4];
    let mut frame = 0usize;
    let mut start = 0usize;
    let log1p_max = (max_token_frames as f64 + 1.0).ln() as f32;
    for &d in durations {
        let d = d.max(0) as usize;
        for _ in 0..d {
            let frac = (frame - start) as f32 + 0.5;
            let fraction = frac / d as f32;
            let base = frame * 4;
            out[base] = 2.0 * fraction - 1.0;
            out[base + 1] = (std::f32::consts::PI * fraction).sin();
            out[base + 2] = (std::f32::consts::PI * fraction).cos();
            out[base + 3] = ((d as f32) + 1.0).ln() / log1p_max;
            frame += 1;
        }
        start += d;
    }
    out
}

// ---------------------------------------------------------------------------
// Pitch periods + de-emphasis
// ---------------------------------------------------------------------------

/// Convert pitch feature value → FARGAN period index within [pitch_min, pitch_max].
pub fn pitch_period(feat: f32, pitch_min: i64, pitch_max: i64) -> i64 {
    let v = 256.0f32 / 2f32.powf(feat.clamp(-10.0, 10.0) + 1.5);
    (v.round() as i64).clamp(pitch_min, pitch_max)
}

/// Apply de-emphasis IIR: out[i] = s[i] + preemph * state.
#[inline]
pub fn deemphasis_chunk(audio: &[f32], preemph: f32, state: &mut f32, out: &mut [f32]) {
    for (i, &s) in audio.iter().enumerate() {
        *state = s + preemph * *state;
        out[i] = *state;
    }
}

// ---------------------------------------------------------------------------
// Full synthesizer
// ---------------------------------------------------------------------------

pub struct PivoModel {
    pub metadata: ModelMetadata,
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub conditioning: Conditioning,
    pub vocoder: VocoderStep,
    pub n_voices: usize,
    pub vocab_size: usize,
    pub pitch_min: i64,
    pub pitch_max: i64,
    /// samples per synthesized frame = subframes * samp_len (== config.hop_length)
    pub frame_len: usize,
    /// conditioning output / subframe dimension
    pub subframes: usize,
    /// feature_dim + phase_dim
    pub expanded_dim: usize,
    /// index of the pitch feature inside the feature vector (schema-derived)
    pub pitch_idx: usize,
}

impl PivoModel {
    /// Load from a verified model directory (manifest + onnx assets).
    pub fn load_dir(dir: impl AsRef<std::path::Path>) -> Result<Self, ModelLoadError> {
        let (root, manifest) = crate::onnx_loader::verify_model_dir(&dir)?;
        let comps = crate::onnx_loader::resolve_components(
            manifest.files.iter().map(|f| f.path.clone()),
        )?;
        let mut files = HashMap::new();
        for f in &manifest.files {
            files.insert(f.path.clone(), std::fs::read(root.join(&f.path))?);
        }
        Self::from_files(&files, &comps)
    }

    /// Load all weights from an in-memory file map + resolved components.
    pub fn from_files(
        files: &HashMap<String, Vec<u8>>,
        comps: &crate::onnx_loader::ModelComponents,
    ) -> Result<Self, ModelLoadError> {
        let get = |p: &str| -> Result<&[u8], ModelLoadError> {
            files.get(p).map(|v| v.as_slice()).ok_or_else(|| {
                ModelLoadError::Validation(format!("model missing asset {}", p))
            })
        };
        let metadata: ModelMetadata = serde_json::from_slice(get(&comps.metadata)?)?;
        crate::onnx_loader::parse_format_version(&metadata.format, "pivo-onnx-")
            .map_err(|_| ModelLoadError::Validation(format!(
                "unsupported export format '{}'", metadata.format
            )))?;
        if !metadata.trained {
            return Err(ModelLoadError::Validation(
                "this export is untrained; supply trained checkpoints before synthesis".into(),
            ));
        }
        if !metadata.feature_schema.is_empty() && !metadata.feature_schema.contains("fargan") {
            eprintln!(
                "[pivo-studio] note: unknown feature_schema '{}' — assuming pitch at index feature_dim-2",
                metadata.feature_schema
            );
        }
        let cfg = &metadata.config;
        let encoder = Encoder::from_onnx_bytes(get(&comps.encoder)?, cfg)?;
        let decoder = Decoder::from_onnx_bytes(get(&comps.decoder)?, cfg)?;
        let conditioning = Conditioning::from_onnx_bytes(get(&comps.conditioning)?, decoder.feature_dim)?;
        let vocoder = VocoderStep::from_onnx_bytes(get(&comps.vocoder)?)?;

        // subframes = cond_dim / 80? no: cond_dim == out_dim/subframes.
        // v1: conditioning out=320, vocoder consumes 80 per subframe → 4 subframes.
        // Derive: subframes = out_dim / (fs_len - pm - samp + ...) — simplest:
        // out_dim = n_subframes * cond_dim_sub where cond input per step is dense2_out/n.
        // Graph-wise: vocoder splits conditioning into cond_dim chunks.
        let subframes = conditioning.out_dim / vocoder.cond_dim;
        if conditioning.out_dim % vocoder.cond_dim != 0 || subframes == 0 {
            return Err(val_err(format!(
                "conditioning out {} not divisible by vocoder cond dim {}",
                conditioning.out_dim, vocoder.cond_dim
            )));
        }
        let frame_len = vocoder.samp_len * subframes;
        if frame_len != cfg.hop_length as usize {
            eprintln!(
                "[pivo-studio] note: hop_length={} but vocoder produces {} samples/frame — using {}",
                cfg.hop_length, frame_len, frame_len
            );
        }
        let pitch_max = (PITCH_MEMORY_LEN - 1) as i64;
        let pitch_min = (PITCH_MEMORY_LEN - conditioning.pitch_rows) as i64;
        let expanded_dim = encoder.width + if cfg.phoneme_phase { 4 } else { 0 };
        // pitch feature index from schema (v1 schema ends with ... pitch, pitch_corr)
        let pitch_idx = decoder.feature_dim.saturating_sub(2);

        Ok(Self {
            metadata: metadata.clone(),
            encoder,
            decoder,
            conditioning,
            vocoder,
            n_voices: metadata.voices.len().max(1),
            vocab_size: metadata.symbols.len().max(1),
            pitch_min,
            pitch_max,
            frame_len,
            subframes,
            expanded_dim,
            pitch_idx,
        })
    }

    /// Load all weights directly from an unpacked `.pivomodel` bundle map.
    pub fn load_bundle_files(files: &HashMap<String, Vec<u8>>) -> Result<Self, ModelLoadError> {
        let comps = crate::onnx_loader::resolve_components(files.keys().cloned())?;
        Self::from_files(files, &comps)
    }

    /// Load from a directory that has already been verified.
    pub fn load_assets(root: &std::path::Path) -> Result<Self, ModelLoadError> {
        let manifest: crate::onnx_loader::ModelManifest = serde_json::from_slice(
            &std::fs::read(root.join("model_manifest.json"))?,
        )?;
        let comps = crate::onnx_loader::resolve_components(
            manifest.files.iter().map(|f| f.path.clone()),
        )?;
        let mut files = HashMap::new();
        for f in &manifest.files {
            files.insert(f.path.clone(), std::fs::read(root.join(&f.path))?);
        }
        Self::from_files(&files, &comps)
    }

    pub fn sample_rate(&self) -> u32 {
        self.metadata.config.sample_rate
    }

    /// Voice name → index (metadata-driven; falls back to Pi/Vo order).
    pub fn voice_id(&self, voice: &str) -> Option<i64> {
        if let Some(&id) = self.metadata.voices.get(voice) {
            return Some(id as i64);
        }
        // case-insensitive / tolerant match
        self.metadata
            .voices
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(voice))
            .map(|(_, &v)| v as i64)
    }

    pub fn voice_names(&self) -> Vec<String> {
        let mut v: Vec<(String, i64)> = self
            .metadata
            .voices
            .iter()
            .map(|(k, &i)| (k.clone(), i))
            .collect();
        v.sort_by_key(|(_, i)| *i);
        v.into_iter().map(|(k, _)| k).collect()
    }

    /// Map a frontend ARPAbet symbol to a model token id using the model's
    /// own symbol table (vocab can change between exports).
    pub fn tokenize_symbols(&self, symbols: &[String]) -> Result<Vec<i64>, crate::text::FrontendError> {
        let map: HashMap<&str, i64> = self
            .metadata
            .symbols
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i as i64))
            .collect();
        let mut ids = Vec::with_capacity(symbols.len());
        for s in symbols {
            match map.get(s.as_str()) {
                Some(&id) => ids.push(id),
                None => {
                    return Err(crate::text::FrontendError::InvalidInput(format!(
                        "phoneme {:?} not in model symbol table",
                        s
                    )))
                }
            }
        }
        Ok(ids)
    }

    /// Encode token ids → (expanded [n_frames*expanded_dim], n_frames) honoring speed.
    fn encode_and_expand(
        &self,
        token_ids: &[i64],
        voice_id: i64,
        speed: f32,
    ) -> Result<(Vec<f32>, usize), crate::text::FrontendError> {
        let c = &self.metadata.config;
        let (encoded, log_durations) = self.encoder.run(token_ids, voice_id, token_ids.len());
        let max_tok = c.max_token_frames as f32;
        let log1p_max = (max_tok + 1.0).ln();
        let mut durations = vec![0i64; token_ids.len()];
        for (i, &tok) in token_ids.iter().enumerate() {
            let ld = log_durations[i].clamp(0.0, log1p_max);
            let mut d = (ld.exp() - 1.0).max(0.0) / speed;
            d = d.round().clamp(0.0, max_tok);
            if tok >= 10 {
                d = d.max(1.0);
            }
            if tok <= 2 {
                d = 0.0;
            }
            durations[i] = d as i64;
        }
        let n_frames: usize = durations.iter().map(|&d| d.max(0) as usize).sum();
        if n_frames == 0 {
            return Err(crate::text::FrontendError::InvalidInput(
                "duration predictor produced no audio frames".into(),
            ));
        }
        if n_frames > c.max_utterance_frames as usize {
            return Err(crate::text::FrontendError::InvalidInput(format!(
                "utterance exceeds {} frames",
                c.max_utterance_frames
            )));
        }
        let width = self.encoder.width;
        let mut expanded = vec![0f32; n_frames * self.expanded_dim];
        let mut dst = 0usize;
        for (t, &d) in durations.iter().enumerate() {
            for _ in 0..d.max(0) {
                expanded[dst * self.expanded_dim..dst * self.expanded_dim + width]
                    .copy_from_slice(&encoded[t * width..(t + 1) * width]);
                dst += 1;
            }
        }
        if self.metadata.config.phoneme_phase {
            let phase = phoneme_phase_features(&durations, c.max_token_frames as usize);
            for f in 0..n_frames {
                expanded[f * self.expanded_dim + width..(f + 1) * self.expanded_dim]
                    .copy_from_slice(&phase[f * 4..(f + 1) * 4]);
            }
        }
        Ok((expanded, n_frames))
    }

    /// Decode all feature frames at once (tests + non-streaming paths).
    /// expanded: [n_frames*expanded_dim]. Returns [n_frames*feature_dim].
    pub fn decode_all_features(&self, expanded: &[f32], n_frames: usize) -> Vec<f32> {
        let n_blocks = self.decoder.blocks.len();
        let mut kv: Vec<KvCache> = (0..n_blocks)
            .map(|_| {
                KvCache::new(
                    self.decoder.kv_heads,
                    self.decoder.head_dim,
                    self.decoder.kv_cap,
                )
            })
            .collect();
        let mut out = Vec::with_capacity(n_frames * self.decoder.feature_dim);
        let mut pos = 0usize;
        while pos < n_frames {
            let end = (pos + 16).min(n_frames);
            let feats = self.decoder.run(
                &expanded[pos * self.expanded_dim..end * self.expanded_dim],
                self.expanded_dim,
                end - pos,
                pos,
                &mut kv,
            );
            out.extend_from_slice(&feats);
            pos = end;
        }
        out
    }

    /// Create a lazy per-clause synthesizer: `next_frame()` decodes on demand and
    /// yields `frame_len` samples at a time (hop_length, 10 ms in v1).
    pub fn clause_stream(
        self: &Arc<Self>,
        token_ids: &[i64],
        voice_id: i64,
        speed: f32,
    ) -> Result<ClauseSynth, crate::text::FrontendError> {
        let (expanded, n_frames) = self.encode_and_expand(token_ids, voice_id, speed)?;
        Ok(ClauseSynth::new(self.clone(), expanded, n_frames))
    }

    /// Build a `ClauseSynth` directly from a pre-expanded buffer (tests).
    pub fn clause_stream_from_expanded(
        self: &Arc<Self>,
        expanded: Vec<f32>,
        n_frames: usize,
    ) -> Result<ClauseSynth, crate::text::FrontendError> {
        Ok(ClauseSynth::new(self.clone(), expanded, n_frames))
    }

    /// Synthesize a whole clause to a single Vec<f32>.
    pub fn synthesize_clause(
        self: &Arc<Self>,
        token_ids: &[i64],
        voice_id: i64,
        speed: f32,
    ) -> Result<Vec<f32>, crate::text::FrontendError> {
        let mut audio = Vec::new();
        let mut s = self.clause_stream(token_ids, voice_id, speed)?;
        while let Some(f) = s.next_frame() {
            audio.extend_from_slice(&f);
        }
        Ok(audio)
    }

    /// Stream synthesized audio: calls `on_chunk` with each frame slice.
    /// Convenience wrapper over `clause_stream`.
    pub fn synthesize_stream<F: FnMut(&[f32])>(
        self: &Arc<Self>,
        token_ids: &[i64],
        voice_id: i64,
        speed: f32,
        mut on_chunk: F,
    ) -> Result<usize, crate::text::FrontendError> {
        let mut s = self.clause_stream(token_ids, voice_id, speed)?;
        let mut produced = 0usize;
        while let Some(frame) = s.next_frame() {
            produced += frame.len();
            on_chunk(&frame);
        }
        Ok(produced)
    }

    /// Full pipeline: normalize+phonemize text → stream audio chunks.
    pub fn synthesize_text_stream<F: FnMut(&[f32])>(
        self: &Arc<Self>,
        text_input: &str,
        voice: &str,
        speed: f32,
        strict: bool,
        overrides: Option<&HashMap<String, Vec<String>>>,
        mut on_chunk: F,
    ) -> Result<usize, crate::text::FrontendError> {
        let vid = self.voice_id(voice).ok_or_else(|| {
            crate::text::FrontendError::InvalidInput(format!("unknown voice '{}'", voice))
        })?;
        let front = text::phonemize(text_input, strict, overrides)?;
        let ids = self.tokenize_symbols(&front.symbols)?;
        self.synthesize_stream(&ids, vid, speed, |f| on_chunk(f))
    }
}

/// Lazily decoded single-clause synthesizer: yields `frame_len`-sample frames on demand.
/// First frame requires encoding the clause plus ~2 decoder frames of lookahead.
pub struct ClauseSynth {
    model: Arc<PivoModel>,
    expanded: Vec<f32>,
    n_frames: usize,
    pos: usize,
    chunk_frames: usize,
    kv: Vec<KvCache>,
    voc: VocoderState,
    deemph: f32,
    queue: VecDeque<Vec<f32>>,
    prev: Option<Vec<f32>>,
    ctx: usize,
}

impl ClauseSynth {
    fn new(model: Arc<PivoModel>, expanded: Vec<f32>, n_frames: usize) -> Self {
        let n_blocks = model.decoder.blocks.len();
        Self {
            kv: (0..n_blocks)
                .map(|_| {
                    KvCache::new(
                        model.decoder.kv_heads,
                        model.decoder.head_dim,
                        model.decoder.kv_cap,
                    )
                })
                .collect(),
            voc: model.vocoder.new_state(),
            ctx: model.conditioning.ctx,
            model,
            expanded,
            n_frames,
            pos: 0,
            chunk_frames: 16,
            deemph: 0.0,
            queue: VecDeque::new(),
            prev: None,
        }
    }

    /// Next frame of `frame_len` samples, or None when the clause is finished.
    pub fn next_frame(&mut self) -> Option<Vec<f32>> {
        let ctx = self.ctx;
        let fd = self.model.decoder.feature_dim;
        let pitch_idx = self.model.pitch_idx;
        let cfg_preemph = self.model.metadata.preemphasis;
        // v1 ctx=3 → need current + 1 lookahead: [prev, cur, next].
        // General: emit when queue has 1 + `after` frames (after = ctx-1-(ctx-1)/2),
        // or at end-of-frames when any context entry remains.
        let before = (ctx - 1) / 2;
        let after = ctx - 1 - before;
        loop {
            if self.queue.len() >= 1 + after || (self.pos >= self.n_frames && !self.queue.is_empty()) {
                let current = self.queue.pop_front().unwrap();
                // window[before] = current; earlier entries = prev (repeated),
                // later entries from queue (clamped to current at stream end).
                let prev = self.prev.clone().unwrap_or_else(|| current.clone());
                let mut window: Vec<Vec<f32>> = vec![prev; ctx];
                window[before] = current.clone();
                for i in 0..after {
                    window[before + 1 + i] = self
                        .queue
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| current.clone());
                }
                let mut context = vec![0f32; ctx * fd];
                for (i, f) in window.iter().enumerate() {
                    context[i * fd..(i + 1) * fd].copy_from_slice(f);
                }
                let periods: Vec<i64> = (0..ctx)
                    .map(|i| {
                        pitch_period(
                            context[i * fd + pitch_idx],
                            self.model.pitch_min,
                            self.model.pitch_max,
                        )
                    })
                    .collect();
                let cond = self.model.conditioning.run(&context, &periods);
                let mid = ctx / 2;
                let mut frame_pcm = vec![0f32; self.model.frame_len];
                for sub in 0..self.model.subframes {
                    let c = &cond[sub * self.model.vocoder.cond_dim..(sub + 1) * self.model.vocoder.cond_dim];
                    let exc = self.model.vocoder.run(c, periods[mid], &mut self.voc);
                    let mut pcm = vec![0f32; exc.len()];
                    deemphasis_chunk(&exc, cfg_preemph, &mut self.deemph, &mut pcm);
                    frame_pcm[sub * self.model.vocoder.samp_len
                        ..sub * self.model.vocoder.samp_len + exc.len()]
                        .copy_from_slice(&pcm);
                }
                self.prev = Some(current);
                return Some(frame_pcm);
            }
            if self.pos >= self.n_frames {
                return None;
            }
            // First decode call: pull only 2 frames so the very first vocoder output
            // arrives after minimal work; afterwards decode 16-frame chunks.
            let chunk = if self.pos == 0 {
                2.min(self.n_frames)
            } else {
                self.chunk_frames
            };
            let end = (self.pos + chunk).min(self.n_frames);
            let feats = self.model.decoder.run(
                &self.expanded[self.pos * self.model.expanded_dim..end * self.model.expanded_dim],
                self.model.expanded_dim,
                end - self.pos,
                self.pos,
                &mut self.kv,
            );
            for t in 0..(end - self.pos) {
                queue_push(&mut self.queue, feats[t * fd..(t + 1) * fd].to_vec());
            }
            self.pos = end;
        }
    }

    pub fn expanded_frames(&self) -> usize {
        self.n_frames
    }
}

fn queue_push(q: &mut VecDeque<Vec<f32>>, v: Vec<f32>) {
    q.push_back(v);
}
