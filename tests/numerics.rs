//! Numerical equivalence tests: compares every stage (encoder, durations,
//! decoder, conditioning, vocoder, end-to-end audio) against reference outputs
//! produced by the Python ONNX Runtime implementation.
//!
//! Run the reference generator first:
//!   python3 tests/gen_reference.py --model-dir ../huggingface --out /tmp/pivo_ref
//! Then:
//!   PIVO_REF_DIR=/tmp/pivo_ref cargo test --test numerics -- --nocapture

use std::path::PathBuf;

fn ref_dir() -> Option<PathBuf> {
    let d = match std::env::var("PIVO_REF_DIR") {
        Ok(d) => d,
        Err(_) if std::env::var_os("PIVO_REQUIRE_REFERENCE").is_some() => panic!("PIVO_REF_DIR required"),
        Err(_) => return None,
    };
    let p = PathBuf::from(d);
    if p.join("tokens.bin").exists() {
        Some(p)
    } else {
        assert!(std::env::var_os("PIVO_REQUIRE_REFERENCE").is_none(), "reference tokens.bin missing");
        None
    }
}


fn reference_case(rd: &PathBuf) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(rd.join("case.json")).expect("missing case manifest")).expect("invalid case manifest")
}

fn model_dir() -> PathBuf {
    std::env::var("PIVO_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../huggingface"))
}

/// Read a little-endian f32 binary file (as dumped by np.tofile).
fn read_f32_bin(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).expect(&format!("missing {}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "malformed f32 tensor");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_i64_bin(path: &PathBuf) -> Vec<i64> {
    let bytes = std::fs::read(path).expect(&format!("missing {}", path.display()));
    assert_eq!(bytes.len() % 8, 0, "malformed i64 tensor");
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([
            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
        ]))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "reference length mismatch");
    assert!(a.iter().chain(b.iter()).all(|v| v.is_finite()), "non-finite reference or output");
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn encoder_matches_reference() {
    let Some(rd) = ref_dir() else {
        eprintln!("skipping: PIVO_REF_DIR not set");
        return;
    };
    let tokens = read_i64_bin(&rd.join("tokens.bin"));
    let ref_encoded = read_f32_bin(&rd.join("encoded.bin"));
    let ref_logdur = read_f32_bin(&rd.join("log_durations.bin"));

    let model = pivo_runner::model::PivoModel::load_dir(model_dir())
        .expect("failed to load model");
    let case = reference_case(&rd);
    let voice_id = model.voice_id(case["voice"].as_str().expect("voice")).expect("unknown voice");
    let (enc, log_dur) = model.encoder.run(&tokens, voice_id, tokens.len());

    let d_enc = max_abs_diff(&enc, &ref_encoded);
    let d_dur = max_abs_diff(&log_dur, &ref_logdur);
    eprintln!("encoder  max|Δ| = {:.3e}", d_enc);
    eprintln!("log_dur  max|Δ| = {:.3e}", d_dur);
    assert!(d_enc < 1e-4, "encoder diverged: max|Δ|={:.3e}", d_enc);
    assert!(d_dur < 1e-4, "duration diverged: max|Δ|={:.3e}", d_dur);
}

#[test]
fn decoder_matches_reference() {
    let Some(rd) = ref_dir() else {
        eprintln!("skipping: PIVO_REF_DIR not set");
        return;
    };
    let expanded = read_f32_bin(&rd.join("expanded.bin"));
    let ref_feats = read_f32_bin(&rd.join("features.bin"));

    let model = std::sync::Arc::new(
        pivo_runner::model::PivoModel::load_dir(model_dir()).expect("failed to load model"),
    );
    // derive dims from the loaded model rather than hardcoding v1 values
    let fd = model.decoder.feature_dim;
    let expanded_dim = model.expanded_dim;
    let n_frames = expanded.len() / expanded_dim;
    assert_eq!(expanded.len() % expanded_dim, 0);
    assert_eq!(ref_feats.len(), n_frames * fd);

    let feats_v = model.decode_all_features(&expanded, n_frames);
    let feats = feats_v;
    let d = max_abs_diff(&feats, &ref_feats);
    eprintln!("decoder  max|Δ| = {:.3e} ({} frames)", d, n_frames);
    assert!(d < 1e-3, "decoder diverged: max|Δ|={:.3e}", d);
}

#[test]
fn end_to_end_audio() {
    let Some(rd) = ref_dir() else {
        eprintln!("skipping: PIVO_REF_DIR not set");
        return;
    };
    let ref_audio = read_f32_bin(&rd.join("audio.bin"));
    let engine = pivo_runner::Engine::load_dir(model_dir()).expect("load failed");
    let opts = pivo_runner::SynthOptions {
        voice: reference_case(&rd)["voice"].as_str().expect("voice").into(),
        speed: reference_case(&rd)["speed"].as_f64().expect("speed") as f32,
        strict: true,
        overrides: Default::default(),
    };
    let audio = engine
        .synthesize_clause(reference_case(&rd)["text"].as_str().expect("text"), &opts)
        .expect("synthesis failed");
    eprintln!("ref {} samples, rust {} samples", ref_audio.len(), audio.len());
    assert_eq!(audio.len(), ref_audio.len(), "sample count mismatch");
    let d = max_abs_diff(&audio, &ref_audio);
    let rms_ref = (ref_audio.iter().map(|v| v * v).sum::<f32>() / ref_audio.len() as f32).sqrt();
    eprintln!("end-to-end audio max|Δ| = {:.3e} (ref RMS {:.4})", d, rms_ref);
    assert!(d < 0.02, "audio diverged: max|Δ|={:.3e}", d);
}
