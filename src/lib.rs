//! `pivo_runner` — a pure-Rust, zero-dependency inference engine for PiVo-1.4M TTS.
//!
//! - Loads the same ONNX weight assets as the Python reference (no ONNX Runtime needed).
//! - Single-file `.pivomodel` bundles for mobile/embedded deployment.
//! - Clause-level + 10 ms frame streaming for sub-50 ms time-to-first-audio on
//!   arbitrarily long text.
//! - `#[no_mangle]` C ABI (`pivo_ffi.h`) ready for JNI (Android), Swift (iOS),
//!   WebAssembly and plain C/C++.

pub mod cpu;
pub mod ffi;
pub mod model;
pub mod onnx_loader;
pub mod server;
pub mod studio;
pub mod text;

use std::collections::HashMap;
use std::sync::Arc;

use model::{ClauseSynth, PivoModel};
use text::FrontendError;

pub use onnx_loader::{pack_model_bundle, unpack_model_bundle, verify_model_dir};
pub use text::{phonemize, split_text_into_clauses, FrontendResult};

/// Maximum characters fed to the encoder in a single clause (the Python engine allows
/// 2,000 chars / 512 tokens per utterance; our clause splitter keeps well below that).
pub const MAX_CLAUSE_CHARS: usize = 900;
/// Absolute safety cap on a single input text.
pub const MAX_INPUT_CHARS: usize = 100_000;

#[derive(Debug)]
pub enum EngineError {
    Frontend(FrontendError),
    Load(onnx_loader::ModelLoadError),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Frontend(e) => write!(f, "{}", e),
            EngineError::Load(e) => write!(f, "{}", e),
        }
    }
}
impl std::error::Error for EngineError {}

impl From<FrontendError> for EngineError {
    fn from(e: FrontendError) -> Self {
        EngineError::Frontend(e)
    }
}
impl From<onnx_loader::ModelLoadError> for EngineError {
    fn from(e: onnx_loader::ModelLoadError) -> Self {
        EngineError::Load(e)
    }
}
impl From<std::io::Error> for EngineError {
    fn from(e: std::io::Error) -> Self {
        EngineError::Load(onnx_loader::ModelLoadError::Io(e))
    }
}

#[derive(Debug, Clone)]
pub struct SynthOptions {
    pub voice: String,
    pub speed: f32,
    /// strict=true reproduces the PiVo v1 contract (OOV words error out).
    /// strict=false spells OOV words letter-by-letter.
    pub strict: bool,
    pub overrides: HashMap<String, Vec<String>>,
}

impl Default for SynthOptions {
    fn default() -> Self {
        Self {
            voice: "Pi".to_string(),
            speed: 1.0,
            strict: true,
            overrides: HashMap::new(),
        }
    }
}

/// Reusable loaded model. Cheap to clone the handle (weights are `Arc`-shared internally).
#[derive(Clone)]
pub struct Engine {
    model: Arc<PivoModel>,
    manifest_name: Option<String>,
}

impl Engine {
    /// Load a verified PiVo model directory (checks `model_manifest.json` hashes).
    pub fn load_dir(dir: impl AsRef<std::path::Path>) -> Result<Self, EngineError> {
        // Eagerly parse the CMU lexicon once so the first request's TTFA stays low.
        let _ = text::lexicon();
        let _ = cpu::backend_name();
        Ok(Self {
            model: Arc::new(PivoModel::load_dir(dir)?),
            manifest_name: None,
        })
    }

    /// Load a `.pivomodel` bundle (single-file format; path or in-memory bytes).
    pub fn load_bundle(bundle: &[u8]) -> Result<Self, EngineError> {
        let _ = text::lexicon();
        let _ = cpu::backend_name();
        let files = unpack_model_bundle(bundle)?;
        // Capture the manifest's display name if present.
        let manifest_name = files
            .get("model_manifest.json")
            .and_then(|b| serde_json::from_slice::<crate::onnx_loader::ModelManifest>(b).ok())
            .and_then(|m| {
                if !m.name.is_empty() {
                    Some(m.name)
                } else if !m.model_name.is_empty() {
                    Some(m.model_name)
                } else {
                    None
                }
            });
        Ok(Self {
            model: Arc::new(PivoModel::load_bundle_files(&files)?),
            manifest_name,
        })
    }

    pub fn load_bundle_file(path: impl AsRef<std::path::Path>) -> Result<Self, EngineError> {
        let bytes = std::fs::read(path)?;
        Self::load_bundle(&bytes)
    }

    /// Load either a model directory or a `.pivomodel` bundle from a path.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, EngineError> {
        let p = path.as_ref();
        if p.is_file() {
            Self::load_bundle_file(p)
        } else {
            Self::load_dir(p)
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.model.sample_rate()
    }

    /// Synthesize a full clause/utterance (must contain speech; ≤ max_utterance_frames).
    pub fn synthesize_clause(
        &self,
        text: &str,
        opts: &SynthOptions,
    ) -> Result<Vec<f32>, EngineError> {
        let vid = self.model.voice_id(&opts.voice).ok_or_else(|| {
            FrontendError::InvalidInput(format!(
                "unknown voice '{}' (available: {})",
                opts.voice,
                self.model.voice_names().join(", ")
            ))
        })?;
        if !(0.5..=2.0).contains(&opts.speed) || !opts.speed.is_finite() {
            return Err(FrontendError::InvalidInput(
                "speed must be finite and between 0.5 and 2.0".into(),
            )
            .into());
        }
        let front = text::phonemize(text, opts.strict, Some(&opts.overrides))?;
        let ids = self.model.tokenize_symbols(&front.symbols)?;
        Ok(self.model.synthesize_clause(&ids, vid, opts.speed)?)
    }

    /// Synthesize arbitrary-length text (auto-split into clauses) to a full PCM buffer.
    pub fn synthesize(&self, text: &str, opts: &SynthOptions) -> Result<Vec<f32>, EngineError> {
        let mut audio = Vec::new();
        let mut stream = self.synthesize_stream(text, opts)?;
        while let Some(frame) = stream.next_frame()? {
            audio.extend_from_slice(&frame);
        }
        Ok(audio)
    }

    /// Create a pull-based streaming session over arbitrarily long text.
    /// Each `next_frame()` call returns the next 160 samples (10 ms) of audio.
    /// The first clause is kept short so time-to-first-audio is sub-50 ms.
    pub fn synthesize_stream(
        &self,
        text_input: &str,
        opts: &SynthOptions,
    ) -> Result<SpeechStream, EngineError> {
        let vid = self.model.voice_id(&opts.voice).ok_or_else(|| {
            FrontendError::InvalidInput(format!(
                "unknown voice '{}' (available: {})",
                opts.voice,
                self.model.voice_names().join(", ")
            ))
        })?;
        if !(0.5..=2.0).contains(&opts.speed) || !opts.speed.is_finite() {
            return Err(FrontendError::InvalidInput(
                "speed must be finite and between 0.5 and 2.0".into(),
            )
            .into());
        }
        if text_input.len() > MAX_INPUT_CHARS {
            return Err(FrontendError::InvalidInput(format!(
                "input exceeds {} characters",
                MAX_INPUT_CHARS
            ))
            .into());
        }
        let clauses = split_text_into_clauses(text_input, MAX_CLAUSE_CHARS);
        if clauses.is_empty() {
            return Err(FrontendError::InvalidInput("input must contain spoken text".into()).into());
        }
        // Phonemize clause 0 eagerly so OOV errors surface before streaming starts;
        // later clauses are phonemized lazily inside `next_frame` — this keeps TTFA
        // minimal on arbitrarily long input.
        let front0 = text::phonemize(&clauses[0], opts.strict, Some(&opts.overrides))?;
        let any_spoken = front0
            .symbols
            .iter()
            .any(|s| !text::SPECIAL.contains(&s.as_str()))
            || clauses.len() > 1; // later clauses may contain speech
        if !any_spoken {
            return Err(
                FrontendError::InvalidInput("input must contain spoken text".into()).into(),
            );
        }
        Ok(SpeechStream {
            model: self.model.clone(),
            clauses,
            front0: Some(front0),
            clause: 0,
            current: None,
            voice_id: vid,
            speed: opts.speed,
            strict: opts.strict,
            overrides: opts.overrides.clone(),
            done: false,
            err: None,
        })
    }

    /// PCM samples per output frame (hop_length: 160 in v1, read from the model).
    pub fn frame_len(&self) -> usize {
        self.model.frame_len
    }

    /// Voice names exposed by the loaded model's metadata.
    pub fn voices(&self) -> Vec<String> {
        self.model.voice_names()
    }

    /// Display name for the loaded model (manifest name > parameters.provisional_name > format).
    pub fn model_name(&self) -> String {
        let md = &self.model.metadata;
        self.manifest_name
            .as_deref()
            .or_else(|| {
                md.parameters
                    .as_ref()
                    .and_then(|p| (!p.provisional_name.is_empty()).then_some(p.provisional_name.as_str()))
            })
            .unwrap_or(&md.format)
            .to_string()
    }

    /// JSON descriptor for `/v1/voices` + `/health` + the C API.
    pub fn model_info(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.model_name(),
            "format": self.model.metadata.format,
            "sample_rate": self.sample_rate(),
            "frame_samples": self.frame_len(),
            "voices": self.voices(),
            "vocab_size": self.model.vocab_size,
        })
    }

    /// Phonemize text for debugging/inspection (`/v1/phonemize`).
    /// Token ids are mapped through the loaded model's own symbol table.
    pub fn phonemize(
        &self,
        text: &str,
        strict: bool,
        overrides: Option<&HashMap<String, Vec<String>>>,
    ) -> Result<FrontendResult, EngineError> {
        let mut front = text::phonemize(text, strict, overrides)?;
        front.token_ids = self.model.tokenize_symbols(&front.symbols)?;
        Ok(front)
    }

    /// Split text into the clause sequence the streaming path will consume.
    pub fn clauses_for(&self, text: &str) -> Vec<String> {
        split_text_into_clauses(text, MAX_CLAUSE_CHARS)
    }
}

/// Pull-based streaming synthesizer over one logical text (potentially many clauses).
/// Each `next_frame()` returns exactly 160 f32 samples (10 ms at 16 kHz), generated
/// lazily — audio for the first clause starts after encoding it plus ~2 decoder
/// frames of lookahead, giving sub-50 ms TTFA even on slow devices.
pub struct SpeechStream {
    model: Arc<PivoModel>,
    clauses: Vec<String>,
    front0: Option<FrontendResult>,
    clause: usize,
    current: Option<ClauseSynth>,
    voice_id: i64,
    speed: f32,
    strict: bool,
    overrides: HashMap<String, Vec<String>>,
    done: bool,
    err: Option<FrontendError>,
}

impl SpeechStream {
    /// Samples per output frame for this model (hop_length; 160 in v1).
    pub fn frame_len(&self) -> usize {
        self.model.frame_len
    }

    /// Produce the next frame (`frame_len` f32 samples, hop_length of the model),
    /// or `None` when finished.
    pub fn next_frame(&mut self) -> Result<Option<Vec<f32>>, EngineError> {
        if let Some(e) = self.err.take() {
            return Err(e.into());
        }
        loop {
            if let Some(cur) = self.current.as_mut() {
                if let Some(frame) = cur.next_frame() {
                    return Ok(Some(frame));
                }
                self.current = None;
            }
            if self.done || self.clause >= self.clauses.len() {
                self.done = true;
                return Ok(None);
            }
            let front = if self.clause == 0 {
                self.front0.take().unwrap()
            } else {
                match text::phonemize(&self.clauses[self.clause], self.strict, Some(&self.overrides))
                {
                    Ok(f) => f,
                    Err(e) => {
                        self.done = true;
                        self.err = Some(e);
                        return Err(self.err.take().unwrap().into());
                    }
                }
            };
            let has_speech = front
                .symbols
                .iter()
                .any(|s| !text::SPECIAL.contains(&s.as_str()));
            self.clause += 1;
            if !has_speech {
                continue;
            }
            let ids = match self.model.tokenize_symbols(&front.symbols) {
                Ok(ids) => ids,
                Err(e) => {
                    self.done = true;
                    self.err = Some(e);
                    return Err(self.err.take().unwrap().into());
                }
            };
            match self.model.clause_stream(&ids, self.voice_id, self.speed) {
                Ok(s) => self.current = Some(s),
                Err(e) => {
                    self.err = Some(e);
                    self.done = true;
                    return Err(self.err.take().unwrap().into());
                }
            }
        }
    }
}

/// Write PCM16 mono WAV bytes (16 kHz) for `audio` (float32, -1..1).
pub fn wav_bytes(audio: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = (audio.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in audio {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header() {
        let w = wav_bytes(&[0.0, 0.5, -0.5], 16000);
        assert_eq!(&w[..4], b"RIFF");
        assert_eq!(w.len(), 44 + 6);
    }
}
