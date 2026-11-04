#!/usr/bin/env python3
"""Generate reference tensors from the Python ONNX Runtime PiVo engine for
numerical-equivalence testing of pivo_runner.

Usage:
  python3 gen_reference.py --model-dir /path/to/huggingface --out /tmp/pivo_ref
"""
import argparse
import numpy as np
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(Path(__file__).parent.parent.parent / "huggingface"))
    ap.add_argument("--out", default="/tmp/pivo_ref")
    ap.add_argument("--text", default="NASA is ready for the next mission.")
    ap.add_argument("--voice", default="Pi")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()
    if not np.isfinite(args.speed) or not 0.5 <= args.speed <= 2.0: ap.error("speed must be finite and between 0.5 and 2.0")

    sys.path.insert(0, args.model_dir)
    from pivo.text import phonemize, VOICE_IDS
    from pivo.phase import phoneme_phase_features
    from pivo.audio import deemphasis
    import onnxruntime as ort

    onnx_dir = Path(args.model_dir) / "onnx"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import json
    (out / "case.json").write_text(json.dumps({"text": args.text, "voice": args.voice, "speed": args.speed}))

    front = phonemize(args.text)
    tokens = np.array([front.token_ids], dtype=np.int64)
    tokens[0].tofile(out / "tokens.bin")

    enc = ort.InferenceSession(str(onnx_dir / "encoder.onnx"))
    encoded, log_dur = enc.run(None, {
        "tokens": tokens,
        "voice": np.array([VOICE_IDS[args.voice]], dtype=np.int64),
        "lengths": np.array([len(front.token_ids)], dtype=np.int64),
    })
    encoded[0].astype(np.float32).tofile(out / "encoded.bin")
    log_dur[0].astype(np.float32).tofile(out / "log_durations.bin")

    cfg = __import__("json").loads((onnx_dir / "metadata.json").read_text())["config"]
    durations = (np.rint(np.expm1(log_dur.clip(0, np.log1p(cfg["max_token_frames"]))) / args.speed)
                 .clip(0, cfg["max_token_frames"]).astype(np.int64))
    durations = np.maximum(durations, (tokens >= 10).astype(np.int64))
    durations[tokens <= 2] = 0
    expanded = np.repeat(encoded[0], durations[0], axis=0)[None]
    phase = phoneme_phase_features(durations[0], cfg["max_token_frames"])[None]
    expanded = np.concatenate((expanded, phase), axis=-1).astype(np.float32)
    expanded[0].tofile(out / "expanded.bin")
    durations[0].astype(np.int64).tofile(out / "durations.bin")

    dec = ort.InferenceSession(str(onnx_dir / "decoder.onnx"))
    cache = {n: np.zeros((1, cfg["kv_heads"], 0, cfg["width"] // cfg["heads"]), dtype=np.float32)
             for n in ["key0", "value0", "key1", "value1"]}
    feats = []
    for pos in range(0, expanded.shape[1], 16):
        r = dec.run(None, {
            "expanded": expanded[:, pos:pos + 16],
            "position": np.array(pos, dtype=np.int64),
            **cache})
        cache = dict(zip(["key0", "value0", "key1", "value1"], r[1:]))
        feats.extend(r[0][0])
    feats = np.array(feats, dtype=np.float32)
    feats.tofile(out / "features.bin")

    cond = ort.InferenceSession(str(onnx_dir / "conditioning.onnx"))
    voc = ort.InferenceSession(str(onnx_dir / "vocoder_step.onnx"))
    state = {n: np.zeros((1, w), dtype=np.float32)
             for n, w in zip(["memory", "gru1", "gru2", "gru3", "frame_state"],
                             (256, 160, 128, 128, 164))}
    audio = []
    prev = None
    de = 0.0
    for i, cur in enumerate(feats):
        following = feats[i + 1] if i + 1 < len(feats) else cur
        previous = prev if prev is not None else cur
        context = np.stack([previous, cur, following])[None].astype(np.float32)
        periods = (np.rint(256 / np.power(2.0, np.clip(context[..., 18], -10, 10) + 1.5))
                   .clip(32, 255).astype(np.int64))
        c = cond.run(None, {"features": context, "periods": periods})[0]
        frame = []
        for sf in range(4):
            r = voc.run(None, {
                "conditioning": c[:, 0, sf * 80:(sf + 1) * 80],
                "period": periods[:, 1],
                **state})
            state = dict(zip(["memory", "gru1", "gru2", "gru3", "frame_state"], r[1:]))
            frame.append(r[0][0])
        pcm, de = deemphasis(np.concatenate(frame), de)
        audio.append(pcm)
        prev = cur
    audio = np.concatenate(audio).astype(np.float32)
    audio.tofile(out / "audio.bin")
    print(f"wrote {out} — {len(feats)} frames, {len(audio)} samples")


if __name__ == "__main__":
    main()
