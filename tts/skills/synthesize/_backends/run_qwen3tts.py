"""Qwen3-TTS — the control path decides the checkpoint:
   --speaker  -> CustomVoice   --instruct -> VoiceDesign   --ref -> Base (clone)
Note: no seed, sampled decoding, so output is not reproducible; --max-seconds is the
only reliable length bound (max_new_tokens caps a single chunk, not the total).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import parse
import torch, numpy as np, soundfile as sf
from qwen_tts import Qwen3TTSModel

a = parse("qwen3tts")
if a.ref:
    ck, mode = "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "clone"
elif a.instruct:
    ck, mode = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "design"
elif a.speaker:
    ck, mode = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "custom"
else:
    sys.exit("qwen3tts needs one of --speaker / --instruct / --ref")

m = Qwen3TTSModel.from_pretrained(ck, device_map="cuda:0", dtype=torch.bfloat16,
                                  attn_implementation="sdpa")  # no flash-attn in this venv
if mode == "clone":
    if not a.ref_text:
        sys.exit("qwen3tts --ref needs --ref-text (or use x_vector_only_mode in your own call)")
    wavs, sr = m.generate_voice_clone(text=a.text, language=a.lang,
                                      ref_audio=a.ref, ref_text=a.ref_text)
elif mode == "design":
    wavs, sr = m.generate_voice_design(text=a.text, language=a.lang, instruct=a.instruct)
else:
    kw = {"instruct": a.instruct} if a.instruct else {}
    wavs, sr = m.generate_custom_voice(text=a.text, language=a.lang, speaker=a.speaker, **kw)

w = wavs[0]
if a.max_seconds:
    w = w[: int(a.max_seconds * sr)]
sf.write(a.out, w, sr)
print(f"qwen3tts[{mode}]: {len(w)/sr:.2f}s @ {sr}Hz rms={np.sqrt((w**2).mean()):.3f} -> {a.out}")
