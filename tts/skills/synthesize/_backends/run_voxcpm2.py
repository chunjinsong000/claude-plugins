"""VoxCPM2 — 48 kHz. Voice design goes in parentheses at the head of the text."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import parse
import numpy as np, soundfile as sf
from voxcpm import VoxCPM

a = parse("voxcpm2")
m = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
sr = m.tts_model.sample_rate
text = a.text
kw = dict(cfg_value=2.0, inference_timesteps=10, seed=a.seed)
if a.instruct:
    # wording is fragile: without "normal volume" some descriptions collapse to a whisper
    ins = a.instruct if "volume" in a.instruct.lower() else a.instruct + ", normal volume"
    text = f"({ins}){text}"
if a.ref:
    if not a.ref_text:
        sys.exit("voxcpm2 --ref needs --ref-text (prompt_wav_path/prompt_text must be paired)")
    kw.update(prompt_wav_path=a.ref, prompt_text=a.ref_text, reference_wav_path=a.ref)
w = m.generate(text=text, **kw)
if a.max_seconds:
    w = w[: int(a.max_seconds * sr)]
sf.write(a.out, w, sr)
print(f"voxcpm2: {len(w)/sr:.2f}s @ {sr}Hz rms={np.sqrt((w**2).mean()):.3f} -> {a.out}")
