"""IndexTTS-2.5 — zero-shot clone; the only engine needing no reference transcript."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import parse, VALKA
import soundfile as sf
from indextts.infer_v2_5 import IndexTTS2

a = parse("indextts25")
if not a.ref:
    sys.exit("indextts25 needs --ref (it is clone-only; there is no built-in voice)")
ck = f"{VALKA}/IndexTTS-2.5/checkpoints"
tts = IndexTTS2(cfg_path=f"{ck}/config.yaml", model_dir=ck, use_bf16=True)
kw = {}
if a.max_seconds:                      # duration_factor is 0.5-2.0; >1 slows down
    kw["duration_factor"] = 1.0
tts.infer(spk_audio_prompt=a.ref, text=a.text, lang=a.lang[:2].upper(),
          output_path=a.out, **kw)
w, sr = sf.read(a.out)
print(f"indextts25: {len(w)/sr:.2f}s @ {sr}Hz -> {a.out}")
