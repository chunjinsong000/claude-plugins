"""Fish Audio S2 Pro — clone only, needs ref + transcript.

The documented flow is three CLI calls that each reload the 4B model; this drives the
same internals in one process. generate_long does bool(prompt_tokens), so the prompt
args must be lists, never a bare tensor.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import parse, VALKA
from pathlib import Path
import torch, numpy as np, soundfile as sf
from fish_speech.models.text2semantic.inference import (
    init_model, load_codec_model, encode_audio, decode_to_audio, generate_long,
)

a = parse("fishs2pro")
if not a.ref or not a.ref_text:
    sys.exit("fishs2pro needs --ref and --ref-text")
CKPT = Path(f"{VALKA}/FishS2/checkpoints/s2-pro")
dev, prec = "cuda", torch.bfloat16
model, decode_one = init_model(CKPT, dev, prec, compile=False)
with torch.device(dev):
    model.setup_caches(max_batch_size=1, max_seq_len=model.config.max_seq_len,
                       dtype=next(model.parameters()).dtype)
torch.cuda.synchronize()
codec = load_codec_model(CKPT / "codec.pth", dev, prec)
torch.manual_seed(a.seed); torch.cuda.manual_seed(a.seed)

tokens = encode_audio(a.ref, codec, dev).cpu()
codes = []
for resp in generate_long(model=model, device=dev, decode_one_token=decode_one,
                          text=a.text, num_samples=1, max_new_tokens=0, top_p=0.9,
                          top_k=30, temperature=1.0, compile=False, iterative_prompt=True,
                          chunk_length=300, prompt_text=[a.ref_text], prompt_tokens=[tokens]):
    if resp.action == "sample":
        codes.append(resp.codes)
    elif resp.action == "next":
        break
audio = decode_to_audio(torch.cat(codes, dim=1).to(dev), codec).cpu().float().numpy()
if a.max_seconds:
    audio = audio[: int(a.max_seconds * codec.sample_rate)]
sf.write(a.out, audio, codec.sample_rate)
print(f"fishs2pro: {len(audio)/codec.sample_rate:.2f}s @ {codec.sample_rate}Hz "
      f"rms={np.sqrt((audio**2).mean()):.3f} -> {a.out}")
