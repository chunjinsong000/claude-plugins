"""FireRedTTS3 — Base clones from ref+transcript, Instruct does voice design."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import parse, VALKA
import torchaudio

a = parse("fireredtts3")
MODELS = f"{VALKA}/FireRedTTS3/pretrained_models"
if a.ref:
    if not a.ref_text:
        sys.exit("fireredtts3 --ref needs --ref-text")
    from fireredtts3.core import FireRedTTS3
    tts = FireRedTTS3(MODELS, use_wetext=True, use_llm_tn=False)
    pa, psr = torchaudio.load(a.ref)
    gen, sr = tts.generate(language=None, prompt_text=a.ref_text, prompt_audio=pa,
                           prompt_audio_sr=psr, text=a.text, do_tn=True)
elif a.instruct:
    from fireredtts3.core import FireRedTTS3Instruct
    inst = FireRedTTS3Instruct(MODELS, use_wetext=True, use_llm_tn=False)
    gen, sr, tags = inst.generate_voice_design(instruction=a.instruct, text=a.text)
    print(f"resolved attributes: {tags}")
else:
    sys.exit("fireredtts3 needs --ref (+--ref-text) for cloning or --instruct for voice design")
if a.max_seconds:
    gen = gen[..., : int(a.max_seconds * sr)]
torchaudio.save(a.out, gen.cpu(), sr)
print(f"fireredtts3: {gen.shape[-1]/sr:.2f}s @ {sr}Hz -> {a.out}")
