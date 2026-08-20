"""Shared argument parsing for the per-engine backends.

Only stdlib here: each backend runs under a different venv, so this file must
import cleanly on python 3.11 and 3.12 with no third-party packages.
"""
import argparse, os, sys

VALKA = os.environ.get("VALKA_ROOT", "/home/ubuntu/chunjin/project/valka-ai")

REPO = {
    "qwen3tts":    "Qwen3-TTS",
    "voxcpm2":     "VoxCPM2",
    "fireredtts3": "FireRedTTS3",
    "fishs2pro":   "FishS2",
    "indextts25":  "IndexTTS-2.5",
}


def parse(engine):
    p = argparse.ArgumentParser(prog=f"tts:{engine}")
    p.add_argument("--text", help="text to speak")
    p.add_argument("--text-file", help="file holding the text (alternative to --text)")
    p.add_argument("--out", required=True, help="output wav path")
    p.add_argument("--ref", help="reference audio for voice cloning")
    p.add_argument("--ref-text", default=None,
                   help="transcript of --ref; required by every engine except indextts25")
    p.add_argument("--instruct", default=None,
                   help="natural-language voice description (voice design)")
    p.add_argument("--speaker", default=None, help="qwen3tts CustomVoice speaker name")
    p.add_argument("--lang", default="English", help="language label where the engine takes one")
    p.add_argument("--seed", type=int, default=42, help="ignored by engines without a seed")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="hard-truncate the output; the only reliable bound for qwen3tts")
    a = p.parse_args()
    if not a.text and not a.text_file:
        p.error("one of --text / --text-file is required")
    if a.text_file:
        a.text = open(a.text_file).read().strip()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    return a


def report(a, wav, sr, engine):
    """One-line stdout summary; keeps every backend's output shape identical."""
    n = len(wav)
    print(f"{engine}: {n/sr:.2f}s @ {sr}Hz -> {a.out}", flush=True)
