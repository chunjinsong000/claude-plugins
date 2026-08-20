#!/usr/bin/env python3
"""Step 2 - turn dealer_phrases_filled.json into IndexTTS-2.5 prompt inputs.

Only `text` is spoken. Everything else in the source (style, the six axes, the
situation's description) is descriptive metadata, so it is converted here into the
engine's actual control signals:

    text            -> infer(text=...)                 the words
    style           -> emo_vector                      professional=calmer, friendly=warmer
    axes.emphasis   -> emo_vector                      main driver
    axes.sentence_type -> emo_vector + duration_factor
    axes.directness -> duration_factor
    axes.length     -> duration_factor + interval_silence
    axes.courtesy   -> emo_vector (slight warmth)
    situation       -> emo_vector                      wins celebrate, losses sympathise
    axes.addressing -> nothing acoustic (it is a property of the wording)

Emotion dims, order fixed by the model:
    [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
A live-dealer voice only ever wants happy / surprised / calm, plus a little
melancholic to sound sympathetic on a loss. angry / afraid / disgusted stay 0.

IMPORTANT: IndexTTS2.infer() does NOT normalise emo_vector - normalize_emo_vec is only
called from webui.py. So this script applies the model's own emo_bias and the
"sum <= 0.8" cap itself, and emits vectors that are safe to pass straight in.

  python make_tts_prompts.py [--out dealer_phrases_tts.json]
"""
import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "dealer_phrases_filled.json")
DST = os.path.join(HERE, "dealer_phrases_tts.json")

DIMS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
# straight from IndexTTS2.normalize_emo_vec - it de-emphasises dims that go strange
EMO_BIAS = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
EMO_SUM_CAP = 0.8

# --- situation -> emotional colour ----------------------------------------
# Situation colour has an additive part and a multiplicative one. The scale matters:
# the source data marks plenty of losing lines as emphasis="enthusiastic" (e.g. "Bust at
# 22!"), which means "forceful", not "cheerful". Adding melancholic on top of a happy
# base is not enough -- happy has to be damped, or a bust gets read gleefully.
# Each carries a name so downstream stages can key off the class without redefining the
# situation list (synth_10s.py compensates pace on celebrate lines only).
CELEBRATE = {"name": "celebrate", "add": {"happy": 0.30, "surprised": 0.10}}   # player won
SYMPATHY  = {"name": "sympathy", "add": {"melancholic": 0.12, "calm": 0.10},
             "scale": {"happy": 0.25}}                                         # player lost
WARM      = {"name": "warm", "add": {"happy": 0.18}}                           # greetings
URGENT    = {"name": "urgent", "add": {"surprised": 0.08}}                     # betting closing

SITUATION_COLOUR = {
    "player_turn/player_blackjack":        CELEBRATE,
    "resolution/player_wins":              CELEBRATE,
    "resolution/congratulate_winners":     CELEBRATE,
    "side_bets/perfect_pairs_win":         CELEBRATE,
    "side_bets/twentyone_plus_three_win":  CELEBRATE,
    "side_bets/insurance_wins":            CELEBRATE,
    "dealer_turn/dealer_busts":            CELEBRATE,   # dealer over 21 = players win
    "player_turn/player_busts":            SYMPATHY,
    "resolution/player_loses":             SYMPATHY,
    "side_bets/side_bet_loss":             SYMPATHY,
    "side_bets/insurance_loses":           SYMPATHY,
    "dealer_turn/dealer_blackjack":        SYMPATHY,     # dealer natural = players lose
    "dealing/good_luck":                   WARM,
    "general/dealer_introduction":         WARM,
    "general/dealer_farewell":             WARM,
    "general/welcome_player":              WARM,
    "betting/last_bets":                   URGENT,
    "betting/no_more_bets":                URGENT,
}

# --- axes -> emotion ------------------------------------------------------
EMPHASIS_BASE = {
    "enthusiastic": {"happy": 0.45, "surprised": 0.10},
    "calm":         {"calm": 0.50},
    "neutral":      {"calm": 0.18},
}
STYLE_MOD = {
    "friendly":     {"happy": 0.12},
    "professional": {"calm": 0.10},
}
SENTENCE_EMO = {
    "exclamatory": {"happy": 0.12, "surprised": 0.12},
    "question":    {"surprised": 0.08},
    "imperative":  {"calm": 0.05},
    "declarative": {},
}
COURTESY_MOD = {"happy": 0.04}

# --- axes -> pace ---------------------------------------------------------
# MEASURED on this checkpoint (same text, deterministic decode):
#   0.7 ->3.34s  0.8 ->4.14s  0.88->4.42s  0.95->4.28s  1.0 ->4.47s
#   1.05->4.64s  1.14->5.82s  1.3 ->6.33s  1.5 ->7.78s
# The 0.88-1.05 band is flat AND non-monotonic (0.88 came out longer than 0.95), so
# small offsets around 1.0 are meaningless. Slowing down is strong (1.14 = +30%),
# speeding up is weak (0.8 = only -7%). Hence: score the axes as integers, then
# quantise onto steps coarse enough to actually be audible.
# 0.72 was dropped: it read as rushed even after enthusiasm stopped buying a pace step,
# so the fastest the corpus goes is 0.85. Only score -3 is affected (96 of 2000 texts:
# 52 enthusiastic, 44 neutral, 0 calm) -- every other score keeps its step.
PACE_STEPS = {-3: 0.85, -2: 0.85, -1: 0.85, 0: 1.00, 1: 1.15, 2: 1.15, 3: 1.30}
LENGTH_PACE     = {"ultra_short": -1, "short": 0, "medium": 0, "long": 1}
DIRECTNESS_PACE = {"direct": -1, "invitation": 0, "suggestion": 1, "observation": 1}
SENTENCE_PACE   = {"imperative": -1, "exclamatory": -1, "question": 0, "declarative": 0}
# "enthusiastic" used to get -1 here, which stacked with sentence_type (exclamatory -1)
# and directness (direct -1) and dropped most of those lines onto the fastest step:
# 310 of 552 enthusiastic texts landed on 0.72. It read as rushed, so enthusiasm no longer
# buys a pace step -- it still ends up slightly quick via the other axes.
EMPHASIS_PACE   = {"enthusiastic": 0, "neutral": 0, "calm": 2}
URGENT_SITUATIONS = {"betting/last_bets", "betting/no_more_bets"}
URGENT_PACE = -2


def build_emo(style, axes, situation):
    acc = {d: 0.0 for d in DIMS}

    def add(mod):
        for k, v in mod.items():
            acc[k] += v

    add(EMPHASIS_BASE.get(axes.get("emphasis"), {}))
    add(STYLE_MOD.get(style, {}))
    add(SENTENCE_EMO.get(axes.get("sentence_type"), {}))
    if axes.get("courtesy"):
        add(COURTESY_MOD)

    colour = SITUATION_COLOUR.get(situation, {})
    for k, f in colour.get("scale", {}).items():   # damp before adding
        acc[k] *= f
    add(colour.get("add", {}))

    vec = [min(1.0, max(0.0, acc[d])) for d in DIMS]
    vec = [v * b for v, b in zip(vec, EMO_BIAS)]           # model's own de-emphasis
    s = sum(vec)
    if s > EMO_SUM_CAP:                                     # model's own hard cap
        vec = [v * EMO_SUM_CAP / s for v in vec]
    return [round(v, 4) for v in vec]


def build_duration(axes, situation):
    """Integer pace score -> a duration_factor step that is actually audible."""
    score = (LENGTH_PACE.get(axes.get("length"), 0)
             + DIRECTNESS_PACE.get(axes.get("directness"), 0)
             + SENTENCE_PACE.get(axes.get("sentence_type"), 0)
             + EMPHASIS_PACE.get(axes.get("emphasis"), 0)
             + (URGENT_PACE if situation in URGENT_SITUATIONS else 0))
    score = max(-3, min(3, score))
    return PACE_STEPS[score], score


def build_silence(axes, situation):
    if situation in URGENT_SITUATIONS:
        return 150
    if axes.get("length") == "long" or axes.get("emphasis") == "calm":
        return 250
    return 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=DST)
    a = ap.parse_args()

    d = json.load(open(a.src))
    items = []
    for pn, situs in d["phases"].items():
        for sn, s in situs.items():
            situation = f"{pn}/{sn}"
            for i, p in enumerate(s["phrases"]):
                axes = p.get("axes") or {}
                style = p.get("style")
                emo = build_emo(style, axes, situation)
                dur, pace_score = build_duration(axes, situation)
                kw = {
                    "duration_factor": dur,
                    "interval_silence": build_silence(axes, situation),
                    "text_normalization": True,
                }
                if any(v > 0 for v in emo):
                    # emo_alpha stays 1.0: the intended strength is already in the vector
                    kw = {"emo_vector": emo, "emo_alpha": 1.0, **kw}
                items.append({
                    "id": f"{pn}.{sn}.{i:04d}",
                    "phase": pn,
                    "situation": sn,
                    "text": p["text"],
                    "infer_kwargs": kw,
                    "source": {
                        "style": style,
                        "axes": axes,
                        "situation_description": s.get("description"),
                        "pace_score": pace_score,
                        "emotion_class": SITUATION_COLOUR.get(situation, {}).get("name", "neutral"),
                        **({"text_template": p["text_template"]} if "text_template" in p else {}),
                    },
                })

    out = {
        "metadata": {
            "engine": "IndexTTS-2.5",
            "generated_from": os.path.basename(a.src),
            "source_metadata": d["metadata"],
            "emo_dims": DIMS,
            "usage": ("tts.infer(spk_audio_prompt=REF, text=item['text'], lang='EN', "
                      "output_path=OUT, **item['infer_kwargs'])"),
            "constraints": {
                "emo_bias_applied": EMO_BIAS,
                "emo_sum_cap": EMO_SUM_CAP,
                "note": ("IndexTTS2.infer does NOT normalise emo_vector (normalize_emo_vec is "
                         "only called by webui.py), so the bias and the sum<=0.8 cap are "
                         "already applied here; pass emo_vector through unchanged."),
                "duration_factor_steps": sorted(set(PACE_STEPS.values())),
                "duration_factor_semantics": (">1.0 is slower. MEASURED: the 0.88-1.05 band is "
                    "flat and non-monotonic on this checkpoint, so only coarse steps are used; "
                    "slowing is strong (1.14=+30% length), speeding up is weak (0.8=-7%)."),
            },
            "mapping": {
                "text": "spoken content",
                "style": "emo_vector (professional=calmer, friendly=warmer)",
                "axes.emphasis": "emo_vector (main driver) + duration_factor",
                "axes.sentence_type": "emo_vector + duration_factor",
                "axes.directness": "duration_factor",
                "axes.length": "duration_factor + interval_silence",
                "axes.courtesy": "emo_vector (slight warmth)",
                "axes.addressing": "not acoustic - it is a property of the wording",
                "situation": "emo_vector (wins celebrate, losses sympathise, greetings warm)",
            },
            "count": len(items),
        },
        "items": items,
    }
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"{len(items)} items -> {a.out}")

    # summary of what the mapping actually produced
    import collections
    dur = collections.Counter(i["infer_kwargs"]["duration_factor"] for i in items)
    print(f"duration_factor: min {min(dur)} max {max(dur)} distinct {len(dur)}")
    sums = [round(sum(i['infer_kwargs'].get('emo_vector', [0])), 3) for i in items]
    print(f"emo sum: min {min(sums)} max {max(sums)} (cap {EMO_SUM_CAP})")
    over = [s for s in sums if s > EMO_SUM_CAP + 1e-6]
    print(f"vectors over cap: {len(over)}")
    no_emo = sum(1 for i in items if "emo_vector" not in i["infer_kwargs"])
    print(f"items with no emo_vector (neutral, uses reference voice as-is): {no_emo}")


if __name__ == "__main__":
    main()
