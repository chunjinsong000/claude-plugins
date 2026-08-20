#!/usr/bin/env python3
"""Stage 2.5 - concatenate DIFFERENT lines into one clip instead of repeating one line.

Repeating a single line for 10 s sounds mechanical, and concatenating whatever shares a
situation is barely better: stage 1 expands each template into dozens of placeholder
siblings, so a naive concat gives "Unfortunately, 27 is a bust. Unfortunately, 28 is a
bust. ..." -- same wording, different number.

So each clip takes **at most one line per text_template**, which yields real variation:
"Unfortunately, 27 is a bust. That's over. 24. I'm afraid that's a bust. 28."

Lines are only combined when their control signal fits all of them, i.e. they share
(phase, situation, style, axes.emphasis). That keeps emotion_class, emphasis and style
constant; the residual spread inside a group comes only from sentence_type, and the
emitted vector is the group median.

  python make_mixed_prompts.py --target-words 34 --out dealer_phrases_mixed.json

Templates are reused ACROSS clips (with different placeholder values), never within one.
"""
import argparse, collections, json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_TTS = os.path.join(HERE, "dealer_phrases_tts.json")
SRC_FILLED = os.path.join(HERE, "dealer_phrases_filled.json")
DST = os.path.join(HERE, "dealer_phrases_mixed.json")

# IndexTTS keeps text in one segment below max_text_tokens_per_segment (default 120), and
# splitting wrecks the audio, so cap the concatenation well under that.
TOKEN_BUDGET_WORDS = 70


def template_index(filled):
    """id -> the template a line came from (its own text when it had no placeholders)."""
    out = {}
    for pn, situs in filled["phases"].items():
        for sn, s in situs.items():
            for i, p in enumerate(s["phrases"]):
                out[f"{pn}.{sn}.{i:04d}"] = p.get("text_template", p["text"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_TTS)
    ap.add_argument("--filled", default=SRC_FILLED)
    ap.add_argument("--out", default=DST)
    ap.add_argument("--target-words", type=int, default=34,
                    help="words per clip; ~34 reaches 10 s at the slower pace steps")
    ap.add_argument("--target-clips", type=int, default=0,
                    help="total clips to emit. Quotas are proportional to each group's "
                         "phrase count, which preserves the emotion mix. Without this the "
                         "count is driven by placeholder combinatorics instead, and neutral "
                         "situations (52 cards x 7 boxes x 18 totals) swamp everything: "
                         "measured 96%% neutral vs 58%% at phrase level.")
    ap.add_argument("--max-clips", type=int, default=0, help="0 = as many as the data allows")
    ap.add_argument("--no-reuse", action="store_true",
                    help="never use the same phrase in two clips. Every clip is then unique "
                         "text, and the count is capped by capacity (non-neutral situations "
                         "have very little). Use this when audio variety comes from many "
                         "reference voices rather than from many distinct texts.")
    a = ap.parse_args()

    tts = json.load(open(a.src))
    tmpl = template_index(json.load(open(a.filled)))

    groups = collections.defaultdict(list)
    for it in tts["items"]:
        key = (it["phase"], it["situation"], it["source"]["style"],
               it["source"]["axes"]["emphasis"])
        groups[key].append(it)

    # Clip quota per group: proportional to its DISTINCT TEMPLATE count, not its phrase
    # count. Phrase counts follow placeholder combinatorics, which is wildly uneven --
    # 99.1% of the cartesian capacity sits in neutral situations (a card+box+points line
    # has 6,552 combinations, "Winner!" has 1), so quotas by phrase count give 96% neutral
    # clips against 47% of the templates. Template count reflects how the phrase bank was
    # actually written, so that is the mix worth reproducing. The cost is that non-neutral
    # groups run out of distinct phrases and reuse them across clips (never within one).
    tmpl_count = {k: len({tmpl.get(x["id"], x["text"]) for x in v}) for k, v in groups.items()}
    quota = {}
    if a.target_clips:
        total = sum(tmpl_count.values())
        raw = {k: a.target_clips * tmpl_count[k] / total for k in groups}
        quota = {k: max(1, int(round(x))) for k, x in raw.items()}
        # trim/extend to hit the target exactly, largest groups absorbing the difference
        diff = a.target_clips - sum(quota.values())
        order = sorted(groups, key=lambda k: -tmpl_count[k])
        i = 0
        while diff != 0 and order:
            k = order[i % len(order)]
            if diff > 0:
                quota[k] += 1; diff -= 1
            elif quota[k] > 1:
                quota[k] -= 1; diff += 1
            i += 1

    items = []
    stats = collections.Counter()
    for key, members in sorted(groups.items()):
        # bucket by template so a clip can take one line from each
        by_tmpl = collections.defaultdict(list)
        for it in members:
            by_tmpl[tmpl.get(it["id"], it["text"])].append(it)
        tmpl_names = sorted(by_tmpl)
        cursor = {t: 0 for t in tmpl_names}          # rotate variants across clips
        if a.no_reuse:                               # each bucket becomes a consumable queue
            pool = {t: list(v) for t, v in by_tmpl.items()}
        # how many clips this group can fill, bounded by its largest template bucket so
        # every clip gets a fresh placeholder variant where possible
        n_clips = quota.get(key) or max(1, max(len(v) for v in by_tmpl.values()))
        t_start = 0
        for c in range(n_clips):
            chosen, words, used_tmpl = [], 0, set()
            # walk templates round-robin, starting at a rotating offset for variety
            order = tmpl_names[t_start:] + tmpl_names[:t_start]
            for t in order:
                if words >= a.target_words:
                    break
                if t in used_tmpl:
                    continue
                if a.no_reuse:
                    if not pool[t]:
                        continue                     # this template is exhausted
                    it = pool[t].pop(0)
                else:
                    bucket = by_tmpl[t]
                    it = bucket[cursor[t] % len(bucket)]
                    cursor[t] += 1
                used_tmpl.add(t)
                chosen.append(it)
                words += len(it["text"].split())
            fallback = None
            if a.no_reuse and words < a.target_words:
                # not enough unused phrases left in this group -- stop here rather than
                # reusing, and drop the partial clip if it is too short to be useful
                if words < a.target_words * 0.6 or not chosen:
                    break
                fallback = "short_no_reuse"
            elif words < a.target_words:
                # group has too few distinct templates -- reuse them, cycling variants
                fallback = "reused_templates"
                guard = 0
                while words < a.target_words and guard < 60:
                    for t in order:
                        if words >= a.target_words:
                            break
                        bucket = by_tmpl[t]
                        it = bucket[cursor[t] % len(bucket)]
                        cursor[t] += 1
                        chosen.append(it)
                        words += len(it["text"].split())
                    guard += 1
            t_start = (t_start + 1) % max(1, len(tmpl_names))

            if words > TOKEN_BUDGET_WORDS:                 # keep it inside one segment
                while len(chosen) > 1 and words > TOKEN_BUDGET_WORDS:
                    words -= len(chosen[-1]["text"].split())
                    chosen.pop()

            text = " ".join(x["text"] for x in chosen)
            evs = np.array([x["infer_kwargs"].get("emo_vector", [0.0] * 8) for x in chosen])
            dfs = collections.Counter(x["infer_kwargs"]["duration_factor"] for x in chosen)
            kw = {
                "emo_vector": [round(float(v), 4) for v in np.median(evs, axis=0)],
                "emo_alpha": 1.0,
                "duration_factor": dfs.most_common(1)[0][0],
                "text_normalization": True,
            }
            phase, situation, style, emphasis = key
            items.append({
                "id": f"{phase}.{situation}.{style}.{emphasis}.{c:04d}",
                "phase": phase, "situation": situation,
                "text": text,
                "infer_kwargs": kw,
                "source": {
                    "style": style,
                    "axes": {"emphasis": emphasis},
                    "emotion_class": chosen[0]["source"].get("emotion_class", "neutral"),
                    "n_lines": len(chosen),
                    "n_distinct_templates": len(used_tmpl),
                    "words": words,
                    "line_ids": [x["id"] for x in chosen],
                    "fallback": fallback,
                },
            })
            stats["clips"] += 1
            stats[fallback or "distinct_templates"] += 1
            if a.max_clips and len(items) >= a.max_clips:
                break
        if a.max_clips and len(items) >= a.max_clips:
            break

    out = {
        "metadata": {
            **{k: v for k, v in tts["metadata"].items() if k != "count"},
            "stage": "mixed",
            "grouping": "phase+situation+style+emphasis (control signal fits every line)",
            "one_line_per_template": True,
        "phrase_reuse": not a.no_reuse,
            "target_words": a.target_words,
            "emo_vector": "group median; emo_alpha left at 1.0 so --emo-scale still applies",
            "count": len(items),
        },
        "items": items,
    }
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    w = np.array([i["source"]["words"] for i in items])
    t = np.array([i["source"]["n_distinct_templates"] for i in items])
    print(f"{len(items)} clips from {len(groups)} groups -> {a.out}")
    print(f"words/clip     : min {w.min()} median {int(np.median(w))} max {w.max()}")
    print(f"templates/clip : min {t.min()} median {int(np.median(t))} max {t.max()}")
    print(f"clips needing template reuse: {stats['reused_templates']} "
          f"({stats['reused_templates']/len(items)*100:.1f}%)")


if __name__ == "__main__":
    main()
