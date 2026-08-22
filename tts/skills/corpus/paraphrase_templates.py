#!/usr/bin/env python3
"""Stage 0.5 - grow the template bank with LLM paraphrases, for text diversity.

The bank has 2,741 templates, and the non-neutral situations are tiny (celebrate has 561
templates and only 1,193 placeholder combinations). Building 10,000 clips from that forces
heavy reuse. This writes new wordings for every template so the clips stop repeating.

Uses the local gemma-3-12b-it (the LTX text encoder) — no API needed.

  python paraphrase_templates.py --variants 8 --batch 16 --out dealer_phrases_aug.json
  python paraphrase_templates.py --limit 20 --variants 8 --out /tmp/probe.json   # smoke test

Every variant is validated: identical placeholder multiset, word count in range, ASCII
punctuation, and not a duplicate of the source or of another variant. Originals are always
kept — a variant is tagged `origin: "paraphrase"` and carries `origin_index`.
"""
import argparse, collections, copy, json, os, re, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "dealer_phrases.json")
DST = os.path.join(HERE, "dealer_phrases_aug.json")
GEMMA = "/home/ubuntu/chunjin/project/valka-ai/LTX-2/models/gemma-3-12b"

PH = re.compile(r"\{[a-z_]+\}")


def norm_key(t):
    """Dedup key. Must be the ONLY definition -- a second, subtly different one (missing
    .strip()) let duplicates through the situation-wide check."""
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()
# the model likes typographic punctuation; the TTS front-end wants plain ASCII
SUBS = {"’": "'", "‘": "'", "“": '"', "”": '"',
        "—": " - ", "–": " - ", "…": "...", " ": " "}

PROMPT = """You write lines for a live blackjack dealer. Rewrite the line below into {n} NEW variants.

Rules:
- keep the same meaning, situation and register
- keep EVERY {{placeholder}} exactly as written, and the same number of them
- vary the wording and the sentence shape; do not just swap a single word
- {lo} to {hi} words each, natural spoken English
- no emoji, no quotes, no numbering, no commentary
- output ONLY the {n} variants, one per line

Situation: {desc}
Register: {style}, {emphasis}{extra}
Line: {line}"""


def clean(s):
    for a, b in SUBS.items():
        s = s.replace(a, b)
    s = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", s)      # stray bullets/numbering
    s = re.sub(r"\s+", " ", s).strip().strip('"')
    return s


def acceptable(cand, src, lo, hi, seen):
    if not cand or len(cand) > 200:
        return False
    if collections.Counter(PH.findall(cand)) != collections.Counter(PH.findall(src)):
        return False                                          # placeholders must survive
    if not (lo <= len(cand.split()) <= hi):
        return False
    if not cand[0].isalnum() and cand[0] not in "'":
        return False
    key = norm_key(cand)
    if key in seen:
        return False
    seen.add(key)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=DST)
    ap.add_argument("--variants", type=int, default=8, help="new wordings per template")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only the first N templates")
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--model", default=GEMMA)
    a = ap.parse_args()

    d = json.load(open(a.src))
    jobs = []                                                  # (phase, situation, index, phrase, desc)
    for pn, situs in d["phases"].items():
        for sn, s in situs.items():
            for i, p in enumerate(s["phrases"]):
                jobs.append((pn, sn, i, p, s.get("description", sn.replace("_", " "))))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} templates x {a.variants} variants", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                device_map="cuda:0").eval()
    print(f"model loaded, {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    def build(job):
        _, _, _, p, desc = job
        w = len(p["text"].split())
        lo, hi = max(2, int(w * 0.6)), max(6, int(w * 1.7))
        ax = p.get("axes") or {}
        extra = ""
        if ax.get("sentence_type"):
            extra = f", {ax['sentence_type']}"
        return PROMPT.format(n=a.variants, lo=lo, hi=hi, desc=desc,
                             style=p.get("style", "neutral"),
                             emphasis=ax.get("emphasis", "neutral"),
                             extra=extra, line=p["text"]), lo, hi

    out = copy.deepcopy(d)
    # Dedup key set PER SITUATION, pre-seeded with every original line there. Scoping this
    # per template (the obvious way) lets variants of different templates collide -- two
    # neighbours both produced "Bets are open, please.", and one even regenerated another
    # template's original text. Stage 2.5 dedupes by template, so identical text under two
    # template ids would slip straight through.
    seen_sit = {}
    for pn, situs in d["phases"].items():
        for sn, s_ in situs.items():
            seen_sit[(pn, sn)] = {norm_key(x["text"]) for x in s_["phrases"]}
    added = collections.Counter()
    stats = collections.Counter()
    t0 = time.time()
    for start in range(0, len(jobs), a.batch):
        chunk = jobs[start:start + a.batch]
        prompts, bounds = [], []
        for j in chunk:
            pr, lo, hi = build(j)
            prompts.append(tok.apply_chat_template([{"role": "user", "content": pr}],
                                                   add_generation_prompt=True, tokenize=False))
            bounds.append((lo, hi))
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=40 * a.variants, do_sample=True,
                                 temperature=a.temperature, top_p=0.95,
                                 pad_token_id=tok.pad_token_id)
        for (pn, sn, i, p, _), g, (lo, hi) in zip(chunk, gen, bounds):
            text = tok.decode(g[enc["input_ids"].shape[-1]:], skip_special_tokens=True)
            seen = seen_sit[(pn, sn)]
            kept = []
            for line in text.splitlines():
                c = clean(line)
                if acceptable(c, p["text"], lo, hi, seen):
                    kept.append(c)
                else:
                    stats["rejected"] += 1
            for k in kept:
                q = copy.deepcopy(p)
                q["text"] = k
                q["origin"] = "paraphrase"
                q["origin_index"] = i
                out["phases"][pn][sn]["phrases"].append(q)
                added[(pn, sn)] += 1
                stats["kept"] += 1
        done = start + len(chunk)
        if done % (a.batch * 8) == 0 or done >= len(jobs):
            el = time.time() - t0
            print(f"  {done}/{len(jobs)}  kept {stats['kept']} rejected {stats['rejected']}  "
                  f"{done/el*60:.0f} tmpl/min  eta {(len(jobs)-done)/max(done/el,1e-9)/60:.1f} min",
                  flush=True)

    n_before = sum(len(s["phrases"]) - added[(pn, sn)]
                   for pn, ss in out["phases"].items() for sn, s in ss.items())
    n_after = sum(len(s["phrases"]) for ss in out["phases"].values() for s in ss.values())
    out["metadata"]["paraphrase"] = {
        "model": os.path.basename(a.model), "variants_requested": a.variants,
        "temperature": a.temperature,
        "templates_in": len(jobs), "kept": stats["kept"], "rejected": stats["rejected"],
        "note": "originals kept; added phrases carry origin='paraphrase' and origin_index",
    }
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"\ntemplates {n_before} -> {n_after}  (+{stats['kept']}, rejected {stats['rejected']})")
    print(f"kept per template: {stats['kept']/max(len(jobs),1):.1f} of {a.variants} requested")
    print(f"wrote {a.out}  in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
