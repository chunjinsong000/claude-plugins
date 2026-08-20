#!/usr/bin/env python3
"""Step 1 - expand every {placeholder} in dealer_phrases.json over its full value domain.

Every placeholder here has a finite domain, so each phrase template is emitted once per
value rather than once with a random pick: all 52 cards, all 13 ranks, every box, every
valid point total, every name. Domains follow blackjack semantics, so a "bust" line is
always over 21, a dealer stand is 17-21, and a pair of Eights is worth 16.

Phrases with several placeholders rotate in lockstep (not a cartesian product): the
phrase is emitted max(domain sizes) times, and each placeholder cycles its own domain.
Every value therefore appears at least once without the output exploding - "{card} for
box {box}." yields 52 lines (all 52 cards, boxes cycling 1-7), not 364.

  python fill_placeholders.py [--seed 20260820] [--names 10] [--cap N]

Writes dealer_phrases_filled.json next to the source.
"""
import argparse, collections, copy, json, os, random, re

# Defaults sit next to this script; point --src/--out at your data instead.
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "dealer_phrases.json")
DST = os.path.join(HERE, "dealer_phrases_filled.json")

# --- card vocabulary -------------------------------------------------------
SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
# rank -> (blackjack value, plural).  Every {rank} in the file is followed by "s"
# ("a pair of {rank}s"), so the plural must be spelled right: Six -> Sixes.
RANKS = {
    "Ace":   (11, "Aces"),   "Two":   (2,  "Twos"),   "Three": (3,  "Threes"),
    "Four":  (4,  "Fours"),  "Five":  (5,  "Fives"),  "Six":   (6,  "Sixes"),
    "Seven": (7,  "Sevens"), "Eight": (8,  "Eights"), "Nine":  (9,  "Nines"),
    "Ten":   (10, "Tens"),   "Jack":  (10, "Jacks"),  "Queen": (10, "Queens"),
    "King":  (10, "Kings"),
}
RANK_LIST = list(RANKS)
CARD_LIST = [f"{r} of {s}" for r in RANK_LIST for s in SUITS]      # all 52
BOX_LIST = list(range(1, 8))                                        # real blackjack seats
SECONDS_LIST = [3, 5, 10, 15, 20, 30]                               # values a dealer calls out

NAME_POOL = [
    "Sarah", "Alex", "Emma", "Daniel", "Olivia", "Ryan", "Chloe", "Marcus", "Sophie",
    "Nathan", "Grace", "Oliver", "Hannah", "Ethan", "Lucy", "Adam", "Isabel", "Connor",
    "Maya", "Julian", "Ruby", "Simon", "Nora", "Felix", "Leah", "Victor", "Iris",
    "Damian", "清", "Elena", "Theo", "Naomi", "Caleb", "Freya", "Miles", "Alice",
]
NAME_POOL = [n for n in NAME_POOL if n.isascii()]


def pair_total(rank):
    """A pair of Aces is soft 12, not 22 - 22 would be a bust, which contradicts the
    split/double lines these totals appear in."""
    return 12 if rank == "Ace" else 2 * RANKS[rank][0]


# --- per-situation point domains (blackjack rules, not an arbitrary range) --
BUST = list(range(22, 31))     # over 21 by definition
LIVE = list(range(4, 22))      # hand still in play: two cards min 4, never over 21
STAND_DEALER = list(range(17, 22))   # dealer stands on 17+
STAND_PLAYER = list(range(12, 22))   # below 12 a player always draws
PUSH = list(range(17, 22))     # ties happen once both sides have stood
POINTS_DOMAIN = {
    "player_turn/player_busts":  BUST,
    "dealer_turn/dealer_busts":  BUST,
    "dealer_turn/dealer_stands": STAND_DEALER,
    "player_turn/player_stands": STAND_PLAYER,
    "resolution/push":           PUSH,
}
PH = re.compile(r"\{[a-z_]+\}")


def domains_for(text, situation, names):
    """Value domain per placeholder present in this template."""
    ms = list(dict.fromkeys(PH.findall(text)))
    pair_line = "{rank}" in ms and "{points}" in ms
    dom = {}
    for m in ms:
        if m == "{card}":
            dom[m] = CARD_LIST
        elif m == "{rank}":
            dom[m] = RANK_LIST
        elif m == "{points}":
            # on a pair line the total is dictated by the rank, so it rides along
            dom[m] = RANK_LIST if pair_line else POINTS_DOMAIN.get(situation, LIVE)
        elif m == "{box}":
            dom[m] = BOX_LIST
        elif m == "{seconds}":
            dom[m] = SECONDS_LIST
        elif m in ("{dealer_name}", "{player_name}"):
            dom[m] = names
        else:
            raise ValueError(f"unhandled placeholder {m} in: {text}")
    return dom, pair_line


def render(text, dom, pair_line, cursor):
    """Emit one variant. Values come from a GLOBAL per-placeholder cursor, so coverage
    accumulates across templates: with a phrase budget below 52 per template, all 52
    cards still appear in the corpus as a whole rather than 52 times per template."""
    used = {}

    def take(m):
        i = cursor[m]
        cursor[m] = i + 1
        return dom[m][i % len(dom[m])]

    if pair_line:
        rank = take("{rank}")
        used["{rank}"] = rank
        used["{points}"] = pair_total(rank)

    out = text
    if "{rank}s" in out:
        rank = used.get("{rank}") or take("{rank}")
        used["{rank}"] = rank
        out = out.replace("{rank}s", RANKS[rank][1])   # before the bare-{rank} rule

    for m in dict.fromkeys(PH.findall(out)):
        v = used.get(m)
        if v is None:
            v = take(m)
        used[m] = v
        out = out.replace(m, str(v))
    return out, used


def allocate(caps, budget):
    """Spread `budget` phrases over templates as evenly as each template's capacity
    allows (water-filling), so no single high-capacity template floods the corpus."""
    k = {i: min(1, c) for i, c in caps.items()}
    left = budget - sum(k.values())
    while left > 0:
        active = [i for i, c in caps.items() if k[i] < c]
        if not active:
            break
        step = max(1, left // len(active))
        for i in active:
            d = min(step, caps[i] - k[i], left)
            k[i] += d
            left -= d
            if left <= 0:
                break
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--names", type=int, default=10, help="size of the random name pool")
    ap.add_argument("--cap", type=int, default=0,
                    help="max variants per phrase (0 = full domain)")
    ap.add_argument("--target", type=int, default=0,
                    help="exact total number of phrases to emit (0 = full enumeration). "
                         "Templates without placeholders always contribute 1; the rest is "
                         "spread as evenly as capacity allows, and placeholder coverage "
                         "becomes corpus-wide instead of per-template.")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=DST)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    names = rng.sample(NAME_POOL, a.names)
    d = json.load(open(a.src))
    out = copy.deepcopy(d)

    # pass 1: capacity of every template that has placeholders
    tmpl = []       # (phase, situation_name, index, domains, pair_line, capacity)
    n_plain = 0
    for pn, situs in d["phases"].items():
        for sn, s in situs.items():
            situation = f"{pn}/{sn}"
            for idx, p in enumerate(s["phrases"]):
                if not PH.search(p["text"]):
                    n_plain += 1
                    continue
                dom, pair_line = domains_for(p["text"], situation, names)
                if pair_line:
                    cap = len(dom["{rank}"])          # points rides along with rank
                else:
                    cap = max(len(v) for v in dom.values())
                if a.target:
                    # with an exact target, a template may exceed its lockstep width by
                    # combining placeholders, so allow up to the cartesian size
                    prod = 1
                    for m, v in dom.items():
                        if not (pair_line and m == "{points}"):
                            prod *= len(v)
                    cap = prod
                if a.cap:
                    cap = min(cap, a.cap)
                tmpl.append([pn, sn, idx, dom, pair_line, cap])

    if a.target:
        budget = a.target - n_plain
        if budget < len(tmpl):
            raise SystemExit(f"--target {a.target} too small: {n_plain} placeholder-free "
                             f"phrases + {len(tmpl)} templates needs >= {n_plain+len(tmpl)}")
        alloc = allocate({i: t[5] for i, t in enumerate(tmpl)}, budget)
    else:
        alloc = {i: t[5] for i, t in enumerate(tmpl)}

    counts = {(t[0], t[1], t[2]): alloc[i] for i, t in enumerate(tmpl)}
    cursor = collections.Counter()

    n_in = n_out = 0
    for pn, situs in d["phases"].items():
        for sn, s in situs.items():
            situation = f"{pn}/{sn}"
            expanded = []
            for idx, p in enumerate(s["phrases"]):
                n_in += 1
                if not PH.search(p["text"]):
                    expanded.append(dict(p))
                    continue
                dom, pair_line = domains_for(p["text"], situation, names)
                n = counts[(pn, sn, idx)]
                for i in range(n):
                    filled, used = render(p["text"], dom, pair_line, cursor)
                    q = dict(p)
                    q["text_template"] = p["text"]
                    q["text"] = filled
                    q["variant"] = i
                    q["placeholder_values"] = used
                    expanded.append(q)
            out["phases"][pn][sn]["phrases"] = expanded
            n_out += len(expanded)

    out["metadata"]["filled"] = {
        "seed": a.seed,
        "name_pool": names,
        "target": a.target or None,
        "expansion": ("each placeholder enumerated over its full domain; multi-placeholder "
                      "phrases rotate in lockstep (max(domain) variants, not a product)")
                     if not a.target else
                     ("exact target: placeholder-free templates contribute 1 each, the rest "
                      "is water-filled evenly across templates up to each one's cartesian "
                      "capacity; placeholder values rotate on GLOBAL cursors so every value "
                      "still appears corpus-wide even though no single template enumerates all"),
        "domains": {
            "{card}": f"all 52 ({len(CARD_LIST)})",
            "{rank}": RANK_LIST,
            "{box}": BOX_LIST,
            "{seconds}": SECONDS_LIST,
            "{dealer_name}/{player_name}": names,
            "{points}": {"default_live": [LIVE[0], LIVE[-1]],
                         **{k: [v[0], v[-1]] for k, v in POINTS_DOMAIN.items()},
                         "pair_lines": "2x rank value; a pair of Aces is 12"},
        },
        "note": "point domains follow blackjack rules rather than a flat range, so bust / "
                "stand / push lines stay semantically valid",
    }

    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"names ({len(names)}): {', '.join(names)}")
    print(f"{n_in} templates -> {n_out} phrases  ({a.out})")
    left = [p["text"] for pn, ss in out["phases"].items() for sn, s in ss.items()
            for p in s["phrases"] if PH.search(p["text"])]
    print(f"unresolved placeholders: {len(left)}")


if __name__ == "__main__":
    main()
