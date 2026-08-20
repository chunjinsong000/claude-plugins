#!/usr/bin/env python3
"""CAM++ speaker similarity of cloned takes vs their reference -- WITH a chance floor.

A bare cosine is meaningless: speakers recorded in one studio already resemble each
other. During this project's comparison the 11 dealer references scored 0.253 mean
cosine against EACH OTHER (max 0.585), so 0.4 is not "cloned successfully".
This always prints that floor plus a per-take mismatched-reference control.

Must run in the FireRedTTS3 venv -- CAM++ and its VoxCeleb checkpoint ship there:
  cd $VALKA_ROOT/FireRedTTS3 && .venv/bin/python <this> --takes out/ --refs refs.json

--refs is a json list of {"id","path"} (prep_refs.py writes one); take filenames must
end in _<id>.wav so each take can be matched to its own reference.
"""
import argparse, glob, itertools, json, os
import numpy as np, torch, torchaudio
from fireredtts3.campp.campp import CamppEmbedding

VALKA = os.environ.get("VALKA_ROOT", "/home/ubuntu/chunjin/project/valka-ai")
ap = argparse.ArgumentParser()
ap.add_argument("--takes", required=True, help="directory of generated wavs")
ap.add_argument("--refs", required=True, help="json list of {id, path}")
ap.add_argument("--ckpt", default=f"{VALKA}/FireRedTTS3/pretrained_models/campp/campplus_voxceleb.bin")
ap.add_argument("--json-out", default=None)
a = ap.parse_args()

m = CamppEmbedding(a.ckpt).cuda()
def emb(p):
    w, sr = torchaudio.load(p)
    return torch.nn.functional.normalize(m(w, sr), dim=-1)[0]

refs = json.load(open(a.refs))
R = {r["id"]: emb(r["path"]) for r in refs}
floor = [float(R[x] @ R[y]) for x, y in itertools.combinations(R, 2)]
print(f"CHANCE FLOOR -- the {len(R)} references vs each other: "
      f"mean {np.mean(floor):.3f}, max {max(floor):.3f}")
print("Read every number below against that floor, not against 0.\n")

rows, groups = {}, {}
for f in sorted(glob.glob(os.path.join(a.takes, "*.wav"))):
    b = os.path.basename(f)[:-4]
    if "_" not in b:
        continue
    system, rid = b.rsplit("_", 1)
    if rid not in R:
        print(f"skip {b}: no reference id '{rid}'"); continue
    e = emb(f)
    matched = float(e @ R[rid])
    mis = [float(e @ R[o]) for o in R if o != rid]
    rows[b] = {"system": system, "ref": rid, "sim": matched,
               "mismatch_mean": float(np.mean(mis))}
    groups.setdefault(system, []).append((matched, np.mean(mis)))

print(f"{'system':<24}{'n':>3}{'spk-sim':>18}{'mismatch':>10}{'margin':>8}")
print("-" * 63)
for s in sorted(groups):
    v = np.array([x[0] for x in groups[s]]); mm = np.array([x[1] for x in groups[s]])
    print(f"{s:<24}{len(v):3d}{v.mean():8.3f} ±{v.std():.3f}{mm.mean():10.3f}"
          f"{v.mean()-np.mean(floor):+8.3f}")
print("\nmargin = mean spk-sim minus the chance floor; that is the real signal.")
if a.json_out:
    json.dump({"takes": rows, "chance_floor_mean": float(np.mean(floor)),
               "chance_floor_max": float(max(floor))}, open(a.json_out, "w"), indent=2)
    print(f"wrote {a.json_out}")
