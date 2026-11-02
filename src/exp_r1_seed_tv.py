#!/usr/bin/env python3
"""R1: species-composition TV under confidence ordering for seed-only and full models (BioCLIP)."""
import json, os, sys
from collections import Counter
import numpy as np
import exp_r1_revision as x
OUT = "outputs/benchmark/r1/r1_seed_tv.json"
if os.path.exists(OUT): sys.exit("exists")
d = x.load("bioclip"); Z, y = d["Z"], d["y"]
crops = [it["crop"] for it in json.load(open("outputs/video_embed_index.json")) ]
seed = set(json.load(open("outputs/seed_crops.json")))
final = {r["crop"] for r in json.load(open("outputs/labels_video_v1.json"))["labels"]}
crops = [c for c in crops if c not in seed and c in final]
hab = {}
for l in open("outputs/video_crops_index.jsonl"):
    r = json.loads(l); hab[r["crop"]] = r["habitat"]
H = Counter(hab.get(c) for c in crops); ok = [h for h, n in H.items() if h is not None and n >= 20]
ix = {h: [i for i, c in enumerate(crops) if hab.get(c) == h] for h in ok}
def tv_at(P, classes, B):
    pred = np.array(classes)[P.argmax(1)]; o = np.argsort(P.max(1)); got = pred.copy()
    got[o[:round(B / 100 * len(y))]] = y[o[:round(B / 100 * len(y))]]
    v = []
    for I in ix.values():
        a, b = Counter(got[I]), Counter(y[I]); v.append(.5 * sum(abs(a[k] / len(I) - b[k] / len(I)) for k in set(a) | set(b)))
    return round(float(np.mean(v)), 4)
rep = {"n_habitats": len(ok)}
for nm, (Zt, yt) in {"seed_only": (d["Z_seed"], d["y_seed"]), "full": (np.vstack([d["Z_seed"], d["Z_still"]]), d["y_seed"] + d["y_still"])}.items():
    clf = x.fit(Zt, yt); P = clf.predict_proba(Z); c = [str(k) for k in clf.classes_]
    rep[nm] = {str(B): tv_at(P, c, B) for B in (0, 30, 50)}
json.dump(rep, open(OUT, "w"), indent=1); print(rep)
