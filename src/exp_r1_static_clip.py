#!/usr/bin/env python3
"""R1: paired bootstrap CIs resampling source clips (static orderings)."""
import json, os, sys
import numpy as np
import exp_r1_revision as x
from pipeline_canonical import badge_order
OUT = "outputs/benchmark/r1/r1_static_clip.json"
if os.path.exists(OUT): sys.exit("exists")
clip = {}
for l in open("outputs/video_crops_index.jsonl"):
    r = json.loads(l); clip[r["crop"]] = r["clip"]
seed = set(json.load(open("outputs/seed_crops.json")))
final = {r["crop"] for r in json.load(open("outputs/labels_video_v1.json"))["labels"]}
crops = [it["crop"] for it in json.load(open("outputs/video_embed_index.json")) if it["crop"] not in seed and it["crop"] in final]
rep = {}
for b in ("bioclip", "dinov2"):
    d = x.load(b); y = d["y"]; n = len(y); assert len(crops) == n
    G = np.array([clip[c] for c in crops]); rare = np.array([v not in x.EXCLUDED for v in y])
    Zb = np.vstack([d["Z_seed"], d["Z_still"]]); yb = d["y_seed"] + d["y_still"]
    clf = x.fit(Zb, yb); cl = [str(c) for c in clf.classes_]; P = clf.predict_proba(d["Z"]); pred = np.array(cl)[P.argmax(1)]
    O = {"confidence": x.uncertainty_rank(P, cl, "confidence"), "rcta": x.uncertainty_rank(P, cl, "rcta"), "coreset": x.kcenter(d["Z"], Zb, n),
         "badge_approx": badge_order(d["Z"], P, np.random.default_rng(6906), pca_components=20),
         "badge_true": x.kmeanspp(x.gradient_embedding(d["Z"], P), n, np.random.default_rng(x.SEED))}
    rep[b] = {"n_clips": len(set(G))}
    for a, c in (("rcta", "confidence"), ("confidence", "coreset"), ("confidence", "badge_approx"), ("confidence", "badge_true")):
        rep[b][f"{a}-{c}"] = x.boot_gap(O[a], O[c], pred, y, rare, G, np.random.default_rng(x.SEED), x.N_BOOT)
    json.dump(rep, open(OUT + ".tmp", "w"), indent=1); print(b, json.dumps(rep[b]), flush=True)
os.replace(OUT + ".tmp", OUT)
