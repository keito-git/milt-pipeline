#!/usr/bin/env python3
"""Post-hoc: audit error-rate intervals resampling source clips (the crop index stores them under 'clip')."""
import json
from analyze_blind_audit import cluster_ci
clip = {json.loads(l)["crop"]: json.loads(l)["clip"] for l in open("../outputs/video_crops_index.jsonl")}
key = {r["id"]: r for r in json.load(open("audit_key.json"))["items"]}
for name in ("annotator_A", "annotator_B"):
    a = {x["id"]: x["label"] for x in json.load(open(f"results/{name}.json"))["answers"]}
    for b in ("[0.0,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,1.0]"):
        rows = [(i, r) for i, r in key.items() if r["band"] == b and not r["is_hidden_repeat"]
                and r["stratum"] in ("S1_all_high", "S2_random") and a[i] != "不明"]
        e = [a[i] != r["auto_label"] for i, r in rows]
        print(name, b, sum(e), len(e), cluster_ci(e, [clip[r["crop"]] for _, r in rows]))
