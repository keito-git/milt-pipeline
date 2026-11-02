#!/usr/bin/env python3
"""R1 sensitivity: Mie second-site RCTA when the common-class list comes from seed labels only.

The published run used a COMMON list whose membership of 'tanuki' comes from the Mie stills,
which are the evaluation set of this experiment. This script writes a new file only.
"""
import json, os, sys
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
OUT = "outputs"
COMMON = {"ニホンジカ", "タヌキ", "イノシシ", "ニホンザル", "ニホンノウサギ"}
UNKNOWN, FALSE_DET = "不明", "誤検出(動物なし)"
RESULT = "outputs/benchmark/r1/mie_common_sensitivity.json"
if os.path.exists(RESULT):
    sys.exit(f"Refusing to overwrite {RESULT}")

ve = np.load(f"{OUT}/video_embeddings.npy").astype(np.float64)
vidx = {it["crop"]: i for i, it in enumerate(json.load(open(f"{OUT}/video_embed_index.json")))}
seed = [r for r in json.load(open(f"{OUT}/seed_labels.json"))["labels"]]
seed_set = set(json.load(open(f"{OUT}/seed_crops.json")))
seed = [r for r in seed if r["crop"] in seed_set and r["crop"] in vidx]
Z = np.stack([ve[vidx[r["crop"]]] for r in seed]); y = [r["label"] for r in seed]
clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(Z, y)
classes = list(clf.classes_)

se = np.load(f"{OUT}/embeddings.npy").astype(np.float64)
pidx = {r["crop"]: i for i, r in enumerate(json.load(open(f"{OUT}/provisional_labels.json")))}
still = [r for r in json.load(open(f"{OUT}/labels_still_v1.json"))["labels"] if r["crop"] in pidx]
P = clf.predict_proba(np.stack([se[pidx[r["crop"]]] for r in still]))
true = [r["label"] for r in still]
NON = {UNKNOWN, FALSE_DET}
seed_top5 = set([c for c, _ in Counter(y).most_common() if c not in NON][:5])

def recall_auc(order, mask):
    hit = np.concatenate([[0], np.cumsum(mask[order])]) / mask.sum()
    return float(np.trapezoid(hit, np.linspace(0, 1, len(hit))))

def capture(order, mask, b):
    return float(mask[order[: int(round(b * len(order)))]].sum() / mask.sum())

def boot_gap(a, b, mask, n=2000):
    rng = np.random.default_rng(20261102)
    ra, rb = np.empty(len(a), int), np.empty(len(b), int)
    ra[a] = np.arange(len(a)); rb[b] = np.arange(len(b))
    gaps = []
    for _ in range(n):
        s = rng.integers(0, len(a), len(a))
        ma = mask[s]
        if ma.sum() == 0:
            continue
        oa, ob = np.argsort(ra[s]), np.argsort(rb[s])
        gaps.append(recall_auc(oa, ma) - recall_auc(ob, ma))
    return [round(float(x), 4) for x in np.percentile(gaps, [2.5, 97.5])]

conf_order = np.argsort(P.max(1))
report = {"seed_counts": Counter(y), "common_published": sorted(COMMON), "common_seed_top5": sorted(seed_top5),
          "mie_counts": Counter(true), "n_eval": len(true), "runs": {}}
for score_name, common in (("published_list", COMMON), ("seed_top5", seed_top5)):
    w = np.array([0.0 if c in common or c in NON else 1.0 for c in classes])
    order = np.argsort(-(P @ w))
    for target_name, tcommon in (("published_list", COMMON), ("seed_top5", seed_top5)):
        mask = np.array([t not in tcommon and t not in NON for t in true])
        report["runs"][f"score={score_name}|target={target_name}"] = {
            "rare_classes_in_model": [c for c, v in zip(classes, w) if v],
            "n_target": int(mask.sum()), "rcta_auc": round(recall_auc(order, mask), 4), "conf_auc": round(recall_auc(conf_order, mask), 4),
            "gap_ci95": boot_gap(order, conf_order, mask),
            "rcta_b10_b30": [round(capture(order, mask, b), 3) for b in (.1, .3)],
            "conf_b10_b30": [round(capture(conf_order, mask, b), 3) for b in (.1, .3)]}
json.dump(report, open(RESULT, "w"), ensure_ascii=False, indent=1)
print(json.dumps(report, ensure_ascii=False, indent=1))
