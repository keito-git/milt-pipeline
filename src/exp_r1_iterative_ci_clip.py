#!/usr/bin/env python3
"""R1: bootstrap CIs for the iterative comparison, and seed-only RCTA list check on video crops."""
import json, os, sys, time
import numpy as np
import exp_r1_revision as x

OUT = "outputs/benchmark/r1/r1_iterative_ci_clip.json"
if os.path.exists(OUT):
    sys.exit(f"Refusing to overwrite {OUT}")
BOOT, SEEDS, ROUNDS = 2000, 5, 10
report = {}
for backbone in ("bioclip", "dinov2"):
    d = x.load(backbone)
    # Group by source clip: the crop index stores the source video under "clip".
    _clip = {}
    for _l in open("outputs/video_crops_index.jsonl"):
        _r = json.loads(_l); _clip[_r["crop"]] = _r["clip"]
    _seed = set(json.load(open("outputs/seed_crops.json")))
    _final = {r["crop"] for r in json.load(open("outputs/labels_video_v1.json"))["labels"]}
    _crops = [it["crop"] for it in json.load(open("outputs/video_embed_index.json")) if it["crop"] not in _seed and it["crop"] in _final]
    assert len(_crops) == len(d["y"])
    d["videos"] = np.array([_clip[c] for c in _crops])
    Z, y = d["Z"], d["y"]
    Z_base = np.vstack([d["Z_seed"], d["Z_still"]]); y_base = d["y_seed"] + d["y_still"]
    n = len(y); batch = round(n * .05)
    def run(strategy, rng):
        reviewed = np.zeros(n, bool); M = []
        for r in range(ROUNDS + 1):
            lab = np.flatnonzero(reviewed)
            clf = x.fit(np.vstack([Z_base, Z[lab]]), list(y_base) + y[lab].tolist())
            classes = [str(c) for c in clf.classes_]
            P_all = clf.predict_proba(Z)
            pred = np.array(classes)[P_all.argmax(1)]; pred[lab] = y[lab]
            M.append(pred == y)
            if r == ROUNDS: break
            pool = np.flatnonzero(~reviewed); P = P_all[pool]
            if strategy == "random": pick = rng.choice(len(pool), batch, replace=False)
            elif strategy == "coreset": pick = x.kcenter(Z[pool], np.vstack([Z_base, Z[lab]]), batch)
            elif strategy == "badge": pick = x.kmeanspp(x.gradient_embedding(Z[pool], P), batch, rng)
            else: pick = x.uncertainty_rank(P, classes, strategy)[:batch]
            reviewed[pool[pick]] = True
        return np.array(M, float)  # rounds x crops
    C = {}
    for s in ("confidence", "coreset", "badge", "random"):
        t = time.time()
        reps = SEEDS if s in ("badge", "random") else 1
        C[s] = np.mean([run(s, np.random.default_rng(x.SEED + i)) for i in range(reps)], axis=0)
        print(backbone, s, round(time.time() - t), flush=True)
    bud = np.arange(ROUNDS + 1) * 5
    auc = lambda M: np.trapezoid(M.mean(1), bud) / bud[-1]
    groups = {}
    for i, v in enumerate(d[x_key] if (x_key := next((k for k in ("videos",) if k in d), None)) else range(n)):
        groups.setdefault(str(v), []).append(i)
    glist = [np.array(v) for v in groups.values()]
    rng = np.random.default_rng(20261102)
    res = {"auc": {s: round(float(auc(M)), 4) for s, M in C.items()}, "group_key": x_key, "n_groups": len(glist), "gaps": {}}
    for b in ("badge", "coreset", "random"):
        gc, gv = [], []
        for _ in range(BOOT):
            s = rng.integers(0, n, n); gc.append(auc(C["confidence"][:, s]) - auc(C[b][:, s]))
            pick = rng.integers(0, len(glist), len(glist)); s = np.concatenate([glist[j] for j in pick])
            gv.append(auc(C["confidence"][:, s]) - auc(C[b][:, s]))
        res["gaps"][b] = {"mean": round(float(auc(C["confidence"]) - auc(C[b])), 4),
                          "ci_crop": [round(float(q), 4) for q in np.percentile(gc, [2.5, 97.5])],
                          "ci_video": [round(float(q), 4) for q in np.percentile(gv, [2.5, 97.5])]}
    # Seed-only model on video crops: published list versus seed-derived list.
    clf = x.fit(d["Z_seed"], d["y_seed"]); classes = [str(c) for c in clf.classes_]; P = clf.predict_proba(Z)
    NON = {x.UNKNOWN, x.FALSE_DET}
    from collections import Counter
    seed5 = set([c for c, _ in Counter(d["y_seed"]).most_common() if c not in NON][:5])
    conf = np.argsort(P.max(1))
    so = {"seed_top5": sorted(seed5), "runs": {}}
    for nm, common in (("published", x.COMMON), ("seed_top5", seed5)):
        w = np.array([0. if c in common or c in NON else 1. for c in classes]); o = np.argsort(-(P @ w))
        for tn, tc in (("published", x.COMMON), ("seed_top5", seed5)):
            m = np.array([str(t) not in tc and str(t) not in NON for t in y])
            cap = lambda ord_, b: round(float(m[ord_[:round(b * n)]].sum() / m.sum()), 3)
            so["runs"][f"score={nm}|target={tn}"] = {"n": int(m.sum()), "rcta": [cap(o, .1), cap(o, .3)], "conf": [cap(conf, .1), cap(conf, .3)]}
    res["seed_only_video"] = so
    report[backbone] = res
    json.dump(report, open(OUT + ".tmp", "w"), ensure_ascii=False, indent=1)
os.replace(OUT + ".tmp", OUT)
print(json.dumps(report, ensure_ascii=False))
