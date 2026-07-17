"""
Additional ecological statistics for the pipeline paper:
 - Ecological summary beyond TV: Shannon diversity error and Spearman rank
   correlation of predicted vs true per-habitat species composition.
 - Bootstrap 95% CI on the AUC(accuracy-vs-budget) gap between confidence order
   and random order (does uncertainty order significantly beat random?).

Standalone (does not modify the pipeline scripts). Seed-out evaluation.
Writes: outputs/benchmark/pipeline_ecology_extra.json
"""
import json
import os
from collections import Counter

import numpy as np

HOME = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HOME, "outputs")
BMK = os.path.join(OUT, "benchmark")
COMMON = {"ニホンジカ", "タヌキ", "イノシシ", "ニホンザル", "ニホンノウサギ"}
BUDGETS = [0, 5, 10, 20, 30, 50, 100]
RNG = np.random.default_rng(0)


def shannon(counter):
    t = sum(counter.values())
    if t == 0:
        return 0.0
    ps = [v / t for v in counter.values() if v > 0]
    return float(-sum(p * np.log(p) for p in ps))


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = a.argsort().argsort(); rb = b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def emb_map():
    m = {}
    se = np.load(os.path.join(OUT, "embeddings.npy"))
    for i, r in enumerate(json.load(open(os.path.join(OUT, "provisional_labels.json")))):
        m[r["crop"]] = se[i]
    ve = np.load(os.path.join(OUT, "video_embeddings.npy"))
    vit = json.load(open(os.path.join(OUT, "video_embed_index.json")))
    for i, it in enumerate(vit):
        m[it["crop"]] = ve[i]
    return m, vit


def main():
    from sklearn.linear_model import LogisticRegression
    EMB, vit = emb_map()
    seed_lab = {r["crop"]: r["label"] for r in json.load(open(os.path.join(OUT, "seed_labels.json")))["labels"]}
    final = {r["crop"]: r["label"] for r in json.load(open(os.path.join(OUT, "labels_video_v1.json")))["labels"]}
    hab = {json.loads(l)["crop"]: json.loads(l)["habitat"] for l in open(os.path.join(OUT, "video_crops_index.jsonl"))}
    seed_set = set(json.load(open(os.path.join(OUT, "seed_crops.json"))))

    X, y = [], []
    for c, l in seed_lab.items():
        if c in EMB:
            X.append(EMB[c]); y.append(l)
    for r in json.load(open(os.path.join(OUT, "labels_still_v1.json")))["labels"]:
        if r["crop"] in EMB:
            X.append(EMB[r["crop"]]); y.append(r["label"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(np.stack(X), y)
    classes = list(clf.classes_)

    crops = [it["crop"] for it in vit if it["crop"] not in seed_set and it["crop"] in final]
    P = clf.predict_proba(np.stack([EMB[c] for c in crops]))
    pred = [classes[i] for i in P.argmax(1)]
    true = [final[c] for c in crops]
    conf = P.max(1); n = len(crops)

    # habitats with enough data
    habs = [h for h in set(hab.get(c) for c in crops)
            if sum(hab.get(crops[i]) == h for i in range(n)) >= 20]
    ci_order = np.argsort(conf)

    def eco(B):
        k = int(round(B / 100 * n)); rev = set(ci_order[:k].tolist())
        got = [true[i] if i in rev else pred[i] for i in range(n)]
        sh_err, sp = [], []
        for h in habs:
            ix = [i for i in range(n) if hab.get(crops[i]) == h]
            keys = sorted(set(true[i] for i in ix) | set(got[i] for i in ix))
            tc = Counter(true[i] for i in ix); gc = Counter(got[i] for i in ix)
            sh_err.append(abs(shannon(gc) - shannon(tc)))
            sp.append(spearman([gc.get(k_, 0) for k_ in keys], [tc.get(k_, 0) for k_ in keys]))
        return round(float(np.mean(sh_err)), 3), round(float(np.nanmean(sp)), 3)

    ecology = {str(B): dict(zip(["shannon_abs_err", "spearman_rank"], eco(B))) for B in [0, 30, 50]}
    n_by_hab = {h: int(sum(hab.get(crops[i]) == h for i in range(n))) for h in habs}

    # AUC gap CI: confidence vs random, bootstrap over crops
    def auc_of(order_idx, idx_sub):
        sub = [i for i in order_idx if i in idx_sub]
        accs = []
        m = len(sub)
        for B in BUDGETS:
            k = int(round(B / 100 * m)); rev = set(sub[:k])
            got = [true[i] if i in rev else pred[i] for i in sub]
            tr = [true[i] for i in sub]
            accs.append(np.mean([got[j] == tr[j] for j in range(m)]))
        return np.trapezoid(accs, [b / 100 for b in BUDGETS]) if hasattr(np, "trapezoid") else np.trapz(accs, [b / 100 for b in BUDGETS])

    gaps = []
    allidx = np.arange(n)
    for _ in range(300):
        bs = set(RNG.integers(0, n, n).tolist())
        conf_auc = auc_of(list(ci_order), bs)
        rnd_auc = np.mean([auc_of(list(RNG.permutation(n)), bs) for _ in range(3)])
        gaps.append(conf_auc - rnd_auc)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    out = {"ecology_by_budget": ecology, "n_by_habitat": n_by_hab,
           "auc_gap_confidence_minus_random": {"mean": round(float(np.mean(gaps)), 4),
                                               "ci95": [round(float(lo), 4), round(float(hi), 4)]}}
    json.dump(out, open(os.path.join(BMK, "pipeline_ecology_extra.json"), "w"), ensure_ascii=False, indent=2)
    print("ecology by budget (Shannon abs err / Spearman):", json.dumps(ecology, ensure_ascii=False))
    print("n by habitat:", n_by_hab)
    print("AUC gap confidence-random:", out["auc_gap_confidence_minus_random"])


if __name__ == "__main__":
    main()
