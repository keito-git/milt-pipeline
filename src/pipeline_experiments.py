"""
Core experiments for the pipeline paper.

Evaluates review order strategies on seed-out crops only (seed crops were used to
train the auto-labeler). Compares confidence (least-confident first), predictive
entropy, margin (top1-top2), and RANDOM by the area under the accuracy-vs-budget
curve. Includes Wilson 95% CIs on key proportions.

Re-derives full class probabilities by retraining the same logistic-regression
auto-labeler (seed + confirmed still labels) on BioCLIP embeddings.

Reads: outputs/seed_labels.json, outputs/labels_video_v1.json, outputs/labels_still_v1.json,
       outputs/video_embeddings.npy, outputs/video_embed_index.json,
       outputs/embeddings.npy, outputs/provisional_labels.json,
       outputs/video_crops_index.jsonl, outputs/seed_crops.json
Writes: outputs/benchmark/pipeline_experiments.json
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


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def tv(a, b, keys):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)


def normc(counter):
    t = sum(counter.values()) or 1
    return {k: v / t for k, v in counter.items()}


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
    os.makedirs(BMK, exist_ok=True)
    EMB, vit = emb_map()
    seed_lab = {r["crop"]: r["label"] for r in json.load(open(os.path.join(OUT, "seed_labels.json")))["labels"]}
    final = {r["crop"]: r["label"] for r in json.load(open(os.path.join(OUT, "labels_video_v1.json")))["labels"]}
    hab = {json.loads(l)["crop"]: json.loads(l)["habitat"] for l in open(os.path.join(OUT, "video_crops_index.jsonl"))}
    seed_set = set(json.load(open(os.path.join(OUT, "seed_crops.json"))))

    # train auto-labeler = seed(video) + confirmed still labels (same as deployment)
    X, y = [], []
    for c, l in seed_lab.items():
        if c in EMB:
            X.append(EMB[c]); y.append(l)
    for r in json.load(open(os.path.join(OUT, "labels_still_v1.json")))["labels"]:
        if r["crop"] in EMB:
            X.append(EMB[r["crop"]]); y.append(r["label"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(np.stack(X), y)
    classes = list(clf.classes_)

    # evaluation set: video crops NOT in seed (seed-out)
    crops = [it["crop"] for it in vit if it["crop"] not in seed_set and it["crop"] in final]
    Z = np.stack([EMB[c] for c in crops])
    P = clf.predict_proba(Z)
    pred = [classes[i] for i in P.argmax(1)]
    true = [final[c] for c in crops]
    n = len(crops)
    conf = P.max(1)
    ent = -(P * np.log(P + 1e-9)).sum(1)
    part = np.sort(P, 1)
    margin = part[:, -1] - part[:, -2]
    rare_idx = [i for i, t in enumerate(true) if t not in COMMON and t != "誤検出(動物なし)"]
    base_acc = float(np.mean([pred[i] == true[i] for i in range(n)]))

    # review orders (indices to review FIRST). random = mean over 20 shuffles.
    orders = {
        "confidence": np.argsort(conf),            # least-confident first
        "entropy": np.argsort(-ent),               # highest-entropy first
        "margin": np.argsort(margin),              # smallest-margin first
    }

    rare_mask = np.zeros(n, bool); rare_mask[rare_idx] = True

    def curve(order_idx):
        accs, rares = [], []
        for B in BUDGETS:
            k = int(round(B / 100 * n)); rev = set(order_idx[:k].tolist())
            got = np.array([true[i] if i in rev else pred[i] for i in range(n)])
            corr = np.array([got[i] == true[i] for i in range(n)])
            accs.append(float(corr.mean()))
            rares.append(float(corr[rare_mask].mean()) if rare_mask.any() else float("nan"))
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        xs = [b / 100 for b in BUDGETS]
        return accs, round(float(trap(accs, xs)), 4), rares, round(float(trap(rares, xs)), 4)

    res = {"n_eval_seed_out": n, "n_rare_seedout": len(rare_idx), "auto_accuracy_seedout": round(base_acc, 3),
           "note": "review->truth is an UPPER BOUND under perfect review; compare ORDERS at fixed budget",
           "orders": {}}
    conf_auc = None
    for name, idx in orders.items():
        accs, auc, rares, rauc = curve(idx)
        res["orders"][name] = {"acc_by_budget": [round(a, 3) for a in accs], "auc_acc_budget": auc,
                               "rare_recall_auc": rauc}
        if name == "confidence":
            conf_auc = auc
    # random baseline: mean+std over shuffles + permutation p-value vs confidence
    B_SH = 1000
    rand_auc = []; rand_racc = np.zeros(len(BUDGETS))
    for _ in range(B_SH):
        idx = RNG.permutation(n)
        a, u, r, ru = curve(idx); rand_auc.append(u); rand_racc += np.array(r)
    pval = (1 + sum(1 for u in rand_auc if u >= conf_auc)) / (B_SH + 1)
    res["orders"]["random"] = {"auc_acc_budget": round(float(np.mean(rand_auc)), 4),
                               "auc_std": round(float(np.std(rand_auc)), 4),
                               "rare_recall_auc": round(float(np.trapezoid(rand_racc / B_SH, [b/100 for b in BUDGETS]) if hasattr(np,"trapezoid") else np.trapz(rand_racc/B_SH,[b/100 for b in BUDGETS])), 4)}
    res["confidence_vs_random_perm_p"] = round(float(pval), 5)

    # detailed confidence-order table with rare recall + composition TV (seed-out)
    detail = []
    ci = np.argsort(conf)
    for B in BUDGETS:
        k = int(round(B / 100 * n)); rev = set(ci[:k].tolist())
        got = [true[i] if i in rev else pred[i] for i in range(n)]
        acc = float(np.mean([got[i] == true[i] for i in range(n)]))
        rr = float(np.mean([got[i] == true[i] for i in rare_idx])) if rare_idx else float("nan")
        tvs = []
        for h in set(hab.get(crops[i]) for i in range(n)):
            ix = [i for i in range(n) if hab.get(crops[i]) == h]
            if len(ix) < 20:
                continue
            keys = set(true[i] for i in ix) | set(got[i] for i in ix)
            tvs.append(tv(normc(Counter(got[i] for i in ix)), normc(Counter(true[i] for i in ix)), keys))
        detail.append({"budget_pct": B, "accuracy_ub": round(acc, 3), "rare_recall_ub": round(rr, 3),
                       "habitat_TV": round(float(np.mean(tvs)), 3) if tvs else None})
    res["confidence_detail"] = detail

    # labeling efficiency (seed-out) with Wilson CI per confidence band
    bands = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    eff = []
    for lo, hi in bands:
        m = (conf >= lo) & (conf < hi)
        nb = int(m.sum()); corr = int(sum(pred[i] != true[i] for i in range(n) if lo <= conf[i] < hi))
        eff.append({"band": f"[{lo},{hi})", "n": nb, "corrected": corr,
                    "rate": round(corr / nb, 3) if nb else None, "wilson95": wilson(corr, nb)})
    res["efficiency_bands"] = eff

    json.dump(res, open(os.path.join(BMK, "pipeline_experiments.json"), "w"), ensure_ascii=False, indent=2)
    print(f"seed-out eval n={n}, rare={len(rare_idx)}, auto-acc={base_acc:.3f}")
    print("=== review ORDER comparison (AUC of accuracy-vs-budget, higher=better) ===")
    for name, d in res["orders"].items():
        print(f"  {name:<11} AUC={d['auc_acc_budget']}  acc@budget={d['acc_by_budget']}")
    print("=== confidence-order detail (upper bound) ===")
    for d in detail:
        print(f"  B={d['budget_pct']:>3}% accUB={d['accuracy_ub']} rareRecallUB={d['rare_recall_ub']} habitatTV={d['habitat_TV']}")
    print("=== efficiency bands (Wilson 95% CI) ===")
    for e in eff:
        print(f"  {e['band']:<10} n={e['n']:>4} rate={e['rate']} CI={e['wilson95']}")


if __name__ == "__main__":
    main()
