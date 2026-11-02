#!/usr/bin/env python3
"""Experiments requested by the Applied Intelligence R1 reviews (APIN-D-26-11064).

Parts, run per backbone (bioclip, dinov2):
  metrics    seed-only and full training: accuracy, macro-F1, balanced accuracy
  static     single-model review orders, including true BADGE
             (k-means++ seeding on full-dimensional gradient embeddings)
  bootstrap  paired 95% CIs at B=10/30 by crop and by source video (N=2000)
  iterative  retrain-after-each-batch comparison (batch = 5% of pool, up to 50%)

Rare = identified species outside COMMON (the 249 "true rare" crops).
Evaluation labels are used only to simulate the oracle and to score.
Outputs are new files in outputs/benchmark/r1/; existing files are never overwritten.

Usage: nice -n 10 .venv/bin/python exp_r1_revision.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score

HOME = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HOME, "outputs")
R1 = os.path.join(OUT, "benchmark", "r1")
COMMON = {"ニホンジカ", "タヌキ", "イノシシ", "ニホンザル", "ニホンノウサギ"}
UNKNOWN, FALSE_DET = "不明", "誤検出(動物なし)"
EXCLUDED = COMMON | {UNKNOWN, FALSE_DET}
BUDGETS = [0, 5, 10, 20, 30, 50, 100]
N_BOOT = 2000
SEED = 20261101
sys.path.insert(0, HOME)


def fit(X: np.ndarray, y: list[str]) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(X, y)


def load(backbone: str) -> dict:
    labels = lambda name: {r["crop"]: r["label"] for r in json.load(open(os.path.join(OUT, name)))["labels"]}
    seed, final = labels("seed_labels.json"), labels("labels_video_v1.json")
    still = json.load(open(os.path.join(OUT, "labels_still_v1.json")))["labels"]
    vit = json.load(open(os.path.join(OUT, "video_embed_index.json")))
    seed_set = set(json.load(open(os.path.join(OUT, "seed_crops.json"))))
    emb: dict[str, np.ndarray] = {}
    if backbone == "bioclip":
        se = np.load(os.path.join(OUT, "embeddings.npy")).astype(np.float64)
        for i, r in enumerate(json.load(open(os.path.join(OUT, "provisional_labels.json")))):
            emb[r["crop"]] = se[i]
        ve = np.load(os.path.join(OUT, "video_embeddings.npy")).astype(np.float64)
        for i, it in enumerate(vit):
            emb[it["crop"]] = ve[i]
    else:
        de = np.load(os.path.join(OUT, "dinov2_embeddings.npy")).astype(np.float64)
        for i, c in enumerate(json.load(open(os.path.join(OUT, "dinov2_embed_index.json")))):
            emb[c] = de[i]
    seed_crops = [c for c in seed if c in emb]
    still_rows = [r for r in still if r["crop"] in emb]
    crops = [it["crop"] for it in vit if it["crop"] not in seed_set and it["crop"] in final]
    index = {}
    for line in open(os.path.join(OUT, "video_crops_index.jsonl")):
        r = json.loads(line)
        index[r["crop"]] = r
    return {
        "Z_seed": np.stack([emb[c] for c in seed_crops]), "y_seed": [seed[c] for c in seed_crops],
        "Z_still": np.stack([emb[r["crop"]] for r in still_rows]), "y_still": [r["label"] for r in still_rows],
        "Z": np.stack([emb[c] for c in crops]), "y": np.array([final[c] for c in crops]),
        "videos": np.array([index.get(c, {}).get("src") or c for c in crops]),
    }


def scores(pred: np.ndarray, y: np.ndarray) -> dict:
    return {"accuracy": round(float((pred == y).mean()), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4)}


def rcta_score(P: np.ndarray, classes: list[str]) -> np.ndarray:
    w = np.array([0.0 if c in EXCLUDED else 1.0 for c in classes])
    return P @ w


def gradient_embedding(Z: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Hypothetical-label gradient of the softmax last layer, full dimension (float32)."""
    delta = P.copy()
    delta[np.arange(len(P)), P.argmax(axis=1)] -= 1.0
    return (delta[:, :, None].astype(np.float32) * Z[:, None, :].astype(np.float32)).reshape(len(P), -1)


def kmeanspp(G: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding as in BADGE; the selection order is the review order."""
    n = len(G)
    first = int(np.argmax((G * G).sum(axis=1)))
    chosen = [first]
    d2 = ((G - G[first]) ** 2).sum(axis=1).astype(np.float64)
    d2[first] = 0.0
    taken = np.zeros(n, dtype=bool)
    taken[first] = True
    while len(chosen) < k:
        total = d2.sum()
        if total > 0:
            idx = int(rng.choice(n, p=d2 / total))
        else:  # only exact duplicates remain
            idx = int(rng.choice(np.flatnonzero(~taken)))
        chosen.append(idx)
        taken[idx] = True
        d2 = np.minimum(d2, ((G - G[idx]) ** 2).sum(axis=1).astype(np.float64))
        d2[taken] = 0.0
    return np.array(chosen)


def kcenter(Z_pool: np.ndarray, Z_lab: np.ndarray, k: int) -> np.ndarray:
    Zp, Zl = Z_pool.astype(np.float32), Z_lab.astype(np.float32)
    # Squared distance through inner products keeps memory bounded.
    d = np.full(len(Zp), np.inf, dtype=np.float32)
    pn = (Zp * Zp).sum(axis=1)
    for s in range(0, len(Zl), 1024):
        b = Zl[s:s + 1024]
        d = np.minimum(d, (pn[:, None] - 2 * Zp @ b.T + (b * b).sum(axis=1)).min(axis=1))
    chosen = []
    for _ in range(k):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, ((Zp - Zp[i]) ** 2).sum(axis=1))
        d[chosen] = -np.inf
    return np.array(chosen)


def uncertainty_rank(P: np.ndarray, classes: list[str], name: str) -> np.ndarray:
    if name == "confidence":
        return np.argsort(P.max(axis=1), kind="stable")
    if name == "entropy":
        return np.argsort((P * np.log(P + 1e-9)).sum(axis=1), kind="stable")
    if name == "margin":
        s = np.sort(P, axis=1)
        return np.argsort(s[:, -1] - s[:, -2], kind="stable")
    if name == "rcta":
        return np.argsort(-rcta_score(P, classes), kind="stable")
    raise ValueError(name)


def curve_static(order: np.ndarray, pred: np.ndarray, y: np.ndarray, rare: np.ndarray) -> dict:
    n, rows = len(y), {}
    for b in BUDGETS:
        cut = round(n * b / 100)
        p = pred.copy()
        p[order[:cut]] = y[order[:cut]]
        rows[str(b)] = {**scores(p, y), "rare_capture": round(float(np.isin(np.flatnonzero(rare), order[:cut]).mean()), 4)}
    acc = [rows[str(b)]["accuracy"] for b in BUDGETS]
    return {"by_budget": rows, "auc_accuracy": round(float(np.trapezoid(acc, BUDGETS) / 100), 4)}


def boot_gap(a: np.ndarray, b: np.ndarray, pred: np.ndarray, y: np.ndarray, rare: np.ndarray,
             groups: np.ndarray | None, rng: np.random.Generator, n_boot: int) -> dict:
    """Paired gaps a-b. Crop weights are resampled; review positions stay fixed."""
    n = len(y)
    correct = pred == y
    def pos(o: np.ndarray) -> np.ndarray:
        p = np.empty(n, dtype=np.int64)
        p[o] = np.arange(n)
        return p
    pa, pb = pos(a), pos(b)
    out: dict = {}
    if groups is not None:
        _, gid = np.unique(groups, return_inverse=True)
        n_g = gid.max() + 1
    gaps: dict[str, list[float]] = {f"rare_capture_B{b}": [] for b in (10, 30)}
    gaps.update({"accuracy_auc": []})
    cuts = {b: round(n * b / 100) for b in BUDGETS}
    for _ in range(n_boot):
        if groups is None:
            w = rng.multinomial(n, np.full(n, 1 / n)).astype(np.float64)
        else:
            w = np.bincount(rng.integers(0, n_g, n_g), minlength=n_g)[gid].astype(np.float64)
        wr = w[rare].sum()
        for bud in (10, 30):
            cut = cuts[bud]
            gaps[f"rare_capture_B{bud}"].append(
                float((w * rare * (pa < cut)).sum() - (w * rare * (pb < cut)).sum()) / wr if wr else 0.0)
        acc_a = [float((w * (correct | (pa < cuts[x]))).sum() / w.sum()) for x in BUDGETS]
        acc_b = [float((w * (correct | (pb < cuts[x]))).sum() / w.sum()) for x in BUDGETS]
        gaps["accuracy_auc"].append(float(np.trapezoid(acc_a, BUDGETS) - np.trapezoid(acc_b, BUDGETS)) / 100)
    for k, v in gaps.items():
        out[k] = {"mean": round(float(np.mean(v)), 4),
                  "ci95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]}
    return out


def run_iterative(strategy: str, d: dict, Z_base: np.ndarray, y_base: list[str], rare: np.ndarray,
                  rng: np.random.Generator, rounds: int) -> dict:
    Z, y = d["Z"], d["y"]
    n = len(y)
    batch = round(n * 0.05)
    pool = np.arange(n)
    reviewed = np.zeros(n, dtype=bool)
    rows: dict = {}
    for r in range(rounds + 1):
        lab = np.flatnonzero(reviewed)
        clf = fit(np.vstack([Z_base, Z[lab]]), list(y_base) + y[lab].tolist())
        classes = [str(c) for c in clf.classes_]
        P_all = clf.predict_proba(Z)
        pred = np.array(classes)[P_all.argmax(axis=1)]
        pred[lab] = y[lab]
        rows[str(5 * r)] = {**scores(pred, y), "rare_capture": round(float(reviewed[rare].mean()), 4)}
        if r == rounds:
            break
        pool = np.flatnonzero(~reviewed)
        P = P_all[pool]
        if strategy == "random":
            pick = rng.choice(len(pool), batch, replace=False)
        elif strategy == "coreset":
            pick = kcenter(Z[pool], np.vstack([Z_base, Z[lab]]), batch)
        elif strategy == "badge":
            pick = kmeanspp(gradient_embedding(Z[pool], P), batch, rng)
        else:
            pick = uncertainty_rank(P, classes, strategy)[:batch]
        reviewed[pool[pick]] = True
    keys = sorted(rows, key=int)
    acc = [rows[k]["accuracy"] for k in keys]
    return {"by_budget": rows, "auc_accuracy_to_B50": round(float(np.trapezoid(acc, [int(k) for k in keys]) / int(keys[-1])), 4)}


def save(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = "_smoke" if args.smoke else ""
    os.makedirs(R1, exist_ok=True)
    n_boot, seeds_static, seeds_iter, rounds = (20, 1, 1, 1) if args.smoke else (N_BOOT, 3, 5, 10)
    for backbone in ("bioclip", "dinov2"):
        path = os.path.join(R1, f"r1_{backbone}{tag}.json")
        if os.path.exists(path):
            sys.exit(f"Refusing to overwrite {path}")
        t0 = time.time()
        d = load(backbone)
        y = d["y"]
        n = len(y)
        rare = np.array([v not in EXCLUDED for v in y])
        Z_base = np.vstack([d["Z_seed"], d["Z_still"]])
        y_base = d["y_seed"] + d["y_still"]
        full = fit(Z_base, y_base)
        classes = [str(c) for c in full.classes_]
        P = full.predict_proba(d["Z"])
        pred = np.array(classes)[P.argmax(axis=1)]
        seed_clf = fit(d["Z_seed"], d["y_seed"])
        seed_pred = np.array([str(c) for c in seed_clf.classes_])[seed_clf.predict_proba(d["Z"]).argmax(axis=1)]
        counts = Counter(y_base)
        state: dict = {
            "backbone": backbone, "n_eval": n, "n_rare_identified": int(rare.sum()), "dim": int(d["Z"].shape[1]),
            "n_train_full": len(y_base), "n_train_seed": len(d["y_seed"]),
            "train_top5_excluding_unknown_and_false_detection":
                [c for c, _ in counts.most_common() if c not in (UNKNOWN, FALSE_DET)][:5],
            "COMMON_used": sorted(COMMON),
            "metrics": {"seed_only": {"n_classes": len(seed_clf.classes_), **scores(seed_pred, y)},
                        "full": {"n_classes": len(classes), **scores(pred, y)}},
        }
        save(path, state)
        print(backbone, state["metrics"], state["train_top5_excluding_unknown_and_false_detection"], flush=True)

        orders = {name: uncertainty_rank(P, classes, name) for name in ("confidence", "entropy", "margin", "rcta")}
        from pipeline_canonical import badge_order
        orders["coreset"] = kcenter(d["Z"], Z_base, n)
        orders["badge_approx_pca20"] = badge_order(d["Z"], P, np.random.default_rng(6906), pca_components=20)
        state["static"] = {k: curve_static(v, pred, y, rare) for k, v in orders.items()}
        rnd = np.random.default_rng(SEED)
        state["static"]["random_mean_5"] = [curve_static(rnd.permutation(n), pred, y, rare)["auc_accuracy"] for _ in range(5)]
        save(path, state)
        print(backbone, "static done", round(time.time() - t0), flush=True)

        G = gradient_embedding(d["Z"], P)
        true_orders = []
        for s in range(seeds_static):
            ts = time.time()
            true_orders.append(kmeanspp(G, n, np.random.default_rng(SEED + s)))
            state["static"].setdefault("badge_true", []).append(curve_static(true_orders[-1], pred, y, rare))
            save(path, state)
            print(backbone, "true BADGE seed", s, state["static"]["badge_true"][-1]["auc_accuracy"], round(time.time() - ts), flush=True)
        del G
        orders["badge_true_seed0"] = true_orders[0]

        boot: dict = {}
        for grp_name, grp in (("by_crop", None), ("by_source_video", d["videos"])):
            boot[grp_name] = {}
            for a, b in (("rcta", "confidence"), ("confidence", "coreset"), ("confidence", "badge_approx_pca20"), ("confidence", "badge_true_seed0")):
                boot[grp_name][f"{a}_minus_{b}"] = boot_gap(orders[a], orders[b], pred, y, rare, grp,
                                                            np.random.default_rng(SEED), n_boot)
        state["bootstrap"] = boot
        save(path, state)
        print(backbone, "bootstrap done", round(time.time() - t0), flush=True)

        state["iterative"] = {"batch_pct_of_pool": 5, "max_budget_pct": 5 * rounds}
        for strat in ("confidence", "entropy", "margin", "rcta", "coreset", "badge", "random"):
            ts = time.time()
            stochastic = strat in ("badge", "random")
            runs = [run_iterative(strat, d, Z_base, y_base, rare, np.random.default_rng(SEED + s), rounds)
                    for s in range(seeds_iter if stochastic else 1)]
            state["iterative"][strat] = runs
            save(path, state)
            print(backbone, "iterative", strat, [x["auc_accuracy_to_B50"] for x in runs], round(time.time() - ts), flush=True)
        print(backbone, "finished", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
