"""
Additional experiments for the pipeline paper.

Experiments:
  Exp1: Coreset (k-center greedy) and BADGE (approx.) AL baselines vs confidence/margin/random
      → outputs/benchmark/pipeline_al_baselines.json
  Exp2: Per-band rare species recall + high-confidence misclassification count
      → outputs/benchmark/pipeline_rare_recall_by_band.json
  Exp3: Ecological KPI (TV / Shannon / Spearman) with random-order permutation control
      → outputs/benchmark/pipeline_ecology_random_control.json
  Exp4: 10,000-shuffle permutation p-value for confidence vs random AUC
      (included in pipeline_al_baselines.json, field: confidence_vs_random_perm_p_10k)

Reproducibility contract:
  - Same data loading, classifier, eval split as pipeline_experiments.py
  - Classifier: LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
  - Eval set : seed-out video crops (crops NOT in seed_crops.json with ground truth in labels_video_v1.json)
  - COMMON    : {'ニホンジカ', 'タヌキ', 'イノシシ', 'ニホンザル', 'ニホンノウサギ'}
  - Rare species: true label not in COMMON and not '誤検出(動物なし)'
  - Coreset RNG seed: np.random.default_rng(seed=6906)  # matches pipeline seed-out size label
  - BADGE   RNG seed: np.random.default_rng(seed=6906)
  - Permutation test RNG seed: np.random.default_rng(seed=6906)
  - Ecology permutation RNG seed: np.random.default_rng(seed=6906)

Reads (never modified):
  outputs/seed_labels.json, outputs/labels_video_v1.json, outputs/labels_still_v1.json,
  outputs/video_embeddings.npy, outputs/video_embed_index.json,
  outputs/embeddings.npy, outputs/provisional_labels.json,
  outputs/video_crops_index.jsonl, outputs/seed_crops.json

Writes (new files only):
  outputs/benchmark/pipeline_al_baselines.json
  outputs/benchmark/pipeline_rare_recall_by_band.json
  outputs/benchmark/pipeline_ecology_random_control.json
"""

import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HOME = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HOME, "outputs")
BMK = os.path.join(OUT, "benchmark")

# ---------------------------------------------------------------------------
# Constants (shared with pipeline_experiments.py)
# ---------------------------------------------------------------------------
COMMON = {"ニホンジカ", "タヌキ", "イノシシ", "ニホンザル", "ニホンノウサギ"}
BUDGETS = [0, 5, 10, 20, 30, 50, 100]
BUDGETS_ECO = [0, 10, 20, 30, 50, 100]   # B3 specification

# ---------------------------------------------------------------------------
# Reproducibility seeds (fixed, documented)
# ---------------------------------------------------------------------------
SEED_MAIN = 6906      # matches n_eval_seed_out for traceability
SEED_BADGE = 6906
SEED_PERM = 6906
SEED_ECO = 6906

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(fn(y, x))


def tv(a: Dict, b: Dict, keys) -> float:
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)


def normc(counter: Counter) -> Dict:
    t = sum(counter.values()) or 1
    return {k: v / t for k, v in counter.items()}


def shannon(counter: Counter) -> float:
    t = sum(counter.values())
    if t == 0:
        return 0.0
    ps = [v / t for v in counter.values() if v > 0]
    return float(-sum(p * np.log(p) for p in ps))


def spearman_rank(a: List[float], b: List[float]) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = a.argsort().argsort()
    rb = b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
# Data loading (identical to pipeline_experiments.py)
# ---------------------------------------------------------------------------

def load_data():
    """Load embeddings, labels, split info. Returns shared eval arrays."""
    from sklearn.linear_model import LogisticRegression

    # Embeddings
    se = np.load(os.path.join(OUT, "embeddings.npy"))
    prov = json.load(open(os.path.join(OUT, "provisional_labels.json")))
    EMB: Dict[str, np.ndarray] = {}
    for i, r in enumerate(prov):
        EMB[r["crop"]] = se[i]

    ve = np.load(os.path.join(OUT, "video_embeddings.npy"))
    vit = json.load(open(os.path.join(OUT, "video_embed_index.json")))
    for i, it in enumerate(vit):
        EMB[it["crop"]] = ve[i]

    # Labels and metadata
    seed_lab = {r["crop"]: r["label"]
                for r in json.load(open(os.path.join(OUT, "seed_labels.json")))["labels"]}
    final = {r["crop"]: r["label"]
             for r in json.load(open(os.path.join(OUT, "labels_video_v1.json")))["labels"]}
    hab = {json.loads(l)["crop"]: json.loads(l)["habitat"]
           for l in open(os.path.join(OUT, "video_crops_index.jsonl"))}
    seed_set = set(json.load(open(os.path.join(OUT, "seed_crops.json"))))

    # Train classifier (exact replica of pipeline_experiments.py)
    X_tr, y_tr = [], []
    for c, l in seed_lab.items():
        if c in EMB:
            X_tr.append(EMB[c])
            y_tr.append(l)
    for r in json.load(open(os.path.join(OUT, "labels_still_v1.json")))["labels"]:
        if r["crop"] in EMB:
            X_tr.append(EMB[r["crop"]])
            y_tr.append(r["label"])
    Z_train = np.stack(X_tr)     # training embeddings (for Coreset anchor)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    clf.fit(Z_train, y_tr)
    classes = list(clf.classes_)

    # Seed-out eval set
    crops = [it["crop"] for it in vit
             if it["crop"] not in seed_set and it["crop"] in final]
    Z_eval = np.stack([EMB[c] for c in crops])
    P = clf.predict_proba(Z_eval)
    pred = [classes[i] for i in P.argmax(1)]
    true = [final[c] for c in crops]
    n = len(crops)
    conf = P.max(1)
    ent = -(P * np.log(P + 1e-9)).sum(1)
    margin = np.sort(P, 1)[:, -1] - np.sort(P, 1)[:, -2]

    rare_mask = np.array([t not in COMMON and t != "誤検出(動物なし)" for t in true], dtype=bool)
    base_correct = np.array([pred[i] == true[i] for i in range(n)], dtype=bool)

    return dict(
        crops=crops, Z_eval=Z_eval, Z_train=Z_train, P=P,
        pred=pred, true=true, n=n, conf=conf, ent=ent, margin=margin,
        classes=classes, clf=clf, hab=hab, rare_mask=rare_mask,
        base_correct=base_correct,
    )


# ---------------------------------------------------------------------------
# AUC curve computation helpers
# ---------------------------------------------------------------------------

def compute_auc(order_idx: np.ndarray, base_correct: np.ndarray,
                n: int, budgets: List[int],
                rare_mask: np.ndarray) -> Tuple[List[float], float, List[float], float]:
    """Compute accuracy- and rare-recall-vs-budget curves + AUC values."""
    xs = [b / 100 for b in budgets]
    accs, rares = [], []
    for B in budgets:
        K = int(round(B / 100 * n))
        reviewed = set(order_idx[:K].tolist())
        correct = base_correct.copy()
        for i in reviewed:
            correct[i] = True        # reviewed crops are corrected to ground truth
        accs.append(float(correct.mean()))
        if rare_mask.any():
            rares.append(float(correct[rare_mask].mean()))
        else:
            rares.append(float("nan"))
    auc_acc = round(_trapz(np.array(accs), np.array(xs)), 4)
    auc_rare = round(_trapz(np.array(rares), np.array(xs)), 4)
    return ([round(a, 4) for a in accs], auc_acc,
            [round(r, 4) for r in rares], auc_rare)


def fast_rand_aucs(n_shuffles: int, base_correct: np.ndarray, budgets: List[int],
                   n: int, rng: np.random.Generator, batch: int = 500) -> np.ndarray:
    """
    Vectorized AUC computation for random permutations.

    Insight: accuracy at budget K for a random permutation p equals
      acc(K) = [K + total_base - cumsum(base_correct[p])[:K]] / n
    where total_base = base_correct.sum() and cumsum is along the permutation.

    This avoids a Python loop over budgets per shuffle.
    """
    xs = np.array([b / 100.0 for b in budgets])
    Ks = np.array([int(round(b / 100 * n)) for b in budgets], dtype=int)
    dB = np.diff(xs)
    total_base = int(base_correct.sum())
    base_int = base_correct.astype(np.int32)

    aucs = np.zeros(n_shuffles)
    for start in range(0, n_shuffles, batch):
        end = min(start + batch, n_shuffles)
        nb = end - start
        # (nb, n) permutations
        perms = np.array([rng.permutation(n) for _ in range(nb)], dtype=np.int32)
        # (nb, n) base_correct in permuted order
        bc_perms = base_int[perms]
        # (nb, n+1) cumulative sums; cum[:, K] = sum of first K elements
        cum = np.hstack([np.zeros((nb, 1), dtype=np.int64),
                         np.cumsum(bc_perms, axis=1, dtype=np.int64)])
        # (nb, len(budgets)) accuracies
        accs_mat = np.zeros((nb, len(Ks)), dtype=np.float64)
        for bi, K in enumerate(Ks):
            accs_mat[:, bi] = (K + total_base - cum[:, K]) / n
        # trapz: (nb,)
        aucs[start:end] = np.sum((accs_mat[:, :-1] + accs_mat[:, 1:]) / 2.0 * dB, axis=1)
    return aucs


def bootstrap_auc_gap(order_a: np.ndarray, order_b: np.ndarray,
                      base_correct: np.ndarray, budgets: List[int],
                      n: int, n_boot: int, rng: np.random.Generator) -> Dict:
    """
    Bootstrap 95% CI on AUC(order_a) - AUC(order_b).

    Strategy: pre-compute the review-position rank for each item under both orderings.
    For each bootstrap resample, re-order the bootstrap items by their pre-computed rank
    (positions 0..n_boot-1 within the resample, not original indices).
    This avoids rerunning Coreset/BADGE from scratch.
    """
    # rank_x[i] = how early item i is reviewed under order_x (0 = first)
    rank_a = np.empty(n, dtype=np.int64)
    rank_a[order_a] = np.arange(n)
    rank_b = np.empty(n, dtype=np.int64)
    rank_b[order_b] = np.arange(n)

    xs = np.array([b / 100.0 for b in budgets])
    gaps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)          # bootstrap sample indices, shape (n,)
        n_bs = len(idx)
        bc_sub = base_correct[idx]           # base_correct for each bootstrap item

        # Positions in bootstrap sample (0..n_bs-1) sorted by pre-computed rank
        # = the bootstrap-internal review order for each method
        pos_a = np.argsort(rank_a[idx])      # pos_a[k] = bootstrap-position reviewed k-th
        pos_b = np.argsort(rank_b[idx])

        def auc_bs(positions: np.ndarray) -> float:
            # positions[k] = bootstrap-position (0..n_bs-1) reviewed at step k
            accs = []
            for B in budgets:
                K = int(round(B / 100 * n_bs))
                reviewed_positions = set(positions[:K].tolist())
                correct = bc_sub.copy()
                for pos in reviewed_positions:
                    correct[pos] = True
                accs.append(float(correct.mean()))
            return _trapz(np.array(accs), xs)

        gaps.append(auc_bs(pos_a) - auc_bs(pos_b))
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {"mean_gap": round(float(np.mean(gaps)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n_bootstrap": n_boot}


# ---------------------------------------------------------------------------
# B1/B4: Active Learning baselines (Coreset, BADGE) + 10k permutation test
# ---------------------------------------------------------------------------

def coreset_order(Z_train: np.ndarray, Z_eval: np.ndarray) -> np.ndarray:
    """
    k-center greedy ordering of Z_eval.

    Returns indices into Z_eval sorted so that the most diverse
    (farthest from training set) sample comes first.

    Training samples serve as the initial covered set.
    At each step we select the uncovered eval sample with the largest
    minimum distance to all covered points (training + previously selected).
    """
    n_eval = len(Z_eval)
    n_train = len(Z_train)
    chunk = 512

    # Initial min-distance: each eval point to its nearest training point
    min_dist = np.full(n_eval, np.inf)
    for start in range(0, n_train, chunk):
        block = Z_train[start:start + chunk]          # (c, d)
        # (n_eval, c) squared Euclidean distances
        diff = Z_eval[:, None, :] - block[None, :, :]  # (n_eval, c, d)
        d2 = (diff ** 2).sum(axis=2)                   # (n_eval, c)
        min_dist = np.minimum(min_dist, d2.min(axis=1))

    order = []
    for step in range(n_eval):
        idx = int(np.argmax(min_dist))
        order.append(idx)
        min_dist[idx] = -np.inf        # mark as selected
        # Update min-dist using newly covered point
        new_pt = Z_eval[idx]
        new_d = ((Z_eval - new_pt) ** 2).sum(axis=1)  # (n_eval,)
        min_dist = np.minimum(min_dist, new_d)
        min_dist[idx] = -np.inf        # keep selected point ineligible
    return np.array(order, dtype=np.int64)


def badge_order(Z_eval: np.ndarray, P: np.ndarray,
                rng: np.random.Generator, pca_components: int = 20) -> np.ndarray:
    """
    BADGE-approximation ordering using k-center greedy on gradient embeddings.

    Gradient embedding for logistic regression (Ash et al., 2019):
      g_i = (p_i - e_{ŷ_i}) ⊗ z_i

    where p_i is the softmax output, e_{ŷ_i} is the one-hot of the predicted class,
    and z_i is the input embedding.  For K=19 classes and d=512 this gives 9728-dim
    vectors; we reduce z to 20 PCA dims first, yielding 380-dim gradient embeddings.
    The k-center greedy strategy applied to this space produces a deterministic
    ordering (most uncertain+diverse first) compatible with AUC comparison.
    """
    from sklearn.decomposition import PCA

    n_eval, d = Z_eval.shape
    K = P.shape[1]
    pred_classes = P.argmax(axis=1)

    # PCA reduction on eval embeddings (20 components)
    actual_components = min(pca_components, d, n_eval)
    pca = PCA(n_components=actual_components, random_state=42)
    Z_proj = pca.fit_transform(Z_eval)          # (n_eval, 20)

    # Gradient embeddings: delta = p - one_hot(argmax)
    delta = P.copy()                             # (n_eval, K)
    delta[np.arange(n_eval), pred_classes] -= 1.0

    # G_i = delta_i ⊗ z_proj_i  (outer product, flattened)
    # Shape: (n_eval, K * actual_components)
    G = (delta[:, :, None] * Z_proj[:, None, :]).reshape(n_eval, K * actual_components)

    # k-center greedy on gradient embedding space
    # Start: first point = item with highest gradient norm (most uncertain)
    norms = np.linalg.norm(G, axis=1)
    first = int(np.argmax(norms))

    min_dist = np.full(n_eval, np.inf)
    new_d = ((G - G[first]) ** 2).sum(axis=1)
    min_dist = np.minimum(min_dist, new_d)
    min_dist[first] = -np.inf

    order = [first]
    for _ in range(1, n_eval):
        idx = int(np.argmax(min_dist))
        order.append(idx)
        min_dist[idx] = -np.inf
        new_d = ((G - G[idx]) ** 2).sum(axis=1)
        min_dist = np.minimum(min_dist, new_d)
        min_dist[idx] = -np.inf
    return np.array(order, dtype=np.int64)


def run_b1_b4(data: Dict) -> Dict:
    """Run B1 (AL baselines) and B4 (10k permutation test)."""
    n = data["n"]
    base_correct = data["base_correct"]
    rare_mask = data["rare_mask"]
    conf = data["conf"]
    ent = data["ent"]
    margin = data["margin"]
    Z_eval = data["Z_eval"]
    Z_train = data["Z_train"]
    P = data["P"]

    rng_main = np.random.default_rng(SEED_MAIN)
    rng_badge = np.random.default_rng(SEED_BADGE)
    rng_perm = np.random.default_rng(SEED_PERM)

    # ----- Deterministic orderings -----
    print("[B1] Computing standard orderings (confidence / entropy / margin)...", flush=True)
    orders_det = {
        "confidence": np.argsort(conf),          # least-confident first
        "entropy":    np.argsort(-ent),           # highest-entropy first
        "margin":     np.argsort(margin),         # smallest-margin first
    }

    print("[B1] Computing Coreset (k-center greedy) ordering...", flush=True)
    t0 = time.time()
    orders_det["coreset"] = coreset_order(Z_train, Z_eval)
    print(f"      Coreset done in {time.time()-t0:.1f}s", flush=True)

    print("[B1] Computing BADGE (gradient embedding k-center greedy) ordering...", flush=True)
    t0 = time.time()
    orders_det["badge_approx"] = badge_order(Z_eval, P, rng_badge, pca_components=20)
    print(f"      BADGE done in {time.time()-t0:.1f}s", flush=True)

    # ----- AUC for all deterministic orderings -----
    result_orders = {}
    for name, idx in orders_det.items():
        accs, auc_acc, rares, auc_rare = compute_auc(idx, base_correct, n, BUDGETS, rare_mask)
        result_orders[name] = {
            "acc_by_budget": accs,
            "auc_acc_budget": auc_acc,
            "rare_recall_by_budget": rares,
            "rare_recall_auc": auc_rare,
        }

    conf_auc = result_orders["confidence"]["auc_acc_budget"]

    # ----- B4: 10,000-shuffle permutation test (vectorized) -----
    print("[B4] Running 10,000-shuffle permutation test (vectorized)...", flush=True)
    t0 = time.time()
    N_PERM = 10_000
    rand_aucs = fast_rand_aucs(N_PERM, base_correct, BUDGETS, n, rng_perm, batch=500)
    pval_10k = float((1 + np.sum(rand_aucs >= conf_auc)) / (N_PERM + 1))
    rand_mean = round(float(rand_aucs.mean()), 4)
    rand_std = round(float(rand_aucs.std()), 4)
    print(f"      10k perm done in {time.time()-t0:.1f}s  "
          f"rand={rand_mean}±{rand_std}  p={pval_10k:.5f}", flush=True)

    # Compute rare_recall_auc for random (mean over shuffles, approx)
    # Use a small separate sample to compute rare recall
    rng_rare = np.random.default_rng(SEED_MAIN + 1)
    N_RARE_SHUFFLES = 2000
    rare_aucs = []
    xs = np.array([b / 100.0 for b in BUDGETS])
    for _ in range(N_RARE_SHUFFLES):
        idx = rng_rare.permutation(n)
        _, _, rares, auc_rare = compute_auc(idx, base_correct, n, BUDGETS, rare_mask)
        rare_aucs.append(auc_rare)
    rare_auc_rand_mean = round(float(np.mean(rare_aucs)), 4)
    rare_auc_rand_std = round(float(np.std(rare_aucs)), 4)

    result_orders["random"] = {
        "auc_acc_budget": rand_mean,
        "auc_acc_budget_std": rand_std,
        "rare_recall_auc": rare_auc_rand_mean,
        "rare_recall_auc_std": rare_auc_rand_std,
        "n_shuffles": N_PERM,
    }

    # ----- Bootstrap CI: confidence vs Coreset, confidence vs BADGE -----
    # Both methods are deterministic; bootstrap resamples crops to get CI on AUC gap.
    print("[B1] Bootstrap CI for confidence vs Coreset / BADGE (n_boot=500)...", flush=True)
    N_BOOT = 500
    rng_boot_cs = np.random.default_rng(SEED_MAIN + 100)
    rng_boot_bg = np.random.default_rng(SEED_MAIN + 200)
    ci_vs_coreset = bootstrap_auc_gap(
        orders_det["confidence"], orders_det["coreset"],
        base_correct, BUDGETS, n, N_BOOT, rng_boot_cs)
    ci_vs_badge = bootstrap_auc_gap(
        orders_det["confidence"], orders_det["badge_approx"],
        base_correct, BUDGETS, n, N_BOOT, rng_boot_bg)
    print("[B1] Bootstrap done.", flush=True)

    out = {
        "experiment": "B1_B4_al_baselines",
        "n_eval_seed_out": n,
        "n_rare_seedout": int(rare_mask.sum()),
        "budgets_pct": BUDGETS,
        "note": (
            "Coreset = k-center greedy on BioCLIP embedding space; "
            "BADGE_approx = k-center greedy on logistic-regression gradient "
            "embedding (PCA-20 projected, K=19 classes → 380-dim); "
            "review order = selected first → most informative/diverse first."
        ),
        "orders": result_orders,
        "confidence_vs_random_perm_p_10k": round(pval_10k, 5),
        "n_shuffles_permutation": N_PERM,
        "bootstrap_auc_gap_confidence_minus": {
            "coreset": ci_vs_coreset,
            "badge_approx": ci_vs_badge,
            "note": (
                "Bootstrap CI on AUC(confidence) - AUC(method). "
                "Positive gap = confidence ordering achieves higher AUC. "
                "CI excludes 0 → difference is significant at alpha=0.05."
            ),
        },
    }
    return out


# ---------------------------------------------------------------------------
# B2: Per-band rare species recall
# ---------------------------------------------------------------------------

def run_b2(data: Dict) -> Dict:
    """Rare species recall and high-confidence misclassification by confidence band."""
    n = data["n"]
    conf = data["conf"]
    pred = data["pred"]
    true = data["true"]
    rare_mask = data["rare_mask"]

    bands = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    band_results = []

    # High-confidence misclassification of rare species (c >= 0.9, pred != true, true is rare)
    high_conf_rare_misclass = [
        {"index": i, "true": true[i], "pred": pred[i], "conf": round(float(conf[i]), 4)}
        for i in range(n)
        if conf[i] >= 0.9 and pred[i] != true[i] and rare_mask[i]
    ]

    for lo, hi in bands:
        in_band = (conf >= lo) & (conf < hi)
        in_band_rare = in_band & rare_mask
        n_all_band = int(in_band.sum())
        n_rare_band = int(in_band_rare.sum())

        if n_rare_band > 0:
            n_rare_correct = int(sum(pred[i] == true[i] for i in range(n)
                                    if in_band_rare[i]))
            recall = round(n_rare_correct / n_rare_band, 4)
            w = wilson(n_rare_correct, n_rare_band)
        else:
            n_rare_correct = 0
            recall = None
            w = (None, None)

        # High-confidence misclassification count for this band
        n_highconf_rare_misclass = int(sum(
            1 for i in range(n) if in_band_rare[i] and pred[i] != true[i]
            and conf[i] >= 0.9
        ))

        band_results.append({
            "band": f"[{lo},{hi})",
            "n_all_in_band": n_all_band,
            "n_rare_in_band": n_rare_band,
            "n_rare_correct_in_band": n_rare_correct,
            "rare_recall_in_band": recall,
            "wilson95_rare_recall": list(w),
            "n_rare_highconf_misclass_in_band": n_highconf_rare_misclass,
        })

    # Overall rare species precision (just for reference)
    overall_rare_correct = int(sum(1 for i in range(n) if rare_mask[i] and pred[i] == true[i]))
    overall_rare_total = int(rare_mask.sum())

    out = {
        "experiment": "B2_rare_recall_by_band",
        "n_eval_seed_out": n,
        "n_rare_seedout": overall_rare_total,
        "n_rare_correct_overall": overall_rare_correct,
        "rare_recall_overall": round(overall_rare_correct / overall_rare_total, 4)
            if overall_rare_total > 0 else None,
        "rare_definition": "true label not in COMMON and not '誤検出(動物なし)'",
        "COMMON_species": sorted(COMMON),
        "confidence_threshold_highconf": 0.9,
        "bands": band_results,
        "high_confidence_rare_misclassifications": {
            "threshold": 0.9,
            "n_total": len(high_conf_rare_misclass),
            "examples": sorted(high_conf_rare_misclass,
                               key=lambda x: -x["conf"])[:10],
        },
    }
    return out


# ---------------------------------------------------------------------------
# Exp 3: Ecology KPI with random-order permutation control
# ---------------------------------------------------------------------------

def compute_eco_kpis(review_order_idx: np.ndarray, base_correct: np.ndarray,
                     pred: List[str], true: List[str], crops: List[str],
                     hab: Dict[str, str], habs_valid: List[str],
                     budgets: List[int], n: int) -> Dict[int, Dict]:
    """
    Compute TV, Shannon abs error, and Spearman rank for each budget.
    Returns {budget: {tv, shannon_abs_err, spearman_rank}}.
    """
    result = {}
    for B in budgets:
        K = int(round(B / 100 * n))
        reviewed = set(review_order_idx[:K].tolist())
        got = [true[i] if i in reviewed else pred[i] for i in range(n)]

        tvs, sh_errs, sps = [], [], []
        for h in habs_valid:
            ix = [i for i in range(n) if hab.get(crops[i]) == h]
            if len(ix) < 20:
                continue
            keys = sorted(set(true[i] for i in ix) | set(got[i] for i in ix))
            tc = Counter(true[i] for i in ix)
            gc = Counter(got[i] for i in ix)
            tvs.append(tv(normc(gc), normc(tc), keys))
            sh_errs.append(abs(shannon(gc) - shannon(tc)))
            sps.append(spearman_rank([gc.get(k_, 0) for k_ in keys],
                                     [tc.get(k_, 0) for k_ in keys]))

        result[B] = {
            "tv_mean": round(float(np.mean(tvs)), 4) if tvs else None,
            "shannon_abs_err_mean": round(float(np.mean(sh_errs)), 4) if sh_errs else None,
            "spearman_rank_mean": round(float(np.nanmean(sps)), 4) if sps else None,
        }
    return result


def run_b3(data: Dict) -> Dict:
    """Ecology KPI random control with permutation test."""
    n = data["n"]
    conf = data["conf"]
    pred = data["pred"]
    true = data["true"]
    crops = data["crops"]
    hab = data["hab"]
    base_correct = data["base_correct"]

    rng_eco = np.random.default_rng(SEED_ECO)

    # Valid habitats (same n>=20 filter as pipeline_ecology_extra.py)
    habs_valid = [h for h in set(hab.get(c) for c in crops)
                  if sum(hab.get(crops[i]) == h for i in range(n)) >= 20]

    # Confidence ordering (least-confident first)
    ci_order = np.argsort(conf)

    # ----- Confidence-order KPIs -----
    print("[Exp 3] Computing confidence-order ecology KPIs...", flush=True)
    eco_conf = compute_eco_kpis(ci_order, base_correct, pred, true,
                                crops, hab, habs_valid, BUDGETS_ECO, n)

    # ----- Random-order KPIs (mean over N_ECO_SHUFFLES) -----
    N_ECO_SHUFFLES = 2000
    print(f"[Exp 3] Computing random-order ecology KPIs ({N_ECO_SHUFFLES} shuffles)...", flush=True)
    t0 = time.time()

    # Accumulate metrics across shuffles
    acc_tv    = {B: [] for B in BUDGETS_ECO}
    acc_sh    = {B: [] for B in BUDGETS_ECO}
    acc_sp    = {B: [] for B in BUDGETS_ECO}

    for s in range(N_ECO_SHUFFLES):
        rand_order = rng_eco.permutation(n)
        eco_s = compute_eco_kpis(rand_order, base_correct, pred, true,
                                 crops, hab, habs_valid, BUDGETS_ECO, n)
        for B in BUDGETS_ECO:
            if eco_s[B]["tv_mean"] is not None:
                acc_tv[B].append(eco_s[B]["tv_mean"])
            if eco_s[B]["shannon_abs_err_mean"] is not None:
                acc_sh[B].append(eco_s[B]["shannon_abs_err_mean"])
            if eco_s[B]["spearman_rank_mean"] is not None:
                acc_sp[B].append(eco_s[B]["spearman_rank_mean"])

    print(f"      Random eco shuffles done in {time.time()-t0:.1f}s", flush=True)

    eco_rand = {}
    for B in BUDGETS_ECO:
        eco_rand[B] = {
            "tv_mean": round(float(np.mean(acc_tv[B])), 4) if acc_tv[B] else None,
            "tv_std": round(float(np.std(acc_tv[B])), 4) if acc_tv[B] else None,
            "shannon_abs_err_mean": round(float(np.mean(acc_sh[B])), 4) if acc_sh[B] else None,
            "shannon_abs_err_std": round(float(np.std(acc_sh[B])), 4) if acc_sh[B] else None,
            "spearman_rank_mean": round(float(np.nanmean(acc_sp[B])), 4) if acc_sp[B] else None,
            "spearman_rank_std": round(float(np.nanstd(acc_sp[B])), 4) if acc_sp[B] else None,
        }

    # ----- Permutation test -----
    # For TV and Shannon (lower = better for confidence ordering):
    #   p = fraction of random shuffles with metric <= confidence metric value
    # For Spearman (higher = better):
    #   p = fraction of random shuffles with metric >= confidence metric value
    perm_test = {}
    for B in BUDGETS_ECO:
        tv_conf = eco_conf[B]["tv_mean"]
        sh_conf = eco_conf[B]["shannon_abs_err_mean"]
        sp_conf = eco_conf[B]["spearman_rank_mean"]
        # +1 correction (Phipson & Smyth 2010) to avoid reporting p=0
        p_tv = ((sum(1 for v in acc_tv[B] if v <= tv_conf) + 1) / (len(acc_tv[B]) + 1)
                if acc_tv[B] and tv_conf is not None else None)
        p_sh = ((sum(1 for v in acc_sh[B] if v <= sh_conf) + 1) / (len(acc_sh[B]) + 1)
                if acc_sh[B] and sh_conf is not None else None)
        p_sp = ((sum(1 for v in acc_sp[B] if not np.isnan(v) and v >= sp_conf) + 1) / (len(acc_sp[B]) + 1)
                if acc_sp[B] and sp_conf is not None else None)
        perm_test[B] = {
            "tv_p_value": round(float(p_tv), 4) if p_tv is not None else None,
            "shannon_p_value": round(float(p_sh), 4) if p_sh is not None else None,
            "spearman_p_value": round(float(p_sp), 4) if p_sp is not None else None,
            "note_tv_shannon": "p = fraction of random shuffles achieving lower (better) value than confidence ordering",
            "note_spearman": "p = fraction of random shuffles achieving higher (better) value than confidence ordering",
        }

    out = {
        "experiment": "B3_ecology_random_control",
        "n_eval_seed_out": n,
        "budgets_pct": BUDGETS_ECO,
        "n_eco_shuffles": N_ECO_SHUFFLES,
        "n_habitats_valid": len(habs_valid),
        "habitats_valid": sorted(habs_valid),
        "n_by_habitat": {h: int(sum(hab.get(crops[i]) == h for i in range(n)))
                         for h in habs_valid},
        "confidence_order": {str(B): eco_conf[B] for B in BUDGETS_ECO},
        "random_order": {str(B): eco_rand[B] for B in BUDGETS_ECO},
        "permutation_test": {str(B): perm_test[B] for B in BUDGETS_ECO},
    }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(BMK, exist_ok=True)
    print("=" * 60, flush=True)
    print("pipeline_extra_experiments.py", flush=True)
    print("Loading data and training classifier...", flush=True)
    t_start = time.time()
    data = load_data()
    print(f"  n_eval_seed_out={data['n']}, n_rare={data['rare_mask'].sum()}", flush=True)

    # ---- B1 + B4 ----
    print("\n--- Exp 1/4: AL baselines + 10k permutation test ---", flush=True)
    out_b1 = run_b1_b4(data)
    path_b1 = os.path.join(BMK, "pipeline_al_baselines.json")
    json.dump(out_b1, open(path_b1, "w"), ensure_ascii=False, indent=2)
    print(f"  Written: {path_b1}", flush=True)

    # ---- B2 ----
    print("\n--- Exp 2: Per-band rare species recall ---", flush=True)
    out_b2 = run_b2(data)
    path_b2 = os.path.join(BMK, "pipeline_rare_recall_by_band.json")
    json.dump(out_b2, open(path_b2, "w"), ensure_ascii=False, indent=2)
    print(f"  Written: {path_b2}", flush=True)

    # ---- B3 ----
    print("\n--- Exp 3: Ecology KPI random control ---", flush=True)
    out_b3 = run_b3(data)
    path_b3 = os.path.join(BMK, "pipeline_ecology_random_control.json")
    json.dump(out_b3, open(path_b3, "w"), ensure_ascii=False, indent=2)
    print(f"  Written: {path_b3}", flush=True)

    # ---- Summary ----
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)

    print("\n[Exp 1/4] Review order AUC comparison (accuracy-vs-budget AUC, higher=better):")
    for name, d in out_b1["orders"].items():
        std = f"  std={d.get('auc_acc_budget_std','N/A')}" if "auc_acc_budget_std" in d else ""
        rare = d.get("rare_recall_auc", "?")
        print(f"  {name:<18}  AUC_acc={d['auc_acc_budget']}{std}  AUC_rare={rare}")
    print(f"\n  confidence vs random p (10k shuffles): "
          f"{out_b1['confidence_vs_random_perm_p_10k']:.5f}")
    ci_c = out_b1["bootstrap_auc_gap_confidence_minus"]["coreset"]
    ci_b = out_b1["bootstrap_auc_gap_confidence_minus"]["badge_approx"]
    print(f"  bootstrap AUC gap confidence-Coreset:  "
          f"mean={ci_c['mean_gap']}  CI95={ci_c['ci95']}")
    print(f"  bootstrap AUC gap confidence-BADGE:    "
          f"mean={ci_b['mean_gap']}  CI95={ci_b['ci95']}")

    print("\n[Exp 2] Rare species recall by confidence band:")
    for b in out_b2["bands"]:
        print(f"  {b['band']:<12}  n_rare={b['n_rare_in_band']:>4}  "
              f"recall={b['rare_recall_in_band']}  CI95={b['wilson95_rare_recall']}")
    hc = out_b2["high_confidence_rare_misclassifications"]
    print(f"\n  High-confidence (c≥0.9) rare misclassifications: {hc['n_total']}")

    print("\n[Exp 3] Ecology KPI: confidence vs random (permutation test p-values):")
    for B in BUDGETS_ECO:
        conf_tv  = out_b3["confidence_order"][str(B)]["tv_mean"]
        rand_tv  = out_b3["random_order"][str(B)]["tv_mean"]
        p_tv     = out_b3["permutation_test"][str(B)]["tv_p_value"]
        conf_sp  = out_b3["confidence_order"][str(B)]["spearman_rank_mean"]
        rand_sp  = out_b3["random_order"][str(B)]["spearman_rank_mean"]
        p_sp     = out_b3["permutation_test"][str(B)]["spearman_p_value"]
        print(f"  B={B:>3}%  TV: conf={conf_tv} rand={rand_tv} p={p_tv}"
              f"  |  Sp: conf={conf_sp} rand={rand_sp} p={p_sp}")

    print(f"\nTotal time: {time.time()-t_start:.1f}s", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
