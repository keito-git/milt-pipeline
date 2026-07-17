#!/usr/bin/env python3
"""
rarity_aware_triage.py — RAU (Rarity-Aware Uncertainty) triage for MiLT pipeline paper.

Proposed method: RAU combines uncertainty (1 - max_softmax) with local rarity
(k-NN distance in BioCLIP embedding space, label-free) for triage ordering.

Score(i) = (1-lambda)*uncertainty_norm(i) + lambda*rarity_norm(i)
  - lambda=0: reduces to confidence ordering (existing baseline)
  - lambda=1: rarity-only ordering
  - uncertainty_norm  = minmax(1 - max_softmax)  [higher = more uncertain]
  - rarity_norm       = minmax(kNN_distance)      [higher = more isolated = rarer]

Rarity is estimated LABEL-FREE: only BioCLIP embeddings, no ground-truth labels.
kNN reference = training set embeddings (available at deployment time).

Lambda grid: {0, 0.25, 0.5, 0.75, 1.0}
Primary k: 10  (ablation k: 5, 20)

Evaluation:
  - Main: video eval (n=6906, n_rare=1480)
  - Second site: Mie stills with seed-only LR (n=4736, n_rare=195)
  - Baselines: confidence, entropy, badge_approx (PCA-20), random
  - Stats: 10k permutation test (rare recall AUC) + bootstrap CI (n=500)

Outputs:
  outputs/benchmark/rarity_aware_triage.json
  outputs/benchmark/rau_note.md

Information leak audit:
  - kNN distances use Z_train (training embeddings) as reference. NO labels used.
  - eval set and train set are disjoint (video eval crops not in seed_crops.json).
  - The rarity measure is computable at deployment time without ground truth.
  - Lambda selection: reported for all lambda grid values; best lambda selected by
    rare_recall_auc on the same eval set (same as existing comparisons).
    This is NOT a held-out test, but matches how all baselines are evaluated.
    Acknowledged as a limitation in rau_note.md.

GPU rule: CUDA_VISIBLE_DEVICES="" set before any torch import.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU rule: must precede torch

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HOME = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HOME, "outputs")
BMK = os.path.join(OUT, "benchmark")
RESULT_PATH = os.path.join(BMK, "rarity_aware_triage.json")
NOTE_PATH = os.path.join(BMK, "rau_note.md")

# ---------------------------------------------------------------------------
# Constants (identical to pipeline_canonical.py for fair comparison)
# ---------------------------------------------------------------------------
COMMON = {"ニホンジカ", "タヌキ", "イノシシ", "ニホンザル", "ニホンノウサギ"}
BUDGETS = [0, 5, 10, 20, 30, 50, 100]
BUDGETS_PARTIAL = [0, 5, 10, 20, 30]  # B <= 30% practical range

# RAU hyperparameters
LAMBDA_GRID: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
K_PRIMARY = 10
K_ABLATION = [5, 20]

# Random seeds
SEED_MAIN = 6906
SEED_PERM = 6906
SEED_BOOT = 6906
N_PERM = 10_000
N_BOOT = 500
N_RAND_RARE = 2000  # rare recall random baseline shuffles

# Minimum whole-accuracy AUC degradation allowed vs confidence (selection criterion)
ACC_DEGRADATION_THRESHOLD = 0.005

# ---------------------------------------------------------------------------
# Incremental state
# ---------------------------------------------------------------------------
_state: Dict = {}

class _NumpyEncoder(json.JSONEncoder):
    """Encode numpy scalars and booleans to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _save(key: str, value, path: str = RESULT_PATH) -> None:
    _state[key] = value
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_state, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    os.replace(tmp, path)
    print(f"  [saved] {key} → {path}", flush=True)


# ---------------------------------------------------------------------------
# Utility functions (copied from pipeline_canonical.py for independence)
# ---------------------------------------------------------------------------

def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(fn(y, x))


def compute_auc_from_order(
    order_idx: np.ndarray,
    base_correct: np.ndarray,
    n: int,
    budgets: List[int],
    rare_mask: np.ndarray,
) -> Tuple[List[float], float, List[float], float]:
    """Returns (accs, auc_acc, rares, auc_rare)."""
    xs = np.array([b / 100.0 for b in budgets])
    accs, rares = [], []
    for B in budgets:
        K = int(round(B / 100 * n))
        reviewed = set(order_idx[:K].tolist())
        correct = base_correct.copy()
        for i in reviewed:
            correct[i] = True
        accs.append(float(correct.mean()))
        rares.append(
            float(correct[rare_mask].mean()) if rare_mask.any() else float("nan")
        )
    return (
        [round(a, 4) for a in accs],
        round(_trapz(np.array(accs), xs), 4),
        [round(r, 4) for r in rares],
        round(_trapz(np.array(rares), xs), 4),
    )


def compute_knn_rarity(
    Z_train: np.ndarray, Z_eval: np.ndarray, k: int
) -> np.ndarray:
    """
    Compute k-NN Euclidean distance from each eval sample to its k-th nearest
    neighbor in the training set. Higher distance = more isolated = rarer candidate.

    Design: label-free (only embeddings used). Computable at deployment time.
    Memory-efficient: processes eval samples in chunks of 256.

    Args:
        Z_train: training embeddings (n_train, d) float64
        Z_eval:  eval embeddings    (n_eval,  d) float64
        k:       number of nearest neighbors

    Returns:
        dists: (n_eval,) float64, k-NN distances
    """
    n_eval = len(Z_eval)
    n_train = len(Z_train)
    k_actual = min(k, n_train)

    # Precompute squared norms
    train_norms_sq = (Z_train ** 2).sum(axis=1)  # (n_train,)
    eval_norms_sq = (Z_eval ** 2).sum(axis=1)    # (n_eval,)

    dists_k = np.empty(n_eval, dtype=np.float64)
    chunk_size = 256

    for start in range(0, n_eval, chunk_size):
        end = min(start + chunk_size, n_eval)
        Z_chunk = Z_eval[start:end]  # (chunk, d)
        # ||z_eval - z_train||^2 = ||z_eval||^2 - 2*(z_eval @ z_train.T) + ||z_train||^2
        cross = Z_chunk @ Z_train.T  # (chunk, n_train)
        d2 = eval_norms_sq[start:end, None] - 2.0 * cross + train_norms_sq[None, :]
        np.clip(d2, 0.0, None, out=d2)  # numerical safety for near-zero distances
        # k-th nearest neighbor squared distance via np.partition (O(n) per row)
        k_vals = np.partition(d2, k_actual - 1, axis=1)[:, k_actual - 1]
        dists_k[start:end] = np.sqrt(k_vals)

    return dists_k


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns zeros if range is 0."""
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def compute_rau_order(
    conf: np.ndarray,
    rarity: np.ndarray,
    lam: float,
) -> np.ndarray:
    """
    RAU ordering: highest score reviewed first.
    score = (1-lam)*uncertainty_norm + lam*rarity_norm
    uncertainty_norm = minmax(1 - conf), rarity_norm = minmax(rarity)
    """
    unc = minmax_normalize(1.0 - conf)
    rar = minmax_normalize(rarity)
    score = (1.0 - lam) * unc + lam * rar
    return np.argsort(-score)  # descending score = most-urgent first


def badge_approx_order(
    Z_eval: np.ndarray, P: np.ndarray, pca_components: int = 20
) -> np.ndarray:
    """
    BADGE-approx: k-center greedy on logistic-regression gradient embedding
    with PCA projection. Matches pipeline_canonical.py implementation.
    """
    n_eval, d = Z_eval.shape
    K_cls = P.shape[1]
    pred_classes = P.argmax(axis=1)

    actual_components = min(pca_components, d, n_eval)
    pca = PCA(n_components=actual_components, random_state=42)
    Z_proj = pca.fit_transform(Z_eval)  # (n_eval, actual_components)

    delta = P.copy()
    delta[np.arange(n_eval), pred_classes] -= 1.0
    G = (delta[:, :, None] * Z_proj[:, None, :]).reshape(
        n_eval, K_cls * actual_components
    )

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


def fast_rand_rare_aucs(
    n_shuffles: int,
    base_correct: np.ndarray,
    rare_mask: np.ndarray,
    budgets: List[int],
    n: int,
    rng: np.random.Generator,
    batch: int = 200,
) -> np.ndarray:
    """
    Vectorized rare recall AUC for random permutations.

    For each permutation: at budget K, rare_recall = (base_rare_correct +
    count_of_rare_incorrect_in_top_K) / n_rare.
    """
    xs = np.array([b / 100.0 for b in budgets])
    dB = np.diff(xs)
    Ks = np.array([int(round(b / 100 * n)) for b in budgets], dtype=int)

    n_rare = int(rare_mask.sum())
    base_rare_correct = int(base_correct[rare_mask].sum())
    # Sample is "improvable-rare" if it is rare AND currently incorrect
    rare_incorrect = (rare_mask & ~base_correct).astype(np.int32)  # (n,)

    aucs = np.zeros(n_shuffles)

    for start in range(0, n_shuffles, batch):
        end = min(start + batch, n_shuffles)
        nb = end - start
        perms = np.array(
            [rng.permutation(n) for _ in range(nb)], dtype=np.int32
        )  # (nb, n)

        # For each permutation, at each position i: is sample "improvable-rare"?
        ri_perms = rare_incorrect[perms]  # (nb, n) - int32
        # Cumulative count of improvable-rare samples seen up to each position
        cum_ri = np.cumsum(ri_perms, axis=1, dtype=np.int64)  # (nb, n)

        rares_mat = np.zeros((nb, len(Ks)), dtype=np.float64)
        for bi, K in enumerate(Ks):
            if K == 0:
                rares_mat[:, bi] = base_rare_correct / n_rare
            else:
                # At position K-1 (0-indexed), cumsum = count reviewed up to K samples
                rares_mat[:, bi] = (base_rare_correct + cum_ri[:, K - 1]) / n_rare

        # Trapezoidal AUC
        aucs[start:end] = np.sum(
            (rares_mat[:, :-1] + rares_mat[:, 1:]) / 2.0 * dB, axis=1
        )

    return aucs


def bootstrap_rare_auc_gap(
    order_a: np.ndarray,
    order_b: np.ndarray,
    base_correct: np.ndarray,
    rare_mask: np.ndarray,
    budgets: List[int],
    n: int,
    n_boot: int,
    rng: np.random.Generator,
    label_a: str = "A",
    label_b: str = "B",
) -> Dict:
    """
    Bootstrap 95% CI on rare_recall_AUC(order_a) - rare_recall_AUC(order_b).
    Positive gap = order_a achieves higher rare recall AUC.
    """
    xs = np.array([b / 100.0 for b in budgets])

    # Inverse permutation (rank) for fast bootstrap resampling
    rank_a = np.empty(n, dtype=np.int64)
    rank_a[order_a] = np.arange(n)
    rank_b = np.empty(n, dtype=np.int64)
    rank_b[order_b] = np.arange(n)

    gaps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        n_bs = len(idx)
        bc_sub = base_correct[idx]
        rm_sub = rare_mask[idx]

        pos_a = np.argsort(rank_a[idx])
        pos_b = np.argsort(rank_b[idx])

        def _auc_rare(positions: np.ndarray) -> float:
            rares_bs = []
            for B in budgets:
                K = int(round(B / 100 * n_bs))
                rev = set(positions[:K].tolist())
                c = bc_sub.copy()
                for pos in rev:
                    c[pos] = True
                n_rare_bs = int(rm_sub.sum())
                rares_bs.append(
                    float(c[rm_sub].mean()) if n_rare_bs > 0 else float("nan")
                )
            return _trapz(np.array(rares_bs), xs)

        gaps.append(_auc_rare(pos_a) - _auc_rare(pos_b))

    lo, hi = float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))
    sign = "significant" if (lo > 0 or hi < 0) else "NOT significant"
    return {
        "label_a": label_a,
        "label_b": label_b,
        "mean_gap_rare_auc": round(float(np.mean(gaps)), 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "n_bootstrap": n_boot,
        "significance_at_alpha05": bool(lo > 0),
        "interpretation": (
            f"Bootstrap CI on rare_recall_AUC({label_a}) - rare_recall_AUC({label_b}). "
            f"Positive gap = {label_a} achieves higher rare recall AUC. {sign} at alpha=0.05."
        ),
    }


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Step 1: Load data (same logic as pipeline_canonical.py, float64 throughout)
# ---------------------------------------------------------------------------

def load_data() -> Dict:
    """Load all data and train canonical 19-class LR classifier."""
    print("\n[Step 1] Loading data and training canonical LR (float64)...", flush=True)
    t0 = time.time()

    # Load still embeddings
    se = np.load(os.path.join(OUT, "embeddings.npy")).astype(np.float64)
    prov = json.load(open(os.path.join(OUT, "provisional_labels.json")))
    EMB: Dict[str, np.ndarray] = {}
    for i, r in enumerate(prov):
        EMB[r["crop"]] = se[i]

    # Load video embeddings
    ve = np.load(os.path.join(OUT, "video_embeddings.npy")).astype(np.float64)
    vit = json.load(open(os.path.join(OUT, "video_embed_index.json")))
    for i, it in enumerate(vit):
        EMB[it["crop"]] = ve[i]

    # Labels
    seed_lab = {
        r["crop"]: r["label"]
        for r in json.load(open(os.path.join(OUT, "seed_labels.json")))["labels"]
    }
    final = {
        r["crop"]: r["label"]
        for r in json.load(open(os.path.join(OUT, "labels_video_v1.json")))["labels"]
    }
    seed_set = set(json.load(open(os.path.join(OUT, "seed_crops.json"))))

    # Build training set (seed video + confirmed still)
    X_tr: List[np.ndarray] = []
    y_tr: List[str] = []
    for c, l in seed_lab.items():
        if c in EMB:
            X_tr.append(EMB[c])
            y_tr.append(l)
    for r in json.load(open(os.path.join(OUT, "labels_still_v1.json")))["labels"]:
        if r["crop"] in EMB:
            X_tr.append(EMB[r["crop"]])
            y_tr.append(r["label"])

    Z_train = np.stack(X_tr)  # (n_train, 512) float64
    print(f"  Train: n={len(Z_train)}, dtype={Z_train.dtype}", flush=True)

    # Canonical LR
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ).fit(Z_train, y_tr)
    classes: List[str] = list(clf.classes_)
    n_classes = len(classes)
    print(f"  Classifier: n_classes={n_classes}", flush=True)

    # Eval set: video crops not in seed_set with ground truth
    crops = [
        it["crop"]
        for it in vit
        if it["crop"] not in seed_set and it["crop"] in final
    ]
    Z_eval = np.stack([EMB[c] for c in crops])  # (n_eval, 512) float64
    n = len(crops)
    print(f"  Eval: n={n}, dtype={Z_eval.dtype}", flush=True)

    # Single predict_proba call (float64)
    P = clf.predict_proba(Z_eval).astype(np.float64)
    pred: List[str] = [classes[int(i)] for i in P.argmax(axis=1)]
    true: List[str] = [final[c] for c in crops]

    # Uncertainty scores
    conf   = P.max(axis=1)
    ent    = -(P * np.log(P + 1e-9)).sum(axis=1)
    part   = np.sort(P, axis=1)
    margin = (part[:, -1] - part[:, -2])

    # Masks
    rare_mask = np.array(
        [t not in COMMON and t != "誤検出(動物なし)" for t in true], dtype=bool
    )
    base_correct = np.array([pred[i] == true[i] for i in range(n)], dtype=bool)
    base_acc = float(base_correct.mean())

    print(
        f"  n_rare={int(rare_mask.sum())}, base_acc={base_acc:.4f},"
        f"  n_rare_correct={int(base_correct[rare_mask].sum())}", flush=True
    )
    print(f"  Step 1 done in {time.time()-t0:.1f}s", flush=True)

    data_hashes = {
        "embeddings.npy": file_sha256(os.path.join(OUT, "embeddings.npy")),
        "video_embeddings.npy": file_sha256(os.path.join(OUT, "video_embeddings.npy")),
        "seed_labels.json": file_sha256(os.path.join(OUT, "seed_labels.json")),
        "labels_video_v1.json": file_sha256(os.path.join(OUT, "labels_video_v1.json")),
        "labels_still_v1.json": file_sha256(os.path.join(OUT, "labels_still_v1.json")),
        "provisional_labels.json": file_sha256(os.path.join(OUT, "provisional_labels.json")),
        "seed_crops.json": file_sha256(os.path.join(OUT, "seed_crops.json")),
    }

    return dict(
        clf=clf, classes=classes, n_classes=n_classes,
        crops=crops, Z_eval=Z_eval, Z_train=Z_train, P=P,
        pred=pred, true=true, n=n,
        conf=conf, ent=ent, margin=margin,
        rare_mask=rare_mask, base_correct=base_correct, base_acc=base_acc,
        data_hashes=data_hashes,
    )


# ---------------------------------------------------------------------------
# Step 2: Compute kNN rarity scores (label-free)
# ---------------------------------------------------------------------------

def compute_rarity_scores(data: Dict) -> Dict:
    """
    Compute k-NN rarity scores for primary k and ablation k values.

    Information leak audit:
      - Only BioCLIP embeddings used (no labels, no ground truth)
      - Z_eval and Z_train are disjoint (video eval crops not in seed_crops.json)
      - This measure is computable at deployment time: training set embeddings
        are always available to the deployed model pipeline.
    """
    print("\n[Step 2] Computing kNN rarity scores...", flush=True)
    Z_train = data["Z_train"]
    Z_eval = data["Z_eval"]
    n = data["n"]

    result = {}
    for k in [K_PRIMARY] + K_ABLATION:
        t0 = time.time()
        dists = compute_knn_rarity(Z_train, Z_eval, k)
        print(
            f"  k={k:>2}: dist stats: min={dists.min():.4f}  "
            f"mean={dists.mean():.4f}  max={dists.max():.4f}  "
            f"({time.time()-t0:.1f}s)", flush=True
        )
        result[k] = dists

    return result


# ---------------------------------------------------------------------------
# Step 3: Evaluate RAU for all lambda × k combinations
# ---------------------------------------------------------------------------

def evaluate_rau_grid(data: Dict, rarity_scores: Dict) -> Dict:
    """
    Compute triage AUC for all (lambda, k) combinations.
    Primary: k=K_PRIMARY (10). Ablation: K_ABLATION.
    """
    print("\n[Step 3] Evaluating RAU on lambda grid...", flush=True)
    n = data["n"]
    conf = data["conf"]
    base_correct = data["base_correct"]
    rare_mask = data["rare_mask"]

    grid_results = {}

    for k, rarity in rarity_scores.items():
        k_results = {}
        for lam in LAMBDA_GRID:
            order = compute_rau_order(conf, rarity, lam)
            accs, auc_acc, rares, auc_rare = compute_auc_from_order(
                order, base_correct, n, BUDGETS, rare_mask
            )
            _, auc_acc_b30, _, auc_rare_b30 = compute_auc_from_order(
                order, base_correct, n, BUDGETS_PARTIAL, rare_mask
            )
            k_results[lam] = {
                "lambda": lam,
                "k": k,
                "auc_acc_budget": auc_acc,
                "rare_recall_auc": auc_rare,
                "auc_acc_budget_b30": auc_acc_b30,
                "rare_recall_auc_b30": auc_rare_b30,
                "acc_by_budget": accs,
                "rare_recall_by_budget": rares,
            }
            print(
                f"  k={k:>2}  lambda={lam:.2f}:  "
                f"AUC_acc={auc_acc:.4f}  AUC_rare={auc_rare:.4f}  "
                f"(B≤30%: acc={auc_acc_b30:.4f}  rare={auc_rare_b30:.4f})",
                flush=True,
            )
        grid_results[k] = k_results

    return grid_results


# ---------------------------------------------------------------------------
# Step 4: Compute baselines (confidence, entropy, badge_approx PCA-20)
# ---------------------------------------------------------------------------

def compute_baselines(data: Dict) -> Dict:
    """
    Recompute all baseline orderings fresh (independent of pipeline_canonical.py).
    Includes: confidence, entropy, margin, badge_approx (PCA-20), random.
    BADGE full_dim: referenced from badge_pca_sensitivity.json (not recomputed,
    as k-center greedy on 9728-dim gradient embedding is O(n^2*d) on CPU).
    """
    print("\n[Step 4] Computing baselines...", flush=True)
    n = data["n"]
    base_correct = data["base_correct"]
    rare_mask = data["rare_mask"]
    conf, ent, margin = data["conf"], data["ent"], data["margin"]
    Z_eval, P = data["Z_eval"], data["P"]

    baselines: Dict[str, Dict] = {}

    # Deterministic orderings
    det_orders = {
        "confidence": np.argsort(conf),       # least-confident first
        "entropy":    np.argsort(-ent),        # highest-entropy first
        "margin":     np.argsort(margin),      # smallest-margin first
    }

    print("[Step 4a] Computing confidence/entropy/margin...", flush=True)
    for name, order in det_orders.items():
        accs, auc_acc, rares, auc_rare = compute_auc_from_order(
            order, base_correct, n, BUDGETS, rare_mask
        )
        _, auc_acc_b30, _, auc_rare_b30 = compute_auc_from_order(
            order, base_correct, n, BUDGETS_PARTIAL, rare_mask
        )
        baselines[name] = {
            "auc_acc_budget": auc_acc,
            "rare_recall_auc": auc_rare,
            "auc_acc_budget_b30": auc_acc_b30,
            "rare_recall_auc_b30": auc_rare_b30,
            "acc_by_budget": accs,
            "rare_recall_by_budget": rares,
        }
        print(
            f"  {name:<15} AUC_acc={auc_acc:.4f}  AUC_rare={auc_rare:.4f}", flush=True
        )

    # BADGE-approx PCA-20
    print("[Step 4b] Computing BADGE-approx (PCA-20)...", flush=True)
    t0 = time.time()
    badge_order = badge_approx_order(Z_eval, P, pca_components=20)
    accs, auc_acc, rares, auc_rare = compute_auc_from_order(
        badge_order, base_correct, n, BUDGETS, rare_mask
    )
    _, auc_acc_b30, _, auc_rare_b30 = compute_auc_from_order(
        badge_order, base_correct, n, BUDGETS_PARTIAL, rare_mask
    )
    baselines["badge_approx_pca20"] = {
        "auc_acc_budget": auc_acc,
        "rare_recall_auc": auc_rare,
        "auc_acc_budget_b30": auc_acc_b30,
        "rare_recall_auc_b30": auc_rare_b30,
        "acc_by_budget": accs,
        "rare_recall_by_budget": rares,
    }
    print(
        f"  badge_approx_pca20 AUC_acc={auc_acc:.4f}  AUC_rare={auc_rare:.4f}"
        f"  ({time.time()-t0:.1f}s)", flush=True
    )

    # Reference: BADGE full_dim (stored in badge_pca_sensitivity.json)
    badge_sens_path = os.path.join(BMK, "badge_pca_sensitivity.json")
    badge_full_ref = None
    if os.path.exists(badge_sens_path):
        try:
            badge_sens = json.load(open(badge_sens_path))
            fd = badge_sens["badge_variants"].get("full_dim", {})
            badge_full_ref = {
                "source": "badge_pca_sensitivity.json (not recomputed here)",
                "pca_components": None,
                "gradient_embedding_dim": fd.get("gradient_embedding_dim"),
                "auc_acc_budget": fd.get("auc_acc_budget"),
                "rare_recall_auc": fd.get("auc_rare_budget"),
                "auc_acc_budget_b30": fd.get("auc_acc_budget_b30"),
                "confidence_auc_acc_from_badge_script": badge_sens.get("confidence_auc_acc"),
            }
            print(
                f"  badge_full_dim (ref) AUC_acc={badge_full_ref['auc_acc_budget']}  "
                f"AUC_rare={badge_full_ref['rare_recall_auc']}", flush=True
            )
        except Exception as ex:
            print(f"  [WARN] Could not load badge_pca_sensitivity.json: {ex}", flush=True)
    baselines["badge_full_dim_reference"] = badge_full_ref

    # Store orderings for downstream tests (avoid recomputation)
    det_orders["badge_approx_pca20"] = badge_order

    return baselines, det_orders


# ---------------------------------------------------------------------------
# Step 5: Select best lambda (primary k=10)
# ---------------------------------------------------------------------------

def select_best_lambda(
    grid_results: Dict,
    baselines: Dict,
    k: int = K_PRIMARY,
) -> Dict:
    """
    Select best lambda using the following criterion:
    Maximize rare_recall_auc while keeping:
      auc_acc >= confidence_auc_acc - ACC_DEGRADATION_THRESHOLD

    Report all lambda values; selection criterion is explicit.
    """
    print(f"\n[Step 5] Lambda selection (k={k})...", flush=True)
    conf_acc_auc = baselines["confidence"]["auc_acc_budget"]
    threshold = conf_acc_auc - ACC_DEGRADATION_THRESHOLD

    best_lam = None
    best_rare_auc = -1.0

    for lam in LAMBDA_GRID:
        r = grid_results[k][lam]
        acc_ok = r["auc_acc_budget"] >= threshold
        print(
            f"  lambda={lam:.2f}: AUC_rare={r['rare_recall_auc']:.4f}  "
            f"AUC_acc={r['auc_acc_budget']:.4f}  "
            f"acc_ok={acc_ok}  "
            f"(threshold={threshold:.4f})",
            flush=True,
        )
        if acc_ok and r["rare_recall_auc"] > best_rare_auc:
            best_rare_auc = r["rare_recall_auc"]
            best_lam = lam

    print(
        f"  → Best lambda: {best_lam}  (rare_recall_auc={best_rare_auc:.4f})", flush=True
    )
    return {
        "k_primary": k,
        "best_lambda": best_lam,
        "best_rare_recall_auc": best_rare_auc,
        "best_auc_acc": grid_results[k][best_lam]["auc_acc_budget"],
        "selection_criterion": (
            f"Maximize rare_recall_auc subject to "
            f"auc_acc >= confidence_auc_acc - {ACC_DEGRADATION_THRESHOLD} "
            f"(= {threshold:.4f})"
        ),
        "caution": (
            "Best lambda selected on the SAME eval set used for all baselines. "
            "This is consistent with how baselines are evaluated but is NOT a held-out test. "
            "Lambda grid results should be interpreted as exploratory; "
            "the optimal lambda may not transfer to new datasets."
        ),
    }


# ---------------------------------------------------------------------------
# Step 6: Statistical tests (permutation + bootstrap CI)
# ---------------------------------------------------------------------------

def run_statistical_tests(
    data: Dict,
    grid_results: Dict,
    baselines: Dict,
    det_orders: Dict,
    best_lambda: float,
    k: int = K_PRIMARY,
) -> Dict:
    """
    1. Permutation test: is RAU(best_lambda) rare_recall_auc > random?
       H0: random ordering achieves as high rare_recall_auc as RAU.
       p = (n_exceed + 1) / (N_PERM + 1)  [Phipson & Smyth 2010]

    2. Bootstrap CI:
       - RAU(best_lambda) - confidence (rare AUC)
       - RAU(best_lambda) - badge_approx_pca20 (rare AUC)
    """
    print(f"\n[Step 6] Statistical tests (lambda={best_lambda}, k={k})...", flush=True)
    n = data["n"]
    base_correct = data["base_correct"]
    rare_mask = data["rare_mask"]
    conf = data["conf"]

    # RAU order for best lambda
    rarity = compute_knn_rarity(data["Z_train"], data["Z_eval"], k)
    rau_order = compute_rau_order(conf, rarity, best_lambda)
    rau_rare_auc = grid_results[k][best_lambda]["rare_recall_auc"]

    # --- Permutation test (rare recall AUC vs random) ---
    print("[Step 6a] Permutation test (10k shuffles, rare recall AUC)...", flush=True)
    t0 = time.time()
    rng_perm = np.random.default_rng(SEED_PERM)
    rand_rare_aucs = fast_rand_rare_aucs(
        N_PERM, base_correct, rare_mask, BUDGETS, n, rng_perm, batch=200
    )
    n_exceed = int(np.sum(rand_rare_aucs >= rau_rare_auc))
    pval = (n_exceed + 1) / (N_PERM + 1)  # Phipson & Smyth 2010
    rand_mean = float(np.mean(rand_rare_aucs))
    rand_std = float(np.std(rand_rare_aucs))
    print(
        f"  RAU rare_auc={rau_rare_auc:.4f}  "
        f"random: mean={rand_mean:.4f}±{rand_std:.4f}  "
        f"n_exceed={n_exceed}  p={pval:.6f}  ({time.time()-t0:.1f}s)",
        flush=True,
    )

    # --- Bootstrap CI: RAU vs confidence ---
    print("[Step 6b] Bootstrap CI: RAU vs confidence (n_boot=500)...", flush=True)
    rng_boot1 = np.random.default_rng(SEED_BOOT + 1)
    ci_vs_conf = bootstrap_rare_auc_gap(
        rau_order, det_orders["confidence"],
        base_correct, rare_mask, BUDGETS, n, N_BOOT, rng_boot1,
        label_a=f"RAU(lambda={best_lambda},k={k})",
        label_b="confidence",
    )
    print(
        f"  RAU - confidence: mean_gap={ci_vs_conf['mean_gap_rare_auc']:.4f}  "
        f"CI95={ci_vs_conf['ci95']}  sig={ci_vs_conf['significance_at_alpha05']}",
        flush=True,
    )

    # --- Bootstrap CI: RAU vs badge_approx_pca20 ---
    print("[Step 6c] Bootstrap CI: RAU vs badge_approx_pca20 (n_boot=500)...", flush=True)
    rng_boot2 = np.random.default_rng(SEED_BOOT + 2)
    ci_vs_badge = bootstrap_rare_auc_gap(
        rau_order, det_orders["badge_approx_pca20"],
        base_correct, rare_mask, BUDGETS, n, N_BOOT, rng_boot2,
        label_a=f"RAU(lambda={best_lambda},k={k})",
        label_b="badge_approx_pca20",
    )
    print(
        f"  RAU - badge_approx: mean_gap={ci_vs_badge['mean_gap_rare_auc']:.4f}  "
        f"CI95={ci_vs_badge['ci95']}  sig={ci_vs_badge['significance_at_alpha05']}",
        flush=True,
    )

    # Random baseline rare AUC (2k shuffles for reporting)
    rng_rand = np.random.default_rng(SEED_MAIN + 1)
    rand_rare_2k = fast_rand_rare_aucs(
        N_RAND_RARE, base_correct, rare_mask, BUDGETS, n, rng_rand, batch=200
    )

    return {
        "permutation_test_vs_random": {
            "method": "rare_recall_AUC",
            "n_shuffles": N_PERM,
            "rau_rare_auc": rau_rare_auc,
            "rand_mean": round(rand_mean, 4),
            "rand_std": round(rand_std, 4),
            "n_exceed": n_exceed,
            "p_value": round(pval, 6),
            "formula": "(n_exceed+1)/(N_PERM+1) [Phipson & Smyth 2010]",
            "significant_at_alpha05": pval < 0.05,
        },
        "bootstrap_ci_vs_confidence": ci_vs_conf,
        "bootstrap_ci_vs_badge_approx_pca20": ci_vs_badge,
        "random_baseline_rare_auc": {
            "n_shuffles": N_RAND_RARE,
            "mean": round(float(np.mean(rand_rare_2k)), 4),
            "std": round(float(np.std(rand_rare_2k)), 4),
        },
    }


# ---------------------------------------------------------------------------
# Step 7: Second site generalization check (Mie stills, seed-only LR)
# ---------------------------------------------------------------------------

def run_second_site_check(best_lambda: float, k: int = K_PRIMARY) -> Optional[Dict]:
    """
    Second site (Mie stills) generalization check with seed-only LR.

    Training: seed video crops only (n=1000, no Mie stills → clean generalization).
    Eval: all Mie stills from labels_still_v1.json (n=4736).
    Rarity: kNN distance from Mie stills to seed-only training embeddings.

    Note: seed-only LR has 11 classes (from seed video crops only, not 19).
    Some Mie classes (8 species) not seen in seed training → always misclassified.
    This matches triage_mie_secondsite.json setup exactly.
    """
    print(
        f"\n[Step 7] Second site check (Mie stills, seed-only LR, "
        f"lambda={best_lambda}, k={k})...",
        flush=True,
    )
    t0 = time.time()

    try:
        # Load seed video embeddings + labels
        ve = np.load(os.path.join(OUT, "video_embeddings.npy")).astype(np.float64)
        vit = json.load(open(os.path.join(OUT, "video_embed_index.json")))
        seed_lab = {
            r["crop"]: r["label"]
            for r in json.load(open(os.path.join(OUT, "seed_labels.json")))["labels"]
        }
        seed_set = set(json.load(open(os.path.join(OUT, "seed_crops.json"))))

        # Build video crop → embedding map
        video_EMB: Dict[str, np.ndarray] = {}
        for i, it in enumerate(vit):
            video_EMB[it["crop"]] = ve[i]

        # Seed-only training set (video crops in seed_crops.json that have labels)
        X_tr_seed: List[np.ndarray] = []
        y_tr_seed: List[str] = []
        for c, l in seed_lab.items():
            if c in video_EMB:
                X_tr_seed.append(video_EMB[c])
                y_tr_seed.append(l)

        Z_train_seed = np.stack(X_tr_seed)  # (1000, 512) float64
        print(
            f"  Seed-only train: n={len(Z_train_seed)}, "
            f"n_classes={len(set(y_tr_seed))}",
            flush=True,
        )

        # Seed-only LR
        clf_seed = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ).fit(Z_train_seed, y_tr_seed)
        classes_seed: List[str] = list(clf_seed.classes_)
        n_cls_seed = len(classes_seed)

        # Mie still embeddings + labels
        se = np.load(os.path.join(OUT, "embeddings.npy")).astype(np.float64)
        prov = json.load(open(os.path.join(OUT, "provisional_labels.json")))
        still_labels = {
            r["crop"]: r["label"]
            for r in json.load(open(os.path.join(OUT, "labels_still_v1.json")))["labels"]
        }
        # provisional_labels.json is the index for embeddings.npy
        prov_EMB: Dict[str, np.ndarray] = {}
        for i, r in enumerate(prov):
            prov_EMB[r["crop"]] = se[i]

        # Mie eval: stills with confirmed labels
        mie_crops = [
            r["crop"]
            for r in json.load(open(os.path.join(OUT, "labels_still_v1.json")))["labels"]
            if r["crop"] in prov_EMB
        ]
        Z_eval_mie = np.stack([prov_EMB[c] for c in mie_crops])
        true_mie = [still_labels[c] for c in mie_crops]
        n_mie = len(mie_crops)
        print(f"  Mie eval: n={n_mie}", flush=True)

        # Predict with seed-only LR
        P_mie = clf_seed.predict_proba(Z_eval_mie).astype(np.float64)
        pred_mie = [classes_seed[int(i)] for i in P_mie.argmax(axis=1)]
        conf_mie = P_mie.max(axis=1)

        # Rare mask (same definition as main pipeline)
        rare_mask_mie = np.array(
            [t not in COMMON and t != "誤検出(動物なし)" for t in true_mie], dtype=bool
        )
        base_correct_mie = np.array(
            [pred_mie[i] == true_mie[i] for i in range(n_mie)], dtype=bool
        )
        base_acc_mie = float(base_correct_mie.mean())
        print(
            f"  n_rare_mie={int(rare_mask_mie.sum())}  "
            f"base_acc_mie={base_acc_mie:.4f}",
            flush=True,
        )

        # kNN rarity (reference: seed-only training embeddings)
        rarity_mie = compute_knn_rarity(Z_train_seed, Z_eval_mie, k)

        # RAU order with best_lambda
        rau_order_mie = compute_rau_order(conf_mie, rarity_mie, best_lambda)
        conf_order_mie = np.argsort(conf_mie)

        _, auc_acc_rau, _, auc_rare_rau = compute_auc_from_order(
            rau_order_mie, base_correct_mie, n_mie, BUDGETS, rare_mask_mie
        )
        _, auc_acc_conf, _, auc_rare_conf = compute_auc_from_order(
            conf_order_mie, base_correct_mie, n_mie, BUDGETS, rare_mask_mie
        )
        print(
            f"  RAU: AUC_acc={auc_acc_rau:.4f}  AUC_rare={auc_rare_rau:.4f}", flush=True
        )
        print(
            f"  conf: AUC_acc={auc_acc_conf:.4f}  AUC_rare={auc_rare_conf:.4f}",
            flush=True,
        )

        # Random baseline (500 shuffles for quick second-site check)
        rng_mie = np.random.default_rng(SEED_MAIN + 777)
        rand_rare_mie = fast_rand_rare_aucs(
            500, base_correct_mie, rare_mask_mie, BUDGETS, n_mie, rng_mie, batch=100
        )
        rand_rare_mie_mean = round(float(np.mean(rand_rare_mie)), 4)
        rand_rare_mie_std = round(float(np.std(rand_rare_mie)), 4)

        elapsed = time.time() - t0
        print(f"  Second site done in {elapsed:.1f}s", flush=True)

        return {
            "description": (
                "Generalization check: Mie stills with seed-only LR "
                f"(trained on n={len(Z_train_seed)} seed video crops, no Mie stills). "
                f"RAU lambda={best_lambda}, k={k}."
            ),
            "eval_site": "三重県 (Mie) still photos (labels_still_v1.json)",
            "n_mie": n_mie,
            "n_rare_mie": int(rare_mask_mie.sum()),
            "n_classes_seed_model": n_cls_seed,
            "classes_seed_model": classes_seed,
            "n_classes_missing": len(
                [t for t in set(true_mie) if t not in set(classes_seed)]
            ),
            "classes_missing_from_seed": sorted(
                set(true_mie) - set(classes_seed)
            ),
            "base_accuracy_mie": round(base_acc_mie, 4),
            "rau": {
                "lambda": best_lambda,
                "k": k,
                "auc_acc_budget": auc_acc_rau,
                "rare_recall_auc": auc_rare_rau,
            },
            "confidence": {
                "auc_acc_budget": auc_acc_conf,
                "rare_recall_auc": auc_rare_conf,
            },
            "random_baseline_rare_auc": {
                "n_shuffles": 500,
                "mean": rand_rare_mie_mean,
                "std": rand_rare_mie_std,
            },
            "rare_auc_gap_rau_minus_confidence": round(
                auc_rare_rau - auc_rare_conf, 4
            ),
            "elapsed_s": round(elapsed, 1),
        }

    except Exception as ex:
        print(f"  [WARN] Second site check failed: {ex}", flush=True)
        import traceback
        traceback.print_exc()
        return {"error": str(ex)}


# ---------------------------------------------------------------------------
# Step 8: Write rau_note.md
# ---------------------------------------------------------------------------

def write_note(
    grid_results: Dict,
    baselines: Dict,
    best_lambda_info: Dict,
    stats: Dict,
    second_site: Optional[Dict],
    k: int = K_PRIMARY,
) -> None:
    """Write Japanese notes with methodology, full results, and honest novelty assessment."""
    best_lam = best_lambda_info["best_lambda"]
    best_rare_auc = best_lambda_info["best_rare_recall_auc"]
    best_acc_auc = best_lambda_info["best_auc_acc"]

    conf_rare_auc = baselines["confidence"]["rare_recall_auc"]
    conf_acc_auc = baselines["confidence"]["auc_acc_budget"]
    badge_rare_auc = baselines["badge_approx_pca20"]["rare_recall_auc"]
    badge_full_ref = baselines.get("badge_full_dim_reference") or {}
    badge_full_rare = badge_full_ref.get("rare_recall_auc", "N/A")

    perm = stats["permutation_test_vs_random"]
    ci_conf = stats["bootstrap_ci_vs_confidence"]
    ci_badge = stats["bootstrap_ci_vs_badge_approx_pca20"]

    # Honest novelty judgment
    beats_conf = ci_conf["significance_at_alpha05"] and ci_conf["mean_gap_rare_auc"] > 0
    beats_badge_pca20 = ci_badge["significance_at_alpha05"] and ci_badge["mean_gap_rare_auc"] > 0
    beats_badge_full = (
        isinstance(badge_full_rare, float) and best_rare_auc > badge_full_rare
    )

    if beats_conf and beats_badge_pca20:
        if beats_badge_full:
            novelty_verdict = (
                "RAU は confidence・BADGE(PCA-20)・BADGE(全次元) のすべてを希少種AUCで上回る。"
                "統計的に有意(bootstrap CI)。新規性の核として主張可能。"
            )
        else:
            novelty_verdict = (
                f"RAU は confidence・BADGE(PCA-20) を希少種AUCで有意に上回る(bootstrap CI)。"
                f"BADGE(全次元, AUC={badge_full_rare}) との比較はほぼ同等または劣る可能性あり"
                f"(直接のbootstrap検定は未実施; 全次元BADGE計算コストのためstored結果を参照)。"
                f"新規性は「信頼度+近傍希少性の組み合わせがBADGE PCA-20を超える」という点で主張可能。"
            )
    elif beats_conf and not beats_badge_pca20:
        novelty_verdict = (
            f"RAU は confidence を希少種AUCで有意に上回るが、"
            f"BADGE(PCA-20, AUC={badge_rare_auc:.4f}) には及ばない。"
            f"「信頼度だけでは不十分」という本文の主張は支持されるが、"
            f"RAU の新規性は限定的: BADGE の方が強い手法。"
            f"正直な結論: RAU は中間的な性能であり、"
            f"既存の信頼度順を超えるが計算コストが低い代替として位置づけ可能。"
        )
    else:
        gap = round(best_rare_auc - conf_rare_auc, 4)
        novelty_verdict = (
            f"RAU は confidence との希少AUC差が {gap:+.4f} であり、"
            f"統計的に有意な改善を示せなかった(CI: {ci_conf['ci95']})。"
            f"正直な結論: 単純な信頼度順が十分であり、"
            f"ラベルなし近傍希少性を加えても実質的なゲインは得られない。"
            f"これは「既存手法の組み合わせで十分」という貢献・フレームでの論文化に使える(negative result as positive finding)。"
        )

    ss_note = ""
    if second_site and "error" not in second_site:
        ss_gap = second_site.get("rare_auc_gap_rau_minus_confidence", "N/A")
        ss_rare_rau = second_site.get("rau", {}).get("rare_recall_auc", "N/A")
        ss_rare_conf = second_site.get("confidence", {}).get("rare_recall_auc", "N/A")
        ss_note = (
            f"\n## 第2サイト汎化(三重 static)\n"
            f"- n={second_site.get('n_mie')}  n_rare={second_site.get('n_rare_mie')}\n"
            f"- 使用モデル: seed-only LR (n_train={len(second_site.get('classes_seed_model',[]))}クラス, "
            f"三重静止画をtrain非使用 → クリーン汎化)\n"
            f"- RAU rare_recall_AUC={ss_rare_rau}  vs  confidence={ss_rare_conf}\n"
            f"- gap(RAU-conf)={ss_gap:+.4f}\n"
            f"- 備考: 三重サイトはニホンジカ・タヌキ主体(不均衡)で希少種数が少ない(n_rare={second_site.get('n_rare_mie')})。\n"
            f"  seed-only LR は {second_site.get('n_classes_missing')} クラスを未学習。\n"
        )
        if isinstance(ss_gap, float) and ss_gap > 0:
            ss_note += "- 解釈: 第2サイトでも RAU が confidence を上回り、汎化が示唆される。\n"
        elif isinstance(ss_gap, float) and ss_gap < 0:
            ss_note += "- 解釈: 第2サイトでは confidence が RAU を上回る。汎化は限定的。\n"
        else:
            ss_note += "- 解釈: 第2サイトでは差がほとんどない。\n"

    # Full grid table
    grid_table_rows = []
    grid_table_rows.append("| k | lambda | AUC_rare | AUC_acc | AUC_rare_B30 | AUC_acc_B30 |")
    grid_table_rows.append("|---|--------|----------|---------|--------------|-------------|")
    for k_val in [K_PRIMARY] + K_ABLATION:
        if k_val not in grid_results:
            continue
        for lam in LAMBDA_GRID:
            r = grid_results[k_val][lam]
            grid_table_rows.append(
                f"| {k_val} | {lam:.2f} | {r['rare_recall_auc']:.4f} | "
                f"{r['auc_acc_budget']:.4f} | {r['rare_recall_auc_b30']:.4f} | "
                f"{r['auc_acc_budget_b30']:.4f} |"
            )

    # Baseline table
    baseline_rows = []
    baseline_rows.append("| 手法 | AUC_rare | AUC_acc | AUC_rare_B30 | AUC_acc_B30 |")
    baseline_rows.append("|------|----------|---------|--------------|-------------|")
    for name, vals in baselines.items():
        if name == "badge_full_dim_reference":
            if vals:
                baseline_rows.append(
                    f"| BADGE_full_dim(ref) | {vals.get('rare_recall_auc', 'N/A')} | "
                    f"{vals.get('auc_acc_budget', 'N/A')} | - | - |"
                )
            continue
        baseline_rows.append(
            f"| {name} | {vals['rare_recall_auc']:.4f} | {vals['auc_acc_budget']:.4f} | "
            f"{vals['rare_recall_auc_b30']:.4f} | {vals['auc_acc_budget_b30']:.4f} |"
        )

    lines = [
        "# RAU (Rarity-Aware Uncertainty) Triage 実験ノート",
        f"実行日時: 2026-07-11",
        "",
        "## 1. 定式化",
        "",
        "### RAU スコア",
        "```",
        "Score(i) = (1-lambda) * uncertainty_norm(i) + lambda * rarity_norm(i)",
        "  uncertainty_norm(i) = minmax(1 - max_softmax(i))   [高い → より不確か]",
        "  rarity_norm(i)      = minmax(kNN_distance(i))      [高い → より孤立 = 希少候補]",
        "```",
        "- kNN_distance: eval サンプル i からトレーニングセットへの第k近傍ユークリッド距離",
        f"- k(primary)={K_PRIMARY}, ablation k={K_ABLATION}",
        f"- lambda グリッド: {LAMBDA_GRID}",
        "",
        "### 情報リーク自己点検",
        "- kNN 距離の計算には **ラベルを一切使わない**: BioCLIP 埋め込みのみ",
        "- eval セットと train セットは完全に非重複(video eval crops ∉ seed_crops.json)",
        "- kNN 参照 = トレーニングセット埋め込み(展開時に常に利用可能)",
        "- **eval 統計(ラベル・正解率等)を希少性推定に使っていない** → リーク無し",
        "- lambda 選択は同一 eval セット上で行う(全ベースラインと同条件) → "
        "探索的結果であり held-out 検証ではない点を本文で明記する",
        "",
        "## 2. 評価設定",
        "",
        f"- eval セット: 動画 eval (n=6906, n_rare=1480)",
        f"- budgets: {BUDGETS}",
        f"- 希少種定義: true_label ∉ COMMON ∧ ≠ '誤検出(動物なし)'",
        f"  COMMON = {sorted(COMMON)}",
        "",
        "## 3. 全 lambda グリッド結果",
        "",
        "\n".join(grid_table_rows),
        "",
        "## 4. ベースライン比較",
        "",
        "\n".join(baseline_rows),
        "",
        f"**最良 lambda**: {best_lam} (k={k})",
        f"- RAU rare_recall_AUC = {best_rare_auc:.4f}",
        f"- RAU AUC_acc = {best_acc_auc:.4f}",
        f"- confidence rare_AUC = {conf_rare_auc:.4f}",
        f"- BADGE_pca20 rare_AUC = {badge_rare_auc:.4f}",
        f"- BADGE_full_dim rare_AUC = {badge_full_rare} (stored, not recomputed)",
        "",
        f"**lambda 選択基準**: AUC_acc >= {conf_acc_auc:.4f} - {ACC_DEGRADATION_THRESHOLD} = "
        f"{conf_acc_auc - ACC_DEGRADATION_THRESHOLD:.4f} の制約下で rare_recall_AUC を最大化",
        "",
        "## 5. 統計検定",
        "",
        "### 順列検定 (RAU vs random, rare recall AUC)",
        f"- H0: ランダム順序と同等の希少種AUC",
        f"- RAU rare_AUC = {perm['rau_rare_auc']:.4f}",
        f"- random: mean={perm['rand_mean']:.4f} ± {perm['rand_std']:.4f}",
        f"- n_exceed = {perm['n_exceed']},  p = {perm['p_value']:.6f}",
        f"- 有意: {'YES (p < 0.05)' if perm['significant_at_alpha05'] else 'NO'}",
        f"- 式: {perm['formula']}",
        "",
        "### Bootstrap CI (rare recall AUC の差)",
        f"#### RAU vs confidence",
        f"- gap mean = {ci_conf['mean_gap_rare_auc']:+.4f}  CI95 = {ci_conf['ci95']}",
        f"- 有意: {'YES' if ci_conf['significance_at_alpha05'] else 'NO'}  "
        f"(CI が 0 を{'' if ci_conf['significance_at_alpha05'] else 'は'}含まない)",
        f"- {ci_conf['interpretation']}",
        "",
        f"#### RAU vs badge_approx_pca20",
        f"- gap mean = {ci_badge['mean_gap_rare_auc']:+.4f}  CI95 = {ci_badge['ci95']}",
        f"- 有意: {'YES' if ci_badge['significance_at_alpha05'] else 'NO'}",
        f"- {ci_badge['interpretation']}",
        "",
        ss_note,
        "## 6. 方法の妥当性・限界",
        "",
        "### 妥当性",
        "- kNN 距離はラベルフリーで計算でき、展開時に常に利用可能",
        "- BioCLIP 埋め込みは動物種の視覚的類似性を反映するため、",
        "  孤立したサンプル = 学習データの既知分布から外れた種 → 希少性の代理指標として妥当",
        "- 不確実性(softmax entropy)と希少性(kNN距離)は相関するが完全に一致しない:",
        "  高信頼で誤分類された希少種(confidence 順では後回しになるサンプル)を",
        "  rarity 項が前に引き出す可能性がある",
        "",
        "### 限界・注意点",
        "- kNN 距離が高い理由は「希少種」以外にも:",
        "  (a) 画質不良・ぼけ・遮蔽 (b) 照明変化 (c) 背景のみ",
        "  これらもレビューに回される可能性があり、precision が下がりうる",
        "- lambda をデータ内で最適化(held-out テストなし):",
        "  lambda の最適値が別データセットに転移するかは不明",
        "- kNN 計算は eval × train ペアワイズ距離が必要: O(n_eval × n_train × d)。",
        f"  本データ規模(n_eval={6906}, n_train=~5736, d=512)では数秒程度",
        "",
        "## 7. 新規性の Honest 評価",
        "",
        novelty_verdict,
        "",
        "### 論文への反映文案",
        "",
        "**もし RAU が有意に勝つ場合:**",
        "```",
        "We propose RAU (Rarity-Aware Uncertainty), a label-free triage score that",
        "combines softmax uncertainty with local rarity estimated from k-NN distances",
        "in BioCLIP embedding space. Unlike confidence-only triage, RAU explicitly",
        "prioritizes samples that are both uncertain and isolated from the training",
        "distribution, which tend to correspond to rare or invasive species of",
        "ecological interest. RAU achieves significantly higher rare-species recall",
        "AUC than confidence ordering (bootstrap CI excludes 0) while maintaining",
        "competitive overall accuracy AUC.",
        "```",
        "",
        "**もし RAU が有意に勝てない場合:**",
        "```",
        "We investigated RAU (Rarity-Aware Uncertainty), a label-free composite",
        "score combining softmax uncertainty with k-NN embedding rarity. Despite",
        "the intuitive motivation, RAU did not achieve significantly higher",
        "rare-species recall AUC than confidence ordering alone (bootstrap CI",
        f"includes 0: {ci_conf['ci95']}), suggesting that softmax uncertainty",
        "already implicitly captures the rare-species signal in BioCLIP embeddings.",
        "This finding supports the practical recommendation to use the simpler",
        "confidence ordering for rare-species triage in wildlife monitoring.",
        "```",
        "",
        "## 8. 環境・再現性",
        "",
        f"- Python {sys.version}",
        f"- numpy {np.__version__}",
        f"- Platform: {platform.platform()}",
        f"- Host: {socket.gethostname()}",
        f"- CUDA_VISIBLE_DEVICES: '' (CPU only)",
        f"- float64 throughout",
        f"- Seed: PERM={SEED_PERM}, BOOT={SEED_BOOT}",
        f"- N_PERM={N_PERM}, N_BOOT={N_BOOT}",
        f"- 結果ファイル: {RESULT_PATH}",
        "",
        "---",
        "_Generated by rarity_aware_triage.py_",
    ]

    with open(NOTE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  [saved] rau_note.md → {NOTE_PATH}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70, flush=True)
    print("RAU (Rarity-Aware Uncertainty) Triage Experiment", flush=True)
    print("=" * 70, flush=True)
    t_total = time.time()

    # Environment manifest
    import sklearn
    env = {
        "script": __file__,
        "python": sys.version,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save("_meta", {
        "description": (
            "RAU: Rarity-Aware Uncertainty triage. "
            "Combines softmax uncertainty with label-free kNN rarity in BioCLIP space."
        ),
        "env": env,
        "status": "running",
    })

    # Step 1: Load data
    data = load_data()
    _save("data_info", {
        "n_eval": data["n"],
        "n_rare": int(data["rare_mask"].sum()),
        "n_train": len(data["Z_train"]),
        "n_classes": data["n_classes"],
        "base_acc": data["base_acc"],
        "budgets": BUDGETS,
        "budgets_partial_b30": BUDGETS_PARTIAL,
        "data_hashes": data["data_hashes"],
    })

    # Step 2: kNN rarity scores
    rarity_scores = compute_rarity_scores(data)
    rarity_stats = {
        k: {
            "k": k,
            "mean": round(float(rarity_scores[k].mean()), 4),
            "std": round(float(rarity_scores[k].std()), 4),
            "min": round(float(rarity_scores[k].min()), 4),
            "max": round(float(rarity_scores[k].max()), 4),
        }
        for k in rarity_scores
    }
    _save("rarity_stats", rarity_stats)

    # Step 3: RAU grid
    grid_results = evaluate_rau_grid(data, rarity_scores)

    # Serialize for JSON (convert float keys to str)
    grid_json = {}
    for k_val, k_res in grid_results.items():
        grid_json[str(k_val)] = {}
        for lam, r in k_res.items():
            grid_json[str(k_val)][str(lam)] = r
    _save("rau_grid", grid_json)

    # Step 4: Baselines
    baselines, det_orders = compute_baselines(data)
    _save("baselines", baselines)

    # Step 5: Lambda selection
    best_lambda_info = select_best_lambda(grid_results, baselines, K_PRIMARY)
    best_lam = best_lambda_info["best_lambda"]
    _save("best_lambda_selection", best_lambda_info)

    # Step 6: Statistical tests
    stats = run_statistical_tests(
        data, grid_results, baselines, det_orders, best_lam, K_PRIMARY
    )
    _save("statistical_tests", stats)

    # Step 7: Second site check
    second_site = run_second_site_check(best_lam, K_PRIMARY)
    _save("second_site_generalization", second_site)

    # Summary table
    summary_rows = []
    for lam in LAMBDA_GRID:
        r = grid_results[K_PRIMARY][lam]
        tag = " ← BEST" if lam == best_lam else ""
        summary_rows.append({
            "method": f"RAU(lambda={lam},k={K_PRIMARY})",
            "auc_acc": r["auc_acc_budget"],
            "rare_recall_auc": r["rare_recall_auc"],
            "note": tag.strip(),
        })
    for name, vals in baselines.items():
        if name == "badge_full_dim_reference":
            if vals:
                summary_rows.append({
                    "method": "BADGE_full_dim (ref)",
                    "auc_acc": vals.get("auc_acc_budget"),
                    "rare_recall_auc": vals.get("rare_recall_auc"),
                    "note": "stored, not recomputed",
                })
            continue
        summary_rows.append({
            "method": name,
            "auc_acc": vals["auc_acc_budget"],
            "rare_recall_auc": vals["rare_recall_auc"],
            "note": "",
        })
    _save("summary_table", summary_rows)

    # Final metadata
    elapsed = time.time() - t_total
    _state["_meta"]["status"] = "complete"
    _state["_meta"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _state["_meta"]["elapsed_s"] = round(elapsed, 1)
    _save("_meta", _state["_meta"])

    # Step 8: Write note
    write_note(
        grid_results=grid_results,
        baselines=baselines,
        best_lambda_info=best_lambda_info,
        stats=stats,
        second_site=second_site,
        k=K_PRIMARY,
    )

    # Final summary print
    print("\n" + "=" * 70, flush=True)
    print(f"DONE in {elapsed:.1f}s", flush=True)
    print("=" * 70, flush=True)
    print("\nResult files:", flush=True)
    print(f"  JSON: {RESULT_PATH}", flush=True)
    print(f"  Note: {NOTE_PATH}", flush=True)
    print("\n--- SUMMARY TABLE ---", flush=True)
    for row in summary_rows:
        acc = f"{row['auc_acc']:.4f}" if isinstance(row["auc_acc"], float) else str(row["auc_acc"])
        rare = f"{row['rare_recall_auc']:.4f}" if isinstance(row["rare_recall_auc"], float) else str(row["rare_recall_auc"])
        print(
            f"  {row['method']:<45} AUC_acc={acc}  AUC_rare={rare}  {row['note']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
