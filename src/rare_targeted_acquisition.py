#!/usr/bin/env python3
"""
rare_targeted_acquisition.py — Rare-Class Prior Targeted Acquisition (RCTA)

Motivation:
  Existing RAU (uncertainty + kNN rarity) beats random but loses to confidence
  on true rare species (n=173, "不明" excluded). kNN embedding rarity over-prioritises
  "不明" crops that are also sparse in embedding space.

Proposed method: RCTA
  score_i = sum_c  p_c(i) * w_c

  where w_c is derived from TRAINING CLASS FREQUENCY ONLY (no eval gt used):
    - w_c = 0  for COMMON classes, "不明", "誤検出(動物なし)"
    - w_c > 0  for true-rare classes (present in training but underrepresented)

  Weight variants:
    binary      : w_c = 1 if true-rare class else 0
    inv_freq    : w_c = N_train / count_c  if true-rare class else 0
    log_inv_freq: w_c = log(N_train / count_c)  if true-rare class else 0

  (Weights are normalised so that max(w_c for true-rare classes) = 1
   inside the score computation, to keep scores on [0,1] scale.)

Combination with uncertainty:
  combined_{alpha}_{wtype}: alpha * rcta_norm + (1-alpha) * uncertainty_norm
  alpha grid: {0.0, 0.25, 0.5, 0.75, 1.0}

Evaluation (primary: true_rare_n173):
  - rare_recall_AUC (full range and B<=30%)
  - capture rate at B=10/20/30%
  - Bootstrap 95% CI vs confidence  (n_boot=500)
  - Permutation test vs random      (n_perm=10000)
  - Overall accuracy AUC (no regression check)

Information-leak audit:
  - w_c computed from training labels only
  - eval ground-truth NOT used to derive weights or scores
  - alpha grid evaluated on the same eval set  --> limitation noted in JSON and note
  - binary weight variant has NO tunable parameter from eval -> cleanest

GPU rule: CUDA_VISIBLE_DEVICES="" is set before any torch import.
Outputs (new files only, existing JSONs not overwritten):
  outputs/benchmark/rare_targeted_acquisition.json
  rare_targeted_note.md  (in trailcam_bench root)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU rule: must precede any torch import

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HOME = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HOME, "outputs")
BMK = os.path.join(OUT, "benchmark")

RESULT_PATH = os.path.join(BMK, "rare_targeted_acquisition.json")
NOTE_PATH = os.path.join(HOME, "rare_targeted_note.md")

# Safety: abort if outputs already exist
for p in [RESULT_PATH, NOTE_PATH]:
    if os.path.exists(p):
        print(f"[ERROR] Output already exists: {p}", flush=True)
        print("  既存ファイルの上書き防止のため終了します。", flush=True)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Constants (identical to rarity_aware_triage.py and rau_true_rare_analysis.py)
# ---------------------------------------------------------------------------
COMMON = {"ニホンジカ", "タヌキ", "イノシシ", "ニホンザル", "ニホンノウサギ"}
UNKNOWN_LABEL = "不明"
FALSE_DET_LABEL = "誤検出(動物なし)"

# Classes explicitly excluded from rare-class weight (not true rare species)
EXCLUDED_FROM_RARE_WEIGHT = COMMON | {UNKNOWN_LABEL, FALSE_DET_LABEL}

BUDGETS = [0, 5, 10, 20, 30, 50, 100]
BUDGETS_PARTIAL = [0, 5, 10, 20, 30]
CAPTURE_BUDGETS = [10, 20, 30]

# Statistical parameters (same seeds as prior scripts)
SEED_PERM = 6906
SEED_BOOT = 6906
N_PERM = 10_000
N_BOOT = 500
N_RAND = 2_000

LAMBDA_PRIMARY = 0.5   # RAU reference (from rarity_aware_triage.py)
K_PRIMARY = 10          # RAU kNN reference
K_PCA_BADGE = 20        # BADGE PCA components

ALPHA_GRID: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
WEIGHT_TYPES = ["binary", "inv_freq", "log_inv_freq"]

# ---------------------------------------------------------------------------
# State + encoder
# ---------------------------------------------------------------------------
_state: Dict = {}


class _NumpyEncoder(json.JSONEncoder):
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


def _save(key: str, value) -> None:
    _state[key] = value
    tmp = RESULT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_state, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    os.replace(tmp, RESULT_PATH)
    print(f"  [saved] {key}", flush=True)


# ---------------------------------------------------------------------------
# Utility functions (copied verbatim from rau_true_rare_analysis.py)
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
    xs = np.array([b / 100.0 for b in budgets])
    accs, rares = [], []
    for B in budgets:
        K = int(round(B / 100 * n))
        reviewed = set(order_idx[:K].tolist())
        correct = base_correct.copy()
        for i in reviewed:
            correct[i] = True
        accs.append(float(correct.mean()))
        n_rare = int(rare_mask.sum())
        rares.append(
            float(correct[rare_mask].mean()) if n_rare > 0 else float("nan")
        )
    return (
        [round(a, 4) for a in accs],
        round(_trapz(np.array(accs), xs), 4),
        [round(r, 4) for r in rares],
        round(_trapz(np.array(rares), xs), 4),
    )


def compute_capture_rate(
    order_idx: np.ndarray,
    n: int,
    rare_mask: np.ndarray,
    budgets: List[int],
) -> Dict[int, float]:
    n_rare = int(rare_mask.sum())
    if n_rare == 0:
        return {B: float("nan") for B in budgets}
    result = {}
    for B in budgets:
        K = int(round(B / 100 * n))
        reviewed = set(order_idx[:K].tolist())
        captured = sum(1 for i in reviewed if rare_mask[i])
        result[B] = round(captured / n_rare, 4)
    return result


def compute_knn_rarity(
    Z_train: np.ndarray, Z_eval: np.ndarray, k: int
) -> np.ndarray:
    n_eval = len(Z_eval)
    n_train = len(Z_train)
    k_actual = min(k, n_train)
    train_norms_sq = (Z_train ** 2).sum(axis=1)
    eval_norms_sq = (Z_eval ** 2).sum(axis=1)
    dists_k = np.empty(n_eval, dtype=np.float64)
    chunk_size = 256
    for start in range(0, n_eval, chunk_size):
        end = min(start + chunk_size, n_eval)
        Z_chunk = Z_eval[start:end]
        cross = Z_chunk @ Z_train.T
        d2 = eval_norms_sq[start:end, None] - 2.0 * cross + train_norms_sq[None, :]
        np.clip(d2, 0.0, None, out=d2)
        k_vals = np.partition(d2, k_actual - 1, axis=1)[:, k_actual - 1]
        dists_k[start:end] = np.sqrt(k_vals)
    return dists_k


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def badge_approx_order(Z_eval: np.ndarray, P: np.ndarray, pca_components: int = 20) -> np.ndarray:
    n_eval, d = Z_eval.shape
    K_cls = P.shape[1]
    pred_classes = P.argmax(axis=1)
    actual_components = min(pca_components, d, n_eval)
    pca = PCA(n_components=actual_components, random_state=42)
    Z_proj = pca.fit_transform(Z_eval)
    delta = P.copy()
    delta[np.arange(n_eval), pred_classes] -= 1.0
    G = (delta[:, :, None] * Z_proj[:, None, :]).reshape(n_eval, K_cls * actual_components)
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
    xs = np.array([b / 100.0 for b in budgets])
    dB = np.diff(xs)
    Ks = np.array([int(round(b / 100 * n)) for b in budgets], dtype=int)
    n_rare = int(rare_mask.sum())
    base_rare_correct = int(base_correct[rare_mask].sum())
    rare_incorrect = (rare_mask & ~base_correct).astype(np.int32)
    aucs = np.zeros(n_shuffles)
    for start in range(0, n_shuffles, batch):
        end = min(start + batch, n_shuffles)
        nb = end - start
        perms = np.array([rng.permutation(n) for _ in range(nb)], dtype=np.int32)
        ri_perms = rare_incorrect[perms]
        cum_ri = np.cumsum(ri_perms, axis=1, dtype=np.int64)
        rares_mat = np.zeros((nb, len(Ks)), dtype=np.float64)
        for bi, K in enumerate(Ks):
            if K == 0:
                rares_mat[:, bi] = base_rare_correct / n_rare
            else:
                rares_mat[:, bi] = (base_rare_correct + cum_ri[:, K - 1]) / n_rare
        aucs[start:end] = np.sum((rares_mat[:, :-1] + rares_mat[:, 1:]) / 2.0 * dB, axis=1)
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
    xs = np.array([b / 100.0 for b in budgets])
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
                rares_bs.append(float(c[rm_sub].mean()) if n_rare_bs > 0 else float("nan"))
            return _trapz(np.array(rares_bs), xs)

        gaps.append(_auc_rare(pos_a) - _auc_rare(pos_b))

    lo = float(np.percentile(gaps, 2.5))
    hi = float(np.percentile(gaps, 97.5))
    return {
        "label_a": label_a,
        "label_b": label_b,
        "mean_gap": round(float(np.mean(gaps)), 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "n_bootstrap": n_boot,
        "significant_at_alpha05": bool(lo > 0),
        "interpretation": (
            f"Bootstrap CI on rare_recall_AUC({label_a}) - rare_recall_AUC({label_b}). "
            f"CI=[{lo:.4f}, {hi:.4f}]. "
            f"{'有意(0を含まない)' if lo > 0 else '非有意(0を含む)'} at alpha=0.05."
        ),
    }


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# RCTA-specific functions
# ---------------------------------------------------------------------------

def compute_class_weights(
    train_labels: List[str],
    classes: List[str],
    weight_type: str,
    excluded: set,
) -> np.ndarray:
    """
    Compute per-class weight vector from training labels.
    Classes in `excluded` get weight 0.
    Weight types:
      binary      : w_c = 1 if not excluded
      inv_freq    : w_c = N / count_c if not excluded
      log_inv_freq: w_c = log(N / count_c) if not excluded
    Weights are then max-normalised so max(w_c) = 1.
    Returns float64 array of length len(classes).
    """
    N = len(train_labels)
    cnt = Counter(train_labels)
    w = np.zeros(len(classes), dtype=np.float64)
    for i, c in enumerate(classes):
        if c in excluded:
            w[i] = 0.0
            continue
        freq = cnt.get(c, 0)
        if weight_type == "binary":
            w[i] = 1.0
        elif weight_type == "inv_freq":
            # If class is unknown (not in training), treat freq=0.5 (pseudocount)
            effective_freq = max(freq, 0.5)
            w[i] = float(N) / effective_freq
        elif weight_type == "log_inv_freq":
            effective_freq = max(freq, 0.5)
            w[i] = math.log(float(N) / effective_freq)
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
    # Max-normalise (so scale is always [0,1])
    max_w = float(w.max())
    if max_w > 0:
        w = w / max_w
    return w


def compute_rcta_score(P: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    RCTA score = sum_c p_c * w_c  for each sample.
    P: (n_eval, n_classes) float64
    w: (n_classes,) float64 (already max-normalised)
    Returns: (n_eval,) float64 in [0, 1]
    """
    return (P @ w).astype(np.float64)


def compute_combined_order(
    rcta: np.ndarray,
    uncertainty: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    combined = alpha * rcta_norm + (1 - alpha) * uncertainty_norm
    Higher score = review first.
    """
    rcta_norm = minmax_normalize(rcta)
    unc_norm = minmax_normalize(uncertainty)
    score = alpha * rcta_norm + (1.0 - alpha) * unc_norm
    return np.argsort(-score)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_total = time.time()
    os.makedirs(BMK, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 0: env log
    # ------------------------------------------------------------------
    env_log = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "script": os.path.abspath(__file__),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": "RCTA: rare-class prior targeted acquisition vs confidence/RAU/BADGE",
        "gpu_rule": "CUDA_VISIBLE_DEVICES='' set before any torch import",
    }
    _save("_meta", {"env": env_log, "status": "running"})

    # ------------------------------------------------------------------
    # Step 1: データ読み込み (rau_true_rare_analysis.py と完全同一)
    # ------------------------------------------------------------------
    print("\n[Step 1] データ読み込み...", flush=True)
    t0 = time.time()

    se = np.load(os.path.join(OUT, "embeddings.npy")).astype(np.float64)
    prov = json.load(open(os.path.join(OUT, "provisional_labels.json")))
    EMB: Dict[str, np.ndarray] = {}
    for i, r in enumerate(prov):
        EMB[r["crop"]] = se[i]

    ve = np.load(os.path.join(OUT, "video_embeddings.npy")).astype(np.float64)
    vit = json.load(open(os.path.join(OUT, "video_embed_index.json")))
    for i, it in enumerate(vit):
        EMB[it["crop"]] = ve[i]

    seed_lab = {
        r["crop"]: r["label"]
        for r in json.load(open(os.path.join(OUT, "seed_labels.json")))["labels"]
    }
    final = {
        r["crop"]: r["label"]
        for r in json.load(open(os.path.join(OUT, "labels_video_v1.json")))["labels"]
    }
    seed_set = set(json.load(open(os.path.join(OUT, "seed_crops.json"))))

    # Training set (identical to rarity_aware_triage.py)
    X_tr: List[np.ndarray] = []
    y_tr: List[str] = []
    for c, l in seed_lab.items():
        if c in EMB:
            X_tr.append(EMB[c])
            y_tr.append(l)
    still_labels_data = json.load(open(os.path.join(OUT, "labels_still_v1.json")))
    for r in still_labels_data["labels"]:
        if r["crop"] in EMB:
            X_tr.append(EMB[r["crop"]])
            y_tr.append(r["label"])
    Z_train = np.stack(X_tr)
    N_train = len(y_tr)

    # Training class frequency (for RCTA weights)
    train_cnt = Counter(y_tr)
    print(f"  Training labels: {dict(sorted(train_cnt.items(), key=lambda x: -x[1]))}", flush=True)

    # Classifier (identical to rarity_aware_triage.py)
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ).fit(Z_train, y_tr)
    classes: List[str] = list(clf.classes_)
    n_classes = len(classes)

    # Eval set (seed-excluded, ground-truth available)
    crops = [
        it["crop"]
        for it in vit
        if it["crop"] not in seed_set and it["crop"] in final
    ]
    Z_eval = np.stack([EMB[c] for c in crops])
    n = len(crops)
    true_labels: List[str] = [final[c] for c in crops]
    P = clf.predict_proba(Z_eval).astype(np.float64)  # (n, n_classes)
    pred_labels: List[str] = [classes[int(i)] for i in P.argmax(axis=1)]
    conf = P.max(axis=1).astype(np.float64)
    base_correct = np.array([pred_labels[i] == true_labels[i] for i in range(n)], dtype=bool)
    base_acc = float(base_correct.mean())

    # Masks
    true_rare_mask = np.array(
        [t not in COMMON and t != FALSE_DET_LABEL and t != UNKNOWN_LABEL
         for t in true_labels], dtype=bool
    )
    unknown_mask = np.array([t == UNKNOWN_LABEL for t in true_labels], dtype=bool)
    rare_all_mask = np.array(
        [t not in COMMON and t != FALSE_DET_LABEL for t in true_labels], dtype=bool
    )

    n_true_rare = int(true_rare_mask.sum())
    n_unknown = int(unknown_mask.sum())
    n_rare_all = int(rare_all_mask.sum())

    print(f"  n_eval={n}, base_acc={base_acc:.4f}", flush=True)
    print(f"  true_rare={n_true_rare}, unknown={n_unknown}, rare_all={n_rare_all}", flush=True)

    data_info = {
        "n_eval": n,
        "n_train": N_train,
        "n_classes": n_classes,
        "base_acc": round(base_acc, 4),
        "n_true_rare": n_true_rare,
        "n_unknown": n_unknown,
        "n_rare_all": n_rare_all,
        "train_class_counts": dict(sorted(train_cnt.items(), key=lambda x: -x[1])),
        "classes_in_classifier": classes,
        "excluded_from_rare_weight": sorted(EXCLUDED_FROM_RARE_WEIGHT),
        "consistency_note": (
            "Eval set, classifier, train set are IDENTICAL to "
            "rarity_aware_triage.py and rau_true_rare_analysis.py. "
            "true_rare_mask: excludes COMMON, 不明, 誤検出. n=173."
        ),
        "data_hashes": {
            "embeddings.npy": file_sha256(os.path.join(OUT, "embeddings.npy")),
            "video_embeddings.npy": file_sha256(os.path.join(OUT, "video_embeddings.npy")),
            "seed_labels.json": file_sha256(os.path.join(OUT, "seed_labels.json")),
            "labels_video_v1.json": file_sha256(os.path.join(OUT, "labels_video_v1.json")),
            "labels_still_v1.json": file_sha256(os.path.join(OUT, "labels_still_v1.json")),
        },
    }
    _save("data_info", data_info)
    print(f"  Step 1 done in {time.time()-t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Step 2: RCTA weight vectors (from training only)
    # ------------------------------------------------------------------
    print("\n[Step 2] RCTA weight vectors 計算...", flush=True)
    weight_vectors: Dict[str, np.ndarray] = {}
    weight_info: Dict[str, Dict] = {}
    for wtype in WEIGHT_TYPES:
        w = compute_class_weights(y_tr, classes, wtype, EXCLUDED_FROM_RARE_WEIGHT)
        weight_vectors[wtype] = w
        per_class = {c: round(float(w[i]), 4) for i, c in enumerate(classes)}
        weight_info[wtype] = {
            "nonzero_classes": {c: v for c, v in per_class.items() if v > 0},
            "zero_classes": [c for c, v in per_class.items() if v == 0],
            "note": "Derived from training label frequency only. No eval gt used.",
        }
        print(f"  [{wtype}] nonzero: {weight_info[wtype]['nonzero_classes']}", flush=True)
    _save("weight_vectors", weight_info)

    # ------------------------------------------------------------------
    # Step 3: RCTA scores
    # ------------------------------------------------------------------
    print("\n[Step 3] RCTA スコア計算...", flush=True)
    rcta_scores: Dict[str, np.ndarray] = {}
    for wtype in WEIGHT_TYPES:
        score = compute_rcta_score(P, weight_vectors[wtype])
        rcta_scores[wtype] = score
        print(
            f"  [{wtype}] score: min={score.min():.4f}  mean={score.mean():.4f}  "
            f"max={score.max():.4f}  nonzero={int((score > 0).sum())}",
            flush=True
        )
    score_stats = {
        wtype: {
            "min": round(float(s.min()), 4),
            "mean": round(float(s.mean()), 4),
            "max": round(float(s.max()), 4),
            "n_nonzero": int((s > 0).sum()),
        }
        for wtype, s in rcta_scores.items()
    }
    _save("rcta_score_stats", score_stats)

    # ------------------------------------------------------------------
    # Step 4: reference baselines orderings
    # ------------------------------------------------------------------
    print("\n[Step 4] Baseline orderings...", flush=True)
    uncertainty = (1.0 - conf).astype(np.float64)
    conf_order = np.argsort(conf)   # least-confident first

    print("  kNN rarity (k=10)...", flush=True)
    t0 = time.time()
    rarity = compute_knn_rarity(Z_train, Z_eval, K_PRIMARY)
    rau_order = np.argsort(
        -(minmax_normalize(uncertainty) * (1.0 - LAMBDA_PRIMARY)
          + minmax_normalize(rarity) * LAMBDA_PRIMARY)
    )

    print(f"  BADGE-approx PCA-{K_PCA_BADGE}...", flush=True)
    badge_order = badge_approx_order(Z_eval, P, pca_components=K_PCA_BADGE)
    print(f"  Baselines done in {time.time()-t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Step 5: RCTA combination grid + pure RCTA orderings
    # ------------------------------------------------------------------
    print("\n[Step 5] RCTA orderings (all weight types x alpha grid)...", flush=True)
    # We collect all orderings keyed by human-readable name
    orderings: Dict[str, np.ndarray] = {
        "confidence": conf_order,
        f"RAU_lambda{LAMBDA_PRIMARY}_k{K_PRIMARY}": rau_order,
        "BADGE_pca20": badge_order,
    }
    for wtype in WEIGHT_TYPES:
        for alpha in ALPHA_GRID:
            key = f"RCTA_{wtype}_alpha{alpha}"
            orderings[key] = compute_combined_order(rcta_scores[wtype], uncertainty, alpha)

    # ------------------------------------------------------------------
    # Step 6: Evaluate all orderings on all masks
    # ------------------------------------------------------------------
    print("\n[Step 6] 全ordering x 全maskでAUC・捕捉率評価...", flush=True)
    masks = {
        "true_rare_n173": true_rare_mask,
        "unknown_n1307": unknown_mask,
        "rare_all_n1480": rare_all_mask,
    }

    def evaluate_ordering(order: np.ndarray, mask: np.ndarray) -> Dict:
        _, auc_acc, _, auc_rare = compute_auc_from_order(
            order, base_correct, n, BUDGETS, mask
        )
        _, auc_acc_b30, _, auc_rare_b30 = compute_auc_from_order(
            order, base_correct, n, BUDGETS_PARTIAL, mask
        )
        cap = compute_capture_rate(order, n, mask, CAPTURE_BUDGETS)
        return {
            "auc_rare_recall": auc_rare,
            "auc_acc": auc_acc,
            "auc_rare_recall_b30": auc_rare_b30,
            "auc_acc_b30": auc_acc_b30,
            "capture_B10pct": cap[10],
            "capture_B20pct": cap[20],
            "capture_B30pct": cap[30],
        }

    all_results: Dict[str, Dict[str, Dict]] = {}
    for mask_name, mask in masks.items():
        all_results[mask_name] = {}
        for method_name, order in orderings.items():
            r = evaluate_ordering(order, mask)
            all_results[mask_name][method_name] = r
        print(f"\n  === {mask_name} ===", flush=True)
        # Print sorted by AUC descending
        sorted_methods = sorted(
            all_results[mask_name].items(),
            key=lambda kv: kv[1]["auc_rare_recall"],
            reverse=True
        )
        for mname, r in sorted_methods:
            print(
                f"    {mname:<45} AUC={r['auc_rare_recall']:.4f}  "
                f"B10={r['capture_B10pct']:.4f}  B20={r['capture_B20pct']:.4f}  B30={r['capture_B30pct']:.4f}",
                flush=True
            )
    _save("all_results", all_results)

    # ------------------------------------------------------------------
    # Step 7: Random baselines (true_rare_n173 focus)
    # ------------------------------------------------------------------
    print("\n[Step 7] Random baseline AUC (2k shuffles, true_rare_n173)...", flush=True)
    random_info: Dict[str, Dict] = {}
    for mask_name, mask in masks.items():
        rng_rand = np.random.default_rng(SEED_PERM + hash(mask_name) % 10000)
        aucs_rand = fast_rand_rare_aucs(N_RAND, base_correct, mask, BUDGETS, n, rng_rand)
        aucs_rand_b30 = fast_rand_rare_aucs(N_RAND, base_correct, mask, BUDGETS_PARTIAL, n, rng_rand)
        random_info[mask_name] = {
            "auc_rare_mean": round(float(np.mean(aucs_rand)), 4),
            "auc_rare_std": round(float(np.std(aucs_rand)), 4),
            "auc_rare_b30_mean": round(float(np.mean(aucs_rand_b30)), 4),
            "auc_rare_b30_std": round(float(np.std(aucs_rand_b30)), 4),
            "n_shuffles": N_RAND,
        }
        print(
            f"  [{mask_name}] random: AUC={random_info[mask_name]['auc_rare_mean']:.4f}"
            f"±{random_info[mask_name]['auc_rare_std']:.4f}",
            flush=True
        )
    _save("random_baselines", random_info)

    # ------------------------------------------------------------------
    # Step 8: 統計検定 (true_rare_n173 フォーカス、全RCTA variant vs confidence)
    # ------------------------------------------------------------------
    print("\n[Step 8] 統計検定 (true_rare_n173, RCTA vs confidence, RCTA vs random)...", flush=True)
    mask = true_rare_mask
    stats_results: Dict[str, Dict] = {}

    # Reference AUCs
    conf_auc_n173 = all_results["true_rare_n173"]["confidence"]["auc_rare_recall"]

    # Permutation test for each RCTA variant vs random
    rng_perm_base = np.random.default_rng(SEED_PERM)
    rand_aucs_perm = fast_rand_rare_aucs(N_PERM, base_correct, mask, BUDGETS, n, rng_perm_base)

    # Bootstrap + permutation for each RCTA variant (all weight types x all alphas)
    boot_counter = 0
    for wtype in WEIGHT_TYPES:
        for alpha in ALPHA_GRID:
            key = f"RCTA_{wtype}_alpha{alpha}"
            order_rcta = orderings[key]
            rcta_auc = all_results["true_rare_n173"][key]["auc_rare_recall"]

            # Permutation test
            n_exceed = int(np.sum(rand_aucs_perm >= rcta_auc))
            pval_perm = (n_exceed + 1) / (N_PERM + 1)

            # Bootstrap CI vs confidence
            rng_boot = np.random.default_rng(SEED_BOOT + boot_counter)
            ci_vs_conf = bootstrap_rare_auc_gap(
                order_rcta, conf_order,
                base_correct, mask, BUDGETS, n, N_BOOT, rng_boot,
                label_a=key, label_b="confidence",
            )
            boot_counter += 1

            # Bootstrap CI vs RAU
            rng_boot2 = np.random.default_rng(SEED_BOOT + boot_counter)
            ci_vs_rau = bootstrap_rare_auc_gap(
                order_rcta, rau_order,
                base_correct, mask, BUDGETS, n, N_BOOT, rng_boot2,
                label_a=key, label_b=f"RAU_lambda{LAMBDA_PRIMARY}",
            )
            boot_counter += 1

            stats_results[key] = {
                "rcta_auc_n173": rcta_auc,
                "perm_p_vs_random": round(pval_perm, 6),
                "perm_significant": bool(pval_perm < 0.05),
                "bootstrap_vs_confidence": ci_vs_conf,
                "bootstrap_vs_rau": ci_vs_rau,
            }
            print(
                f"  {key}: AUC={rcta_auc:.4f}  vs_conf_gap={ci_vs_conf['mean_gap']:+.4f}"
                f"  CI={ci_vs_conf['ci95']}  sig_vs_conf={ci_vs_conf['significant_at_alpha05']}  "
                f"perm_p={pval_perm:.4f}",
                flush=True
            )

    _save("statistical_tests_true_rare_n173", stats_results)

    # ------------------------------------------------------------------
    # Step 9: サマリー集計
    # ------------------------------------------------------------------
    print("\n[Step 9] サマリー集計...", flush=True)

    # Best RCTA by AUC on true_rare_n173
    rcta_methods = [k for k in all_results["true_rare_n173"].keys()
                    if k.startswith("RCTA_")]
    best_rcta_key = max(rcta_methods,
                        key=lambda k: all_results["true_rare_n173"][k]["auc_rare_recall"])
    best_rcta_auc = all_results["true_rare_n173"][best_rcta_key]["auc_rare_recall"]

    # Binary-alpha0.0 (pure RCTA, no uncertainty, no tunable alpha from eval)
    # This is the cleanest: weights from training frequency, no eval-based selection
    rcta_binary_pure_key = "RCTA_binary_alpha1.0"  # alpha=1.0 means pure RCTA (no uncertainty)
    rcta_binary_pure_auc = all_results["true_rare_n173"][rcta_binary_pure_key]["auc_rare_recall"]

    # alpha=0.5 binary (canonical combination)
    rcta_binary_half_key = "RCTA_binary_alpha0.5"
    rcta_binary_half_auc = all_results["true_rare_n173"][rcta_binary_half_key]["auc_rare_recall"]

    summary = {
        "focus": "true_rare_n173",
        "baselines": {
            m: all_results["true_rare_n173"][m]
            for m in ["confidence", f"RAU_lambda{LAMBDA_PRIMARY}_k{K_PRIMARY}", "BADGE_pca20"]
        },
        "random": random_info["true_rare_n173"],
        "confidence_auc": conf_auc_n173,
        "best_rcta_by_eval_auc": {
            "key": best_rcta_key,
            "auc": best_rcta_auc,
            "note": "Selected by same eval set — subject to overfitting bias",
        },
        "rcta_binary_pure": {
            "key": rcta_binary_pure_key,
            "auc": rcta_binary_pure_auc,
            "note": "alpha=1.0 = pure rare-class mass, no uncertainty. No alpha selection from eval.",
        },
        "rcta_binary_half": {
            "key": rcta_binary_half_key,
            "auc": rcta_binary_half_auc,
            "note": "alpha=0.5 = equal mix of RCTA and uncertainty.",
        },
        "beats_confidence": {
            "best_rcta": bool(best_rcta_auc > conf_auc_n173),
            "best_rcta_auc": best_rcta_auc,
            "confidence_auc": conf_auc_n173,
        },
    }
    _save("summary", summary)

    # ------------------------------------------------------------------
    # Step 10: 最終 meta 更新
    # ------------------------------------------------------------------
    elapsed = round(time.time() - t_total, 1)
    _save("_meta", {
        "env": env_log,
        "status": "completed",
        "elapsed_s": elapsed,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    print(f"\n[Done] JSON saved: {RESULT_PATH}  ({elapsed}s)", flush=True)

    # ------------------------------------------------------------------
    # Step 11: メモ (rare_targeted_note.md) 生成
    # ------------------------------------------------------------------
    write_note(
        all_results=all_results,
        stats_results=stats_results,
        random_info=random_info,
        summary=summary,
        weight_info=weight_info,
        conf_auc_n173=conf_auc_n173,
        n=n,
        n_true_rare=n_true_rare,
        n_unknown=n_unknown,
        n_rare_all=n_rare_all,
        best_rcta_key=best_rcta_key,
        best_rcta_auc=best_rcta_auc,
        rcta_binary_pure_key=rcta_binary_pure_key,
        rcta_binary_pure_auc=rcta_binary_pure_auc,
        rcta_binary_half_key=rcta_binary_half_key,
        rcta_binary_half_auc=rcta_binary_half_auc,
        lambda_primary=LAMBDA_PRIMARY,
    )
    print(f"[Done] Note saved: {NOTE_PATH}", flush=True)


def write_note(
    all_results: Dict,
    stats_results: Dict,
    random_info: Dict,
    summary: Dict,
    weight_info: Dict,
    conf_auc_n173: float,
    n: int,
    n_true_rare: int,
    n_unknown: int,
    n_rare_all: int,
    best_rcta_key: str,
    best_rcta_auc: float,
    rcta_binary_pure_key: str,
    rcta_binary_pure_auc: float,
    rcta_binary_half_key: str,
    rcta_binary_half_auc: float,
    lambda_primary: float,
) -> None:
    """日本語メモ: RCTA評価の誠実な分析と論文反映文案"""

    r_n173 = all_results["true_rare_n173"]
    rau_key = f"RAU_lambda{lambda_primary}_k{K_PRIMARY}"
    rand = random_info["true_rare_n173"]

    rau_auc = r_n173[rau_key]["auc_rare_recall"]
    badge_auc = r_n173["BADGE_pca20"]["auc_rare_recall"]

    # Honest verdict construction
    rcta_binary_pure_sig = stats_results[rcta_binary_pure_key]["bootstrap_vs_confidence"]["significant_at_alpha05"]
    rcta_binary_half_sig = stats_results[rcta_binary_half_key]["bootstrap_vs_confidence"]["significant_at_alpha05"]
    best_rcta_sig = stats_results[best_rcta_key]["bootstrap_vs_confidence"]["significant_at_alpha05"]

    ci_pure = stats_results[rcta_binary_pure_key]["bootstrap_vs_confidence"]
    ci_half = stats_results[rcta_binary_half_key]["bootstrap_vs_confidence"]
    ci_best = stats_results[best_rcta_key]["bootstrap_vs_confidence"]

    # Determine verdict
    if best_rcta_sig and best_rcta_auc > conf_auc_n173:
        if best_rcta_key == rcta_binary_pure_key or best_rcta_key == rcta_binary_half_key:
            # Best is also a principled choice (no alpha overfitting)
            verdict_level = "STRONG"
        else:
            verdict_level = "CONDITIONAL"  # best selected on eval
        verdict = (
            f"RCTA(最良: {best_rcta_key}, AUC={best_rcta_auc:.4f})はconfidence"
            f"(AUC={conf_auc_n173:.4f})を真希少種n=173で有意に上回る"
            f"(bootstrap CI {ci_best['ci95']}, α=0.05)。\n"
            f"ただし最良手法の選択は同じevalセットで行っており、alpha過適合リスクあり。"
        )
    else:
        verdict_level = "NEGATIVE"
        verdict = (
            f"RCTAはいずれの重み・alpha設定でも真希少種n=173においてconfidence"
            f"(AUC={conf_auc_n173:.4f})を有意に上回ることができなかった"
            f"(bootstrap CI, α=0.05, すべて非有意)。\n"
            f"最良RCTA: {best_rcta_key} AUC={best_rcta_auc:.4f}。\n"
            f"純粋RCTA(binary,alpha=1.0): AUC={rcta_binary_pure_auc:.4f}。"
        )

    # Build full grid table rows for true_rare_n173
    grid_rows = []
    for wtype in WEIGHT_TYPES:
        for alpha in ALPHA_GRID:
            key = f"RCTA_{wtype}_alpha{alpha}"
            r = r_n173[key]
            ci = stats_results[key]["bootstrap_vs_confidence"]
            sig_mark = "**SIG**" if ci["significant_at_alpha05"] else "n.s."
            grid_rows.append(
                f"| {key:<45} | {r['auc_rare_recall']:.4f} | "
                f"{r['capture_B10pct']:.1%} | {r['capture_B20pct']:.1%} | "
                f"{r['capture_B30pct']:.1%} | {ci['mean_gap']:+.4f} | {ci['ci95']} | {sig_mark} |"
            )

    lines = [
        "# rare_targeted_note.md",
        f"目的: 希少クラス事前確率質量スコア(RCTA)による真希少種(n=173)の直接狙い撃ち実証",
        "",
        "## 1. 背景・問題設定",
        "",
        f"- 既存RAU(埋め込みkNN希少性 + 不確実性): 真希少種n=173ではconfidence(AUC={conf_auc_n173:.4f})に劣る"
        f"(RAU AUC={rau_auc:.4f}, 非有意)。",
        f"- 原因: kNN埋め込み密度が「不明」n={n_unknown}を過剰に前倒しする。",
        f"- 新手法(RCTA): 分類器の予測確率のうち真希少クラスに割かれた確率質量を取得スコアとする。",
        f"  w_c = 0 for COMMON, 不明, 誤検出  →  不明への過剰優先を構造的に排除。",
        f"  w_c > 0 for 真希少クラス(訓練頻度が低い同定済み種)。",
        "",
        "## 2. 重みベクトル (訓練ラベル頻度由来・evalラベル不使用)",
        "",
    ]
    for wtype, info in weight_info.items():
        lines.append(f"### {wtype}")
        lines.append(f"  非ゼロクラス: {info['nonzero_classes']}")
        lines.append(f"  ゼロクラス: {info['zero_classes']}")
        lines.append("")

    lines += [
        "## 3. 全グリッド結果 (true_rare_n173, チェリーピック禁止)",
        "",
        "### 参照ベースライン",
        f"| 手法 | AUC | B10 | B20 | B30 |",
        f"|------|-----|-----|-----|-----|",
        f"| confidence | {conf_auc_n173:.4f} | {r_n173['confidence']['capture_B10pct']:.1%} | "
        f"{r_n173['confidence']['capture_B20pct']:.1%} | {r_n173['confidence']['capture_B30pct']:.1%} |",
        f"| RAU(λ=0.5,k=10) | {rau_auc:.4f} | {r_n173[rau_key]['capture_B10pct']:.1%} | "
        f"{r_n173[rau_key]['capture_B20pct']:.1%} | {r_n173[rau_key]['capture_B30pct']:.1%} |",
        f"| BADGE_pca20 | {badge_auc:.4f} | {r_n173['BADGE_pca20']['capture_B10pct']:.1%} | "
        f"{r_n173['BADGE_pca20']['capture_B20pct']:.1%} | {r_n173['BADGE_pca20']['capture_B30pct']:.1%} |",
        f"| random (mean) | {rand['auc_rare_mean']:.4f}±{rand['auc_rare_std']:.4f} | - | - | - |",
        "",
        "### RCTA全グリッド (vs confidence: bootstrap 95% CI)",
        "",
        "| 手法 | AUC | B10 | B20 | B30 | gap_vs_conf | CI95 | sig |",
        "|------|-----|-----|-----|-----|-------------|------|-----|",
    ]
    lines += grid_rows
    lines += [
        "",
        "## 4. Honest判定",
        "",
        f"**判定レベル: {verdict_level}**",
        "",
        verdict,
        "",
        "## 5. 誠実性確認・リーク監査",
        "",
        "- w_c は訓練ラベル頻度のみから計算。evalラベルを取得基準に使っていない。",
        "- alpha グリッド選択を同一evalセットで行っている → alpha過適合リスクあり(明記)。",
        "- binary weight(alpha=1.0)は eval から選んだパラメータなし = 最もリーク-free。",
        "- eval set / 分類器 / 訓練setはrarity_aware_triage.pyと完全一致。",
        "- 全グリッド値を報告(チェリーピック禁止)。",
        "",
        "## 6. 論文反映文案",
        "",
    ]

    if verdict_level == "NEGATIVE":
        lines += [
            "【新規性として維持不可 / 主張精緻化の文案】",
            "",
            "「We further investigated whether a rare-class probability mass score (RCTA)—",
            "which weights predicted probabilities by inverse training frequency,",
            "assigning zero weight to the 'Unknown' category—could surpass confidence",
            "on identified rare species (n=173). Across all weight variants (binary,",
            "inv_freq, log_inv_freq) and combination coefficients α∈{0,0.25,0.5,0.75,1.0},",
            f"no configuration significantly outperformed confidence (AUC={conf_auc_n173:.4f})",
            f"on this strict subset (bootstrap 95% CI, α=0.05, all non-significant).",
            "These results indicate that for verified rare species identification,",
            "confidence-based triage is already an effective baseline.",
            "The contribution of RAU therefore lies in triage efficiency across all",
            "uncertain crops—including unresolvable 'Unknown' cases—rather than",
            "specifically targeting definitively identified rare wildlife.」",
            "",
            "→ 論文の新規性の主軸を「不確実クロップ全体(不明含む)の効率的トリアージ」として"
            "正直に精緻化。捏造なし。",
        ]
    elif verdict_level == "CONDITIONAL":
        lines += [
            "【条件付き新規性の文案 (alpha選択のバイアスを明記)】",
            "",
            f"「To target genuinely identifiable rare species, we propose RCTA, which",
            f"weights classifier softmax probabilities by inverse training frequency,",
            f"assigning zero weight to the 'Unknown' class.",
            f"The best-performing configuration ({best_rcta_key}, AUC={best_rcta_auc:.4f})",
            f"significantly outperforms confidence (AUC={conf_auc_n173:.4f})",
            f"on identified rare species n=173 (bootstrap CI {ci_best['ci95']}, α=0.05).",
            f"We note that this α was selected on the same evaluation set;",
            f"held-out validation would be required for deployment-time α selection.」",
        ]
    else:
        lines += [
            f"【強い新規性の文案】",
            "",
            f"「RCTA ({best_rcta_key}) achieves rare-recall AUC={best_rcta_auc:.4f}",
            f"on n=173 identified rare species, significantly outperforming",
            f"confidence (AUC={conf_auc_n173:.4f}, bootstrap CI {ci_best['ci95']}, α=0.05)",
            f"without using evaluation ground-truth labels for weight selection.」",
        ]

    with open(NOTE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
