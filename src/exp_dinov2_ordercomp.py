#!/usr/bin/env python3
"""
exp_dinov2_ordercomp.py — E1: DINOv2 backbone order comparison (backbone robustness).

Purpose:
  Replicate pipeline_canonical.py order-comparison experiment using DINOv2 (768-dim)
  embeddings instead of BioCLIP (512-dim) embeddings.  Demonstrates that the result
  "confidence ordering ≥ active-learning baselines" is backbone-independent.

Protocol (identical to pipeline_canonical.py):
  - Same training split: seed_labels.json + labels_still_v1.json
  - Same eval split:     video crops NOT in seed_crops, with ground-truth in labels_video_v1
  - Same LR:            class_weight='balanced', max_iter=2000, random_state=42, float64
  - Same ordering seeds: SEED_BADGE=6906, SEED_PERM=6906, SEED_BOOT=6906
  - Same N_PERM=10000, N_BOOT=500, BUDGETS=[0,5,10,20,30,50,100]
  - Same Coreset (k-center greedy), BADGE-approx (PCA-20 gradient embedding)
  - Same p-value formula: (n_exceed+1)/(N_PERM+1) [Phipson & Smyth 2010]

Additions beyond canonical:
  - RCTA-binary: rare-class prior score (sum of softmax probs over true-rare classes),
    evaluated on true_rare_n173 mask (same as rare_targeted_acquisition.py definition).

GPU rule: CUDA_VISIBLE_DEVICES="" set before any torch import.
No GPU operations — LR + numpy only.

Output: outputs/benchmark/pipeline_dinov2_ordercomp.json
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
from typing import Dict, List, Tuple

# GPU rule: must precede any torch import
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HOME = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HOME, "outputs")
BMK = os.path.join(OUT, "benchmark")
RESULT_PATH = os.path.join(BMK, "pipeline_dinov2_ordercomp.json")

# Safety: abort if output already exists
if os.path.exists(RESULT_PATH):
    print(f"[ERROR] Output already exists: {RESULT_PATH}")
    print("  既存ファイルの上書き防止のため終了します。")
    sys.exit(1)

os.makedirs(BMK, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants — must match pipeline_canonical.py exactly
# ---------------------------------------------------------------------------
COMMON = {"ニホンジカ", "タヌキ", "イノシシ", "ニホンザル", "ニホンノウサギ"}
BUDGETS = [0, 5, 10, 20, 30, 50, 100]
BUDGETS_PARTIAL = [0, 5, 10, 20, 30]
SEED_BADGE = 6906
SEED_PERM = 6906
SEED_BOOT = 6906
N_PERM = 10_000
N_BOOT = 500

# ---------------------------------------------------------------------------
# Utility functions (identical to pipeline_canonical.py)
# ---------------------------------------------------------------------------

def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(fn(y, x))


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def file_sha256(path: str, n_bytes: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(n_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_auc_from_order(
    order_idx: np.ndarray,
    base_correct: np.ndarray,
    n: int,
    budgets: List[int],
    rare_mask: np.ndarray,
) -> Tuple[List[float], float, List[float], float]:
    """Returns (accs, auc_acc, rares, auc_rare) — identical to pipeline_canonical.py."""
    xs = np.array([b / 100.0 for b in budgets])
    accs, rares = [], []
    for B in budgets:
        K = int(round(B / 100 * n))
        reviewed = set(order_idx[:K].tolist())
        correct = base_correct.copy()
        for i in reviewed:
            correct[i] = True
        accs.append(float(correct.mean()))
        rares.append(float(correct[rare_mask].mean()) if rare_mask.any() else float("nan"))
    return (
        [round(a, 4) for a in accs],
        round(_trapz(np.array(accs), xs), 4),
        [round(r, 4) for r in rares],
        round(_trapz(np.array(rares), xs), 4),
    )


def fast_rand_aucs(
    n_shuffles: int, base_correct: np.ndarray, budgets: List[int],
    n: int, rng: np.random.Generator, batch: int = 500
) -> np.ndarray:
    """Vectorized AUC computation for random permutations — identical to pipeline_canonical.py."""
    xs = np.array([b / 100.0 for b in budgets])
    Ks = np.array([int(round(b / 100 * n)) for b in budgets], dtype=int)
    dB = np.diff(xs)
    total_base = int(base_correct.sum())
    base_int = base_correct.astype(np.int32)

    aucs = np.zeros(n_shuffles)
    for start in range(0, n_shuffles, batch):
        end = min(start + batch, n_shuffles)
        nb = end - start
        perms = np.array([rng.permutation(n) for _ in range(nb)], dtype=np.int32)
        bc_perms = base_int[perms]
        cum = np.hstack([np.zeros((nb, 1), dtype=np.int64),
                         np.cumsum(bc_perms, axis=1, dtype=np.int64)])
        accs_mat = np.zeros((nb, len(Ks)), dtype=np.float64)
        for bi, K in enumerate(Ks):
            accs_mat[:, bi] = (K + total_base - cum[:, K]) / n
        aucs[start:end] = np.sum(
            (accs_mat[:, :-1] + accs_mat[:, 1:]) / 2.0 * dB, axis=1
        )
    return aucs


def bootstrap_auc_gap(
    order_a: np.ndarray, order_b: np.ndarray,
    base_correct: np.ndarray, budgets: List[int],
    n: int, n_boot: int, rng: np.random.Generator
) -> Dict:
    """Bootstrap 95% CI on AUC(order_a) - AUC(order_b) — identical to pipeline_canonical.py."""
    rank_a = np.empty(n, dtype=np.int64)
    rank_a[order_a] = np.arange(n)
    rank_b = np.empty(n, dtype=np.int64)
    rank_b[order_b] = np.arange(n)

    xs = np.array([b / 100.0 for b in budgets])
    gaps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        n_bs = len(idx)
        bc_sub = base_correct[idx]

        pos_a = np.argsort(rank_a[idx])
        pos_b = np.argsort(rank_b[idx])

        def auc_bs(positions: np.ndarray) -> float:
            accs_bs = []
            for B in budgets:
                K = int(round(B / 100 * n_bs))
                reviewed_pos = set(positions[:K].tolist())
                correct_bs = bc_sub.copy()
                for pos in reviewed_pos:
                    correct_bs[pos] = True
                accs_bs.append(float(correct_bs.mean()))
            return _trapz(np.array(accs_bs), xs)

        gaps.append(auc_bs(pos_a) - auc_bs(pos_b))

    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {
        "mean_gap": round(float(np.mean(gaps)), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "n_bootstrap": n_boot,
        "note": (
            "Bootstrap CI on AUC(A) - AUC(B). "
            "Positive gap = A achieves higher AUC. "
            "CI excludes 0 → significant at alpha=0.05."
        ),
    }


def coreset_order(Z_train: np.ndarray, Z_eval: np.ndarray) -> np.ndarray:
    """k-center greedy ordering — identical to pipeline_canonical.py."""
    n_eval, n_train = len(Z_eval), len(Z_train)
    chunk = 512

    min_dist = np.full(n_eval, np.inf)
    for start in range(0, n_train, chunk):
        block = Z_train[start:start + chunk]
        diff = Z_eval[:, None, :] - block[None, :, :]
        d2 = (diff ** 2).sum(axis=2)
        min_dist = np.minimum(min_dist, d2.min(axis=1))

    order = []
    for _ in range(n_eval):
        idx = int(np.argmax(min_dist))
        order.append(idx)
        min_dist[idx] = -np.inf
        new_d = ((Z_eval - Z_eval[idx]) ** 2).sum(axis=1)
        min_dist = np.minimum(min_dist, new_d)
        min_dist[idx] = -np.inf
    return np.array(order, dtype=np.int64)


def badge_order(
    Z_eval: np.ndarray, P: np.ndarray,
    rng: np.random.Generator, pca_components: int = 20
) -> np.ndarray:
    """
    BADGE-approx: k-center greedy on LR gradient embedding (PCA-20 projected).
    Identical to pipeline_canonical.py (PCA-20 regardless of embedding dimensionality).
    """
    n_eval, d = Z_eval.shape
    K = P.shape[1]
    pred_classes = P.argmax(axis=1)

    actual_components = min(pca_components, d, n_eval)
    pca = PCA(n_components=actual_components, random_state=42)
    Z_proj = pca.fit_transform(Z_eval)   # (n_eval, 20)

    delta = P.copy()
    delta[np.arange(n_eval), pred_classes] -= 1.0

    G = (delta[:, :, None] * Z_proj[:, None, :]).reshape(n_eval, K * actual_components)

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


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    t_global = time.time()
    result: Dict = {}

    print("=" * 70, flush=True)
    print("exp_dinov2_ordercomp.py  — E1: DINOv2 backbone robustness check", flush=True)
    print(f"Output: {RESULT_PATH}", flush=True)
    print("=" * 70, flush=True)

    # ------------------------------------------------------------------
    # Step 1: Load DINOv2 embeddings and build EMB dict
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading DINOv2 embeddings and data...", flush=True)
    t0 = time.time()

    dino_emb_raw = np.load(os.path.join(OUT, "dinov2_embeddings.npy"))
    print(f"  DINOv2 embeddings shape: {dino_emb_raw.shape}, dtype: {dino_emb_raw.dtype}", flush=True)

    with open(os.path.join(OUT, "dinov2_embed_index.json")) as f:
        dino_idx: List[str] = json.load(f)

    # Map crop → DINOv2 embedding (float64)
    EMB: Dict[str, np.ndarray] = {}
    for i, crop in enumerate(dino_idx):
        EMB[crop] = dino_emb_raw[i].astype(np.float64)

    print(f"  EMB built: {len(EMB)} entries (float64 cast)", flush=True)

    # Load label files (same as pipeline_canonical.py)
    seed_lab_data = json.load(open(os.path.join(OUT, "seed_labels.json")))
    seed_lab: Dict[str, str] = {r["crop"]: r["label"] for r in seed_lab_data["labels"]}

    final_data = json.load(open(os.path.join(OUT, "labels_video_v1.json")))
    final: Dict[str, str] = {r["crop"]: r["label"] for r in final_data["labels"]}

    vid_idx_data = json.load(open(os.path.join(OUT, "video_embed_index.json")))
    hab: Dict[str, str] = {it["crop"]: it.get("habitat") for it in vid_idx_data}
    seed_set: set = set(json.load(open(os.path.join(OUT, "seed_crops.json"))))

    still_data = json.load(open(os.path.join(OUT, "labels_still_v1.json")))

    # ------------------------------------------------------------------
    # Step 2: Build training set (seed video + confirmed stills)
    # ------------------------------------------------------------------
    print("\n[Step 2] Building training set...", flush=True)
    X_tr: List[np.ndarray] = []
    y_tr: List[str] = []
    for c, l in seed_lab.items():
        if c in EMB:
            X_tr.append(EMB[c])
            y_tr.append(l)
    for r in still_data["labels"]:
        if r["crop"] in EMB:
            X_tr.append(EMB[r["crop"]])
            y_tr.append(r["label"])

    Z_train = np.stack(X_tr)
    print(f"  Training set: n={len(Z_train)}, dtype={Z_train.dtype}", flush=True)

    # ------------------------------------------------------------------
    # Step 3: Build eval set (same 6906 seed-out crops with final labels)
    # ------------------------------------------------------------------
    print("\n[Step 3] Building eval set...", flush=True)
    crops: List[str] = [
        it["crop"] for it in vid_idx_data
        if it["crop"] not in seed_set and it["crop"] in final
    ]
    assert all(c in EMB for c in crops), "Some eval crops missing from DINOv2 index!"
    Z_eval = np.stack([EMB[c] for c in crops])
    n = len(crops)
    print(f"  Eval set: n={n}, dtype={Z_eval.dtype}", flush=True)
    assert n == 6906, f"Expected 6906 eval crops, got {n}"

    # ------------------------------------------------------------------
    # Step 4: Fit LR (identical hyperparams)
    # ------------------------------------------------------------------
    print("\n[Step 4] Fitting LogisticRegression (DINOv2, float64)...", flush=True)
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ).fit(Z_train, y_tr)
    classes: List[str] = list(clf.classes_)
    n_classes = len(classes)
    print(f"  n_classes={n_classes}, classes={sorted(classes)}", flush=True)

    # ------------------------------------------------------------------
    # Step 5: Single predict_proba call (float64) — all metrics derived from P
    # ------------------------------------------------------------------
    print("\n[Step 5] predict_proba (single call)...", flush=True)
    P = clf.predict_proba(Z_eval)   # (n, n_classes), float64
    print(f"  P dtype: {P.dtype}", flush=True)

    pred: List[str] = [classes[int(i)] for i in P.argmax(axis=1)]
    true: List[str] = [final[c] for c in crops]

    # Uncertainty scores
    conf   = P.max(axis=1).astype(np.float64)
    ent    = -(P * np.log(P + 1e-9)).sum(axis=1).astype(np.float64)
    part   = np.sort(P, axis=1)
    margin = (part[:, -1] - part[:, -2]).astype(np.float64)

    # Masks
    rare_mask = np.array(
        [t not in COMMON and t != "誤検出(動物なし)" for t in true], dtype=bool
    )
    # True-rare mask: same definition as rau_true_rare_n173
    # (excl. 不明, same as unknown_exclusion_sensitivity in canonical)
    true_rare_mask = np.array(
        [t not in COMMON and t not in ("誤検出(動物なし)", "不明") for t in true], dtype=bool
    )
    base_correct = np.array([pred[i] == true[i] for i in range(n)], dtype=bool)
    base_acc = float(base_correct.mean())

    n_rare = int(rare_mask.sum())
    n_true_rare = int(true_rare_mask.sum())
    print(f"  n_rare={n_rare}, n_true_rare={n_true_rare}, base_acc={base_acc:.4f}", flush=True)
    print(f"  Band [0.9,1.01) n={int(((conf >= 0.9) & (conf < 1.01)).sum())}", flush=True)
    print(f"  Step 1-5 done in {time.time()-t0:.1f}s", flush=True)

    result["data_info"] = {
        "backbone": "DINOv2 (dinov2_vitb14, 768-dim)",
        "n_eval_seed_out": n,
        "n_rare_seedout": n_rare,
        "n_true_rare_seedout": n_true_rare,
        "n_train": len(Z_train),
        "n_classes": n_classes,
        "classes": sorted(classes),
        "base_accuracy": round(base_acc, 4),
        "embedding_dim": 768,
        "embedding_dtype": "float64",
        "rare_def": "true not in COMMON and not '誤検出(動物なし)'",
        "true_rare_def": "true not in COMMON and not '誤検出(動物なし)' and not '不明'",
    }

    # ------------------------------------------------------------------
    # Step 6: Deterministic orderings
    # ------------------------------------------------------------------
    print("\n[Step 6] Deterministic orderings (conf/ent/margin)...", flush=True)
    orders_det: Dict[str, np.ndarray] = {
        "confidence": np.argsort(conf),       # least-confident first
        "entropy":    np.argsort(-ent),        # highest-entropy first
        "margin":     np.argsort(margin),      # smallest-margin first
    }

    # Coreset
    print("[Step 6a] Coreset ordering...", flush=True)
    t0 = time.time()
    orders_det["coreset"] = coreset_order(Z_train, Z_eval)
    print(f"  Coreset done in {time.time()-t0:.1f}s", flush=True)

    # BADGE-approx (PCA-20, same as canonical)
    print("[Step 6b] BADGE-approx ordering (PCA-20 gradient embedding)...", flush=True)
    t0 = time.time()
    rng_badge = np.random.default_rng(SEED_BADGE)
    orders_det["badge_approx"] = badge_order(Z_eval, P, rng_badge, pca_components=20)
    print(f"  BADGE done in {time.time()-t0:.1f}s", flush=True)

    # RCTA-binary: rare-class prior targeting (binary weights)
    # score_i = sum_{c: c is true-rare class in training} p_c(i)
    # true-rare training classes: in clf.classes_, not in COMMON, not '誤検出(動物なし)', not '不明'
    print("[Step 6c] RCTA-binary ordering...", flush=True)
    true_rare_cls_idx = [
        i for i, c in enumerate(classes)
        if c not in COMMON and c not in ("誤検出(動物なし)", "不明")
    ]
    rcta_score = P[:, true_rare_cls_idx].sum(axis=1)  # higher = more likely rare
    orders_det["rcta_binary"] = np.argsort(-rcta_score)  # highest score first
    print(f"  RCTA-binary: {len(true_rare_cls_idx)} true-rare classes in training", flush=True)

    # ------------------------------------------------------------------
    # Step 7: AUC for all orderings
    # ------------------------------------------------------------------
    print("\n[Step 7] Computing AUCs...", flush=True)
    result_orders: Dict[str, Dict] = {}
    for name, idx in orders_det.items():
        accs, auc_acc, rares, auc_rare = compute_auc_from_order(
            idx, base_correct, n, BUDGETS, rare_mask
        )
        _, _, tr_rares, auc_tr_rare = compute_auc_from_order(
            idx, base_correct, n, BUDGETS, true_rare_mask
        )
        result_orders[name] = {
            "acc_by_budget": accs,
            "auc_acc_budget": auc_acc,
            "rare_recall_by_budget": rares,
            "rare_recall_auc": auc_rare,
            "true_rare_recall_by_budget": tr_rares,
            "true_rare_recall_auc": auc_tr_rare,
        }
        print(f"  {name:<20}  AUC_acc={auc_acc}  AUC_rare={auc_rare}  "
              f"AUC_true_rare={auc_tr_rare}", flush=True)

    conf_auc = result_orders["confidence"]["auc_acc_budget"]

    # ------------------------------------------------------------------
    # Step 8: 10k permutation test (confidence vs random)
    # ------------------------------------------------------------------
    print("\n[Step 8] 10k permutation test...", flush=True)
    t0 = time.time()
    rng_perm = np.random.default_rng(SEED_PERM)
    rand_aucs = fast_rand_aucs(N_PERM, base_correct, BUDGETS, n, rng_perm, batch=500)
    n_exceed = int(np.sum(rand_aucs >= conf_auc))
    pval = (n_exceed + 1) / (N_PERM + 1)   # Phipson & Smyth 2010
    rand_mean = round(float(rand_aucs.mean()), 4)
    rand_std = round(float(rand_aucs.std()), 4)
    print(f"  10k perm done in {time.time()-t0:.1f}s  "
          f"rand={rand_mean}±{rand_std}  n_exceed={n_exceed}  p={pval:.6f}", flush=True)

    result_orders["random"] = {
        "auc_acc_budget": rand_mean,
        "auc_acc_budget_std": rand_std,
        "n_shuffles": N_PERM,
    }

    result["confidence_vs_random_perm"] = {
        "n_shuffles": N_PERM,
        "n_exceed": n_exceed,
        "p_value": round(pval, 6),
        "formula": "(n_exceed+1)/(N_PERM+1) [Phipson & Smyth 2010]",
    }

    # ------------------------------------------------------------------
    # Step 9: Bootstrap CI (confidence vs Coreset, confidence vs BADGE)
    # ------------------------------------------------------------------
    print("\n[Step 9] Bootstrap CI (conf vs Coreset/BADGE, n_boot=500)...", flush=True)
    rng_boot_cs = np.random.default_rng(SEED_BOOT + 100)
    rng_boot_bg = np.random.default_rng(SEED_BOOT + 200)
    ci_vs_coreset = bootstrap_auc_gap(
        orders_det["confidence"], orders_det["coreset"],
        base_correct, BUDGETS, n, N_BOOT, rng_boot_cs
    )
    ci_vs_badge = bootstrap_auc_gap(
        orders_det["confidence"], orders_det["badge_approx"],
        base_correct, BUDGETS, n, N_BOOT, rng_boot_bg
    )
    print(f"  conf - coreset: {ci_vs_coreset}", flush=True)
    print(f"  conf - badge:   {ci_vs_badge}", flush=True)

    result["bootstrap_auc_gap_confidence_minus"] = {
        "coreset": ci_vs_coreset,
        "badge_approx": ci_vs_badge,
    }

    # Also bootstrap for RCTA-binary vs confidence (true_rare AUC)
    rng_boot_rcta = np.random.default_rng(SEED_BOOT + 300)
    # RCTA-binary vs confidence for true_rare recall AUC
    rank_rcta = np.empty(n, dtype=np.int64)
    rank_rcta[orders_det["rcta_binary"]] = np.arange(n)
    rank_conf = np.empty(n, dtype=np.int64)
    rank_conf[orders_det["confidence"]] = np.arange(n)

    xs = np.array([b / 100.0 for b in BUDGETS])
    rcta_gaps = []
    for _ in range(N_BOOT):
        idx = rng_boot_rcta.integers(0, n, n)
        n_bs = len(idx)
        bc_sub = base_correct[idx]
        tr_sub = true_rare_mask[idx]

        pos_rcta = np.argsort(rank_rcta[idx])
        pos_conf  = np.argsort(rank_conf[idx])

        def auc_tr_bs(positions: np.ndarray) -> float:
            rares_bs = []
            for B in BUDGETS:
                K = int(round(B / 100 * n_bs))
                reviewed_pos = set(positions[:K].tolist())
                correct_bs = bc_sub.copy()
                for pos in reviewed_pos:
                    correct_bs[pos] = True
                if tr_sub.any():
                    rares_bs.append(float(correct_bs[tr_sub].mean()))
                else:
                    rares_bs.append(float("nan"))
            return _trapz(np.array(rares_bs), xs)

        rcta_gaps.append(auc_tr_bs(pos_rcta) - auc_tr_bs(pos_conf))

    lo_r, hi_r = np.percentile(rcta_gaps, [2.5, 97.5])
    ci_rcta_vs_conf_true_rare = {
        "mean_gap": round(float(np.mean(rcta_gaps)), 4),
        "ci95": [round(float(lo_r), 4), round(float(hi_r), 4)],
        "n_bootstrap": N_BOOT,
        "note": "Bootstrap CI on true_rare_recall_AUC: RCTA_binary − confidence. Positive = RCTA wins.",
    }
    result["bootstrap_rcta_binary_minus_confidence_true_rare"] = ci_rcta_vs_conf_true_rare
    print(f"  RCTA-binary vs conf (true_rare AUC): {ci_rcta_vs_conf_true_rare}", flush=True)

    # ------------------------------------------------------------------
    # Final assembly
    # ------------------------------------------------------------------
    result["orders"] = result_orders
    result["budgets_pct"] = BUDGETS
    result["note"] = (
        "DINOv2 (768-dim) replication of pipeline_canonical.py order comparison. "
        "Protocol: identical LR config, same training/eval split, same seeds. "
        "RCTA-binary uses LR softmax probs summed over true-rare classes (no '不明'). "
        "All orderings: least-confident/highest-ent/smallest-margin/coreset-first/"
        "badge-first/rcta-high-score-first reviewed first. "
        "Coreset = k-center greedy on DINOv2 768-dim embedding; "
        "BADGE_approx = k-center greedy on LR gradient embedding (PCA-20 projected, "
        "K=19 classes → ~380-dim). "
        "p-value: (n_exceed+1)/(N_PERM+1) [Phipson & Smyth 2010]."
    )

    # Environment log
    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HOME, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unavailable"

    import sklearn
    result["env_log"] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "git_commit": git_commit,
        "script": __file__,
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed_badge": SEED_BADGE,
        "seed_perm": SEED_PERM,
        "seed_boot": SEED_BOOT,
        "data_hashes": {
            "dinov2_embeddings.npy": file_sha256(os.path.join(OUT, "dinov2_embeddings.npy")),
            "dinov2_embed_index.json": file_sha256(os.path.join(OUT, "dinov2_embed_index.json")),
            "seed_labels.json": file_sha256(os.path.join(OUT, "seed_labels.json")),
            "labels_video_v1.json": file_sha256(os.path.join(OUT, "labels_video_v1.json")),
            "labels_still_v1.json": file_sha256(os.path.join(OUT, "labels_still_v1.json")),
            "seed_crops.json": file_sha256(os.path.join(OUT, "seed_crops.json")),
        },
        "elapsed_total_s": round(time.time() - t_global, 1),
    }
    result["_meta"] = {
        "status": "complete",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t_global, 1),
        "script": __file__,
        "canonical_reference": os.path.join(BMK, "pipeline_canonical.json"),
        "description": (
            "E1 DINOv2 backbone robustness: replicates canonical 6-method order comparison "
            "using DINOv2 (768-dim) instead of BioCLIP (512-dim). "
            "Adds RCTA-binary true-rare targeting."
        ),
    }

    # Write output
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total = time.time() - t_global
    print("\n" + "=" * 70, flush=True)
    print(f"DONE in {total:.1f}s", flush=True)
    print(f"Output: {RESULT_PATH}", flush=True)
    print("=" * 70, flush=True)

    # Summary
    print("\n=== DINOv2 ORDER COMPARISON SUMMARY ===", flush=True)
    print(f"n_eval={n}  n_rare={n_rare}  n_true_rare={n_true_rare}  "
          f"base_acc={base_acc:.4f}", flush=True)
    print("\nMethod               AUC_acc  AUC_rare  AUC_true_rare", flush=True)
    print("-" * 55, flush=True)
    for name, d in result_orders.items():
        if name == "random":
            print(f"{'random':<20} {d['auc_acc_budget']}±{d['auc_acc_budget_std']}", flush=True)
        else:
            print(f"{name:<20} {d['auc_acc_budget']:<8} "
                  f"{d.get('rare_recall_auc', 'N/A'):<9} "
                  f"{d.get('true_rare_recall_auc', 'N/A')}", flush=True)
    perm_r = result["confidence_vs_random_perm"]
    print(f"\nPermutation test: n_exceed={perm_r['n_exceed']}/{N_PERM} → p={perm_r['p_value']:.6f}", flush=True)
    print(f"Bootstrap CI (conf-coreset): {result['bootstrap_auc_gap_confidence_minus']['coreset']['ci95']}", flush=True)
    print(f"Bootstrap CI (conf-badge):   {result['bootstrap_auc_gap_confidence_minus']['badge_approx']['ci95']}", flush=True)
    print(f"Bootstrap CI (rcta-conf, true_rare): {ci_rcta_vs_conf_true_rare['ci95']}", flush=True)


if __name__ == "__main__":
    main()
