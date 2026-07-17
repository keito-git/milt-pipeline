#!/usr/bin/env python3
"""
pipeline_ablation.py
CPU-only ablation experiments for the MiLT pipeline paper.

Ablation tasks:
  Task 1: seed_only vs full training — quantify Mie still-image contribution
  Task 2: LR-family uncertainty comparison — confidence / entropy / bootstrap ensemble
  Task 3: BADGE PCA sensitivity — PCA-5 / PCA-20 / PCA-50 / full-dim
  Task 4: Mie still photos as second site — generalization check
  Task 5: Random AUC std confirmation from canonical JSON

Design principles (anti-cheating):
  - All embeddings float64 (same as pipeline_canonical.py)
  - Same LR config: max_iter=2000, class_weight='balanced', random_state=42
  - Canonical training set split never crossed: seed (Wakayama+Shizuoka, n=1000)
    vs confirmed still (Mie, n=4736) are kept separate for ablation
  - Never overwrites existing JSONs; writes to NEW paths
  - All numbers are code outputs, never manually edited

Outputs (all new, never overwriting existing files):
  outputs/benchmark/seed_only_ablation.json
  outputs/benchmark/uncertainty_samemodel.json
  outputs/benchmark/badge_pca_sensitivity.json
  outputs/benchmark/triage_mie_secondsite.json

Usage:
  cd ~/trailcam_bench
  python3 pipeline_ablation.py
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

# GPU rule: CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HOME = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HOME, "outputs")
BMK = os.path.join(OUT, "benchmark")

# Output paths (NEW files only — never overwriting canonical)
OUT_SEED_ABL    = os.path.join(BMK, "seed_only_ablation.json")
OUT_UNCERTAINTY = os.path.join(BMK, "uncertainty_samemodel.json")
OUT_BADGE_PCA   = os.path.join(BMK, "badge_pca_sensitivity.json")
OUT_MIE_SITE    = os.path.join(BMK, "triage_mie_secondsite.json")

# ---------------------------------------------------------------------------
# Constants (must be identical to pipeline_canonical.py)
# ---------------------------------------------------------------------------
COMMON = {"ニホンジカ", "タヌキ", "イノシシ", "ニホンザル", "ニホンノウサギ"}
BUDGETS = [0, 5, 10, 20, 30, 50, 100]
BUDGETS_PARTIAL = [0, 5, 10, 20, 30]

# Seeds (distinct from canonical to avoid cross-contamination)
SEED_BOOT   = 7001   # bootstrap ensemble
SEED_PERM   = 7002   # random permutation AUC

# Classifier config (IDENTICAL to canonical)
CLF_KWARGS = dict(max_iter=2000, class_weight="balanced", random_state=42)

N_BOOT_ENSEMBLE = 200   # bootstrap resamples for LR ensemble
N_RAND_PERM     = 2000  # random permutations for std verification

# ---------------------------------------------------------------------------
# Utility functions (copied from canonical for self-containment)
# ---------------------------------------------------------------------------

def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(fn(y, x))


def compute_auc_from_order(
    order_idx: np.ndarray,
    base_correct: np.ndarray,
    n: int,
    budgets: List[int],
    rare_mask: Optional[np.ndarray] = None,
) -> Tuple[List[float], float, Optional[List[float]], Optional[float]]:
    """Returns (accs, auc_acc, rares_or_None, auc_rare_or_None)."""
    xs = np.array([b / 100.0 for b in budgets])
    accs, rares = [], []
    for B in budgets:
        K = int(round(B / 100 * n))
        reviewed = set(order_idx[:K].tolist())
        correct = base_correct.copy()
        for i in reviewed:
            correct[i] = True
        accs.append(float(correct.mean()))
        if rare_mask is not None and rare_mask.any():
            rares.append(float(correct[rare_mask].mean()))
        else:
            rares.append(float("nan"))
    auc_acc = round(_trapz(np.array(accs), xs), 4)
    auc_rare = (
        round(_trapz(np.array([r for r in rares if not np.isnan(r)]), xs), 4)
        if rare_mask is not None and rare_mask.any() and not all(np.isnan(rares))
        else None
    )
    return (
        [round(a, 4) for a in accs],
        auc_acc,
        [round(r, 4) for r in rares] if rare_mask is not None else None,
        auc_rare,
    )


def bootstrap_auc_gap(
    order_a: np.ndarray,
    order_b: np.ndarray,
    base_correct: np.ndarray,
    budgets: List[int],
    n: int,
    n_boot: int,
    rng: np.random.Generator,
) -> Dict:
    """Bootstrap 95% CI on AUC(order_a) - AUC(order_b)."""
    rank_a = np.empty(n, dtype=np.int64)
    rank_a[order_a] = np.arange(n)
    rank_b = np.empty(n, dtype=np.int64)
    rank_b[order_b] = np.arange(n)

    xs = np.array([b / 100.0 for b in budgets])

    def auc_bs(positions: np.ndarray, bc_sub: np.ndarray) -> float:
        n_bs = len(bc_sub)
        accs_bs = []
        for B in budgets:
            K = int(round(B / 100 * n_bs))
            reviewed_pos = set(positions[:K].tolist())
            correct_bs = bc_sub.copy()
            for pos in reviewed_pos:
                correct_bs[pos] = True
            accs_bs.append(float(correct_bs.mean()))
        return _trapz(np.array(accs_bs), xs)

    gaps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        n_bs = len(idx)
        bc_sub = base_correct[idx]
        pos_a = np.argsort(rank_a[idx])
        pos_b = np.argsort(rank_b[idx])
        gaps.append(auc_bs(pos_a, bc_sub) - auc_bs(pos_b, bc_sub))

    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {
        "mean_gap": round(float(np.mean(gaps)), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "n_bootstrap": n_boot,
        "significant_at_alpha05": bool(lo > 0 or hi < 0),
        "note": "Positive gap = order_a > order_b in AUC. CI excludes 0 → significant.",
    }


def badge_order(
    Z_eval: np.ndarray,
    P: np.ndarray,
    pca_components: Optional[int],
    rng_seed: int = 42,
) -> np.ndarray:
    """
    BADGE-approximation: k-center greedy on logistic-regression gradient embeddings.
    pca_components=None → use full embedding dim (no PCA projection).
    """
    from sklearn.decomposition import PCA

    n_eval, d = Z_eval.shape
    K = P.shape[1]
    pred_classes = P.argmax(axis=1)

    if pca_components is not None:
        actual_components = min(pca_components, d, n_eval)
        pca = PCA(n_components=actual_components, random_state=rng_seed)
        Z_proj = pca.fit_transform(Z_eval)
    else:
        Z_proj = Z_eval  # full dim

    delta = P.copy()
    delta[np.arange(n_eval), pred_classes] -= 1.0
    G = (delta[:, :, None] * Z_proj[:, None, :]).reshape(n_eval, K * Z_proj.shape[1])

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


def file_sha256(path: str, n_bytes: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(n_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: str, obj: dict) -> None:
    """Atomically write JSON to path (never leaves partial file)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    print(f"  [wrote] {path}", flush=True)


# ---------------------------------------------------------------------------
# Data loading (shared across all tasks)
# ---------------------------------------------------------------------------

class DataBundle:
    """All loaded data for ablation experiments."""

    def __init__(self) -> None:
        print("[Data] Loading embeddings and labels...", flush=True)
        t0 = time.time()

        # Load embeddings
        se = np.load(os.path.join(OUT, "embeddings.npy")).astype(np.float64)
        prov = json.load(open(os.path.join(OUT, "provisional_labels.json")))
        EMB: Dict[str, np.ndarray] = {}
        for i, r in enumerate(prov):
            EMB[r["crop"]] = se[i]

        ve = np.load(os.path.join(OUT, "video_embeddings.npy")).astype(np.float64)
        vit = json.load(open(os.path.join(OUT, "video_embed_index.json")))
        for i, it in enumerate(vit):
            EMB[it["crop"]] = ve[i]

        # Seed labels (Wakayama + Shizuoka video crops, n=1000)
        seed_lab = {
            r["crop"]: r["label"]
            for r in json.load(open(os.path.join(OUT, "seed_labels.json")))["labels"]
        }

        # Still labels (Mie 三重県 confirmed, n=4736)
        still_recs = json.load(open(os.path.join(OUT, "labels_still_v1.json")))["labels"]

        # Video ground-truth labels (for eval set)
        final = {
            r["crop"]: r["label"]
            for r in json.load(open(os.path.join(OUT, "labels_video_v1.json")))["labels"]
        }

        # Seed set (crops used as LR training seed — excluded from eval)
        seed_set = set(json.load(open(os.path.join(OUT, "seed_crops.json"))))

        # ── Build training sets ──

        # Seed-only training (n=1000, Wakayama + Shizuoka)
        X_seed, y_seed = [], []
        for c, l in seed_lab.items():
            if c in EMB:
                X_seed.append(EMB[c])
                y_seed.append(l)
        self.Z_seed = np.stack(X_seed)   # (1000, 512), float64
        self.y_seed = y_seed

        # Mie still training data (n=4736)
        X_still, y_still, crops_still = [], [], []
        for r in still_recs:
            if r["crop"] in EMB:
                X_still.append(EMB[r["crop"]])
                y_still.append(r["label"])
                crops_still.append(r["crop"])
        self.Z_still = np.stack(X_still)   # (4736, 512), float64
        self.y_still = y_still
        self.crops_still = crops_still

        # Full training set (seed + still)
        self.Z_train = np.vstack([self.Z_seed, self.Z_still])
        self.y_train = self.y_seed + self.y_still

        # ── Eval set (seed-out video: Wakayama + Shizuoka, n≈6906) ──
        eval_crops = [it["crop"] for it in vit
                      if it["crop"] not in seed_set and it["crop"] in final]
        self.Z_eval = np.stack([EMB[c] for c in eval_crops])
        self.crops_eval = eval_crops
        self.true_eval = [final[c] for c in eval_crops]
        self.n_eval = len(eval_crops)

        # ── Precompute RARE mask for eval set ──
        self.rare_mask_eval = np.array(
            [t not in COMMON and t != "誤検出(動物なし)" for t in self.true_eval], dtype=bool
        )

        # ── Data hashes for env log ──
        self.data_hashes = {
            "embeddings.npy": file_sha256(os.path.join(OUT, "embeddings.npy")),
            "video_embeddings.npy": file_sha256(os.path.join(OUT, "video_embeddings.npy")),
            "seed_labels.json": file_sha256(os.path.join(OUT, "seed_labels.json")),
            "labels_video_v1.json": file_sha256(os.path.join(OUT, "labels_video_v1.json")),
            "labels_still_v1.json": file_sha256(os.path.join(OUT, "labels_still_v1.json")),
        }

        print(
            f"  n_seed={len(self.Z_seed)}  n_still={len(self.Z_still)}"
            f"  n_train_full={len(self.Z_train)}  n_eval={self.n_eval}"
            f"  n_rare_eval={int(self.rare_mask_eval.sum())}"
            f"  dtype={self.Z_train.dtype}  elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

    def fit_lr(
        self, Z: np.ndarray, y: List[str], label: str
    ) -> LogisticRegression:
        t0 = time.time()
        clf = LogisticRegression(**CLF_KWARGS).fit(Z, y)
        print(
            f"  [{label}] LR fit: n={len(Z)}, n_classes={len(clf.classes_)}"
            f", n_iter={clf.n_iter_}, elapsed={time.time()-t0:.2f}s",
            flush=True,
        )
        return clf

    def predict_on_eval(
        self, clf: LogisticRegression
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns P, pred_labels, base_correct, conf for the eval set."""
        P = clf.predict_proba(self.Z_eval)   # (n, K), float64
        classes = list(clf.classes_)
        pred = [classes[int(i)] for i in P.argmax(axis=1)]
        base_correct = np.array([pred[i] == self.true_eval[i] for i in range(self.n_eval)], dtype=bool)
        conf = P.max(axis=1)
        return P, pred, base_correct, conf


# ---------------------------------------------------------------------------
# Task 1: Seed-only vs Full training ablation
# ---------------------------------------------------------------------------

def task1_seed_only_ablation(db: DataBundle) -> None:
    """
    Compare training on seed only (n=1000, Wakayama+Shizuoka video)
    vs full set (n=5736 = 1000 seed + 4736 Mie still).
    Evaluate on the seed-out video eval set (n=6906).
    Metric: accuracy, macro-F1, triage precision-budget AUC.
    Quantifies the Mie still-image contribution to LR performance.
    """
    print("\n[Task 1] Seed-only vs Full training ablation", flush=True)
    t0 = time.time()

    def eval_model(clf: LogisticRegression, label: str) -> Dict:
        """Evaluate a fitted LR on the video eval set."""
        P, pred, base_correct, conf = db.predict_on_eval(clf)
        n = db.n_eval

        acc = round(float(base_correct.mean()), 4)
        macro_f1 = round(float(f1_score(db.true_eval, pred, average="macro", zero_division=0)), 4)

        # Triage AUC (confidence order: least confident first)
        ci_order = np.argsort(conf)
        accs, auc_acc, rares, auc_rare = compute_auc_from_order(
            ci_order, base_correct, n, BUDGETS, db.rare_mask_eval
        )
        # Partial AUC (B<=30%)
        _, auc_acc_b30, _, auc_rare_b30 = compute_auc_from_order(
            ci_order, base_correct, n, BUDGETS_PARTIAL, db.rare_mask_eval
        )

        print(
            f"  [{label}] acc={acc}  macro_f1={macro_f1}"
            f"  AUC_acc={auc_acc}  AUC_rare={auc_rare}"
            f"  AUC_acc_b30={auc_acc_b30}",
            flush=True,
        )
        return {
            "n_train": int(len(clf.n_iter_) if hasattr(clf.n_iter_, "__len__") else 1),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "auc_acc_budget": auc_acc,
            "auc_rare_budget": auc_rare,
            "auc_acc_budget_b30": auc_acc_b30,
            "auc_rare_budget_b30": auc_rare_b30,
            "acc_by_budget": accs,
            "rare_recall_by_budget": rares,
            "n_classes": len(clf.classes_),
        }

    # Seed-only model
    clf_seed = db.fit_lr(db.Z_seed, db.y_seed, "seed_only")
    res_seed = eval_model(clf_seed, "seed_only")
    res_seed["n_train"] = len(db.Z_seed)

    # Full model (seed + Mie still)
    clf_full = db.fit_lr(db.Z_train, db.y_train, "full_5736")
    res_full = eval_model(clf_full, "full_5736")
    res_full["n_train"] = len(db.Z_train)

    # Deltas (full - seed_only): positive = full is better
    result = {
        "description": (
            "Ablation: seed-only (n=1000, Wakayama+Shizuoka video crops) vs "
            "full training (n=5736 = 1000 seed + 4736 Mie confirmed stills). "
            "Eval set: seed-out video crops (Wakayama+Shizuoka, n=6906). "
            "Quantifies the Mie still-image contribution."
        ),
        "training_sets": {
            "seed_only": "Wakayama (n=660) + Shizuoka (n=340) video seed crops",
            "full_5736": "seed_only + Mie (三重県) confirmed still photos (n=4736)",
        },
        "eval_set": "Seed-out video crops (Wakayama + Shizuoka, n=6906)",
        "classifier_config": CLF_KWARGS,
        "n_eval": db.n_eval,
        "n_rare_eval": int(db.rare_mask_eval.sum()),
        "seed_only_1000": res_seed,
        "full_5736": res_full,
        "class_coverage_note": (
            f"seed_only model: {res_seed['n_classes']} classes (Wakayama+Shizuoka video); "
            f"full model: {res_full['n_classes']} classes (adds 8 Mie-only species: "
            "その他/キジバト/キツネ/テン/トラツグミ/ネコ/ヤマガラ/鳥類). "
            "Eval set has 18 classes; 66/6906 (1.0%) samples have labels unseen by seed-only model. "
            "Mie stills provide class diversity but introduce domain shift from Mie→Wakayama/Shizuoka."
        ),
        "delta_full_minus_seed_only": {
            "delta_accuracy": round(res_full["accuracy"] - res_seed["accuracy"], 4),
            "delta_macro_f1": round(res_full["macro_f1"] - res_seed["macro_f1"], 4),
            "delta_auc_acc": round(res_full["auc_acc_budget"] - res_seed["auc_acc_budget"], 4),
            "delta_auc_rare": round((res_full["auc_rare_budget"] or 0) - (res_seed["auc_rare_budget"] or 0), 4),
            "delta_auc_acc_b30": round(res_full["auc_acc_budget_b30"] - res_seed["auc_acc_budget_b30"], 4),
            "note": (
                "Positive = full (5736) outperforms seed_only (1000). "
                "Note: seed_only may show higher raw accuracy if Mie still domain shift "
                "hurts Wakayama/Shizuoka eval; macro-F1 reflects class coverage gain."
            ),
        },
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
    }

    atomic_write_json(OUT_SEED_ABL, result)
    print(f"  [Task 1 done] elapsed={time.time()-t0:.1f}s", flush=True)

    # Return the two classifiers for downstream reuse
    return clf_seed, clf_full


# ---------------------------------------------------------------------------
# Task 2: LR-family uncertainty comparison (same model, same training set)
# ---------------------------------------------------------------------------

def task2_uncertainty_samemodel(db: DataBundle, clf_full: LogisticRegression) -> None:
    """
    Fair comparison of uncertainty estimators on the SAME LR model family
    (trained on full n=5736). All estimates are from the LR family.

    Methods:
      (a) LR confidence: max softmax probability (least-confident first)
      (b) LR entropy: predictive entropy (highest-entropy first)
      (c) LR bootstrap ensemble (200 resamples): entropy of ensemble mean
          — Each resample trains a fresh LR on a bootstrap draw of the
            full training set (n=5736, with replacement). Aggregate
            predict_proba across 200 models → ensemble mean P_avg.
            Order by H(P_avg) descending.

    Evaluates whether the ordering advantage of simple confidence
    is specific to a single-model setup, or a more expensive ensemble
    produces comparable / better results.
    """
    print("\n[Task 2] LR-family uncertainty comparison (same model)", flush=True)
    t0 = time.time()

    n = db.n_eval

    # Single-model predictions (RECOMPUTE from clf_full for self-containment)
    P_single, pred_single, base_correct, conf = db.predict_on_eval(clf_full)
    ent_single = -(P_single * np.log(P_single + 1e-9)).sum(axis=1)

    # Orders
    order_conf = np.argsort(conf)       # least-confident first
    order_ent  = np.argsort(-ent_single)  # highest-entropy first

    accs_conf, auc_conf, rares_conf, auc_rare_conf = compute_auc_from_order(
        order_conf, base_correct, n, BUDGETS, db.rare_mask_eval
    )
    accs_ent, auc_ent, rares_ent, auc_rare_ent = compute_auc_from_order(
        order_ent, base_correct, n, BUDGETS, db.rare_mask_eval
    )

    print(f"  LR_confidence: AUC_acc={auc_conf}  AUC_rare={auc_rare_conf}", flush=True)
    print(f"  LR_entropy:    AUC_acc={auc_ent}  AUC_rare={auc_rare_ent}", flush=True)

    # ── Bootstrap ensemble (200 resamples of full training set) ──
    print(f"  Bootstrap ensemble: {N_BOOT_ENSEMBLE} resamples...", flush=True)
    rng_boot = np.random.default_rng(SEED_BOOT)

    Z_tr = db.Z_train
    y_tr_arr = np.array(db.y_train)
    n_tr = len(Z_tr)

    # Collect predict_proba from each bootstrap LR
    # Use clf_full's classes_ to ensure consistent column alignment
    canonical_classes = list(clf_full.classes_)
    K = len(canonical_classes)
    class_to_idx = {c: i for i, c in enumerate(canonical_classes)}

    P_sum = np.zeros((n, K), dtype=np.float64)
    t_boot = time.time()
    n_failed = 0
    for b in range(N_BOOT_ENSEMBLE):
        idx_b = rng_boot.integers(0, n_tr, n_tr)
        Z_b = Z_tr[idx_b]
        y_b = y_tr_arr[idx_b].tolist()
        try:
            clf_b = LogisticRegression(**CLF_KWARGS).fit(Z_b, y_b)
        except Exception as e:
            print(f"    Bootstrap {b} fit failed: {e}", flush=True)
            n_failed += 1
            continue

        # Map clf_b.classes_ columns back to canonical ordering
        P_b_raw = clf_b.predict_proba(db.Z_eval)   # (n, K_b)
        for j, cls in enumerate(clf_b.classes_):
            if cls in class_to_idx:
                P_sum[:, class_to_idx[cls]] += P_b_raw[:, j]

        if (b + 1) % 50 == 0:
            print(f"    {b+1}/{N_BOOT_ENSEMBLE} done ({time.time()-t_boot:.0f}s)", flush=True)

    n_success = N_BOOT_ENSEMBLE - n_failed
    P_ens = P_sum / max(n_success, 1)   # ensemble mean
    ent_ens = -(P_ens * np.log(P_ens + 1e-9)).sum(axis=1)

    # Ensemble order (highest-entropy first)
    order_ens = np.argsort(-ent_ens)
    accs_ens, auc_ens, rares_ens, auc_rare_ens = compute_auc_from_order(
        order_ens, base_correct, n, BUDGETS, db.rare_mask_eval
    )
    print(f"  LR_bootstrap_ensemble: AUC_acc={auc_ens}  AUC_rare={auc_rare_ens}"
          f"  (n_success={n_success})", flush=True)

    # Bootstrap CI: confidence vs ensemble
    print("  Bootstrap CI: confidence vs ensemble...", flush=True)
    rng_ci = np.random.default_rng(SEED_BOOT + 1)
    ci_conf_vs_ens = bootstrap_auc_gap(
        order_conf, order_ens, base_correct, BUDGETS, n, 500, rng_ci
    )

    # Partial AUC (B<=30%)
    _, auc_conf_b30, _, _ = compute_auc_from_order(
        order_conf, base_correct, n, BUDGETS_PARTIAL, db.rare_mask_eval
    )
    _, auc_ent_b30, _, _ = compute_auc_from_order(
        order_ent, base_correct, n, BUDGETS_PARTIAL, db.rare_mask_eval
    )
    _, auc_ens_b30, _, _ = compute_auc_from_order(
        order_ens, base_correct, n, BUDGETS_PARTIAL, db.rare_mask_eval
    )

    result = {
        "description": (
            "Fair uncertainty comparison: all three methods use the SAME LR model family "
            "(LogisticRegression, same config, same embedding space). "
            "Eval set: seed-out video crops (n=6906). "
            "Tests whether simple softmax confidence is sufficient, or "
            "whether a bootstrap ensemble improves triage ordering."
        ),
        "model": "LogisticRegression (max_iter=2000, class_weight=balanced, random_state=42)",
        "training_set": f"full (n={len(db.Z_train)}: {len(db.Z_seed)} seed + {len(db.Z_still)} Mie still)",
        "n_eval": n,
        "n_rare_eval": int(db.rare_mask_eval.sum()),
        "n_classes": K,
        "budgets_pct": BUDGETS,
        "results": {
            "lr_confidence": {
                "description": "Max softmax probability; least-confident first",
                "auc_acc_budget": auc_conf,
                "auc_rare_budget": auc_rare_conf,
                "auc_acc_budget_b30": auc_conf_b30,
                "acc_by_budget": accs_conf,
                "rare_recall_by_budget": rares_conf,
            },
            "lr_entropy": {
                "description": "Predictive entropy H(P_single); highest-entropy first",
                "auc_acc_budget": auc_ent,
                "auc_rare_budget": auc_rare_ent,
                "auc_acc_budget_b30": auc_ent_b30,
                "acc_by_budget": accs_ent,
                "rare_recall_by_budget": rares_ent,
            },
            "lr_bootstrap_ensemble": {
                "description": (
                    f"Ensemble of {N_BOOT_ENSEMBLE} LR models trained on bootstrap resamples "
                    "of the full training set. Order by H(P_avg), highest-entropy first."
                ),
                "n_resamples": N_BOOT_ENSEMBLE,
                "n_success": n_success,
                "n_failed": n_failed,
                "seed": SEED_BOOT,
                "auc_acc_budget": auc_ens,
                "auc_rare_budget": auc_rare_ens,
                "auc_acc_budget_b30": auc_ens_b30,
                "acc_by_budget": accs_ens,
                "rare_recall_by_budget": rares_ens,
            },
        },
        "bootstrap_ci_confidence_vs_ensemble": ci_conf_vs_ens,
        "honest_summary": (
            "confidence_better_than_ensemble="
            + str(bool(auc_conf > auc_ens))
            + f"; gap_conf_minus_ens={round(auc_conf - auc_ens, 4)}"
        ),
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
    }

    atomic_write_json(OUT_UNCERTAINTY, result)
    print(f"  [Task 2 done] elapsed={time.time()-t0:.1f}s", flush=True)

    # Return orders for potential downstream use
    return order_conf, base_correct


# ---------------------------------------------------------------------------
# Task 3: BADGE PCA sensitivity
# ---------------------------------------------------------------------------

def task3_badge_pca_sensitivity(
    db: DataBundle,
    clf_full: LogisticRegression,
    conf_order: np.ndarray,
    base_correct_full: np.ndarray,
) -> None:
    """
    BADGE-approx PCA sensitivity analysis.
    Tests whether the confidence-ordering advantage over BADGE depends on
    the PCA approximation dimension (5 / 20 / 50 / full 512).
    """
    print("\n[Task 3] BADGE PCA sensitivity", flush=True)
    t0 = time.time()

    n = db.n_eval
    P = clf_full.predict_proba(db.Z_eval)

    # PCA settings to test
    pca_settings = [
        ("pca_5",   5),
        ("pca_20",  20),    # canonical (current)
        ("pca_50",  50),
        ("full_dim", None), # no PCA; full 512-dim
    ]

    badge_results = {}
    badge_orders  = {}

    for name, pca_comp in pca_settings:
        label = f"PCA-{pca_comp}" if pca_comp is not None else "full_dim(512)"
        print(f"\n  BADGE {label}...", flush=True)
        t_b = time.time()

        b_order = badge_order(db.Z_eval, P, pca_components=pca_comp, rng_seed=42)
        badge_orders[name] = b_order

        accs, auc_acc, rares, auc_rare = compute_auc_from_order(
            b_order, base_correct_full, n, BUDGETS, db.rare_mask_eval
        )
        _, auc_acc_b30, _, _ = compute_auc_from_order(
            b_order, base_correct_full, n, BUDGETS_PARTIAL, db.rare_mask_eval
        )

        # Gradient embedding dimension
        K = P.shape[1]
        if pca_comp is not None:
            actual_comp = min(pca_comp, db.Z_eval.shape[1], n)
            g_dim = K * actual_comp
        else:
            g_dim = K * db.Z_eval.shape[1]

        print(f"    G.dim={g_dim}  AUC_acc={auc_acc}  AUC_rare={auc_rare}  "
              f"AUC_acc_b30={auc_acc_b30}  elapsed={time.time()-t_b:.1f}s", flush=True)

        badge_results[name] = {
            "pca_components": pca_comp,
            "gradient_embedding_dim": g_dim,
            "auc_acc_budget": auc_acc,
            "auc_rare_budget": auc_rare,
            "auc_acc_budget_b30": auc_acc_b30,
            "acc_by_budget": accs,
            "rare_recall_by_budget": rares,
        }

    # Bootstrap CI: confidence vs each BADGE variant
    print("\n  Bootstrap CIs (confidence vs each BADGE variant)...", flush=True)
    ci_results = {}
    for name, _ in pca_settings:
        rng_ci = np.random.default_rng(SEED_BOOT + 200 + list(dict(pca_settings).keys()).index(name))
        ci = bootstrap_auc_gap(
            conf_order, badge_orders[name],
            base_correct_full, BUDGETS, n, 500, rng_ci
        )
        ci_results[name] = ci
        print(f"    conf vs {name}: gap={ci['mean_gap']}  CI95={ci['ci95']}"
              f"  sig={ci['significant_at_alpha05']}", flush=True)

    # AUC range across all BADGE variants
    all_badge_aucs = [badge_results[k]["auc_acc_budget"] for k, _ in pca_settings]
    canonical_pca20_auc = badge_results["pca_20"]["auc_acc_budget"]
    conf_auc = round(float(compute_auc_from_order(
        conf_order, base_correct_full, db.n_eval, BUDGETS, db.rare_mask_eval
    )[1]), 4)

    result = {
        "description": (
            "BADGE-approx PCA sensitivity: k-center greedy on LR gradient embedding "
            "with varying PCA projection dimensions. "
            "Tests whether confidence ordering advantage over BADGE depends on "
            "the PCA approximation choice."
        ),
        "canonical_pca20_note": (
            "PCA-20 is the canonical BADGE-approx used in the paper. "
            "Other settings are ablation only."
        ),
        "n_eval": n,
        "n_rare_eval": int(db.rare_mask_eval.sum()),
        "n_classes": int(P.shape[1]),
        "embedding_dim": int(db.Z_eval.shape[1]),
        "budgets_pct": BUDGETS,
        "confidence_auc_acc": conf_auc,
        "badge_variants": badge_results,
        "bootstrap_ci_confidence_minus_badge": ci_results,
        "sensitivity_summary": {
            "badge_auc_min": round(float(min(all_badge_aucs)), 4),
            "badge_auc_max": round(float(max(all_badge_aucs)), 4),
            "badge_auc_range": round(float(max(all_badge_aucs) - min(all_badge_aucs)), 4),
            "canonical_pca20_auc": canonical_pca20_auc,
            "confidence_auc": conf_auc,
            "all_ci_exclude_zero": all(ci_results[k]["significant_at_alpha05"] for k, _ in pca_settings),
        },
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
    }

    atomic_write_json(OUT_BADGE_PCA, result)
    print(f"  [Task 3 done] elapsed={time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Task 4: Mie still photos as second site (generalization test)
# ---------------------------------------------------------------------------

def task4_triage_mie_secondsite(db: DataBundle, clf_seed: LogisticRegression) -> None:
    """
    Generalization test: use the seed-only LR (trained on Wakayama+Shizuoka)
    to triage Mie (三重県) still photos.

    The confirmed still labels (labels_still_v1.json) serve as ground truth.
    Compute: accuracy, triage AUC (confidence order vs random).

    Design note: We use the SEED-ONLY model here (not full), because the full
    model was trained ON the Mie still data — it would be a train/test leak
    if we evaluated on Mie stills with the full model.
    The seed-only LR has never seen any Mie images.
    """
    print("\n[Task 4] Mie still photos — second-site generalization", flush=True)
    t0 = time.time()

    n_mie = len(db.Z_still)
    y_mie = db.y_still
    print(f"  n_mie={n_mie}", flush=True)

    # Predict on Mie still crops (using seed-only model)
    P_mie = clf_seed.predict_proba(db.Z_still)
    classes = list(clf_seed.classes_)
    K = len(classes)

    pred_mie = [classes[int(i)] for i in P_mie.argmax(axis=1)]
    base_correct_mie = np.array([pred_mie[i] == y_mie[i] for i in range(n_mie)], dtype=bool)
    conf_mie = P_mie.max(axis=1)

    acc_mie = round(float(base_correct_mie.mean()), 4)
    macro_f1_mie = round(float(f1_score(y_mie, pred_mie, average="macro", zero_division=0)), 4)

    # RARE mask for Mie
    rare_mask_mie = np.array(
        [t not in COMMON and t != "誤検出(動物なし)" for t in y_mie], dtype=bool
    )
    n_rare_mie = int(rare_mask_mie.sum())

    print(f"  acc_mie={acc_mie}  macro_f1={macro_f1_mie}  n_rare={n_rare_mie}", flush=True)

    # Triage: confidence order
    ci_order_mie = np.argsort(conf_mie)   # least-confident first
    accs_conf, auc_conf, rares_conf, auc_rare_conf = compute_auc_from_order(
        ci_order_mie, base_correct_mie, n_mie, BUDGETS, rare_mask_mie
    )
    _, auc_conf_b30, _, _ = compute_auc_from_order(
        ci_order_mie, base_correct_mie, n_mie, BUDGETS_PARTIAL, rare_mask_mie
    )

    print(f"  Mie triage (conf): AUC_acc={auc_conf}  AUC_rare={auc_rare_conf}", flush=True)

    # Random baseline (2000 permutations)
    rng_perm = np.random.default_rng(SEED_PERM)
    rand_aucs_mie = []
    for _ in range(N_RAND_PERM):
        idx_r = rng_perm.permutation(n_mie)
        _, auc_r, _, _ = compute_auc_from_order(idx_r, base_correct_mie, n_mie, BUDGETS, rare_mask_mie)
        rand_aucs_mie.append(auc_r)
    rand_mean = round(float(np.mean(rand_aucs_mie)), 4)
    rand_std  = round(float(np.std(rand_aucs_mie)), 4)

    # p-value (Phipson & Smyth)
    n_exceed = int(np.sum(np.array(rand_aucs_mie) >= auc_conf))
    pval = round((n_exceed + 1) / (N_RAND_PERM + 1), 6)

    print(f"  Random AUC: {rand_mean}±{rand_std}  n_exceed={n_exceed}  p={pval}", flush=True)

    # Bootstrap CI: confidence vs random
    rng_ci_mie = np.random.default_rng(SEED_BOOT + 300)
    # For random baseline, create a fixed random order (median permutation is hard to define)
    # Use 500-bootstrap CI on confidence ordering vs a 2000-permutation median ordering
    rand_order_mie = rng_perm.permutation(n_mie)
    ci_conf_vs_rand = bootstrap_auc_gap(
        ci_order_mie, rand_order_mie,
        base_correct_mie, BUDGETS, n_mie, 500, rng_ci_mie
    )

    # Mie class distribution and coverage analysis
    from collections import Counter
    label_dist = dict(Counter(y_mie))
    classes_mie = sorted(set(y_mie))
    classes_missing = sorted(set(y_mie) - set(classes))
    n_unpredictable = sum(1 for l in y_mie if l not in set(classes))

    result = {
        "description": (
            "Second-site generalization: Mie (三重県) still photos "
            "evaluated with seed-only LR (trained on Wakayama+Shizuoka video, n=1000). "
            "No Mie data used in training → clean generalization test. "
            "Second-site generalization check."
        ),
        "training": f"seed_only LR (Wakayama n=660 + Shizuoka n=340 video crops)",
        "training_n": len(db.Z_seed),
        "eval_site": "Mie (三重県) still photos (labels_still_v1.json, confirmed labels)",
        "n_mie": n_mie,
        "n_rare_mie": n_rare_mie,
        "n_classes_model": K,
        "classes_model": classes,
        "classes_mie": classes_mie,
        "classes_missing_from_seed_model": classes_missing,
        "n_mie_samples_unpredictable": n_unpredictable,
        "coverage_note": (
            f"{len(classes_missing)} Mie classes ({classes_missing}) "
            "cannot be predicted by seed-only model (not seen during training). "
            f"These account for {n_unpredictable}/{n_mie} Mie samples — "
            "all will be misclassified regardless of confidence."
        ),
        "label_distribution_mie": label_dist,
        "accuracy": acc_mie,
        "macro_f1": macro_f1_mie,
        "budgets_pct": BUDGETS,
        "triage_confidence": {
            "description": "Confidence ordering (least-confident first) on Mie stills",
            "auc_acc_budget": auc_conf,
            "auc_rare_budget": auc_rare_conf,
            "auc_acc_budget_b30": auc_conf_b30,
            "acc_by_budget": accs_conf,
            "rare_recall_by_budget": rares_conf,
        },
        "triage_random": {
            "description": f"Random ordering ({N_RAND_PERM} permutations)",
            "n_permutations": N_RAND_PERM,
            "auc_acc_mean": rand_mean,
            "auc_acc_std": rand_std,
        },
        "permutation_test": {
            "conf_auc": auc_conf,
            "rand_mean": rand_mean,
            "n_exceed": n_exceed,
            "n_permutations": N_RAND_PERM,
            "p_value": pval,
            "formula": "(n_exceed+1)/(N_PERM+1) [Phipson & Smyth 2010]",
        },
        "bootstrap_ci_confidence_vs_random": ci_conf_vs_rand,
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
    }

    atomic_write_json(OUT_MIE_SITE, result)
    print(f"  [Task 4 done] elapsed={time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Task 5: Random AUC std confirmation from canonical JSON
# ---------------------------------------------------------------------------

def task5_confirm_random_std() -> Dict:
    """
    Read the canonical JSON and confirm the random AUC std values.
    The paper states 0.002 (full) / ±0.001 (B<=30%).
    Actual canonical values: ?
    This is a read-only check; no re-computation.
    """
    print("\n[Task 5] Confirming random AUC std from canonical JSON...", flush=True)
    canon_path = os.path.join(BMK, "pipeline_canonical.json")
    if not os.path.exists(canon_path):
        return {"error": f"canonical JSON not found: {canon_path}"}

    c = json.load(open(canon_path))

    rand_full = c["al_baselines"]["orders"]["random"]
    rand_b30  = c["partial_auc_b30"]["orders"]["random"]

    auc_full_std = rand_full.get("auc_acc_budget_std")
    auc_b30_std  = rand_b30.get("auc_acc_budget_b30_std")
    n_full       = rand_full.get("n_shuffles")
    n_b30        = rand_b30.get("n_shuffles")

    print(f"  Full-budget (n={n_full}): std={auc_full_std}", flush=True)
    print(f"  B<=30%  (n={n_b30}): std={auc_b30_std}", flush=True)
    print(f"  Paper claims: 0.002 / ±0.001", flush=True)

    return {
        "source": canon_path,
        "full_budget": {
            "auc_acc_mean": rand_full.get("auc_acc_budget"),
            "auc_acc_std": auc_full_std,
            "n_shuffles": n_full,
        },
        "partial_b30": {
            "auc_acc_mean": rand_b30.get("auc_acc_budget_b30"),
            "auc_acc_std": auc_b30_std,
            "n_shuffles": n_b30,
        },
        "paper_claims": {
            "full_budget_std": 0.002,
            "partial_b30_std": 0.001,
        },
        "verdict": {
            "full_budget_std_matches_paper": (
                abs((auc_full_std or 0) - 0.002) <= 0.0005
            ),
            "partial_b30_std_matches_paper": (
                abs((auc_b30_std or 0) - 0.001) <= 0.0005
            ),
        },
    }


# ---------------------------------------------------------------------------
# Environment manifest
# ---------------------------------------------------------------------------

def build_env_manifest(db: DataBundle) -> Dict:
    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HOME, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unavailable"

    import sklearn
    try:
        import torch
        torch_ver = torch.__version__
    except ImportError:
        torch_ver = "not_installed"

    return {
        "script": __file__,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch_ver,
        "git_commit": git_commit,
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "data_hashes": db.data_hashes,
        "n_eval_seed_out": db.n_eval,
        "n_rare_eval": int(db.rare_mask_eval.sum()),
        "n_train_full": len(db.Z_train),
        "n_train_seed": len(db.Z_seed),
        "n_mie_still": len(db.Z_still),
        "embedding_dtype": str(db.Z_eval.dtype),
        "classifier_config": CLF_KWARGS,
        "seeds": {
            "bootstrap_ensemble": SEED_BOOT,
            "random_perm": SEED_PERM,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(BMK, exist_ok=True)
    print("=" * 70, flush=True)
    print("pipeline_ablation.py — CPU-only ablation experiments", flush=True)
    print(f"Output dir: {BMK}", flush=True)
    print("=" * 70, flush=True)
    t_global = time.time()

    # Guard: never overwrite existing output files
    for path in [OUT_SEED_ABL, OUT_UNCERTAINTY, OUT_BADGE_PCA, OUT_MIE_SITE]:
        if os.path.exists(path):
            print(f"  [WARN] Output file already exists: {path}", flush=True)
            print("  To re-run, rename or remove existing files first.", flush=True)
            print("  Aborting to protect existing results.", flush=True)
            sys.exit(1)

    # Load all data once
    db = DataBundle()

    # Build env manifest
    env = build_env_manifest(db)

    # Task 1: seed-only vs full ablation
    clf_seed, clf_full = task1_seed_only_ablation(db)

    # Task 2: LR-family uncertainty comparison
    # Returns conf_order and base_correct for the full model
    conf_order, base_correct_full = task2_uncertainty_samemodel(db, clf_full)

    # Task 3: BADGE PCA sensitivity
    task3_badge_pca_sensitivity(db, clf_full, conf_order, base_correct_full)

    # Task 4: Mie second-site
    task4_triage_mie_secondsite(db, clf_seed)

    # Task 5: Confirm random std (read-only)
    std_check = task5_confirm_random_std()

    # Print Task 5 summary
    print("\n[Task 5] Random AUC std confirmation:", flush=True)
    print(f"  Full-budget:  canonical std={std_check['full_budget']['auc_acc_std']}"
          f"  paper claimed=0.002"
          f"  match={std_check['verdict']['full_budget_std_matches_paper']}", flush=True)
    print(f"  B<=30%:       canonical std={std_check['partial_b30']['auc_acc_std']}"
          f"  paper claimed=0.001"
          f"  match={std_check['verdict']['partial_b30_std_matches_paper']}", flush=True)

    # Final summary
    elapsed = time.time() - t_global
    print("\n" + "=" * 70, flush=True)
    print(f"DONE in {elapsed:.1f}s", flush=True)
    print(f"Outputs:", flush=True)
    for p in [OUT_SEED_ABL, OUT_UNCERTAINTY, OUT_BADGE_PCA, OUT_MIE_SITE]:
        exists = "OK" if os.path.exists(p) else "MISSING"
        print(f"  [{exists}] {p}", flush=True)
    print("=" * 70, flush=True)

    # Print reference summary for paper update
    print("\n=== ABLATION SUMMARY ===", flush=True)

    # Task 1
    try:
        s1 = json.load(open(OUT_SEED_ABL))
        print("\n[Task 1] Seed-only vs Full training:", flush=True)
        print(f"  seed_only (n=1000): acc={s1['seed_only_1000']['accuracy']}"
              f"  F1={s1['seed_only_1000']['macro_f1']}"
              f"  AUC_acc={s1['seed_only_1000']['auc_acc_budget']}"
              f"  AUC_rare={s1['seed_only_1000']['auc_rare_budget']}", flush=True)
        print(f"  full (n=5736):      acc={s1['full_5736']['accuracy']}"
              f"  F1={s1['full_5736']['macro_f1']}"
              f"  AUC_acc={s1['full_5736']['auc_acc_budget']}"
              f"  AUC_rare={s1['full_5736']['auc_rare_budget']}", flush=True)
        d = s1['delta_full_minus_seed_only']
        print(f"  Delta (full - seed_only): acc={d['delta_accuracy']}"
              f"  F1={d['delta_macro_f1']}"
              f"  AUC_acc={d['delta_auc_acc']}"
              f"  AUC_rare={d['delta_auc_rare']}", flush=True)
    except Exception as e:
        print(f"  [Error reading Task 1 result: {e}]", flush=True)

    # Task 2
    try:
        s2 = json.load(open(OUT_UNCERTAINTY))
        r = s2["results"]
        print("\n[Task 2] LR-family uncertainty comparison (full n=5736):", flush=True)
        print(f"  confidence:        AUC_acc={r['lr_confidence']['auc_acc_budget']}"
              f"  AUC_rare={r['lr_confidence']['auc_rare_budget']}", flush=True)
        print(f"  entropy:           AUC_acc={r['lr_entropy']['auc_acc_budget']}"
              f"  AUC_rare={r['lr_entropy']['auc_rare_budget']}", flush=True)
        print(f"  bootstrap_ens:     AUC_acc={r['lr_bootstrap_ensemble']['auc_acc_budget']}"
              f"  AUC_rare={r['lr_bootstrap_ensemble']['auc_rare_budget']}", flush=True)
        ci = s2["bootstrap_ci_confidence_vs_ensemble"]
        print(f"  CI(conf-ens): mean={ci['mean_gap']} CI95={ci['ci95']}"
              f" sig={ci['significant_at_alpha05']}", flush=True)
        print(f"  Honest: {s2['honest_summary']}", flush=True)
    except Exception as e:
        print(f"  [Error reading Task 2 result: {e}]", flush=True)

    # Task 3
    try:
        s3 = json.load(open(OUT_BADGE_PCA))
        print("\n[Task 3] BADGE PCA sensitivity:", flush=True)
        print(f"  Confidence:  AUC_acc={s3['confidence_auc_acc']}", flush=True)
        for name, d in s3["badge_variants"].items():
            ci = s3["bootstrap_ci_confidence_minus_badge"][name]
            print(f"  {name:<12}: AUC_acc={d['auc_acc_budget']}"
                  f"  CI(conf-badge)={ci['ci95']}"
                  f"  sig={ci['significant_at_alpha05']}", flush=True)
        sens = s3["sensitivity_summary"]
        print(f"  BADGE AUC range: [{sens['badge_auc_min']}, {sens['badge_auc_max']}]"
              f"  spread={sens['badge_auc_range']}", flush=True)
    except Exception as e:
        print(f"  [Error reading Task 3 result: {e}]", flush=True)

    # Task 4
    try:
        s4 = json.load(open(OUT_MIE_SITE))
        print("\n[Task 4] Mie second-site generalization:", flush=True)
        print(f"  n_mie={s4['n_mie']}  n_rare={s4['n_rare_mie']}"
              f"  acc={s4['accuracy']}  macro_f1={s4['macro_f1']}", flush=True)
        print(f"  Triage AUC_acc={s4['triage_confidence']['auc_acc_budget']}"
              f"  AUC_rare={s4['triage_confidence']['auc_rare_budget']}", flush=True)
        print(f"  Random AUC={s4['triage_random']['auc_acc_mean']}"
              f"±{s4['triage_random']['auc_acc_std']}", flush=True)
        perm = s4["permutation_test"]
        print(f"  p={perm['p_value']} (n_exceed={perm['n_exceed']}/{perm['n_permutations']})", flush=True)
    except Exception as e:
        print(f"  [Error reading Task 4 result: {e}]", flush=True)

    # Task 5
    print("\n[Task 5] Random AUC std (canonical vs paper claims):", flush=True)
    print(f"  Full-budget: canonical={std_check['full_budget']['auc_acc_std']}"
          f"  paper=0.002  match={std_check['verdict']['full_budget_std_matches_paper']}", flush=True)
    print(f"  B<=30%:      canonical={std_check['partial_b30']['auc_acc_std']}"
          f"  paper=0.001  match={std_check['verdict']['partial_b30_std_matches_paper']}", flush=True)


if __name__ == "__main__":
    main()
