#!/usr/bin/env python3
"""
pipeline_canonical.py — single-source-of-truth script for the MiLT pipeline paper.

Numerical consistency notes:
  - Uses float64 embeddings throughout: same precision for fit AND predict_proba.
    All band counts derive from a single conf array.
  - TV values use single habitat computation with explicit None-exclusion,
    consistent 4-decimal rounding.
  - p-value = (n_exceed + 1) / (n_shuffle + 1) consistently (Phipson & Smyth 2010).

Outputs:
  outputs/benchmark/pipeline_canonical.json   ← main output, written incrementally
  outputs/benchmark/pipeline_canonical_env.json ← environment manifest

Usage:
  cd ~/trailcam_bench
  python3 pipeline_canonical.py

GPU rules: CUDA disabled; MPS used for MC-Dropout MLP only (lightweight embedding MLP).
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

# GPU rule: set CUDA_VISIBLE_DEVICES before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HOME = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HOME, "outputs")
BMK = os.path.join(OUT, "benchmark")
CANONICAL_PATH = os.path.join(BMK, "pipeline_canonical.json")
ENV_PATH = os.path.join(BMK, "pipeline_canonical_env.json")

# ---------------------------------------------------------------------------
# Constants (must match paper definitions exactly)
# ---------------------------------------------------------------------------
COMMON = {"ニホンジカ", "タヌキ", "イノシシ", "ニホンザル", "ニホンノウサギ"}
# Rare = true label not in COMMON and not '誤検出(動物なし)'
BUDGETS = [0, 5, 10, 20, 30, 50, 100]
BUDGETS_ECO = [0, 10, 20, 30, 50, 100]
BUDGETS_PARTIAL = [0, 5, 10, 20, 30]  # B≤30% partial AUC
CONF_HIGHCONF = 0.9
SEED_MAIN = 6906        # matches n_eval_seed_out for traceability
SEED_BADGE = 6906
SEED_PERM = 6906
SEED_ECO = 6906
SEED_BOOT = 6906
SEED_MCDROP = 42
N_PERM = 10_000
N_ECO_SHUFFLES = 2000
N_BOOT = 500
MC_PASSES = 15

# ---------------------------------------------------------------------------
# Incremental JSON save (write after each section; guarded against partial loss)
# ---------------------------------------------------------------------------

_state: Dict = {}

def _save(key: str, value, path: str = CANONICAL_PATH) -> None:
    """Update _state[key] and atomically overwrite the canonical JSON."""
    _state[key] = value
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    print(f"  [saved] {key} → {path}", flush=True)


# ---------------------------------------------------------------------------
# Utility functions
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


def tv(a: Dict, b: Dict, keys) -> float:
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def normc(counter: Counter) -> Dict:
    t = sum(counter.values()) or 1
    return {k: v / t for k, v in counter.items()}


def shannon(counter: Counter) -> float:
    t = sum(counter.values())
    if t == 0:
        return 0.0
    return float(-sum((v / t) * np.log(v / t) for v in counter.values() if v > 0))


def spearman_rank(a: List[float], b: List[float]) -> float:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if len(a_arr) < 2 or np.std(a_arr) == 0 or np.std(b_arr) == 0:
        return float("nan")
    ra = a_arr.argsort().argsort()
    rb = b_arr.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def file_sha256(path: str, n_bytes: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(n_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Step 1: Load data, train LR (float64), compute conf/ent/margin ONCE
# ---------------------------------------------------------------------------

def load_and_fit() -> Dict:
    """Load all data and train the LR classifier. Returns the canonical data dict."""
    from sklearn.linear_model import LogisticRegression

    print("\n[Step 1] Loading data and training LR classifier (float64)...", flush=True)
    t0 = time.time()

    # Load embeddings as float64 for numerical stability at band boundaries
    se = np.load(os.path.join(OUT, "embeddings.npy")).astype(np.float64)
    prov = json.load(open(os.path.join(OUT, "provisional_labels.json")))
    EMB: Dict[str, np.ndarray] = {}
    for i, r in enumerate(prov):
        EMB[r["crop"]] = se[i]

    ve = np.load(os.path.join(OUT, "video_embeddings.npy")).astype(np.float64)
    vit = json.load(open(os.path.join(OUT, "video_embed_index.json")))
    for i, it in enumerate(vit):
        EMB[it["crop"]] = ve[i]

    # Labels and metadata
    seed_lab = {r["crop"]: r["label"]
                for r in json.load(open(os.path.join(OUT, "seed_labels.json")))["labels"]}
    final = {r["crop"]: r["label"]
             for r in json.load(open(os.path.join(OUT, "labels_video_v1.json")))["labels"]}
    hab = {json.loads(line)["crop"]: json.loads(line)["habitat"]
           for line in open(os.path.join(OUT, "video_crops_index.jsonl"))}
    seed_set = set(json.load(open(os.path.join(OUT, "seed_crops.json"))))

    # Build training set (seed video labels + confirmed still labels)
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

    Z_train = np.stack(X_tr)   # float64
    print(f"  Training set: n={len(Z_train)}, dtype={Z_train.dtype}", flush=True)

    # Fit canonical classifier
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ).fit(Z_train, y_tr)
    classes: List[str] = list(clf.classes_)
    n_classes = len(classes)
    print(f"  Classifier: n_classes={n_classes}, classes={sorted(classes)}", flush=True)

    # Seed-out evaluation set (never in seed_crops, has ground truth)
    crops = [it["crop"] for it in vit
             if it["crop"] not in seed_set and it["crop"] in final]
    Z_eval = np.stack([EMB[c] for c in crops])   # float64
    n = len(crops)
    print(f"  Eval set: n={n}, dtype={Z_eval.dtype}", flush=True)

    # *** SINGLE predict_proba call — all downstream metrics use this P ***
    P = clf.predict_proba(Z_eval)   # shape (n, n_classes), float64
    print(f"  P dtype: {P.dtype}", flush=True)

    pred: List[str] = [classes[int(i)] for i in P.argmax(axis=1)]
    true: List[str] = [final[c] for c in crops]

    # Uncertainty scores (all float64)
    conf   = P.max(axis=1).astype(np.float64)
    ent    = -(P * np.log(P + 1e-9)).sum(axis=1).astype(np.float64)
    part   = np.sort(P, axis=1)
    margin = (part[:, -1] - part[:, -2]).astype(np.float64)

    # Masks
    rare_mask = np.array(
        [t not in COMMON and t != "誤検出(動物なし)" for t in true], dtype=bool
    )
    base_correct = np.array([pred[i] == true[i] for i in range(n)], dtype=bool)
    base_acc = float(base_correct.mean())

    # Valid habitats (>= 20 crops, None excluded explicitly)
    hab_all = [hab.get(crops[i]) for i in range(n)]
    habs_counter: Counter = Counter(h for h in hab_all if h is not None)
    habs_valid: List[str] = sorted(h for h, cnt in habs_counter.items() if cnt >= 20)

    print(f"  n_rare={int(rare_mask.sum())}, base_acc={base_acc:.4f}", flush=True)
    print(f"  Valid habitats ({len(habs_valid)}): {habs_valid}", flush=True)
    print(f"  Band [0.9,1.01) n={int(((conf >= 0.9) & (conf < 1.01)).sum())}", flush=True)
    print(f"  Step 1 done in {time.time()-t0:.1f}s", flush=True)

    # Data file hashes for env log
    data_hashes = {
        "embeddings.npy":           file_sha256(os.path.join(OUT, "embeddings.npy")),
        "video_embeddings.npy":     file_sha256(os.path.join(OUT, "video_embeddings.npy")),
        "seed_labels.json":         file_sha256(os.path.join(OUT, "seed_labels.json")),
        "labels_video_v1.json":     file_sha256(os.path.join(OUT, "labels_video_v1.json")),
        "labels_still_v1.json":     file_sha256(os.path.join(OUT, "labels_still_v1.json")),
        "provisional_labels.json":  file_sha256(os.path.join(OUT, "provisional_labels.json")),
        "seed_crops.json":          file_sha256(os.path.join(OUT, "seed_crops.json")),
    }

    return dict(
        clf=clf, classes=classes, n_classes=n_classes,
        crops=crops, Z_eval=Z_eval, Z_train=Z_train, P=P,
        pred=pred, true=true, n=n,
        conf=conf, ent=ent, margin=margin,
        rare_mask=rare_mask, base_correct=base_correct, base_acc=base_acc,
        hab=hab, hab_all=hab_all, habs_valid=habs_valid,
        data_hashes=data_hashes,
    )


# ---------------------------------------------------------------------------
# Step 2: Efficiency bands (Wilson CI)
# ---------------------------------------------------------------------------

def compute_efficiency_bands(data: Dict) -> List[Dict]:
    """Compute labeling efficiency per confidence band with Wilson 95% CI."""
    print("\n[Step 2] Efficiency bands...", flush=True)
    n, conf = data["n"], data["conf"]
    pred, true = data["pred"], data["true"]

    bands = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    result = []
    for lo, hi in bands:
        mask = (conf >= lo) & (conf < hi)
        nb = int(mask.sum())
        # corrected = mislabeled crops (pred != true) that would need correction
        n_corr = int(sum(pred[i] != true[i] for i in range(n) if mask[i]))
        result.append({
            "band": f"[{lo},{hi})",
            "n": nb,
            "corrected": n_corr,
            "rate": round(n_corr / nb, 4) if nb else None,
            "wilson95": list(wilson(n_corr, nb)),
        })
        print(f"  {result[-1]['band']:<12} n={nb:>4} corrected={n_corr} "
              f"rate={result[-1]['rate']} CI={result[-1]['wilson95']}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Step 3: Triage curves (accuracy + rare recall + ecology KPIs per budget)
# ---------------------------------------------------------------------------

def _build_hab_indices(data: Dict) -> Dict[str, List[int]]:
    """Pre-compute {habitat: [crop_indices]} once; reused across all shuffles."""
    n, crops, hab, habs_valid = data["n"], data["crops"], data["hab"], data["habs_valid"]
    return {h: [i for i in range(n) if hab.get(crops[i]) == h] for h in habs_valid}


def compute_eco_kpis_for_order(
    order_idx: np.ndarray, data: Dict, budgets: List[int],
    hab_indices: Dict[str, List[int]] = None
) -> Dict[int, Dict]:
    """
    Compute TV/Shannon/Spearman for a given review order.
    Pass hab_indices (pre-computed by _build_hab_indices) to avoid O(n) recomputation
    in the inner loop — critical for the 2000-shuffle ecology control.
    """
    n = data["n"]
    pred, true = data["pred"], data["true"]

    if hab_indices is None:
        hab_indices = _build_hab_indices(data)

    result: Dict[int, Dict] = {}
    for B in budgets:
        K = int(round(B / 100 * n))
        reviewed = set(order_idx[:K].tolist())
        got = [true[i] if i in reviewed else pred[i] for i in range(n)]

        tvs, sh_errs, sps = [], [], []
        for h, ix in hab_indices.items():
            if len(ix) < 20:
                continue
            keys = sorted(set(true[i] for i in ix) | set(got[i] for i in ix))
            tc = Counter(true[i] for i in ix)
            gc = Counter(got[i] for i in ix)
            tvs.append(tv(normc(gc), normc(tc), keys))
            sh_errs.append(abs(shannon(gc) - shannon(tc)))
            sps.append(spearman_rank(
                [gc.get(k_, 0) for k_ in keys],
                [tc.get(k_, 0) for k_ in keys]
            ))

        result[B] = {
            "tv_mean": round(float(np.mean(tvs)), 4) if tvs else None,
            "shannon_abs_err_mean": round(float(np.mean(sh_errs)), 4) if sh_errs else None,
            "spearman_rank_mean": round(float(np.nanmean(sps)), 4) if sps else None,
        }
    return result


def compute_triage_curves(data: Dict) -> Dict:
    """Full triage curve for confidence ordering including ecology KPIs."""
    print("\n[Step 3] Triage curves (confidence order + ecology KPIs)...", flush=True)
    n = data["n"]
    conf, rare_mask = data["conf"], data["rare_mask"]
    base_correct = data["base_correct"]
    pred, true = data["pred"], data["true"]
    rare_idx = [i for i in range(n) if rare_mask[i]]

    ci_order = np.argsort(conf)   # least-confident first
    xs = [b / 100.0 for b in BUDGETS]
    hab_indices = _build_hab_indices(data)

    # Per-budget: accuracy upper bound, rare recall upper bound, ecology KPIs
    eco_kpis = compute_eco_kpis_for_order(ci_order, data, BUDGETS, hab_indices)
    rows = []
    for B in BUDGETS:
        K = int(round(B / 100 * n))
        reviewed = set(ci_order[:K].tolist())
        correct_arr = base_correct.copy()
        for i in reviewed:
            correct_arr[i] = True
        acc = float(correct_arr.mean())
        rr = float(correct_arr[rare_mask].mean()) if rare_mask.any() else float("nan")
        row = {
            "budget_pct": B,
            "accuracy_ub": round(acc, 4),
            "rare_recall_ub": round(rr, 4),
            "tv_mean": eco_kpis[B]["tv_mean"],
            "shannon_abs_err_mean": eco_kpis[B]["shannon_abs_err_mean"],
            "spearman_rank_mean": eco_kpis[B]["spearman_rank_mean"],
        }
        rows.append(row)
        print(f"  B={B:>3}%  acc={acc:.4f}  rareRecall={rr:.4f}  "
              f"TV={eco_kpis[B]['tv_mean']}  Sp={eco_kpis[B]['spearman_rank_mean']}", flush=True)

    accs = [r["accuracy_ub"] for r in rows]
    rares = [r["rare_recall_ub"] for r in rows]
    auc_acc = round(_trapz(np.array(accs), np.array(xs)), 4)
    auc_rare = round(_trapz(np.array(rares), np.array(xs)), 4)
    print(f"  AUC_acc={auc_acc}  AUC_rare={auc_rare}", flush=True)

    return {
        "n_eval_seed_out": n,
        "n_rare_seedout": int(rare_mask.sum()),
        "auto_accuracy_seedout": round(float(base_correct.mean()), 4),
        "confidence_order_index": ci_order.tolist(),  # saved for downstream reuse
        "budgets_pct": BUDGETS,
        "triage_detail": rows,
        "auc_acc_budget": auc_acc,
        "auc_rare_budget": auc_rare,
    }


# ---------------------------------------------------------------------------
# Helpers: AUC computation and random permutation AUC
# ---------------------------------------------------------------------------

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
    """Vectorized AUC computation for random permutations (acc only)."""
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
    """Bootstrap 95% CI on AUC(order_a) - AUC(order_b)."""
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


# ---------------------------------------------------------------------------
# Step 4: AL baselines (Coreset, BADGE-approx, 10k permutation, bootstrap CI)
# ---------------------------------------------------------------------------

def coreset_order(Z_train: np.ndarray, Z_eval: np.ndarray) -> np.ndarray:
    """k-center greedy ordering of Z_eval (most diverse from training set first)."""
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
    BADGE-approximation: k-center greedy on logistic-regression gradient embeddings.

    Gradient embedding: g_i = (p_i - e_{argmax(p_i)}) ⊗ z_proj_i
    where z_proj is obtained via PCA-20 projection of BioCLIP embeddings.
    This yields 19*20=380-dim vectors for k-center greedy.
    BADGE-approx (PCA-20 projected) is the method used throughout this paper.
    """
    from sklearn.decomposition import PCA

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


def compute_al_baselines(data: Dict) -> Dict:
    """B1: All AL baselines + B4: 10k permutation + bootstrap CIs."""
    print("\n[Step 4] AL baselines + permutation test + bootstrap CI...", flush=True)
    n = data["n"]
    base_correct = data["base_correct"]
    rare_mask = data["rare_mask"]
    conf, ent, margin = data["conf"], data["ent"], data["margin"]
    Z_eval, Z_train, P = data["Z_eval"], data["Z_train"], data["P"]

    rng_badge = np.random.default_rng(SEED_BADGE)
    rng_perm = np.random.default_rng(SEED_PERM)

    # Deterministic orderings (shared with triage curves)
    orders_det = {
        "confidence": np.argsort(conf),       # least-confident first
        "entropy":    np.argsort(-ent),        # highest-entropy first
        "margin":     np.argsort(margin),      # smallest-margin first
    }

    print("[Step 4a] Coreset ordering...", flush=True)
    t0 = time.time()
    orders_det["coreset"] = coreset_order(Z_train, Z_eval)
    print(f"  Coreset done in {time.time()-t0:.1f}s", flush=True)

    print("[Step 4b] BADGE-approx ordering (PCA-20 gradient embedding)...", flush=True)
    t0 = time.time()
    orders_det["badge_approx"] = badge_order(Z_eval, P, rng_badge, pca_components=20)
    print(f"  BADGE done in {time.time()-t0:.1f}s", flush=True)

    # AUC for all deterministic orderings
    result_orders = {}
    for name, idx in orders_det.items():
        accs, auc_acc, rares, auc_rare = compute_auc_from_order(
            idx, base_correct, n, BUDGETS, rare_mask
        )
        result_orders[name] = {
            "acc_by_budget": accs,
            "auc_acc_budget": auc_acc,
            "rare_recall_by_budget": rares,
            "rare_recall_auc": auc_rare,
        }
        print(f"  {name:<18}  AUC_acc={auc_acc}  AUC_rare={auc_rare}", flush=True)

    conf_auc = result_orders["confidence"]["auc_acc_budget"]

    # 10k permutation test (p = (n_exceed+1)/(N_PERM+1))
    print("[Step 4c] 10k permutation test (vectorized)...", flush=True)
    t0 = time.time()
    rand_aucs = fast_rand_aucs(N_PERM, base_correct, BUDGETS, n, rng_perm, batch=500)
    n_exceed = int(np.sum(rand_aucs >= conf_auc))
    # Canonical p-value: (n_exceed+1)/(N_PERM+1) — Phipson & Smyth 2010
    pval_10k = (n_exceed + 1) / (N_PERM + 1)
    rand_mean = round(float(rand_aucs.mean()), 4)
    rand_std = round(float(rand_aucs.std()), 4)
    print(f"  10k perm done in {time.time()-t0:.1f}s  "
          f"rand={rand_mean}±{rand_std}  n_exceed={n_exceed}  p={pval_10k:.6f}", flush=True)

    # Rare recall AUC for random (2000 shuffles)
    rng_rare = np.random.default_rng(SEED_MAIN + 1)
    N_RARE_SH = 2000
    rare_aucs = []
    for _ in range(N_RARE_SH):
        idx = rng_rare.permutation(n)
        _, _, _, auc_r = compute_auc_from_order(idx, base_correct, n, BUDGETS, rare_mask)
        rare_aucs.append(auc_r)
    rare_auc_rand_mean = round(float(np.mean(rare_aucs)), 4)
    rare_auc_rand_std = round(float(np.std(rare_aucs)), 4)

    result_orders["random"] = {
        "auc_acc_budget": rand_mean,
        "auc_acc_budget_std": rand_std,
        "rare_recall_auc": rare_auc_rand_mean,
        "rare_recall_auc_std": rare_auc_rand_std,
        "n_shuffles": N_PERM,
    }

    # Bootstrap CI: confidence vs Coreset, confidence vs BADGE
    print("[Step 4d] Bootstrap CI (confidence vs Coreset/BADGE, n_boot=500)...", flush=True)
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
    print("[Step 4d] Bootstrap done.", flush=True)

    return {
        "n_eval_seed_out": n,
        "n_rare_seedout": int(rare_mask.sum()),
        "budgets_pct": BUDGETS,
        "note": (
            "Coreset = k-center greedy on BioCLIP embedding space (float64); "
            "BADGE_approx = k-center greedy on LR gradient embedding (PCA-20 projected, "
            "K=19 classes → 380-dim); "
            "review order = most informative/diverse first. "
            "p-value formula: (n_exceed+1)/(N_PERM+1) [Phipson & Smyth 2010]."
        ),
        "orders": result_orders,
        "confidence_vs_random_perm": {
            "n_shuffles": N_PERM,
            "n_exceed": n_exceed,
            "p_value": round(pval_10k, 6),
            "formula": "(n_exceed+1)/(N_PERM+1)",
        },
        "bootstrap_auc_gap_confidence_minus": {
            "coreset": ci_vs_coreset,
            "badge_approx": ci_vs_badge,
        },
        # Save orders for partial AUC and rare gap computation
        "_orders_det": {k: v.tolist() for k, v in orders_det.items()},
    }


# ---------------------------------------------------------------------------
# Step 5: Rare recall by band
# ---------------------------------------------------------------------------

def compute_rare_recall_by_band(data: Dict) -> Dict:
    """Per-band rare species recall + high-confidence misclassification."""
    print("\n[Step 5] Rare recall by confidence band...", flush=True)
    n, conf = data["n"], data["conf"]
    pred, true = data["pred"], data["true"]
    rare_mask = data["rare_mask"]

    bands = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    band_results = []

    high_conf_rare_misclass = [
        {"index": i, "true": true[i], "pred": pred[i], "conf": round(float(conf[i]), 4)}
        for i in range(n)
        if conf[i] >= CONF_HIGHCONF and pred[i] != true[i] and rare_mask[i]
    ]

    for lo, hi in bands:
        in_band = (conf >= lo) & (conf < hi)
        in_band_rare = in_band & rare_mask
        n_all_band = int(in_band.sum())
        n_rare_band = int(in_band_rare.sum())

        if n_rare_band > 0:
            n_rare_correct = int(
                sum(pred[i] == true[i] for i in range(n) if in_band_rare[i])
            )
            recall = round(n_rare_correct / n_rare_band, 4)
            w = wilson(n_rare_correct, n_rare_band)
        else:
            n_rare_correct = 0
            recall = None
            w = (None, None)

        n_hc_misclass = int(sum(
            1 for i in range(n)
            if in_band_rare[i] and pred[i] != true[i] and conf[i] >= CONF_HIGHCONF
        ))
        band_results.append({
            "band": f"[{lo},{hi})",
            "n_all_in_band": n_all_band,
            "n_rare_in_band": n_rare_band,
            "n_rare_correct_in_band": n_rare_correct,
            "rare_recall_in_band": recall,
            "wilson95_rare_recall": list(w),
            "n_rare_highconf_misclass_in_band": n_hc_misclass,
        })
        print(f"  {band_results[-1]['band']:<12}  n_all={n_all_band:>4}  "
              f"n_rare={n_rare_band:>4}  recall={recall}  CI={list(w)}", flush=True)

    overall_rare_correct = int(sum(1 for i in range(n) if rare_mask[i] and pred[i] == true[i]))
    overall_rare_total = int(rare_mask.sum())

    print(f"  High-conf rare misclassifications (c≥{CONF_HIGHCONF}): "
          f"{len(high_conf_rare_misclass)}", flush=True)

    return {
        "n_eval_seed_out": n,
        "n_rare_seedout": overall_rare_total,
        "n_rare_correct_overall": overall_rare_correct,
        "rare_recall_overall": round(overall_rare_correct / overall_rare_total, 4)
            if overall_rare_total > 0 else None,
        "rare_definition": "true label not in COMMON and not '誤検出(動物なし)'",
        "COMMON_species": sorted(COMMON),
        "confidence_threshold_highconf": CONF_HIGHCONF,
        "bands": band_results,
        "high_confidence_rare_misclassifications": {
            "threshold": CONF_HIGHCONF,
            "n_total": len(high_conf_rare_misclass),
            "examples": sorted(high_conf_rare_misclass, key=lambda x: -x["conf"])[:10],
        },
    }


# ---------------------------------------------------------------------------
# Step 6: Ecology random control (2000 shuffles, permutation test)
# ---------------------------------------------------------------------------

def compute_ecology_random_control(data: Dict) -> Dict:
    """Ecological KPI with 2000-shuffle random control."""
    print("\n[Step 6] Ecology random control (2000 shuffles)...", flush=True)
    n, conf = data["n"], data["conf"]
    pred, true, crops = data["pred"], data["true"], data["crops"]
    hab, habs_valid = data["hab"], data["habs_valid"]
    base_correct = data["base_correct"]

    rng_eco = np.random.default_rng(SEED_ECO)
    ci_order = np.argsort(conf)

    # Pre-compute habitat→crop-index map ONCE (avoids O(n) recomputation per shuffle)
    hab_indices = _build_hab_indices(data)

    # Confidence-order KPIs
    eco_conf = compute_eco_kpis_for_order(ci_order, data, BUDGETS_ECO, hab_indices)

    # Random-order KPIs
    acc_tv = {B: [] for B in BUDGETS_ECO}
    acc_sh = {B: [] for B in BUDGETS_ECO}
    acc_sp = {B: [] for B in BUDGETS_ECO}

    t0 = time.time()
    for s in range(N_ECO_SHUFFLES):
        rand_order = rng_eco.permutation(n)
        eco_s = compute_eco_kpis_for_order(rand_order, data, BUDGETS_ECO, hab_indices)
        for B in BUDGETS_ECO:
            if eco_s[B]["tv_mean"] is not None:
                acc_tv[B].append(eco_s[B]["tv_mean"])
            if eco_s[B]["shannon_abs_err_mean"] is not None:
                acc_sh[B].append(eco_s[B]["shannon_abs_err_mean"])
            if eco_s[B]["spearman_rank_mean"] is not None:
                acc_sp[B].append(eco_s[B]["spearman_rank_mean"])
        if (s + 1) % 500 == 0:
            print(f"  shuffle {s+1}/{N_ECO_SHUFFLES}  elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"  Ecology shuffles done in {time.time()-t0:.1f}s", flush=True)

    eco_rand = {}
    for B in BUDGETS_ECO:
        eco_rand[B] = {
            "tv_mean": round(float(np.mean(acc_tv[B])), 4) if acc_tv[B] else None,
            "tv_std":  round(float(np.std(acc_tv[B])), 4) if acc_tv[B] else None,
            "shannon_abs_err_mean": round(float(np.mean(acc_sh[B])), 4) if acc_sh[B] else None,
            "shannon_abs_err_std":  round(float(np.std(acc_sh[B])), 4) if acc_sh[B] else None,
            "spearman_rank_mean": round(float(np.nanmean(acc_sp[B])), 4) if acc_sp[B] else None,
            "spearman_rank_std":  round(float(np.nanstd(acc_sp[B])), 4) if acc_sp[B] else None,
        }

    # Permutation test ((n_exceed+1)/(N+1))
    perm_test = {}
    for B in BUDGETS_ECO:
        tv_c = eco_conf[B]["tv_mean"]
        sh_c = eco_conf[B]["shannon_abs_err_mean"]
        sp_c = eco_conf[B]["spearman_rank_mean"]

        def pval_lower(vals, obs):
            # lower is better; p = P(random <= obs)
            if not vals or obs is None:
                return None
            n_exc = sum(1 for v in vals if v <= obs)
            return round((n_exc + 1) / (len(vals) + 1), 4)

        def pval_higher(vals, obs):
            # higher is better; p = P(random >= obs)
            if not vals or obs is None:
                return None
            n_exc = sum(1 for v in vals if not np.isnan(v) and v >= obs)
            return round((n_exc + 1) / (len(vals) + 1), 4)

        perm_test[B] = {
            "tv_p_value": pval_lower(acc_tv[B], tv_c),
            "shannon_p_value": pval_lower(acc_sh[B], sh_c),
            "spearman_p_value": pval_higher(acc_sp[B], sp_c),
            "formula": "(n_exceed+1)/(N_shuffles+1)",
            "note_tv_shannon": "p = P(random ≤ confidence_order); lower TV/Shannon is better",
            "note_spearman": "p = P(random ≥ confidence_order); higher Spearman is better",
        }
        print(f"  B={B:>3}%  TV: conf={tv_c}  p={perm_test[B]['tv_p_value']}"
              f"  |  Sp: conf={sp_c}  p={perm_test[B]['spearman_p_value']}", flush=True)

    return {
        "n_eval_seed_out": n,
        "budgets_pct": BUDGETS_ECO,
        "n_eco_shuffles": N_ECO_SHUFFLES,
        "n_habitats_valid": len(habs_valid),
        "habitats_valid": habs_valid,
        "n_by_habitat": {h: int(sum(hab.get(crops[i]) == h for i in range(n)))
                         for h in habs_valid},
        "confidence_order": {str(B): eco_conf[B] for B in BUDGETS_ECO},
        "random_order": {str(B): eco_rand[B] for B in BUDGETS_ECO},
        "permutation_test": {str(B): perm_test[B] for B in BUDGETS_ECO},
    }


# ---------------------------------------------------------------------------
# Step 7: MC-Dropout entropy triage (MC-Dropout MLP on BioCLIP embeddings)
# ---------------------------------------------------------------------------

def compute_mc_dropout_triage(data: Dict) -> Dict:
    """
    MC-Dropout triage comparison.

    Design: Train a 2-layer MLP with Dropout(0.2) on the same BioCLIP embeddings
    and training split as the LR classifier (seed_lab + confirmed still labels,
    all 19 classes). Then compute 15-pass MC-Dropout predictive entropy ordering
    on the seed-out eval set and compare triage AUC to LR confidence ordering.

    Rationale: The existing ViT-B FT (exp5/train_finetune.py) is a 5-class
    open-set detector, NOT a 19-class triage classifier, so its MC-Dropout
    entropy cannot be directly reused for this comparison. The MC-Dropout MLP
    on BioCLIP embeddings is a lightweight equivalent that:
      (a) uses the same embedding space as LR
      (b) enables genuine MC-Dropout uncertainty (LR has no dropout)
      (c) covers all 19 pipeline classes
    """
    import torch
    import torch.nn as nn

    print("\n[Step 7] MC-Dropout MLP triage...", flush=True)
    t0 = time.time()

    n = data["n"]
    Z_train, Z_eval = data["Z_train"].astype(np.float32), data["Z_eval"].astype(np.float32)
    y_tr = []
    # Rebuild y_tr in same order as Z_train
    seed_lab = {r["crop"]: r["label"]
                for r in json.load(open(os.path.join(OUT, "seed_labels.json")))["labels"]}
    still = json.load(open(os.path.join(OUT, "labels_still_v1.json")))["labels"]
    prov = json.load(open(os.path.join(OUT, "provisional_labels.json")))
    EMB_tmp: Dict[str, np.ndarray] = {}
    se = np.load(os.path.join(OUT, "embeddings.npy")).astype(np.float64)
    for i, r in enumerate(prov):
        EMB_tmp[r["crop"]] = se[i]
    ve = np.load(os.path.join(OUT, "video_embeddings.npy")).astype(np.float64)
    vit = json.load(open(os.path.join(OUT, "video_embed_index.json")))
    for i, it in enumerate(vit):
        EMB_tmp[it["crop"]] = ve[i]
    for c, l in seed_lab.items():
        if c in EMB_tmp:
            y_tr.append(l)
    for r in still:
        if r["crop"] in EMB_tmp:
            y_tr.append(r["label"])
    assert len(y_tr) == len(Z_train), f"y_tr/Z_train mismatch: {len(y_tr)} vs {len(Z_train)}"

    classes = data["classes"]
    n_classes = len(classes)
    label_to_idx = {l: i for i, l in enumerate(classes)}
    y_idx = np.array([label_to_idx[l] for l in y_tr], dtype=np.int64)

    # Class weights (balanced)
    counts = np.bincount(y_idx, minlength=n_classes).astype(np.float32)
    weights = len(y_idx) / (n_classes * np.clip(counts, 1.0, None))

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}", flush=True)

    # MC-Dropout MLP
    class MCDropoutMLP(nn.Module):
        def __init__(self, in_dim: int, hidden: int, n_cls: int, p: float = 0.2):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(p)
            self.fc2 = nn.Linear(hidden, n_cls)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(self.drop(self.relu(self.fc1(x))))

    torch.manual_seed(SEED_MCDROP)
    np.random.seed(SEED_MCDROP)

    in_dim = Z_train.shape[1]   # 512 (BioCLIP)
    model = MCDropoutMLP(in_dim, 256, n_classes, p=0.2).to(device)

    X_t = torch.tensor(Z_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_idx, dtype=torch.long).to(device)
    w_t = torch.tensor(weights, dtype=torch.float32).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=w_t)

    N_EPOCHS = 80
    model.train()
    for ep in range(N_EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        opt.step()
        if (ep + 1) % 20 == 0:
            print(f"  epoch {ep+1}/{N_EPOCHS}  loss={loss.item():.4f}", flush=True)

    elapsed_train = time.time() - t0
    print(f"  MLP training done in {elapsed_train:.1f}s", flush=True)

    # MC-Dropout inference (15 passes)
    X_eval_t = torch.tensor(Z_eval, dtype=torch.float32).to(device)
    model.train()   # keep Dropout active
    with torch.no_grad():
        probs_acc = torch.zeros(n, n_classes, dtype=torch.float32, device=device)
        for _ in range(MC_PASSES):
            logits = model(X_eval_t)
            probs_acc += torch.softmax(logits, dim=1)
    P_mc = (probs_acc / MC_PASSES).cpu().numpy()

    mc_ent = -(P_mc * np.log(P_mc + 1e-9)).sum(axis=1)
    mc_order = np.argsort(-mc_ent)   # highest-entropy first (= most uncertain first)

    base_correct = data["base_correct"]
    rare_mask = data["rare_mask"]

    _, auc_mc_acc, _, auc_mc_rare = compute_auc_from_order(
        mc_order, base_correct, n, BUDGETS, rare_mask
    )
    conf_order = np.argsort(data["conf"])
    _, auc_conf_acc, _, auc_conf_rare = compute_auc_from_order(
        conf_order, base_correct, n, BUDGETS, rare_mask
    )
    lr_ent_order = np.argsort(-data["ent"])
    _, auc_lr_ent_acc, _, auc_lr_ent_rare = compute_auc_from_order(
        lr_ent_order, base_correct, n, BUDGETS, rare_mask
    )

    # Partial AUC for B≤30% (acc)
    accs_mc, auc_mc_partial, _, _ = compute_auc_from_order(
        mc_order, base_correct, n, BUDGETS_PARTIAL, rare_mask
    )
    accs_conf, auc_conf_partial, _, _ = compute_auc_from_order(
        conf_order, base_correct, n, BUDGETS_PARTIAL, rare_mask
    )

    elapsed_total = time.time() - t0
    print(f"  AUC_acc: LR_conf={auc_conf_acc}  LR_ent={auc_lr_ent_acc}  MC_ent={auc_mc_acc}", flush=True)
    print(f"  AUC_rare: LR_conf={auc_conf_rare}  LR_ent={auc_lr_ent_rare}  MC_ent={auc_mc_rare}", flush=True)

    return {
        "description": (
            "MC-Dropout entropy triage using a 2-layer MLP (512→256→19, Dropout=0.2) "
            "trained on BioCLIP embeddings (same training split as LR canonical classifier). "
            "MC passes=15, seed=42. "
            "Compares: (a) LR confidence order, (b) LR predictive entropy order, "
            "(c) MC-Dropout MLP entropy order. "
            "Note: exp5/train_finetune.py ViT-B is 5-class open-set detector "
            "and cannot be reused for 19-class triage."
        ),
        "model_config": {
            "architecture": "MLP-2layer (512→256→19)",
            "dropout": 0.2,
            "mc_passes": MC_PASSES,
            "epochs": N_EPOCHS,
            "lr": 1e-3,
            "optimizer": "AdamW",
            "class_weight": "balanced",
            "seed": SEED_MCDROP,
            "device": str(device),
            "elapsed_train_s": round(elapsed_train, 1),
            "elapsed_total_s": round(elapsed_total, 1),
        },
        "budgets_pct": BUDGETS,
        "results": {
            "lr_confidence": {
                "description": "LR max softmax probability ordering (least-confident first)",
                "auc_acc_budget": auc_conf_acc,
                "auc_rare_budget": auc_conf_rare,
            },
            "lr_entropy": {
                "description": "LR predictive entropy ordering (highest-entropy first)",
                "auc_acc_budget": auc_lr_ent_acc,
                "auc_rare_budget": auc_lr_ent_rare,
            },
            "mc_dropout_entropy": {
                "description": "MC-Dropout MLP predictive entropy ordering (highest-entropy first)",
                "auc_acc_budget": auc_mc_acc,
                "auc_rare_budget": auc_mc_rare,
            },
        },
        "interpretation": (
            "If MC-Dropout entropy AUC ≈ LR confidence AUC, "
            "the ordering effect is robust to model choice / uncertainty estimator. "
            "If inferior, the concern that 'simpler confidence is not reliable' "
            "may warrant further investigation."
        ),
    }


# ---------------------------------------------------------------------------
# Step 8: 「不明」 exclusion sensitivity
# ---------------------------------------------------------------------------

def compute_unknown_exclusion_sensitivity(data: Dict) -> Dict:
    """
    Recompute rare_recall_by_band and high-conf misclassification
    after excluding '不明' from the rare species definition.
    """
    print("\n[Step 8] '不明' exclusion sensitivity...", flush=True)
    n, conf = data["n"], data["conf"]
    pred, true = data["pred"], data["true"]

    # Rare (excluding 不明): true label not in COMMON, not '誤検出(動物なし)', not '不明'
    rare_excl_unknown = np.array(
        [t not in COMMON and t not in ("誤検出(動物なし)", "不明") for t in true], dtype=bool
    )
    n_rare_excl = int(rare_excl_unknown.sum())
    print(f"  n_rare (original)={int(data['rare_mask'].sum())}  "
          f"n_rare (excl 不明)={n_rare_excl}", flush=True)

    bands = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    band_results = []

    high_conf_rare_excl = [
        {"index": i, "true": true[i], "pred": pred[i], "conf": round(float(conf[i]), 4)}
        for i in range(n)
        if conf[i] >= CONF_HIGHCONF and pred[i] != true[i] and rare_excl_unknown[i]
    ]

    for lo, hi in bands:
        in_band = (conf >= lo) & (conf < hi)
        in_band_rare = in_band & rare_excl_unknown
        n_all_band = int(in_band.sum())
        n_rare_band = int(in_band_rare.sum())

        if n_rare_band > 0:
            n_rare_correct = int(
                sum(pred[i] == true[i] for i in range(n) if in_band_rare[i])
            )
            recall = round(n_rare_correct / n_rare_band, 4)
            w = wilson(n_rare_correct, n_rare_band)
        else:
            n_rare_correct = 0
            recall = None
            w = (None, None)

        n_hc_misclass = int(sum(
            1 for i in range(n)
            if in_band_rare[i] and pred[i] != true[i] and conf[i] >= CONF_HIGHCONF
        ))
        band_results.append({
            "band": f"[{lo},{hi})",
            "n_all_in_band": n_all_band,
            "n_rare_in_band": n_rare_band,
            "n_rare_correct_in_band": n_rare_correct,
            "rare_recall_in_band": recall,
            "wilson95_rare_recall": list(w),
            "n_rare_highconf_misclass_in_band": n_hc_misclass,
        })
        print(f"  {band_results[-1]['band']:<12}  n_rare={n_rare_band:>4}  "
              f"recall={recall}  hc_misclass={n_hc_misclass}", flush=True)

    # 不明ラベルが何件あるか
    n_unknown = int(sum(1 for t in true if t == "不明"))
    n_unknown_rare_in_orig = int(sum(
        1 for i in range(n) if true[i] == "不明" and data["rare_mask"][i]
    ))
    print(f"  n_unknown in eval={n_unknown}  n_unknown in orig rare={n_unknown_rare_in_orig}", flush=True)
    print(f"  High-conf rare misclassifications (excl 不明): {len(high_conf_rare_excl)}", flush=True)

    return {
        "description": (
            "Rare species definition with '不明' excluded. "
            "Rare = true not in COMMON, not '誤検出(動物なし)', not '不明'. "
            "Tests whether high-confidence misclassifications persist for "
            "genuine novel species (excluding the ambiguous '不明' category)."
        ),
        "n_eval_seed_out": n,
        "n_rare_original": int(data["rare_mask"].sum()),
        "n_rare_excl_unknown": n_rare_excl,
        "n_unknown_in_eval": n_unknown,
        "n_unknown_in_original_rare": n_unknown_rare_in_orig,
        "confidence_threshold_highconf": CONF_HIGHCONF,
        "bands": band_results,
        "high_confidence_rare_misclassifications_excl_unknown": {
            "threshold": CONF_HIGHCONF,
            "n_total": len(high_conf_rare_excl),
            "examples": sorted(high_conf_rare_excl, key=lambda x: -x["conf"])[:10],
        },
    }


# ---------------------------------------------------------------------------
# Step 9: B≤30% partial AUC (acc and rare recall)
# ---------------------------------------------------------------------------

def compute_partial_auc_b30(data: Dict, al_result: Dict) -> Dict:
    """Partial AUC over B∈[0,5,10,20,30]% grid for practical budget range."""
    print("\n[Step 9] B≤30% partial AUC...", flush=True)
    n = data["n"]
    base_correct = data["base_correct"]
    rare_mask = data["rare_mask"]

    orders_det = {k: np.array(v) for k, v in al_result["_orders_det"].items()}
    # Include random from existing 10k result
    rand_auc_partial = None

    result_orders = {}
    for name, idx in orders_det.items():
        accs, auc_acc, rares, auc_rare = compute_auc_from_order(
            idx, base_correct, n, BUDGETS_PARTIAL, rare_mask
        )
        result_orders[name] = {
            "acc_by_budget": accs,
            "auc_acc_budget_b30": auc_acc,
            "rare_recall_by_budget": rares,
            "rare_recall_auc_b30": auc_rare,
        }
        print(f"  {name:<18}  partial_AUC_acc={auc_acc}  partial_AUC_rare={auc_rare}", flush=True)

    # Random baseline for partial AUC
    rng_part = np.random.default_rng(SEED_MAIN + 999)
    N_RAND = 2000
    rand_aucs_partial = []
    for _ in range(N_RAND):
        idx = rng_part.permutation(n)
        _, auc_acc, _, _ = compute_auc_from_order(idx, base_correct, n, BUDGETS_PARTIAL, rare_mask)
        rand_aucs_partial.append(auc_acc)
    result_orders["random"] = {
        "auc_acc_budget_b30": round(float(np.mean(rand_aucs_partial)), 4),
        "auc_acc_budget_b30_std": round(float(np.std(rand_aucs_partial)), 4),
        "n_shuffles": N_RAND,
    }
    print(f"  {'random':<18}  partial_AUC_acc={result_orders['random']['auc_acc_budget_b30']}"
          f"±{result_orders['random']['auc_acc_budget_b30_std']}", flush=True)

    return {
        "description": (
            "Partial AUC computed over practical budget range B∈{0,5,10,20,30}%. "
            "This isolates performance in the realistic low-budget regime where "
            "researchers can only review a fraction of the data."
        ),
        "budgets_pct": BUDGETS_PARTIAL,
        "orders": result_orders,
    }


# ---------------------------------------------------------------------------
# Step 10: BADGE vs confidence rare recall AUC gap (bootstrap)
# ---------------------------------------------------------------------------

def compute_badge_conf_rare_gap(data: Dict, al_result: Dict) -> Dict:
    """Bootstrap CI on BADGE_approx vs confidence rare recall AUC gap."""
    print("\n[Step 10] BADGE vs confidence rare recall AUC gap (bootstrap)...", flush=True)
    n = data["n"]
    base_correct = data["base_correct"]
    rare_mask = data["rare_mask"]

    conf_order = np.array(al_result["_orders_det"]["confidence"])
    badge_order_arr = np.array(al_result["_orders_det"]["badge_approx"])

    _, _, _, auc_conf_rare = compute_auc_from_order(
        conf_order, base_correct, n, BUDGETS, rare_mask
    )
    _, _, _, auc_badge_rare = compute_auc_from_order(
        badge_order_arr, base_correct, n, BUDGETS, rare_mask
    )
    observed_gap = round(auc_badge_rare - auc_conf_rare, 4)  # BADGE - confidence
    print(f"  AUC_rare: BADGE={auc_badge_rare}  conf={auc_conf_rare}  "
          f"gap(BADGE-conf)={observed_gap}", flush=True)

    # Bootstrap CI on gap (BADGE - confidence rare recall AUC)
    rng_boot = np.random.default_rng(SEED_BOOT + 500)

    # Use bootstrap_auc_gap with RARE recall
    # Need a version for rare_recall AUC
    rank_badge = np.empty(n, dtype=np.int64)
    rank_badge[badge_order_arr] = np.arange(n)
    rank_conf = np.empty(n, dtype=np.int64)
    rank_conf[conf_order] = np.arange(n)

    xs = np.array([b / 100.0 for b in BUDGETS])
    gaps = []
    for _ in range(N_BOOT):
        idx = rng_boot.integers(0, n, n)
        n_bs = len(idx)
        bc_sub = base_correct[idx]
        rare_sub = rare_mask[idx]

        pos_badge = np.argsort(rank_badge[idx])
        pos_conf  = np.argsort(rank_conf[idx])

        def auc_rare_bs(positions: np.ndarray) -> float:
            rares_bs = []
            for B in BUDGETS:
                K = int(round(B / 100 * n_bs))
                reviewed_pos = set(positions[:K].tolist())
                correct_bs = bc_sub.copy()
                for pos in reviewed_pos:
                    correct_bs[pos] = True
                if rare_sub.any():
                    rares_bs.append(float(correct_bs[rare_sub].mean()))
                else:
                    rares_bs.append(float("nan"))
            return _trapz(np.array(rares_bs), xs)

        gaps.append(auc_rare_bs(pos_badge) - auc_rare_bs(pos_conf))

    lo, hi = np.percentile(gaps, [2.5, 97.5])
    ci = {"mean_gap": round(float(np.mean(gaps)), 4),
          "ci95": [round(float(lo), 4), round(float(hi), 4)],
          "n_bootstrap": N_BOOT}
    print(f"  Bootstrap CI (BADGE-conf rare gap): mean={ci['mean_gap']} "
          f"CI95={ci['ci95']}", flush=True)

    # Significance note
    sig = "significant" if (lo > 0 or hi < 0) else "not significant"
    print(f"  Gap is {sig} at alpha=0.05 (CI {'excludes' if sig == 'significant' else 'includes'} 0)", flush=True)

    return {
        "description": (
            "Bootstrap 95% CI on rare recall AUC: BADGE_approx − confidence ordering. "
            "Positive gap = BADGE achieves higher rare recall AUC. "
            "BADGE_approx = k-center greedy on LR gradient embedding (PCA-20 projected, "
            "K=19 classes → 380-dim)."
        ),
        "auc_rare_confidence": auc_conf_rare,
        "auc_rare_badge_approx": auc_badge_rare,
        "observed_gap_badge_minus_confidence": observed_gap,
        "bootstrap_ci": ci,
        "significance": sig,
        "budgets_pct": BUDGETS,
    }


# ---------------------------------------------------------------------------
# Environment manifest
# ---------------------------------------------------------------------------

def build_env_manifest(data: Dict) -> Dict:
    import sklearn
    import torch

    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HOME, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unavailable"

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "git_commit": git_commit,
        "script": __file__,
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed_main": SEED_MAIN,
        "seed_badge": SEED_BADGE,
        "seed_perm": SEED_PERM,
        "seed_eco": SEED_ECO,
        "seed_boot": SEED_BOOT,
        "seed_mcdrop": SEED_MCDROP,
        "data_hashes": data["data_hashes"],
        "n_eval_seed_out": data["n"],
        "n_rare_seedout": int(data["rare_mask"].sum()),
        "classifier_config": {
            "class": "LogisticRegression",
            "max_iter": 2000,
            "class_weight": "balanced",
            "random_state": 42,
            "embedding_dtype": "float64",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(BMK, exist_ok=True)
    print("=" * 70, flush=True)
    print("pipeline_canonical.py  — single-source-of-truth run", flush=True)
    print(f"Output: {CANONICAL_PATH}", flush=True)
    print("=" * 70, flush=True)
    t_global = time.time()

    # Initialize canonical JSON
    _save("_meta", {
        "status": "running",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": __file__,
        "description": (
            "Canonical single-run output for the MiLT pipeline. "
            "All numbers derived from a single clf.predict_proba call (float64). "
            "Canonical band count, TV values, and p-value formula."
        ),
    })

    # ── Step 1: Load and fit ──
    data = load_and_fit()
    _save("classifier_info", {
        "n_classes": data["n_classes"],
        "classes": data["classes"],
        "n_eval_seed_out": data["n"],
        "n_rare_seedout": int(data["rare_mask"].sum()),
        "n_train": len(data["Z_train"]),
        "base_accuracy": round(data["base_acc"], 4),
        "embedding_dtype": str(data["Z_eval"].dtype),
    })

    # ── Step 2: Efficiency bands ──
    eff_bands = compute_efficiency_bands(data)
    _save("efficiency_bands", eff_bands)

    # ── Step 3: Triage curves ──
    triage = compute_triage_curves(data)
    # Don't persist the large index array in the main output
    ci_order_idx = np.array(triage.pop("confidence_order_index"))
    _save("triage_curves", triage)

    # ── Step 4: AL baselines ──
    al_result = compute_al_baselines(data)
    _save("al_baselines", al_result)

    # ── Step 5: Rare recall by band ──
    rare_band = compute_rare_recall_by_band(data)
    _save("rare_recall_by_band", rare_band)

    # ── Step 6: Ecology random control ──
    eco_ctrl = compute_ecology_random_control(data)
    _save("ecology_random_control", eco_ctrl)

    # ── Step 7: MC-Dropout triage ──
    mc_result = compute_mc_dropout_triage(data)
    _save("mc_dropout_triage", mc_result)

    # ── Step 8: 不明 exclusion sensitivity ──
    unk_sensitivity = compute_unknown_exclusion_sensitivity(data)
    _save("unknown_exclusion_sensitivity", unk_sensitivity)

    # ── Step 9: B≤30% partial AUC ──
    partial_auc = compute_partial_auc_b30(data, al_result)
    _save("partial_auc_b30", partial_auc)

    # ── Step 10: BADGE vs conf rare gap ──
    badge_gap = compute_badge_conf_rare_gap(data, al_result)
    _save("badge_vs_conf_rare_gap", badge_gap)

    # ── Environment manifest ──
    env = build_env_manifest(data)
    _save("env_log", env)
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)

    # ── Final metadata update ──
    elapsed = time.time() - t_global
    _save("_meta", {
        "status": "complete",
        "started": _state["_meta"].get("started", ""),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(elapsed, 1),
        "script": __file__,
        "description": _state["_meta"]["description"],
    })

    print("\n" + "=" * 70, flush=True)
    print(f"DONE in {elapsed:.1f}s", flush=True)
    print(f"Output: {CANONICAL_PATH}", flush=True)
    print(f"Env:    {ENV_PATH}", flush=True)
    print("=" * 70, flush=True)

    # Print summary for quick reference
    print("\n=== CANONICAL VALUES (update paper from these) ===", flush=True)
    print(f"n_eval_seed_out = {data['n']}", flush=True)
    print(f"n_rare_seedout  = {int(data['rare_mask'].sum())}", flush=True)
    print(f"base_accuracy   = {data['base_acc']:.4f}", flush=True)
    print("\n[Efficiency bands]", flush=True)
    for b in eff_bands:
        print(f"  {b['band']:<12} n={b['n']:>4} rate={b['rate']} CI={b['wilson95']}", flush=True)
    print("\n[Triage TV values (B=0,20,50) — canonical]", flush=True)
    for row in triage["triage_detail"]:
        if row["budget_pct"] in (0, 20, 50):
            print(f"  B={row['budget_pct']:>3}%  TV={row['tv_mean']}", flush=True)
    print("\n[AL baselines AUC_acc]", flush=True)
    for name, d in al_result["orders"].items():
        print(f"  {name:<18}  AUC_acc={d['auc_acc_budget']}", flush=True)
    perm = al_result["confidence_vs_random_perm"]
    print(f"\n[Permutation p-value] n_exceed={perm['n_exceed']} / {perm['n_shuffles']} → "
          f"p={perm['p_value']:.6f}", flush=True)


if __name__ == "__main__":
    main()
