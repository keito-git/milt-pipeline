#!/usr/bin/env python3
"""
rcta_mie_secondsite.py — RCTA(binary, alpha=1.0) on Mie second site.

RCTA second-site evaluation: triage_mie_secondsite.json にはconfidence triageのみで
RCTAの第2サイト評価が無い。本スクリプトでRCTA(binary, alpha=1.0,
パラメータフリー)を三重静止画(n=4736, 訓練未使用サイト)に適用する。

評価対象:
  - 真希少種: COMMON, 誤検出(動物なし), 不明 を除いた同定済み希少種
  - 比較: RCTA(binary,alpha=1.0) vs confidence vs random
  - 統計: bootstrap CI (N=2000) + 順列検定

Honest note:
  seed-only LRは11クラス(コジュケイ・アライグマ・ニホンアナグマ・ハクビシンのみ
  がRCTA nonzero)。三重の真希少種の多くはこのモデルが予測できないため
  (トラツグミn=33, キジバトn=42, キツネn=8等は未学習クラス)、RCTAが
  confidenceを下回る可能性がある。結果はそのまま報告する。

GPU rule: CUDA_VISIBLE_DEVICES="" set before any torch import.
Output (new file only, existing JSONs not overwritten):
  outputs/benchmark/rcta_mie_secondsite.json
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
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HOME = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HOME, "outputs")
BMK = os.path.join(OUT, "benchmark")

RESULT_PATH = os.path.join(BMK, "rcta_mie_secondsite.json")

# Safety: abort if output already exists
if os.path.exists(RESULT_PATH):
    print(f"[ERROR] Output already exists: {RESULT_PATH}", flush=True)
    print("  既存ファイルの上書き防止のため終了します。", flush=True)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants (identical to rare_targeted_acquisition.py)
# ---------------------------------------------------------------------------
COMMON = {"ニホンジカ", "タヌキ", "イノシシ", "ニホンザル", "ニホンノウサギ"}
UNKNOWN_LABEL = "不明"
FALSE_DET_LABEL = "誤検出(動物なし)"
EXCLUDED_FROM_RARE_WEIGHT = COMMON | {UNKNOWN_LABEL, FALSE_DET_LABEL}

BUDGETS = [0, 5, 10, 20, 30, 50, 100]
BUDGETS_PARTIAL = [0, 5, 10, 20, 30]
CAPTURE_BUDGETS = [10, 20, 30]

SEED_PERM = 6906
SEED_BOOT = 7412  # distinct seed for this script
N_PERM = 10_000
N_BOOT = 2_000   # bootstrap iterations
N_RAND = 2_000

# ---------------------------------------------------------------------------
# State
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
# Utility functions (verbatim from rare_targeted_acquisition.py)
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


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


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


def compute_rcta_binary_weights(
    classes: List[str],
    excluded: set,
) -> np.ndarray:
    """
    Binary RCTA weight: w_c = 1 if class not in excluded, else 0.
    Max-normalized (max(w) = 1).
    Derived purely from class list (training labels only).
    No eval ground-truth used.
    """
    w = np.zeros(len(classes), dtype=np.float64)
    for i, c in enumerate(classes):
        if c not in excluded:
            w[i] = 1.0
    max_w = float(w.max())
    if max_w > 0:
        w = w / max_w
    return w


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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
        "purpose": (
            "RCTA(binary,alpha=1.0) on Mie second site (n=4736). "
            "seed-only LR (Wakayama+Shizuoka video, training-unused site)."
        ),
        "gpu_rule": "CUDA_VISIBLE_DEVICES='' set before any torch import",
    }
    _save("_meta", {"env": env_log, "status": "running"})

    # ------------------------------------------------------------------
    # Step 1: Load seed-only training data (identical to triage_mie_secondsite.json)
    # ------------------------------------------------------------------
    print("\n[Step 1] seed-only training data 読み込み...", flush=True)
    t0 = time.time()

    ve = np.load(os.path.join(OUT, "video_embeddings.npy")).astype(np.float64)
    vit = json.load(open(os.path.join(OUT, "video_embed_index.json")))
    seed_lab = {
        r["crop"]: r["label"]
        for r in json.load(open(os.path.join(OUT, "seed_labels.json")))["labels"]
    }
    seed_set = set(json.load(open(os.path.join(OUT, "seed_crops.json"))))

    # Video crop → embedding map
    video_EMB: Dict[str, np.ndarray] = {}
    for i, it in enumerate(vit):
        video_EMB[it["crop"]] = ve[i]

    # Seed-only training set (video crops in seed_crops.json with labels)
    X_tr_seed: List[np.ndarray] = []
    y_tr_seed: List[str] = []
    for c, lbl in seed_lab.items():
        if c in video_EMB:
            X_tr_seed.append(video_EMB[c])
            y_tr_seed.append(lbl)

    Z_train_seed = np.stack(X_tr_seed).astype(np.float64)  # (n_seed, 512)
    train_cnt = Counter(y_tr_seed)
    n_train = len(y_tr_seed)
    print(f"  Seed-only train: n={n_train}, n_classes={len(set(y_tr_seed))}", flush=True)
    print(f"  Train counts: {dict(sorted(train_cnt.items(), key=lambda x: -x[1]))}", flush=True)

    # Seed-only LR classifier (11 classes)
    clf_seed = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ).fit(Z_train_seed, y_tr_seed)
    classes_seed: List[str] = list(clf_seed.classes_)
    n_cls_seed = len(classes_seed)
    print(f"  Seed-only LR classes ({n_cls_seed}): {classes_seed}", flush=True)
    print(f"  Step 1 done in {time.time()-t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Step 2: Load Mie stills eval set
    # ------------------------------------------------------------------
    print("\n[Step 2] Mie stills eval データ読み込み...", flush=True)
    t0 = time.time()

    se = np.load(os.path.join(OUT, "embeddings.npy")).astype(np.float64)
    prov = json.load(open(os.path.join(OUT, "provisional_labels.json")))
    still_labels_raw = json.load(open(os.path.join(OUT, "labels_still_v1.json")))
    still_labels_dict = {r["crop"]: r["label"] for r in still_labels_raw["labels"]}

    # provisional_labels.json is the index for embeddings.npy
    prov_EMB: Dict[str, np.ndarray] = {}
    for i, r in enumerate(prov):
        prov_EMB[r["crop"]] = se[i]

    # Mie eval: all stills in labels_still_v1.json with embeddings
    mie_crops = [
        r["crop"]
        for r in still_labels_raw["labels"]
        if r["crop"] in prov_EMB
    ]
    Z_eval_mie = np.stack([prov_EMB[c] for c in mie_crops]).astype(np.float64)
    true_mie: List[str] = [still_labels_dict[c] for c in mie_crops]
    n_mie = len(mie_crops)
    print(f"  Mie eval: n={n_mie}", flush=True)

    # Label distribution
    mie_label_cnt = Counter(true_mie)
    print(f"  Mie label distribution: {dict(sorted(mie_label_cnt.items(), key=lambda x: -x[1]))}", flush=True)

    # ------------------------------------------------------------------
    # Step 3: Predict with seed-only LR
    # ------------------------------------------------------------------
    print("\n[Step 3] seed-only LR で三重静止画を予測...", flush=True)
    P_mie = clf_seed.predict_proba(Z_eval_mie).astype(np.float64)  # (n_mie, 11)
    pred_mie: List[str] = [classes_seed[int(i)] for i in P_mie.argmax(axis=1)]
    conf_mie = P_mie.max(axis=1).astype(np.float64)
    base_correct_mie = np.array(
        [pred_mie[i] == true_mie[i] for i in range(n_mie)], dtype=bool
    )
    base_acc_mie = float(base_correct_mie.mean())

    # Masks
    # true_rare_mie: verified (identified) rare species = not COMMON, not 誤検出, not 不明
    true_rare_mask_mie = np.array(
        [t not in COMMON and t != FALSE_DET_LABEL and t != UNKNOWN_LABEL
         for t in true_mie], dtype=bool
    )
    # rare_all_mie (including 不明)
    rare_all_mask_mie = np.array(
        [t not in COMMON and t != FALSE_DET_LABEL for t in true_mie], dtype=bool
    )

    n_true_rare_mie = int(true_rare_mask_mie.sum())
    n_rare_all_mie = int(rare_all_mask_mie.sum())
    n_unknown_mie = int(sum(t == UNKNOWN_LABEL for t in true_mie))

    print(f"  base_acc_mie={base_acc_mie:.4f}", flush=True)
    print(f"  n_true_rare_mie (identified)={n_true_rare_mie}", flush=True)
    print(f"  n_rare_all_mie (incl. 不明)={n_rare_all_mie}", flush=True)
    print(f"  n_unknown_mie={n_unknown_mie}", flush=True)

    # Classes missing from seed model (cannot be predicted → always wrong)
    classes_missing = sorted(set(true_mie) - set(classes_seed))
    n_unpredictable = int(sum(t in classes_missing for t in true_mie))
    print(f"  Missing classes: {classes_missing}", flush=True)
    print(f"  Unpredictable samples: {n_unpredictable}/{n_mie}", flush=True)

    # True rare breakdown at Mie
    true_rare_label_cnt_mie = Counter(t for t in true_mie if true_rare_mask_mie[true_mie.index(t)])
    # More robust: use mask
    true_rare_label_cnt_mie = Counter(
        true_mie[i] for i in range(n_mie) if true_rare_mask_mie[i]
    )
    print(f"  True rare label counts at Mie: {dict(sorted(true_rare_label_cnt_mie.items(), key=lambda x: -x[1]))}", flush=True)

    # ------------------------------------------------------------------
    # Step 4: RCTA binary weight vector (from seed-only training, no eval gt)
    # ------------------------------------------------------------------
    print("\n[Step 4] RCTA binary weight (seed-only training classes)...", flush=True)
    w_binary = compute_rcta_binary_weights(classes_seed, EXCLUDED_FROM_RARE_WEIGHT)
    per_class_w = {c: round(float(w_binary[i]), 4) for i, c in enumerate(classes_seed)}
    nonzero_classes = {c: v for c, v in per_class_w.items() if v > 0}
    zero_classes = [c for c, v in per_class_w.items() if v == 0]
    print(f"  Nonzero classes: {nonzero_classes}", flush=True)
    print(f"  Zero classes: {zero_classes}", flush=True)
    print(
        f"  NOTE: Many true rare species at Mie (トラツグミ,キジバト,キツネ,テン,ネコ等) "
        f"are NOT in seed model → RCTA mass goes to {list(nonzero_classes.keys())} only.",
        flush=True
    )

    # RCTA score (binary, alpha=1.0 = pure RCTA)
    rcta_score_mie = (P_mie @ w_binary).astype(np.float64)
    print(
        f"  RCTA score stats: min={rcta_score_mie.min():.4f}  "
        f"mean={rcta_score_mie.mean():.4f}  max={rcta_score_mie.max():.4f}",
        flush=True
    )

    # Orderings
    # RCTA(binary, alpha=1.0): highest RCTA score reviewed first
    rcta_order_mie = np.argsort(-rcta_score_mie)
    # confidence: least-confident first
    conf_order_mie = np.argsort(conf_mie)

    weight_info = {
        "weight_type": "binary",
        "alpha": 1.0,
        "classes_in_seed_model": classes_seed,
        "nonzero_classes": nonzero_classes,
        "zero_classes": zero_classes,
        "note": (
            "Binary weight: w_c=1 for classes not in COMMON|{不明,誤検出(動物なし)}, else 0. "
            "Max-normalized. Derived from training class list only. No eval gt used."
        ),
    }

    # ------------------------------------------------------------------
    # Step 5: Evaluate on true_rare_mie and rare_all_mie
    # ------------------------------------------------------------------
    print("\n[Step 5] 評価: AUC, 捕捉率...", flush=True)

    def evaluate_ordering(order: np.ndarray, mask: np.ndarray) -> Dict:
        _, auc_acc, _, auc_rare = compute_auc_from_order(
            order, base_correct_mie, n_mie, BUDGETS, mask
        )
        _, auc_acc_b30, _, auc_rare_b30 = compute_auc_from_order(
            order, base_correct_mie, n_mie, BUDGETS_PARTIAL, mask
        )
        _, rares_by_budget = compute_auc_from_order(
            order, base_correct_mie, n_mie, BUDGETS, mask
        )[2], compute_auc_from_order(
            order, base_correct_mie, n_mie, BUDGETS, mask
        )[2]
        cap = compute_capture_rate(order, n_mie, mask, CAPTURE_BUDGETS)
        return {
            "auc_rare_recall": auc_rare,
            "auc_acc": auc_acc,
            "auc_rare_recall_b30": auc_rare_b30,
            "auc_acc_b30": auc_acc_b30,
            "capture_B10pct": cap[10],
            "capture_B20pct": cap[20],
            "capture_B30pct": cap[30],
        }

    masks = {
        "true_rare_mie": true_rare_mask_mie,
        "rare_all_mie": rare_all_mask_mie,
    }

    all_results: Dict[str, Dict] = {}
    for mask_name, mask in masks.items():
        all_results[mask_name] = {}
        for method_name, order in [("RCTA_binary_alpha1.0", rcta_order_mie),
                                    ("confidence", conf_order_mie)]:
            r = evaluate_ordering(order, mask)
            all_results[mask_name][method_name] = r
            print(
                f"  [{mask_name}] {method_name:<30} AUC={r['auc_rare_recall']:.4f}  "
                f"B10={r['capture_B10pct']:.4f}  B20={r['capture_B20pct']:.4f}  B30={r['capture_B30pct']:.4f}",
                flush=True
            )

    # ------------------------------------------------------------------
    # Step 6: Random baselines
    # ------------------------------------------------------------------
    print("\n[Step 6] Random baseline (2k shuffles)...", flush=True)
    random_baselines: Dict[str, Dict] = {}
    for mask_name, mask in masks.items():
        rng_rand = np.random.default_rng(SEED_PERM + hash(mask_name) % 10000)
        aucs_rand = fast_rand_rare_aucs(N_RAND, base_correct_mie, mask, BUDGETS, n_mie, rng_rand)
        aucs_rand_b30 = fast_rand_rare_aucs(N_RAND, base_correct_mie, mask, BUDGETS_PARTIAL, n_mie, rng_rand)
        random_baselines[mask_name] = {
            "auc_rare_mean": round(float(np.mean(aucs_rand)), 4),
            "auc_rare_std": round(float(np.std(aucs_rand)), 4),
            "auc_rare_b30_mean": round(float(np.mean(aucs_rand_b30)), 4),
            "auc_rare_b30_std": round(float(np.std(aucs_rand_b30)), 4),
            "n_shuffles": N_RAND,
        }
        print(
            f"  [{mask_name}] random AUC={random_baselines[mask_name]['auc_rare_mean']:.4f}"
            f"±{random_baselines[mask_name]['auc_rare_std']:.4f}",
            flush=True
        )

    # ------------------------------------------------------------------
    # Step 7: Statistical tests (true_rare_mie フォーカス)
    # ------------------------------------------------------------------
    print("\n[Step 7] 統計検定 (true_rare_mie)...", flush=True)
    mask = true_rare_mask_mie
    rcta_auc = all_results["true_rare_mie"]["RCTA_binary_alpha1.0"]["auc_rare_recall"]
    conf_auc = all_results["true_rare_mie"]["confidence"]["auc_rare_recall"]

    # Permutation test: RCTA vs random
    rng_perm = np.random.default_rng(SEED_PERM)
    rand_aucs_perm = fast_rand_rare_aucs(N_PERM, base_correct_mie, mask, BUDGETS, n_mie, rng_perm)
    n_exceed_rcta = int(np.sum(rand_aucs_perm >= rcta_auc))
    pval_rcta = (n_exceed_rcta + 1) / (N_PERM + 1)
    n_exceed_conf = int(np.sum(rand_aucs_perm >= conf_auc))
    pval_conf = (n_exceed_conf + 1) / (N_PERM + 1)

    print(f"  RCTA: perm p={pval_rcta:.6f}  (n_exceed={n_exceed_rcta})", flush=True)
    print(f"  conf: perm p={pval_conf:.6f}  (n_exceed={n_exceed_conf})", flush=True)

    # Bootstrap CI: RCTA vs confidence (N=2000)
    print(f"  Bootstrap CI RCTA vs confidence (N={N_BOOT})...", flush=True)
    t0 = time.time()
    rng_boot = np.random.default_rng(SEED_BOOT)
    ci_rcta_vs_conf = bootstrap_rare_auc_gap(
        rcta_order_mie, conf_order_mie,
        base_correct_mie, mask, BUDGETS, n_mie, N_BOOT, rng_boot,
        label_a="RCTA_binary_alpha1.0",
        label_b="confidence",
    )
    print(f"  gap={ci_rcta_vs_conf['mean_gap']:+.4f}  CI={ci_rcta_vs_conf['ci95']}  "
          f"sig={ci_rcta_vs_conf['significant_at_alpha05']}  ({time.time()-t0:.1f}s)", flush=True)

    stats_results = {
        "primary_focus": "true_rare_mie",
        "RCTA_binary_alpha1.0": {
            "rcta_auc": rcta_auc,
            "permutation_test_vs_random": {
                "n_perm": N_PERM,
                "n_exceed": n_exceed_rcta,
                "p_value": round(pval_rcta, 6),
                "formula": "(n_exceed+1)/(N_PERM+1) [Phipson & Smyth 2010]",
                "significant_at_alpha05": bool(pval_rcta < 0.05),
            },
            "bootstrap_vs_confidence": ci_rcta_vs_conf,
        },
        "confidence": {
            "conf_auc": conf_auc,
            "permutation_test_vs_random": {
                "n_perm": N_PERM,
                "n_exceed": n_exceed_conf,
                "p_value": round(pval_conf, 6),
                "formula": "(n_exceed+1)/(N_PERM+1) [Phipson & Smyth 2010]",
                "significant_at_alpha05": bool(pval_conf < 0.05),
            },
        },
    }

    # ------------------------------------------------------------------
    # Step 8: Summary + honest verdict
    # ------------------------------------------------------------------
    print("\n[Step 8] サマリー...", flush=True)
    rcta_beats_conf = bool(rcta_auc > conf_auc)
    rcta_sig = ci_rcta_vs_conf["significant_at_alpha05"]
    gap = round(rcta_auc - conf_auc, 4)

    if rcta_beats_conf and rcta_sig:
        verdict = (
            f"RCTA(binary,alpha=1.0) AUC={rcta_auc:.4f} > confidence AUC={conf_auc:.4f} "
            f"(gap={gap:+.4f}) — 第2サイトでも有意に上回る。"
            f"CI={ci_rcta_vs_conf['ci95']} (N={N_BOOT}), p<0.05。"
        )
        verdict_level = "POSITIVE"
    elif rcta_beats_conf and not rcta_sig:
        verdict = (
            f"RCTA(binary,alpha=1.0) AUC={rcta_auc:.4f} > confidence AUC={conf_auc:.4f} "
            f"(gap={gap:+.4f}) だが統計的に非有意。"
            f"CI={ci_rcta_vs_conf['ci95']} (N={N_BOOT})。"
        )
        verdict_level = "MARGINAL"
    else:
        verdict = (
            f"RCTA(binary,alpha=1.0) AUC={rcta_auc:.4f} <= confidence AUC={conf_auc:.4f} "
            f"(gap={gap:+.4f}) — 第2サイトではRCTAがconfidenceを下回る。"
            f"CI={ci_rcta_vs_conf['ci95']} (N={N_BOOT})。"
            f"\n  Honest reason: seed-only LRは三重の真希少種(トラツグミ{true_rare_label_cnt_mie.get('トラツグミ',0)}件, "
            f"キジバト{true_rare_label_cnt_mie.get('キジバト',0)}件, "
            f"キツネ{true_rare_label_cnt_mie.get('キツネ',0)}件等)を未学習。"
            f"RCTAは{list(nonzero_classes.keys())}の確率質量でしかスコアリングできない。"
        )
        verdict_level = "NEGATIVE"

    print(f"\n  === Honest Verdict ===", flush=True)
    print(f"  Level: {verdict_level}", flush=True)
    print(f"  {verdict}", flush=True)

    summary = {
        "site": "Mie stills (三重県静止画, training-unused second site)",
        "n_eval": n_mie,
        "n_true_rare": n_true_rare_mie,
        "n_rare_all": n_rare_all_mie,
        "n_unknown": n_unknown_mie,
        "true_rare_label_counts_mie": dict(sorted(true_rare_label_cnt_mie.items(), key=lambda x: -x[1])),
        "model": "seed-only LR (Wakayama+Shizuoka video crops, n_train=1000, 11 classes)",
        "classes_seed_model": classes_seed,
        "classes_missing_from_seed_model": classes_missing,
        "n_unpredictable_samples": n_unpredictable,
        "rcta_method": "RCTA_binary_alpha1.0",
        "rcta_nonzero_classes_in_seed_model": list(nonzero_classes.keys()),
        "true_rare_mie": {
            "RCTA_binary_alpha1.0": all_results["true_rare_mie"]["RCTA_binary_alpha1.0"],
            "confidence": all_results["true_rare_mie"]["confidence"],
            "random": {
                "auc_rare_mean": random_baselines["true_rare_mie"]["auc_rare_mean"],
                "auc_rare_std": random_baselines["true_rare_mie"]["auc_rare_std"],
            },
        },
        "honest_verdict_level": verdict_level,
        "honest_verdict": verdict,
        "rcta_beats_confidence": rcta_beats_conf,
        "rcta_significant_vs_confidence": rcta_sig,
        "gap_rcta_minus_conf": gap,
    }

    # ------------------------------------------------------------------
    # Save all results to JSON
    # ------------------------------------------------------------------
    data_info = {
        "n_eval_mie": n_mie,
        "n_train_seed": n_train,
        "n_classes_seed_model": n_cls_seed,
        "classes_seed_model": classes_seed,
        "base_acc_mie": round(base_acc_mie, 4),
        "n_true_rare_mie": n_true_rare_mie,
        "n_rare_all_mie": n_rare_all_mie,
        "n_unknown_mie": n_unknown_mie,
        "label_distribution_mie": dict(sorted(mie_label_cnt.items(), key=lambda x: -x[1])),
        "true_rare_label_counts_mie": dict(sorted(true_rare_label_cnt_mie.items(), key=lambda x: -x[1])),
        "classes_missing_from_seed_model": classes_missing,
        "n_unpredictable_samples": n_unpredictable,
        "true_rare_mask_definition": (
            "not in COMMON={ニホンジカ,タヌキ,イノシシ,ニホンザル,ニホンノウサギ} "
            "and != 誤検出(動物なし) and != 不明 → identified rare species only"
        ),
        "data_hashes": {
            "embeddings.npy": file_sha256(os.path.join(OUT, "embeddings.npy")),
            "video_embeddings.npy": file_sha256(os.path.join(OUT, "video_embeddings.npy")),
            "seed_labels.json": file_sha256(os.path.join(OUT, "seed_labels.json")),
            "labels_still_v1.json": file_sha256(os.path.join(OUT, "labels_still_v1.json")),
        },
    }

    _save("data_info", data_info)
    _save("weight_info", weight_info)
    _save("all_results", all_results)
    _save("random_baselines", random_baselines)
    _save("statistical_tests", stats_results)
    _save("summary", summary)

    elapsed = round(time.time() - t_total, 1)
    _save("_meta", {
        "env": env_log,
        "status": "completed",
        "elapsed_s": elapsed,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    print(f"\n[Done] JSON saved: {RESULT_PATH}  ({elapsed}s)", flush=True)


if __name__ == "__main__":
    main()
