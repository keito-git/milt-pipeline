#!/usr/bin/env python3
"""
rcta_bootstrap2000.py — Bootstrap N=2000 for RCTA_binary_alpha1.0 vs confidence.

Bootstrap rerun for rare_targeted_acquisition.json の真希少種n=173.
RCTA_binary_alpha1.0 vs confidence の bootstrap CI を N=2000(可能なら5000)で計算し、
CI境界の安定性を確認する。

比較: RCTA_binary_alpha1.0 vs confidence on true_rare_n173
データ: rare_targeted_acquisition.py と同一(eval set・classifier・train set)
出力: outputs/benchmark/rcta_bootstrap2000.json

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

RESULT_PATH = os.path.join(BMK, "rcta_bootstrap2000.json")
ORIG_PATH = os.path.join(BMK, "rare_targeted_acquisition.json")

# Safety: abort if output already exists
if os.path.exists(RESULT_PATH):
    print(f"[ERROR] Output already exists: {RESULT_PATH}", flush=True)
    print("  既存ファイルの上書き防止のため終了します。", flush=True)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants (IDENTICAL to rare_targeted_acquisition.py)
# ---------------------------------------------------------------------------
COMMON = {"ニホンジカ", "タヌキ", "イノシシ", "ニホンザル", "ニホンノウサギ"}
UNKNOWN_LABEL = "不明"
FALSE_DET_LABEL = "誤検出(動物なし)"
EXCLUDED_FROM_RARE_WEIGHT = COMMON | {UNKNOWN_LABEL, FALSE_DET_LABEL}

BUDGETS = [0, 5, 10, 20, 30, 50, 100]

# N_BOOT values to test (2000 main, also record 5000 for convergence check)
N_BOOT_LIST = [2_000, 5_000]
SEED_BOOT_2K = 9173   # distinct seed (original used 6906+counter; use new seed here)

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


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


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
    Identical logic to rare_targeted_acquisition.py.
    """
    xs = np.array([b / 100.0 for b in budgets])
    rank_a = np.empty(n, dtype=np.int64)
    rank_a[order_a] = np.arange(n)
    rank_b = np.empty(n, dtype=np.int64)
    rank_b[order_b] = np.arange(n)
    gaps = []
    for it in range(n_boot):
        if it % 500 == 0:
            print(f"    bootstrap iteration {it}/{n_boot}...", flush=True)
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
        "gaps_stats": {
            "mean": round(float(np.mean(gaps)), 6),
            "std": round(float(np.std(gaps)), 6),
            "p025": round(lo, 6),
            "p975": round(hi, 6),
            "p05": round(float(np.percentile(gaps, 5.0)), 6),
            "p95": round(float(np.percentile(gaps, 95.0)), 6),
        },
    }


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
            "Bootstrap N=2000(+5000) re-run for "
            "RCTA_binary_alpha1.0 vs confidence on true_rare_n173. "
            "Verifies CI stability vs original N=500."
        ),
        "gpu_rule": "CUDA_VISIBLE_DEVICES='' set before any torch import",
    }
    _save("_meta", {"env": env_log, "status": "running"})

    # ------------------------------------------------------------------
    # Step 1: Load data (IDENTICAL to rare_targeted_acquisition.py)
    # ------------------------------------------------------------------
    print("\n[Step 1] データ読み込み (rare_targeted_acquisition.py と完全同一)...", flush=True)
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

    # Training set (identical)
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
    Z_train = np.stack(X_tr).astype(np.float64)
    N_train = len(y_tr)

    # Classifier (identical)
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ).fit(Z_train, y_tr)
    classes: List[str] = list(clf.classes_)
    n_classes = len(classes)

    # Eval set
    crops = [
        it["crop"]
        for it in vit
        if it["crop"] not in seed_set and it["crop"] in final
    ]
    Z_eval = np.stack([EMB[c] for c in crops]).astype(np.float64)
    n = len(crops)
    true_labels: List[str] = [final[c] for c in crops]
    P = clf.predict_proba(Z_eval).astype(np.float64)
    pred_labels: List[str] = [classes[int(i)] for i in P.argmax(axis=1)]
    conf = P.max(axis=1).astype(np.float64)
    base_correct = np.array([pred_labels[i] == true_labels[i] for i in range(n)], dtype=bool)
    base_acc = float(base_correct.mean())

    # true_rare_mask (identical to rare_targeted_acquisition.py)
    true_rare_mask = np.array(
        [t not in COMMON and t != FALSE_DET_LABEL and t != UNKNOWN_LABEL
         for t in true_labels], dtype=bool
    )
    n_true_rare = int(true_rare_mask.sum())

    print(f"  n_eval={n}  n_train={N_train}  n_classes={n_classes}", flush=True)
    print(f"  base_acc={base_acc:.4f}  n_true_rare={n_true_rare}", flush=True)
    print(f"  Step 1 done in {time.time()-t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Step 2: RCTA binary weight (identical logic)
    # ------------------------------------------------------------------
    print("\n[Step 2] RCTA binary weight...", flush=True)
    train_cnt = Counter(y_tr)
    w = np.zeros(len(classes), dtype=np.float64)
    for i, c in enumerate(classes):
        if c not in EXCLUDED_FROM_RARE_WEIGHT:
            w[i] = 1.0
    max_w = float(w.max())
    if max_w > 0:
        w = w / max_w
    nonzero_cls = {c: round(float(w[i]), 4) for i, c in enumerate(classes) if w[i] > 0}
    print(f"  Nonzero classes: {list(nonzero_cls.keys())}", flush=True)

    # RCTA score (binary)
    rcta_score = (P @ w).astype(np.float64)

    # Uncertainty (1 - confidence)
    uncertainty = (1.0 - conf).astype(np.float64)

    # Orderings
    # RCTA(binary, alpha=1.0): pure RCTA, no uncertainty
    rcta_order = np.argsort(-minmax_normalize(rcta_score))
    # confidence (alpha=0.0): least-confident first
    conf_order = np.argsort(conf)

    # ------------------------------------------------------------------
    # Step 3: Cross-verify AUC values match original JSON
    # ------------------------------------------------------------------
    print("\n[Step 3] AUC cross-verification vs original JSON...", flush=True)

    def compute_auc_simple(
        order_idx: np.ndarray,
        base_correct: np.ndarray,
        n: int,
        budgets: List[int],
        rare_mask: np.ndarray,
    ) -> float:
        xs = np.array([b / 100.0 for b in budgets])
        rares = []
        for B in budgets:
            K = int(round(B / 100 * n))
            reviewed = set(order_idx[:K].tolist())
            correct = base_correct.copy()
            for i in reviewed:
                correct[i] = True
            n_rare = int(rare_mask.sum())
            rares.append(float(correct[rare_mask].mean()) if n_rare > 0 else float("nan"))
        fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        return round(float(fn(np.array(rares), xs)), 4)

    rcta_auc_check = compute_auc_simple(rcta_order, base_correct, n, BUDGETS, true_rare_mask)
    conf_auc_check = compute_auc_simple(conf_order, base_correct, n, BUDGETS, true_rare_mask)

    # Expected from rare_targeted_acquisition.json
    EXPECTED_RCTA_AUC = 0.8256
    EXPECTED_CONF_AUC = 0.7922
    EXPECTED_N_TRUE_RARE = 173

    auc_match_rcta = abs(rcta_auc_check - EXPECTED_RCTA_AUC) < 0.0001
    auc_match_conf = abs(conf_auc_check - EXPECTED_CONF_AUC) < 0.0001
    n_rare_match = (n_true_rare == EXPECTED_N_TRUE_RARE)

    print(f"  RCTA AUC: computed={rcta_auc_check}  expected={EXPECTED_RCTA_AUC}  match={auc_match_rcta}", flush=True)
    print(f"  conf AUC: computed={conf_auc_check}  expected={EXPECTED_CONF_AUC}  match={auc_match_conf}", flush=True)
    print(f"  n_true_rare: computed={n_true_rare}  expected={EXPECTED_N_TRUE_RARE}  match={n_rare_match}", flush=True)

    if not (auc_match_rcta and auc_match_conf and n_rare_match):
        msg = (
            f"[ERROR] AUC mismatch vs rare_targeted_acquisition.json. "
            f"RCTA: {rcta_auc_check} vs {EXPECTED_RCTA_AUC}. "
            f"conf: {conf_auc_check} vs {EXPECTED_CONF_AUC}. "
            f"n_rare: {n_true_rare} vs {EXPECTED_N_TRUE_RARE}. "
            "Data loading is NOT identical. Aborting."
        )
        print(msg, flush=True)
        _save("ERROR", msg)
        sys.exit(1)

    print("  [PASS] AUC values and n_true_rare match original JSON.", flush=True)

    cross_verify = {
        "rcta_auc_computed": rcta_auc_check,
        "rcta_auc_expected": EXPECTED_RCTA_AUC,
        "rcta_auc_match": auc_match_rcta,
        "conf_auc_computed": conf_auc_check,
        "conf_auc_expected": EXPECTED_CONF_AUC,
        "conf_auc_match": auc_match_conf,
        "n_true_rare_computed": n_true_rare,
        "n_true_rare_expected": EXPECTED_N_TRUE_RARE,
        "n_true_rare_match": n_rare_match,
        "status": "PASS",
    }
    _save("cross_verify", cross_verify)

    # ------------------------------------------------------------------
    # Step 4: Bootstrap CI at N=2000 and N=5000
    # ------------------------------------------------------------------
    results_by_n: Dict[str, Dict] = {}
    for n_boot in N_BOOT_LIST:
        print(f"\n[Step 4] Bootstrap CI N={n_boot}...", flush=True)
        t0 = time.time()
        rng_boot = np.random.default_rng(SEED_BOOT_2K + n_boot)
        ci = bootstrap_rare_auc_gap(
            rcta_order, conf_order,
            base_correct, true_rare_mask, BUDGETS, n,
            n_boot, rng_boot,
            label_a="RCTA_binary_alpha1.0",
            label_b="confidence",
        )
        elapsed_boot = round(time.time() - t0, 1)
        print(
            f"  N={n_boot}: mean_gap={ci['mean_gap']:+.4f}  "
            f"CI95={ci['ci95']}  sig={ci['significant_at_alpha05']}  ({elapsed_boot}s)",
            flush=True
        )
        results_by_n[f"N{n_boot}"] = {
            **ci,
            "elapsed_s": elapsed_boot,
        }

    _save("bootstrap_results_by_n", results_by_n)

    # ------------------------------------------------------------------
    # Step 5: Original N=500 reference from JSON (for comparison)
    # ------------------------------------------------------------------
    print("\n[Step 5] Original N=500 reference loading...", flush=True)
    original_n500 = None
    if os.path.exists(ORIG_PATH):
        try:
            orig = json.load(open(ORIG_PATH))
            orig_key = orig.get("statistical_tests_true_rare_n173", {}).get("RCTA_binary_alpha1.0", {})
            bs_orig = orig_key.get("bootstrap_vs_confidence", {})
            original_n500 = {
                "label_a": bs_orig.get("label_a"),
                "label_b": bs_orig.get("label_b"),
                "mean_gap": bs_orig.get("mean_gap"),
                "ci95": bs_orig.get("ci95"),
                "n_bootstrap": bs_orig.get("n_bootstrap"),
                "significant_at_alpha05": bs_orig.get("significant_at_alpha05"),
                "source": ORIG_PATH,
            }
            print(
                f"  Original N=500: mean_gap={original_n500['mean_gap']}  "
                f"CI95={original_n500['ci95']}  sig={original_n500['significant_at_alpha05']}",
                flush=True
            )
        except Exception as ex:
            print(f"  [WARN] Could not load original: {ex}", flush=True)

    _save("original_n500_reference", original_n500)

    # ------------------------------------------------------------------
    # Step 6: Summary + stability verdict
    # ------------------------------------------------------------------
    print("\n[Step 6] Summary + stability verdict...", flush=True)

    r2k = results_by_n["N2000"]
    r5k = results_by_n["N5000"]

    # CI stability check: lower bound of CI95 at N=2000
    ci_lo_2k = r2k["ci95"][0]
    ci_hi_2k = r2k["ci95"][1]
    ci_lo_5k = r5k["ci95"][0]
    ci_hi_5k = r5k["ci95"][1]

    stable_significant = bool(ci_lo_2k > 0 and ci_lo_5k > 0)

    if stable_significant:
        stability_verdict = (
            f"CI lower bound is >0 at both N=2000 ({ci_lo_2k:.4f}) and N=5000 ({ci_lo_5k:.4f}). "
            f"RCTA_binary_alpha1.0 vs confidence の有意差は安定して確認される。"
        )
    elif ci_lo_2k > 0 or ci_lo_5k > 0:
        stability_verdict = (
            f"CI lower bound: N=2000 → {ci_lo_2k:.4f}, N=5000 → {ci_lo_5k:.4f}. "
            f"いずれかのNで0を含む。安定性は限定的。"
        )
    else:
        stability_verdict = (
            f"CI lower bound: N=2000 → {ci_lo_2k:.4f}, N=5000 → {ci_lo_5k:.4f}. "
            f"両方で0を含む → N=500での有意は不安定だった可能性あり。"
        )

    print(f"  {stability_verdict}", flush=True)

    summary = {
        "purpose": "N=500 bootstrap CI stability check",
        "comparison": "RCTA_binary_alpha1.0 vs confidence on true_rare_n173",
        "n_eval": n,
        "n_true_rare": n_true_rare,
        "rcta_auc_n173": rcta_auc_check,
        "conf_auc_n173": conf_auc_check,
        "original_N500": original_n500,
        "N2000": {
            "mean_gap": r2k["mean_gap"],
            "ci95": r2k["ci95"],
            "significant_at_alpha05": r2k["significant_at_alpha05"],
        },
        "N5000": {
            "mean_gap": r5k["mean_gap"],
            "ci95": r5k["ci95"],
            "significant_at_alpha05": r5k["significant_at_alpha05"],
        },
        "ci_lower_bound_stable_above_0": stable_significant,
        "stability_verdict": stability_verdict,
        "data_hashes": {
            "embeddings.npy": file_sha256(os.path.join(OUT, "embeddings.npy")),
            "video_embeddings.npy": file_sha256(os.path.join(OUT, "video_embeddings.npy")),
            "seed_labels.json": file_sha256(os.path.join(OUT, "seed_labels.json")),
            "labels_video_v1.json": file_sha256(os.path.join(OUT, "labels_video_v1.json")),
            "labels_still_v1.json": file_sha256(os.path.join(OUT, "labels_still_v1.json")),
        },
    }
    _save("summary", summary)

    elapsed = round(time.time() - t_total, 1)
    _save("_meta", {
        "env": env_log,
        "status": "completed",
        "elapsed_s": elapsed,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    print(f"\n[Done] JSON saved: {RESULT_PATH}  ({elapsed}s)", flush=True)
    print(f"\n=== Final Summary ===", flush=True)
    print(f"  N=2000: mean_gap={r2k['mean_gap']:+.4f}  CI=[{r2k['ci95'][0]:.4f}, {r2k['ci95'][1]:.4f}]  sig={r2k['significant_at_alpha05']}", flush=True)
    print(f"  N=5000: mean_gap={r5k['mean_gap']:+.4f}  CI=[{r5k['ci95'][0]:.4f}, {r5k['ci95'][1]:.4f}]  sig={r5k['significant_at_alpha05']}", flush=True)
    print(f"  Stability: {stability_verdict}", flush=True)


if __name__ == "__main__":
    main()
