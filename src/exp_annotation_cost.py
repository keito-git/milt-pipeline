#!/usr/bin/env python3
"""
exp_annotation_cost.py — E2: Annotation labor / cost model.

Computes "how many expert-review crops are needed to reach target accuracy"
under three review strategies:
  (i)   Confidence ordering (least-confident first — MiLT protocol)
  (ii)  Random ordering
  (iii) Full review (all crops)

Input: pipeline_canonical.json (pre-computed; no new model training).

Method:
  1. For each target accuracy τ ∈ {0.80, 0.85, 0.90}, find the minimal review
     budget B* such that the accuracy-at-budget curve reaches τ.
     B* is interpolated linearly between the two adjacent budget points.
  2. Report B* (fraction of 6906), n_crops* (absolute count), and
     relative effort vs full review (n_crops* / 6906).
  3. Cost model: assuming each expert review costs t (arbitrary unit,
     value is a parameter — only ratios are meaningful).
     Report cost_conf / cost_fullreview and cost_conf / cost_random
     at each target τ.  Assumption: correction cost at high-conf band
     is already factored into B* (crops in B* are all reviewed once).
  4. Correction-adjusted model: an expert reviewing a high-conf crop
     (conf ≥ 0.9) saves very little time because the error rate is only
     1.49% (from efficiency_bands).  Model the "effective corrections"
     saved per reviewed crop in each ordering.

All numbers derived solely from pipeline_canonical.json.
No GPU, no new training, no new embedding computation.

Output: outputs/benchmark/annotation_cost_model.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HOME = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HOME, "outputs")
BMK = os.path.join(OUT, "benchmark")
CANONICAL_PATH = os.path.join(BMK, "pipeline_canonical.json")
RESULT_PATH = os.path.join(BMK, "annotation_cost_model.json")

# Safety: abort if output already exists
if os.path.exists(RESULT_PATH):
    print(f"[ERROR] Output already exists: {RESULT_PATH}")
    print("  既存ファイルの上書き防止のため終了します。")
    sys.exit(1)

os.makedirs(BMK, exist_ok=True)

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def interp_budget_for_target(
    budgets: List[int], accs: List[float], target: float, n_total: int
) -> Optional[Dict]:
    """
    Find minimal B* (percent) and n_crops* (absolute) to reach target accuracy.
    Uses linear interpolation between adjacent budget points.
    Returns None if target is never reached within 0–100%.
    """
    for i in range(len(budgets) - 1):
        if accs[i] <= target <= accs[i + 1]:
            b_lo, b_hi = budgets[i], budgets[i + 1]
            a_lo, a_hi = accs[i], accs[i + 1]
            if abs(a_hi - a_lo) < 1e-9:
                b_star_pct = float(b_lo)
            else:
                b_star_pct = b_lo + (target - a_lo) / (a_hi - a_lo) * (b_hi - b_lo)
            n_star = b_star_pct / 100.0 * n_total
            return {
                "budget_pct": round(b_star_pct, 2),
                "n_crops": round(n_star),
                "reduction_vs_full_review": round(1.0 - b_star_pct / 100.0, 4),
            }
    # Check if already met at B=0
    if accs[0] >= target:
        return {"budget_pct": 0.0, "n_crops": 0, "reduction_vs_full_review": 1.0}
    return None  # target not reached


def main() -> None:
    t_global = time.time()

    print("=" * 70, flush=True)
    print("exp_annotation_cost.py  — E2: Annotation labor / cost model", flush=True)
    print(f"Input:  {CANONICAL_PATH}", flush=True)
    print(f"Output: {RESULT_PATH}", flush=True)
    print("=" * 70, flush=True)

    # ------------------------------------------------------------------
    # Load canonical results
    # ------------------------------------------------------------------
    with open(CANONICAL_PATH) as f:
        canon = json.load(f)

    n_total: int = canon["al_baselines"]["n_eval_seed_out"]
    budgets: List[int] = canon["al_baselines"]["budgets_pct"]
    eff_bands: List[Dict] = canon["efficiency_bands"]

    conf_accs: List[float] = canon["al_baselines"]["orders"]["confidence"]["acc_by_budget"]
    rand_auc_mean: float   = canon["al_baselines"]["orders"]["random"]["auc_acc_budget"]
    rand_std: float        = canon["al_baselines"]["orders"]["random"]["auc_acc_budget_std"]
    rand_n_shuffles: int   = canon["al_baselines"]["orders"]["random"]["n_shuffles"]

    # For random baseline, we use the expected accuracy at each budget:
    # acc_random(B) ≈ base_acc + B/100 * (1 - base_acc)
    # (randomly selected B% are reviewed and become correct; rest stay at base_acc)
    # This is the expected value under random ordering.
    base_acc: float = canon["classifier_info"]["base_accuracy"]
    rand_accs: List[float] = [
        round(base_acc + (b / 100.0) * (1.0 - base_acc), 4) for b in budgets
    ]

    print(f"\n  n_total={n_total}, base_acc={base_acc}", flush=True)
    print(f"  Confidence accs: {conf_accs}", flush=True)
    print(f"  Random expected accs (E[acc(B)]): {rand_accs}", flush=True)
    print(f"  Budgets: {budgets}", flush=True)

    # ------------------------------------------------------------------
    # Target accuracy analysis
    # ------------------------------------------------------------------
    TARGET_ACCS = [0.80, 0.85, 0.90]
    STRATEGIES = {
        "confidence_order": conf_accs,
        "random_order": rand_accs,
    }

    print("\n[Budget-to-target accuracy]", flush=True)
    targets_result: Dict = {}
    for tau in TARGET_ACCS:
        tau_key = f"tau_{int(tau*100):02d}pct"
        targets_result[tau_key] = {
            "target_accuracy": tau,
            "full_review": {
                "budget_pct": 100.0,
                "n_crops": n_total,
                "reduction_vs_full_review": 0.0,
                "note": "Baseline: review every crop",
            },
        }
        for strategy, accs in STRATEGIES.items():
            interp = interp_budget_for_target(budgets, accs, tau, n_total)
            targets_result[tau_key][strategy] = interp
            if interp:
                print(f"  τ={tau:.2f}  {strategy:<25} B*={interp['budget_pct']:.2f}%  "
                      f"n*={interp['n_crops']:>5}  reduction={interp['reduction_vs_full_review']:.3f}", flush=True)
            else:
                print(f"  τ={tau:.2f}  {strategy:<25} NOT REACHABLE within budget grid", flush=True)

    # ------------------------------------------------------------------
    # Relative cost model (cost per reviewed crop = t, constant)
    # ------------------------------------------------------------------
    print("\n[Relative cost model (t = unit cost per reviewed crop)]", flush=True)
    cost_model: Dict = {
        "assumption": (
            "Each reviewed crop costs t (arbitrary unit). "
            "t is a free parameter; only RATIOS are meaningful. "
            "Full review = t * n_total. "
            "Confidence order at B* = t * n_crops*. "
            "Random order at B* = t * n_crops_random*."
        ),
        "by_target": {},
    }
    for tau in TARGET_ACCS:
        tau_key = f"tau_{int(tau*100):02d}pct"
        conf_r = targets_result[tau_key].get("confidence_order")
        rand_r = targets_result[tau_key].get("random_order")

        entry: Dict = {"target_accuracy": tau}
        if conf_r:
            entry["cost_confidence_order_per_fullreview"] = round(
                conf_r["n_crops"] / n_total, 4
            )
            if rand_r:
                entry["cost_confidence_per_random"] = round(
                    conf_r["n_crops"] / max(rand_r["n_crops"], 1), 4
                )
                entry["crops_saved_vs_random"] = rand_r["n_crops"] - conf_r["n_crops"]
        cost_model["by_target"][tau_key] = entry
        print(f"  τ={tau:.2f}  cost_conf/full={entry.get('cost_confidence_order_per_fullreview')}  "
              f"cost_conf/rand={entry.get('cost_confidence_per_random')}  "
              f"crops_saved_vs_rand={entry.get('crops_saved_vs_random')}", flush=True)

    # ------------------------------------------------------------------
    # Correction-adjusted efficiency (from efficiency_bands)
    # ------------------------------------------------------------------
    # The fraction of crops in each confidence band that would need correction
    # (i.e., are auto-labeled wrong) — this is the "savings" of not reviewing them.
    print("\n[Correction-rate per confidence band]", flush=True)
    corr_model: Dict = {
        "description": (
            "If you SKIP reviewing a crop, the auto-label is used. "
            "'Correction rate' = fraction of skipped crops that would need human correction. "
            "Confidence ordering prioritizes LOW-confidence crops (high correction rate) first, "
            "so the expert budget is efficiently targeted at crops likely to be wrong."
        ),
        "bands": [],
    }
    total_n = sum(b["n"] for b in eff_bands)
    total_corrections_if_all_skipped = sum(b["corrected"] for b in eff_bands)
    for b in eff_bands:
        entry = {
            "band": b["band"],
            "n_crops": b["n"],
            "pct_of_total": round(b["n"] / n_total * 100, 2),
            "auto_correction_rate": b["rate"],
            "wilson95_correction_rate": b["wilson95"],
            "expected_corrections_if_skipped": b["corrected"],
        }
        corr_model["bands"].append(entry)
        print(f"  {b['band']:<12} n={b['n']:>5} ({b['n']/n_total*100:.1f}%)  "
              f"correction_rate={b['rate']}  CI={b['wilson95']}", flush=True)

    # Expected corrections per crop reviewed (marginal benefit of confidence ordering)
    # At budget B=30%: the 30% reviewed by confidence order are the least confident
    # → they come from the low-conf bands where correction rate is highest.
    # Compute expected correction rate of the B=30% reviewed under confidence ordering.
    # Under confidence ordering, we pick the lowest-confidence crops first.
    # Approximate: the 30% of crops reviewed include ~all of [0,0.5) band + partial [0.5,0.7)
    b_30_conf_n = int(round(30 / 100 * n_total))   # 2072 crops
    bands_sorted = sorted(eff_bands, key=lambda x: float(x["band"].split(",")[0].replace("[","").replace("(", "")))

    # Fill up the 30% budget from lowest-confidence band first
    remaining = b_30_conf_n
    expected_corrections_conf_B30 = 0
    conf_band_fill: List[Dict] = []
    for b in bands_sorted:
        take = min(remaining, b["n"])
        expected_corrections = round(take * (b["rate"] or 0.0))
        conf_band_fill.append({
            "band": b["band"],
            "reviewed": take,
            "expected_corrections": expected_corrections,
            "correction_rate": b["rate"],
        })
        expected_corrections_conf_B30 += expected_corrections
        remaining -= take
        if remaining <= 0:
            break

    # Under random ordering at B=30%: proportion of each band = its share of total
    expected_corrections_rand_B30 = 0
    rand_band_fill: List[Dict] = []
    for b in bands_sorted:
        take = round(b_30_conf_n * b["n"] / n_total)
        expected_corrections = round(take * (b["rate"] or 0.0))
        rand_band_fill.append({
            "band": b["band"],
            "reviewed": take,
            "expected_corrections": expected_corrections,
            "correction_rate": b["rate"],
        })
        expected_corrections_rand_B30 += expected_corrections

    correction_efficiency: Dict = {
        "budget_B30_n_crops": b_30_conf_n,
        "confidence_order_B30": {
            "expected_corrections_found": expected_corrections_conf_B30,
            "band_breakdown": conf_band_fill,
            "corrections_per_crop_reviewed": round(
                expected_corrections_conf_B30 / b_30_conf_n, 4),
        },
        "random_order_B30": {
            "expected_corrections_found": expected_corrections_rand_B30,
            "band_breakdown": rand_band_fill,
            "corrections_per_crop_reviewed": round(
                expected_corrections_rand_B30 / b_30_conf_n, 4),
        },
        "ratio_corrections_conf_vs_random": round(
            expected_corrections_conf_B30 / max(expected_corrections_rand_B30, 1), 4),
        "note": (
            "Approximation: assumes crops within each confidence band are uniformly "
            "distributed in the confidence ordering; confidence ordering fills bands "
            "from [0,0.5) upward. "
            "Actual ratio will differ slightly from interpolation artifacts."
        ),
    }
    print(f"\n  B=30%: conf_order corrections={expected_corrections_conf_B30}  "
          f"random corrections={expected_corrections_rand_B30}  "
          f"ratio={correction_efficiency['ratio_corrections_conf_vs_random']}", flush=True)

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    result: Dict = {
        "_meta": {
            "status": "complete",
            "script": __file__,
            "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "input_canonical": CANONICAL_PATH,
            "elapsed_s": round(time.time() - t_global, 2),
            "description": (
                "E2: Annotation labor/cost model derived from pipeline_canonical.json. "
                "No new training or embedding computation."
            ),
        },
        "n_total_eval": n_total,
        "base_accuracy": base_acc,
        "budgets_pct": budgets,
        "confidence_accs_by_budget": conf_accs,
        "random_expected_accs_by_budget": rand_accs,
        "random_auc_observed_mean": rand_auc_mean,
        "random_auc_observed_std": rand_std,
        "random_auc_n_shuffles": rand_n_shuffles,
        "targets": targets_result,
        "cost_model": cost_model,
        "correction_rate_by_band": corr_model,
        "correction_efficiency_at_B30": correction_efficiency,
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total_time = time.time() - t_global
    print(f"\nDONE in {total_time:.2f}s", flush=True)
    print(f"Output: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
