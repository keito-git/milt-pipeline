#!/usr/bin/env python3
"""Analyse blinded audit results for the MiLT R1 revision.

Usage:
  ../.venv/bin/python analyze_blind_audit.py --key KEY --out OUT.json RESULT.json [RESULT2.json]

The analysis plan is fixed in ANALYSIS_PLAN.md, written before any answer exists.
The output file must not already exist.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BOOT = 2000
SEED = 20261101
BANDS = ["[0.0,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,1.0]"]
UNKNOWN = "不明"


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def kappa(a: list[str], b: list[str]) -> float | None:
    if not a:
        return None
    labels = set(a) | set(b)
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum(a.count(l) / len(a) * b.count(l) / len(b) for l in labels)
    return round(float((po - pe) / (1 - pe)), 4) if pe < 1 else 1.0


def cluster_ci(errors: list[bool], clusters: list[str]) -> list[float] | None:
    """Percentile interval resampling source videos, since crops of one video are correlated."""
    if not errors:
        return None
    groups = defaultdict(list)
    for e, c in zip(errors, clusters):
        groups[c].append(e)
    per_group = [(sum(v), len(v)) for v in groups.values()]
    rng = np.random.default_rng(SEED)
    k = np.array([p[0] for p in per_group])
    n = np.array([p[1] for p in per_group])
    idx = rng.integers(0, len(per_group), size=(BOOT, len(per_group)))
    rates = k[idx].sum(axis=1) / n[idx].sum(axis=1)
    return [round(float(np.percentile(rates, 2.5)), 4), round(float(np.percentile(rates, 97.5)), 4)]


def error_summary(rows: list[dict], reference: str) -> dict:
    """Auto-label or gold error against the blind label, under two treatments of 'unknown'."""
    decided = [r for r in rows if r["blind"] != UNKNOWN]
    errors = [r[reference] != r["blind"] for r in decided]
    k, n = sum(errors), len(decided)
    strict = sum(r[reference] != r["blind"] for r in rows)
    return {
        "n": len(rows),
        "n_blind_unknown": len(rows) - n,
        "excluding_unknown": {"errors": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": wilson(k, n),
                              "video_cluster_boot95": cluster_ci(errors, [r["src"] or r["crop"] for r in decided])},
        "counting_unknown_as_error": {"errors": strict, "n": len(rows),
                                      "rate": round(strict / len(rows), 4) if rows else None,
                                      "wilson95": wilson(strict, len(rows))},
    }


def analyse(body: dict, key: dict[str, dict]) -> dict:
    answered = {r["id"]: r for r in body["answers"] if r.get("label")}
    unknown_ids = set(answered) - set(key)
    if unknown_ids:
        sys.exit(f"Result holds ids missing from the key: {sorted(unknown_ids)[:5]}")
    rows, repeats = [], []
    by_crop: dict[str, dict] = {}
    for id_, k in key.items():
        if id_ in answered:
            row = {**k, "blind": answered[id_]["label"], "seconds": answered[id_].get("seconds")}
            (repeats if k["is_hidden_repeat"] else rows).append(row)
    for r in rows:
        by_crop[r["crop"]] = r
    pairs = [(by_crop[r["crop"]]["blind"], r["blind"]) for r in repeats if r["crop"] in by_crop]
    seconds = [r["seconds"] for r in rows + repeats if r["seconds"] is not None]

    out: dict = {"auditor": body["auditor"], "n_items": len(key), "n_answered": len(answered)}
    out["auto_label_by_band_random_only"] = {
        b: error_summary([r for r in rows if r["band"] == b and r["stratum"] in ("S1_all_high", "S2_random")], "auto_label")
        for b in BANDS}
    out["gold_error_by_band_random_only"] = {
        b: error_summary([r for r in rows if r["band"] == b and r["stratum"] in ("S1_all_high", "S2_random")], "gold")
        for b in BANDS}
    high = [r for r in rows if r["band"] == BANDS[3]]
    out["high_conf_common_vs_rare"] = {
        "auto_common": error_summary([r for r in high if not r["auto_rare"]], "auto_label"),
        "auto_rare": error_summary([r for r in high if r["auto_rare"]], "auto_label")}
    out["rare_all_bands_descriptive"] = {
        "gold_rare_recall_of_blind_rare": {
            b: error_summary([r for r in rows if r["band"] == b and r["gold_rare"]], "auto_label") for b in BANDS},
        "auto_rare_precision": {
            b: error_summary([r for r in rows if r["band"] == b and r["auto_rare"]], "auto_label") for b in BANDS}}
    out["blind_vs_gold_kappa_random_only"] = kappa(
        *map(list, zip(*[(r["blind"], r["gold"]) for r in rows if r["stratum"] in ("S1_all_high", "S2_random")]))
    ) if any(r["stratum"] in ("S1_all_high", "S2_random") for r in rows) else None
    out["intra_rater"] = {"n_pairs": len(pairs),
                          "agreement": round(float(np.mean([a == b for a, b in pairs])), 4) if pairs else None}
    out["seconds_per_item"] = {"median": float(np.median(seconds)), "total_hours": round(sum(seconds) / 3600, 2)} if seconds else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"Refusing to overwrite {args.out}")
    key = {r["id"]: r for r in json.loads(args.key.read_text(encoding="utf-8"))["items"]}
    bodies = [json.loads(p.read_text(encoding="utf-8")) for p in args.results]
    report: dict = {"per_auditor": [analyse(b, key) for b in bodies]}
    if len(bodies) == 2:
        a, b = [{r["id"]: r.get("label") for r in x["answers"] if r.get("label")} for x in bodies]
        ids = [i for i in key if i in a and i in b and not key[i]["is_hidden_repeat"]]
        report["between_auditors"] = {"n": len(ids), "kappa": kappa([a[i] for i in ids], [b[i] for i in ids]),
                                      "agreement": round(float(np.mean([a[i] == b[i] for i in ids])), 4) if ids else None}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
