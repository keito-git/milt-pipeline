#!/usr/bin/env python3
"""Build the blinded audit package for the MiLT R1 revision (APIN-D-26-11064).

Sampling plan (fixed before any audit answer exists):
  S1  all crops with auto-label confidence >= 0.9
  S2  simple random sample of 80 crops from each lower confidence band
  S3  rare supplement per lower band: up to 20 identified rare crops by gold label
      and up to 20 crops auto-labelled as rare, excluding crops already selected
  S4  30 hidden repeats drawn from S1-S3 for intra-rater agreement

The shared package holds only anonymous file names and images. Everything that
could reveal model output, gold labels, source video or sampling stratum is written
to the private key, which must never be sent to auditors.
Outputs are created in new directories; source data are read-only.
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pipeline_canonical as pc  # noqa: E402

SEED = 20261101
N_PER_LOWER_BAND = 80
N_RARE_SUPPLEMENT = 20
N_REPEATS = 30
BANDS = [("[0.0,0.5)", 0.0, 0.5), ("[0.5,0.7)", 0.5, 0.7), ("[0.7,0.9)", 0.7, 0.9), ("[0.9,1.0]", 0.9, 1.01)]
NON_SPECIES = {"不明", "誤検出(動物なし)"}
SPECIES = ["ニホンジカ", "イノシシ", "タヌキ", "ニホンアナグマ", "ハクビシン", "アライグマ", "キツネ", "ニホンザル",
           "ニホンノウサギ", "テン", "イタチ", "ニホンリス", "カモシカ", "ネコ", "イヌ", "ネズミ類", "キジバト", "トラツグミ",
           "ヤマガラ", "コジュケイ", "鳥類", "その他", "不明", "誤検出(動物なし)"]
PKG = HERE / "種同定チェック"
KEY = HERE / "audit_key.json"


def is_rare(label: str) -> bool:
    return label not in pc.COMMON and label not in NON_SPECIES


def main() -> None:
    for p in (PKG / "crops", KEY):
        if p.exists():
            sys.exit(f"Refusing to overwrite {p}")
    d = pc.load_and_fit()
    crops, pred, true, conf = d["crops"], d["pred"], d["true"], d["conf"]
    canon = {b["band"]: b for b in json.load(open(pc.CANONICAL_PATH))["efficiency_bands"]}
    index = {json.loads(l)["crop"]: json.loads(l) for l in open(Path(pc.OUT) / "video_crops_index.jsonl")}
    n = len(crops)
    band = np.empty(n, dtype=object)
    for name, lo, hi in BANDS:
        mask = (conf >= lo) & (conf < hi)
        band[mask] = name
        err = sum(pred[i] != true[i] for i in np.flatnonzero(mask))
        ref = canon[f"[{lo},{hi})"]
        print(f"{name}: n={mask.sum()} (canonical {ref['n']}), errors={err} (canonical {ref['corrected']})")
        if mask.sum() != ref["n"]:
            sys.exit("Band size differs from pipeline_canonical.json; stop and investigate.")

    rng = np.random.default_rng(SEED)
    stratum: dict[int, str] = {int(i): "S1_all_high" for i in np.flatnonzero(band == "[0.9,1.0]")}
    for name, _, _ in BANDS[:3]:
        members = np.flatnonzero(band == name)
        for i in rng.choice(members, N_PER_LOWER_BAND, replace=False):
            stratum[int(i)] = "S2_random"
        for tag, get in (("S3_gold_rare", true), ("S3_pred_rare", pred)):
            pool = [int(i) for i in members if is_rare(get[i]) and int(i) not in stratum]
            for i in rng.permutation(pool)[:N_RARE_SUPPLEMENT]:
                stratum[int(i)] = tag
    selected = sorted(stratum)
    repeats = [int(i) for i in rng.choice(selected, N_REPEATS, replace=False)]
    items = [(i, False) for i in selected] + [(i, True) for i in repeats]
    ids = rng.permutation(len(items))

    (PKG / "crops").mkdir(parents=True)
    key_rows, public = [], []
    for (i, is_repeat), num in zip(items, ids):
        anon = f"a{int(num):04d}"
        shutil.copyfile(Path(pc.OUT) / "video_crops" / crops[i], PKG / "crops" / f"{anon}.jpg")
        public.append(anon)
        key_rows.append({"id": anon, "crop": crops[i], "src": index.get(crops[i], {}).get("src"), "band": band[i],
                         "conf": float(conf[i]), "auto_label": pred[i], "gold": true[i], "stratum": stratum[i],
                         "is_hidden_repeat": is_repeat,
                         "gold_rare": is_rare(true[i]), "auto_rare": is_rare(pred[i])})
    public.sort()
    (PKG / "data.js").write_text(
        "const AUDIT=" + json.dumps({"version": "R1-1", "species": SPECIES, "items": public},
                                    ensure_ascii=False) + ";\n", encoding="utf-8")
    meta = {"seed": SEED, "n_population": n, "n_items": len(items),
            "band_population": dict(Counter(band.tolist())),
            "stratum_counts": dict(Counter(r["stratum"] for r in key_rows if not r["is_hidden_repeat"])),
            "sampled_by_band": dict(Counter(r["band"] for r in key_rows if not r["is_hidden_repeat"]))}
    KEY.write_text(json.dumps({"meta": meta, "items": key_rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
