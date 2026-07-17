# MiLT: Model-in-the-Loop Uncertainty-aware Labeling and Triage

MiLT is a lightweight, model-in-the-loop pipeline for annotating and triaging night camera-trap
data in agroforestry (*satoyama*) landscapes. It combines foundation-model embeddings, seed
confirmation, a lightweight classifier, confidence-ordered review, and uncertainty triage —
**without large-scale GPU fine-tuning** — and adds RCTA (Rare-Class-Targeted Acquisition), a
parameter-free criterion that surfaces already-identified rare species early in the review queue.

This repository accompanies the pipeline paper. The dataset itself (crops, metadata, splits,
embeddings) is released separately as **SatoyamaCT** (see *Data* below); this repo contains only
the pipeline code and the pre-computed experiment outputs.

## Pipeline Stages

1. **Embed & cluster** — BioCLIP (512-d) embeddings, k-means clusters of visually similar animals.
2. **Seed confirmation** — experts confirm a stratified set of representative seed crops.
3. **Auto-label** — a multiclass logistic regression labels all remaining crops with confidence.
4. **Confidence-ordered review** — experts confirm high-confidence predictions in bulk.
5. **Uncertainty triage + RCTA** — route ambiguous / rare crops to the front of the review queue.

Frozen BioCLIP embeddings + a lightweight classifier keep the whole pipeline GPU-fine-tuning-free.

## Key Results (all reproduced from `results/`)

| Claim | Value | Result file |
|-------|-------|-------------|
| High-confidence auto-labels are near-correct | correction rate 1.5% (lower bound) | `results/pipeline_canonical.json` |
| Confidence order beats random, matches AL baselines | acc–budget AUC bootstrap-CI gain over Coreset/BADGE | `results/pipeline_experiments.json`, `results/pipeline_al_baselines.json` |
| Backbone-independence (DINOv2) | conf AUC 0.919 vs random 0.870 (*p*=1.0×10⁻⁴), ≈ BADGE | `results/pipeline_dinov2_ordercomp.json` |
| External validation (iWildCam) | conf AUC 0.887 vs random 0.812 (*p*=1.0×10⁻⁴) | `results/external_iwildcam_ordercomp.json` |
| Annotation-cost saving | target acc 0.80 reached at 28.4% reviewed (−71.7%) | `results/annotation_cost_model.json` |
| RCTA surfaces true rare species early | capture 22%→57% at 10% budget (bootstrap 95% CI excludes 0) | `results/rau_true_rare_n173.json`, `results/rcta_bootstrap2000.json` |
| Full-review triage upper bound | acc 0.81 / TV 0.26→0.12 at 30% human | `results/rarity_aware_triage.json` |

## Code → Result Map

| Script | Produces | Paper table/figure |
|--------|----------|--------------------|
| `src/pipeline_canonical.py` | canonical pipeline run, correction rates | Auto-label ablation, confidence bands |
| `src/pipeline_experiments.py` | review-order comparison (AUC) | order-comparison table |
| `src/pipeline_ablation.py` | seed-only ablation | ablation table |
| `src/exp_dinov2_ordercomp.py` | DINOv2 backbone robustness | DINOv2 table |
| `src/exp_external_iwildcam_ordercomp.py` | external validation on iWildCam | external-validation paragraph |
| `src/exp_annotation_cost.py` | accuracy–budget cost model | cost model |
| `src/rare_targeted_acquisition.py`, `src/rcta_bootstrap2000.py`, `src/rau_true_rare_analysis.py` | RCTA rare-recall + bootstrap CI | rare-recall table |
| `src/rarity_aware_triage.py` | full-review triage upper bound | triage table |
| `src/compute_iaa.py` | inter-annotator agreement | IAA analysis |

## Installation

```bash
pip install -r requirements.txt
```

## Data

The pipeline runs on the **SatoyamaCT** dataset (crops, metadata, recommended splits, and
pre-computed embeddings). Metadata and splits ship in the [SatoyamaCT GitHub repository](https://github.com/keito-git/satoyama-ct);
the crop images and embeddings are distributed as GitHub Release assets under CC BY-NC 4.0.
Download them first and point the scripts at the local dataset directory.

The external-validation experiment additionally uses the public **iWildCam** subset
(WILDS), downloaded via its official loader.

## Reproduction

```bash
# after downloading SatoyamaCT crops + embeddings
python src/pipeline_experiments.py         # review-order comparison
python src/exp_dinov2_ordercomp.py         # backbone robustness
python src/exp_annotation_cost.py          # cost model
python src/exp_external_iwildcam_ordercomp.py --gpu-id ""   # external validation (CPU)
python src/rcta_bootstrap2000.py           # RCTA bootstrap CI
```

Pre-computed outputs are in `results/` so tables can be checked without a full rerun.

## Citation

```bibtex
@article{milt_2026,
  author        = {Inoshita, Keito and Nojiri, Kota and Sugeno, Haruto},
  title         = {{MiLT}: Model-in-the-Loop Uncertainty-aware Labeling and Triage
                   for Night Camera-Trap Data},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
}
```

## License

Code is released under the MIT License (see `LICENSE`). The SatoyamaCT dataset it operates on
is licensed separately under CC BY-NC 4.0.

## Ethics and Location Privacy

Precise GPS coordinates and site names are withheld to protect rare and sensitive species.
No geographic coordinates are contained in this repository or in the released dataset.
