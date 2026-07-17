#!/usr/bin/env python3
"""
exp_external_iwildcam_ordercomp.py — E3: External validation on iWildCam (WILDS)

Purpose:
  Validate that "confidence ordering >= active-learning baselines >> random"
  holds on an INDEPENDENT external camera-trap dataset (iWildCam via WILDS),
  using the same protocol as pipeline_canonical.py (SatoyamaCT).

Data:
  WILDS iWildCam — multi-location camera-trap dataset with 182 species.
  OOD test split (different camera locations from train) used as eval pool.
  Inter-location domain shift is directly analogous to SatoyamaCT.

Subset design (constants at top of file — NOT derived from training data):
  TOP_K_SPECIES   = 20   top-20 species by GLOBAL count (all splits combined)
  TOP_K_COMMON    = 5    top-5 of those 20 = "common"; rest (15) = "rare"
  N_SEED_PER_CLASS= 50   labeled seed images per class from WILDS train split
  N_EVAL_MAX      = 8000 max eval images from OOD test (stratified if larger)

Protocol (IDENTICAL to pipeline_canonical.py):
  LR: class_weight='balanced', max_iter=2000, random_state=42, float64
  Seeds: SEED_BADGE = SEED_PERM = SEED_BOOT = 6906
  N_PERM = 10_000, N_BOOT = 500
  BUDGETS = [0, 5, 10, 20, 30, 50, 100]
  Coreset: k-center greedy on embedding space
  BADGE-approx: k-center greedy on PCA-20 gradient embedding
  p-value: (n_exceed+1)/(N_PERM+1) [Phipson & Smyth 2010]

--smoke-test:
  Bypasses all WILDS / BioCLIP / GPU code. Uses random embeddings of the
  correct production shape to verify every code path (LR fit, 6-method
  ordering, 10k permutation, 500-bootstrap, JSON save) without downloading
  data or needing a GPU. Numbers produced are MEANINGLESS — output JSON is
  flagged with "smoke_test": true and MUST NOT be cited as results.

GPU rule:
  GPU is required ONLY for BioCLIP/DINOv2 embedding extraction (production).
  CUDA_VISIBLE_DEVICES is set before any torch import (see GPU-rule block).
  In --smoke-test mode CUDA_VISIBLE_DEVICES is forced to "" (CPU only).
  Never hardcode cuda:0 / cuda:1.

Output:
  outputs/benchmark/external_iwildcam_ordercomp.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# GPU rule: parse --smoke-test / --gpu-id from sys.argv BEFORE any torch import
# ---------------------------------------------------------------------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--smoke-test", action="store_true")
_pre.add_argument("--gpu-id", type=str, default=None,
                  help="Value to set CUDA_VISIBLE_DEVICES (e.g. '0'). "
                       "Overridden to '' in --smoke-test mode.")
_pre_args, _ = _pre.parse_known_args()

if _pre_args.smoke_test:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
elif _pre_args.gpu_id is not None:
    # cuda:0 / cuda:1 hardcoding forbidden; user supplies the device index via CLI
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu_id
# else: respect CUDA_VISIBLE_DEVICES already set in environment by caller

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HOME = Path(__file__).resolve().parent
OUT_DIR = HOME / "outputs"
BMK_DIR = OUT_DIR / "benchmark"
RESULT_PATH = BMK_DIR / "external_iwildcam_ordercomp.json"

# Safety: do not overwrite existing results
if RESULT_PATH.exists():
    print(f"[ERROR] Output already exists: {RESULT_PATH}")
    print("  既存ファイルの上書き防止のため終了します。")
    sys.exit(1)

BMK_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Subset-design constants
# (all are DATASET-LEVEL constants — not derived from the training split)
# ---------------------------------------------------------------------------

# Number of species to include. We take the TOP_K_SPECIES most frequent
# species across the ENTIRE iWildCam dataset (train+val+test combined).
# This is metadata-level knowledge; no model is trained before this selection.
TOP_K_SPECIES: int = 20

# Of those TOP_K_SPECIES, the top TOP_K_COMMON by global frequency are
# labelled "common"; the remaining TOP_K_SPECIES - TOP_K_COMMON are "rare".
# Directly mirrors the 5-species COMMON set in pipeline_canonical.py.
TOP_K_COMMON: int = 5

# Number of training images sampled from WILDS train split per species.
# Total seed = TOP_K_SPECIES * N_SEED_PER_CLASS = 20 * 50 = 1,000 labels.
N_SEED_PER_CLASS: int = 50

# Maximum eval set size from OOD test split. Stratified-subsampled if larger.
N_EVAL_MAX: int = 8_000

# Backbone: BioCLIP (same as SatoyamaCT) is default.
BACKBONE_DEFAULT: str = "bioclip"
EMB_DIM: Dict[str, int] = {"bioclip": 512, "dinov2": 768}
BIOCLIP_MODEL_ID: str = "hf-hub:imageomics/bioclip"
DINOV2_MODEL_ID: str = "dinov2_vitb14"

# Protocol constants — must match pipeline_canonical.py exactly
BUDGETS: List[int] = [0, 5, 10, 20, 30, 50, 100]
SEED_BADGE: int = 6906
SEED_PERM: int = 6906
SEED_BOOT: int = 6906
N_PERM: int = 10_000
N_BOOT: int = 500

# Smoke-test sizes (small enough to finish in seconds on CPU)
N_SMOKE_TRAIN: int = TOP_K_SPECIES * N_SEED_PER_CLASS   # 1000
N_SMOKE_EVAL: int = 400

# WILDS download root (override via environment variable)
WILDS_ROOT: str = os.environ.get("WILDS_ROOT",
                                  str(Path.home() / "wilds_data"))

# ---------------------------------------------------------------------------
# Utility functions — IDENTICAL to pipeline_canonical.py
# (copy-paste intentional: this script must be self-contained and runnable
#  without importing pipeline_canonical)
# ---------------------------------------------------------------------------

def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(fn(y, x))


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
    """Returns (accs, auc_acc, rares, auc_rare). Identical to pipeline_canonical.py."""
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
    n_shuffles: int,
    base_correct: np.ndarray,
    budgets: List[int],
    n: int,
    rng: np.random.Generator,
    batch: int = 500,
) -> np.ndarray:
    """Vectorised AUC for random permutations. Identical to pipeline_canonical.py."""
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
        cum = np.hstack([
            np.zeros((nb, 1), dtype=np.int64),
            np.cumsum(bc_perms, axis=1, dtype=np.int64),
        ])
        accs_mat = np.zeros((nb, len(Ks)), dtype=np.float64)
        for bi, K in enumerate(Ks):
            accs_mat[:, bi] = (K + total_base - cum[:, K]) / n
        aucs[start:end] = np.sum(
            (accs_mat[:, :-1] + accs_mat[:, 1:]) / 2.0 * dB, axis=1
        )
    return aucs


def bootstrap_auc_gap(
    order_a: np.ndarray,
    order_b: np.ndarray,
    base_correct: np.ndarray,
    budgets: List[int],
    n: int,
    n_boot: int,
    rng: np.random.Generator,
) -> Dict:
    """Bootstrap 95% CI on AUC(a)-AUC(b). Identical to pipeline_canonical.py."""
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
            "Bootstrap CI on AUC(A)-AUC(B). "
            "Positive gap = A achieves higher AUC. "
            "CI excludes 0 => significant at alpha=0.05."
        ),
    }


def coreset_order(Z_train: np.ndarray, Z_eval: np.ndarray) -> np.ndarray:
    """k-center greedy ordering. Identical to pipeline_canonical.py."""
    n_eval = len(Z_eval)
    n_train = len(Z_train)
    chunk = 2048

    # Memory-efficient squared distances via matmul; avoids the
    # (n_eval, chunk, d) broadcast temporary that OOMs under an 8 GiB cgroup.
    # Euclidean distances are numerically identical to the broadcast form.
    eval_sq = (Z_eval ** 2).sum(axis=1)
    min_dist = np.full(n_eval, np.inf)
    for start in range(0, n_train, chunk):
        block = Z_train[start:start + chunk]
        d2 = eval_sq[:, None] + (block ** 2).sum(axis=1)[None, :] - 2.0 * (Z_eval @ block.T)
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
    Z_eval: np.ndarray,
    P: np.ndarray,
    rng: np.random.Generator,
    pca_components: int = 20,
) -> np.ndarray:
    """BADGE-approx: k-center greedy on PCA-20 gradient embedding.
    Identical to pipeline_canonical.py."""
    from sklearn.decomposition import PCA

    n_eval, d = Z_eval.shape
    K = P.shape[1]
    pred_classes = P.argmax(axis=1)

    actual_components = min(pca_components, d, n_eval)
    pca = PCA(n_components=actual_components, random_state=42)
    Z_proj = pca.fit_transform(Z_eval)

    delta = P.copy()
    delta[np.arange(n_eval), pred_classes] -= 1.0
    G = (delta[:, :, None] * Z_proj[:, None, :]).reshape(
        n_eval, K * actual_components
    )

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
# Smoke-test data generator
# ---------------------------------------------------------------------------

def make_smoke_data(
    n_train: int = N_SMOKE_TRAIN,
    n_eval: int = N_SMOKE_EVAL,
    n_classes: int = TOP_K_SPECIES,
    emb_dim: int = 512,
    seed: int = 0,
) -> Dict:
    """
    Generate purely synthetic data for --smoke-test mode.

    Produces random embeddings of the SAME SHAPE as production data so that
    every downstream code path (LR fit, 6-method ordering, 10k permutation,
    500-bootstrap, JSON serialisation) can be exercised without any data
    download, GPU access, or real embedding extraction.

    WARNING: The numbers produced are statistically meaningless.
    The output JSON is tagged smoke_test=true and MUST NOT be cited or used
    as experimental results.
    """
    rng = np.random.default_rng(seed)
    Z_train = rng.standard_normal((n_train, emb_dim)).astype(np.float64)
    y_train_int = np.array([i % n_classes for i in range(n_train)], dtype=int)
    Z_eval = rng.standard_normal((n_eval, emb_dim)).astype(np.float64)
    y_eval_int = rng.integers(0, n_classes, n_eval)

    class_names = [f"species_{i:02d}" for i in range(n_classes)]
    common_set = set(class_names[:TOP_K_COMMON])

    y_train_str = [class_names[y] for y in y_train_int]
    y_eval_str = [class_names[y] for y in y_eval_int]
    rare_mask = np.array([name not in common_set for name in y_eval_str], dtype=bool)

    species_counts_eval = {
        name: int(np.sum(y_eval_int == i))
        for i, name in enumerate(class_names)
    }

    return {
        "Z_train": Z_train,
        "y_train_str": y_train_str,
        "Z_eval": Z_eval,
        "y_eval_str": y_eval_str,
        "class_names": class_names,
        "common_names": sorted(common_set),
        "rare_mask": rare_mask,
        "species_counts_eval": species_counts_eval,
        "emb_dim": emb_dim,
        "n_train": n_train,
        "n_eval": n_eval,
    }


# ---------------------------------------------------------------------------
# Production: WILDS iWildCam data loader
# ---------------------------------------------------------------------------

def _resolve_category_names(dataset) -> List[str]:
    """
    Return a list of category names indexed by y-value.
    Tries multiple WILDS attribute spellings; falls back to integer strings.
    """
    for attr in ("categories", "_categories_list", "category_names"):
        obj = getattr(dataset, attr, None)
        if obj is not None:
            return list(obj)
    # Last resort
    n_cls = int(getattr(dataset, "n_classes", 200))
    return [str(i) for i in range(n_cls)]


def load_iwildcam_subset(backbone: str) -> Dict:
    """
    Load WILDS iWildCam, select TOP_K_SPECIES by global count,
    extract embeddings (BioCLIP or DINOv2), and return train/eval arrays.
    """
    try:
        from wilds import get_dataset
    except ImportError as exc:
        raise ImportError(
            "wilds package not found.\n"
            "Install: pip install wilds\n"
            "Then set WILDS_ROOT and re-run in production mode."
        ) from exc

    print(f"  [WILDS] Loading iWildCam metadata from {WILDS_ROOT}...", flush=True)
    dataset = get_dataset(dataset="iwildcam", download=True, root_dir=WILDS_ROOT)

    # ---- Label array ----
    y_all = (dataset.y_array.numpy()
             if hasattr(dataset.y_array, "numpy")
             else np.array(dataset.y_array))
    n_total = len(y_all)
    print(f"  Total dataset size: {n_total}", flush=True)

    cat_names = _resolve_category_names(dataset)
    print(f"  Total categories: {len(cat_names)}", flush=True)

    # ---- Identify empty/background classes (dataset-level, not train-derived) ----
    empty_ids: set = set()
    for cid, cname in enumerate(cat_names):
        if cname.lower().strip() in {"empty", "background", "no animal", ""}:
            empty_ids.add(cid)
    if not empty_ids:
        empty_ids = {0}   # iWildCam convention: label 0 = "empty"
    print(f"  Empty/background class IDs: {empty_ids}", flush=True)

    # ---- Count species across ALL splits (dataset-level, NOT train-derived) ----
    from collections import Counter
    valid_mask = ~np.isin(y_all, list(empty_ids))
    species_global_counts: Counter = Counter(int(y) for y in y_all[valid_mask])

    top_species_ids: List[int] = [
        cid for cid, _ in species_global_counts.most_common(TOP_K_SPECIES)
    ]
    top_species_set = set(top_species_ids)

    class_names: List[str] = [cat_names[i] for i in top_species_ids]
    common_set = set(class_names[:TOP_K_COMMON])
    cat_to_local: Dict[int, int] = {cid: i for i, cid in enumerate(top_species_ids)}

    print(f"  Top-{TOP_K_SPECIES} species (global count, COMMON=top-{TOP_K_COMMON}):",
          flush=True)
    for rank, (cid, cnt) in enumerate(
            species_global_counts.most_common(TOP_K_SPECIES)):
        tag = "COMMON" if rank < TOP_K_COMMON else "rare"
        cname = cat_names[cid] if cid < len(cat_names) else str(cid)
        print(f"    [{rank+1:2d}] {cname:<30} global_n={cnt:6d}  ({tag})",
              flush=True)

    # ---- Split indices ----
    split_arr = (dataset.split_array.numpy()
                 if hasattr(dataset.split_array, "numpy")
                 else np.array(dataset.split_array))
    split_dict = dataset.split_dict   # {'train': 0, 'val': 1, 'test': 2, ...}
    train_sid = split_dict.get("train", 0)
    test_sid  = split_dict.get("test",  2)   # OOD test: different camera locations

    in_top = np.isin(y_all, list(top_species_set))
    train_idx = np.where((split_arr == train_sid) & in_top)[0]
    test_idx  = np.where((split_arr == test_sid)  & in_top)[0]
    print(f"  Train pool (top-{TOP_K_SPECIES}): {len(train_idx):,}", flush=True)
    print(f"  OOD test pool (top-{TOP_K_SPECIES}): {len(test_idx):,}", flush=True)

    rng_sel = np.random.default_rng(42)

    # ---- Stratified seed: N_SEED_PER_CLASS images/class from train ----
    seed_indices: List[int] = []
    for cid in top_species_ids:
        cand = train_idx[y_all[train_idx] == cid]
        n_sel = min(N_SEED_PER_CLASS, len(cand))
        if n_sel > 0:
            chosen = rng_sel.choice(cand, size=n_sel, replace=False)
            seed_indices.extend(chosen.tolist())
    seed_indices_arr = np.array(seed_indices)
    print(f"  Seed set: {len(seed_indices_arr)} images", flush=True)

    # ---- Eval set: OOD test, stratified subsample to N_EVAL_MAX ----
    if len(test_idx) > N_EVAL_MAX:
        eval_list: List[int] = []
        per_class = N_EVAL_MAX // TOP_K_SPECIES
        for cid in top_species_ids:
            cand = test_idx[y_all[test_idx] == cid]
            n_sel = min(per_class, len(cand))
            if n_sel > 0:
                chosen = rng_sel.choice(cand, size=n_sel, replace=False)
                eval_list.extend(chosen.tolist())
        eval_indices_arr = np.array(eval_list)
    else:
        eval_indices_arr = test_idx
    print(f"  Eval set: {len(eval_indices_arr)} images", flush=True)

    # ---- Embedding extraction ----
    device_str = _resolve_device()
    print(f"  Embedding device: {device_str}", flush=True)
    if backbone == "bioclip":
        Z_seed, Z_eval_arr = _extract_bioclip(
            dataset, seed_indices_arr, eval_indices_arr, device_str
        )
    elif backbone == "dinov2":
        Z_seed, Z_eval_arr = _extract_dinov2(
            dataset, seed_indices_arr, eval_indices_arr, device_str
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    y_seed_str = [class_names[cat_to_local[int(y_all[i])]] for i in seed_indices_arr]
    y_eval_str = [class_names[cat_to_local[int(y_all[i])]] for i in eval_indices_arr]
    rare_mask  = np.array([name not in common_set for name in y_eval_str], dtype=bool)

    from collections import Counter as _Ctr
    ec = _Ctr(y_eval_str)
    species_counts_eval = {name: ec.get(name, 0) for name in class_names}

    def _arr_hash(arr: np.ndarray) -> str:
        return hashlib.sha256(arr.tobytes()).hexdigest()[:16]

    return {
        "Z_train": Z_seed.astype(np.float64),
        "y_train_str": y_seed_str,
        "Z_eval": Z_eval_arr.astype(np.float64),
        "y_eval_str": y_eval_str,
        "class_names": class_names,
        "common_names": sorted(common_set),
        "rare_mask": rare_mask,
        "species_counts_eval": species_counts_eval,
        "emb_dim": int(Z_seed.shape[1]),
        "n_train": len(seed_indices_arr),
        "n_eval": len(eval_indices_arr),
        "data_hashes": {
            "seed_indices_sha256_prefix": _arr_hash(seed_indices_arr),
            "eval_indices_sha256_prefix": _arr_hash(eval_indices_arr),
            "wilds_root": WILDS_ROOT,
            "dataset": "iwildcam",
        },
    }


def _resolve_device() -> str:
    """Return torch device string based on CUDA_VISIBLE_DEVICES."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible in ("", "-1"):
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _pil_from_wilds_item(x) -> "PIL.Image.Image":
    """Convert a WILDS dataset item (PIL or tensor) to PIL Image."""
    from PIL import Image as PImage
    if isinstance(x, PImage.Image):
        return x.convert("RGB")
    try:
        import torch
        import torchvision.transforms.functional as TF
        if isinstance(x, torch.Tensor):
            return TF.to_pil_image(x).convert("RGB")
    except ImportError:
        pass
    import numpy as _np
    arr = _np.array(x)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != _np.uint8:
        arr = (arr * 255).clip(0, 255).astype(_np.uint8)
    from PIL import Image as PImage2
    return PImage2.fromarray(arr).convert("RGB")


def _extract_bioclip(
    dataset,
    seed_idx: np.ndarray,
    eval_idx: np.ndarray,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract BioCLIP (hf-hub:imageomics/bioclip) embeddings."""
    import torch
    import open_clip

    print(f"  Loading BioCLIP ({BIOCLIP_MODEL_ID})...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        BIOCLIP_MODEL_ID, pretrained=True
    )
    model = model.to(device).eval()
    print("  BioCLIP loaded.", flush=True)

    def embed_indices(indices: np.ndarray, tag: str) -> np.ndarray:
        all_embs = []
        bs = 64
        with torch.no_grad():
            for start in range(0, len(indices), bs):
                batch_i = indices[start:start + bs]
                imgs = []
                for gi in batch_i:
                    raw_x, _, _ = dataset[int(gi)]
                    imgs.append(preprocess(_pil_from_wilds_item(raw_x)))
                batch_t = torch.stack(imgs).to(device)
                feats = model.encode_image(batch_t)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_embs.append(feats.cpu().float().numpy())
                if (start // bs) % 20 == 0:
                    print(f"    {tag}: {start}/{len(indices)}", flush=True)
        return np.concatenate(all_embs, axis=0)

    Z_seed = embed_indices(seed_idx, "seed")
    Z_eval = embed_indices(eval_idx, "eval")
    return Z_seed, Z_eval


def _extract_dinov2(
    dataset,
    seed_idx: np.ndarray,
    eval_idx: np.ndarray,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract DINOv2 (dinov2_vitb14) embeddings (optional backbone)."""
    import torch
    from torchvision import transforms as T

    print(f"  Loading DINOv2 ({DINOV2_MODEL_ID})...", flush=True)
    model = torch.hub.load("facebookresearch/dinov2", DINOV2_MODEL_ID)
    model = model.to(device).eval()

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def embed_indices(indices: np.ndarray, tag: str) -> np.ndarray:
        all_embs = []
        bs = 64
        with torch.no_grad():
            for start in range(0, len(indices), bs):
                batch_i = indices[start:start + bs]
                imgs = []
                for gi in batch_i:
                    raw_x, _, _ = dataset[int(gi)]
                    imgs.append(transform(_pil_from_wilds_item(raw_x)))
                batch_t = torch.stack(imgs).to(device)
                feats = model(batch_t)
                all_embs.append(feats.cpu().float().numpy())
                if (start // bs) % 20 == 0:
                    print(f"    {tag}: {start}/{len(indices)}", flush=True)
        return np.concatenate(all_embs, axis=0)

    Z_seed = embed_indices(seed_idx, "seed")
    Z_eval = embed_indices(eval_idx, "eval")
    return Z_seed, Z_eval


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(
        description="E3: External validation — iWildCam order comparison"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run all code paths with random data (no WILDS/GPU required). "
             "Numbers produced are MEANINGLESS — tagged smoke_test=true in JSON.",
    )
    parser.add_argument(
        "--backbone", choices=["bioclip", "dinov2"], default=BACKBONE_DEFAULT,
        help=f"Embedding backbone (default: {BACKBONE_DEFAULT})",
    )
    parser.add_argument(
        "--gpu-id", type=str, default=None,
        help="Already handled before torch import; kept for --help visibility.",
    )
    args = parser.parse_args()

    t_global = time.time()
    result: Dict = {}

    mode_str = ("SMOKE TEST — random data, numbers MEANINGLESS"
                if args.smoke_test else "PRODUCTION")
    print("=" * 70, flush=True)
    print("exp_external_iwildcam_ordercomp.py  — E3: External validation",
          flush=True)
    print(f"Mode:     {mode_str}", flush=True)
    print(f"Backbone: {args.backbone}", flush=True)
    print(f"Output:   {RESULT_PATH}", flush=True)
    print("=" * 70, flush=True)

    # ------------------------------------------------------------------
    # Step 1: Data loading
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading data...", flush=True)
    t0 = time.time()

    if args.smoke_test:
        print("  Generating random smoke-test embeddings.", flush=True)
        data = make_smoke_data(emb_dim=EMB_DIM[args.backbone])
    else:
        import pickle as _pickle
        _emb_cache = Path(f"_e3_embcache_{args.backbone}.pkl")
        if _emb_cache.exists():
            print(f"  Loading cached embeddings from {_emb_cache}", flush=True)
            with open(_emb_cache, "rb") as _cf:
                data = _pickle.load(_cf)
        else:
            data = load_iwildcam_subset(backbone=args.backbone)
            with open(_emb_cache, "wb") as _cf:
                _pickle.dump(data, _cf)
            print(f"  Cached embeddings to {_emb_cache}", flush=True)

    Z_train: np.ndarray = data["Z_train"]
    Z_eval: np.ndarray  = data["Z_eval"]
    y_train_str: List[str] = data["y_train_str"]
    y_eval_str: List[str]  = data["y_eval_str"]
    class_names: List[str] = data["class_names"]
    common_set = set(data["common_names"])
    rare_mask: np.ndarray  = data["rare_mask"]
    n = len(Z_eval)

    assert Z_train.dtype == np.float64, f"Z_train dtype mismatch: {Z_train.dtype}"
    assert Z_eval.dtype  == np.float64, f"Z_eval dtype mismatch: {Z_eval.dtype}"

    print(f"  Z_train: {Z_train.shape} {Z_train.dtype}", flush=True)
    print(f"  Z_eval:  {Z_eval.shape} {Z_eval.dtype}", flush=True)
    print(f"  n_classes: {len(class_names)}  n_rare_eval: {int(rare_mask.sum())}/{n}",
          flush=True)
    print(f"  Step 1 done in {time.time()-t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Step 2: Fit LR — IDENTICAL hyperparams to pipeline_canonical.py
    # ------------------------------------------------------------------
    print("\n[Step 2] Fitting LogisticRegression (float64)...", flush=True)
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ).fit(Z_train, y_train_str)

    classes: List[str] = list(clf.classes_)
    n_classes = len(classes)
    print(f"  n_classes={n_classes}", flush=True)

    # ------------------------------------------------------------------
    # Step 3: Single predict_proba call (float64) — all metrics derived here
    # ------------------------------------------------------------------
    print("\n[Step 3] predict_proba (single call, float64)...", flush=True)
    P = clf.predict_proba(Z_eval)          # (n, n_classes), float64
    assert P.dtype == np.float64, f"P dtype: {P.dtype}"

    pred_str: List[str] = [classes[int(i)] for i in P.argmax(axis=1)]

    conf   = P.max(axis=1).astype(np.float64)
    ent    = -(P * np.log(P + 1e-9)).sum(axis=1).astype(np.float64)
    part   = np.sort(P, axis=1)
    margin = (part[:, -1] - part[:, -2]).astype(np.float64)

    base_correct = np.array(
        [pred_str[i] == y_eval_str[i] for i in range(n)], dtype=bool
    )
    base_acc = float(base_correct.mean())
    n_rare = int(rare_mask.sum())
    print(f"  base_acc={base_acc:.4f}  n_rare={n_rare}", flush=True)

    result["data_info"] = {
        "dataset": "iWildCam (WILDS)",
        "dataset_url": "https://wilds.stanford.edu/datasets/",
        "citation": "Koh et al., NeurIPS 2021 (WILDS benchmark)",
        "smoke_test": args.smoke_test,
        "backbone": args.backbone,
        "emb_dim": data["emb_dim"],
        "n_train": data["n_train"],
        "n_eval": n,
        "n_classes": n_classes,
        "top_k_species": TOP_K_SPECIES,
        "top_k_common": TOP_K_COMMON,
        "n_seed_per_class": N_SEED_PER_CLASS,
        "n_eval_max": N_EVAL_MAX,
        "classes": sorted(classes),
        "common_names": sorted(common_set),
        "rare_def": (
            f"species not in top-{TOP_K_COMMON} by global count "
            f"(across all WILDS splits — not derived from train split)"
        ),
        "n_rare_eval": n_rare,
        "base_accuracy": round(base_acc, 4),
        "eval_split": "OOD test (different camera locations from train)",
        "species_counts_eval": data.get("species_counts_eval", {}),
    }
    if "data_hashes" in data:
        result["data_info"]["data_hashes"] = data["data_hashes"]

    # ------------------------------------------------------------------
    # Step 4: Orderings (5 deterministic + 1 random via permutation test)
    # ------------------------------------------------------------------
    print("\n[Step 4] Computing orderings...", flush=True)

    orders: Dict[str, np.ndarray] = {
        "confidence": np.argsort(conf),    # least-confident first
        "entropy":    np.argsort(-ent),    # highest-entropy first
        "margin":     np.argsort(margin),  # smallest-margin first
    }

    print("  [Step 4a] Coreset...", flush=True)
    t0 = time.time()
    orders["coreset"] = coreset_order(Z_train, Z_eval)
    print(f"  Coreset done in {time.time()-t0:.1f}s", flush=True)

    print("  [Step 4b] BADGE-approx (PCA-20 gradient embedding)...", flush=True)
    t0 = time.time()
    rng_badge = np.random.default_rng(SEED_BADGE)
    orders["badge_approx"] = badge_order(Z_eval, P, rng_badge, pca_components=20)
    print(f"  BADGE done in {time.time()-t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Step 5: AUC for all deterministic orderings
    # ------------------------------------------------------------------
    print("\n[Step 5] Computing AUCs...", flush=True)
    result_orders: Dict[str, Dict] = {}
    for name, idx in orders.items():
        accs, auc_acc, rares, auc_rare = compute_auc_from_order(
            idx, base_correct, n, BUDGETS, rare_mask
        )
        result_orders[name] = {
            "acc_by_budget": accs,
            "auc_acc_budget": auc_acc,
            "rare_recall_by_budget": rares,
            "rare_recall_auc": auc_rare,
        }
        print(f"  {name:<22}  AUC_acc={auc_acc}  AUC_rare={auc_rare}", flush=True)

    conf_auc = result_orders["confidence"]["auc_acc_budget"]

    # ------------------------------------------------------------------
    # Step 6: 10k permutation test — confidence vs random
    # ------------------------------------------------------------------
    print(f"\n[Step 6] 10k permutation test (N={N_PERM})...", flush=True)
    t0 = time.time()
    rng_perm = np.random.default_rng(SEED_PERM)
    rand_aucs = fast_rand_aucs(N_PERM, base_correct, BUDGETS, n, rng_perm, batch=500)
    n_exceed = int(np.sum(rand_aucs >= conf_auc))
    pval = (n_exceed + 1) / (N_PERM + 1)    # Phipson & Smyth 2010
    rand_mean = round(float(rand_aucs.mean()), 4)
    rand_std  = round(float(rand_aucs.std()),  4)
    print(f"  Done in {time.time()-t0:.1f}s  "
          f"rand={rand_mean}+/-{rand_std}  n_exceed={n_exceed}  p={pval:.6f}",
          flush=True)

    result_orders["random"] = {
        "auc_acc_budget":     rand_mean,
        "auc_acc_budget_std": rand_std,
        "n_shuffles": N_PERM,
    }
    result["confidence_vs_random_perm"] = {
        "n_shuffles": N_PERM,
        "n_exceed":   n_exceed,
        "p_value":    round(pval, 6),
        "formula":    "(n_exceed+1)/(N_PERM+1) [Phipson & Smyth 2010]",
    }

    # ------------------------------------------------------------------
    # Step 7: Bootstrap CI — confidence vs Coreset, confidence vs BADGE
    # ------------------------------------------------------------------
    print(f"\n[Step 7] Bootstrap CI (n_boot={N_BOOT})...", flush=True)
    rng_boot_cs = np.random.default_rng(SEED_BOOT + 100)
    rng_boot_bg = np.random.default_rng(SEED_BOOT + 200)

    ci_vs_coreset = bootstrap_auc_gap(
        orders["confidence"], orders["coreset"],
        base_correct, BUDGETS, n, N_BOOT, rng_boot_cs,
    )
    ci_vs_badge = bootstrap_auc_gap(
        orders["confidence"], orders["badge_approx"],
        base_correct, BUDGETS, n, N_BOOT, rng_boot_bg,
    )
    print(f"  conf - coreset: {ci_vs_coreset}", flush=True)
    print(f"  conf - badge:   {ci_vs_badge}", flush=True)

    result["bootstrap_auc_gap_confidence_minus"] = {
        "coreset":      ci_vs_coreset,
        "badge_approx": ci_vs_badge,
    }

    # ------------------------------------------------------------------
    # Assemble final result
    # ------------------------------------------------------------------
    result["orders"]      = result_orders
    result["budgets_pct"] = BUDGETS

    if args.smoke_test:
        result["SMOKE_TEST_WARNING"] = (
            "THIS IS A SMOKE TEST RUN. All numerical results are based on "
            "RANDOM EMBEDDINGS and are STATISTICALLY MEANINGLESS. "
            "Do NOT cite, compare, or use any value from this file as an "
            "experimental result. Purpose: verify code paths execute correctly."
        )

    result["note"] = (
        f"E3 External validation — iWildCam (WILDS, OOD test split). "
        f"Backbone: {args.backbone}. "
        f"Protocol identical to pipeline_canonical.py: "
        f"LR(balanced, max_iter=2000, seed=42, float64), "
        f"BUDGETS={BUDGETS}, SEED=6906, N_PERM={N_PERM}, N_BOOT={N_BOOT}. "
        f"Subset: top-{TOP_K_SPECIES} species by global count (all WILDS splits). "
        f"Common: top-{TOP_K_COMMON} species. Rare: remaining {TOP_K_SPECIES-TOP_K_COMMON}. "
        f"Seed: {N_SEED_PER_CLASS} img/class from WILDS train "
        f"({TOP_K_SPECIES * N_SEED_PER_CLASS} total). "
        f"Eval: OOD test, <=N_EVAL_MAX={N_EVAL_MAX} images. "
        f"p-value: (n_exceed+1)/(N_PERM+1) [Phipson & Smyth 2010]."
    )

    # Environment log
    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(HOME), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = "unavailable"

    import sklearn
    env_log: Dict = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "git_commit": git_commit,
        "script": str(Path(__file__).resolve()),
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed_badge": SEED_BADGE,
        "seed_perm":  SEED_PERM,
        "seed_boot":  SEED_BOOT,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"),
        "smoke_test": args.smoke_test,
        "backbone": args.backbone,
    }
    try:
        import torch
        env_log["torch"] = torch.__version__
        if torch.cuda.is_available():
            env_log["cuda"] = torch.version.cuda
            # Physical GPU UUIDs — no hardcoded device indices
            env_log["gpu_uuids"] = [
                str(torch.cuda.get_device_properties(i).uuid)
                for i in range(torch.cuda.device_count())
            ]
    except Exception:
        pass
    env_log["elapsed_total_s"] = round(time.time() - t_global, 1)
    result["env_log"] = env_log

    result["_meta"] = {
        "status": "complete",
        "smoke_test": args.smoke_test,
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t_global, 1),
        "script": str(Path(__file__).resolve()),
        "description": (
            "E3: External validity. Validates that confidence ordering "
            ">= Coreset/BADGE and >> random on iWildCam (OOD test split), "
            "independent of SatoyamaCT."
        ),
        "canonical_reference": str(BMK_DIR / "pipeline_canonical.json"),
    }

    # Write output JSON
    with open(str(RESULT_PATH), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total = time.time() - t_global
    print("\n" + "=" * 70, flush=True)
    label = "SMOKE TEST DONE" if args.smoke_test else "DONE"
    print(f"{label} in {total:.1f}s", flush=True)
    print(f"Output: {RESULT_PATH}", flush=True)
    print("=" * 70, flush=True)

    # Summary table
    print("\n=== ORDER COMPARISON SUMMARY ===", flush=True)
    if args.smoke_test:
        print("  *** SMOKE TEST — numbers below are MEANINGLESS ***", flush=True)
    print(f"n_eval={n}  n_rare={n_rare}  base_acc={base_acc:.4f}", flush=True)
    print(f"\n{'Method':<24} {'AUC_acc':>8} {'AUC_rare':>9}", flush=True)
    print("-" * 44, flush=True)
    for name, d in result_orders.items():
        if name == "random":
            print(f"{'random':<24} {d['auc_acc_budget']:>8}"
                  f"+-{d['auc_acc_budget_std']}", flush=True)
        else:
            print(f"{name:<24} {d['auc_acc_budget']:>8}"
                  f" {d.get('rare_recall_auc', 'N/A'):>9}", flush=True)
    pr = result["confidence_vs_random_perm"]
    print(f"\nPerm test: n_exceed={pr['n_exceed']}/{N_PERM}  p={pr['p_value']:.6f}",
          flush=True)
    boot = result["bootstrap_auc_gap_confidence_minus"]
    print(f"Boot CI conf-coreset: {boot['coreset']['ci95']}", flush=True)
    print(f"Boot CI conf-badge:   {boot['badge_approx']['ci95']}", flush=True)
    if args.smoke_test:
        print("\n*** ALL NUMBERS ABOVE ARE FROM RANDOM DATA — DO NOT CITE ***",
              flush=True)


if __name__ == "__main__":
    main()
