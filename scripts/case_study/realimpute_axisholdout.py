#!/usr/bin/env python
"""Axis-holdout real imputation for both case studies.

Protocol: the proteins that define the evaluation axis
(liver: 215 HSS gene-set proteins; Brunner: 22 quantified G2M/S cycle-score
markers) are REMOVED from the training matrix entirely. Models train on the
remaining proteins' observed entries only (missing.enabled=False, no artificial
injection — liver realimpute protocol, now unified across both case studies).
The imputed matrix covers the kept (in-matrix) proteins; the axis proteins are
additionally predicted ZERO-SHOT from their sequence prior (SABER-only side
output; baselines have no column for them).

Recipe per dataset:
  liver   <- configs/case_study/cellembed_saber.yaml
  brunner <- configs/case_study/brunner_impute_missingon.yaml

Outputs (per dataset): $SABER_OUTPUTS_ROOT/<name>_realimpute_axisholdout/matrices.npz with
  observed_reduced (raw, NaN missing), saber_seeds (z-space, kept proteins),
  zeroshot_axis_seeds (z-space, axis proteins), knn, cellmean, acs_keep, acs_axis.
Run:  python scripts/case_study/realimpute_axisholdout.py --dataset liver
"""
from __future__ import annotations
import argparse, sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import KNNImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from saber.config import load_config
from saber.data import load_data_bundle
from saber.expansion import _build_model, subset_data_bundle
from saber.paths import data_root, outputs_root
from saber.train import train_model, predict_all

REPO = Path(__file__).resolve().parents[2]


def _datasets() -> dict[str, dict[str, str]]:
    return {
        "liver": dict(
            cfg=str(REPO / "configs/case_study/cellembed_saber.yaml"),
            axis=str(data_root() / "easyscp_liver/markers/axis_targets.txt"),
            out=str(outputs_root() / "easyscp_realimpute_axisholdout"),
        ),
        "brunner": dict(
            cfg=str(REPO / "configs/case_study/brunner_impute_missingon.yaml"),
            axis=str(data_root() / "brunner2022/markers/axis_targets.txt"),
            out=str(outputs_root() / "brunner_realimpute_axisholdout"),
        ),
    }


DATASETS = None  # resolved lazily (needs configured roots)
SEEDS = [42, 123, 2024, 2025, 7, 0, 1, 2, 3, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="liver | brunner")
    ap.add_argument("--smoke", action="store_true", help="2 epochs, seed 42 only")
    ap.add_argument("--out", default="", help="override the output directory")
    ap.add_argument("--config", default="", help="override the training config (e.g. the no-PLM variant)")
    args = ap.parse_args()
    datasets = _datasets()
    if args.dataset not in datasets:
        ap.error(f"--dataset must be one of {sorted(datasets)}")
    spec = datasets[args.dataset]
    if args.out:
        spec = dict(spec, out=args.out)
    if args.config:
        spec = dict(spec, cfg=str(Path(args.config).resolve()))

    cfg = load_config(spec["cfg"])
    cfg = replace(cfg, missing=replace(cfg.missing, enabled=False))  # no injection
    if args.smoke:
        from dataclasses import replace as _r
        cfg = _r(cfg, training=_r(cfg.training, epochs=2))

    axis_ids = {ln.strip() for ln in open(spec["axis"]) if ln.strip()}
    full = load_data_bundle(cfg.data, cfg.missing, cfg.model, seed=42)
    pids = [str(p) for p in full.protein_ids]
    keep_idx = np.array([i for i, p in enumerate(pids) if p not in axis_ids])
    axis_idx = np.array([i for i, p in enumerate(pids) if p in axis_ids])
    data = subset_data_bundle(full, keep_idx)
    print(f"[{args.dataset}] kept {keep_idx.size} / axis-heldout {axis_idx.size} proteins; "
          f"cells {data.x_true.shape[0]}; observed frac {np.asarray(data.available_mask).mean():.2f}")

    saber_mats, zs_mats = [], []
    seeds = [42] if args.smoke else SEEDS
    for s in seeds:
        model = _build_model(cfg, data)
        res = train_model(model, data, cfg.training, seed=s)
        pred, _, _ = predict_all(model, data.x_input, res.device)
        saber_mats.append(pred.astype(np.float32))
        # zero-shot prediction of the axis proteins from their sequence prior
        model.eval()
        dev = next(model.parameters()).device
        with torch.no_grad():
            z = model.encode_cells(torch.as_tensor(data.x_input, dtype=torch.float32, device=dev))
            e_axis = model.encode_proteins(
                torch.as_tensor(full.protein_features[axis_idx], dtype=torch.float32, device=dev))
            z_use = z @ model.W if model.W is not None else z
            zs = (z_use @ e_axis.T).cpu().numpy().astype(np.float32)
        zs_mats.append(zs)
        print(f"[{args.dataset}] seed {s} done (imputed {pred.shape}, zeroshot {zs.shape})")

    observed = data.x_true.astype(np.float32)           # raw scale, NaN = real missing
    avail = np.isfinite(observed)
    knn = KNNImputer(n_neighbors=5, weights="distance").fit_transform(
        np.where(avail, observed, np.nan)).astype(np.float32)
    cellmean = np.where(avail, observed,
                        np.nanmean(np.where(avail, observed, np.nan), axis=1, keepdims=True)).astype(np.float32)

    out = Path(spec["out"]); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "matrices.npz",
        observed_reduced=observed, saber_seeds=np.asarray(saber_mats),
        zeroshot_axis_seeds=np.asarray(zs_mats), knn=knn, cellmean=cellmean,
        acs_keep=np.asarray([pids[i] for i in keep_idx], dtype=str),
        acs_axis=np.asarray([pids[i] for i in axis_idx], dtype=str),
        seeds=np.asarray(seeds))
    print(f"[{args.dataset}] saved -> {out/'matrices.npz'}")


if __name__ == "__main__":
    main()
