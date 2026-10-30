#!/usr/bin/env python
"""Species-only LFQ control imputer, recomputed under the 10-seed campaign protocol.

Fills each missing entry with its (condition x species) block mean of observed
raw-scale values (see saber.evaluate.lfq._species_block_mean_baseline), then scores
fold-change MAE on the has-missing proteins exactly as the campaign LFQ evaluation
does (median reconstructed intensities per condition, per-protein log2FC vs expected
mixture ratios; saber.evaluate.lfq.evaluate_lfq_manifest protocol).

Deterministic: the result is identical across seeds; we still evaluate on three
seeds' label/imputation artifacts to confirm and emit per-seed rows that merge
cleanly with source_metrics_long.csv (method='species_mean', one row per seed).

Usage:
  conda run -n saber python scripts/figures/lfq_species_only.py \
      --run paper_benchmark_20260801_final_10seed [--merge]

--merge appends/updates the species_mean rows in the run's
source_data/source_metrics_long.csv (dedup on task,dataset,method,lfq_subset,seed).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))  # resolve `saber` from this checkout, not the env

from saber.evaluate.lfq import (
    _split_masks,
    _species_block_mean_baseline,
    compute_per_protein_log2fc,
    normalize_conditions,
    normalize_species,
    summarize_lfq_records,
)



def evaluate_seed(run_dir: Path, seed_dir: str) -> list[dict]:
    base = run_dir / "seeds" / seed_dir / "runs" / "PXD071205_LFQ_full_saber"
    manifest = base / "artifacts" / "artifact_manifest.json"
    # conditions/species are dataset constants: pull them from the prepared input
    # AnnData the campaign trained on (obs.condition / var.species per saber.evaluate.lfq).
    import anndata as ad

    prepared = run_dir / "seeds" / seed_dir / "prepared_inputs" / "PXD071205_LFQ"
    h5 = sorted(prepared.glob("*.h5ad"))[0]
    adata = ad.read_h5ad(h5)
    conditions = normalize_conditions(adata.obs["condition"].to_numpy())
    species = normalize_species(adata.var["species"].to_numpy())
    x_raw = np.asarray(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, dtype=np.float32)

    imp = np.load(base / "artifacts" / "imputation.npz")
    observed_mask = imp["observed_mask"] if "observed_mask" in imp.files else np.isfinite(imp["x_input"])
    if observed_mask.shape != x_raw.shape:
        # masks live in model space identical to raw here for LFQ (no cell filtering)
        raise ValueError(f"mask shape {observed_mask.shape} != raw {x_raw.shape}")

    filled = _species_block_mean_baseline(x_raw, observed_mask, conditions, species)
    records = compute_per_protein_log2fc(filled, conditions, species)
    summary = summarize_lfq_records(records)
    splits = _split_masks(x_raw)
    rows = [dict(metric=k, value=v, lfq_subset="all") for k, v in summary.items() if v is not None]
    for subset, mask in splits.items():
        if int(mask.sum()) == 0:
            continue
        sub = summarize_lfq_records(
            compute_per_protein_log2fc(filled[:, mask], conditions, species[mask])
        )
        rows += [dict(metric=k, value=v, lfq_subset=subset) for k, v in sub.items() if v is not None]
    seed = int(seed_dir.removeprefix("seed"))
    for r in rows:
        r.update(task="lfq", dataset="PXD071205_LFQ", method="species_mean",
                 condition="saber", seed=seed, scale_mode="none",
                 artifact_path=str(manifest), holdout_mode=np.nan, mask_key=np.nan,
                 method_kind="baseline", missing_fraction=np.nan, mnar_fraction=np.nan,
                 protein_embedding_layer=np.nan, run_name="PXD071205_LFQ_species_only",
                 target_fraction=np.nan)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="paper_benchmark_20260801_final_10seed")
    ap.add_argument("--seeds", nargs="*", default=["seed42", "seed123", "seed2024", "seed2025",
                                                    "seed7", "seed2026", "seed2027", "seed2028",
                                                    "seed2029", "seed2030"],
                    help="campaign seed dirs (default: the ten-seed campaign used in the paper)")
    ap.add_argument("--merge", action="store_true", help="update source_metrics_long.csv in place")
    ap.add_argument("--out", default=None, help="optional standalone CSV path for the rows")
    args = ap.parse_args()

    from saber.paths import campaign_run_dir
    run_dir = campaign_run_dir(args.run)
    all_rows: list[dict] = []
    for sd in args.seeds:
        rows = evaluate_seed(run_dir, sd)
        all_rows += rows
        head = {r["metric"]: round(r["value"], 3) for r in rows
                if r["lfq_subset"] == "has_missing" and r["metric"] in
                ("mean_abs_error", "mean_abs_error_human", "mean_abs_error_yeast", "mean_abs_error_ecoli")}
        print(f"{sd}: {head}")

    df = pd.DataFrame(all_rows)
    out = Path(args.out) if args.out else run_dir / "source_data" / "lfq_species_only_rows.csv"
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows -> {out}")

    if args.merge:
        long_csv = run_dir / "source_data" / "source_metrics_long.csv"
        old = pd.read_csv(long_csv)
        old = old[~((old.method == "species_mean") & (old.task == "lfq"))]
        cols = [c for c in old.columns if c in df.columns]
        merged = pd.concat([old, df[cols]], ignore_index=True)
        merged.to_csv(long_csv, index=False)
        print(f"merged species_mean rows into {long_csv} (now {len(merged)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
