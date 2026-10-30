"""SI (2026-10-13, group-meeting Q&A): reconstruction fidelity at OBSERVED entries.

Grouped bar (dataset x {SABER, SABER no PLM} x {observed, missing} @ miss80
mnar50) of per-protein Pearson. Answers "is the reconstruction close where the
truth was not masked?" and pairs it with the missing-entry score of the SAME
runs, so the observed->missing drop is read directly.

WHY ONLY SABER + no-PLM ARE PLOTTED (verified 2026-10-13 on grid artifacts):
PIMMS-DAE, PIMMS-VAE, omicsGMF and KNN return the observed values UNCHANGED at
unmasked positions — their predictions are bit-identical to the input there
(max |pred - true| = 0, per-protein Pearson = 1 by construction). Observed-entry
fidelity is therefore only meaningful for methods that reconstruct through a
bottleneck (SABER and its no-PLM ablation). All methods' observed-entry numbers
are still written to the long CSV for transparency; cell mean is computed on the
fly exactly as saber/evaluate/imputation.py does; protein mean is a per-protein
constant and undefined at per-protein correlation.

Data: existing artifacts only, zero retraining.

Run:  conda run -n saber python scripts/figures/observed_reconstruction_figures.py \
          --run paper_benchmark_20260904_grid
"""
from __future__ import annotations

import argparse
import glob
import re

import numpy as np
import pandas as pd

import style  # noqa: E402
from style import (  # noqa: E402
    get_plt, save_panel, write_source, log_panel, write_manifest,
    dataset_label, DATASETS, color_for,
)
from imputation_figures import DS_ORDER  # noqa: E402  canonical dataset order

plt = get_plt()

# npz key inside the *saber* run artifact (noplm read from its own run)
PRED_KEYS = {
    "SABER": "pred__saber",
    "SABER no PLM": "noplm",    # sentinel: read pred__saber from the noplm run
    "PIMMS-DAE": "pred__pimms_dae",
    "omicsGMF": "pred__omicsgmf",
    "KNN": "pred__knn",
    "Cell mean": None,          # computed on the fly
}
PLOT_METHODS = ["SABER", "SABER no PLM"]
PLOT_POS = [("observed", False), ("missing", True)]   # (position, hatched)


def _cell_mean(x_input: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    obs = observed_mask.astype(bool) & np.isfinite(x_input)
    counts = obs.sum(axis=1, keepdims=True)
    sums = np.where(obs, x_input, 0.0).sum(axis=1, keepdims=True, dtype=np.float64)
    means = np.zeros((x_input.shape[0], 1), dtype=np.float32)
    np.divide(sums, counts, out=means, where=counts > 0)
    return np.broadcast_to(means, x_input.shape).copy().astype(np.float32)


def _per_protein_pearson(x_true, pred, mask):
    """Mean per-protein Pearson over `mask`. Mirrors
    saber/evaluate/imputation.py::_per_protein_metrics (>=2 valid cells, finite
    guard, constant predictions dropped)."""
    vals = []
    for j in range(x_true.shape[1]):
        m = mask[:, j] & np.isfinite(x_true[:, j]) & np.isfinite(pred[:, j])
        if m.sum() >= 2:
            t, p = x_true[m, j], pred[m, j]
            if p.std() < 1e-12:
                continue
            c = float(np.corrcoef(t, p)[0, 1])
            if np.isfinite(c):
                vals.append(c)
    return float(np.mean(vals)) if vals else float("nan")


def _seed_from_path(p: str) -> int:
    return int(re.search(r"seed(\d+)", p).group(1))


def collect(run_dir, datasets: list[str]) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        pat = str(run_dir / "seeds" / "seed*" / "runs" / f"{ds}_miss80_mnar50_saber"
                  / "artifacts" / "imputation.npz")
        for p in sorted(glob.glob(pat)):
            seed = _seed_from_path(p)
            with np.load(p) as d:
                x_true = d["x_true"]
                obs_mask = d["observed_mask"].astype(bool)
                inj_mask = d["injected_missing_mask"].astype(bool)
                x_input = d["x_input"]
                preds = {lbl: np.asarray(d[key], dtype=np.float32)
                         for lbl, key in PRED_KEYS.items() if key and key != "noplm"}
                preds["Cell mean"] = _cell_mean(x_input, obs_mask)
            noplm = str(p).replace("_mnar50_saber", "_mnar50_noplm")
            try:
                with np.load(noplm) as d:
                    preds["SABER no PLM"] = np.asarray(d["pred__saber"], dtype=np.float32)
            except FileNotFoundError:
                pass
            for lbl, pred in preds.items():
                rows.append(dict(
                    dataset=ds, dataset_label=dataset_label(ds), seed=seed,
                    method_label=lbl,
                    observed_pearson_mean=_per_protein_pearson(x_true, pred, obs_mask),
                    missing_pearson_mean=_per_protein_pearson(x_true, pred, inj_mask)))
            print(f"  {ds} seed{seed}: done", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="campaign run dir under outputs/")
    args = ap.parse_args()
    style.configure(args.run)

    print(f"collecting observed/missing per-protein Pearson from {style.RUN_DIR} ...")
    long = collect(style.RUN_DIR, DATASETS)
    out_long = style.SRC_ROOT / "imputation" / "observed_reconstruction_long.csv"
    out_long.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(out_long, index=False)

    # figure: dataset groups x {SABER, no PLM} x {observed (solid), missing (hatch)}
    summ = (long[long.method_label.isin(PLOT_METHODS)]
            .melt(id_vars=["dataset", "method_label"], value_vars=["observed_pearson_mean", "missing_pearson_mean"],
                  var_name="position", value_name="value")
            .assign(position=lambda t: t.position.str.replace("_pearson_mean", "", regex=False))
            .groupby(["dataset", "method_label", "position"])["value"]
            .agg(["mean", "std", "count"]).reset_index().rename(columns={"std": "sd"}))
    summ["sd"] = summ["sd"].fillna(0.0)

    bars = [(m, pos) for m in PLOT_METHODS for pos, _ in PLOT_POS]
    n = len(bars); w = 0.82 / n
    offs = (np.arange(n) - (n - 1) / 2) * w
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for i, (m, pos) in enumerate(bars):
        means, sds = [], []
        for ds in DS_ORDER:
            r = summ[(summ.dataset == ds) & (summ.method_label == m) & (summ.position == pos)]
            means.append(float(r["mean"].iloc[0]) if not r.empty else np.nan)
            sds.append(float(r["sd"].iloc[0]) if not r.empty else 0.0)
        ax.bar(np.arange(len(DS_ORDER)) + offs[i], means, width=w * 0.92, yerr=sds,
               capsize=1.6, color=color_for(m), edgecolor="white",
               hatch="//" if dict(PLOT_POS)[pos] else None,
               error_kw=dict(lw=0.8, ecolor="#222222"),
               label=f"{m} ({pos})")
    ax.set_xticks(np.arange(len(DS_ORDER)))
    ax.set_xticklabels([dataset_label(d) for d in DS_ORDER], rotation=18, ha="right")
    ax.set_ylabel("Pearson r")
    ax.legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.26),
              frameon=False)
    name = "figS_imputation_observed_vs_missing"
    save_panel(fig, name); plt.close(fig)
    write_source(summ, name)
    log_panel("2a-si-obs", "observed_vs_missing_bar", f"{name}.png", f"{name}_source.csv",
              "per-protein Pearson at OBSERVED vs injected-MISSING entries @ miss80 mnar50 "
              "(SABER + no-PLM only: external imputers copy observed values verbatim, "
              "Pearson = 1 by construction there), 10-seed mean±SD",
              n_rows=len(summ),
              params=dict(methods=PLOT_METHODS, missing_fraction="0.8"))
    write_manifest(task="imputation", name="observed_reconstruction_manifest.json")
    print(f"\nDONE. figure -> {style.FIG_ROOT / 'imputation' / (name + '.png')}")
    print(f"long csv -> {out_long}")
    print(summ.pivot(index=["method_label", "position"], columns="dataset",
                     values="mean").round(3))


if __name__ == "__main__":
    main()
