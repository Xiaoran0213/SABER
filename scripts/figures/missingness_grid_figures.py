"""SABER supp figure — missingness robustness grid (0723 plan §1).

Three panels per dataset (+ pooled): SABER mean Pearson over the missing-fraction x
MNAR-share grid; SABER minus cell-mean; SABER minus no-PLM (where the sequence prior
starts to help as imputation gets harder). Reads the imputation rows of
source_metrics_long.csv (produced by the flagged missingness-grid arm).

Run:  conda run -n saber python scripts/figures/missingness_grid_figures.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import style  # noqa: E402
from style import (  # noqa: E402
    get_plt, save_panel, write_source, log_panel, write_manifest, reset_manifest,
    dataset_label, DATASETS,
)

plt = get_plt()
METRIC = "pearson_missing_mean"


def _load(csv: str | None) -> pd.DataFrame:
    df = style.load_long() if not csv else pd.read_csv(csv)
    df = df[df.task == "imputation"].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for c in ("missing_fraction", "mnar_fraction", "seed"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["value", "missing_fraction", "mnar_fraction"])


def _series(df: pd.DataFrame, method: str, condition: str | None) -> pd.DataFrame:
    """mean over seeds of METRIC by (dataset, missing_fraction, mnar_fraction)."""
    if condition is None:
        sub = df[df.method == method]
    else:
        sub = df[(df.method == method) & (df.condition.fillna("") == condition)]
    sub = style.select_metric(sub, "imputation", METRIC)
    per = sub.groupby(["dataset", "missing_fraction", "mnar_fraction", "seed"])["value"].mean().reset_index()
    return per.groupby(["dataset", "missing_fraction", "mnar_fraction"])["value"].mean().reset_index()


def _pivot(s: pd.DataFrame, dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = s[s.dataset == dataset]
    fr = np.sort(d.missing_fraction.unique())
    mn = np.sort(d.mnar_fraction.unique())
    M = np.full((len(fr), len(mn)), np.nan)
    for _, r in d.iterrows():
        M[list(fr).index(r.missing_fraction), list(mn).index(r.mnar_fraction)] = r.value
    return fr, mn, M


def _heat(ax, fr, mn, M, title, diverging=False):
    cmap = "RdBu_r" if diverging else "viridis"
    if diverging:
        lo, hi = np.nanmin(M), np.nanmax(M)
        if lo >= 0:
            # all differences one-signed (e.g. SABER - cell mean): scale to the
            # data range, a symmetric +-max norm made every cell the same dark red
            vmin, vmax = lo, hi
        else:
            mx = max(abs(lo), abs(hi)) or 1.0
            vmin, vmax = -mx, mx
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap)
    ax.grid(False)  # house grid drew white lines through the cell annotations
    ax.set_xticks(range(len(mn))); ax.set_xticklabels([f"{x*100:.0f}" for x in mn], fontsize=7)
    ax.set_yticks(range(len(fr))); ax.set_yticklabels([f"{x*100:.0f}" for x in fr], fontsize=7)
    ax.set_xlabel("MNAR share (%)", fontsize=8); ax.set_ylabel("Missing fraction (%)", fontsize=8)
    ax.set_title(title, fontsize=9)
    vlo, vhi = im.get_clim()
    cmap_obj = plt.get_cmap(cmap)
    for i in range(len(fr)):
        for j in range(len(mn)):
            if np.isfinite(M[i, j]):
                r, g, b, _ = cmap_obj((M[i, j] - vlo) / (vhi - vlo) if vhi > vlo else 0.5)
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if lum < 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def make_panels(csv: str | None = None):
    reset_manifest()
    df = _load(csv)
    saber = _series(df, "saber", "saber")
    noplm = _series(df, "saber", "noplm")
    cmean = _series(df, "cell_mean", None)
    if saber.empty:
        print("[missingness_grid] no imputation rows found; nothing to plot (run the flagged grid arm first).")
        return
    datasets = [d for d in DATASETS if d in set(saber.dataset)] + ["_pooled"]

    for ds in datasets:
        s = saber if ds == "_pooled" else saber[saber.dataset == ds]
        n = noplm if ds == "_pooled" else noplm[noplm.dataset == ds]
        c = cmean if ds == "_pooled" else cmean[cmean.dataset == ds]
        fr, mn, S = (_pivot(s, "_pooled") if ds == "_pooled" else _pivot(saber, ds))
        _, _, C = (_pivot(c, "_pooled") if ds == "_pooled" else _pivot(cmean, ds))
        _, _, N = (_pivot(n, "_pooled") if ds == "_pooled" else _pivot(noplm, ds))
        if not np.isfinite(S).any():
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
        lab = "pooled" if ds == "_pooled" else dataset_label(ds)
        _heat(axes[0], fr, mn, S, f"{lab} — SABER Pearson")
        _heat(axes[1], fr, mn, S - C, f"{lab} — SABER − cell mean", diverging=True)
        _heat(axes[2], fr, mn, S - N, f"{lab} — SABER − no-PLM", diverging=True)
        fig.tight_layout()
        name = f"figS_missingness_grid_{ds}"
        save_panel(fig, name, task="missingness_grid"); plt.close(fig)
        write_source(s.rename(columns={"value": "saber_pearson"}), name, task="missingness_grid")
        log_panel("S", "missingness_heatmap", f"missingness_grid/{name}.png",
                  f"missingness_grid/{name}_source.csv",
                  f"{lab}: SABER imputation Pearson over missing×MNAR grid + Δcell-mean + Δno-PLM (mean over seeds)",
                  n_rows=len(s), task="missingness_grid")
    write_manifest(task="missingness_grid")
    print("[missingness_grid] done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=None, help="override source_metrics_long.csv (for testing on pilot data)")
    args = ap.parse_args()
    if args.csv:
        # write panels + source CSVs into the run that owns the --csv (outputs/<run>/...),
        # not style's default run.
        style.configure(Path(args.csv).resolve().parent.parent.name)
    make_panels(args.csv)
