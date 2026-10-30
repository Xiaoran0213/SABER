#!/usr/bin/env python3
"""Layer (b) coherence figure (multi-seed, mean±sd).

Panel A: # significant GO BP terms per cell type by method @ 80% holdout
         (PLM-dependence headline: SABER/refs >> no-PLM/cell-mean).
Panel B: SABER Derks PDAC top GO BP terms (-log10 FDR) -- keratin showcase (seed42).
Panel C: SABER #sig GO BP terms across holdout {50,80,90}% per cell type.

Reads the aggregated coherence_summary_{ds}_holdout{frac}_agg.csv (mean±sd across
seeds) produced by go_coherence.py, + seed42 ORA detail for the showcase panel.
Relocated from outputs/proteome-wide/scripts/ -> scripts/proteome_wide/.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from saber.paths import outputs_root  # noqa: E402
RES = outputs_root() / "proteome-wide" / "results" / "go"
OUT = outputs_root() / "proteome-wide" / "figures"
DEFAULT_SEEDS = (42, 123, 2024)

METHODS = [("saber", "SABER", "#0072B2"), ("plm_ridge", "PLM-ridge", "#CC79A7"),
           ("plm_knn", "PLM-kNN", "#009E73"), ("oracle", "oracle (truth)", "#E69F00"),
           ("noplm", "no-PLM", "#D55E00"), ("cell_mean", "cell mean", "#999999")]
DATASETS = [("Derks_2022", ["Melanoma", "PDAC", "U-937"]), ("Specht_2019", ["sc_m0", "sc_u"])]


def _agg_row(ds: str, fr: int, cell: str, method: str):
    p = RES / f"coherence_summary_{ds}_holdout{fr}_agg.csv"
    if not p.exists():
        return None
    for r in csv.DictReader(open(p)):
        if r["cell_type"] == cell and r["method"] == method:
            return r
    return None


def n_sig_agg(ds: str, fr: int, cell: str, method: str, col="n_sig_GO_BP"):
    r = _agg_row(ds, fr, cell, method)
    if not r:
        return 0.0, 0.0
    m = r.get(f"{col}_mean", "0"); s = r.get(f"{col}_sd", "0")
    return (float(m) if m not in ("", "nan") else 0.0), (float(s) if s not in ("", "nan") else 0.0)


def main() -> int:
    import argparse
    global RES, OUT
    _p = argparse.ArgumentParser(); _p.add_argument("--out-suffix", default=""); suf = _p.parse_args().out_suffix
    RES = outputs_root() / f"proteome-wide{suf}" / "results" / "go"
    OUT = outputs_root() / f"proteome-wide{suf}" / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 4.4))
    axA = fig.add_subplot(1, 3, 1)
    axB = fig.add_subplot(1, 3, 2)
    axC = fig.add_subplot(1, 3, 3)

    # Panel A: grouped bars @80% with error bars (mean±sd across seeds)
    cells = [(ds, c) for ds, cs in DATASETS for c in cs]
    labels = [f"{c}\n({ds[:5]})" for ds, c in cells]
    x = np.arange(len(cells))
    bw = 0.13
    for i, (mkey, mlab, col) in enumerate(METHODS):
        means = [n_sig_agg(ds, 80, c, mkey)[0] for ds, c in cells]
        sds = [n_sig_agg(ds, 80, c, mkey)[1] for ds, c in cells]
        axA.bar(x + (i - 2.5) * bw, means, bw, yerr=sds, capsize=2,
                label=mlab, color=col, edgecolor="white", linewidth=0.4)
    axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=8)
    axA.set_ylabel("# significant GO BP terms (FDR<0.05)")
    axA.set_title("A.  Proteome-wide coherence @ 80% holdout", fontsize=10)
    axA.axhline(0, color="#cccccc", linewidth=0.8)
    axA.legend(fontsize=7, ncol=2, loc="upper right")
    axA.grid(axis="y", alpha=0.25)

    # Panel B: SABER Derks PDAC top GO BP terms (seed42 showcase)
    detail = RES / "Derks_2022_holdout80_seed42" / "ora_details" / "saber__PDAC__go.csv"
    rows = list(csv.DictReader(open(detail)))[:8][::-1] if detail.exists() else []
    names = [r["term_name"][:34] for r in rows]
    neglog = [-np.log10(max(float(r["fdr"]), 1e-300)) for r in rows]
    yB = np.arange(len(rows))
    axB.barh(yB, neglog, color="#0072B2", edgecolor="white", linewidth=0.4)
    axB.set_yticks(yB); axB.set_yticklabels(names, fontsize=8)
    axB.set_xlabel("-log10(FDR)")
    axB.set_title("B.  SABER Derks PDAC top GO terms\n(keratin filament is canonical)", fontsize=10)
    axB.axvline(-np.log10(0.05), color="#D55E00", linestyle="--", linewidth=1, label="FDR=0.05")
    axB.legend(fontsize=7, loc="lower right")
    axB.grid(axis="x", alpha=0.25)

    # Panel C: SABER #sig GO BP terms across fractions, mean±sd
    fracs = [50, 80, 90]
    for ds, cs in DATASETS:
        for c in cs:
            ys = [n_sig_agg(ds, fr, c, "saber")[0] for fr in fracs]
            ysd = [n_sig_agg(ds, fr, c, "saber")[1] for fr in fracs]
            xs = [f / 100 for f in fracs]
            axC.errorbar(xs, ys, yerr=ysd, marker="o", markersize=5, capsize=2,
                         label=f"{c} ({ds[:5]})")
    axC.set_xticks([0.5, 0.8, 0.9])
    axC.set_xlabel("held-out protein fraction")
    axC.set_ylabel("SABER # significant GO BP terms")
    axC.set_title("C.  Coherence is robust to holdout fraction", fontsize=10)
    axC.legend(fontsize=7, ncol=2, loc="best")
    axC.grid(alpha=0.25)

    fig.suptitle(f"Proteome-wide expansion: predicted cell-type markers are biologically coherent (PLM-dependent, n={len(DEFAULT_SEEDS)} seeds)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"proteome_wide_go_coherence.{ext}", dpi=180)
    print(f"[plot_coherence] wrote {OUT}/proteome_wide_go_coherence.png|pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
