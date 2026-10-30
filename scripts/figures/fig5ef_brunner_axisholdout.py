#!/usr/bin/env python
"""Brunner cell-cycle panels of the merged Fig 5 (axis-holdout protocol).

  5e  class oscillation gradient by method: gold-APC (n=8 in-matrix) /
      degron (n=621) / ribosomal (n=69), z-consistent metric, 22 cycle-score
      markers removed from every method's training matrix. SABER representative
      seed + 10-seed whisker on the gold class.
  5f-h  one protein per panel (NCAPH, NCAPD2, RPS14): observed boxes and
      points, SABER reconstructed on every cell (including measured cells),
      dashed line through the reconstructed medians.

Sources: $SABER_OUTPUTS_ROOT/easyscp_expansion_logs/brunner_metrics_axisholdout_{methods,perseed}.csv
         $SABER_OUTPUTS_ROOT/brunner_realimpute_axisholdout/matrices.npz
Writes:  $SABER_FIGURES_ROOT/fig5/fig5{e,f}_*
Run:     python scripts/figures/fig5ef_brunner_axisholdout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))  # resolve `saber` from this checkout, not the env
sys.path.insert(0, str(REPO / "scripts" / "figures"))
from saber.paths import data_root, figures_root, outputs_root  # noqa: E402
import style  # noqa: E402
from style import get_plt  # noqa: E402

plt = get_plt()

import argparse as _argparse
_ap = _argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--out", default="", help="figure output dir (default: $SABER_FIGURES_ROOT/fig5)")
_args = _ap.parse_args()

B = str(data_root() / "brunner2022")
OUTDIR = str(outputs_root() / "brunner_realimpute_axisholdout")
LOGS = outputs_root() / "easyscp_expansion_logs"
OUT = Path(_args.out) if _args.out else figures_root() / "fig5"
OUT.mkdir(parents=True, exist_ok=True)

C_SAB, C_PIMS, C_OGMF, C_KNN = "#0072B2", "#009E73", "#CC79A7", "#8C6D8F"
C_CMEAN, C_OBS = "#999999", "#9e9e9e"


def _save(fig, name, src_df=None):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    if src_df is not None:
        src_df.to_csv(OUT / f"{name}_source.csv", index=False)
    plt.close(fig)
    print("panel ->", OUT / name)


def panel_5e():
    """Delta bars: oscillation-ordering difference per method,
    Δ = mean |g| (APC/C substrates) − mean |g| (ribosomal).
    Positive Δ = the biology-prescribed ordering (substrates above ribosomal).
    Zero line = no ordering; the raw class means sit on an axis-agnostic floor
    (~0.27 against random axes), which the difference cancels. SABER carries a
    ten-seed error bar; other methods are point estimates (3-seed mean for the
    externals, deterministic for KNN / cell-mean / observed).
    """
    df = pd.read_csv(LOGS / "brunner_metrics_axisholdout_methods.csv")
    order = ["observed", "cell-mean", "KNN", "omicsgmf", "pimms_dae", "SABER"]
    labels = ["observed", "cell-mean", "KNN", "omicsGMF", "PIMMS-DAE", "SABER"]
    colors = [C_OBS, "#E69F00", C_KNN, C_OGMF, C_PIMS, C_SAB]
    deltas = {m: float(df[df.method == m].iloc[0].gold - df[df.method == m].iloc[0].ribo)
              for m in order}
    src = pd.DataFrame([dict(method=lab, delta=deltas[m]) for m, lab in zip(order, labels)])
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(order))
    ax.bar(x, [deltas[m] for m in order], width=0.6, color=colors)
    ax.axhline(0, color="0.2", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=18, ha="right")
    ax.set_ylabel(r"$\Delta$ = substrates $-$ ribosomal ($|g|$)")
    ax.set_ylim(-0.17, 0.08)
    _save(fig, "fig5e_brunner_classgrad", src)


def panel_5f(showcase=(("NCAPH", "f", "APC/C"),
                       ("NCAPD2", "g", "APC/C"),
                       ("RPS14", "h", "ribosomal")),
             nbins=10):
    """Per-protein cell-cycle panels (f, g, h).

    Ten equal-count bins of the cell-cycle score. Per bin: observed cells
    (grey boxes and points) and SABER reconstructed on every cell (light
    blue), including cells in which the protein was measured. A dashed line
    joins the medians of the reconstructed boxes. The reconstructed matrix
    is the ten-seed ensemble of the model output, not observed values with
    holes filled.
    """
    a = ad.read_h5ad(f"{B}/prepared/Brunner2022.h5ad")
    cc = np.asarray(a.obs["cell_cycle_score"], float)
    d = pd.read_csv(f"{B}/brunner_derisk2_results.csv")
    ac2g = {ac: (str(g) if pd.notna(g) else "") for ac, g in zip(d.AC_base, d.gene)}
    Z = dict(np.load(f"{OUTDIR}/matrices.npz"))
    acs = [str(x) for x in Z["acs_keep"]]
    genes = [ac2g.get(ac, "") for ac in acs]
    g2i = {g: i for i, g in enumerate(genes)}
    obs = np.asarray(Z["observed_reduced"], float)
    sab = np.asarray(Z["saber_seeds"], float).mean(0)   # ensemble mean, model output
    sc = np.nanmean(np.where(np.isfinite(obs), obs, np.nan), axis=0)
    ss = np.nanstd(np.where(np.isfinite(obs), obs, np.nan), axis=0)
    ss = np.where(np.isfinite(ss) & (ss > 1e-8), ss, 1.0)
    zobs = (obs - sc) / ss
    zm = np.nanmean(np.where(np.isfinite(zobs), zobs, np.nan), axis=1)
    bins = pd.qcut(cc, nbins, labels=False)

    prepared = []
    wlos, whis = [], []
    for gene, letter, kind in showcase:
        j = g2i[gene]
        dc_obs = zobs[:, j] - zm
        dc_sab = sab[:, j] - zm
        det = np.isfinite(dc_obs)
        per_bin = []
        for b in range(nbins):
            msk = bins == b
            vo = dc_obs[msk][det[msk]]
            v_all = dc_sab[msk]
            per_bin.append((vo, v_all))
            for v in (vo, v_all):
                if len(v) == 0:
                    continue
                q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
                wlos.append(max(v.min(), q1 - 1.5 * (q3 - q1)))
                whis.append(min(v.max(), q3 + 1.5 * (q3 - q1)))
        prepared.append((gene, letter, kind, det.mean(), per_bin))
    ylo, yhi = min(wlos), max(whis)
    pad = 0.08 * (yhi - ylo)

    def bx(ax, x, v, fc, ec, w=0.32):
        if len(v) == 0:
            return
        q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
        ax.bxp([{"med": np.median(v), "q1": q1, "q3": q3,
                 "whislo": max(v.min(), q1 - 1.5 * (q3 - q1)),
                 "whishi": min(v.max(), q3 + 1.5 * (q3 - q1))}],
               positions=[x], widths=w, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=fc, edgecolor=ec),
               medianprops=dict(color=ec, lw=1.2),
               whiskerprops=dict(color=ec), capprops=dict(color=ec))

    rng = np.random.RandomState(0)
    for gene, letter, kind, det_frac, per_bin in prepared:
        fig, ax = plt.subplots(figsize=(4.0, 3.15))
        med_x, med_y = [], []
        rows = []
        for b, (vo, v_all) in enumerate(per_bin):
            bx(ax, b - 0.18, vo, "0.78", "0.35")
            bx(ax, b + 0.18, v_all, "#7fb3d8", "#2878b5")
            if len(vo):
                jit = rng.uniform(-0.07, 0.07, len(vo))
                ax.plot(np.full(len(vo), b - 0.18) + jit, vo, ".", ms=2.2,
                        color="0.15", alpha=0.55, zorder=3)
            if len(v_all):
                med_x.append(b + 0.18)
                med_y.append(float(np.median(v_all)))
            rows.append(dict(gene=gene, bin=b, n_observed=int(len(vo)),
                             n_cells=int(len(v_all)),
                             median_observed=float(np.median(vo)) if len(vo) else np.nan,
                             median_reconstructed=float(np.median(v_all)) if len(v_all) else np.nan))
        ax.plot(med_x, med_y, ls="--", color="#2878b5", lw=1.3, zorder=4)
        counts = [r["n_observed"] for r in rows]
        ax.set_xticks(range(nbins))
        ax.set_xticklabels([str(n) for n in counts], fontsize=7)
        ax.set_xlabel("cell cycle score", fontsize=8)
        ax.set_ylabel("abundance (z)", fontsize=8)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(f"{gene} ({kind})", fontsize=9, pad=26)
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor="0.78", edgecolor="0.35"),
                   plt.Rectangle((0, 0), 1, 1, facecolor="#7fb3d8", edgecolor="#2878b5")]
        ax.legend(handles, ["observed", "SABER reconstructed"],
                  fontsize=7, frameon=False, loc="lower left",
                  bbox_to_anchor=(0.0, 1.02), ncol=2, borderaxespad=0,
                  handlelength=1.2, columnspacing=0.8)
        _save(fig, f"fig5{letter}_{gene}", pd.DataFrame(rows))


if __name__ == "__main__":
    panel_5e()
    panel_5f()
