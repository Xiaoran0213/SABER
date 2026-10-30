#!/usr/bin/env python
"""Supplementary Fig. S11 — per-substrate oscillation-gradient deltas (Brunner).

One panel per in-matrix gold APC/C substrate (n=8): Delta = |g_substrate| -
mean|g_ribosomal| for the six Fig 5e methods (observed / cell-mean / KNN /
omicsGMF / PIMMS-DAE / SABER), in the z-consistent axis-holdout metric.
Gradients are recomputed per protein from the axis-holdout matrices (same
recipe as scripts/case_study/brunner_metrics_axisholdout.py); deterministic,
no model rerun. Shared y-limits across panels.

Writes: $SABER_FIGURES_ROOT/supp/figS11_brunner_per_substrate.{png,pdf}
Run:    conda run -n saber python scripts/figures/figS11_brunner_per_substrate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))  # resolve `saber` from this checkout, not the env
sys.path.insert(0, str(REPO / "scripts" / "figures"))
sys.path.insert(0, str(REPO / "scripts" / "case_study"))

import style  # noqa: E402
from style import get_plt  # noqa: E402

plt = get_plt()

from saber.paths import data_root, outputs_root  # noqa: E402
from saber.paths import figures_root  # noqa: E402

BRUNNER = str(data_root() / "brunner2022")
OUTDIR = str(outputs_root() / "brunner_realimpute_axisholdout")
LOGS = outputs_root() / "easyscp_expansion_logs"
SUPP = figures_root() / "supp"
ASM_S = SUPP  # single destination; kept as an alias for the save helper

METHODS = ["observed", "cell-mean", "KNN", "omicsgmf", "pimms_dae", "SABER"]
LABELS = ["observed", "cell-mean", "KNN", "omicsGMF", "PIMMS-DAE", "SABER"]
COLORS = ["#9e9e9e", "#E69F00", "#8C6D8F", "#CC79A7", "#009E73", "#0072B2"]


def main():
    import anndata as ad
    a = ad.read_h5ad(f"{BRUNNER}/prepared/Brunner2022.h5ad")
    cc = np.asarray(a.obs["cell_cycle_score"], float)
    Z = dict(np.load(f"{OUTDIR}/matrices.npz"))
    acs = [str(x) for x in Z["acs_keep"]]
    obs_raw = np.asarray(Z["observed_reduced"], float)
    sc = np.nanmean(np.where(np.isfinite(obs_raw), obs_raw, np.nan), axis=0)
    ss = np.nanstd(np.where(np.isfinite(obs_raw), obs_raw, np.nan), axis=0)
    ss = np.where(np.isfinite(ss) & (ss > 1e-8), ss, 1.0)
    zobs = (obs_raw - sc) / ss
    zcm = np.nanmean(np.where(np.isfinite(zobs), zobs, np.nan), axis=1)

    def corr(x, y):
        msk = np.isfinite(x) & np.isfinite(y)
        if msk.sum() < 20 or np.std(x[msk]) < 1e-9 or np.std(y[msk]) < 1e-9:
            return np.nan
        return float(np.corrcoef(x[msk], y[msk])[0, 1])

    def grads(Mz):
        return np.array([corr(Mz[:, j] - zcm, cc) for j in range(Mz.shape[1])])

    def z_of(M):
        return (np.asarray(M, float) - sc) / ss

    mats = {"observed": zobs,
            "SABER": np.asarray(Z["saber_seeds"], float)[0],
            "KNN": z_of(Z["knn"]),
            "cell-mean": z_of(Z["cell-mean"] if "cell-mean" in Z else Z["cellmean"])}
    ext = {}
    for k in Z:
        if k.startswith(("omicsgmf_seed", "pimms_dae_seed")):
            ext.setdefault(k.rsplit("_seed", 1)[0], []).append(np.asarray(Z[k], float))
    for nm, Ms in ext.items():
        mats[nm] = z_of(np.mean(Ms, axis=0))

    G = {nm: np.abs(grads(M)) for nm, M in mats.items()}

    pp = pd.read_csv(LOGS / "brunner_metrics_axisholdout_perprotein.csv")
    ribo_ac = set(pp.loc[pp.ribo, "AC"])
    ribo_mean = {nm: float(np.mean([G[nm][i] for i, ac in enumerate(acs)
                                    if ac in ribo_ac])) for nm in METHODS}

    sub = pp[pp.gold].copy()
    sub["d_SABER"] = sub.AC.map(lambda ac: G["SABER"][acs.index(ac)]) - ribo_mean["SABER"]
    sub = sub.sort_values("d_SABER", ascending=False).reset_index(drop=True)

    # consistency vs the published class aggregates (Fig 5e)
    ref = pd.read_csv(LOGS / "brunner_metrics_axisholdout_methods.csv").set_index("method")
    for nm in METHODS:
        d_mean = np.mean([G[nm][acs.index(ac)] - ribo_mean[nm] for ac in sub.AC])
        diff = d_mean - (ref.loc[nm, "gold"] - ref.loc[nm, "ribo"])
        assert abs(diff) < 1e-3, f"{nm}: aggregate mismatch {diff}"
    print("aggregate cross-check vs Fig 5e: OK (all |diff| < 1e-3)")

    det = dict(zip(pp.AC, pp.obsfrac))
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 5.6), sharey=True)
    all_d = np.array([(G[nm][acs.index(r.AC)] - ribo_mean[nm])
                            for _, r in sub.iterrows() for nm in METHODS])
    ylim = (all_d.min() - 0.05, all_d.max() + 0.05)
    for ax, (_, r) in zip(axes.ravel(), sub.iterrows()):
        deltas = [G[nm][acs.index(r.AC)] - ribo_mean[nm] for nm in METHODS]
        ax.bar(np.arange(6), deltas, width=0.6, color=COLORS)
        ax.axhline(0, color="0.2", lw=0.8)
        ax.set_xticks(np.arange(6))
        ax.set_xticklabels(LABELS, fontsize=6, rotation=25, ha="right")
        ax.set_title(f"{r.gene}  (detected {r.obsfrac:.0%})", fontsize=8.5)
        ax.tick_params(labelsize=6.5)
        ax.set_ylim(*ylim)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Delta$ = substrate $-$ ribosomal ($|g|$)", fontsize=7.5)
    fig.tight_layout()

    SUPP.mkdir(parents=True, exist_ok=True)
    ASM_S.mkdir(parents=True, exist_ok=True)
    for d in (SUPP, ASM_S):
        for ext_ in ("png", "pdf"):
            fig.savefig(d / f"figS11_brunner_per_substrate.{ext_}",
                        dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("->", SUPP / "figS11_brunner_per_substrate")


if __name__ == "__main__":
    main()
