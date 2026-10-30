"""SABER Fig 5 (liver case study): one fitted model on real liver data.

Beyond the expansion panels of Fig. 2, this figure collects the remaining
outputs of the fitted model on the liver-zonation dataset (EasySCP, PXD070185),
evaluated against the genuine 36% missingness:
  5a  real imputation (axis-holdout): mean gate-stratified gradient of the
      zonated proteins after each method fills the genuine missing values,
      with the 215 HSS axis proteins removed from every training matrix
      (de-confounded; in-matrix subset 1,003 + SABER zero-shot 120 separate)
  5b  protein embedding: pericentral vs periportal classification AUC
      (5-fold CV) across marker tiers, SABER vs raw ESM-2 vs omicsGMF
  5c  zone coherence: within-zone minus between-zone cosine similarity of the
      protein embeddings across marker tiers

Writes directly to docs/manuscript/figures/assembly_v2/fig5/ (png+pdf+source
CSV), independent of the campaign paper_figures tree.

Sources:
  outputs/easyscp_expansion_logs/realimpute_metrics_axisholdout.csv  (5a, 2026-09-09
    axis-holdout rerun: the 215 HSS proteins were removed from every method's
    training matrix; bars = in-matrix zonated subset, SABER zero-shot bar separate)
  outputs/easyscp_expansion_logs/proteinemb_metrics_10seed.csv
Run:  conda run -n saber python scripts/figures/fig5_liver_casestudy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))  # resolve `saber` from this checkout, not the env
sys.path.insert(0, str(REPO / "scripts" / "figures"))

from saber.paths import data_root, figures_root, outputs_root  # noqa: E402

import style  # noqa: E402
from style import get_plt  # noqa: E402

plt = get_plt()

import argparse as _argparse
_ap = _argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--suffix", default="", help="run-tree suffix, appended to run names")
_ap.add_argument("--out", default="", help="figure output dir (default: $SABER_FIGURES_ROOT/fig5)")
_args = _ap.parse_args()
SUFFIX = _args.suffix

REALIMPUTE = outputs_root() / "easyscp_expansion_logs" / f"realimpute_metrics_axisholdout{SUFFIX}.csv"
PROTEINEMB = outputs_root() / "easyscp_expansion_logs" / f"proteinemb_metrics_10seed{SUFFIX}.csv"
OUT = Path(_args.out) if _args.out else figures_root() / "fig5"
OUT.mkdir(parents=True, exist_ok=True)

# colours consistent with the Brunner case-study panels (brunner_fig6panels.py)
C_SAB, C_PIMS, C_OGMF, C_KNN = "#0072B2", "#009E73", "#CC79A7", "#8C6D8F"
C_CMEAN, C_OBS, C_RAW = "#999999", "#9e9e9e", "#B3B3B3"


def _save(fig, name, src_df=None):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    if src_df is not None:
        src_df.to_csv(OUT / f"{name}_source.csv", index=False)
    plt.close(fig)
    print("panel ->", OUT / name)


def _pm(s):
    """Parse '0.972±0.040' / '+0.337±0.032' strings from proteinemb CSV."""
    s = str(s)
    if "±" in s:
        m, sd = s.split("±")
        return float(m), float(sd)
    try:
        return float(s), np.nan
    except ValueError:
        return np.nan, np.nan


# --------------------------------------------------------------------------- #
def panel_5a():
    """Real imputation under the axis-holdout protocol (215 HSS proteins removed
    from every training matrix): single series = full-matrix gate-stratified
    gradient of the 1,003 in-matrix zonated proteins. Zero-shot numbers and the
    filled-entries-only decomposition are in the text / Supplementary Note 5.3.
    Y-axis truncated (bars span 0.46-0.49) to resolve the differences; the
    truncation is disclosed in the caption."""
    df = pd.read_csv(REALIMPUTE)
    d = df[(df.tier == "zonated") & (df.subset == "in_matrix")]
    g = d.groupby("method")["signed"].agg(["mean", "std", "count"])
    order = ["observed-only", "cell-mean", "omicsGMF", "KNN", "PIMMS", "SABER"]
    labels = ["observed", "cell-mean", "omicsGMF", "KNN", "PIMMS-DAE", "SABER"]
    colors = [C_OBS, "#E69F00", C_OGMF, C_KNN, C_PIMS, C_SAB]  # cell-mean: orange (was grey)
    src = pd.DataFrame(dict(
        method=labels,
        mean_gradient=[g.loc[m, "mean"] for m in order],
        sd=[g.loc[m, "std"] for m in order],
        n_seeds=[int(g.loc[m, "count"]) for m in order]))

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(src))
    ax.bar(x, src.mean_gradient, width=0.62, color=colors,
           yerr=src.sd.where(src.n_seeds > 1), error_kw=dict(lw=0.8, capsize=2))
    ax.axhline(float(src.mean_gradient[0]), color=C_OBS, lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(src.method, fontsize=8, rotation=18, ha="right")
    ax.set_ylabel("zonation gradient")
    ax.set_ylim(0.44, 0.52)
    _save(fig, "fig5a_liver_realimpute_gradient", src)


# --------------------------------------------------------------------------- #
def panel_5b():
    """Protein embedding: CV/PV classification AUC by marker tier."""
    df = pd.read_csv(PROTEINEMB)
    tiers = ["canonical", "hss_markers", "zonated"]
    # two-line labels with explicit n, same convention as Fig 2a/b
    tier_label = {"canonical": "Canonical\n(n=18)", "hss_markers": "HSS-defined\n(n=215)",
                  "zonated": "Zonated\n(n=1,123)"}
    cols = [("SABER_AUC", "SABER", C_SAB), ("rawESM2_AUC", "raw ESM-2", C_RAW),
            ("omicsGMF_AUC", "omicsGMF", C_OGMF)]
    rows = []
    for t in tiers:
        r = df[df.tier == t].iloc[0]
        for col, lab, _ in cols:
            m, sd = _pm(r[col])
            rows.append({"tier": t, "n": int(r["n"]), "method": lab,
                         "auc_mean": m, "auc_sd": sd})
    src = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    x = np.arange(len(tiers)); w = 0.8 / len(cols)
    for k, (col, lab, c) in enumerate(cols):
        sub = src[src.method == lab]
        ax.bar(x + (k - (len(cols) - 1) / 2) * w, sub.auc_mean, width=w * 0.92,
               color=c, label=lab, yerr=sub.auc_sd, error_kw=dict(lw=0.8, capsize=1.5))
    ax.set_xticks(x)
    ax.set_xticklabels([tier_label[t] for t in tiers], fontsize=8)
    ax.set_ylabel("CV/PV classification AUC")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="lower left",
              bbox_to_anchor=(0, 1.01), columnspacing=1.0, handlelength=1.2,
              handletextpad=0.4)
    _save(fig, "fig5b_liver_proteinemb_auc", src)


# --------------------------------------------------------------------------- #
def panel_5c():
    """Zone coherence: within-zone minus between-zone cosine similarity."""
    df = pd.read_csv(PROTEINEMB)
    tiers = ["canonical", "hss_markers", "zonated"]
    # two-line labels with explicit n, same convention as Fig 2a/b
    tier_label = {"canonical": "Canonical\n(n=18)", "hss_markers": "HSS-defined\n(n=215)",
                  "zonated": "Zonated\n(n=1,123)"}
    cols = [("SABER_gap", "SABER", C_SAB), ("rawESM2_gap", "raw ESM-2", C_RAW),
            ("omicsGMF_gap", "omicsGMF", C_OGMF)]
    rows = []
    for t in tiers:
        r = df[df.tier == t].iloc[0]
        for col, lab, _ in cols:
            m, sd = _pm(r[col])
            rows.append({"tier": t, "n": int(r["n"]), "method": lab,
                         "gap_mean": m, "gap_sd": sd})
    src = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    x = np.arange(len(tiers)); w = 0.8 / len(cols)
    for k, (col, lab, c) in enumerate(cols):
        sub = src[src.method == lab]
        ax.bar(x + (k - (len(cols) - 1) / 2) * w, sub.gap_mean, width=w * 0.92,
               color=c, label=lab, yerr=sub.gap_sd, error_kw=dict(lw=0.8, capsize=1.5))
    ax.axhline(0, color="0.2", lw=0.6, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([tier_label[t] for t in tiers], fontsize=8)
    ax.set_ylabel("within$-$between zone cosine")
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="lower left",
              bbox_to_anchor=(0, 1.01), columnspacing=1.0, handlelength=1.2,
              handletextpad=0.4)
    _save(fig, "fig5c_liver_zone_coherence", src)


# --------------------------------------------------------------------------- #
def panel_5d():
    """Protein-embedding UMAP: curated CV/PV markers separate under SABER but
    not under raw ESM-2 (omicsGMF in between). Promoted from the SI liver
    figure (old figS5d) 2026-09-04."""
    import glob
    import anndata as ad
    import umap
    from sklearn.decomposition import PCA

    BASE = data_root() / "easyscp_liver"
    RUNS = outputs_root()
    CV, PV = "#C0392B", "#2471A3"          # same zone colours as the overview heatmap
    acs = list(ad.read_h5ad(BASE / "prepared" / "EasySCP_liver.h5ad").var_names)
    E = np.load(sorted(glob.glob(str(RUNS / f"easyscp_cellembed_saber{SUFFIX}"
                                    / "*" / "artifacts" / "embeddings.npz")))[-1]
                )["protein_embeddings__saber"]
    rf = np.load(BASE / "prepared" / "EasySCP_liver_features.npz")
    raw = rf["features"] if "features" in rf else rf["embeddings"]
    sidecar = RUNS / f"easyscp_realimpute_pilot{SUFFIX}" / "externals" / "wd_omicsgmf_seed42" / "omicsgmf_imputed.csv.protein.csv"
    omg = (np.loadtxt(sidecar, delimiter=",") if sidecar.is_file()
           else np.load(RUNS / f"easyscp_realimpute_pilot{SUFFIX}" / "omicsgmf_protein_embedding.npy"))
    hv = pd.read_csv(BASE / "markers" / "hss_markers_table.csv")
    hzone = dict(zip(hv["AC"], hv["zone"]))
    midx = np.array([i for i, ac in enumerate(acs) if hzone.get(ac) in ("CV", "PV")])
    iscvm = np.array([hzone[acs[i]] == "CV" for i in midx])

    def proj(Msub):
        X2 = PCA(20, random_state=0).fit_transform(Msub) if Msub.shape[1] > 20 else Msub
        return umap.UMAP(n_neighbors=15, random_state=42).fit_transform(X2)

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))
    for ax, M, ttl in [(axes[0], E, "SABER"), (axes[1], omg, "omicsGMF"),
                       (axes[2], raw, "raw ESM-2")]:
        P = proj(M[midx])
        ax.scatter(P[~iscvm, 0], P[~iscvm, 1], s=14, c=PV, alpha=0.8,
                   edgecolor="none", label="PV-zone")
        ax.scatter(P[iscvm, 0], P[iscvm, 1], s=14, c=CV, alpha=0.8,
                   edgecolor="none", label="CV-zone")
        ax.set_title(ttl, fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=7, loc="lower left", markerscale=1.3,
                   handletextpad=0.2, frameon=False)
    _save(fig, "fig5d_liver_proteinemb_umap")


if __name__ == "__main__":
    panel_5a()
    panel_5b()
    panel_5c()
    panel_5d()
