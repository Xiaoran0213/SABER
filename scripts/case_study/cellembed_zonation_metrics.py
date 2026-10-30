#!/usr/bin/env python
"""
Case-study cell-embedding metrics on the EasySCP liver CONTINUOUS zonation axis.

SABER is trained for reconstruction with no cell-embedding objective, so the cell
latent Z is a byproduct. The main paper's cell-emb claim is "competitive, not
superior" on CATEGORICAL cell types. Here we test the harder, continuous case:
does SABER's 16-dim cell latent order hepatocytes along the portal-central axis
(HSS), comparably to dedicated cell-embedding methods and to a trivial expression
baseline?

Metrics (5-fold CV, linear probe embedding -> HSS):
  * HSS-probe Pearson  : Pearson(predicted HSS, true HSS) over held-out folds.
  * gate ARI (k=8)     : KMeans(8) on the embedding vs the 8 FACS gate labels.
Embeddings: SABER z (ESM-2), no-PLM z (random features), raw-expression PCA(16),
cell-mean (1-d). NOTE HSS is derived from expression, so raw-PCA is a strong /
honest ceiling -- the claim is "SABER latent recovers the continuous axis
unsupervised, on par with what raw expression carries", not "beats expression".
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from saber.paths import data_root, outputs_root  # noqa: E402

BASE = data_root() / "easyscp_liver"
PREP_H5AD = BASE / "prepared" / "EasySCP_liver.h5ad"
RUNS = outputs_root()
SUFFIX = ""  # run-tree suffix (set via --suffix)


def load_z(cond, suffix=None):
    suffix = SUFFIX if suffix is None else suffix
    f = sorted(glob.glob(str(RUNS / f"easyscp_cellembed_{cond}{suffix}" / "*" / "artifacts" / "embeddings.npz")))[-1]
    return np.load(f)["z__saber"]


def load_z_seeds(cond, suffix=None):
    """Cell embeddings across the ten seed runs (base run = seed 42, then _seed{s}
    variants); missing runs are skipped. no-PLM now has the same ten-run coverage
    as SABER, so the cell-level comparison reports seed error bars for both."""
    suffix = SUFFIX if suffix is None else suffix
    out = []
    for s in [42, 123, 7, 2024, 2025, 0, 1, 2, 3, 4]:
        suff = "" if s == 42 else f"_seed{s}"
        fs = sorted(glob.glob(str(RUNS / f"easyscp_cellembed_{cond}{suff}{suffix}" / "*" / "artifacts" / "embeddings.npz")))
        if fs:
            out.append(np.load(fs[-1])["z__saber"])
    return out


def cv_hss_pearson(emb, hss, seed=0, n_folds=5, alpha=1.0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(hss))
    folds = np.array_split(idx, n_folds)
    pred = np.zeros_like(hss)
    for k in range(n_folds):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        m = Ridge(alpha=alpha).fit(emb[tr], hss[tr])
        pred[te] = m.predict(emb[te])
    if np.std(pred) < 1e-9:
        return np.nan
    return float(np.corrcoef(pred, hss)[0, 1])


def gate_ari(emb, gate, seed=0, k=8):
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(emb)
    return float(adjusted_rand_score(gate, km.labels_))


def main():
    global SUFFIX
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suffix", default="", help="run-tree suffix, appended to run names")
    ap.add_argument("--out", default="", help="output CSV path")
    args = ap.parse_args()
    SUFFIX = args.suffix

    a = ad.read_h5ad(PREP_H5AD)
    hss = np.asarray(a.obs["hss_score"], float)
    gate = np.asarray(a.obs["gate"], str)
    # one cell carries a missing HSS score; drop it everywhere so every embedding
    # and the expression baselines are evaluated on the identical cell set
    keep = np.isfinite(hss)
    if not keep.all():
        print(f"dropping {int((~keep).sum())} cell(s) with missing HSS")
    hss = hss[keep]
    gate = gate[keep]
    # map gate labels A-H to integers for ARI
    gl = sorted(set(gate))
    gmap = {g: i for i, g in enumerate(gl)}
    gate_i = np.array([gmap[g] for g in gate])

    # raw expression matrix (observed, z-scored) for PCA + cell-mean baselines
    lab = np.load(sorted(glob.glob(str(RUNS / f"easyscp_cellembed_saber{SUFFIX}" / "*" / "artifacts" / "labels.npz")))[-1])
    X = np.asarray(lab["x_raw"], float)[keep]   # cells x proteins, may have NaN
    X = np.where(np.isfinite(X), X, np.nan)
    cmean = np.nanmean(X, axis=1)
    Xz = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
    Xz = np.where(np.isfinite(Xz), Xz, 0.0)
    pca16 = PCA(n_components=16, random_state=0).fit_transform(Xz)
    pca2 = PCA(n_components=2, random_state=0).fit_transform(Xz)

    embs = {
        "SABER z (ESM-2)": load_z("saber")[keep],
        "no-PLM z (random)": load_z("noplm")[keep],
        "raw-expression PCA(16)": pca16,
        "raw-expression PCA(2)": pca2,
        "cell-mean (1-d)": cmean[:, None],
    }
    print(f"cells={len(hss)} | gate levels={gl}\n")
    print(f"{'embedding':>24} | {'HSS-probe r':>12} {'gate ARI(k=8)':>14}")
    print("-" * 56)
    rows = []
    for name, e in embs.items():
        r = cv_hss_pearson(e, hss)
        ari = gate_ari(e, gate_i) if e.shape[1] >= 2 else np.nan
        print(f"{name:>24} | {r:>12.3f} {ari:>14.3f}")
        rows.append({"embedding": name, "hss_probe_r": r, "gate_ari": ari})
    # Seed-level error bars for the trained embeddings (SABER and no-PLM now both
    # have ten runs); mean ± SD written alongside the single-run table.
    for cond, label in [("saber", "SABER"), ("noplm", "no-PLM")]:
        zs = [z[keep] for z in load_z_seeds(cond)]
        if len(zs) < 2:
            continue
        rs = [cv_hss_pearson(z, hss) for z in zs]
        aris = [gate_ari(z, gate_i) for z in zs]
        print(f"{label + ' (10-seed)':>24} | {np.mean(rs):>7.3f}±{np.std(rs):.3f} {np.mean(aris):>7.3f}±{np.std(aris):.3f}  (n={len(zs)})")
        rows.append({"embedding": f"{label} ({len(zs)}-seed mean)", "hss_probe_r": float(np.mean(rs)),
                     "hss_probe_sd": float(np.std(rs)), "gate_ari": float(np.mean(aris)),
                     "gate_ari_sd": float(np.std(aris)), "n_seeds": len(zs)})
    out_csv = Path(args.out) if args.out else RUNS / "easyscp_expansion_logs" / f"cellembed_zonation_metrics{SUFFIX}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nsaved -> {out_csv}")
    # gate-mean HSS ordering check (sanity)
    print("\ngate mean HSS:", {g: round(float(np.mean(hss[gate == g])), 3) for g in gl})


if __name__ == "__main__":
    main()
