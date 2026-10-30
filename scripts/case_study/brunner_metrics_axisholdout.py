#!/usr/bin/env python
"""Brunner cell-cycle metrics on the axis-holdout runs.

Protocol: the 22 quantified cycle-score markers were removed from the training
matrix of every method (SABER / KNN / omicsGMF / PIMMS-DAE / cell-mean). The
cell-cycle score itself stays the observed 22-marker axis (external ground
truth). Metric = consistent z space (scale-bug fix "metric B"): every raw
matrix is z-standardised per protein with the observed stats of the reduced
matrix, the per-cell mean of the observed z matrix is subtracted, and the
gradient g_j = corr_i(x_ij - m_i, cc_i).

Class definitions (cleaned): axis members are absent from the matrix by
construction, so classes are automatically marker-free:
  gold-APC n=8 in-matrix + 7 axis members reported separately (zero-shot)
  degron   = D-box|KEN|gold among kept proteins
  ribo     = RPL/RPS among kept proteins (no axis overlap)

Writes $SABER_OUTPUTS_ROOT/easyscp_expansion_logs/brunner_metrics_axisholdout_{methods,
perseed,perprotein,zeroshot}.csv and prints the summary.
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from saber.paths import data_root, outputs_root  # noqa: E402
GOLD_APC = set(("CCNA2 CCNB1 CCNB2 CCND1 CCNE1 CDK1 CDC20 PTTG1 GMNN BUB1 BUB1B PLK1 AURKA AURKB "
                "KIF11 KIF4A KIF20A KIF23 KIF18A TPX2 MKI67 TOP2A SGO1 CDCA5 NUSAP1 BIRC5 UBE2C UBE2S "
                "CDCA8 PRC1 ANLN RACGAP1 NCAPG NCAPD2 NCAPH FOXM1").split())


def corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20 or np.std(x[m]) < 1e-9 or np.std(y[m]) < 1e-9:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brunner-dir", default="",
                    help="brunner2022 data dir (default: $SABER_DATA_ROOT/brunner2022)")
    ap.add_argument("--axisholdout-dir", default="",
                    help="axis-holdout run dir (default: $SABER_OUTPUTS_ROOT/brunner_realimpute_axisholdout)")
    ap.add_argument("--out-dir", default="",
                    help="metrics output dir (default: $SABER_OUTPUTS_ROOT/easyscp_expansion_logs)")
    args = ap.parse_args()
    B = Path(args.brunner_dir) if args.brunner_dir else data_root() / "brunner2022"
    OUT = Path(args.axisholdout_dir) if args.axisholdout_dir else outputs_root() / "brunner_realimpute_axisholdout"
    LOGS = Path(args.out_dir) if args.out_dir else outputs_root() / "easyscp_expansion_logs"
    LOGS.mkdir(parents=True, exist_ok=True)

    a = ad.read_h5ad(B / "prepared" / "Brunner2022.h5ad")
    cc = np.asarray(a.obs["cell_cycle_score"], float)
    Z = dict(np.load(OUT / "matrices.npz"))
    acs_keep = [str(x) for x in Z["acs_keep"]]
    acs_axis = [str(x) for x in Z["acs_axis"]]
    obs_raw = np.asarray(Z["observed_reduced"], float)      # raw, NaN = missing

    d = pd.read_csv(B / "brunner_derisk2_results.csv")
    ac2g = {ac: (str(g) if pd.notna(g) else "") for ac, g in zip(d.AC_base, d.gene)}
    ac2deg = dict(zip(d.AC_base, d.dbox | d.ken))
    genes = np.array([ac2g.get(ac, "") for ac in acs_keep])
    gold = np.array([g in GOLD_APC for g in genes])
    ribo = np.array([bool(re.match(r"^RP[SL]\d", g)) for g in genes])
    degm = np.array([bool(ac2deg.get(ac, False)) or g in GOLD_APC for ac, g in zip(acs_keep, genes)])
    print(f"kept {len(acs_keep)} | gold {gold.sum()} | degron {degm.sum()} | ribo {ribo.sum()}")

    # ---- metric (B): common z space from observed stats of the reduced matrix ----
    sc = np.nanmean(np.where(np.isfinite(obs_raw), obs_raw, np.nan), axis=0)
    ss = np.nanstd(np.where(np.isfinite(obs_raw), obs_raw, np.nan), axis=0)
    ss = np.where(np.isfinite(ss) & (ss > 1e-8), ss, 1.0)
    zobs = (obs_raw - sc) / ss
    zcellmean = np.nanmean(np.where(np.isfinite(zobs), zobs, np.nan), axis=1)

    def grads(Mz):
        return np.array([corr(Mz[:, j] - zcellmean, cc) for j in range(Mz.shape[1])])

    def z_of(Mraw):
        return (np.asarray(Mraw, float) - sc) / ss

    mats = {"observed": zobs,
            "SABER": np.asarray(Z["saber_seeds"], float)[0],          # representative seed 42
            "KNN": z_of(Z["knn"]),
            "cell-mean": z_of(Z["cellmean"])}
    ext = {}
    for k in Z:
        if k.startswith(("omicsgmf_seed", "pimms_dae_seed")):
            ext.setdefault(k.rsplit("_seed", 1)[0], []).append(np.asarray(Z[k], float))
    for nm, Ms in ext.items():                                        # mean over externals' seeds
        mats[nm] = z_of(np.mean(Ms, axis=0))

    rows = []
    for nm, M in mats.items():
        g = grads(M)
        rows.append(dict(method=nm, gold=np.abs(g[gold]).mean(), degron=np.abs(g[degm]).mean(),
                         ribo=np.abs(g[ribo]).mean(),
                         p_gold_vs_ribo=mannwhitneyu(np.abs(g[gold]), np.abs(g[ribo]), alternative="greater")[1],
                         p_deg_vs_ribo=mannwhitneyu(np.abs(g[degm]), np.abs(g[ribo]), alternative="greater")[1]))
    t = pd.DataFrame(rows)
    t.to_csv(f"{LOGS}/brunner_metrics_axisholdout_methods.csv", index=False)
    print("=== methods (z-consistent, axis-heldout) ===")
    print(t.round(3).to_string(index=False))

    # ---- per-seed SABER ----
    per = []
    for k, s in enumerate(Z["seeds"]):
        g = grads(np.asarray(Z["saber_seeds"], float)[k])
        per.append(dict(seed=int(s), gold=np.abs(g[gold]).mean(), degron=np.abs(g[degm]).mean(),
                        ribo=np.abs(g[ribo]).mean(),
                        p_deg_vs_ribo=mannwhitneyu(np.abs(g[degm]), np.abs(g[ribo]), alternative="greater")[1]))
    per = pd.DataFrame(per); per.to_csv(f"{LOGS}/brunner_metrics_axisholdout_perseed.csv", index=False)
    print(f"\nSABER per-seed gold: {per.gold.mean():.3f} ± {per.gold.std():.3f} (n seeds {len(per)})")

    # ---- per-protein + showcase candidates ----
    g_obs = grads(zobs); g_sab = grads(np.asarray(Z["saber_seeds"], float)[0])
    pp = pd.DataFrame({"AC": acs_keep, "gene": genes, "g_obs": g_obs, "g_imp": g_sab,
                       "gold": gold, "degron": degm, "ribo": ribo,
                       "obsfrac": np.isfinite(obs_raw).mean(0)})
    pp.to_csv(f"{LOGS}/brunner_metrics_axisholdout_perprotein.csv", index=False)
    show = pp[pp.gold].sort_values("g_imp", key=lambda s: s.abs(), ascending=False)
    print("\ngold-APC in-matrix (showcase candidates, by |g_imp|):")
    print(show[["gene", "obsfrac", "g_obs", "g_imp"]].round(3).to_string(index=False))

    # ---- zero-shot side note: gradients of the 22 axis proteins themselves ----
    zs_rows = []
    genes_ax = [ac2g.get(ac, "") for ac in acs_axis]
    for k, s in enumerate(Z["seeds"]):
        gzs = grads(np.asarray(Z["zeroshot_axis_seeds"], float)[k])
        zs_rows.append(np.abs(gzs))
    gzs_mean = np.mean(zs_rows, axis=0)
    ax_gold = np.array([g in GOLD_APC for g in genes_ax])
    zs = pd.DataFrame({"AC": acs_axis, "gene": genes_ax, "g_zeroshot_abs": gzs_mean, "gold": ax_gold})
    zs.to_csv(f"{LOGS}/brunner_metrics_axisholdout_zeroshot.csv", index=False)
    print(f"\nzero-shot axis proteins: gold-members |g| = {np.nanmean(gzs_mean[ax_gold]):.3f} (n={ax_gold.sum()}), "
          f"other axis |g| = {np.nanmean(gzs_mean[~ax_gold]):.3f}")


if __name__ == "__main__":
    main()
