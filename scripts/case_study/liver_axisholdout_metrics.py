#!/usr/bin/env python
"""Liver real-imputation metrics on the axis-holdout runs.

The 215 HSS-defining proteins were removed from every method's training matrix.
Axis = gate-mean HSS (dataset authors' GSVA score; observed values = ground
truth). De-confounding follows the liver convention (each matrix's own per-cell
mean — internally scale-consistent per method, unlike the Brunner bug).

Reported per tier:
  in-matrix subset  (tier ∩ kept proteins) — all methods comparable, main bars
  zero-shot subset  (tier ∩ axis proteins) — SABER-only side note
Writes $SABER_OUTPUTS_ROOT/easyscp_expansion_logs/realimpute_metrics_axisholdout.csv.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from saber.paths import data_root, outputs_root  # noqa: E402

GATES = list("ABCDEFGH")
TIERS = ["canonical", "hss_markers", "zonated"]


def sc_corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def ggrad(M, j, gate, avail, axis):
    gm = np.array([np.nanmean(M[(gate == g) & avail[:, j], j]) if ((gate == g) & avail[:, j]).sum() else np.nan
                   for g in GATES])
    m = np.isfinite(gm)
    return sc_corr(gm[m], axis[m]) if m.sum() >= 4 else np.nan


def _ggrad_1d(v, gate, m, axis):
    gm = np.array([np.nanmean(v[(gate == g) & m]) if ((gate == g) & m).sum() else np.nan
                   for g in GATES])
    mfin = np.isfinite(gm)
    return sc_corr(gm[mfin], axis[mfin]) if mfin.sum() >= 4 else np.nan


def metrics_one(M, gate, axis, idx, zones, miss_mask=None):
    """miss_mask: (cells, proteins) True where the entry was ORIGINALLY missing.
    When given, also computes the gradient restricted to filled entries only
    (the fill's own zonation signal; cell-mean is constant there by construction)."""
    avail = np.isfinite(M)
    gc = np.array([ggrad(M, j, gate, avail, axis) for j in idx])
    signed = np.array([(1.0 if z == "CV" else -1.0) * g for z, g in zip(zones, gc)])
    dec = signed[np.isfinite(signed) & (np.abs(gc) > 0.05)]
    out = [float(np.sum(dec > 0) / dec.size) if dec.size else np.nan,
           float(np.nanmean(signed)) if np.isfinite(signed).any() else np.nan,
           float(np.nanmean(np.abs(gc))) if np.isfinite(gc).any() else np.nan]
    if miss_mask is not None:
        gcm = np.array([_ggrad_1d(M[:, j], gate, miss_mask[:, k], axis) for k, j in enumerate(idx)])
        sm = np.array([(1.0 if z == "CV" else -1.0) * g for z, g in zip(zones, gcm)])
        out += [float(np.nanmean(sm)) if np.isfinite(sm).any() else np.nan,
                float(np.nanmean(np.abs(gcm))) if np.isfinite(gcm).any() else np.nan]
    return tuple(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--liver-dir", default="",
                    help="easyscp_liver data dir (default: $SABER_DATA_ROOT/easyscp_liver)")
    ap.add_argument("--axisholdout-dir", default="",
                    help="axis-holdout run dir (default: $SABER_OUTPUTS_ROOT/easyscp_realimpute_axisholdout)")
    ap.add_argument("--out-dir", default="",
                    help="metrics output dir (default: $SABER_OUTPUTS_ROOT/easyscp_expansion_logs)")
    args = ap.parse_args()
    BASE = Path(args.liver_dir) if args.liver_dir else data_root() / "easyscp_liver"
    OUT = Path(args.axisholdout_dir) if args.axisholdout_dir else outputs_root() / "easyscp_realimpute_axisholdout"
    LOGS = Path(args.out_dir) if args.out_dir else outputs_root() / "easyscp_expansion_logs"
    LOGS.mkdir(parents=True, exist_ok=True)

    a = ad.read_h5ad(BASE / "prepared" / "EasySCP_liver.h5ad")
    gate = np.asarray(a.obs["gate"], str)
    hss = np.asarray(a.obs["hss_score"], float)
    axis = np.array([np.nanmean(hss[gate == g]) for g in GATES])

    Z = dict(np.load(OUT / "matrices.npz"))
    acs_keep = [str(x) for x in Z["acs_keep"]]
    acs_axis = [str(x) for x in Z["acs_axis"]]

    # ---- metric (B), same convention as the Brunner fix: one common z space.
    # Every raw matrix is z-standardised per protein with the observed stats of
    # the reduced matrix; the per-cell mean of the OBSERVED z matrix is then
    # subtracted from every method, so the de-confounded residual is the same
    # functional for all methods (SABER / zero-shot matrices are already z).
    obs_raw = np.asarray(Z["observed_reduced"], float)
    sc = np.nanmean(np.where(np.isfinite(obs_raw), obs_raw, np.nan), axis=0)
    ss = np.nanstd(np.where(np.isfinite(obs_raw), obs_raw, np.nan), axis=0)
    ss = np.where(np.isfinite(ss) & (ss > 1e-8), ss, 1.0)
    zcellmean = np.nanmean(np.where(np.isfinite(obs_raw), (obs_raw - sc) / ss, np.nan), axis=1)

    def deconf_z(M, is_z=False):
        Mz = np.asarray(M, float) if is_z else (np.asarray(M, float) - sc) / ss
        return Mz - zcellmean[:, None]
    ac2zone_all = {}
    for tier in TIERS:
        df = pd.read_csv(BASE / "markers" / f"{tier}_table.csv")
        ac2zone_all[tier] = dict(zip(df["AC"].dropna().astype(str), df["zone"]))
    keep_pos = {ac: i for i, ac in enumerate(acs_keep)}
    axis_pos = {ac: i for i, ac in enumerate(acs_axis)}

    mats = {"observed-only": [deconf_z(Z["observed_reduced"])],
            "cell-mean": [deconf_z(Z["cellmean"])],
            "KNN": [deconf_z(Z["knn"])],
            "SABER": [deconf_z(m, is_z=True) for m in Z["saber_seeds"]]}
    for m, disp in (("omicsgmf", "omicsGMF"), ("pimms_dae", "PIMMS")):
        if f"{m}_seed42" in Z:
            mats[disp] = [deconf_z(Z[f"{m}_seed{s}"]) for s in (42, 123, 7) if f"{m}_seed{s}" in Z]

    rows = []
    miss_mask = ~np.isfinite(obs_raw)          # originally-missing entries
    for tier in TIERS:
        zone = ac2zone_all[tier]
        idx_in = [keep_pos[ac] for ac in acs_keep if zone.get(ac) in ("CV", "PV")]
        zon_in = [zone[ac] for ac in acs_keep if zone.get(ac) in ("CV", "PV")]
        idx_zs = [axis_pos[ac] for ac in acs_axis if zone.get(ac) in ("CV", "PV")]
        zon_zs = [zone[ac] for ac in acs_axis if zone.get(ac) in ("CV", "PV")]
        print(f"\n===== tier={tier}: in-matrix {len(idx_in)}, zero-shot(axis) {len(idx_zs)} =====")
        for label, Ms in mats.items():
            res = np.array([metrics_one(M, gate, axis, idx_in, zon_in,
                                        miss_mask=miss_mask[:, idx_in]) for M in Ms])
            pol, sg, gr, sgm, grm = (res[:, k] for k in range(res.shape[1]))
            f = (lambda x: f"{np.nanmean(x):.2f}±{np.nanstd(x):.2f}") if len(Ms) > 1 else (lambda x: f"{np.nanmean(x):.2f}")
            print(f"  {label:>14} polarity={f(pol)} signed={f(sg)} |grad|={f(gr)} | fill-only signed={f(sgm)} |g|={f(grm)}")
            for r_ in res:
                rows.append(dict(tier=tier, subset="in_matrix", method=label, nseeds=len(Ms),
                                 n_prot=len(idx_in), polarity=r_[0], signed=r_[1], absgrad=r_[2],
                                 fill_signed=r_[3], fill_absgrad=r_[4]))
        # SABER zero-shot on tier∩axis (side note; baselines have no columns for these)
        res = np.array([metrics_one(deconf_z(M, is_z=True), gate, axis, idx_zs, zon_zs)
                        for M in Z["zeroshot_axis_seeds"]])
        print(f"  {'SABER-zeroshot':>14} (axis∩tier n={len(idx_zs)}) |grad|={np.nanmean(res[:,2]):.2f}±{np.nanstd(res[:,2]):.2f}")
        for r_ in res:
            rows.append(dict(tier=tier, subset="zeroshot_axis", method="SABER-zeroshot", nseeds=len(res),
                             n_prot=len(idx_zs), polarity=r_[0], signed=r_[1], absgrad=r_[2]))

    pd.DataFrame(rows).to_csv(LOGS / "realimpute_metrics_axisholdout.csv", index=False)
    print(f"\nsaved -> {LOGS/'realimpute_metrics_axisholdout.csv'}")


if __name__ == "__main__":
    main()
