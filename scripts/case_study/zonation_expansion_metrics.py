#!/usr/bin/env python
"""
Case-study zonation metrics for the EasySCP liver explicit-marker expansion holdout.

Aggregates across the multi-seed campaign (base seed 42 plus the per-seed
variants launched by run_case_study.py). Two methodological safeguards:

  (1) GATE-STRATIFIED. Zonation markers are sparse/bimodal, so we average each
      protein's abundance within each of the 8 FACS gates (A..H, CV->PV) and
      correlate the 8 gate-means with the gate-mean HSS axis.

  (2) DE-CONFOUNDED. A cell-mean baseline recovers "polarity" trivially from the
      bulk cell-level abundance gradient. We subtract the per-cell cell-mean
      baseline (from observed training proteins) from every prediction AND the
      truth, isolating the PROTEIN-SPECIFIC, sequence-driven signal. After this,
      cell-mean -> 0 -> chance, and SABER(ESM-2) vs no-PLM(random) is the clean
      sequence contrast.

Per method, per tier (mean over seeds): polarity_rate (fraction of markers whose
de-confounded gate-gradient corr has the zone-implied sign), mean_signed_orient,
mean_abs_grad_corr, mean_pp_pearson (standard expansion Pearson). Ceiling = truth.
"""
from __future__ import annotations
import argparse, glob, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from saber.paths import data_root, outputs_root  # noqa: E402

BASE = data_root() / "easyscp_liver"
PREP_H5AD = BASE / "prepared" / "EasySCP_liver.h5ad"
MARKERS = BASE / "markers"
RUNS_ROOT = outputs_root()
SUFFIX = ""  # run-tree suffix (set via --suffix); reads easyscp_expansion_*{SUFFIX}
TIERS = ["canonical", "hss_markers", "zonated"]
GATES = list("ABCDEFGH")
DEC_THR = 0.05


def _runs(tier, cond, axisholdout=False):
    """Return list of (seed, saber_npz, noplm_npz) across all seeds present.

    axisholdout=True reads the tier-UNION-axis runs (canonicalax/zonatedax),
    where the 215 HSS axis proteins were also removed from training; the hss
    tier already holds out all axis proteins, so it keeps its plain dirs.
    """
    dirt = f"{tier}ax" if (axisholdout and tier in ("canonical", "zonated")) else tier
    out = []
    for seed in (42, 123, 2024, 2025, 7, 0, 1, 2, 3, 4):
        suff = "" if seed == 42 else f"_seed{seed}"
        saber_p = sorted(glob.glob(str(RUNS_ROOT / f"easyscp_expansion_{dirt}_saber{suff}{SUFFIX}" / "*" / "artifacts" / "expansion.npz")))
        noplm_p = sorted(glob.glob(str(RUNS_ROOT / f"easyscp_expansion_{dirt}_noplm{suff}{SUFFIX}" / "*" / "artifacts" / "expansion.npz")))
        if saber_p and noplm_p:
            out.append((seed, saber_p[-1], noplm_p[-1]))
    return out


def safe_corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def grad_corr(M, j, gate, avail, axis_vals):
    gm = np.full(8, np.nan)
    for gi, g in enumerate(GATES):
        sel = (gate == g) & avail[:, j]
        if sel.sum():
            gm[gi] = np.nanmean(M[sel, j])
    m = np.isfinite(gm)
    return safe_corr(gm[m], axis_vals[m]) if m.sum() >= 4 else np.nan


def per_prot_pearson(pred, true, emask):
    rs = []
    for j in range(pred.shape[1]):
        mm = emask[:, j].astype(bool)
        if mm.sum() >= 3:
            r = safe_corr(pred[mm, j], true[mm, j])
            if np.isfinite(r): rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def metrics_for(pred, true, emask, gate, avail, axis_vals, zones):
    n = pred.shape[1]
    gc = np.array([grad_corr(pred, j, gate, avail, axis_vals) for j in range(n)])
    signed = np.full(n, np.nan)
    for j, z in enumerate(zones):
        if z in ("CV", "PV"):
            signed[j] = (1.0 if z == "CV" else -1.0) * gc[j]
    dec = signed[np.isfinite(signed) & (np.abs(gc) > DEC_THR)]
    return dict(
        polarity_rate=float(np.sum(dec > 0) / dec.size) if dec.size else np.nan,
        mean_signed_orient=float(np.nanmean(signed)) if np.isfinite(signed).any() else np.nan,
        mean_abs_grad_corr=float(np.nanmean(np.abs(gc))) if np.isfinite(gc).any() else np.nan,
        mean_pp_pearson=per_prot_pearson(pred, true, emask),
    )


def analyse_seed(seed, s_npz, n_npz, tier, gate, axis_vals, zone_map):
    se = np.load(s_npz); lab = np.load(s_npz.replace("expansion.npz", "labels.npz")); ne = np.load(n_npz)
    zones_all = [zone_map.get(str(ac)) for ac in lab["target_protein_ids"]]
    # tier-union-axis runs hold out extra axis proteins; report TIER members only
    # (zone present in the tier table), so numbers stay comparable across protocols.
    keep = np.array([z is not None for z in zones_all])
    zones = [z for z, k in zip(zones_all, keep) if k]
    sl = lambda A: A[:, keep] if A.ndim == 2 else A[keep]
    true = se["x_true_target"][:, keep]; avail = se["available_target_mask"][:, keep].astype(bool)
    emask = se["eval_target_mask"][:, keep]
    xtr = np.where(np.isfinite(se["x_true_train"]), se["x_true_train"], np.nan)
    cm = np.broadcast_to(np.nanmean(xtr, axis=1)[:, None], true.shape).astype(np.float32)
    preds = {"SABER": se["pred_target__saber"][:, keep], "no-PLM": ne["pred_target__saber"][:, keep],
             "PLM-ridge": se["pred_target__plm_ridge"][:, keep], "PLM-kNN": se["pred_target__plm_knn"][:, keep],
             "PLM-kNN(k10)": se["pred_target__plm_knn_k10"][:, keep], "cell-mean": cm}
    rows = []
    for variant, base in [("deconf", cm), ("raw", np.zeros_like(true))]:
        rows.append(dict(seed=seed, tier=tier, variant=variant, method="ceiling(true)",
                         **metrics_for(true - base, true, emask, gate, avail, axis_vals, zones)))
        for method, pred in preds.items():
            rows.append(dict(seed=seed, tier=tier, variant=variant, method=method,
                             **metrics_for(pred - base, true, emask, gate, avail, axis_vals, zones)))
    return rows


def main():
    global SUFFIX
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suffix", default=SUFFIX,
                    help="run-tree suffix (reads easyscp_expansion_*{suffix} dirs)")
    ap.add_argument("--out", default="",
                    help="output CSV (default: $SABER_OUTPUTS_ROOT/easyscp_expansion_logs/"
                         "zonation_expansion_metrics{suffix}{_ax}.csv)")
    ap.add_argument("--axisholdout", dest="axisholdout", action="store_true", default=True,
                    help="read tier-union-axis runs for canonical/zonated (hss unchanged) "
                         "[default: the Fig. 2 protocol]")
    ap.add_argument("--no-axisholdout", dest="axisholdout", action="store_false",
                    help="read the plain per-tier runs instead")
    args = ap.parse_args()
    SUFFIX = args.suffix
    if not args.out:
        ax = "_ax" if args.axisholdout else ""
        args.out = str(RUNS_ROOT / "easyscp_expansion_logs" / f"zonation_expansion_metrics{SUFFIX}{ax}.csv")
    a = ad.read_h5ad(PREP_H5AD)
    gate = np.asarray(a.obs["gate"], str); hss = np.asarray(a.obs["hss_score"], float)
    axis_vals = np.array([np.nanmean(hss[gate == g]) for g in GATES])
    print("gate mean HSS axis (A->H, CV->PV):", np.round(axis_vals, 3), "\n")
    rows = []
    for tier in TIERS:
        zone_map = dict(zip(pd.read_csv(MARKERS / f"{tier}_table.csv")["AC"],
                            pd.read_csv(MARKERS / f"{tier}_table.csv")["zone"]))
        runs = _runs(tier, "saber", axisholdout=args.axisholdout)
        print(f"[{tier}] seeds found: {[s for s, _, _ in runs]}")
        for seed, sp, np_ in runs:
            rows.extend(analyse_seed(seed, sp, np_, tier, gate, axis_vals, zone_map))
    df = pd.DataFrame(rows)
    agg = df.groupby(["tier", "variant", "method"]).agg(
        polarity_rate=("polarity_rate", "mean"), polarity_sd=("polarity_rate", "std"),
        signed=("mean_signed_orient", "mean"), signed_sd=("mean_signed_orient", "std"),
        absgrad=("mean_abs_grad_corr", "mean"), pp=("mean_pp_pearson", "mean"), pp_sd=("mean_pp_pearson", "std"),
        nseeds=("seed", "nunique")).reset_index()
    for tier in TIERS:
        for variant in ["deconf", "raw"]:
            sub = agg[(agg.tier == tier) & (agg.variant == variant)].set_index("method")
            if sub.empty: continue
            ceil = sub.loc["ceiling(true)"]
            print(f"=== tier={tier} [{variant}, nseed={int(ceil['nseeds'])}]  ceiling: "
                  f"polarity={ceil['polarity_rate']:.2f}, signed={ceil['signed']:+.2f}, pp={ceil['pp']:.2f} ===")
            print(sub.loc[["SABER", "no-PLM", "PLM-ridge", "PLM-kNN", "cell-mean"],
                        ["polarity_rate", "polarity_sd", "signed", "pp", "pp_sd"]].round(3).to_string())
            print()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(args.out).replace(".csv", "_long.csv"), index=False)
    agg.to_csv(args.out, index=False)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
