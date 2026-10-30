"""Fig. 2 (liver proteome expansion) — panels a-c + a fig5e companion panel.

  2a  liver expansion: per-protein Pearson by marker tier x method (de-conf, 10 seeds)
  2b  liver expansion: zonation polarity rate by tier x method (de-conf, 10 seeds)
  2c  liver expansion: per-protein Pearson distribution on the zonated tier (violin)
  2d  GO enrichment heatmap (reused from the expansion figure group, not regenerated)
Also emits an external GO:0000278 validation panel for the cell-cycle data.

Writes to $SABER_FIGURES_ROOT/fig2/ (png+pdf), independent of the campaign
paper_figures tree, plus source CSVs alongside.

Sources:
  liver  $SABER_OUTPUTS_ROOT/easyscp_expansion_logs/zonation_expansion_metrics{suffix}_ax.csv
         $SABER_OUTPUTS_ROOT/easyscp_expansion_{canonical,zonated}ax_{saber,noplm}{suffix}/
Run:  python scripts/figures/fig2_v2_expansion.py [--suffix <s>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))  # resolve `saber` from this checkout, not the env
sys.path.insert(0, str(REPO / "scripts" / "figures"))

from saber.paths import data_root, figures_root, outputs_root  # noqa: E402

import style  # noqa: E402
from style import get_plt, color_for  # noqa: E402

plt = get_plt()

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--metrics", default="",
                 help="zonation metrics CSV (default: $SABER_OUTPUTS_ROOT/easyscp_expansion_logs/"
                      "zonation_expansion_metrics{suffix}_ax.csv)")
_ap.add_argument("--suffix", default="", help="run-tree suffix, appended to run names")
_ap.add_argument("--out", default="", help="figure output dir (default: $SABER_FIGURES_ROOT/fig2)")
_args = _ap.parse_args()
SUFFIX = _args.suffix
LIVER = (Path(_args.metrics) if _args.metrics
         else outputs_root() / "easyscp_expansion_logs" / f"zonation_expansion_metrics{SUFFIX}_ax.csv")


def _tier_dir(tier: str) -> str:
    """Axis-holdout protocol: canonical/zonated read the tier-union-axis reruns;
    hss_markers already held out all 215 axis proteins and is unchanged."""
    return f"{tier}ax" if tier in ("canonical", "zonated") else tier
EXTGO_JSON = outputs_root() / "easyscp_expansion_logs" / "m3_external_validation.json"
OUT = Path(_args.out) if _args.out else figures_root() / "fig2"
OUT.mkdir(parents=True, exist_ok=True)

# liver visualization panels: per-protein views of the expansion prediction
# along the portal-central axis. Same seed-42 runs as Fig 2a-c.
LIV_BASE = data_root() / "easyscp_liver"
LIV_MARKERS = LIV_BASE / "markers"
LIV_H5AD = LIV_BASE / "prepared" / "EasySCP_liver.h5ad"
_run_can_name = f"easyscp_expansion_canonicalax_saber{SUFFIX}"
_run_zon_name = f"easyscp_expansion_zonatedax_saber{SUFFIX}"
RUN_CAN = outputs_root() / _run_can_name / _run_can_name / "artifacts"
RUN_ZON = outputs_root() / _run_zon_name / _run_zon_name / "artifacts"
ZONE_C = {"CV": "#0072B2", "PV": "#D95F02"}


def _liver_run(art_dir):
    """Seed-42 run artifacts + shared cell axis, de-confounded.

    Returns dict with observed (NaN-masked), SABER prediction, target protein
    table (gene/AC/zone), and cell ordering by HSS. De-confounding subtracts
    each cell's observed mean over ALL run proteins (same convention as the
    tier metrics), so a constant imputation carries no protein-specific signal.
    """
    import anndata as ad
    d = np.load(art_dir / "expansion.npz", allow_pickle=True)
    lab = np.load(art_dir / "labels.npz", allow_pickle=True)
    a = ad.read_h5ad(LIV_H5AD)
    pid = [str(p) for p in lab["protein_ids"]]
    col = {p: i for i, p in enumerate(a.var_names)}
    assert all(p in col for p in pid)
    Xfull = np.asarray(a.X[:, [col[p] for p in pid]], dtype=float)
    m_i = np.nanmean(Xfull, axis=1)
    hss = np.asarray(a.obs["hss_score"], dtype=float)
    gate = np.asarray(a.obs["gate"], dtype=str)
    obs = np.where(d["eval_target_mask"] == 1, d["x_true_target"], np.nan) - m_i[:, None]
    pred = d["pred_target__saber"] - m_i[:, None]
    tgt = [str(t) for t in lab["target_protein_ids"]]
    return {"obs": obs, "pred": pred, "tgt": tgt, "hss": hss, "gate": gate}


def _gate_gradient(vals, gate, hss, min_gates=6):
    """Gate-stratified gradient of a protein: corr of gate means vs gate-mean HSS."""
    gs, gh = [], []
    for g in sorted(set(gate)):
        m = gate == g
        if not np.any(np.isfinite(vals[m])):
            continue
        gs.append(np.nanmean(vals[m]))
        gh.append(np.nanmean(hss[m]))
    if len(gs) < min_gates or np.std(gs) < 1e-9 or np.std(gh) < 1e-9:
        return np.nan
    return float(np.corrcoef(gs, gh)[0, 1])

METHODS = ["SABER", "SABER no PLM", "PLM-kNN", "PLM-ridge"]
# display label -> row label in the liver CSV (only the default k=5 reference
# is plotted; k=10 is a side variant kept for the source table)
LIVER_LABEL = {"SABER": "SABER", "SABER no PLM": "no-PLM", "PLM-kNN": "PLM-kNN",
               "PLM-kNN (k=10)": "PLM-kNN(k10)", "PLM-ridge": "PLM-ridge"}
TIERS = ["canonical", "hss_markers", "zonated"]
# two-line tier labels with explicit n: a bare "(18)" read as unexplained
TIER_LABEL = {"canonical": "Canonical\n(n=18)", "hss_markers": "HSS-defined\n(n=215)",
              "zonated": "Zonated\n(n=1,123)"}


def _save(fig, name, src_df=None):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    if src_df is not None:
        src_df.to_csv(OUT / f"{name}_source.csv", index=False)
    plt.close(fig)
    print("panel ->", OUT / name)


def _ref_label(m, c):
    if m == "saber":
        return "SABER" if c == "saber" else "SABER no PLM"
    return {"plm_ridge": "PLM-ridge", "plm_knn": "PLM-kNN", "plm_knn_k10": "PLM-kNN (k=10)",
            "cell_mean": "Cell mean"}.get(m, m)


# --------------------------------------------------------------------------- #
def panel_2a():
    df = pd.read_csv(LIVER)
    df = df[df.variant == "deconf"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    n = len(METHODS); w = 0.8 / n
    for ti, tier in enumerate(TIERS):
        sub = df[df.tier == tier].set_index("method")
        for mi, m in enumerate(METHODS):
            lm = LIVER_LABEL[m]
            if lm not in sub.index:
                continue
            v = sub.loc[lm, "pp"]; sd = sub.loc[lm, "pp_sd"]
            x = ti + (mi - (n - 1) / 2) * w
            ax.bar(x, v, width=w * 0.92, color=color_for(m),
                   yerr=sd if np.isfinite(sd) else None,
                   error_kw=dict(lw=0.8, capsize=1.5))
    ax.set_xticks(range(len(TIERS)))
    ax.set_xticklabels([TIER_LABEL[t] for t in TIERS], fontsize=8)
    ax.set_ylabel("per-protein Pearson r")
    ax.axhline(0, color="0.2", lw=0.6, ls="--")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_for(m)) for m in METHODS]
    ax.legend(handles, METHODS, ncol=5, fontsize=6.5, frameon=False,
              loc="lower left", bbox_to_anchor=(0, 1.01), columnspacing=1.0,
              handlelength=1.2, handletextpad=0.4)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
    _save(fig, "fig2a_liver_expansion_pearson", df)


def panel_2b():
    df = pd.read_csv(LIVER)
    df = df[df.variant == "deconf"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    n = len(METHODS); w = 0.8 / n
    for ti, tier in enumerate(TIERS):
        sub = df[df.tier == tier].set_index("method")
        for mi, m in enumerate(METHODS):
            lm = LIVER_LABEL[m]
            if lm not in sub.index or pd.isna(sub.loc[lm, "polarity_rate"]):
                continue
            v = sub.loc[lm, "polarity_rate"]; sd = sub.loc[lm, "polarity_sd"]
            x = ti + (mi - (n - 1) / 2) * w
            ax.bar(x, v, width=w * 0.92, color=color_for(m),
                   yerr=sd if np.isfinite(sd) else None,
                   error_kw=dict(lw=0.8, capsize=1.5))
    ax.set_xticks(range(len(TIERS)))
    ax.set_xticklabels([TIER_LABEL[t] for t in TIERS], fontsize=8)
    ax.set_ylabel("zonation polarity rate")
    ax.set_ylim(0.3, 1.02)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_for(m)) for m in METHODS]
    ax.legend(handles, METHODS, ncol=4, fontsize=6.5, frameon=False,
              loc="lower left", bbox_to_anchor=(0, 1.01), columnspacing=1.0,
              handlelength=1.2, handletextpad=0.4)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.02)
    _save(fig, "fig2c_liver_polarity", df)


# --------------------------------------------------------------------------- #
def _liver_perprotein(tier="zonated"):
    """Per-protein Pearson r on a liver tier, all methods, seed 42 (canonical run).

    The no-PLM run's npz also carries the plm_* reference predictions (recomputed
    on its masks); only pred_target__saber from that run is the no-PLM model.
    The plm_* references are taken from the saber run only.
    """
    recs = []
    td = _tier_dir(tier)
    tier_acs = set(pd.read_csv(LIV_MARKERS / f"{tier}_table.csv")["AC"].dropna().astype(str))
    for cond in ("saber", "noplm"):
        _name = f"easyscp_expansion_{td}_{cond}{SUFFIX}"
        f = outputs_root() / _name / _name / "artifacts" / "expansion.npz"
        if not f.exists():
            print(f"  [warn] missing {f}")
            continue
        d = np.load(f, allow_pickle=True)
        lab = np.load(str(f).replace("expansion.npz", "labels.npz"), allow_pickle=True)
        tgt = [str(t) for t in lab["target_protein_ids"]]
        keep = np.array([t in tier_acs for t in tgt])  # tier members only (axis extras excluded)
        X, M = d["x_true_target"][:, keep], d["eval_target_mask"][:, keep]
        entries = ([("SABER", "pred_target__saber")] if cond == "saber"
                   else [("SABER no PLM", "pred_target__saber")])
        if cond == "saber":
            entries += [("PLM-kNN", "pred_target__plm_knn"),
                        ("PLM-kNN (k=10)", "pred_target__plm_knn_k10"),
                        ("PLM-ridge", "pred_target__plm_ridge")]
        for meth, key in entries:
            P = d[key][:, keep]
            for j in range(X.shape[1]):
                idx = M[:, j] == 1
                if idx.sum() < 10:
                    continue
                a, b = X[idx, j], P[idx, j]
                if a.std() < 1e-9 or b.std() < 1e-9:
                    continue
                recs.append({"method": meth, "pearson_r": float(np.corrcoef(a, b)[0, 1])})
    return pd.DataFrame(recs)


def panel_2c():
    """Per-protein distribution on the zonated tier (violin), SABER vs references.

    v2.1 2026-08-18: replaces the benchmark bar + oracle-tick panel. The oracle
    analysis was withdrawn after review — single-best-neighbour is not an upper
    bound for a multi-protein decoder, and max-|r| over ~10^3 training proteins
    has an uncontrolled noise floor at small n_eval.
    """
    df = _liver_perprotein("zonated")
    order = ["SABER", "SABER no PLM", "PLM-kNN", "PLM-ridge"]
    stats = (df.groupby("method")["pearson_r"]
             .agg(mean="mean", median="median",
                  frac_gt03=lambda s: (s > 0.3).mean(),
                  frac_gt05=lambda s: (s > 0.5).mean(), n="count")
             .reindex(order))
    # per-protein paired Wilcoxon SABER vs references (main-text §expansion stats;
    # n = proteins on the zonated tier, seed 42 canonical run)
    saber = df[df.method == "SABER"]["pearson_r"].values
    for ref in ("SABER no PLM", "PLM-kNN", "PLM-ridge"):
        other = df[df.method == ref]["pearson_r"].values
        n = min(len(saber), len(other))
        _, p = wilcoxon(saber[:n], other[:n])
        stats.loc[ref, "wilcoxon_p_vs_saber"] = p
    print(stats.round(3).to_string())

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    data = [df[df.method == m]["pearson_r"].values for m in order]
    parts = ax.violinplot(data, positions=range(len(order)), showextrema=False, widths=0.85)
    for pc, m in zip(parts["bodies"], order):
        pc.set_facecolor(color_for(m)); pc.set_alpha(0.55); pc.set_edgecolor("none")
    for i, m in enumerate(order):
        med = float(np.median(df[df.method == m]["pearson_r"]))
        ax.plot([i, i], [med, med], color=color_for(m), lw=2.4)
        ax.text(i, med + 0.04, f"{med:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=8, rotation=18, ha="right")
    ax.set_ylabel("per-protein Pearson r")
    ax.axhline(0, color="0.2", lw=0.6, ls="--")
    _save(fig, "fig2d_liver_zonated_violin", stats.reset_index().rename(columns={"index": "method"}))


# --------------------------------------------------------------------------- #
def panel_2traj():
    """Fig 2b: canonical-marker trajectories along the zonation axis (v3).

    Gates A-H ordered central->portal (HSS = CV-minus-PV score; A = highest
    HSS). Per zone, two canonical markers: the strongest OBSERVED gradients
    whose predicted profiles recover the marker (positive per-protein r) --
    CYP2E1 + CYP1A2 (CV), PCK1 + ARG1 (PV). Gate-mean +/- s.e. step lines on
    the de-confounded, z-scored prediction; dots = observed cells.
    Selection uses observed data + recovery (positive r), not direction.
    Within a zone the second marker is a lighter tint of the zone hue with a
    square marker (GENE_STYLE) so the two are distinguishable.
    """
    GATES = list("ABCDEFGH")

    def gate_order(hss, gate):
        order = []
        for g in GATES:
            m = np.where(gate == g)[0]
            order.extend(m[np.argsort(hss[m])])
        return np.array(order)

    R = _liver_run(RUN_CAN)
    tab = pd.read_csv(LIV_MARKERS / "canonical_table.csv")
    tab = tab[tab.in_data.astype(bool)] if tab.in_data.dtype == bool else tab[tab.in_data == True]  # noqa: E712
    idx = {ac: j for j, ac in enumerate(R["tgt"])}
    rows = []
    for _, r in tab.iterrows():
        if r.AC not in idx:
            continue
        j = idx[r.AC]
        g_obs = _gate_gradient(R["obs"][:, j], R["gate"], R["hss"])
        g_pre = _gate_gradient(R["pred"][:, j], R["gate"], R["hss"])
        ok = np.isfinite(R["obs"][:, j])
        rr = (float(np.corrcoef(R["obs"][ok, j], R["pred"][ok, j])[0, 1])
              if ok.sum() >= 10 and np.std(R["obs"][ok, j]) > 1e-9
              and np.std(R["pred"][ok, j]) > 1e-9 else np.nan)
        rows.append({"gene": str(r.gene).upper(), "AC": r.AC, "zone": r.zone,
                     "g_obs": g_obs, "g_pred": g_pre, "pearson_r": rr})
    src = pd.DataFrame(rows)

    pick = src[src.gene.isin(["CYP2E1", "CYP1A2", "PCK1", "ARG1"])].reset_index(drop=True)

    # Per-marker style: the two markers of a zone keep the zone hue
    # family (caption: "pericentral (blue) / periportal (orange)") but must be
    # distinguishable, so the second marker of each zone takes a lighter
    # Okabe-Ito tint of the same hue plus a square marker (colour+shape double
    # coding, colourblind- and grayscale-safe). First marker = ZONE_C, unchanged.
    GENE_STYLE = {"CYP2E1": ("#0072B2", "o"), "CYP1A2": ("#56B4E9", "s"),
                  "PCK1": ("#D95F02", "o"), "ARG1": ("#E69F00", "s")}

    order = gate_order(R["hss"], R["gate"])
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    xg = np.arange(len(GATES))
    for _, r in pick.iterrows():
        j = idx[r.AC]
        o = R["obs"][order, j]; p = R["pred"][order, j]
        oz = (o - np.nanmean(o)) / (np.nanstd(o) + 1e-9)
        pz = (p - p.mean()) / (p.std() + 1e-9)
        gm, gs = [], []
        for g in GATES:
            m = R["gate"][order] == g
            gm.append(np.mean(pz[m]))
            gs.append(np.std(pz[m]) / np.sqrt(m.sum()) if m.sum() > 1 else 0.0)
        c, mk = GENE_STYLE.get(r.gene, (ZONE_C[r.zone], "o"))
        ax.errorbar(xg, gm, yerr=gs, color=c, lw=1.8, marker=mk, ms=3.5,
                    capsize=2, label=f"{r.gene} ({r.zone}, r={r.pearson_r:.2f})")
        m = np.isfinite(oz)
        gpos = np.array([GATES.index(g) for g in R["gate"][order]])
        ax.plot(gpos[m] + np.random.RandomState(0).uniform(-0.18, 0.18, m.sum()),
                oz[m], ".", color=c, ms=2.0, alpha=0.22)
    ax.set_xticks(xg)
    ax.set_xticklabels([f"{g}" for g in GATES], fontsize=8)
    ax.set_xlabel("Lobule position (FACS gates, central $\\to$ portal)")
    ax.set_ylabel("abundance (z)")
    ax.set_ylim(-2.5, 2.5)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=2, columnspacing=1.0, handlelength=1.4, handletextpad=0.4)
    _save(fig, "fig2b_liver_marker_trajectories", src)


def _plm_preds():
    """PLM-kNN / PLM-ridge target predictions from the canonicalax run,
    de-confounded with the same per-cell mean m_i as _liver_run."""
    import anndata as ad
    d = np.load(RUN_CAN / "expansion.npz", allow_pickle=True)
    lab = np.load(RUN_CAN / "labels.npz", allow_pickle=True)
    a = ad.read_h5ad(str(LIV_H5AD))
    pid = [str(p) for p in lab["protein_ids"]]
    col = {p: i for i, p in enumerate(a.var_names)}
    Xfull = np.asarray(a.X[:, [col[p] for p in pid]], dtype=float)
    m_i = np.nanmean(Xfull, axis=1)
    return (d["pred_target__plm_knn"] - m_i[:, None],
            d["pred_target__plm_ridge"] - m_i[:, None])


def panel_2b_marker(gene, zone, name):
    """Fig 2b (2026-09-22 restyle): ONE canonical marker per panel; observed
    and every method's prediction as gate-mean +/- s.e. lines (no single-cell
    dots). Method colours follow Fig 2a (PALETTE2 via color_for): SABER dark
    blue, no-PLM light blue (solid), PLM-kNN vermillion, PLM-ridge gold;
    observed dark grey (not a Fig 2a method). Tight y-axis covers every
    plotted point incl. error bars, floored at +/-1; per-gate observed-cell
    counts ride the tick labels. Main-text markers: CYP1A2 (CV), PCK1 (PV)."""
    _nop_name = f"easyscp_expansion_canonicalax_noplm{SUFFIX}"
    RUN_NOP = outputs_root() / _nop_name / _nop_name / "artifacts"
    R = _liver_run(RUN_CAN)
    Rn = _liver_run(RUN_NOP)
    pk_all, pr_all = _plm_preds()
    tab = pd.read_csv(LIV_MARKERS / "canonical_table.csv")
    tab = tab[tab.in_data.astype(bool)] if tab.in_data.dtype == bool else tab[tab.in_data == True]  # noqa: E712
    gene2ac = {str(r.gene).upper(): r.AC for _, r in tab.iterrows()}
    idx = {ac: j for j, ac in enumerate(R["tgt"])}
    gates = list("ABCDEFGH")
    order = np.concatenate([np.where(R["gate"] == g)[0][np.argsort(
        R["hss"][np.where(R["gate"] == g)[0]])] for g in gates])
    j = idx[gene2ac[gene]]
    o, p, pa = R["obs"][order, j], R["pred"][order, j], Rn["pred"][order, j]
    pk, pr = pk_all[:, j], pr_all[:, j]
    oz = (o - np.nanmean(o)) / (np.nanstd(o) + 1e-9)
    pz = (p - p.mean()) / (p.std() + 1e-9)
    paz = (pa - pa.mean()) / (pa.std() + 1e-9)
    pkz = (pk - pk.mean()) / (pk.std() + 1e-9)
    prz = (pr - pr.mean()) / (pr.std() + 1e-9)

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    xg = np.arange(len(gates))
    gm_o, gs_o, gm_p, gs_p, gm_a, gs_a, gm_k, gm_r, n_o = ([] for _ in range(9))
    for g in gates:
        msk = R["gate"][order] == g
        fo = np.isfinite(oz[msk])
        n_o.append(int(fo.sum()))
        gm_o.append(np.mean(oz[msk][fo]) if fo.any() else np.nan)
        gs_o.append(np.std(oz[msk][fo]) / np.sqrt(fo.sum()) if fo.sum() > 1 else 0.0)
        gm_p.append(np.mean(pz[msk]))
        gs_p.append(np.std(pz[msk]) / np.sqrt(msk.sum()) if msk.sum() > 1 else 0.0)
        gm_a.append(np.mean(paz[msk]))
        gs_a.append(np.std(paz[msk]) / np.sqrt(msk.sum()) if msk.sum() > 1 else 0.0)
        gm_k.append(np.mean(pkz[msk]))
        gm_r.append(np.mean(prz[msk]))

    ax.errorbar(xg, gm_o, yerr=gs_o, color="0.25", lw=1.8, marker="o",
                ms=3.5, capsize=2, label="observed")
    ax.errorbar(xg, gm_k, color=color_for("PLM-kNN"), lw=1.2, marker="^",
                ms=3.0, label="PLM-kNN")
    ax.errorbar(xg, gm_r, color=color_for("PLM-ridge"), lw=1.2, marker="s",
                ms=3.0, label="PLM-ridge")
    ax.errorbar(xg, gm_a, yerr=gs_a, color=color_for("SABER no PLM"), lw=1.3,
                marker="o", ms=2.8, capsize=2, label="SABER (no PLM)")
    ax.errorbar(xg, gm_p, yerr=gs_p, color=color_for("SABER"), lw=1.8,
                marker="o", ms=3.5, capsize=2, label="SABER")
    ax.set_xticks(xg)
    ax.set_xticklabels([f"{g}\n({n})" for g, n in zip(gates, n_o)], fontsize=8)
    ax.set_xlabel("FACS gates, central $\\to$ portal  ((n) = observed cells)",
                  fontsize=8)
    ax.set_ylabel("abundance")
    exts = []
    for arr, gs in ((gm_o, gs_o), (gm_p, gs_p), (gm_a, gs_a)):
        for v, s in zip(arr, gs):
            if np.isfinite(v):
                exts += [abs(v - s), abs(v + s)]
    exts += [abs(v) for v in list(gm_k) + list(gm_r) if np.isfinite(v)]
    half = 1.0 if max(exts) <= 1.0 else max(exts) * 1.06
    ax.set_ylim(-half, half)
    ax.set_title(f"{gene} ({zone} marker)", fontsize=9.5, pad=26)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left",
              bbox_to_anchor=(0, 1.01), ncol=3, columnspacing=1.0,
              handlelength=1.4, handletextpad=0.4)
    _save(fig, name)


def panel_2b1():
    panel_2b_marker("CYP1A2", "CV", "fig2b1_liver_marker_traj_CYP1A2")


def panel_2b2():
    panel_2b_marker("PCK1", "PV", "fig2b2_liver_marker_traj_PCK1")


# --------------------------------------------------------------------------- #
def panel_2heat():
    """New Fig 2c: zonation heatmap, observed (top) vs SABER-predicted (bottom).

    Row selection (deterministic, method-independent): the 30 strongest CV-high
    and 30 strongest PV-high zonated proteins by OBSERVED gate gradient; rows
    sorted by observed gradient within each zone block. Columns = 608 cells
    ordered by HSS; shared column order across both blocks.
    """
    R = _liver_run(RUN_ZON)
    tab = pd.read_csv(LIV_MARKERS / "zonated_table.csv")
    zone_of = dict(zip(tab.AC, tab.zone))
    idx = {ac: j for j, ac in enumerate(R["tgt"])}
    rows = []
    for ac, zone in zone_of.items():
        if zone not in ZONE_C or ac not in idx:
            continue
        j = idx[ac]
        g_obs = _gate_gradient(R["obs"][:, j], R["gate"], R["hss"])
        if np.isfinite(g_obs):
            rows.append({"AC": ac, "zone": zone, "g_obs": g_obs})
    cand = pd.DataFrame(rows)
    top = pd.concat([cand[cand.zone == "CV"].nlargest(30, "g_obs"),
                     cand[cand.zone == "PV"].nsmallest(30, "g_obs")]).reset_index(drop=True)
    j_idx = [idx[a] for a in top.AC]

    order = np.argsort(R["hss"])
    obs = R["obs"][:][:, j_idx][order, :].T   # rows = proteins, cols = cells
    pre = R["pred"][:][:, j_idx][order, :].T
    # per-protein z-score for display (obs on observed stats, pred on all cells)
    oz = (obs - np.nanmean(obs, axis=1, keepdims=True)) / (np.nanstd(obs, axis=1, keepdims=True) + 1e-9)
    pz = (pre - pre.mean(axis=1, keepdims=True)) / (pre.std(axis=1, keepdims=True) + 1e-9)
    oz, pz = np.clip(oz, -3, 3), np.clip(pz, -3, 3)

    gate_seq = R["gate"][order]
    gate_cmap = np.array([plt.cm.tab10(ord(g) - ord("A")) for g in gate_seq])[None, :, :3]

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 4.9),
                             gridspec_kw={"height_ratios": [0.06, 1, 1], "hspace": 0.06})
    gax, axo, axp = axes
    gax.imshow(gate_cmap.astype(float), aspect="auto", interpolation="nearest")
    gax.set_xticks([]); gax.set_yticks([])
    gax.set_ylabel("gate", fontsize=7, rotation=0, ha="right", va="center")
    for ax, M, ttl in ((axo, oz, "observed"), (axp, pz, "SABER-predicted")):
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                       interpolation="nearest")
        ax.set_xticks([])
        ax.set_ylabel(ttl, fontsize=8)
        for y in (29.5,):
            ax.axhline(y, color="k", lw=0.8)
        # zone side bars
        for k, (zn, n) in enumerate((("CV", 30), ("PV", 30))):
            ax.text(-0.012, (0 if k == 0 else 30) + n / 2 - 0.5, zn, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=7, color=ZONE_C[zn], fontweight="bold")
    axp.set_xlabel("cells ordered by HSS (portal $\\to$ central)", fontsize=8)
    cb = fig.colorbar(im, ax=[gax, axo, axp], shrink=0.55, pad=0.015)
    cb.set_label("abundance (z)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    src = top.copy()
    g_pre = [_gate_gradient(R["pred"][:, idx[a]], R["gate"], R["hss"]) for a in top.AC]
    src["g_pred"] = g_pre
    _save(fig, "fig2c_liver_zonation_heatmap", src)


# --------------------------------------------------------------------------- #
def panel_external_go():
    """External GO:0000278 validation of the cell-cycle recovery (companion to Fig. 5e)."""
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    methods = ["observed", "SABER", "pimms_dae", "omicsgmf", "KNN"]
    vals = {"observed": (0.214, 0.194), "SABER": (0.368, 0.335),
            "pimms_dae": (0.255, 0.232), "omicsgmf": (0.239, 0.218), "KNN": (0.245, 0.234)}
    colors = {"observed": "#BBBBBB", "SABER": color_for("SABER"),
              "pimms_dae": "#009E73", "omicsgmf": "#CC79A7", "KNN": color_for("KNN")}
    labels = {"observed": "Observed", "SABER": "SABER", "pimms_dae": "PIMMS-DAE",
              "omicsgmf": "omicsGMF", "KNN": "KNN"}
    x = np.arange(len(methods)); w = 0.38
    for i, m in enumerate(methods):
        go, rest = vals[m]
        ax.bar(i - w / 2, go, width=w * 0.9, color=colors[m])
        ax.bar(i + w / 2, rest, width=w * 0.9, color=colors[m], alpha=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in methods], fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("oscillation gradient |g|")
    ax.text(1, vals["SABER"][0] + 0.012, "*", ha="center", fontsize=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.4"),
               plt.Rectangle((0, 0), 1, 1, color="0.4", alpha=0.35)]
    # legend above the axes: at 'upper left' it overlaps the tall SABER bar
    ax.legend(handles, ["GO:0000278 mitotic (n=115)", "rest of proteome"],
              fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=2, columnspacing=1.0, handlelength=1.2, handletextpad=0.4)
    _save(fig, "fig2_external_go_cellcycle",
          pd.DataFrame([{"method": m, "go0000278": vals[m][0], "rest": vals[m][1]}
                        for m in methods]))


def main():
    style.configure()
    panel_2a()
    panel_2b()
    panel_2c()
    panel_2traj()   # 4-marker dot version; not used in the manuscript
    panel_2b1()     # fig2b (left): CYP1A2 one-marker line restyle
    panel_2b2()     # fig2b (right): PCK1 one-marker line restyle
    panel_external_go()


if __name__ == "__main__":
    main()
