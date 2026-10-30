"""SABER Fig 2 (imputation) — ONE script produces ALL imputation panels.

Panels (canonical 5 non-LFQ datasets; Derks_2023 excluded):
  2a  method x dataset grouped bar @ miss80                      [main]
  2a' method x dataset grouped bar @ miss50                      [secondary parity panel]
  2b  per-fraction line (miss20/50/80), no error bars            [main]
  2c  paired per-protein scatter (SABER vs PIMMS-DAE), 5 datasets [main]
  2d  expression-map showcase: cell-type markers @ miss50,
      True / Observed (missing shown light gray) / Predicted      [main]
  E   component ablation (SABER / no-PLM / no-W) @ miss50         [supp]
  F   heatmap (method x dataset @ miss50) + MAE/RMSE bar          [supp]

Style rules (project-wide): no verbose titles (panel letters + caption carry
meaning at assembly), no parentheticals on titles/axis labels, methodological
detail (n, error-bar def, subset, metric definition) lives in the caption /
FIGURE_PROVENANCE.md. Axis labels kept short ("Pearson r").

Run:  conda run -n saber python scripts/figures/imputation_figures.py [--run <run_dir>]
      --run selects the campaign under outputs/ (default: locked 10-seed frozen+uniform
      run). Reads source_metrics_long.csv + per-protein CSVs + seed42 artifacts from
      outputs/<run>/. Each panel -> png+pdf + <name>_source.csv + manifest entry.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

import style  # noqa: E402
from style import (  # noqa: E402
    get_plt, save_panel, write_source, log_panel, write_manifest, reset_manifest,
    color_for, method_label, dataset_label, DATASETS, keep_datasets,
    load_long, select_metric, agg_mean_sd,
)

plt = get_plt()

# Per-run paths (RUN_DIR, source_data/) are resolved at runtime from style.RUN_DIR,
# which main() sets via style.configure(args.run). No hard-coded run here.

DS_ORDER = DATASETS
FRACS = ["0.2", "0.5", "0.8"]
FRAC_LAB = {"0.2": "20%", "0.5": "50%", "0.8": "80%"}

# SABER no W dropped 2026-09-04 (user): the no-W ablation lives in SI S1;
# the main panel compares SABER against distinct baselines only
M_2A = ["SABER", "SABER no PLM", "PIMMS-DAE", "omicsGMF", "KNN", "Cell mean"]
M_2B = ["SABER", "SABER no PLM", "PIMMS-DAE", "omicsGMF"]
M_ABL = ["SABER", "SABER no PLM", "SABER no W"]
YL = "Pearson r"  # short axis label; definition in caption


# --------------------------------------------------------------------------- #
def load_impute_long() -> pd.DataFrame:
    df = keep_datasets(load_long(), include_lfq=False)
    df = df[df.task == "imputation"].copy()
    df["missing_fraction"] = df["missing_fraction"].astype(str)
    return df


def _grouped_bar(ax, summ, x_col, x_order, methods, ylabel=""):
    x = np.arange(len(x_order)); n = len(methods); w = 0.82 / max(n, 1)
    offs = (np.arange(n) - (n - 1) / 2) * w
    for i, m in enumerate(methods):
        means, sds = [], []
        for xv in x_order:
            r = summ[(summ[x_col] == xv) & (summ.method_label == m)]
            means.append(float(r["mean"].iloc[0]) if not r.empty else np.nan)
            sds.append(float(r["sd"].iloc[0]) if not r.empty else 0.0)
        ax.bar(x + offs[i], means, width=w * 0.92, yerr=sds, capsize=1.6, label=m,
               color=color_for(m), edgecolor="white", error_kw=dict(lw=0.8, ecolor="#222222"))
    ax.set_xticks(x)
    ax.set_xticklabels([dataset_label(v) for v in x_order], rotation=18, ha="right")
    if ylabel:
        ax.set_ylabel(ylabel)


# --------------------------------------------------------------------------- #
def panel_2a(df):
    """Main imputation bar panel.

    v2.1 2026-08-18: MAIN setting switched miss50 -> miss80. Reasons (from the
    0801 10-seed campaign, Derks_2023 excluded, n=50):
      miss50 SABER-PIMMS-DAE  +0.009 CI [-0.003,+0.026]  (crosses 0)
      miss80 SABER-PIMMS-DAE  +0.028 CI [+0.005,+0.056]  (excludes 0)
      miss80 SABER-omicsGMF   +0.062 CI [+0.016,+0.105]  (excludes 0)
    and 80% matches the real MS-SCP missingness regime (50-90%). miss50 stays
    as a secondary panel (panel_2a_miss50) and in the text as the parity claim.
    """
    sub = select_metric(df, "imputation", "pearson_missing_mean", missing_fraction="0.8")
    # mnar50 slice, same rule as panel_2b: the grid campaign carries 5 MNAR
    # shares for saber/noplm but externals only exist at mnar50 — an unfiltered
    # mean would mix slices across methods. (Restored 2026-10-13: the
    # published fig3b source CSV — SABER count=10/seed — was rendered with
    # this filter; it had been lost from the file.)
    if "mnar_fraction" in sub.columns:
        sub = sub[sub["mnar_fraction"].astype(str) == "0.5"]
    summ = agg_mean_sd(sub, ["dataset", "method_label"])
    summ = summ[summ.method_label.isin(M_2A)]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    _grouped_bar(ax, summ, "dataset", DS_ORDER, M_2A, ylabel=YL)
    ax.axhline(0, color="#999", lw=0.5)
    # '*' over the SABER bar where it significantly beats the best COMPETING
    # method in that dataset (one-sided paired Wilcoxon over 10 seeds, p<0.05;
    # 'SABER no PLM' is our own ablation, not a competitor — excluded from the
    # runner-up pick but kept in the plot) — decision 2026-09-04
    from scipy.stats import wilcoxon
    competitors = [m for m in M_2A if m not in ("SABER", "SABER no PLM")]
    n = len(M_2A); w = 0.82 / n
    x_off = (0 - (n - 1) / 2) * w          # SABER is the first bar in each group
    for xi, d in enumerate(DS_ORDER):
        r = summ[(summ.dataset == d) & (summ.method_label == "SABER")]
        others = summ[(summ.dataset == d) & (summ.method_label.isin(competitors))]
        if r.empty or others.empty:
            continue
        best = others.loc[others["mean"].idxmax(), "method_label"]
        piv = sub[sub.dataset == d].pivot_table(index="seed", columns="method_label",
                                                values="value")
        if {"SABER", best}.issubset(piv.columns):
            pair = piv[["SABER", best]].dropna()
            if len(pair) >= 6 and (pair["SABER"] > pair[best]).any():
                stat, p = wilcoxon(pair["SABER"], pair[best], alternative="greater")
                if p < 0.05:
                    y = float(r["mean"].iloc[0]) + float(r["sd"].iloc[0])
                    ax.text(xi + x_off, y + 0.008, "*", ha="center", va="bottom",
                            fontsize=11)
    ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.26),
              frameon=False)
    name = "fig2a_imputation_bar_miss80"
    save_panel(fig, name); plt.close(fig)
    write_source(summ, name)
    log_panel("2a", "main_bar", f"{name}.png", f"{name}_source.csv",
              "method x dataset per-protein Pearson @ miss80, 10-seed mean±SD; "
              "* = SABER > best other method (10-seed paired Wilcoxon p<0.05)",
              n_rows=len(summ), params=dict(methods=M_2A, missing_fraction="0.8"))


def panel_2a_miss50(df):
    """Secondary parity panel at the conventional miss50 setting (SI / text)."""
    sub = select_metric(df, "imputation", "pearson_missing_mean", missing_fraction="0.5")
    summ = agg_mean_sd(sub, ["dataset", "method_label"])
    summ = summ[summ.method_label.isin(M_2A)]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    _grouped_bar(ax, summ, "dataset", DS_ORDER, M_2A, ylabel=YL)
    ax.axhline(0, color="#999", lw=0.5)
    ax.legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.26),
              frameon=False)
    name = "fig2a_imputation_bar_miss50"
    save_panel(fig, name); plt.close(fig)
    write_source(summ, name)
    log_panel("2a", "parity_bar_miss50", f"{name}.png", f"{name}_source.csv",
              "method x dataset per-protein Pearson @ miss50, 10-seed mean±SD",
              n_rows=len(summ), params=dict(methods=M_2A, missing_fraction="0.5"))


def panel_2b(df):
    sub = select_metric(df, "imputation", "pearson_missing_mean")
    # grid campaigns carry several MNAR shares; the by-fraction curve is the
    # balanced mnar50 slice (matches the canonical miss{20,50,80}_mnar50 runs)
    if "mnar_fraction" in sub.columns:
        sub = sub[sub["mnar_fraction"].astype(str) == "0.5"]
    summ = agg_mean_sd(sub, ["missing_fraction", "method_label"])
    summ = summ[summ.method_label.isin(M_2B)]
    fracs = sorted(f for f in summ["missing_fraction"].unique() if float(f) > 0)
    n = len(fracs)
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    for m in M_2B:
        r = summ[summ.method_label == m].set_index("missing_fraction").reindex(fracs)
        ax.plot(range(n), r["mean"], marker="o", ms=4.5, lw=1.9, label=m, color=color_for(m))
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{int(float(f) * 100)}%" for f in fracs], fontsize=8)
    ax.set_xlabel("Total missingness"); ax.set_ylabel(YL)
    ax.legend(fontsize=6, frameon=False)
    name = "fig2b_imputation_byfraction"
    save_panel(fig, name); plt.close(fig)
    write_source(summ, name)
    log_panel("2b", "main_line", f"{name}.png", f"{name}_source.csv",
              "per-fraction per-protein Pearson at mnar50 (mean over datasets x seeds); no error bars",
              n_rows=len(summ), params=dict(methods=M_2B, fracs=fracs))


# --------------------------------------------------------------------------- #
# SI metric companions (2026-10-13, group-meeting Q&A): the two main imputation
# panels — miss80 bar (2a = paper fig3b) and by-fraction line (2b = paper
# fig3a) — under alternative metrics. Same data slice, method set and style as
# the corresponding main panel; ONLY the metric changes. Significance stars
# follow panel_2a's rule with direction flipped for error metrics.
# --------------------------------------------------------------------------- #
SI_METRICS = [
    ("spearman_missing_mean", "spearman", "Spearman ρ", True),
    ("mae_missing", "mae", "MAE", False),
]


def panel_2a_metric(df, metric, short, ylabel, higher):
    """SI: method x dataset bar @ miss80 mnar50 under an alternative metric."""
    sub = select_metric(df, "imputation", metric, missing_fraction="0.8")
    if "mnar_fraction" in sub.columns:
        sub = sub[sub["mnar_fraction"].astype(str) == "0.5"]
    summ = agg_mean_sd(sub, ["dataset", "method_label"])
    summ = summ[summ.method_label.isin(M_2A)]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    _grouped_bar(ax, summ, "dataset", DS_ORDER, M_2A, ylabel=ylabel)
    if higher:
        ax.axhline(0, color="#999", lw=0.5)
    from scipy.stats import wilcoxon
    competitors = [m for m in M_2A if m not in ("SABER", "SABER no PLM")]
    n = len(M_2A); w = 0.82 / n
    x_off = (0 - (n - 1) / 2) * w          # SABER is the first bar in each group
    ymax = float((summ["mean"] + summ["sd"]).max())
    for xi, d in enumerate(DS_ORDER):
        r = summ[(summ.dataset == d) & (summ.method_label == "SABER")]
        others = summ[(summ.dataset == d) & (summ.method_label.isin(competitors))]
        if r.empty or others.empty:
            continue
        best = others.loc[others["mean"].idxmax() if higher else others["mean"].idxmin(),
                          "method_label"]
        piv = sub[sub.dataset == d].pivot_table(index="seed", columns="method_label",
                                                values="value")
        if {"SABER", best}.issubset(piv.columns):
            pair = piv[["SABER", best]].dropna()
            win = (pair["SABER"] > pair[best]).any() if higher else (pair["SABER"] < pair[best]).any()
            if len(pair) >= 6 and win:
                stat, p = wilcoxon(pair["SABER"], pair[best],
                                   alternative="greater" if higher else "less")
                if p < 0.05:
                    y = float(r["mean"].iloc[0]) + float(r["sd"].iloc[0])
                    ax.text(xi + x_off, y + 0.012 * ymax, "*", ha="center", va="bottom",
                            fontsize=11)
    ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.26),
              frameon=False)
    name = f"fig2a_imputation_bar_miss80_{short}"
    save_panel(fig, name); plt.close(fig)
    write_source(summ, name)
    log_panel("2a-si", f"main_bar_{short}", f"{name}.png", f"{name}_source.csv",
              f"method x dataset {metric} @ miss80 mnar50, 10-seed mean±SD; "
              f"* = SABER {'>' if higher else '<'} best other method (paired Wilcoxon p<0.05)",
              n_rows=len(summ), params=dict(methods=M_2A, missing_fraction="0.8", metric=metric))


def panel_2b_metric(df, metric, short, ylabel, higher):
    """SI: per-fraction line @ mnar50 under an alternative metric."""
    sub = select_metric(df, "imputation", metric)
    if "mnar_fraction" in sub.columns:
        sub = sub[sub["mnar_fraction"].astype(str) == "0.5"]
    summ = agg_mean_sd(sub, ["missing_fraction", "method_label"])
    summ = summ[summ.method_label.isin(M_2B)]
    fracs = sorted(f for f in summ["missing_fraction"].unique() if float(f) > 0)
    n = len(fracs)
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    for m in M_2B:
        r = summ[summ.method_label == m].set_index("missing_fraction").reindex(fracs)
        ax.plot(range(n), r["mean"], marker="o", ms=4.5, lw=1.9, label=m, color=color_for(m))
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{int(float(f) * 100)}%" for f in fracs], fontsize=8)
    ax.set_xlabel("Total missingness"); ax.set_ylabel(ylabel)
    ax.legend(fontsize=6, frameon=False)
    name = f"fig2b_imputation_byfraction_{short}"
    save_panel(fig, name); plt.close(fig)
    write_source(summ, name)
    log_panel("2b-si", f"main_line_{short}", f"{name}.png", f"{name}_source.csv",
              f"per-fraction {metric} at mnar50 (mean over datasets x seeds); no error bars",
              n_rows=len(summ), params=dict(methods=M_2B, fracs=fracs, metric=metric))


def _montage(names, out_name, gap=40, bg="white"):
    """Horizontal montage of already-saved panel PNGs (SI assembly convenience)."""
    from PIL import Image
    ims = [Image.open(str(style.FIG_ROOT / "imputation" / f"{n}.png")) for n in names]
    h = max(im.height for im in ims)
    ims = [im.resize((int(im.width * h / im.height), h)) for im in ims]
    W = sum(im.width for im in ims) + gap * (len(ims) - 1)
    canvas = Image.new("RGB", (W, h), bg)
    x = 0
    for im in ims:
        canvas.paste(im, (x, 0)); x += im.width + gap
    p = style.FIG_ROOT / "imputation" / f"{out_name}.png"
    canvas.save(p, dpi=(300, 300))
    return p


def _load_perprotein(dataset, frac_int=50, cond="saber"):
    pat = str(style.RUN_DIR / "source_data" / "seed*" / f"{dataset}_miss{frac_int}_mnar50_{cond}"
              / "imputation" / "source_imputation_per_protein.csv")
    frames = []
    for f in glob.glob(pat):
        d = pd.read_csv(f); d["seed"] = int(re.search(r"seed(\d+)", f).group(1)); frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def panel_2c(df, frac=50):
    """Paired per-protein scatter: SABER vs PIMMS-DAE, as FIVE SEPARATE per-dataset
    figures (author picks which to feature). Points above the dashed diagonal =
    proteins SABER predicts better; inset % quantifies the consistent per-protein
    win that a mean bar / overlapping violin hides."""
    base = "PIMMS-DAE"
    for ds in DS_ORDER:
        pp = _load_perprotein(ds, frac, "saber")
        if pp.empty:
            print(f"  [skip] panel 2c {ds}: per-protein CSVs need the campaign run tree (--run <run>)")
            continue
        pp["method_label"] = [method_label(m, "") for m in pp.method]
        pp = pp[pp.pearson_missing.notna() & pp.method_label.isin(["SABER", base])]
        piv = (pp.groupby(["method_label", "protein_id"])["pearson_missing"].mean()
                  .unstack("method_label").dropna())
        if piv.empty:
            continue
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        ax.scatter(piv[base], piv["SABER"], s=9, alpha=0.32, color=color_for("SABER"),
                   edgecolors="none")
        lo = float(np.nanmin([piv[base].min(), piv["SABER"].min()]))
        hi = float(np.nanmax([piv[base].max(), piv["SABER"].max()]))
        ax.plot([lo, hi], [lo, hi], color="#888", lw=0.9, ls="--")
        above = (piv["SABER"] > piv[base]).mean() * 100
        ax.text(0.04, 0.96, f"{above:.0f}% above diagonal", transform=ax.transAxes,
                fontsize=7, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb", lw=0.5))
        ax.set_xlabel(f"{base} " + YL, fontsize=7.5)
        ax.set_ylabel("SABER " + YL, fontsize=7.5)
        ax.set_title(dataset_label(ds), fontsize=8.5)
        fig.tight_layout()
        name = f"fig2c_imputation_pairedscatter_{ds}"
        save_panel(fig, name); plt.close(fig)
        write_source(piv.assign(dataset=ds), name)
        log_panel("2c", "main_paired_scatter", f"{name}.png", f"{name}_source.csv",
                  f"paired per-protein Pearson: SABER vs {base} @ miss{frac}, {ds}; "
                  "points above dashed diagonal = SABER better for that protein",
                  n_rows=len(piv), params=dict(baseline=base, dataset=ds, missing_fraction="0.5"))


# --------------------------------------------------------------------------- #
# 2d: data-driven cell-type markers (Derks_2022 @ miss50), True/Obs(gray miss)/Pred
# --------------------------------------------------------------------------- #
def _pca2(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :2] * S[:2]


def _cell_type_labels(prepared_h5ad, n_cells):
    try:
        import anndata as ad
        obs = ad.read_h5ad(prepared_h5ad).obs
        for cand in ["cell_type", "Celltype", "cell_type_raw"]:
            if cand in obs.columns:
                vals = list(obs[cand].astype(str).values)
                return vals[:n_cells] if len(vals) >= n_cells else None
    except Exception:
        return None
    return None


def panel_2d():
    ds = "Derks_2022"
    art = style.RUN_DIR / "seeds" / "seed42" / "runs" / f"{ds}_miss50_mnar50_saber" / "artifacts"
    imp = np.load(art / "imputation.npz", allow_pickle=True)
    lab = np.load(art / "labels.npz", allow_pickle=True)
    x_true = imp["x_true"]; obs_mask = imp["observed_mask"].astype(bool)
    pred = imp["pred__saber"]
    ctypes = lab["cell_types"]; pids = lab["protein_ids"]
    prepared = glob.glob(str(style.RUN_DIR / "seeds" / "seed42" / "prepared_inputs" / ds / "*.h5ad"))
    ct_names = _cell_type_labels(prepared[0], x_true.shape[0]) if prepared else None
    uniq = sorted(np.unique(ctypes).tolist())
    if ct_names:
        name_of = {}
        for i in uniq:
            idx = np.where(ctypes == i)[0]
            vals, counts = np.unique([ct_names[j] for j in idx], return_counts=True)
            name_of[i] = str(vals[np.argmax(counts)])
    else:
        name_of = {i: f"type {i}" for i in uniq}

    group_mean = {i: x_true[ctypes == i].mean(axis=0) for i in uniq}
    spec = np.full(len(pids), -np.inf); arg_type = np.zeros(len(pids), int)
    for j in range(len(pids)):
        v = np.array([group_mean[i][j] for i in uniq])
        o = np.argsort(v)[::-1]
        arg_type[j] = uniq[o[0]]; spec[j] = v[o[0]] - v[o[1]] if len(uniq) > 1 else v[o[0]]
    spec_df = pd.DataFrame({"protein_id": pids, "cell_type_int": arg_type,
                            "cell_type": [name_of[t] for t in arg_type], "specificity": spec})
    pp = _load_perprotein(ds, 50, "saber")
    pearson = pp[pp.method == "saber"].groupby("protein_id")["pearson_missing"].mean()
    spec_df["pearson"] = spec_df.protein_id.map(pearson)
    spec_df = spec_df.dropna(subset=["pearson"]).copy()
    med = spec_df.pearson.median()
    chosen = []
    for t in uniq:
        sub = spec_df[spec_df.cell_type_int == t].sort_values("specificity", ascending=False).head(15)
        chosen.extend(sub[sub.pearson >= med].head(3).index.tolist())
    write_source(spec_df.loc[list(dict.fromkeys(chosen))].assign(reason="top-15 specific & pearson>=median, top3/type"),
                 "fig2d_imputation_marker_candidates")

    emb = _pca2(np.nan_to_num(x_true))
    drawn = []
    for jx in dict.fromkeys(chosen):
        j = spec_df.index.get_loc(jx)
        pid = spec_df.loc[jx, "protein_id"]; ct = spec_df.loc[jx, "cell_type"]
        true = x_true[:, j]; obs = np.where(obs_mask[:, j], x_true[:, j], np.nan); prd = pred[:, j]
        if np.isfinite(true).sum() < 10:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), sharex=True, sharey=True)
        vmax = np.nanmax(np.abs([true, prd]))
        for ax, (ttl, val) in zip(axes, [("True", true), ("Observed", obs), ("Predicted", prd)]):
            fin = np.isfinite(val)
            if (~fin).any():  # missing entries -> explicit light gray
                ax.scatter(emb[~fin, 0], emb[~fin, 1], c="#dcdcdc", s=7, edgecolors="none")
            ax.scatter(emb[fin, 0], emb[fin, 1], c=val[fin], cmap="viridis", s=7,
                       vmin=-vmax, vmax=vmax, edgecolors="none")
            ax.set_title(ttl, fontsize=7); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{dataset_label(ds)} · {pid} · {ct}", fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fname = f"fig2d_expression_map_{pid}"
        save_panel(fig, fname); plt.close(fig)
        drawn.append(pid)
    log_panel("2d", "main_expression_map", "fig2d_expression_map_<pid>.png",
              "fig2d_imputation_marker_candidates_source.csv",
              f"True/Observed(missing=gray)/Predicted on Derks_2022 @ miss50; markers: {drawn}",
              n_rows=len(drawn), params=dict(dataset=ds, missing_fraction="0.5", method="saber"))


# --------------------------------------------------------------------------- #
def panel_E(df):
    sub = select_metric(df, "imputation", "pearson_missing_mean", missing_fraction="0.5")
    summ = agg_mean_sd(sub, ["dataset", "method_label"])
    summ = summ[summ.method_label.isin(M_ABL)]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    _grouped_bar(ax, summ, "dataset", DS_ORDER, M_ABL, ylabel=YL)
    ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              frameon=False)
    name = "figE_imputation_ablation_miss50"
    save_panel(fig, name); plt.close(fig)
    write_source(summ, name)
    log_panel("E", "supp_ablation", f"{name}.png", f"{name}_source.csv",
              "component ablation SABER / no-PLM / no-W @ miss50", n_rows=len(summ))


def panel_F(df):
    sub = select_metric(df, "imputation", "pearson_missing_mean", missing_fraction="0.5")
    summ = agg_mean_sd(sub, ["dataset", "method_label"])
    allm = [m for m in ["SABER", "SABER no PLM", "SABER no W", "PIMMS-DAE", "PIMMS-VAE",
                        "omicsGMF", "KNN", "Cell mean"] if m in set(summ.method_label)]
    piv = summ.pivot(index="method_label", columns="dataset", values="mean").reindex(index=allm, columns=DS_ORDER)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=0.72)
    ax.set_xticks(range(len(DS_ORDER)))
    ax.set_xticklabels([dataset_label(d) for d in DS_ORDER], rotation=18, ha="right")
    ax.set_yticks(range(len(allm))); ax.set_yticklabels(allm, fontsize=7)
    for i in range(len(allm)):
        for j in range(len(DS_ORDER)):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                        color="white" if v < 0.42 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, label=YL)
    name = "figF_imputation_heatmap_miss50"
    save_panel(fig, name); plt.close(fig)
    write_source(piv.reset_index(), name)
    log_panel("F1", "supp_heatmap", f"{name}.png", f"{name}_source.csv",
              "method x dataset Pearson heatmap @ miss50", n_rows=len(piv))

    mets = [("mae_missing", "MAE"), ("rmse_missing", "RMSE")]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    g_all = []
    for ax, (met, ylab) in zip(axes, mets):
        g = agg_mean_sd(select_metric(df, "imputation", met), ["method_label"])
        g = g[g.method_label.isin(allm)].sort_values("mean")
        g_all.append(g.assign(metric=met))
        ax.barh(range(len(g)), g["mean"], xerr=g["sd"],
                color=[color_for(m) for m in g.method_label], edgecolor="white",
                error_kw=dict(lw=0.7, ecolor="#222"))
        ax.set_yticks(range(len(g))); ax.set_yticklabels(g.method_label, fontsize=6.5)
        ax.set_xlabel(ylab)
    fig.tight_layout()
    name = "figF_imputation_mae_rmse"
    save_panel(fig, name); plt.close(fig)
    write_source(pd.concat(g_all, ignore_index=True), name)
    log_panel("F2", "supp_errorbar", f"{name}.png", f"{name}_source.csv",
              "MAE/RMSE by method (overall)", n_rows=len(g_all))


# --------------------------------------------------------------------------- #
def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="SABER Fig 2 (imputation) panels.")
    p.add_argument("--run", default=None,
                   help="campaign run dir under outputs/ (default: locked 10-seed run)")
    return p.parse_args()


def main():
    args = _parse_args()
    style.configure(args.run)
    reset_manifest()
    df = load_impute_long()
    panel_2a(df)
    panel_2a_miss50(df)
    panel_2b(df)
    panel_2c(df)
    try:
        panel_2d()
    except Exception as e:
        import traceback; print("[2d] skipped:", repr(e)); traceback.print_exc()
    panel_E(df)
    panel_F(df)
    for metric, short, ylabel, higher in SI_METRICS:
        panel_2a_metric(df, metric, short, ylabel, higher)
        panel_2b_metric(df, metric, short, ylabel, higher)
    m1 = _montage(["fig2b_imputation_byfraction_spearman", "fig2b_imputation_byfraction_mae"],
                  "figS_imputation_byfraction_2metric")
    m2 = _montage(["fig2a_imputation_bar_miss80_spearman", "fig2a_imputation_bar_miss80_mae"],
                  "figS_imputation_bar_2metric")
    print(f"SI montages -> {m1}, {m2}")
    print(f"\nDONE. manifest -> {write_manifest(task='imputation')}")
    print(f"figures -> {style.FIG_ROOT / 'imputation'}")


if __name__ == "__main__":
    main()
