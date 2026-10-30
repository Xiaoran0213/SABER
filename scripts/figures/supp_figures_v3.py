"""Supplementary figures: one script, one style, all SI panels.

Renders every SI panel from the collected source tables with a common layout
(183 mm width, 7 pt text, legends above the axes, SABER always blue). Panel
conventions: S3 references use ESM-2 features only (condition == "saber");
S7 shows per-seed points; S10 uses the 'has_missing' subset, matching the main
text; S12 is Spearman only; S14 SABER only.

Long tables are read from pickles cached under .cache/supp_v3/ (see CACHE).

Run:  python scripts/figures/supp_figures_v3.py [--only S3 S8 ...]
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import style  # noqa: E402
from style import (  # noqa: E402
    get_plt, color_for, dataset_label, DATASETS, keep_datasets, select_metric, agg_mean_sd,
)

plt = get_plt()
from matplotlib.patches import Ellipse, Patch  # noqa: E402

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "legend.frameon": False, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
})

from saber.paths import figures_root, outputs_root  # noqa: E402

PW_SUFFIX = ""  # proteome-wide campaign tree suffix (set by --pw-suffix)

REPO = style.REPO
OUTS = outputs_root()
OUT = figures_root() / "supp"
# Pickle cache for the (large) long tables, repo-local and gitignored; delete
# .cache/ to force a reload from the run trees.
_CACHE_DIR = REPO / ".cache" / "supp_v3"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE = {
    "final": _CACHE_DIR / "saber_long.pkl",        # paper_benchmark_20260801_final_10seed
    "grid": _CACHE_DIR / "saber_long_grid.pkl",    # paper_benchmark_20260904_grid
    "h0709": _CACHE_DIR / "saber_long_0709.pkl",   # paper_benchmark_20260709_5seed
}
RUNS = {"final": "paper_benchmark_20260801_final_10seed",
        "grid": "paper_benchmark_20260904_grid",
        "h0709": "paper_benchmark_20260709_5seed"}
FULL_W = 7.2
DS = DATASETS  # Derks 2022, Specht 2021, Huffman 2023, Woo 2021
ERR = dict(lw=0.6, ecolor="#333333", capsize=1.4, capthick=0.6)

_long: dict[str, pd.DataFrame] = {}


def fig_src(run_key: str, task: str, name: str) -> Path:
    """Resolve a per-panel source CSV from the figure output tree (current
    location) or the campaign run tree (legacy location)."""
    roots = [figures_root() / "benchmark" / "source_data" / "figures" / task,
             OUTS / RUNS[run_key] / "source_data" / "figures" / task]
    for root in roots:
        for fname in (f"{name}_source.csv", f"{name}.csv"):
            if (root / fname).is_file():
                return root / fname
    return roots[-1] / f"{name}_source.csv"


def long(key: str) -> pd.DataFrame:
    if key not in _long:
        if CACHE[key].exists():
            _long[key] = pd.read_pickle(CACHE[key])
        else:
            style.configure(RUNS[key])
            _long[key] = style.load_long()
            _long[key].to_pickle(CACHE[key])
    return _long[key]


def save(fig, name: str, src: pd.DataFrame | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    if src is not None:
        src.to_csv(OUT / f"{name}_source.csv", index=False)
    print("  wrote", name, flush=True)


def letter(ax, s, x=-0.10, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def top_legend(ax, handles=None, labels=None, ncol=6, y=1.02):
    kw = dict(loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, handlelength=1.2,
              columnspacing=1.0, borderaxespad=0.2)
    if handles is not None:
        ax.legend(handles, labels, **kw)
    else:
        ax.legend(**kw)


def grouped_bar(ax, summ, methods, groups=DS, group_col="dataset", label_fn=dataset_label,
                hatch=None):
    n = len(methods); w = 0.8 / n
    offs = (np.arange(n) - (n - 1) / 2) * w
    for i, m in enumerate(methods):
        r = summ[summ.method_label == m].set_index(group_col).reindex(groups)
        ax.bar(np.arange(len(groups)) + offs[i], r["mean"], width=w * 0.9, yerr=r["sd"],
               color=color_for(m), edgecolor="white", linewidth=0.3, label=m,
               hatch=(hatch or {}).get(m), error_kw=ERR)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([label_fn(g) for g in groups])
    return offs


def imp_slice(df, metric, frac=None):
    sub = select_metric(keep_datasets(df), "imputation", metric)
    sub = sub[sub["mnar_fraction"].astype(float) == 0.5]
    if frac is not None:
        sub = sub[sub["missing_fraction"].astype(float) == frac]
    return sub


# --------------------------------------------------------------------------- S1
def fig_S1():
    df = long("grid")
    methods = ["SABER", "SABER no PLM"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.3), sharey=True)
    out = []
    for k, (ax, frac) in enumerate(zip(axes, (0.5, 0.8))):
        sub = imp_slice(df, "pearson_missing_mean", frac)
        sub = sub.groupby(["dataset", "method_label", "seed"], as_index=False)["value"].mean()
        summ = agg_mean_sd(sub, ["dataset", "method_label"])
        summ = summ[summ.method_label.isin(methods)]
        grouped_bar(ax, summ, methods)
        ax.set_title(f"{int(frac * 100)}% injected missingness", pad=3)
        if k == 0:
            ax.set_ylabel("Per-protein Pearson r")
        letter(ax, "ab"[k], x=-0.12 if k == 0 else -0.05)
        out.append(summ.assign(missing_fraction=frac))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    save(fig, "figS1_ablation_50_80", pd.concat(out))


# --------------------------------------------------------------------------- S2
def fig_S2():
    import missingness_grid_figures as mg
    df = long("grid")
    df = df[df.task == "imputation"].copy()
    for c in ("value", "missing_fraction", "mnar_fraction", "seed"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["value", "missing_fraction", "mnar_fraction"])
    ds = "Derks_2022"
    fr, mn, S = mg._pivot(mg._series(df, "saber", "saber"), ds)
    _, _, C = mg._pivot(mg._series(df, "cell_mean", None), ds)
    _, _, N = mg._pivot(mg._series(df, "saber", "noplm"), ds)
    panels = [("a", "SABER Pearson", S, "viridis", None),
              ("b", "SABER minus cell mean", S - C, "Reds", None),
              ("c", "SABER minus no-PLM", S - N, "RdBu_r", "sym")]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.45))
    for ax, (lt, title, M, cmap, norm) in zip(axes, panels):
        if norm == "sym":
            mx = np.nanmax(np.abs(M)); im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, vmin=-mx, vmax=mx)
        else:
            im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap)
        ax.grid(False)
        ax.set_xticks(range(len(mn))); ax.set_xticklabels([f"{x*100:.0f}" for x in mn])
        ax.set_yticks(range(len(fr))); ax.set_yticklabels([f"{x*100:.0f}" for x in fr])
        ax.set_xlabel("MNAR share (%)")
        if lt == "a":
            ax.set_ylabel("Injected missingness (%)")
        ax.set_title(title, pad=6)
        lo, hi = im.get_clim(); cm = plt.get_cmap(cmap)
        for i in range(len(fr)):
            for j in range(len(mn)):
                v = M[i, j]
                if np.isfinite(v):
                    r, g, b, _ = cm((v - lo) / (hi - lo) if hi > lo else 0.5)
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.6,
                            color="white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black")
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=5.5)
        letter(ax, lt, x=-0.3, y=1.06)
    fig.tight_layout(w_pad=1.6)
    src = pd.DataFrame([dict(missing_fraction=f, mnar_share=m, saber=S[i, j], cell_mean=C[i, j], noplm=N[i, j])
                        for i, f in enumerate(fr) for j, m in enumerate(mn)])
    save(fig, "figS2_grid_Derks_2022", src)


# --------------------------------------------------------------------------- S3
def fig_S3():
    src = fig_src("h0709", "expansion", "fig4e_expansion_sequence_holdout")
    df = pd.read_csv(src).drop_duplicates()
    meth = [("SABER", "saber", "saber", "-"), ("SABER no PLM", "saber", "noplm", "--"),
            ("PLM-kNN", "plm_knn", "saber", "-"), ("PLM-ridge", "plm_ridge", "saber", "-")]
    seq_ds = ["Derks_2022", "Specht_2019", "Huffman_2023_melanoma_HEK"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.2))
    rows = []
    for k, (ax, ds) in enumerate(zip(axes, seq_ds)):
        for lab, m, c, ls in meth:
            a = df[(df.dataset == ds) & (df.method == m) & (df.condition == c)].sort_values("target_fraction")
            ax.plot(a.target_fraction * 100, a.value, marker="o", ms=2.8, lw=1.1, ls=ls,
                    color=color_for(lab), label=lab)
            rows += [dict(dataset=ds, method=lab, fraction=f, mean=v) for f, v in zip(a.target_fraction, a.value)]
        ax.axhline(0, color="#bbbbbb", lw=0.5)
        ax.set_title(dataset_label(ds), pad=3)
        ax.set_xlabel("Proteins held out (%)")
        ax.set_xticks([5, 10, 20, 40])
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        if k == 0:
            ax.set_ylabel("Per-protein Pearson r")
        letter(ax, "abc"[k], x=-0.2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    fig.tight_layout()
    save(fig, "figS3_sequence_holdout", pd.DataFrame(rows))


# --------------------------------------------------------------------------- S6
def fig_S6():
    style.configure(RUNS["final"])
    pw = style.load_proteome_wide()
    deg = pw[(pw.task == "proteome_wide_degradation") & (pw.method == "saber")]
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.2))
    rows = []
    for k, (ax, ds) in enumerate(zip(axes, ["Derks_2022", "Specht_2019"])):
        for lab, cond, ls in [("SABER", "saber", "-"), ("SABER no PLM", "noplm", "--")]:
            pts = []
            for f in (0.5, 0.8, 0.9):
                v = deg[(deg.dataset == ds) & (deg.target_fraction == f) & (deg.condition == cond)]["value"]
                pts.append((f * 100, v.mean(), v.std() if len(v) > 1 else 0.0, len(v)))
                rows.append(dict(dataset=ds, method=lab, fraction=f, mean=v.mean(), sd=v.std(), n=len(v)))
            p = np.array(pts)
            ax.errorbar(p[:, 0], p[:, 1], yerr=p[:, 2], marker="o", ms=3, lw=1.2, ls=ls,
                        color=color_for(lab), label=lab, capsize=1.5, elinewidth=0.6)
        ax.axhline(0, color="#bbbbbb", lw=0.5)
        ax.set_xticks([50, 80, 90]); ax.set_xlabel("Proteins held out (%)")
        ax.set_title(dataset_label(ds), pad=3)
        if k == 0:
            ax.set_ylabel("Per-protein Pearson r")
        letter(ax, "ab"[k], x=-0.28)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    save(fig, "figS6_proteome_wide_degradation", pd.DataFrame(rows))


# --------------------------------------------------------------------------- S7
def fig_S7():
    go = OUTS / f"proteome-wide{PW_SUFFIX}" / "results" / "go"
    key = {"SABER": "saber", "SABER no PLM": "noplm", "PLM-kNN": "plm_knn",
           "PLM-ridge": "plm_ridge", "Cell mean": "cell_mean"}
    cts = ["Melanoma", "PDAC", "U-937"]
    per = pd.concat([pd.read_csv(f).assign(seed=int(re.search(r"seed(\d+)", f.name).group(1)))
                     for f in sorted(go.glob("coherence_summary_Derks_2022_holdout80_seed*.csv"))])
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    methods = list(key)
    n = len(methods); w = 0.8 / n; offs = (np.arange(n) - (n - 1) / 2) * w
    rng = np.random.default_rng(0)
    rows = []
    for i, m in enumerate(methods):
        sub = per[per.method == key[m]]
        for xi, c in enumerate(cts):
            v = sub[sub.cell_type == c]["n_sig_GO_BP"].astype(float).to_numpy()
            mu = float(v.mean()) if len(v) else 0.0
            ax.bar(xi + offs[i], mu, width=w * 0.88, color=color_for(m), edgecolor=color_for(m),
                   linewidth=0.5, label=m if xi == 0 else None)
            ax.scatter(xi + offs[i] + rng.uniform(-w * 0.18, w * 0.18, len(v)), v, s=5,
                       color="#222222", zorder=3, linewidths=0)
            rows.append(dict(method=m, cell_type=c, mean=mu, seeds=";".join(f"{x:g}" for x in v)))
    ax.set_xticks(range(len(cts))); ax.set_xticklabels(cts)
    ax.set_ylabel("Significant GO BP terms")
    ax.set_ylim(bottom=0)
    top_legend(ax, ncol=5)
    save(fig, "figS7_expansion_go_coherence", pd.DataFrame(rows))


# --------------------------------------------------------------------------- S8
def fig_S8():
    from common import method_label
    df = keep_datasets(long("final"))
    df = df[df.task == "cell_embedding"].copy()
    df["ml"] = [method_label(m, "") for m in df.method]
    df = df[df.ml.isin(["SABER", "scProtein", "omicsGMF"])]
    ari = df[df.metric == "ARI_cell"].groupby(["dataset", "ml"])["value"].mean().rename("ari")
    ili = df[df.metric == "iLISI"].groupby(["dataset", "ml"])["value"].mean().rename("ilisi")
    m = pd.concat([ari, ili], axis=1).dropna().reset_index()
    disp = {"scProtein": "scPROTEIN"}
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for ds, g in m.groupby("dataset"):
        cx, cy = g.ilisi.mean(), g.ari.mean()
        wd = (g.ilisi.max() - g.ilisi.min()) + 1.3
        ht = (g.ari.max() - g.ari.min()) + 0.11
        ax.add_patch(Ellipse((cx, cy), wd, ht, facecolor="#f2f2f2", edgecolor="#9a9a9a",
                             lw=0.7, ls="--", zorder=0))
        ax.text(cx, cy + ht / 2 + 0.015, dataset_label(ds), ha="center", va="bottom", fontsize=6.5,
                color="#444444")
    for ml in ["SABER", "scProtein", "omicsGMF"]:
        g = m[m.ml == ml]
        ax.scatter(g.ilisi, g.ari, s=26, color=color_for(ml), edgecolor="white", linewidth=0.5,
                   label=disp.get(ml, ml), zorder=3)
    ax.set_xlabel("Batch mixing (iLISI)")
    ax.set_ylabel("Cell-type agreement (ARI)")
    ax.set_ylim(0, 1.12); ax.set_xlim(0, 12.5)
    top_legend(ax, ncol=3)
    save(fig, "figS8_cell_embedding_quadrant", m)


# --------------------------------------------------------------------------- S9
def fig_S9():
    import csv
    from collections import defaultdict
    src = fig_src("final", "expansion", "fig4a_expansion_by_dataset_bar")
    fus_csv = OUTS / "fusion_paper_frozen_20260827" / "fusion_frozen_long.csv"
    if not fus_csv.is_file():
        fus_csv = REPO / "results" / "source_data" / "fusion_frozen_long.csv"
    ref = {(r["dataset"], r["ml"]): (float(r["mean"]), float(r["std"])) for r in csv.DictReader(open(src))}
    fus, shu = defaultdict(list), defaultdict(list)
    for r in csv.DictReader(open(fus_csv)):
        (fus if r["plm"] == "fusion_esm2_scgpt" else shu)[r["dataset"]].append(float(r["saber"]))
    methods = [("Cell mean", "#999999", None), ("SABER no PLM", color_for("SABER no PLM"), None),
               ("PLM-ridge", color_for("PLM-ridge"), None), ("PLM-kNN", color_for("PLM-kNN"), None),
               ("SABER", color_for("SABER"), None), ("SABER, fused prior", "#003A5D", None),
               ("Fused, gene rows permuted", "#A9CBE3", "////")]
    n = len(methods); w = 0.8 / n; offs = (np.arange(n) - (n - 1) / 2) * w
    fig, ax = plt.subplots(figsize=(FULL_W, 2.5))
    rows = []
    for i, (lab, col, hatch) in enumerate(methods):
        ys, es = [], []
        for ds in DS:
            if lab == "SABER, fused prior":
                v = np.array(fus[ds]); mu, sd = v.mean(), v.std(ddof=1)
            elif lab.startswith("Fused, gene"):
                v = np.array(shu[ds]); mu, sd = v.mean(), v.std(ddof=1)
            else:
                mu, sd = ref[(ds, lab)]
            ys.append(mu); es.append(sd)
            rows.append(dict(dataset=ds, method=lab, mean=mu, sd=sd))
        ax.bar(np.arange(len(DS)) + offs[i], ys, width=w * 0.9, yerr=es, color=col,
               edgecolor="white" if hatch is None else "#5a8fb8", linewidth=0.3, hatch=hatch,
               label=lab, error_kw=ERR)
    ax.axhline(0, color="#333333", lw=0.5)
    ax.set_xticks(range(len(DS))); ax.set_xticklabels([dataset_label(d) for d in DS])
    ax.set_ylabel("Per-protein Pearson r")
    top_legend(ax, ncol=7)
    save(fig, "figS9_benchmark_expansion_bar", pd.DataFrame(rows))


# --------------------------------------------------------------------------- S10
def fig_S10():
    df = long("final")
    rows = []
    for sp, met in [("Yeast", "mean_abs_error_yeast"), ("E. coli", "mean_abs_error_ecoli")]:
        s = agg_mean_sd(select_metric(df, "lfq", met, lfq_subset="has_missing"), ["method_label"])
        rows.append(s[s.method_label.isin(["SABER", "SABER no PLM", "Species only"])].assign(species=sp))
    summ = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    grouped_bar(ax, summ, ["SABER", "SABER no PLM", "Species only"], groups=["Yeast", "E. coli"],
                group_col="species", label_fn=lambda s: s)
    ax.set_xticklabels(["Yeast", r"$\it{E.\ coli}$"])
    ax.set_ylabel(r"Fold-change MAE ($\log_2$)")
    top_legend(ax, ncol=3)
    save(fig, "figS10_species_only_control", summ)


# --------------------------------------------------------------------------- S11
def fig_S11():
    sys.path.insert(0, str(REPO / "scripts" / "case_study"))
    import anndata as ad
    import figS11_brunner_per_substrate as b
    a = ad.read_h5ad(f"{b.BRUNNER}/prepared/Brunner2022.h5ad")
    cc = np.asarray(a.obs["cell_cycle_score"], float)
    Z = dict(np.load(f"{b.OUTDIR}/matrices.npz"))
    acs = [str(x) for x in Z["acs_keep"]]
    obs_raw = np.asarray(Z["observed_reduced"], float)
    sc = np.nanmean(obs_raw, axis=0); ss = np.nanstd(obs_raw, axis=0)
    ss = np.where(np.isfinite(ss) & (ss > 1e-8), ss, 1.0)
    zobs = (obs_raw - sc) / ss
    zcm = np.nanmean(zobs, axis=1)

    def corr(x, y):
        k = np.isfinite(x) & np.isfinite(y)
        if k.sum() < 20 or np.std(x[k]) < 1e-9 or np.std(y[k]) < 1e-9:
            return np.nan
        return float(np.corrcoef(x[k], y[k])[0, 1])

    z_of = lambda M: (np.asarray(M, float) - sc) / ss  # noqa: E731
    mats = {"observed": zobs, "SABER": np.asarray(Z["saber_seeds"], float)[0],
            "KNN": z_of(Z["knn"]), "cell-mean": z_of(Z["cell-mean"] if "cell-mean" in Z else Z["cellmean"])}
    ext = {}
    for k in Z:
        if k.startswith(("omicsgmf_seed", "pimms_dae_seed")):
            ext.setdefault(k.rsplit("_seed", 1)[0], []).append(np.asarray(Z[k], float))
    for nm, Ms in ext.items():
        mats[nm] = z_of(np.mean(Ms, axis=0))
    G = {nm: np.abs(np.array([corr(M[:, j] - zcm, cc) for j in range(M.shape[1])])) for nm, M in mats.items()}
    pp = pd.read_csv(b.LOGS / "brunner_metrics_axisholdout_perprotein.csv")
    ribo = set(pp.loc[pp.ribo, "AC"])
    rmean = {nm: float(np.mean([G[nm][i] for i, ac in enumerate(acs) if ac in ribo])) for nm in b.METHODS}
    sub = pp[pp.gold].copy()
    sub["d"] = sub.AC.map(lambda ac: G["SABER"][acs.index(ac)]) - rmean["SABER"]
    sub = sub.sort_values("d", ascending=False).reset_index(drop=True)
    ref = pd.read_csv(b.LOGS / "brunner_metrics_axisholdout_methods.csv").set_index("method")
    for nm in b.METHODS:
        dm = np.mean([G[nm][acs.index(ac)] - rmean[nm] for ac in sub.AC])
        assert abs(dm - (ref.loc[nm, "gold"] - ref.loc[nm, "ribo"])) < 1e-3, nm
    colors = dict(zip(b.METHODS, ["#5a5a5a", "#bdbdbd", color_for("KNN"),
                                  color_for("omicsGMF"), color_for("PIMMS-DAE"), color_for("SABER")]))
    fig, axes = plt.subplots(2, 4, figsize=(FULL_W, 3.3), sharey=True)
    vals = np.array([G[nm][acs.index(ac)] - rmean[nm] for ac in sub.AC for nm in b.METHODS])
    rows = []
    for ax, (_, r) in zip(axes.ravel(), sub.iterrows()):
        d = [G[nm][acs.index(r.AC)] - rmean[nm] for nm in b.METHODS]
        ax.bar(np.arange(6), d, width=0.72, color=[colors[nm] for nm in b.METHODS])
        ax.axhline(0, color="#333333", lw=0.6)
        ax.set_xticks([])
        ax.set_title(f"{r.gene} ({r.obsfrac:.0%} detected)", pad=2, fontsize=6.8)
        ax.set_ylim(vals.min() - 0.04, vals.max() + 0.04)
        rows += [dict(gene=r.gene, method=lb, delta=v) for lb, v in zip(b.LABELS, d)]
    fig.supylabel(r"$\Delta$ (substrate $-$ ribosomal)", fontsize=7, x=0.01)
    handles = [Patch(color=colors[nm], label=lb) for nm, lb in zip(b.METHODS, b.LABELS)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=6)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, "figS11_brunner_per_substrate", pd.DataFrame(rows))


# --------------------------------------------------------------------------- S12
def fig_S12():
    sub = imp_slice(long("grid"), "spearman_missing_mean")
    summ = agg_mean_sd(sub, ["missing_fraction", "method_label"])
    methods = ["SABER", "SABER no PLM", "PIMMS-DAE", "omicsGMF"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    fr = sorted(summ.missing_fraction.astype(float).unique())
    for m in methods:
        r = summ[summ.method_label == m].assign(f=lambda t: t.missing_fraction.astype(float)).set_index("f").reindex(fr)
        ax.plot(np.array(fr) * 100, r["mean"], marker="o", ms=2.8, lw=1.2, color=color_for(m), label=m,
                ls="--" if m == "SABER no PLM" else "-")
    ax.set_xticks([10, 30, 50, 70, 90]); ax.set_xlabel("Injected missingness (%)")
    ax.set_ylabel(r"Per-protein Spearman $\rho$")
    top_legend(ax, ncol=2)
    save(fig, "figS12_imputation_byfraction_spearman", summ[summ.method_label.isin(methods)])


# --------------------------------------------------------------------------- S13
def fig_S13():
    from scipy.stats import wilcoxon
    df = long("grid")
    methods = ["SABER", "SABER no PLM", "PIMMS-DAE", "omicsGMF", "KNN", "Cell mean"]
    comp = [m for m in methods if m not in ("SABER", "SABER no PLM")]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 4.0))
    out = []
    for k, (ax, (met, ylab, higher)) in enumerate(zip(axes, [
            ("spearman_missing_mean", r"Spearman $\rho$", True),
            ("mae_missing", "MAE", False)])):
        sub = imp_slice(df, met, 0.8)
        sub = sub.groupby(["dataset", "method_label", "seed"], as_index=False)["value"].mean()
        summ = agg_mean_sd(sub, ["dataset", "method_label"])
        summ = summ[summ.method_label.isin(methods)]
        offs = grouped_bar(ax, summ, methods)
        ymax = float((summ["mean"] + summ["sd"]).max())
        for xi, d in enumerate(DS):
            o = summ[(summ.dataset == d) & summ.method_label.isin(comp)]
            best = o.loc[o["mean"].idxmax() if higher else o["mean"].idxmin(), "method_label"]
            piv = sub[sub.dataset == d].pivot_table(index="seed", columns="method_label", values="value")
            pair = piv[["SABER", best]].dropna()
            if len(pair) >= 6:
                p = wilcoxon(pair["SABER"], pair[best], alternative="greater" if higher else "less").pvalue
                if p < 0.05:
                    r = summ[(summ.dataset == d) & (summ.method_label == "SABER")].iloc[0]
                    ax.text(xi + offs[0], r["mean"] + r["sd"] + 0.01 * ymax, "*", ha="center",
                            va="bottom", fontsize=9)
        ax.axhline(0, color="#bbbbbb", lw=0.5)
        ax.set_ylabel(ylab)
        letter(ax, "ab"[k], x=-0.085)
        if k == 0:
            top_legend(ax, ncol=6, y=1.06)
        out.append(summ.assign(metric=met))
    fig.tight_layout(h_pad=1.0)
    save(fig, "figS13_imputation_bar_2metric", pd.concat(out))


# --------------------------------------------------------------------------- S14
def fig_S14():
    _lg = fig_src("grid", "imputation", "observed_reconstruction_long")
    lg = pd.read_csv(_lg)
    lg = lg[(lg.method_label == "SABER") & lg.dataset.isin(DS)]
    rows = []
    for pos, col in [("observed", "observed_pearson_mean"), ("missing", "missing_pearson_mean")]:
        g = lg.groupby("dataset")[col].agg(["mean", "std"]).reset_index()
        rows.append(g.assign(position=pos))
    summ = pd.concat(rows).rename(columns={"std": "sd"})
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    w = 0.36
    for i, (pos, lab, hatch) in enumerate([("observed", "Observed entries (model input)", None),
                                           ("missing", "Injected-missing entries", "////")]):
        r = summ[summ.position == pos].set_index("dataset").reindex(DS)
        ax.bar(np.arange(len(DS)) + (i - 0.5) * w, r["mean"], width=w * 0.92, yerr=r["sd"],
               color=color_for("SABER"), edgecolor="white", linewidth=0.3, hatch=hatch, label=lab,
               error_kw=ERR)
    ax.set_xticks(range(len(DS))); ax.set_xticklabels([dataset_label(d) for d in DS])
    ax.set_ylabel("Per-protein Pearson r"); ax.set_ylim(0, 1)
    top_legend(ax, ncol=2)
    save(fig, "figS14_imputation_observed_vs_missing", summ)


FIGS = {"S1": fig_S1, "S2": fig_S2, "S3": fig_S3, "S6": fig_S6, "S7": fig_S7, "S8": fig_S8, "S9": fig_S9,
        "S10": fig_S10, "S11": fig_S11, "S12": fig_S12, "S13": fig_S13, "S14": fig_S14}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--pw-suffix", default="",
                    help="proteome-wide campaign tree suffix")
    ap.add_argument("--pw-csv", default="",
                    help="explicit source_proteome_wide_long.csv (overrides style.PW_LONG)")
    args = ap.parse_args()
    global PW_SUFFIX
    PW_SUFFIX = args.pw_suffix
    if args.pw_csv:
        style.PW_LONG = Path(args.pw_csv)
    for k in (args.only or list(FIGS)):
        print(k, flush=True)
        try:
            FIGS[k]()
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
