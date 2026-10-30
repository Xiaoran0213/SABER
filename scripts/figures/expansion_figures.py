"""SABER Fig 4 (proteome expansion) — 4 main (a/b/c/d) + 1 supplement.

  4a  by-dataset grouped bar (SABER vs refs x 5 datasets), seed-SD error bars
  4b  horizontal per-protein ρ violins, 5 methods, Derks_2022 (shared x -1..1)
  4c  GO enrichment heatmap (Derks holdout80, SABER; pcolormesh seamless)
  4d  GO coherence count bar (cell type x method; no-PLM/cell-mean ≈ 0)
  supp proteome-wide degradation curve

Methods (PLM-kNN k10 dropped): SABER, SABER no PLM, PLM-kNN, PLM-ridge, Cell mean.
Run:  conda run -n saber python scripts/figures/expansion_figures.py [--run <run_dir>] [--proteome-wide-dir <dir>]
      --run selects the main campaign under outputs/ (default: locked 10-seed run).
      --proteome-wide-dir selects the separate proteome-wide campaign (GO enrichment,
      degradation); default outputs/proteome-wide.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

import style  # noqa: E402
from style import (  # noqa: E402
    get_plt, save_panel, write_source, log_panel, write_manifest, reset_manifest,
    color_for, dataset_label, DATASETS, keep_datasets, load_long, load_proteome_wide, select_metric,
)

plt = get_plt()
REPO = Path(__file__).resolve().parents[2]
from saber.paths import outputs_root  # noqa: E402
PW = outputs_root() / "proteome-wide"  # default; overridable via --proteome-wide-dir

# Per-run main-campaign paths (RUN_DIR, source_data/) are resolved at runtime from
# style.RUN_DIR, which main() sets via style.configure(args.run). No hard-coded run here.

M_MAIN = ["SABER", "SABER no PLM", "PLM-kNN", "PLM-ridge", "Cell mean"]
YL = "pearson r"


def _ref_label(m, c):
    if m == "saber":
        return "SABER" if c == "saber" else "SABER no PLM"
    return {"plm_ridge": "PLM-ridge", "plm_knn": "PLM-kNN",
            "cell_mean": "Cell mean", "protein_mean": "Protein mean"}.get(m, m)


# --------------------------------------------------------------------------- #
def panel_4a():
    df = keep_datasets(load_long(), include_lfq=False)
    ex = df[df.task == "proteome_expansion"].copy()
    ex["ml"] = [_ref_label(m, c) for m, c in zip(ex.method, ex.condition.fillna(""))]
    p = select_metric(ex, "proteome_expansion", "pearson_target_mean").dropna(subset=["value"])
    p = p[p.ml.isin(M_MAIN)]
    per_seed = p.groupby(["dataset", "ml", "seed"])["value"].mean().reset_index()
    summ = per_seed.groupby(["dataset", "ml"])["value"].agg(["mean", "std"]).reset_index()
    summ["std"] = summ["std"].fillna(0)
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    x = np.arange(len(DATASETS)); n = len(M_MAIN); w = 0.82 / n
    offs = (np.arange(n) - (n - 1) / 2) * w
    for i, m in enumerate(M_MAIN):
        means = [float(summ[(summ.dataset == d) & (summ.ml == m)]["mean"].iloc[0])
                 if ((summ.dataset == d) & (summ.ml == m)).any() else np.nan for d in DATASETS]
        sds = [float(summ[(summ.dataset == d) & (summ.ml == m)]["std"].iloc[0])
               if ((summ.dataset == d) & (summ.ml == m)).any() else 0 for d in DATASETS]
        ax.bar(x + offs[i], means, width=w * 0.92, yerr=sds, capsize=1.4, label=m,
               color=color_for(m), edgecolor="white", error_kw=dict(lw=0.7, ecolor="#222"))
    ax.set_xticks(x); ax.set_xticklabels([dataset_label(d) for d in DATASETS], rotation=15, ha="right", fontsize=7)
    ax.set_ylabel(YL); ax.axhline(0, color="#bbb", lw=0.5)
    ax.legend(fontsize=6, ncol=n, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    name = "fig4a_expansion_by_dataset_bar"
    save_panel(fig, name, task="expansion"); plt.close(fig)
    write_source(summ, name, task="expansion")
    log_panel("4a", "main_grouped_bar", f"expansion/{name}.png", f"expansion/{name}_source.csv",
              "target Pearson by dataset (SABER vs refs); error = SD over 5 seeds",
              n_rows=len(summ), params=dict(methods=M_MAIN), task="expansion")


def panel_4b():
    d = "Derks_2022"
    fs_s = glob.glob(str(style.RUN_DIR / "source_data" / "seed*" / f"{d}_holdout5_saber" / "proteome_expansion" / "source_expansion_per_protein.csv"))
    fs_n = glob.glob(str(style.RUN_DIR / "source_data" / "seed*" / f"{d}_holdout5_noplm" / "proteome_expansion" / "source_expansion_per_protein.csv"))
    pp_s = pd.concat([pd.read_csv(f) for f in fs_s]) if fs_s else pd.DataFrame()
    pp_n = pd.concat([pd.read_csv(f) for f in fs_n]) if fs_n else pd.DataFrame()
    if pp_s.empty:
        print("  [skip] panel 4b: per-protein CSVs need the campaign run tree (--run <run>)")
        return
    per_method = {}
    for m_raw, lbl in [("saber", "SABER"), ("plm_knn", "PLM-kNN"), ("plm_ridge", "PLM-ridge"), ("cell_mean", "Cell mean")]:
        sub = pp_s[(pp_s.method == m_raw) & pp_s.pearson_target.notna()]
        per_method[lbl] = sub.groupby("protein_id")["pearson_target"].mean().values
    if not pp_n.empty:
        sub = pp_n[(pp_n.method == "saber") & pp_n.pearson_target.notna()]
        per_method["SABER no PLM"] = sub.groupby("protein_id")["pearson_target"].mean().values
    data = [per_method.get(m, np.array([])) for m in M_MAIN]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    pos = np.arange(len(M_MAIN))
    parts = ax.violinplot(data, positions=pos, vert=False, showmedians=True, showextrema=False)
    for body, m in zip(parts["bodies"], M_MAIN):
        body.set_facecolor(color_for(m)); body.set_alpha(0.6); body.set_edgecolor(color_for(m))
    parts["cmedians"].set_color("white"); parts["cmedians"].set_lw(1.0)
    ax.set_xlim(-1, 1)
    ax.set_yticks(pos); ax.set_yticklabels(M_MAIN, fontsize=7)
    ax.set_xlabel("per-protein ρ")
    name = "fig4b_expansion_violin_derks"
    save_panel(fig, name, task="expansion"); plt.close(fig)
    src_df = pd.DataFrame({m: pd.Series(v) for m, v in per_method.items()})
    src_df = src_df.melt(var_name="method", value_name="pearson_target").dropna()
    write_source(src_df, name, task="expansion")
    log_panel("4b", "main_violin", f"expansion/{name}.png", f"expansion/{name}_source.csv",
              "per-protein ρ distribution (Derks_2022 holdout5, 5 methods, per-protein mean over 5 seed)",
              task="expansion")


def panel_supp_degradation():
    pw = load_proteome_wide()
    deg = pw[pw.task == "proteome_wide_degradation"]
    rows = []
    for d in ["Derks_2022", "Specht_2019"]:
        for frac in [0.5, 0.8, 0.9]:
            for cond, lab in [("saber", "SABER"), ("noplm", "SABER no PLM")]:
                v = deg[(deg.dataset == d) & (deg.target_fraction == frac) & (deg.condition == cond)
                        & (deg.method == "saber")]["value"]
                rows.append(dict(dataset=d, fraction=frac, method=lab,
                                 mean=float(v.mean()) if len(v) else np.nan,
                                 sd=float(v.std()) if len(v) else 0, n=len(v)))
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ds_label = {"Derks_2022": "Derks 2022", "Specht_2019": "Specht 2019"}
    for d, mk in [("Derks_2022", "o"), ("Specht_2019", "s")]:
        for lab, col in [("SABER", color_for("SABER")), ("SABER no PLM", color_for("SABER no PLM"))]:
            sub = df[(df.dataset == d) & (df.method == lab)].sort_values("fraction")
            ls = "-" if lab == "SABER" else "--"
            ax.errorbar(sub["fraction"] * 100, sub["mean"], yerr=sub["sd"], marker=mk, ms=4.5,
                        lw=1.7, color=col, ls=ls, capsize=2,
                        label=f"{lab} — {ds_label[d]}")
    ax.set_xlabel("% proteins held out"); ax.set_ylabel(YL)
    # one legend entry per line: before, Derks vs Specht differed only by a
    # marker invisible at this size (found 2026-09-04 visual check); placed
    # below the axes because every in-axes corner sits on a line
    ax.legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              frameon=False)
    name = "figS_expansion_proteome_wide_degradation"
    save_panel(fig, name, task="expansion"); plt.close(fig)
    write_source(df, name, task="expansion")
    log_panel("4c-supp", "supplement_line", f"expansion/{name}.png", f"expansion/{name}_source.csv",
              "SUPPLEMENT: proteome-wide degradation (3-seed)", n_rows=len(df), task="expansion")


def _go_saber_terms(dataset, frac, cell_types):
    out = {ct: {} for ct in cell_types}
    if not (PW / "results" / "go").is_dir():
        print("  [skip] GO term panels need the proteome-wide campaign results (--proteome-wide-dir)")
        return out
    for ct in cell_types:
        fs = glob.glob(str(PW / "results" / "go" / f"{dataset}_holdout{int(frac*100)}_seed*"
                           / "ora_details" / f"saber__{ct}__go.csv"))
        if not fs:
            fs = glob.glob(str(PW / "results" / "go" / f"{dataset}_holdout{int(frac*100)}"
                               / "ora_details" / f"saber__{ct}__go.csv"))
        frames = []
        for f in fs:
            d = pd.read_csv(f); d["neglog"] = -np.log10(d.fdr.clip(lower=1e-30))
            frames.append(d[["term_name", "neglog"]])
        if frames:
            out[ct] = pd.concat(frames).groupby("term_name")["neglog"].mean().to_dict()
    return out


def panel_4c():
    """GO BP enrichment: top-5 terms per cell type, horizontal bar colored by cell type.
    (Each cell type enriches for distinct, cell-type-appropriate biology.)"""
    from matplotlib.patches import Patch
    cts = ["Melanoma", "PDAC", "U-937"]
    td = _go_saber_terms("Derks_2022", 0.8, cts)
    ct_color = {"Melanoma": "#0072B2", "PDAC": "#D55E00", "U-937": "#009E73"}
    rows = []
    for ct in cts:
        for t, v in sorted(td.get(ct, {}).items(), key=lambda kv: -kv[1])[:5]:
            rows.append(dict(cell_type=ct, term=t, neglog=min(v, 8.0)))
    d = pd.DataFrame(rows)
    if d.empty:
        print("  [skip] panel 4c: no GO enrichment details (proteome-wide results absent)")
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d["neglog"], color=[ct_color[c] for c in d["cell_type"]], edgecolor="white", height=0.74)
    ax.set_yticks(y); ax.set_yticklabels([f"{r.term}  ({r.cell_type})" for r in d.itertuples()], fontsize=6.5)
    ax.set_xlabel("pathway enrichment (−log10 FDR)")
    ax.axvline(1.3, color="#999", ls=":", lw=0.7)
    ax.legend(handles=[Patch(color=ct_color[c], label=c) for c in cts],
              fontsize=6, loc="lower right", frameon=False)
    name = "fig4c_expansion_go_enrichment_bar"
    save_panel(fig, name, task="expansion"); plt.close(fig)
    write_source(d, name, task="expansion")
    log_panel("4c", "main_bar", f"expansion/{name}.png", f"expansion/{name}_source.csv",
              "GO BP enrichment (Derks holdout80, SABER, 3-seed); top-5 terms/cell type, colored; dotted=FDR0.05; "
              "PDAC→keratin, Melanoma→ECM, U-937→RNA", task="expansion")


def panel_4d():
    """GO coherence count bar (cell type x method, Derks_2022 holdout80).

    Reads the campaign's aggregated table directly so the bars are the true
    3-seed mean +- s.d.: the flattened proteome-wide long table keeps only the
    mean (seed column empty), so the previous .iloc[0] version plotted one seed
    while the caption claimed a 3-seed mean (found 2026-09-04).
    """
    agg = PW / "results" / "go" / "coherence_summary_Derks_2022_holdout80_agg.csv"
    if not agg.is_file():
        print("  [skip] GO coherence panel needs the proteome-wide campaign results (--proteome-wide-dir)")
        return
    t = pd.read_csv(agg)
    key = {"SABER": "saber", "SABER no PLM": "noplm", "PLM-kNN": "plm_knn",
           "PLM-ridge": "plm_ridge", "Cell mean": "cell_mean"}
    cts = ["Melanoma", "PDAC", "U-937"]
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    x = np.arange(len(cts)); n = len(M_MAIN); w = 0.82 / n
    offs = (np.arange(n) - (n - 1) / 2) * w
    rows = []
    for i, m in enumerate(M_MAIN):
        sub = t[t.method == key[m]].set_index("cell_type")
        means = [float(sub.loc[c, "n_sig_GO_BP_mean"]) if c in sub.index else 0.0 for c in cts]
        sds = [float(sub.loc[c, "n_sig_GO_BP_sd"]) if c in sub.index else 0.0 for c in cts]
        # zero-height bars keep their coloured outline so 'ran, recovered 0'
        # stays visible (edgecolor='white' made them vanish entirely)
        ax.bar(x + offs[i], means, width=w * 0.92, label=m, yerr=sds,
               color=color_for(m), edgecolor=color_for(m), linewidth=0.7,
               error_kw=dict(lw=0.8, capsize=1.5))
        for c, mn, sd in zip(cts, means, sds):
            rows.append(dict(method=m, cell_type=c, n_sig_GO_BP_mean=mn, n_sig_GO_BP_sd=sd))
    ax.set_xticks(x); ax.set_xticklabels(cts)
    ax.set_ylabel("# significant GO BP terms")
    ax.legend(fontsize=6, ncol=n, loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False)
    name = "fig4d_expansion_go_coherence_count"
    save_panel(fig, name, task="expansion"); plt.close(fig)
    write_source(pd.DataFrame(rows), name, task="expansion")
    log_panel("4d", "main_grouped_bar", f"expansion/{name}.png", f"expansion/{name}_source.csv",
              "# significant GO BP terms by cell type x method (Derks holdout80); "
              "3-seed mean+-sd from coherence_summary agg; no-PLM/Cell mean ≈ 0 → PLM-driven coherence",
              task="expansion")


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="SABER Fig 4 (expansion) panels.")
    p.add_argument("--run", default=None,
                   help="campaign run dir under outputs/ (default: locked 10-seed run)")
    p.add_argument("--proteome-wide-dir", default=str(REPO / "outputs" / "proteome-wide"),
                   help="proteome-wide campaign dir (GO enrichment + degradation); "
                        "separate from --run")
    return p.parse_args()


def main():
    global PW
    args = _parse_args()
    style.configure(args.run)
    PW = Path(args.proteome_wide_dir)
    reset_manifest()
    panel_4a()
    panel_4b()
    panel_4c()
    panel_4d()
    panel_supp_degradation()
    print(f"\nDONE. manifest -> {write_manifest(task='expansion')}")
    print(f"figures -> {style.FIG_ROOT / 'expansion'}")


if __name__ == "__main__":
    main()
