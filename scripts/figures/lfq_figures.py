"""SABER Fig 3 (LFQ — spike-in fold-change recovery, PXD071205) — ONE script.

Panels (5 main):
  3a  MAE bar by method (has_missing) + raw reference line (light)         [main]
  3b  species-split MAE (HUMAN/YEAST/ECOLI) x methods (no raw, no no-PLM)  [main]
  3c  PLM ablation: Δ MAE (no-PLM − SABER) per species — PLM helps signal  [main]
  3d  per-comparison (A_vs_B / A_vs_C / B_vs_C) MAE by method             [main]
  3e  species-confound control: FC MAE on signal species (YEAST/ECOLI) for
      SABER vs SABER-no-PLM vs a species-only imputer; species-only far
      worse -> PLM benefit is per-protein signal, not taxonomy             [main]

Style: same convention (no verbose titles, no parentheticals, short labels,
detail -> caption/provenance). Overwrites. Run:
  conda run -n saber python scripts/figures/lfq_figures.py [--run <run_dir>]
      --run selects the campaign under outputs/ (default: locked 10-seed run).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import style  # noqa: E402
from style import (  # noqa: E402
    get_plt, save_panel, write_source, log_panel, write_manifest, reset_manifest,
    color_for, method_label, SEEDS, load_long, select_metric, agg_mean_sd,
)

plt = get_plt()

# Per-run paths (RUN_DIR, source_data/) are resolved at runtime from style.RUN_DIR,
# which main() sets via style.configure(args.run). No hard-coded run here.

M_3A = ["SABER", "PIMMS-DAE", "omicsGMF", "KNN", "PIMMS-VAE", "Cell mean", "Protein mean", "Species only"]
M_3B = ["SABER", "PIMMS-DAE", "omicsGMF", "KNN", "PIMMS-VAE"]
M_3D = ["SABER", "PIMMS-DAE", "omicsGMF", "KNN"]
M_3E = ["SABER", "SABER no PLM", "Species only"]
M_3E_SPECIES = [("YEAST", "mean_abs_error_yeast"), ("ECOLI", "mean_abs_error_ecoli")]
# LFQ in the main campaign runs all ten seeds (the shared benchmark SEEDS is the
# 5-seed subset); per-protein details for panel 3d therefore use ten too.
LFQ_SEEDS = [42, 123, 2024, 2025, 7, 2026, 2027, 2028, 2029, 2030]
YL = "mean |log2FC error|"


def load_lfq_summary() -> pd.DataFrame:
    df = load_long()
    df = df[df.task == "lfq"].copy()
    df["method_label"] = [method_label(m, c) for m, c in zip(df.method, df.condition.fillna(""))]
    return df


def _cache_dir() -> Path:
    """Per-run cache dir for LFQ detail re-evaluation (keyed by run name so
    caches from different campaigns never mix)."""
    c = style.SRC_ROOT / "lfq" / f"_cache_{style.RUN}"
    c.mkdir(parents=True, exist_ok=True)
    return c


def _lfq_details_seed(seed: int) -> pd.DataFrame:
    from saber.evaluate.lfq import evaluate_lfq_manifest
    manifest = style.RUN_DIR / "seeds" / f"seed{seed}" / "runs" / "PXD071205_LFQ_full_saber" / "artifact_manifest.json"
    data_path = style.RUN_DIR / "seeds" / f"seed{seed}" / "prepared_inputs" / "PXD071205_LFQ" / "PXD071205_LFQ.h5ad"
    res = evaluate_lfq_manifest(manifest, _cache_dir() / f"seed{seed}", data_path=str(data_path))
    d = pd.DataFrame(res["details"])
    d = d[d["subset"] == "all"].copy()
    d["seed"] = seed
    d["method_label"] = [method_label(m, "") for m in d["method"]]
    return d[["method_label", "species", "comparison", "expected", "observed", "abs_error", "seed"]]


def get_details() -> pd.DataFrame:
    cache = _cache_dir() / "lfq_details_allseeds.csv"
    if cache.exists():
        return pd.read_csv(cache)
    manifest = (style.RUN_DIR / "seeds" / "seed42" / "runs" /
                "PXD071205_LFQ_full_saber" / "artifact_manifest.json")
    if not manifest.is_file():
        # e.g. `--run results` (shipped tables): no manifests to re-evaluate
        print("  [skip] per-comparison details need the campaign run tree (--run <run>)")
        return pd.DataFrame(columns=["method_label", "species", "comparison",
                                     "expected", "observed", "abs_error", "seed"])
    d = pd.concat([_lfq_details_seed(s) for s in LFQ_SEEDS], ignore_index=True)
    d.to_csv(cache, index=False)
    return d


# --------------------------------------------------------------------------- #
def panel_3a(df):
    sub = select_metric(df, "lfq", "mean_abs_error", lfq_subset="has_missing")
    summ = agg_mean_sd(sub, ["method_label"])
    summ = summ[summ.method_label.isin(M_3A)]
    raw = float(agg_mean_sd(select_metric(df, "lfq", "mean_abs_error", lfq_subset="has_missing"), ["method_label"])
                .query("method_label == 'Raw reference'")["mean"].iloc[0])
    order = summ.set_index("method_label").reindex(M_3A).dropna()
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.bar(range(len(order)), order["mean"], yerr=order["sd"], capsize=2,
           color=[color_for(m) for m in order.index], edgecolor="white",
           error_kw=dict(lw=0.8, ecolor="#222"))
    ax.axhline(raw, color="#bbbbbb", ls=":", lw=1.0)  # light reference line; caption explains
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order.index, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel(YL)
    name = "fig3a_lfq_mae_hasmissing"
    save_panel(fig, name, task="lfq"); plt.close(fig)
    write_source(summ.assign(raw_reference=raw), name, task="lfq")
    log_panel("3a", "main_bar", f"lfq/{name}.png", f"lfq/{name}_source.csv",
              "MAE by method (has_missing); dotted line = raw un-imputed reference",
              n_rows=len(summ), params=dict(subset="has_missing", raw_reference=raw), task="lfq")


def panel_3b(df):
    rows = []
    for sp, met in [("HUMAN", "mean_abs_error_human"), ("YEAST", "mean_abs_error_yeast"), ("ECOLI", "mean_abs_error_ecoli")]:
        s = agg_mean_sd(select_metric(df, "lfq", met, lfq_subset="all"), ["method_label"])
        s = s[s.method_label.isin(M_3B)].assign(species=sp)
        rows.append(s)
    summ = pd.concat(rows, ignore_index=True)
    sp_order = ["HUMAN", "YEAST", "ECOLI"]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    x = np.arange(len(sp_order)); n = len(M_3B); w = 0.82 / n
    offs = (np.arange(n) - (n - 1) / 2) * w
    for i, m in enumerate(M_3B):
        means = [float(summ[(summ.species == sp) & (summ.method_label == m)]["mean"].iloc[0]) for sp in sp_order]
        sds = [float(summ[(summ.species == sp) & (summ.method_label == m)]["sd"].iloc[0]) for sp in sp_order]
        ax.bar(x + offs[i], means, width=w * 0.92, yerr=sds, capsize=1.6, label=m,
               color=color_for(m), edgecolor="white", error_kw=dict(lw=0.8, ecolor="#222"))
    ax.set_xticks(x); ax.set_xticklabels(sp_order)
    ax.set_ylabel(YL)
    ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    name = "fig3b_lfq_species_split"
    save_panel(fig, name, task="lfq"); plt.close(fig)
    write_source(summ, name, task="lfq")
    log_panel("3b", "main_grouped_bar", f"lfq/{name}.png", f"lfq/{name}_source.csv",
              "MAE by species (HUMAN/YEAST/ECOLI) x method; no raw, no no-PLM",
              n_rows=len(summ), params=dict(methods=M_3B), task="lfq")


def panel_3c(df):
    """PLM ablation: Δ MAE (no-PLM − SABER) per species. PLM helps signal species
    (YEAST/ECOLI, real fold-change), barely helps HUMAN (null)."""
    species = [("HUMAN", "mean_abs_error_human"), ("YEAST", "mean_abs_error_yeast"), ("ECOLI", "mean_abs_error_ecoli")]
    rows = []
    for sp, met in species:
        s = df[(df.metric == met) & (df.lfq_subset == "all")].groupby("method_label")["value"].mean()
        sab = float(s.get("SABER", np.nan)); nop = float(s.get("SABER no PLM", np.nan))
        rows.append(dict(species=sp, SABER=sab, no_PLM=nop, delta=nop - sab))
    d = pd.DataFrame(rows)
    # 2026-09-04 polish: match sibling panel width, slimmer bars, quieter labels
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.bar(range(len(d)), d["delta"], width=0.52, color=color_for("SABER"), edgecolor="white")
    for i, row in d.iterrows():
        ax.text(i, row["delta"], f"+{row['delta']:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(d))); ax.set_xticklabels(d["species"])
    ax.set_ylabel("Δ MAE (no-PLM − SABER)")
    ax.set_ylim(0, max(d["delta"]) * 1.22)
    name = "fig3c_lfq_plm_ablation_species"
    save_panel(fig, name, task="lfq"); plt.close(fig)
    write_source(d, name, task="lfq")
    log_panel("3c", "main_ablation", f"lfq/{name}.png", f"lfq/{name}_source.csv",
              "PLM ablation: Δ MAE (no-PLM − SABER) per species; PLM helps YEAST/ECOLI (signal), "
              "barely HUMAN (null)", n_rows=len(d), task="lfq")


def panel_3d(details):
    """per-comparison MAE by method, 10-seed mean ± SD."""
    if details.empty:
        return
    g = (details[details.method_label.isin(M_3D)]
         .groupby(["method_label", "comparison", "seed"])["abs_error"].mean().reset_index())
    summ = g.groupby(["method_label", "comparison"])["abs_error"].agg(["mean", "std"]).reset_index()
    comp_order = ["A_vs_B", "A_vs_C", "B_vs_C"]
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    x = np.arange(len(comp_order))
    for m in M_3D:
        r = summ[summ.method_label == m].set_index("comparison").reindex(comp_order)
        ax.errorbar(x, r["mean"], yerr=r["std"], marker="o", ms=4.5, lw=1.7, label=m,
                    color=color_for(m), capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(comp_order)
    ax.set_xlabel("comparison"); ax.set_ylabel(YL)
    ax.legend(fontsize=6, frameon=False)
    name = "fig3d_lfq_per_comparison"
    save_panel(fig, name, task="lfq"); plt.close(fig)
    write_source(summ, name, task="lfq")
    log_panel("3d", "main_line", f"lfq/{name}.png", f"lfq/{name}_source.csv",
              "per-comparison MAE (10-seed mean±SD); A_vs_C has largest expected FC (yeast 3.32)",
              n_rows=len(summ), params=dict(methods=M_3D), task="lfq")


def panel_3e(df):
    """Species-confound control. FC MAE on
    the signal species (YEAST/ECOLI -- the ones with non-zero expected fold change)
    for SABER vs SABER-no-PLM vs a species-only imputer that fills each entry with
    its (condition x species) block mean. The species-only imputer uses taxonomy +
    condition alone, no sequence, no per-protein learning; if it matched SABER the
    PLM benefit would be species identification rather than per-protein biology."""
    rows = []
    for sp, met in M_3E_SPECIES:
        s = agg_mean_sd(select_metric(df, "lfq", met, lfq_subset="all"), ["method_label"])
        s = s[s.method_label.isin(M_3E)].assign(species=sp)
        rows.append(s)
    summ = pd.concat(rows, ignore_index=True)
    sp_order = [sp for sp, _ in M_3E_SPECIES]
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    x = np.arange(len(sp_order)); n = len(M_3E); w = 0.82 / n
    offs = (np.arange(n) - (n - 1) / 2) * w
    for i, m in enumerate(M_3E):
        means = [float(summ[(summ.species == sp) & (summ.method_label == m)]["mean"].iloc[0]) for sp in sp_order]
        sds = [float(summ[(summ.species == sp) & (summ.method_label == m)]["sd"].iloc[0]) for sp in sp_order]
        ax.bar(x + offs[i], means, width=w * 0.92, yerr=sds, capsize=1.8, label=m,
               color=color_for(m), edgecolor="white", error_kw=dict(lw=0.8, ecolor="#222"))
    ax.set_xticks(x); ax.set_xticklabels(sp_order)
    ax.set_ylabel(YL)
    ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    name = "fig3e_lfq_species_control"
    save_panel(fig, name, task="lfq"); plt.close(fig)
    write_source(summ, name, task="lfq")
    log_panel("3e", "main_grouped_bar", f"lfq/{name}.png", f"lfq/{name}_source.csv",
              "Species-confound control: FC MAE on signal species (YEAST/ECOLI) for SABER vs "
              "SABER-no-PLM vs species-only imputer; species-only far worse -> PLM benefit is "
              "per-protein signal, not taxonomy", n_rows=len(summ), params=dict(methods=M_3E), task="lfq")


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="SABER Fig 3 (LFQ) panels.")
    p.add_argument("--run", default=None,
                   help="campaign run dir under outputs/ (default: locked 10-seed run)")
    return p.parse_args()


def main():
    args = _parse_args()
    style.configure(args.run)
    reset_manifest()
    df = load_lfq_summary()
    panel_3a(df)
    panel_3b(df)
    panel_3c(df)
    details = get_details()
    panel_3d(details)
    panel_3e(df)
    print(f"\nDONE. manifest -> {write_manifest(task='lfq')}")
    print(f"figures -> {style.FIG_ROOT / 'lfq'}")


if __name__ == "__main__":
    main()
