"""SABER Fig 5 (representation learning) — 5 panel groups.

Protein (Derks_2022 + Specht 2021; SABER vs raw ESM-2 vs omicsGMF):
  5a  GO pathway hit@10, per-dataset                                [retrieval]
  5b  STRING edge AUPRC, pooled                                     [PPI recovery]
  (pathway-ratio panel dropped from the paper 2026-08-16)
Cell (5 datasets; SABER vs scProtein vs omicsGMF):
  5c  t-SNE of cell embeddings (cell-type / batch colouring)
  5d  cell-type agreement vs batch mixing (ARI vs iLISI quadrant)

Run:
  conda run -n saber python scripts/figures/representation_figures.py [--run <run_dir>]
      --run selects the campaign under outputs/ (default: locked 10-seed run).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import style  # noqa: E402
from style import (  # noqa: E402
    get_plt, save_panel, write_source, log_panel, write_manifest, reset_manifest,
    color_for, method_label, dataset_label, DATASETS, keep_datasets, load_long, select_metric, agg_mean_sd,
)

plt = get_plt()
REPO = Path(__file__).resolve().parents[2]

PROT_DS = ["Derks_2022", "Specht_2019"]
PROT_M = ["SABER", "raw ESM-2", "omicsGMF"]
CELL_M = ["SABER", "scProtein", "omicsGMF"]


def _prot(method_raw):
    return {
        "saber": "SABER",
        "raw_plm": "raw ESM-2",
        "omicsgmf": "omicsGMF",
        "esm2_pca16": "ESM-2 PCA (16D)",
        "esm2_rp16": "Random projection (16D)",
        "expr_pca16": "Expression PCA (16D)",
    }.get(method_raw, method_raw)


def _prot_df():
    df = load_long()
    df = df[df.task == "protein_embedding"].copy()
    df["ml"] = [_prot(m) for m in df.method]
    return df[df.ml.isin(PROT_M)]


def _cell_df():
    df = keep_datasets(load_long(), include_lfq=False)
    df = df[df.task == "cell_embedding"].copy()
    df["ml"] = [method_label(m, "") for m in df.method]
    return df[df.ml.isin(CELL_M)]


def _per_dataset_bar(df, metric, datasets, methods, ylabel, name, panel, desc, higher_better=True):
    sub = df[df.metric == metric].dropna(subset=["value"])
    summ = agg_mean_sd(sub, ["dataset", "ml"])
    summ = summ[summ.ml.isin(methods)]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    x = np.arange(len(datasets)); n = len(methods); w = 0.82 / n
    offs = (np.arange(n) - (n - 1) / 2) * w
    for i, m in enumerate(methods):
        means = [float(summ[(summ.dataset == d) & (summ.ml == m)]["mean"].iloc[0])
                 if ((summ.dataset == d) & (summ.ml == m)).any() else np.nan for d in datasets]
        sds = [float(summ[(summ.dataset == d) & (summ.ml == m)]["sd"].iloc[0])
               if ((summ.dataset == d) & (summ.ml == m)).any() else 0 for d in datasets]
        ax.bar(x + offs[i], means, width=w * 0.92, yerr=sds, capsize=1.6, label=m,
               color=color_for(m), edgecolor="white", error_kw=dict(lw=0.7, ecolor="#222"))
    ax.set_xticks(x); ax.set_xticklabels([dataset_label(d) for d in datasets], fontsize=7.5)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=6, ncol=n, loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False)
    save_panel(fig, name, task="representation"); plt.close(fig)
    write_source(summ, name, task="representation")
    log_panel(panel, "main_grouped_bar", f"representation/{name}.png", f"representation/{name}_source.csv",
              desc, n_rows=len(summ), params=dict(metric=metric), task="representation")


def _pooled_bar(df, metric, methods, ylabel, name, panel, desc, lower_better=False):
    sub = df[df.metric == metric].dropna(subset=["value"])
    summ = agg_mean_sd(sub, ["ml"])
    summ = summ[summ.ml.isin(methods)].set_index("ml").reindex(methods)
    fig, ax = plt.subplots(figsize=(3.8, 3.3))
    ax.bar(range(len(methods)), summ["mean"], yerr=summ["sd"], capsize=2.5,
           color=[color_for(m) for m in methods], edgecolor="white",
           error_kw=dict(lw=0.8, ecolor="#222"))
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=7, rotation=12, ha="right")
    ax.set_ylabel(ylabel)
    save_panel(fig, name, task="representation"); plt.close(fig)
    write_source(summ.reset_index(), name, task="representation")
    log_panel(panel, "main_bar", f"representation/{name}.png", f"representation/{name}_source.csv",
              desc + (" (lower better)" if lower_better else ""), n_rows=len(summ),
              params=dict(metric=metric, lower_better=lower_better), task="representation")


# --------------------------------------------------------------------------- #
# Figure-letter map (2026-08-16, manuscript refresh): the pathway-ratio panel was
# dropped from the paper, so panels are now a=GO hit, b=STRING AUPRC, c=t-SNE,
# d=ARI/iLISI quadrant. panel_pathway_ratio() is kept below (unexported) for the
# deprecated metric; file names match the caption letters exactly.

def panel_5a(df):
    _per_dataset_bar(df, "go.pathway_hit_at_10", PROT_DS, PROT_M, "GO pathway hit@10",
                    "fig5a_protein_go_hit", "5a",
                    "GO pathway neighbor hit@10 (retrieval), per dataset; raw ESM-2 slightly higher (dimension advantage)")


def panel_5b(df):
    _pooled_bar(df, "string.edge_auprc", PROT_M, "STRING edge AUPRC",
                "fig5b_protein_string_auprc", "5b",
                "STRING PPI edge recovery AUPRC (pooled 2 datasets); raw ESM-2 slightly higher")


def _panel_pathway_ratio(df):
    """Deprecated (dropped from the paper 2026-08-16); retained for reference only."""
    _pooled_bar(df, "reactome.intra_inter_ratio", PROT_M, "intra/inter distance ratio",
                "figS_pathway_ratio_DEPRECATED", "Sx",
                "Reactome intra-/inter-pathway distance ratio (pooled); lower = tighter pathways",
                lower_better=True)


def _cat_palette(cats):
    cats = sorted(set(cats))
    cmap = plt.cm.get_cmap("tab20")
    return {c: cmap(i % 20) for i, c in enumerate(cats)}


def panel_5c_tsne():
    """5 datasets x (2 rows: cell-type / batch coloring x 3 methods) t-SNE. Author picks one."""
    from sklearn.manifold import TSNE
    methods = [("z__saber", "SABER"), ("z__scprotein", "scProtein"), ("z__omicsgmf", "omicsGMF")]
    for d in DATASETS:
        art = style.RUN_DIR / "seeds" / "seed42" / "runs" \
              / f"{d}_miss50_mnar50_saber" / "artifacts"
        if not (art / "embeddings.npz").exists():
            print(f"[5c] skip {d}: no embeddings.npz"); continue
        emb = np.load(art / "embeddings.npz", allow_pickle=True)
        lab = np.load(art / "labels.npz", allow_pickle=True)
        ct = np.asarray(lab["cell_types"]).astype(str)
        bt = np.asarray(lab["batches"]).astype(str)
        n = len(ct); perp = min(30, max(3, n // 4))
        coords = {}
        for key, ml in methods:
            if key not in emb.files:
                coords[ml] = None; continue
            z = np.asarray(emb[key], float)
            coords[ml] = TSNE(n_components=2, perplexity=perp, random_state=42,
                              init="pca", learning_rate="auto").fit_transform(z)
        ctp = _cat_palette(ct); btp = _cat_palette(bt)
        fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.4))
        for j, (key, ml) in enumerate(methods):
            xy = coords[ml]
            if xy is None:
                axes[0, j].set_visible(False); axes[1, j].set_visible(False); continue
            for c in sorted(set(ct)):
                m = ct == c
                axes[0, j].scatter(xy[m, 0], xy[m, 1], s=8, color=ctp[c], label=c, edgecolors="none")
            for b in sorted(set(bt)):
                m = bt == b
                axes[1, j].scatter(xy[m, 0], xy[m, 1], s=8, color=btp[b], label=b, edgecolors="none")
            for r in (0, 1):
                axes[r, j].set_title(ml if r == 0 else "", fontsize=8)
                axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[0, 0].set_ylabel("colored by cell type", fontsize=7)
        axes[1, 0].set_ylabel("colored by batch", fontsize=7)
        # compact legends
        h1 = [plt.Line2D([], [], marker="o", ls="", color=ctp[c], markersize=4, label=c) for c in sorted(set(ct))]
        axes[0, 2].legend(handles=h1, fontsize=5, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, title="cell type", title_fontsize=5.5)
        if len(set(bt)) <= 12:
            h2 = [plt.Line2D([], [], marker="o", ls="", color=btp[b], markersize=4, label=b) for b in sorted(set(bt))]
            axes[1, 2].legend(handles=h2, fontsize=5, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, title="batch", title_fontsize=5.5)
        fig.tight_layout(rect=[0, 0, 0.9, 1.0])
        name = f"fig5c_tsne_{d}"
        save_panel(fig, name, task="representation"); plt.close(fig)
        src_rows = []
        for ml, xy in coords.items():
            if xy is None:
                continue
            for k in range(n):
                src_rows.append({"method": ml, "cell": k, "tsne1": xy[k, 0], "tsne2": xy[k, 1],
                                 "cell_type": ct[k], "batch": bt[k]})
        write_source(pd.DataFrame(src_rows), name, task="representation")
        log_panel("5c", "main_tsne", f"representation/{name}.png", f"representation/{name}_source.csv",
                  f"t-SNE of cell embeddings, {d} (seed42); rows=cell-type/batch coloring, cols=SABER/scProtein/omicsGMF",
                  params=dict(dataset=d, seed=42), task="representation")
        print(f"[5c] {d}: {n} cells, {len(set(ct))} cell types, {len(set(bt))} batches")


def panel_5d_quadrant(df):
    """bio-conservation (ARI) vs batch-mixing (iLISI), raw axes, per-dataset points."""
    ari = agg_mean_sd(df[df.metric == "ARI_cell"], ["dataset", "ml"]).rename(columns={"mean": "ari"})
    ili = agg_mean_sd(df[df.metric == "iLISI"], ["dataset", "ml"]).rename(columns={"mean": "ili"})
    m = ari[["dataset", "ml", "ari"]].merge(ili[["dataset", "ml", "ili"]], on=["dataset", "ml"])
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for ml in CELL_M:
        s = m[m.ml == ml]
        ax.scatter(s["ili"], s["ari"], s=55, color=color_for(ml), label=ml, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(m["ili"].median(), color="#ccc", lw=0.6, ls="--")
    ax.axhline(m["ari"].median(), color="#ccc", lw=0.6, ls="--")
    ax.set_xlabel("batch mixing (iLISI)")
    ax.set_ylabel("bio conservation (ARI)")
    ax.legend(fontsize=6, loc="lower left", frameon=False)
    name = "fig5d_quadrant"
    save_panel(fig, name, task="representation"); plt.close(fig)
    write_source(m[["dataset", "ml", "ari", "ili"]], name, task="representation")
    log_panel("5d", "main_scatter", f"representation/{name}.png", f"representation/{name}_source.csv",
              "bio-conservation (ARI) vs batch-mixing (iLISI), raw axes; per-dataset points; dashed = median",
              task="representation")


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="SABER Fig 5 (representation) panels.")
    p.add_argument("--run", default=None,
                   help="campaign run dir under outputs/ (default: locked 10-seed run)")
    return p.parse_args()


def main():
    args = _parse_args()
    style.configure(args.run)
    reset_manifest()
    prot = _prot_df()
    panel_5a(prot); panel_5b(prot)
    cell = _cell_df()
    panel_5c_tsne()
    panel_5d_quadrant(cell)
    print(f"\nDONE. manifest -> {write_manifest(task='representation')}")
    print(f"figures -> {style.FIG_ROOT / 'representation'}")


if __name__ == "__main__":
    main()
