"""SABER Fig 4e + supp — sequence-cluster holdout (0723 plan §2).

Leakage-controlled expansion: targets are whole MMseqs2 sequence-identity clusters, so
close homologs leave training too. Panels:
  4e   SABER vs PLM-kNN/PLM-ridge vs SABER-no-PLM by holdout fraction, sequence mode
       (pooled over identity thresholds), per human dataset.
  supp random vs sequence holdout (SABER) by fraction, sequence split by identity threshold.
Reads proteome_expansion rows of source_metrics_long.csv (holdout_mode column added by
collect). Run: conda run -n saber python scripts/figures/expansion_holdout_figures.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import style  # noqa: E402
from style import (  # noqa: E402
    get_plt, save_panel, write_source, log_panel, write_manifest, reset_manifest,
    color_for, dataset_label,
)

plt = get_plt()
METRIC = "pearson_target_mean"
SEQ_DATASETS = ["Derks_2022", "Specht_2019", "Huffman_2023_melanoma_HEK"]


def _load(csv: str | None) -> pd.DataFrame:
    df = style.load_long() if not csv else pd.read_csv(csv)
    df = df[df.task == "proteome_expansion"].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["target_fraction"] = pd.to_numeric(df["target_fraction"], errors="coerce")
    df["holdout_mode"] = df.get("holdout_mode", pd.Series(["random"] * len(df))).fillna("random").astype(str)
    # identity threshold only meaningful for sequence mode; parse from run_name (_idNN_)
    df["id_thr"] = df.run_name.fillna("").map(lambda s: (int(m.group(1)) / 100.0) if (m := re.search(r"_id(\d+)_", str(s))) else np.nan)
    df = style.select_metric(df, "proteome_expansion", METRIC)
    return df.dropna(subset=["value", "target_fraction"])


def _agg(df: pd.DataFrame, **filt) -> pd.DataFrame:
    sub = df.copy()
    for k, v in filt.items():
        if k in sub:
            sub = sub[sub[k] == v]
    per = sub.groupby(["dataset", "target_fraction", "seed"])["value"].mean().reset_index()
    return per.groupby(["dataset", "target_fraction"])["value"].mean().reset_index()


def _mlabel(m, c):
    if m == "saber":
        return "SABER" if c == "saber" else "SABER no PLM"
    return {"plm_ridge": "PLM-ridge", "plm_knn": "PLM-kNN"}.get(m, m)


def panel_4e(df: pd.DataFrame):
    seq = df[df.holdout_mode == "sequence"]
    if seq.empty:
        print("[holdout] no sequence-holdout rows; skipping 4e (run the flagged sequence-holdout arm).")
        return
    seq = seq[seq.dataset.isin(SEQ_DATASETS)]
    fig, axes = plt.subplots(1, len(SEQ_DATASETS), figsize=(11.5, 3.4), sharey=True)
    methods = [("saber", "saber", "SABER", "-"), ("saber", "no_plm", "SABER no PLM", "--"),
               ("plm_knn", None, "PLM-kNN", "-"), ("plm_ridge", None, "PLM-ridge", "-")]
    for ax, ds in zip(axes, SEQ_DATASETS):
        for m, c, lab, ls in methods:
            a = _agg(seq, dataset=ds, method=m) if c is None else _agg(seq, dataset=ds, method=m, condition=c)
            if a.empty:
                continue
            ax.plot(a.target_fraction, a.value, marker="o", ms=3, ls=ls,
                    color=color_for(lab), label=lab, lw=1.3)
        ax.set_title(dataset_label(ds), fontsize=9); ax.set_xlabel("holdout fraction", fontsize=8)
        ax.axhline(0, color="#ccc", lw=0.5); ax.grid(alpha=0.25)
    axes[0].set_ylabel("target Pearson r", fontsize=8)
    axes[0].legend(fontsize=6, loc="best", frameon=False)
    fig.tight_layout()
    name = "fig4e_expansion_sequence_holdout"
    save_panel(fig, name, task="expansion"); plt.close(fig)
    write_source(seq.groupby(["dataset", "method", "condition", "target_fraction"])["value"].mean().reset_index(), name, task="expansion")
    log_panel("4e", "sequence_holdout_lines", f"expansion/{name}.png", f"expansion/{name}_source.csv",
              "sequence-cluster holdout (MMseqs2): SABER vs PLM refs vs no-PLM by fraction (pooled over id thresholds)",
              task="expansion")


def panel_random_vs_sequence(df: pd.DataFrame):
    seq = df[(df.holdout_mode == "sequence") & (df.method == "saber") & (df.condition == "saber")]
    rnd = df[(df.holdout_mode == "random") & (df.method == "saber") & (df.condition == "saber")]
    if seq.empty and rnd.empty:
        print("[holdout] no rows for random-vs-sequence supp.")
        return
    ds_list = [d for d in SEQ_DATASETS if d in set(df.dataset)]
    fig, axes = plt.subplots(1, len(ds_list), figsize=(11.5, 3.2), sharey=True)
    if len(ds_list) == 1:
        axes = [axes]
    for ax, ds in zip(axes, ds_list):
        if not rnd.empty:
            a = _agg(rnd, dataset=ds); ax.plot(a.target_fraction, a.value, marker="o", color="#888", label="random holdout", lw=1.3)
        for thr, col in sorted([(t, c) for t, c in zip([0.30, 0.50, 0.70], ["#1f77b4", "#2ca02c", "#d62728"]) if False] or [(0.30, "#1f77b4"), (0.50, "#2ca02c"), (0.70, "#d62728")]):
            sub = seq[(seq.dataset == ds) & (np.isclose(seq.id_thr.fillna(-1), thr))]
            if sub.empty:
                continue
            a = _agg(sub, dataset=ds)
            ax.plot(a.target_fraction, a.value, marker="o", color=col, label=f"sequence id{int(thr*100)}", lw=1.3)
        ax.set_title(dataset_label(ds), fontsize=9); ax.set_xlabel("holdout fraction", fontsize=8)
        ax.axhline(0, color="#ccc", lw=0.5); ax.grid(alpha=0.25)
    axes[0].set_ylabel("target Pearson r (SABER)", fontsize=8)
    axes[0].legend(fontsize=6, loc="best", frameon=False)
    fig.tight_layout()
    name = "figS_expansion_random_vs_sequence"
    save_panel(fig, name, task="expansion"); plt.close(fig)
    src = pd.concat(
        [
            rnd.assign(holdout="random", id_thr=np.nan),
            seq.assign(holdout="sequence"),
        ],
        ignore_index=True,
    )[["dataset", "holdout", "id_thr", "target_fraction", "value"]]
    write_source(src.groupby(["dataset", "holdout", "id_thr", "target_fraction"], dropna=False)["value"].mean().reset_index(), name, task="expansion")
    log_panel("S", "random_vs_sequence", f"expansion/{name}.png", f"expansion/{name}_source.csv",
              "SABER target Pearson: random vs sequence-cluster holdout (by identity threshold)",
              task="expansion")


def make_panels(csv: str | None = None):
    reset_manifest()
    df = _load(csv)
    panel_4e(df)
    panel_random_vs_sequence(df)
    write_manifest(task="expansion")
    print("[holdout] done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    if args.csv:
        # write panels into the run that owns the --csv (outputs/<run>/...), not style's default.
        style.configure(Path(args.csv).resolve().parent.parent.name)
    make_panels(args.csv)
