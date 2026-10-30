#!/usr/bin/env python3
"""SABER efficiency panel (NEW supp figure) — runtime + peak GPU memory.

Story = MEMORY not speed: SABER (frozen ESM-2 prior) trains with 28-87 MB peak GPU;
wall-time is comparable to external baselines (so no 'SABER is faster' claim).
Reads the run's manifests + run_external.json.
Run: conda run -n saber python scripts/figures/efficiency_figures.py [--run <run_dir>]
"""
from __future__ import annotations

import glob, json
from pathlib import Path

import numpy as np
import pandas as pd

import style  # noqa: E402
from style import get_plt, save_panel, write_source, log_panel, write_manifest, reset_manifest  # noqa: E402

plt = get_plt()
REPO = Path(__file__).resolve().parents[2]
RUN = style.RUN_DIR  # re-bound in make_panel() via style.configure(--run)
PAP = ["Derks_2022", "Specht_2019", "Huffman_2023_melanoma_HEK", "Woo_2021"]  # Dou removed 2026-10-13


def _saber_resources():
    rows = []
    for m in glob.glob(str(RUN / "seeds/*/runs/*_miss50_mnar50_saber/artifact_manifest.json")):
        ds = Path(m).name  # not the run dir name; use parent
        run_dir = Path(m).parent.name
        dsname = next((d for d in PAP if run_dir.startswith(d + "_")), run_dir.split("_")[0])
        try:
            r = json.load(open(m)).get("resources", {})
        except Exception:
            continue
        if r:
            rows.append({"dataset": dsname, "train_sec": r.get("training_time_sec"),
                         "peak_gpu_mb": r.get("peak_gpu_memory_mb")})
    return pd.DataFrame(rows)


def _external_wall():
    rows = []
    for e in glob.glob(str(RUN / "seeds/*/external_work/*/*/*/run_external.json")):
        try:
            d = json.load(open(e))
        except Exception:
            continue
        method = Path(e).parent.name
        w = d.get("wall_time_sec")
        if w is not None:
            rows.append({"method": method, "wall_sec": w})
    return pd.DataFrame(rows)


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="SABER efficiency (runtime + GPU memory) panel.")
    p.add_argument("--run", default=None,
                   help="campaign run dir under outputs/ (default: locked 10-seed run)")
    return p.parse_args()


def make_panel():
    global RUN
    args = _parse_args()
    style.configure(args.run)
    RUN = style.RUN_DIR
    reset_manifest()
    sr = _saber_resources()
    ew = _external_wall()
    if "train_sec" not in sr.columns:
        print("  [skip] efficiency panel: run manifests (train_sec) need the campaign run tree")
        return
    saber_mean_train = sr["train_sec"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # Panel A: wall-time by method (SABER mean over datasets; externals mean over all)
    method_time = {}
    if not ew.empty:
        for meth, g in ew.groupby("method"):
            method_time[meth] = g["wall_sec"].median()
    method_time["saber"] = saber_mean_train
    order = ["saber", "pimms_dae", "pimms_vae", "omicsgmf", "knn"]
    labels = {"saber": "SABER", "pimms_dae": "PIMMS-DAE", "pimms_vae": "PIMMS-VAE",
              "omicsgmf": "omicsGMF", "knn": "KNN"}
    xs = [labels[m] for m in order if m in method_time]
    ys = [method_time[m] for m in order if m in method_time]
    colors = [style.color_for("SABER") if m == "saber" else "#9aa0a6" for m in order if m in method_time]
    axes[0].bar(range(len(xs)), ys, color=colors, edgecolor="white")
    axes[0].set_xticks(range(len(xs))); axes[0].set_xticklabels(xs, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("wall time per run (s, median)", fontsize=8)
    axes[0].set_title("a  Runtime (comparable across methods)", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel B: SABER peak GPU memory by dataset (the headline — tiny)
    ds_order = [d for d in PAP if d in set(sr.dataset)]
    gpus = [sr[sr.dataset == d]["peak_gpu_mb"].median() for d in ds_order]
    short = [d.split("_")[0] for d in ds_order]
    axes[1].bar(range(len(ds_order)), gpus, color=style.color_for("SABER"), edgecolor="white")
    axes[1].set_xticks(range(len(ds_order))); axes[1].set_xticklabels(short, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("peak GPU memory (MB)", fontsize=8)
    axes[1].set_title("b  SABER peak GPU memory (frozen ESM-2 prior)", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(gpus):
        axes[1].text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=7)

    fig.tight_layout()
    name = "figS_efficiency_runtime_memory"
    save_panel(fig, name, task="efficiency"); plt.close(fig)
    src = pd.DataFrame({
        "panel": ["a_runtime_by_method"] * len(xs) + ["b_saber_by_dataset"] * len(ds_order),
        "method": list(xs) + ["SABER"] * len(ds_order),
        "dataset": ["-"] * len(xs) + [d for d in ds_order],
        "value": list(ys) + list(gpus),
        "unit": ["wall_sec_median"] * len(xs) + ["peak_gpu_mb_median"] * len(ds_order),
    })
    write_source(src, name, task="efficiency")
    log_panel("S", "efficiency", f"efficiency/{name}.png", f"efficiency/{name}_source.csv",
              f"Runtime (median wall-time by method) + SABER peak GPU memory by dataset. "
              f"SABER train median {saber_mean_train:.1f}s.",
              task="efficiency")
    write_manifest(task="efficiency")
    print(f"[efficiency] SABER train median {saber_mean_train:.1f}s; peak GPU by dataset: {dict(zip(short, [round(g) for g in gpus]))} MB")
    print(f"[efficiency] external median wall: { {labels.get(m,m): round(t,1) for m,t in method_time.items() if m!='saber'} }")


if __name__ == "__main__":
    make_panel()
