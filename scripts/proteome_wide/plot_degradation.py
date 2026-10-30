#!/usr/bin/env python3
"""Layer (a): per-protein rho degradation curve (multi-seed, mean±sd).

Reads outputs/proteome-wide/results/<ds>_holdout<frac>_<cond>_seed<S>/expansion_metrics.csv
and plots mean per-protein Pearson (rho) vs holdout fraction, per method, with
error bars across seeds. Derks_2022 (main) and Specht_2019 (aux).

Pairing:
  saber    = real SABER, from the saber-condition run
  no-PLM   = no-PLM model, from the noplm-condition run's "saber" key
  plm_ridge / plm_knn / cell_mean = from the saber-condition run (real PLM refs)
Relocated from outputs/proteome-wide/scripts/ -> scripts/proteome_wide/.
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from saber.paths import outputs_root  # noqa: E402
RES = outputs_root() / "proteome-wide" / "results"
OUT = outputs_root() / "proteome-wide" / "figures"
DATASETS = ["Derks_2022", "Specht_2019"]
FRACTIONS = [50, 80, 90]
SEEDS = [42, 123, 2024]
TITLES = {"Derks_2022": "Derks 2022 (main)", "Specht_2019": "Specht 2019 (aux)"}

METHODS = [
    ("saber", "SABER", "#0072B2", "o", "-"),
    ("noplm", "SABER no-PLM", "#D55E00", "s", "--"),
    ("plm_knn", "PLM-kNN", "#009E73", "^", ":"),
    ("plm_ridge", "PLM-ridge", "#CC79A7", "D", ":"),
    ("cell_mean", "cell mean", "#999999", "x", ":"),
]


def read_rho(ds: str, frac: int, cond: str, method: str, seed: int):
    p = RES / f"{ds}_holdout{frac}_{cond}_seed{seed}" / "expansion_metrics.csv"
    if not p.exists():
        return None
    for r in csv.DictReader(open(p)):
        if r["method"] == method:
            v = r.get("pearson_target_mean", "")
            return float(v) if v not in ("", "nan", "None") else None
    return None


def collect(ds: str, key: str, cond: str):
    """Return (xs, means, sds) across fractions, aggregating seeds."""
    fr_vals, means, sds = [], [], []
    for fr in FRACTIONS:
        per_seed = [v for v in (read_rho(ds, fr, cond, key, s) for s in SEEDS) if v is not None]
        if not per_seed:
            continue
        fr_vals.append(fr / 100)
        means.append(statistics.mean(per_seed))
        sds.append(statistics.pstdev(per_seed) if len(per_seed) > 1 else 0.0)
    return fr_vals, means, sds


def main() -> int:
    import argparse
    global RES, OUT
    _p = argparse.ArgumentParser(); _p.add_argument("--out-suffix", default=""); suf = _p.parse_args().out_suffix
    RES = outputs_root() / f"proteome-wide{suf}" / "results"
    OUT = outputs_root() / f"proteome-wide{suf}" / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        for key, label, color, marker, ls in METHODS:
            cond = "noplm" if key == "noplm" else "saber"
            xs, ms, ss = collect(ds, "saber" if key == "noplm" else key, cond)
            if not xs:
                continue
            ax.errorbar(xs, ms, yerr=ss, ls=ls, color=color, marker=marker, label=label,
                        markersize=6, linewidth=2, capsize=3)
        ax.set_title(TITLES[ds], fontsize=11)
        ax.set_xlabel("held-out protein fraction")
        ax.set_ylabel("mean per-protein Pearson ρ")
        ax.set_xticks([0.5, 0.8, 0.9]); ax.set_xlim(0.46, 0.94)
        ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Proteome-wide expansion: per-protein ρ vs holdout fraction (n={len(SEEDS)} seeds, mean±sd)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"proteome_wide_degradation.{ext}", dpi=180)
    print(f"[plot_degradation] wrote {OUT}/proteome_wide_degradation.png|pdf")
    # tidy CSV (mean across seeds)
    rows = []
    for ds in DATASETS:
        for key, label, *_ in METHODS:
            cond = "noplm" if key == "noplm" else "saber"
            _, ms, _ = collect(ds, "saber" if key == "noplm" else key, cond)
            rows.append({"dataset": ds, "method": label,
                         "rho_50": ms[0] if len(ms) > 0 else "", "rho_80": ms[1] if len(ms) > 1 else "",
                         "rho_90": ms[2] if len(ms) > 2 else ""})
    p = OUT / "degradation_rho_table.csv"
    with p.open("w") as f:
        f.write("dataset,method,rho_50,rho_80,rho_90\n")
        for r in rows:
            f.write(f"{r['dataset']},{r['method']},{r['rho_50']},{r['rho_80']},{r['rho_90']}\n")
    print(f"[plot_degradation] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
