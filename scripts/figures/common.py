"""Shared loaders, labels, palette, and stats for the figure scripts.

Single place where "which campaign run are we plotting" and "where do figures
go" are decided:

- inputs resolve through ``saber.paths`` (``SABER_OUTPUTS_ROOT`` /
  ``.saber_env``); ``configure(run)`` accepts a run *name* under the outputs
  root or an explicit path (``--run results`` reads the shipped source tables
  in ``results/source_data``);
- outputs (rendered panels + per-panel source CSVs) go under
  ``SABER_FIGURES_ROOT/benchmark`` (default ``./figures``), separate from the
  campaign output tree.

Method labels/palette extend ``saber/plots.py`` for the reference baselines
(PLM-ridge / PLM-kNN / raw ESM-2) used by the expansion panels.
"""
from __future__ import annotations

import json
import re as _re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# scripts/figures/common.py -> repo root; lets the figure scripts run without
# `pip install -e .` (no effect when the package is installed).
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from saber.paths import campaign_run_dir, figures_root  # noqa: E402

# Default campaign run for all figure scripts; override per invocation with
# --run (configure() below, called by each script's main()).
DEFAULT_RUN = "paper_benchmark_20260801_final_10seed"
RUN = DEFAULT_RUN
OUT_ROOT = campaign_run_dir(RUN)
SOURCE_LONG = OUT_ROOT / "source_data" / "source_metrics_long.csv"
SOURCE_BENCH = figures_root() / "benchmark" / "source_data"
FIGURES = figures_root() / "benchmark"
SEEDS = [42, 123, 2024, 2025, 7]


def configure(run: str | None = None) -> str:
    """Set the campaign run that all loaders read from.

    Rebinds the module-level OUT_ROOT/SOURCE_LONG/SOURCE_BENCH/FIGURES so that
    load_long(), save(), and write_csv() — which look these names up at call time —
    all agree on the same run. Called once at the start of a figure script's
    main() (typically via style.configure(run), which forwards here and also
    re-derives the style-layer output dirs).

    Args:
        run: directory name under ``$SABER_OUTPUTS_ROOT`` (e.g.
             ``paper_benchmark_20260801_final_10seed``) or an explicit path to a
             run directory — ``--run results`` points at the tables shipped in
             the repository. None falls back to DEFAULT_RUN.
    Returns the resolved run name."""
    global RUN, OUT_ROOT, SOURCE_LONG, SOURCE_BENCH, FIGURES
    RUN = run or DEFAULT_RUN
    OUT_ROOT = campaign_run_dir(RUN)
    SOURCE_LONG = OUT_ROOT / "source_data" / "source_metrics_long.csv"
    SOURCE_BENCH = figures_root() / "benchmark" / "source_data"
    FIGURES = figures_root() / "benchmark"
    return RUN

# ---------------------------------------------------------------------------
# Method -> display label (mirrors saber/plots.py for SABER variants; adds the
# new reference / baseline methods that the core METHOD_LABELS table lacks).
# ---------------------------------------------------------------------------
def method_label(method: str, condition: str = "") -> str:
    method = str(method)
    cond = str(condition).lower()
    if method == "saber":
        if "noplm" in cond or "no_plm" in cond:
            return "SABER no PLM"
        if "now" in cond or "no_w" in cond:
            return "SABER no W"
        return "SABER"
    return {
        "plm_ridge": "PLM-ridge",
        "plm_knn": "PLM-kNN",
        "plm_knn_k10": "PLM-kNN (k=10)",
        "cell_mean": "Cell mean",
        "protein_mean": "Protein mean",
        "pimms_dae": "PIMMS-DAE",
        "pimms_vae": "PIMMS-VAE",
        "omicsgmf": "omicsGMF",
        "knn": "KNN",
        "scprotein": "scProtein",
        "raw_plm": "raw ESM-2",
        "raw_true": "Raw reference",
        "species_mean": "Species only",
    }.get(method, method.replace("_", " "))


# Colorblind-friendly palette (Okabe-Ito backbone + extensions).
PALETTE2 = {
    "SABER": "#0072B2",
    "SABER no PLM": "#56B4E9",
    "SABER no W": "#9ad2f0",
    "PLM-ridge": "#E69F00",
    "PLM-kNN": "#D55E00",
    "PLM-kNN (k=10)": "#B36000",
    "Cell mean": "#999999",
    "Protein mean": "#c8c8c8",
    "PIMMS-DAE": "#009E73",
    "PIMMS-VAE": "#005A41",
    "omicsGMF": "#CC79A7",
    "KNN": "#8C6D8F",
    "scProtein": "#7E3F8F",
    "raw ESM-2": "#5a5a5a",
    "Raw reference": "#4B5563",
    "raw PCA": "#b0b0b0",
    "Species only": "#7A7A7A",  # neutral grey: a baseline, not an accent
    "ESM-2 PCA (16D)": "#882255",
    "Random projection (16D)": "#CC6677",
    "Expression PCA (16D)": "#999933",
}
GRAY = "#4B5563"


def color_for(label: object) -> str:
    return PALETTE2.get(str(label), GRAY)


# Method display order for axes/legends.
METHOD_ORDER2 = [
    "SABER", "SABER no PLM", "SABER no W",
    "PLM-ridge", "PLM-kNN", "PLM-kNN (k=10)",
    "PIMMS-DAE", "PIMMS-VAE", "omicsGMF", "KNN",
    "scProtein", "raw ESM-2", "Raw reference",
    "Cell mean", "Protein mean", "Species only", "raw PCA",
]

# Dou_2019 is not part of the manuscript datasets.
DATASET_ORDER2 = [
    "Derks_2022", "Derks_2023", "Specht_2019",
    "Huffman_2023_melanoma_HEK", "Woo_2021",
    "PXD071205_LFQ",
]

DATASET_LABELS2 = {
    "Derks_2022": "Derks 2022",
    "Derks_2023": "Derks 2023",
    "Specht_2019": "Specht 2021",
    "Huffman_2023_melanoma_HEK": "Huffman 2023",
    "Woo_2021": "Woo 2021",
    "PXD071205_LFQ": "PXD071205 LFQ",
}


def dataset_label(name: object) -> str:
    return DATASET_LABELS2.get(str(name), str(name).replace("_", " "))


def dataset_sort(name: object) -> tuple[int, str]:
    try:
        return (DATASET_ORDER2.index(str(name)), str(name))
    except ValueError:
        return (len(DATASET_ORDER2), str(name))


def method_sort(label: object) -> tuple[int, str]:
    try:
        return (METHOD_ORDER2.index(str(label)), str(label))
    except ValueError:
        return (len(METHOD_ORDER2), str(label))


# Metric polarity: True = higher is better.
HIGHER_BETTER = {
    "pearson_missing_mean": True, "pearson_missing_all": True, "spearman_missing_mean": True,
    "pearson_target_mean": True, "pearson_target_all": True, "spearman_target_mean": True,
    "pearson_log2fc": True, "spearman_log2fc": True,
    "ARI_cell": True, "NMI_cell": True, "ARI_batch": True, "NMI_batch": True,
    "ASW_cell": True, "PS_cell": True, "iLISI": True, "graph_connectivity": True,
    "batch_mixing_score": True,
    "effective_rank": None,  # structural, no direction
    "string.edge_auroc": True, "string.edge_auprc": True,
    "reactome.layer3_mean_gap": True, "go.global_silhouette": True,
}
LOWER_BETTER_METRICS = {
    "mae_missing", "mse_missing", "rmse_missing",
    "mae_target", "mse_target", "rmse_target",
    "mean_abs_error", "mean_abs_error_yeast_ecoli", "rmse_log2fc", "mean_error",
    "kBET_reject_rate", "cLISI",
    "reactome.intra_inter_ratio", "go.intra_inter_ratio",
}


def higher_is_better(metric: str) -> bool | None:
    if metric in LOWER_BETTER_METRICS:
        return False
    return HIGHER_BETTER.get(metric, None)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_long() -> pd.DataFrame:
    if not SOURCE_LONG.is_file():
        raise SystemExit(
            f"Source table not found: {SOURCE_LONG}\n"
            "Pass a campaign run with --run <name-or-path> (names resolve under "
            "SABER_OUTPUTS_ROOT; `--run results` uses the tables shipped in the "
            "repository), or check SABER_OUTPUTS_ROOT / .saber_env."
        )
    df = pd.read_csv(SOURCE_LONG)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["dataset"] = df["dataset"].astype(str)
    df["method"] = df["method"].astype(str)
    df["condition"] = df["condition"].fillna("").astype(str)
    df["method_label"] = [method_label(m, c) for m, c in zip(df["method"], df["condition"])]
    # Recover the TRUE seed from the artifact_path. The LFQ rows are all
    # mislabeled seed=42 in the CSV column; the real seed lives in the path
    # (.../seeds/seed{N}/runs/...). seed_true is correct for every task and is
    # what we pair/aggregate on. (Verified: matches `seed` for non-LFQ tasks.)
    df["seed_true"] = df["artifact_path"].map(
        lambda p: int(m.group(1)) if (m := _re.search(r"seed(\d+)", str(p))) else None
    )
    return df.dropna(subset=["value"])


def select_metric(df: pd.DataFrame, task: str, metric: str, **filters) -> pd.DataFrame:
    sub = df[(df["task"].astype(str) == task) & (df["metric"].astype(str) == metric)].copy()
    for key, value in filters.items():
        if key in sub.columns:
            sub = sub[sub[key].astype(str) == str(value)]
    return sub


def agg_mean_sd(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Mean ± SD across seeds (and any other non-listed dims). Returns columns
    [..group_cols, 'mean', 'sd', 'count']."""
    g = df.groupby(group_cols, dropna=False)["value"].agg(["mean", "std", "count"]).reset_index()
    g = g.rename(columns={"std": "sd"})
    g["sd"] = g["sd"].fillna(0.0)
    return g


# ---------------------------------------------------------------------------
# Stats: paired Wilcoxon + Holm
# ---------------------------------------------------------------------------
def _holm(pvals: list[float]) -> list[float]:
    pvals = list(pvals)
    m = sum(1 for p in pvals if p is not None and not (isinstance(p, float) and np.isnan(p)))
    order = sorted(range(len(pvals)), key=lambda i: (pvals[i] if pvals[i] is not None and not np.isnan(pvals[i]) else np.inf))
    adj = [np.nan] * len(pvals)
    running = 0.0
    for rank, idx in enumerate(order):
        p = pvals[idx]
        if p is None or np.isnan(p):
            adj[idx] = np.nan
            continue
        cand = min(1.0, p * (m - rank))
        running = max(running, cand)
        adj[idx] = running
    return adj


def wilcoxon_vs_baseline(
    df: pd.DataFrame,
    task: str,
    metric: str,
    ref_spec: tuple[str, str],
    baseline_specs: list[tuple[str, str]],
    pair_cols: list[str],
    extra_filter: dict | None = None,
) -> pd.DataFrame:
    """Paired Wilcoxon of (ref) vs each baseline across pair_cols (seed,dataset,...).

    ref_spec / baseline_specs are (method, condition) tuples. condition "" = any.
    Returns one row per baseline with W, raw p, Holm-adj p, n_pairs, median_delta,
    direction ('SABER better' / 'baseline better' / 'tie'), higher_is_better.
    """
    sub = df[(df["task"].astype(str) == task) & (df["metric"].astype(str) == metric)].copy()
    if extra_filter:
        for key, value in extra_filter.items():
            if key in sub.columns:
                sub = sub[sub[key].astype(str) == str(value)]

    def pick(spec: tuple[str, str]) -> pd.DataFrame:
        m, c = spec
        s = sub[sub["method"].astype(str) == m]
        if c:
            s = s[s["condition"].astype(str) == c]
        return s

    ref = pick(ref_spec)
    rows = []
    raw_p = []
    for spec in baseline_specs:
        base = pick(spec)
        if ref.empty or base.empty:
            rows.append(dict(baseline=method_label(spec[0], spec[1]), n_pairs=0))
            raw_p.append(np.nan)
            continue
        merged = ref.merge(
            base[list(pair_cols) + ["value"]],
            on=pair_cols, suffixes=("_ref", "_base"),
        )
        merged = merged.dropna(subset=["value_ref", "value_base"])
        x = merged["value_ref"].to_numpy(dtype=float)
        y = merged["value_base"].to_numpy(dtype=float)
        n = len(x)
        delta = x - y
        med_delta = float(np.median(delta)) if n else np.nan
        p = np.nan
        W = np.nan
        reason = ""
        if n >= 2 and np.any(delta != 0):
            try:
                from scipy.stats import wilcoxon
                res = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
                W = float(res.statistic)
                p = float(res.pvalue)
            except Exception as exc:  # all-zero diffs, too few nonzero, etc.
                reason = f"wilcoxon_failed: {exc}"
        elif n < 2:
            reason = "n<2"
        else:
            reason = "all diffs zero"
        hb = higher_is_better(metric)
        # Direction from the median delta; if the median is exactly 0 (common with
        # ceiling-clamped metrics like ARI/graph_connectivity), fall back to the
        # mean so a significant test is not mislabeled "tie".
        sign_delta = med_delta
        if sign_delta == 0 and n > 0:
            sign_delta = float(np.mean(delta))
        if hb is None or n == 0:
            direction = "n/a"
        elif sign_delta == 0:
            direction = "tie"
        else:
            # delta = SABER - baseline. higher-better: SABER wins if delta>0.
            saber_better = (sign_delta > 0) if hb else (sign_delta < 0)
            direction = "SABER better" if saber_better else "baseline better"
        rows.append(dict(
            task=task, metric=metric,
            baseline=method_label(spec[0], spec[1]),
            ref=method_label(ref_spec[0], ref_spec[1]),
            n_pairs=n, W_stat=W, p_raw=p, median_delta_saber_minus_baseline=med_delta,
            higher_is_better=("" if hb is None else ("yes" if hb else "no")),
            direction=direction, note=reason,
        ))
        raw_p.append(p)
    adj = _holm(raw_p)
    for r, a in zip(rows, adj):
        r["p_holm"] = a
    cols = ["task", "metric", "ref", "baseline", "n_pairs", "W_stat", "p_raw", "p_holm",
            "median_delta_saber_minus_baseline", "higher_is_better", "direction", "note"]
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])


# ---------------------------------------------------------------------------
# Plotting helpers (import style utils read-only from saber.plots)
# ---------------------------------------------------------------------------
def get_plt():
    from saber.plots import apply_paper_style, require_matplotlib
    require_matplotlib()
    apply_paper_style()
    import matplotlib.pyplot as plt
    return plt


def save(fig, name: str, formats=("png", "pdf")) -> list[str]:
    from saber.plots import save_figure
    FIGURES.mkdir(parents=True, exist_ok=True)
    return save_figure(fig, FIGURES / name, formats=formats)


def grouped_bar_errbar(
    ax,
    summary: pd.DataFrame,
    x_col: str,
    method_col: str = "method_label",
    x_order: list[str] | None = None,
    method_order: list[str] | None = None,
    ylabel: str = "",
    x_label_func=dataset_label,
    rotate: int = 25,
):
    """Grouped bar plot with mean±SD error bars from a `summary` frame that has
    columns [x_col, method_col, 'mean', 'sd']. Returns the list of method labels
    actually plotted (in order)."""
    if x_order is None:
        x_order = sorted(summary[x_col].dropna().unique(), key=dataset_sort)
    if method_order is None:
        present = summary[method_col].dropna().unique()
        method_order = sorted(present, key=method_sort)
    x = np.arange(len(x_order))
    n = len(method_order)
    bar_w = 0.82 / max(n, 1)
    offsets = (np.arange(n) - (n - 1) / 2) * bar_w
    for i, m in enumerate(method_order):
        means, sds = [], []
        for xv in x_order:
            row = summary[(summary[x_col] == xv) & (summary[method_col] == m)]
            if not row.empty:
                means.append(float(row["mean"].iloc[0]))
                sds.append(float(row["sd"].iloc[0]))
            else:
                means.append(np.nan)
                sds.append(0.0)
        ax.bar(x + offsets[i], means, width=bar_w * 0.92, yerr=sds, capsize=2.2,
               label=m, color=color_for(m), edgecolor="white",
               error_kw=dict(linewidth=0.9, ecolor="#222222"))
    ax.set_xticks(x)
    ax.set_xticklabels([x_label_func(v) for v in x_order], rotation=rotate, ha="right")
    if ylabel:
        ax.set_ylabel(ylabel)
    return method_order



def write_csv(df: pd.DataFrame, name: str) -> str:
    SOURCE_BENCH.mkdir(parents=True, exist_ok=True)
    path = SOURCE_BENCH / name
    df.to_csv(path, index=False)
    return str(path)


def write_md(text: str, name: str) -> str:
    SOURCE_BENCH.mkdir(parents=True, exist_ok=True)
    path = SOURCE_BENCH / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def fmt_pm(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f}±{sd:.{digits}f}"


def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.3f}"


def stars(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"
