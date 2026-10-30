from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .plots import (
    DATASET_ORDER,
    MARKERS,
    METHOD_ORDER,
    apply_paper_style,
    color_for,
    dataset_label,
    method_label,
    panel_label,
    place_legend,
    require_columns,
    require_matplotlib,
    save_figure,
)


DEFAULT_FORMATS = ("png", "pdf")


@dataclass(frozen=True)
class PaperFigureInputs:
    source_data: Path
    outdir: Path
    formats: tuple[str, ...] = DEFAULT_FORMATS


def _read_source_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    require_columns(["task", "dataset", "method", "metric", "value"], df.columns, "source data")
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["dataset"] = df["dataset"].astype(str)
    df["method"] = df["method"].astype(str)
    if "condition" not in df.columns:
        df["condition"] = ""
    df["method_label"] = [method_label(m, c) for m, c in zip(df["method"], df["condition"])]
    return df.dropna(subset=["value"])


def _dataset_sort_key(name: object) -> tuple[int, str]:
    text = str(name)
    try:
        return (DATASET_ORDER.index(text), text)
    except ValueError:
        return (len(DATASET_ORDER), text)


def _method_sort_key(name: object) -> tuple[int, str]:
    text = str(name)
    try:
        return (METHOD_ORDER.index(text), text)
    except ValueError:
        return (len(METHOD_ORDER), text)


def _bar_offsets(n: int, width: float = 0.78) -> tuple[np.ndarray, float]:
    if n <= 0:
        return np.asarray([]), width
    bar_width = width / n
    offsets = (np.arange(n) - (n - 1) / 2) * bar_width
    return offsets, bar_width * 0.92


def _write_source(df: pd.DataFrame, outdir: Path, name: str) -> str:
    source_dir = outdir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / name
    df.to_csv(path, index=False)
    return str(path.resolve())


def _save(fig, outdir: Path, filename: str, formats: Sequence[str]) -> list[str]:
    return save_figure(fig, outdir / filename, formats=formats)


def _metric_rows(df: pd.DataFrame, task: str, metric: str, **filters: object) -> pd.DataFrame:
    sub = df[(df["task"].astype(str) == task) & (df["metric"].astype(str) == metric)].copy()
    for key, value in filters.items():
        if key in sub.columns and value is not None:
            sub = sub[sub[key].astype(str) == str(value)]
    return sub


def _summarize_metric(df: pd.DataFrame, task: str, metric: str, **filters: object) -> pd.DataFrame:
    sub = _metric_rows(df, task, metric, **filters)
    if sub.empty:
        return sub
    group_cols = [col for col in ["dataset", "method_label", "missing_fraction", "mnar_fraction", "target_fraction", "lfq_subset", "protein_embedding_layer"] if col in sub.columns]
    return sub.groupby(group_cols, dropna=False, as_index=False)["value"].mean()


def plot_imputation_figures(df: pd.DataFrame, outdir: Path, formats: Sequence[str]) -> dict[str, Any]:
    plt = require_matplotlib()
    summary = _summarize_metric(df, "imputation", "pearson_missing_mean")
    if summary.empty:
        raise ValueError("No imputation pearson_missing_mean rows found")
    summary = summary.sort_values(["dataset", "method_label"], key=lambda s: s.map(_dataset_sort_key) if s.name == "dataset" else s.map(_method_sort_key))
    generated: dict[str, Any] = {"files": [], "source_data": []}
    generated["source_data"].append(_write_source(summary, outdir, "paper_imputation_pearson.csv"))

    if "missing_fraction" in summary.columns and summary["missing_fraction"].notna().any():
        fractions = pd.to_numeric(summary["missing_fraction"], errors="coerce")
        if fractions.notna().any():
            closest = fractions.sub(0.40).abs().sort_values().index[0]
            target_value = summary.loc[closest, "missing_fraction"]
            plot_df = summary[summary["missing_fraction"].astype(str) == str(target_value)].copy()
        else:
            plot_df = summary.copy()
    else:
        plot_df = summary.copy()

    datasets = sorted(plot_df["dataset"].dropna().unique(), key=_dataset_sort_key)
    methods = sorted(plot_df["method_label"].dropna().unique(), key=_method_sort_key)
    x = np.arange(len(datasets))
    offsets, bar_w = _bar_offsets(len(methods))
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for i, method in enumerate(methods):
        vals = []
        for dataset in datasets:
            row = plot_df[(plot_df["dataset"] == dataset) & (plot_df["method_label"] == method)]
            vals.append(float(row["value"].mean()) if not row.empty else np.nan)
        ax.bar(x + offsets[i], vals, width=bar_w, label=method, color=color_for(method), edgecolor="white")
    ax.set_ylabel("Pearson r on masked values")
    ax.set_xticks(x)
    ax.set_xticklabels([dataset_label(d) for d in datasets], rotation=25, ha="right")
    ax.set_title("Imputation benchmark")
    if np.isfinite(plot_df["value"]).any():
        ymin = min(0.0, float(np.nanmin(plot_df["value"])) - 0.05)
        ymax = min(1.05, max(0.2, float(np.nanmax(plot_df["value"])) + 0.05))
        ax.set_ylim(ymin, ymax)
    panel_label(ax, "A")
    place_legend(ax, ncol=2)
    generated["files"].extend(_save(fig, outdir, "paper_imputation_pearson.png", formats))
    return generated


def plot_lfq_figures(df: pd.DataFrame, outdir: Path, formats: Sequence[str]) -> dict[str, Any]:
    plt = require_matplotlib()
    subset = "has_missing" if "lfq_subset" in df.columns and (df["lfq_subset"].astype(str) == "has_missing").any() else "all"
    mae = _summarize_metric(df, "lfq", "mean_abs_error_yeast_ecoli", lfq_subset=subset)
    if mae.empty:
        mae = _summarize_metric(df, "lfq", "mean_abs_error", lfq_subset=subset)
    corr = _summarize_metric(df, "lfq", "pearson_log2fc", lfq_subset=subset)
    if mae.empty and corr.empty:
        raise ValueError("No LFQ summary rows found")
    generated: dict[str, Any] = {"files": [], "source_data": []}
    if not mae.empty:
        generated["source_data"].append(_write_source(mae, outdir, "paper_lfq_mean_abs_error.csv"))
    if not corr.empty:
        generated["source_data"].append(_write_source(corr, outdir, "paper_lfq_pearson.csv"))

    methods = sorted(set(mae.get("method_label", pd.Series(dtype=str)).dropna()) | set(corr.get("method_label", pd.Series(dtype=str)).dropna()), key=_method_sort_key)
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    if not corr.empty:
        vals = [float(corr[corr["method_label"] == method]["value"].mean()) if not corr[corr["method_label"] == method].empty else np.nan for method in methods]
        axes[0].bar(x, vals, color=[color_for(m) for m in methods], edgecolor="white")
    axes[0].set_ylabel("Pearson r")
    axes[0].set_title("Correlation")
    if not mae.empty:
        vals = [float(mae[mae["method_label"] == method]["value"].mean()) if not mae[mae["method_label"] == method].empty else np.nan for method in methods]
        axes[1].bar(x, vals, color=[color_for(m) for m in methods], edgecolor="white")
    axes[1].set_ylabel("Mean absolute error")
    axes[1].set_title("Error")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
    panel_label(axes[0], "B")
    fig.suptitle(f"LFQ benchmark ({subset})", y=1.03)
    generated["files"].extend(_save(fig, outdir, "paper_lfq_summary.png", formats))
    return generated


def plot_cell_embedding_figures(df: pd.DataFrame, outdir: Path, formats: Sequence[str]) -> dict[str, Any]:
    plt = require_matplotlib()
    metrics = ["ARI_cell", "NMI_cell", "PS_cell", "iLISI", "kBET_reject_rate"]
    sub = df[(df["task"].astype(str) == "cell_embedding") & (df["metric"].astype(str).isin(metrics))].copy()
    if sub.empty:
        raise ValueError("No cell embedding metric rows found")
    generated: dict[str, Any] = {"files": [], "source_data": []}
    generated["source_data"].append(_write_source(sub, outdir, "paper_cell_embedding_metrics.csv"))
    datasets = sorted(sub["dataset"].dropna().unique(), key=_dataset_sort_key)
    methods = sorted(sub["method_label"].dropna().unique(), key=_method_sort_key)
    fig, axes = plt.subplots(1, len(metrics), figsize=(10.4, 2.9), sharex=False)
    for ax, metric in zip(np.asarray(axes).reshape(-1), metrics):
        metric_df = sub[sub["metric"] == metric]
        x = np.arange(len(datasets))
        offsets, bar_w = _bar_offsets(len(methods))
        for i, method in enumerate(methods):
            vals = []
            for dataset in datasets:
                row = metric_df[(metric_df["dataset"] == dataset) & (metric_df["method_label"] == method)]
                vals.append(float(row["value"].mean()) if not row.empty else np.nan)
            ax.bar(x + offsets[i], vals, width=bar_w, color=color_for(method), label=method, edgecolor="white")
        ax.set_title(metric.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels([dataset_label(d) for d in datasets], rotation=35, ha="right")
    panel_label(np.asarray(axes).reshape(-1)[0], "C")
    place_legend(np.asarray(axes).reshape(-1)[-1], ncol=2)
    generated["files"].extend(_save(fig, outdir, "paper_cell_embedding_metrics.png", formats))
    return generated


def plot_protein_embedding_figures(df: pd.DataFrame, outdir: Path, formats: Sequence[str]) -> dict[str, Any]:
    plt = require_matplotlib()
    generated: dict[str, Any] = {"files": [], "source_data": []}
    sub = df[df["task"].astype(str) == "protein_embedding"].copy()
    if sub.empty:
        raise ValueError("No protein embedding metric rows found")

    rank = sub[sub["metric"].astype(str).isin(["effective_rank", "embedding_dim", "n_proteins"])].copy()
    if not rank.empty:
        generated["source_data"].append(_write_source(rank, outdir, "paper_protein_embedding_intrinsic.csv"))
        rank_plot = rank[rank["metric"].astype(str).eq("effective_rank")].copy()
        if rank_plot.empty:
            rank_plot = rank[rank["metric"].astype(str).eq("embedding_dim")].copy()
        if not rank_plot.empty:
            datasets = sorted(rank_plot["dataset"].dropna().unique(), key=_dataset_sort_key)
            methods = sorted(rank_plot["method_label"].dropna().unique(), key=_method_sort_key)
            x = np.arange(len(datasets))
            offsets, bar_w = _bar_offsets(len(methods))
            fig, ax = plt.subplots(figsize=(6.7, 3.2))
            for i, method in enumerate(methods):
                vals = []
                for dataset in datasets:
                    row = rank_plot[(rank_plot["dataset"] == dataset) & (rank_plot["method_label"] == method)]
                    vals.append(float(row["value"].mean()) if not row.empty else np.nan)
                ax.bar(x + offsets[i], vals, width=bar_w, label=method, color=color_for(method), edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels([dataset_label(d) for d in datasets], rotation=25, ha="right")
            ax.set_ylabel("Effective rank" if rank_plot["metric"].astype(str).eq("effective_rank").any() else "Embedding dimension")
            ax.set_title("Protein embedding intrinsic structure")
            panel_label(ax, "D")
            place_legend(ax, ncol=2)
            generated["files"].extend(_save(fig, outdir, "paper_protein_embedding_intrinsic.png", formats))

    hit = sub[sub["metric"].astype(str).str.match(r"^hit_at_\d+$", na=False)].copy()
    if not hit.empty:
        hit["k"] = pd.to_numeric(hit["metric"].astype(str).str.extract(r"hit_at_(\d+)")[0], errors="coerce")
        hit = hit.dropna(subset=["k"])
        generated["source_data"].append(_write_source(hit, outdir, "paper_protein_embedding_hit_curves.csv"))
        fig, ax = plt.subplots(figsize=(6.8, 3.5))
        datasets = sorted(hit["dataset"].dropna().unique(), key=_dataset_sort_key)
        methods = sorted(hit["method_label"].dropna().unique(), key=_method_sort_key)
        for d_idx, dataset in enumerate(datasets):
            for method in methods:
                g = hit[(hit["dataset"] == dataset) & (hit["method_label"] == method)].groupby("k", as_index=False)["value"].mean().sort_values("k")
                if g.empty:
                    continue
                label = f"{dataset_label(dataset)} {method}"
                ax.plot(g["k"], g["value"], marker=MARKERS[d_idx % len(MARKERS)], color=color_for(method), label=label)
        ax.set_xlabel("k")
        ax.set_ylabel("Hit rate")
        ax.set_title("Protein-neighborhood recovery")
        ax.set_xticks(sorted(hit["k"].dropna().unique()))
        panel_label(ax, "D")
        place_legend(ax, ncol=2, outside=True)
        generated["files"].extend(_save(fig, outdir, "paper_protein_embedding_hit_curves.png", formats))

    external_metrics = [
        "string.edge_auroc",
        "string.edge_auprc",
        "reactome.intra_inter_ratio",
        "go.intra_inter_ratio",
        "reactome.layer3_mean_gap",
    ]
    ext = sub[sub["metric"].astype(str).isin(external_metrics)].copy()
    if not ext.empty:
        labels = {
            "string.edge_auroc": "STRING AUROC",
            "string.edge_auprc": "STRING AUPRC",
            "reactome.intra_inter_ratio": "Reactome intra/inter",
            "go.intra_inter_ratio": "GO intra/inter",
            "reactome.layer3_mean_gap": "Layer 3 gap",
        }
        ext["metric_label"] = ext["metric"].map(labels)
        generated["source_data"].append(_write_source(ext, outdir, "paper_protein_embedding_external.csv"))
        metrics = [labels[m] for m in external_metrics if labels[m] in set(ext["metric_label"])]
        methods = sorted(ext["method_label"].dropna().unique(), key=_method_sort_key)
        x = np.arange(len(metrics))
        offsets, bar_w = _bar_offsets(len(methods))
        fig, ax = plt.subplots(figsize=(7.0, 3.3))
        for i, method in enumerate(methods):
            vals = []
            for metric in metrics:
                row = ext[(ext["method_label"] == method) & (ext["metric_label"] == metric)]
                vals.append(float(row["value"].mean()) if not row.empty else np.nan)
            ax.bar(x + offsets[i], vals, width=bar_w, label=method, color=color_for(method), edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=25, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("External biological structure")
        panel_label(ax, "E")
        place_legend(ax, ncol=2)
        generated["files"].extend(_save(fig, outdir, "paper_protein_embedding_external.png", formats))

    if not generated["files"]:
        raise ValueError("Protein embedding rows are present, but no plottable protein embedding metrics were found")
    return generated


def plot_expansion_figures(df: pd.DataFrame, outdir: Path, formats: Sequence[str]) -> dict[str, Any]:
    plt = require_matplotlib()
    summary = _summarize_metric(df, "proteome_expansion", "pearson_target_mean")
    if summary.empty:
        raise ValueError("No proteome expansion pearson_target_mean rows found")
    generated: dict[str, Any] = {"files": [], "source_data": []}
    generated["source_data"].append(_write_source(summary, outdir, "paper_expansion_pearson.csv"))

    if "target_fraction" in summary.columns and summary["target_fraction"].notna().any():
        frac = pd.to_numeric(summary["target_fraction"], errors="coerce")
        line_df = summary.assign(target_fraction_numeric=frac).dropna(subset=["target_fraction_numeric"])
    else:
        line_df = pd.DataFrame()

    if not line_df.empty and line_df["target_fraction_numeric"].nunique() > 1:
        fig, ax = plt.subplots(figsize=(6.7, 3.2))
        datasets = sorted(line_df["dataset"].dropna().unique(), key=_dataset_sort_key)
        for idx, dataset in enumerate(datasets):
            g = line_df[(line_df["dataset"] == dataset) & (line_df["method_label"] == "SABER")].sort_values("target_fraction_numeric")
            if g.empty:
                continue
            ax.plot(g["target_fraction_numeric"] * 100.0, g["value"], marker=MARKERS[idx % len(MARKERS)], label=dataset_label(dataset))
        ax.set_xlabel("Held-out proteins (%)")
        ax.set_ylabel("Pearson r")
        ax.set_title("Proteome expansion")
        panel_label(ax, "F")
        place_legend(ax, ncol=2)
        generated["files"].extend(_save(fig, outdir, "paper_expansion_fraction.png", formats))
    else:
        plot_df = line_df if not line_df.empty else summary.copy()
        datasets = sorted(plot_df["dataset"].dropna().unique(), key=_dataset_sort_key)
        methods = sorted(plot_df["method_label"].dropna().unique(), key=_method_sort_key)
        x = np.arange(len(datasets))
        offsets, bar_w = _bar_offsets(len(methods))
        fig, ax = plt.subplots(figsize=(6.7, 3.2))
        for i, method in enumerate(methods):
            vals = []
            for dataset in datasets:
                row = plot_df[(plot_df["dataset"] == dataset) & (plot_df["method_label"] == method)]
                vals.append(float(row["value"].mean()) if not row.empty else np.nan)
            ax.bar(x + offsets[i], vals, width=bar_w, label=method, color=color_for(method), edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([dataset_label(d) for d in datasets], rotation=25, ha="right")
        ax.set_ylabel("Pearson r")
        ax.set_title("Proteome expansion")
        panel_label(ax, "F")
        place_legend(ax, ncol=2)
        generated["files"].extend(_save(fig, outdir, "paper_expansion_pearson.png", formats))
    return generated


def make_paper_figures(
    source_data: str | Path,
    outdir: str | Path,
    formats: Sequence[str] = DEFAULT_FORMATS,
    tasks: Sequence[str] | None = None,
) -> dict[str, Any]:
    apply_paper_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_data)
    df = _read_source_data(source_path)
    task_set = set(tasks) if tasks is not None else {"imputation", "lfq", "cell_embedding", "protein_embedding", "proteome_expansion"}
    manifest: dict[str, Any] = {
        "source_data": str(source_path.resolve()),
        "outdir": str(outdir.resolve()),
        "formats": list(formats),
        "tasks": sorted(task_set),
        "figures": {},
        "warnings": [],
    }
    jobs = [
        ("imputation", plot_imputation_figures),
        ("lfq", plot_lfq_figures),
        ("cell_embedding", plot_cell_embedding_figures),
        ("protein_embedding", plot_protein_embedding_figures),
        ("proteome_expansion", plot_expansion_figures),
    ]
    for name, fn in jobs:
        if name not in task_set:
            continue
        try:
            manifest["figures"][name] = fn(df, outdir, formats)
        except Exception as exc:
            manifest["warnings"].append({"figure_group": name, "error": str(exc)})
    manifest_path = outdir / "paper_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest
