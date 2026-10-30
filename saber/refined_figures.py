from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from .paper_figures import _read_source_data
from .plots import apply_paper_style, color_for, dataset_label, method_label, place_legend, require_matplotlib, save_figure


def _read_spec(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        if path.suffix.lower() == ".json":
            payload = json.load(f)
        else:
            payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Figure spec must be a mapping: {path}")
    return payload


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _filter_df(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    sub = df.copy()
    for col in ["task", "metric"]:
        value = spec.get(col)
        if value is not None and col in sub.columns:
            sub = sub[sub[col].astype(str) == str(value)]
    for key, value in (spec.get("filters") or {}).items():
        if key in sub.columns:
            numeric = pd.to_numeric(sub[key], errors="coerce")
            try:
                target = float(value)
            except (TypeError, ValueError):
                target = None
            if target is not None and numeric.notna().any():
                sub = sub[np.isclose(numeric.astype(float), target, rtol=1e-7, atol=1e-9)]
            else:
                sub = sub[sub[key].astype(str) == str(value)]
    datasets = [str(x) for x in _as_list(spec.get("datasets"))]
    if datasets and "dataset" in sub.columns:
        sub = sub[sub["dataset"].astype(str).isin(datasets)]
    methods = [str(x) for x in _as_list(spec.get("methods"))]
    if methods:
        if "method_label" not in sub.columns:
            sub["method_label"] = [method_label(m, c) for m, c in zip(sub["method"], sub.get("condition", ""))]
        sub = sub[sub["method_label"].astype(str).isin(methods)]
    return sub.copy()


def make_refined_figure(
    source_data: str | Path,
    spec_path: str | Path,
    outdir: str | Path,
    formats: Sequence[str] = ("png", "pdf"),
) -> dict[str, Any]:
    apply_paper_style()
    plt = require_matplotlib()
    spec = _read_spec(spec_path)
    df = _read_source_data(source_data)
    sub = _filter_df(df, spec)
    if sub.empty:
        raise ValueError("No rows matched refined figure spec")
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.dropna(subset=["value"])
    if sub.empty:
        raise ValueError("Rows matched spec, but all values are non-numeric")

    plot_type = str(spec.get("plot_type", "bar"))
    if plot_type != "bar":
        raise ValueError("Only plot_type='bar' is currently supported")

    if "method_label" not in sub.columns:
        sub["method_label"] = [method_label(m, c) for m, c in zip(sub["method"], sub.get("condition", ""))]
    group_cols = [col for col in ["dataset", "method_label"] if col in sub.columns]
    plot_df = sub.groupby(group_cols, as_index=False, dropna=False)["value"].mean()
    dataset_order = [str(x) for x in _as_list(spec.get("datasets"))] or sorted(plot_df["dataset"].dropna().unique())
    method_order = [str(x) for x in _as_list(spec.get("methods"))] or sorted(plot_df["method_label"].dropna().unique())

    x = np.arange(len(dataset_order))
    width = 0.78 / max(1, len(method_order))
    offsets = (np.arange(len(method_order)) - (len(method_order) - 1) / 2) * width
    fig, ax = plt.subplots(figsize=tuple(spec.get("figsize", [7.0, 3.2])))
    for i, method in enumerate(method_order):
        vals = []
        for dataset in dataset_order:
            row = plot_df[(plot_df["dataset"].astype(str) == str(dataset)) & (plot_df["method_label"].astype(str) == str(method))]
            vals.append(float(row["value"].mean()) if not row.empty else np.nan)
        ax.bar(x + offsets[i], vals, width=width * 0.92, label=method, color=color_for(method), edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([dataset_label(d) for d in dataset_order], rotation=float(spec.get("x_rotation", 25)), ha="right")
    ax.set_ylabel(str(spec.get("ylabel", spec.get("metric", "value"))))
    title = str(spec.get("title", ""))
    if title:
        ax.set_title(title)
    ylim = _as_list(spec.get("ylim"))
    if len(ylim) == 2:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    place_legend(ax, ncol=int(spec.get("legend_ncol", 2)))

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name = str(spec.get("name", Path(spec_path).stem))
    files = save_figure(fig, outdir / f"{name}.png", formats=formats)
    source_path = outdir / f"{name}_source.csv"
    plot_source_path = outdir / f"{name}_plot_source.csv"
    sub.to_csv(source_path, index=False)
    plot_df.to_csv(plot_source_path, index=False)
    manifest = {
        "source_data": str(Path(source_data).resolve()),
        "spec": str(Path(spec_path).resolve()),
        "outdir": str(outdir.resolve()),
        "name": name,
        "files": files,
        "figure_source": str(source_path.resolve()),
        "plot_source": str(plot_source_path.resolve()),
        "n_rows": int(sub.shape[0]),
    }
    manifest_path = outdir / f"{name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest
