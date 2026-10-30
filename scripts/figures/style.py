"""Shared style + provenance for SABER result figures.

Single import point so all per-task figure scripts (imputation/LFQ/cell/
protein/expansion) stay stylistically unified.

Reuses, does NOT duplicate:
  - common (scripts/figures/common.py) -> palette (Okabe-Ito), method labels,
    dataset orders, load_long, agg_mean_sd, wilcoxon, fmt/stars helpers.
  - saber.plots -> apply_paper_style: nature-figure-compliant rcParams
    (pdf.fonttype=42 editable text, top/right spines off, clean sans-serif).

Adds: per-panel provenance (figure + exact source CSV + manifest JSON entry),
so every published panel is traceable to the rows that made it.

Rendered panels and their source CSVs are written under SABER_FIGURES_ROOT
(default ./figures/benchmark), separate from the campaign output tree.

Run figure scripts with:  python scripts/figures/<task>_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# scripts/figures/style.py -> parents[2] = repo root
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common import (  # noqa: E402
    PALETTE2, METHOD_ORDER2, DATASET_ORDER2, DATASET_LABELS2, SEEDS,
    method_label, color_for, method_sort, dataset_sort, dataset_label,
    load_long, select_metric, agg_mean_sd, higher_is_better,
    fmt_pm, fmt_p, stars,
)
from saber.paths import figures_root, outputs_root  # noqa: E402
from saber.plots import apply_paper_style, require_matplotlib, save_figure  # noqa: E402

from common import DEFAULT_RUN  # noqa: E402  default campaign run name

RUN = DEFAULT_RUN
RUN_DIR = outputs_root() / RUN
FIG_ROOT = figures_root() / "benchmark"           # <task>/<name>.{png,pdf}
SRC_ROOT = figures_root() / "benchmark" / "source_data" / "figures"  # <task>/<name>_source.csv
# Proteome-wide degradation table (supp). Resolved lazily by configure(): a run
# that contains it wins; otherwise we fall back to the 0709 run where it lives for
# the frozen campaign (source_proteome_wide_long.csv is not produced by the main
# collect, only by scripts/collect_proteome_wide.py on the proteome-wide campaign).
PW_LONG = outputs_root() / "paper_benchmark_20260709_5seed" / "source_data" / "source_proteome_wide_long.csv"


def configure(run: str | None = None, proteome_wide_csv: str | None = None) -> str:
    """Set the campaign run all figure scripts read from. Call once at the start
    of a figure script's main() (right after argparse).

    Forwards to common.configure(run) so load_long()/save()/write_csv() and
    this module's RUN_DIR all agree. Also resolves the proteome-wide table: an
    explicit --proteome-wide-csv wins; else an in-run copy; else the 0709
    fallback. Returns the resolved run name.

    Args:
        run: run name under $SABER_OUTPUTS_ROOT or an explicit run path
             (e.g. ``results`` for the shipped tables). None -> common.DEFAULT_RUN.
        proteome_wide_csv: optional explicit path to source_proteome_wide_long.csv.
    """
    global RUN, RUN_DIR, FIG_ROOT, SRC_ROOT, PW_LONG
    import common
    RUN = common.configure(run)  # propagates to load_long/save/write_csv
    RUN_DIR = common.OUT_ROOT
    FIG_ROOT = figures_root() / "benchmark"
    SRC_ROOT = figures_root() / "benchmark" / "source_data" / "figures"
    in_run = RUN_DIR / "source_data" / "source_proteome_wide_long.csv"
    shipped = REPO / "results" / "source_data" / "source_proteome_wide_long.csv"
    if proteome_wide_csv:
        PW_LONG = Path(proteome_wide_csv)
    elif in_run.exists():
        PW_LONG = in_run
    elif shipped.is_file():
        PW_LONG = shipped
    # else: keep the 0709 fallback bound at import
    return RUN


# --------------------------------------------------------------------------- #
# Canonical dataset list. Derks_2023 is the same data as Derks_2022 (a
# different version) and is excluded everywhere; historical rows for it in
# source_metrics_long.csv are filtered out by keep_datasets().
# --------------------------------------------------------------------------- #
EXCLUDED_DATASETS = {"Derks_2023"}
LFQ_DATASETS = ["PXD071205_LFQ"]
DATASETS = [d for d in DATASET_ORDER2
            if d not in EXCLUDED_DATASETS and d not in LFQ_DATASETS]  # 5 non-LFQ


def keep_datasets(df: pd.DataFrame, include_lfq: bool = False) -> pd.DataFrame:
    """Filter to the canonical dataset list (drop Derks_2023; LFQ opt-in)."""
    cols = DATASETS + (LFQ_DATASETS if include_lfq else [])
    return df[df["dataset"].isin(cols)].copy()


def load_proteome_wide() -> pd.DataFrame:
    """Load the proteome-wide sub-campaign long table (degradation ρ + GO n_sig),
    produced by scripts/collect_proteome_wide.py. Same access pattern as load_long();
    keeps proteome-wide inside the expansion framework (one data source)."""
    df = pd.read_csv(PW_LONG)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


# --------------------------------------------------------------------------- #
# matplotlib
# --------------------------------------------------------------------------- #
def get_plt():
    """Apply paper style once and return pyplot (nature-figure-compliant)."""
    require_matplotlib()
    apply_paper_style()
    import matplotlib.pyplot as plt
    return plt


def save_panel(fig, name: str, task: str = "imputation", formats=("png", "pdf")) -> list[str]:
    """Save a panel to paper_figures/<task>/<name>.{png,pdf}."""
    out = FIG_ROOT / task
    out.mkdir(parents=True, exist_ok=True)
    return save_figure(fig, out / name, formats=formats)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def write_source(df: pd.DataFrame, name: str, task: str = "imputation") -> str:
    """Write the exact rows behind a panel as <name>_source.csv."""
    d = SRC_ROOT / task
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}_source.csv"
    df.to_csv(p, index=False)
    return str(p)


_MANIFEST: list[dict] = []


def reset_manifest() -> None:
    _MANIFEST.clear()


def log_panel(panel: str, kind: str, figure: str, source: str,
              description: str, n_rows: int | None = None,
              params: dict | None = None, task: str | None = None) -> None:
    _MANIFEST.append(dict(
        panel=panel, kind=kind, task=task, figure=str(figure), source=str(source),
        description=description, n_rows=n_rows, params=params or {},
    ))


def write_manifest(task: str = "imputation", name: str | None = None) -> str:
    d = SRC_ROOT / task
    d.mkdir(parents=True, exist_ok=True)
    name = name or f"{task}_figures_manifest.json"
    p = d / name
    p.write_text(json.dumps(_MANIFEST, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(p)
