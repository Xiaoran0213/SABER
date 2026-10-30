#!/usr/bin/env python
"""One-click regeneration of all SABER paper figures from a campaign run.

Reads the already-collected source CSVs of a run (produced by collect at the end
of a benchmark farm) and renders every main + supplementary panel via the
per-task scripts under ``scripts/figures/``. Each panel is traceable to its
source CSV (``<run>/source_data/figures/<task>/<name>_source.csv``) and a per-task
manifest, so figure<->number provenance is exact.

Liver zonation (Fig. 2, Fig. 5a-d) and the Brunner cell-cycle case study
(Fig. 5e-h) have their own pipeline under ``scripts/case_study/`` and
``scripts/figures/fig2_v2_expansion.py`` / ``fig5_*.py``. They are not part
of this driver.

Usage:
  python scripts/plot_all.py --run <campaign_dir>
  python scripts/plot_all.py --run <campaign_dir> --panels main,supp

Panel groups (current manuscript numbering):
  main : Fig. 3 imputation, Fig. 4 LFQ
  supp : Fig. S7 cell embedding, Fig. S8 random-holdout expansion,
         missingness grid, sequence-cluster holdout, runtime/memory
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "scripts" / "figures"

# (label, script, arg-kind). arg-kind: "run" = --run; "run_pw" = --run + --proteome-wide-dir;
# "csv" = --csv <grid-run>/source_data/source_metrics_long.csv.
MAIN = [
    ("Fig. 3 imputation", "imputation_figures.py", "run"),
    ("Fig. 4 LFQ", "lfq_figures.py", "run"),
]
SUPP = [
    ("Fig. S7 cell / protein embedding", "representation_figures.py", "run"),
    ("Fig. S8 random-holdout expansion", "expansion_figures.py", "run_pw"),
    ("Fig. S2 missingness grid", "missingness_grid_figures.py", "csv"),
    ("Fig. S3 sequence-cluster holdout", "expansion_holdout_figures.py", "csv"),
    ("Supp efficiency", "efficiency_figures.py", "run"),
]


def _build(script: str, kind: str, run: str, grid_csv: str, pw_dir: str, env: str) -> list[str]:
    # Children are plain python scripts; they resolve the `saber` package from
    # this checkout themselves. --conda-env wraps them for the authors' cluster.
    cmd = ([sys.executable, str(FIG / script)] if not env
           else ["conda", "run", "--no-capture-output", "-n", env, "python", str(FIG / script)])
    if kind == "run":
        cmd += ["--run", run]
    elif kind == "run_pw":
        cmd += ["--run", run, "--proteome-wide-dir", pw_dir]
    elif kind == "csv":
        cmd += ["--csv", grid_csv]
    return cmd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sys.path.insert(0, str(REPO))
    from saber.paths import campaign_run_dir, outputs_root
    p.add_argument("--run", default="paper_benchmark_20260801_final_10seed",
                   help="main campaign run (name under $SABER_OUTPUTS_ROOT or a path; "
                        "'results' uses the tables shipped in the repository)")
    p.add_argument("--grid-run", default="paper_benchmark_20260801_final_grid_5seed",
                   help="grid+sequence campaign run (for supp grid/seq panels)")
    p.add_argument("--proteome-wide-dir", default=str(outputs_root() / "proteome-wide"),
                   help="proteome-wide campaign dir (GO enrichment + degradation; expansion panels)")
    p.add_argument("--panels", default="main,supp",
                   help="comma list of: main, supp")
    p.add_argument("--conda-env", default="",
                   help="optional conda env to wrap child invocations (default: current python)")
    args = p.parse_args()

    grid_csv_path = campaign_run_dir(args.grid_run) / "source_data" / "source_metrics_long.csv"
    if not grid_csv_path.is_file():
        shipped_grid = REPO / "results" / "source_data" / "source_metrics_grid_long.csv"
        grid_csv_path = shipped_grid if shipped_grid.is_file() else grid_csv_path
    grid_csv = str(grid_csv_path)
    args.run = str(campaign_run_dir(args.run))
    selected = {s.strip() for s in args.panels.split(",") if s.strip()}
    groups: list[tuple[str, str, str]] = []
    if "main" in selected:
        groups += MAIN
    if "supp" in selected:
        groups += SUPP
    if not groups:
        print("[plot_all] no panel groups selected (--panels). Nothing to do.")
        return 0

    env = {**os.environ, "MPLBACKEND": "Agg"}
    print(f"[plot_all] run={args.run}  grid-run={args.grid_run}  proteome-wide={args.proteome_wide_dir}")
    print(f"[plot_all] panels: {', '.join(label for label, _, _ in groups)}\n")

    rc_total = 0
    for label, script, kind in groups:
        cmd = _build(script, kind, args.run, grid_csv, args.proteome_wide_dir, args.conda_env)
        print(f"=== {label} ===\n  $ {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, cwd=str(REPO), env=env)
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"  -> {status}\n", flush=True)
        if rc != 0:
            rc_total = rc

    print(f"[plot_all] done. {'All panels OK.' if rc_total == 0 else f'Some panels failed (last rc={rc_total}).'}")
    n_ok = sum(1 for label, _, _ in groups) - (1 if rc_total else 0)
    return rc_total


if __name__ == "__main__":
    sys.exit(main())
