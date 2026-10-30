#!/usr/bin/env python
"""One-click SABER benchmark: materialize -> submit (SLURM farm) -> collect.

Thin orchestrator over:
  - ``scripts/run_5seed_campaign.py``  (materialize one full benchmark per seed)
  - ``scripts/submit_farm.py``         (two-phase train -> external -> collect farmer)

Nothing is submitted unless ``--submit`` is passed (default: materialize only, so
you can review the generated configs/commands first). The collect step runs
automatically at the end of the farm and writes ``source_data/source_metrics_long.csv``;
after that, regenerate every figure with ``scripts/plot_all.py``.

The two robustness arms (missingness grid + sequence-cluster holdout) are a
separate campaign: run this driver once without them for the main benchmark, and
once WITH ``--missingness-grid --sequence-holdout`` (typically 5 seeds) for the
grid/sequence run that feeds the supplementary (and now main) panels.

Usage:
  # 1) main benchmark, 10 seeds, materialize only (review before submit)
  conda run -n saber python scripts/run_all.py --tag review \
      --seeds 42,123,2024,2025,7,2026,2027,2028,2029,2030
  # 2) submit it on the farm (4 GPUs)
  conda run -n saber python scripts/run_all.py --tag review --seeds ... --submit --farmers 4
  # 3) grid + sequence-cluster holdout, 5 seeds, submit
  conda run -n saber python scripts/run_all.py --tag grid \\
      --seeds 42,123,2024,2025,7 --missingness-grid --sequence-holdout --submit

Outputs land in ``outputs/paper_benchmark_<date>_<tag>/``.
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(REPO))  # resolve `saber` from this checkout, not the env
DEFAULT_SEEDS_10 = "42,123,2024,2025,7,2026,2027,2028,2029,2030"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="run tag -> outputs/paper_benchmark_<date>_<tag>")
    p.add_argument("--seeds", default=DEFAULT_SEEDS_10)
    p.add_argument("--date", default="", help="YYYYMMDD (default: today)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--missingness-grid", action="store_true",
                   help="add full missing-fraction x MNAR-share imputation grid")
    p.add_argument("--sequence-holdout", action="store_true",
                   help="add sequence-cluster (MMseqs2) expansion holdout on human datasets")
    p.add_argument("--submit", action="store_true",
                   help="submit the SLURM farm after materializing (else just materialize)")
    p.add_argument("--farmers", type=int, default=4, help="farmer array size ~= #GPUs")
    p.add_argument("--partition", default="stat")
    p.add_argument("--time", default="08:00:00")
    p.add_argument("--conda-env", default="saber")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    date = args.date or datetime.date.today().strftime("%Y%m%d")
    run_name = f"paper_benchmark_{date}_{args.tag}"
    from saber.paths import outputs_root
    run_dir = outputs_root() / run_name

    # 1) materialize per seed
    mat = ["conda", "run", "--no-capture-output", "-n", args.conda_env, "python",
           str(REPO / "scripts" / "run_5seed_campaign.py"),
           "--date", date, "--tag", args.tag, "--seeds", args.seeds,
           "--epochs", str(args.epochs)]
    if args.missingness_grid:
        mat.append("--missingness-grid")
    if args.sequence_holdout:
        mat.append("--sequence-holdout")
    print(f"[run_all] 1) materialize -> {run_dir}")
    print("  $ " + " ".join(mat), flush=True)
    rc = subprocess.call(mat, cwd=str(REPO))
    if rc != 0:
        print(f"[run_all] materialize FAILED (rc={rc}). Aborting.")
        return rc

    # 2) submit farm (optional)
    if args.submit:
        sub = ["conda", "run", "--no-capture-output", "-n", args.conda_env, "python",
               str(REPO / "scripts" / "submit_farm.py"),
               "--campaign-root", str(run_dir),
               "--farmers", str(args.farmers), "--partition", args.partition, "--time", args.time]
        print(f"\n[run_all] 2) submit farm -> {run_dir}")
        print("  $ " + " ".join(sub), flush=True)
        rc = subprocess.call(sub, cwd=str(REPO))
        if rc != 0:
            print(f"[run_all] submit_farm.py FAILED (rc={rc}). The campaign is materialized; "
                  f"generate + submit manually:\n    python scripts/submit_farm.py "
                  f"--campaign-root {run_dir} --farmers {args.farmers}\n    bash {run_dir}/submit_farm.sh")
            return rc
        # submit_farm.py only WRITES submit_farm.sh (it is a generator, like
        # run_5seed_campaign.py); execute it to actually sbatch the train->ext->collect chain.
        submit_script = run_dir / "submit_farm.sh"
        print(f"\n[run_all] 3) sbatch the farm chain -> {submit_script}")
        print("  $ bash " + str(submit_script), flush=True)
        rc = subprocess.call(["bash", str(submit_script)], cwd=str(REPO))
        if rc != 0:
            print(f"[run_all] submit_farm.sh FAILED (rc={rc}). Submit manually: bash {submit_script}")
            return rc
        print(f"\n[run_all] submitted. Monitor: squeue -u $USER")
        print(f"[run_all] collect auto-runs at the end of the farm -> {run_dir}/source_data/source_metrics_long.csv")
        print(f"[run_all] then regenerate figures: conda run -n saber python scripts/plot_all.py "
              f"--run {run_name}"
              + (f" --grid-run {run_name}" if args.missingness_grid or args.sequence_holdout else ""))
    else:
        print(f"\n[run_all] materialize-only done -> {run_dir}")
        print(f"[run_all] review, then submit with --submit, or:")
        print(f"    conda run -n saber python scripts/submit_farm.py --campaign-root {run_dir} --farmers {args.farmers}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
