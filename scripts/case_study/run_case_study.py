#!/usr/bin/env python
"""Launcher for the liver-zonation case study (Fig. 2 and Fig. 5a-d).

Each case-study run is one YAML under ``configs/case_study/``:

  - ``expansion_<tier>_<cond>.yaml``  -> ``saber train-expansion``
      tiers: canonical | canonicalax | hss_markers | zonated | zonatedax
      conds: saber | noplm
      The ``*ax`` variants additionally hold out the 215 HSS axis proteins
      (tier-union-axis protocol used by the Fig. 2 metrics).
  - ``cellembed_<cond>.yaml``         -> ``saber train``
      plain training for the protein/cell-embedding panels (Fig. 5b-d).

The base config of each group is seed 42; per-seed variants for the 10-seed
protocol are generated on the fly (``project.name``/``outdir`` gain a
``_seed<s>`` suffix and every ``seed`` field is overridden), so the repository
ships one YAML per group instead of a hundred near-duplicates.

Run everything (all tiers x conds x 10 seeds):
    python scripts/case_study/run_case_study.py

Slice (e.g. just the zonated tier, saber, seed 42 + 123):
    python scripts/case_study/run_case_study.py --tier zonated --cond saber --seed 42,123

Dry-run (list the commands without executing):
    python scripts/case_study/run_case_study.py --dry-run

Notes:
  - Outputs go to each config's ``project.outdir`` (under $SABER_OUTPUTS_ROOT).
  - Brunner cell-cycle panels (Fig. 5e-h) use the ``brunner_*`` config and the
    axis-holdout chain; they are launched separately
    (scripts/case_study/realimpute_axisholdout.py --dataset brunner).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from saber.paths import outputs_root  # noqa: E402

CFG_DIR = REPO / "configs" / "case_study"

TIERS = ("canonical", "canonicalax", "hss_markers", "zonated", "zonatedax")
CONDS = ("saber", "noplm")
# base seed = 42 (no filename suffix); multi-seed variants are generated.
MULTI_SEEDS = (0, 1, 2, 3, 4, 7, 123, 2024, 2025)
BASE_SEED = 42
ALL_SEEDS = (BASE_SEED, *MULTI_SEEDS)


def _seed_variant_path(cfg: Path, seed: int) -> Path:
    out = outputs_root() / "_generated_configs"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{cfg.stem}_seed{seed}.yaml"


def _patched_config(cfg: Path, name_suffix: str, seed: int | None = None) -> Path:
    """Write a copy of ``cfg`` under $SABER_OUTPUTS_ROOT/_generated_configs with
    ``project.name``/``outdir`` gaining ``name_suffix`` (and, for seed variants,
    every ``seed`` field overridden)."""
    raw = yaml.safe_load(cfg.read_text())
    suffixed = f"{raw['project']['name']}{name_suffix}"
    raw["project"]["name"] = suffixed
    raw["project"]["outdir"] = str(Path(raw["project"]["outdir"]).parent / suffixed)
    if seed is not None:
        raw["project"]["seed"] = seed
        for section in ("missing", "expansion"):
            if isinstance(raw.get(section), dict) and "seed" in raw[section]:
                raw[section]["seed"] = seed
    out = outputs_root() / "_generated_configs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{suffixed}.yaml"
    dst.write_text(yaml.safe_dump(raw, sort_keys=False))
    return dst


def _seed_variant(cfg: Path, seed: int, suffix: str = "") -> Path:
    # name order: <name>_seed<s>{<suffix>} — the metrics scripts glob this.
    return _patched_config(cfg, f"_seed{seed}{suffix}", seed=seed)


def _configs_for(kind: str, tier: str | None, cond: str, seeds: list[int], suffix: str = "") -> list[Path]:
    """Config files to run for a (kind, [tier,] cond) group.

    The base config represents seed 42 (a suffixed copy is generated when
    --suffix is given, so runs land in a separate tree); every additional
    requested seed gets a generated variant. Cellembed has no tier."""
    if kind == "expansion":
        base = CFG_DIR / f"expansion_{tier}_{cond}.yaml"
    else:  # cellembed
        base = CFG_DIR / f"cellembed_{cond}.yaml"
    out: list[Path] = []
    if base.exists():
        if BASE_SEED in seeds:
            out.append(base if not suffix else _patched_config(base, suffix))
        for s in MULTI_SEEDS:
            if s in seeds:
                out.append(_seed_variant(base, s, suffix))
    else:
        print(f"[case-study] WARNING: missing config {base}")
    return out


def _saber_command(kind: str, cfg: Path) -> list[str]:
    sub = "train-expansion" if kind == "expansion" else "train"
    return [sys.executable, "-m", "saber.cli", sub, "--config", str(cfg)]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", default="all", choices=("all", "expansion", "cellembed"))
    p.add_argument("--tier", default="all", choices=("all", *TIERS),
                   help="expansion tier (ignored for cellembed)")
    p.add_argument("--cond", default="all", choices=("all", *CONDS))
    p.add_argument("--seed", default="all",
                   help="comma-separated seeds, or 'all' (42,0,1,2,3,4,7,123,2024,2025)")
    p.add_argument("--suffix", default="",
                   help="run-tree suffix: run names/outdirs gain the suffix, so an "
                        "existing tree is never overwritten")
    p.add_argument("--dry-run", action="store_true", help="list commands, run nothing")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    kinds = ("expansion", "cellembed") if args.kind == "all" else (args.kind,)
    tiers = TIERS if args.tier == "all" else (args.tier,)
    conds = CONDS if args.cond == "all" else (args.cond,)
    if args.seed.lower() in ("all", ""):
        seeds = list(ALL_SEEDS)
    else:
        seeds = [int(s) for s in args.seed.split(",") if s.strip()]

    jobs: list[tuple[str, Path, list[str]]] = []
    for kind in kinds:
        kind_tiers = tiers if kind == "expansion" else (None,)  # cellembed has no tier
        for tier in kind_tiers:
            for cond in conds:
                for cfg in _configs_for(kind, tier, cond, seeds, args.suffix):
                    jobs.append((kind, cfg, _saber_command(kind, cfg)))

    print(f"[case-study] {len(jobs)} config(s) to run.")
    for kind, cfg, _ in jobs:
        print(f"  + {kind}/{cfg.name}")

    if args.dry_run:
        print("\n[dry-run] commands:")
        for _, _, cmd in jobs:
            print("  " + " ".join(cmd))
        return 0

    if not jobs:
        print("[case-study] nothing to run.")
        return 0

    rc_total = 0
    for i, (kind, cfg, cmd) in enumerate(jobs, 1):
        print(f"\n=== [{i}/{len(jobs)}] {kind}: {cfg.name} ===", flush=True)
        print("    $ " + " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=str(REPO))
        if rc != 0:
            rc_total = rc
            print(f"    !! non-zero exit ({rc}); continuing to next job.", flush=True)
    print(f"\n[case-study] done. {len(jobs)} run(s). "
          f"{'All OK.' if rc_total == 0 else f'Some failed (last rc={rc_total}).'}")
    return rc_total


if __name__ == "__main__":
    sys.exit(main())
