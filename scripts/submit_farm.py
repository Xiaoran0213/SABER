#!/usr/bin/env python3
"""Generate a farmer-style SLURM submit for a materialized SABER campaign.

The main `materialize_full_benchmark` emits `train_array.sbatch` as a raw
`--array=0-N`, which exceeds MaxSubmitJobPerUser (~100) for large campaigns
(e.g. the 10-seed grid). This wraps the SAME materialized command lists
(`commands/train_commands.txt`, `commands/external_commands.txt`) in a
two-phase farmer with a dependency chain:

    train_farm (NF farmers, strided)  --afterok-->  external_farm (NF farmers)
        --afterok-->  collect_and_plot (materialize's own collect sbatch)

Each farmer is one SLURM job holding 1 GPU; NF farmers == concurrent GPUs
(set --farmers to respect the shared-GPU budget, e.g. 4). Scales to any grid
size while keeping submitted-job count == NF (well under the limit).

Usage:
  python scripts/submit_farm.py --run outputs/paper_benchmark_<date>_<tag>/seeds/seed42 \
      [--farmers 4] [--partition stat] [--time 08:00:00] [--mem 48G] [--cpus 4]
  # then:  bash <run>/submit_farm.sh
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(REPO))  # resolve `saber` from this checkout, not the env


def _farm_sbatch(*, path: Path, job_name: str, commands_file: Path, logs_dir: Path,
                 partition: str, time_limit: str, mem: str, cpus: int, farmers: int,
                 gres: str, repo_root: Path) -> None:
    """Write a strided farmer sbatch over a commands file (one full command per line)."""
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --array=0-{farmers - 1}",
    ]
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    lines += [
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output={logs_dir}/{job_name}_%A_%a.out",
        f"#SBATCH --error={logs_dir}/{job_name}_%A_%a.err",
        "set -euo pipefail",
        f"cd {repo_root}",
        f"mapfile -t CMDS < {commands_file}",
        "N=${#CMDS[@]}",
        'if [ "$N" -eq 0 ]; then echo "[farmer $SLURM_ARRAY_TASK_ID] no commands"; exit 0; fi',
        f'echo "[farmer $SLURM_ARRAY_TASK_ID] sweeping $N commands (stride {farmers})"',
        f"for ((i=SLURM_ARRAY_TASK_ID; i<N; i+={farmers})); do",
        '  echo "[farmer $SLURM_ARRAY_TASK_ID] ($i/$N)"',
        '  eval "${CMDS[$i]}"',
        "done",
        'echo "[farmer $SLURM_ARRAY_TASK_ID] DONE"',
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def _grouped_external_farm(*, sbatch_path: Path, commands_file: Path, per_farmer_dir: Path,
                           job_name: str, logs_dir: Path, partition: str, time_limit: str,
                           mem: str, cpus: int, farmers: int, gres: str, repo_root: Path) -> None:
    """Race-safe external farmer. External methods of one SABER manifest share + rewrite
    that manifest's imputation.npz; if farmed by stride they land on different farmers and
    rewrite concurrently -> BadZipFile on the reader. This groups commands by --manifest and
    assigns whole manifest-groups round-robin to farmers, so each manifest's methods run
    sequentially on ONE farmer (and different farmers touch disjoint manifests)."""
    import re as _re
    lines = [l.strip() for l in Path(commands_file).open() if l.strip()]
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for idx, l in enumerate(lines):
        m = _re.search(r"--manifest (\S+)", l)
        key = m.group(1) if m else f"__nomanifest_{idx}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(l)
    per_farmer_dir.mkdir(parents=True, exist_ok=True)
    buckets: list[list[str]] = [[] for _ in range(farmers)]
    for i, key in enumerate(order):
        buckets[i % farmers].extend(groups[key])
    for k in range(farmers):
        (per_farmer_dir / f"{job_name}_{k}.txt").write_text(
            "\n".join(buckets[k]) + ("\n" if buckets[k] else ""), encoding="utf-8")
    flines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --array=0-{farmers - 1}",
    ]
    if gres:
        flines.append(f"#SBATCH --gres={gres}")
    flines += [
        f"#SBATCH --cpus-per-task={cpus}", f"#SBATCH --mem={mem}", f"#SBATCH --time={time_limit}",
        f"#SBATCH --output={logs_dir}/{job_name}_%A_%a.out", f"#SBATCH --error={logs_dir}/{job_name}_%A_%a.err",
        "set -euo pipefail", f"cd {repo_root}",
        f'FILE={per_farmer_dir}/{job_name}_$SLURM_ARRAY_TASK_ID.txt',
        'mapfile -t CMDS < "$FILE"', 'N=${#CMDS[@]}',
        'if [ "$N" -eq 0 ]; then echo "[farmer $SLURM_ARRAY_TASK_ID] empty"; exit 0; fi',
        f'echo "[farmer $SLURM_ARRAY_TASK_ID] $N commands (manifest-grouped, sequential)"',
        'for ((i=0; i<N; i++)); do',
        '  echo "[farmer $SLURM_ARRAY_TASK_ID] ($i/$N)"',
        '  eval "${CMDS[$i]}"',
        "done",
        'echo "[farmer $SLURM_ARRAY_TASK_ID] DONE"',
        "",
    ]
    sbatch_path.write_text("\n".join(flines), encoding="utf-8")
    sbatch_path.chmod(0o755)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="", help="a single materialized seed dir (contains commands/)")
    ap.add_argument("--campaign-root", default="", help="a campaign root with seeds/seedN/ subdirs (pools ALL seeds)")
    ap.add_argument("--farmers", type=int, default=4, help="# concurrent farmer jobs == # GPUs to occupy")
    ap.add_argument("--partition", default="stat")
    ap.add_argument("--time", default="08:00:00")
    ap.add_argument("--mem", default="48G")
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--gres", default="gpu:1", help="GPU per farmer (set '' for CPU-only external phase)")
    ap.add_argument("--external-gres", default="gpu:1", help="GPU per farmer for the external phase")
    ap.add_argument("--repo-root", default=str(REPO))
    args = ap.parse_args()

    if not args.run and not args.campaign_root:
        raise SystemExit("provide --run <seed_dir> or --campaign-root <campaign with seeds/>")
    repo_root = Path(args.repo_root).resolve()

    # Resolve the working dir + command lists (single seed, or pooled across seeds).
    if args.campaign_root:
        root = Path(args.campaign_root).resolve()
        seed_dirs = sorted(root.glob("seeds/seed*"))
        if not seed_dirs:
            raise SystemExit(f"no seeds/seed* under {root}")
        run = root  # sbatch/submit artifacts live at campaign root
        pooled = run / "commands_pooled"
        pooled.mkdir(parents=True, exist_ok=True)
        train_cmds = pooled / "train_commands.txt"
        ext_cmds = pooled / "external_commands.txt"
        # concatenate every seed's command lists into one pooled list
        with train_cmds.open("w") as ft, ext_cmds.open("w") as fe:
            for sd in seed_dirs:
                tc = sd / "commands" / "train_commands.txt"
                ec = sd / "commands" / "external_commands.txt"
                if tc.exists():
                    ft.write(tc.read_text())
                if ec.exists() and ec.stat().st_size > 0:
                    fe.write(ec.read_text())
        print(f"[submit_farm] pooled {len(seed_dirs)} seeds -> {train_cmds}")
    else:
        run = Path(args.run).resolve()

    cmds = run / "commands" if not args.campaign_root else run / "commands_pooled"
    slurm = run / "slurm"
    logs = run / "logs" / "slurm"
    slurm.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    if not args.campaign_root:
        train_cmds = cmds / "train_commands.txt"
        ext_cmds = cmds / "external_commands.txt"
    if not train_cmds.exists() or train_cmds.stat().st_size == 0:
        raise SystemExit(f"missing/empty {train_cmds} (run materialize_full_benchmark first)")

    train_sb = slurm / "train_farm.sbatch"
    _farm_sbatch(path=train_sb, job_name="saber_train_farm", commands_file=train_cmds,
                 logs_dir=logs, partition=args.partition, time_limit=args.time,
                 mem=args.mem, cpus=args.cpus, farmers=args.farmers, gres=args.gres, repo_root=repo_root)

    has_ext = ext_cmds.exists() and ext_cmds.stat().st_size > 0
    ext_sb = slurm / "external_farm.sbatch"
    if has_ext:
        # External methods of a manifest share+rewrite its imputation.npz -> MUST run the
        # manifest's methods on one farmer (grouped), not strided, or concurrent rewrites race.
        _grouped_external_farm(sbatch_path=ext_sb, commands_file=ext_cmds,
                               per_farmer_dir=slurm / "ext_per_farmer", job_name="saber_ext_farm",
                               logs_dir=logs, partition=args.partition, time_limit=args.time,
                               mem=args.mem, cpus=args.cpus, farmers=args.farmers, gres=args.external_gres, repo_root=repo_root)

    # collect: per-seed (single --run) uses materialize's collect_and_plot.sbatch;
    # campaign-root uses the multiseed collect_5seed.py wrapper if present.
    collect_sb = slurm / "collect_and_plot.sbatch"
    multiseed_collect = run / "collect_5seed.py" if args.campaign_root else None
    submit = run / "submit_farm.sh"
    chain = ["#!/usr/bin/env bash", "set -euo pipefail", f'cd {repo_root}']
    chain.append(f'TRAIN=$(sbatch --parsable {train_sb})')
    chain.append(f'echo "train_farm: $TRAIN"')
    if has_ext:
        chain.append(f'EXT=$(sbatch --parsable --dependency=afterok:$TRAIN {ext_sb})')
        chain.append(f'echo "external_farm: $EXT"')
        prev = "$EXT"
    else:
        prev = "$TRAIN"
    if args.campaign_root and multiseed_collect and multiseed_collect.exists():
        collect_run = slurm / "collect_multiseed_farm.sbatch"
        collect_run.write_text(
            "#!/usr/bin/env bash\n"
            "#SBATCH --job-name=saber_collect_farm\n#SBATCH --partition=stat\n"
            "#SBATCH --cpus-per-task=4\n#SBATCH --mem=32G\n#SBATCH --time=12:00:00\n"
            f"#SBATCH --output={logs}/collect_farm_%A.out\n"
            "set -euo pipefail\n"
            f"cd {repo_root}\n"
            f"conda run -n saber python {multiseed_collect}\n",
            encoding="utf-8",
        )
        collect_run.chmod(0o755)
        chain.append(f'COLLECT=$(sbatch --parsable --dependency=afterok:{prev} {collect_run})')
        chain.append(f'echo "collect_multiseed: $COLLECT"')
        chain.append(f'echo "$COLLECT" > {run}/collect_jobid.txt')
    elif collect_sb.exists():
        chain.append(f'COLLECT=$(sbatch --parsable --dependency=afterok:{prev} {collect_sb})')
        chain.append(f'echo "collect: $COLLECT"')
    chain.append(f'echo "submitted farmer campaign: {run}"')
    submit.write_text("\n".join(chain) + "\n", encoding="utf-8")
    submit.chmod(0o755)

    print(f"[submit_farm] run: {run}")
    print(f"[submit_farm] farmers: {args.farmers} (== GPUs occupied)")
    print(f"[submit_farm] train_farm -> {'external_farm -> ' if has_ext else ''}collect")
    print(f"[submit_farm] launch: bash {submit}")


if __name__ == "__main__":
    main()
