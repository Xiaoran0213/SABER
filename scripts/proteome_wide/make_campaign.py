#!/usr/bin/env python3
"""Materialize the SABER proteome-wide leave-large-fraction-out campaign.

Generates expansion configs + a train+eval command file + a sequential SLURM
sbatch + submit script, all under outputs/proteome-wide/. Consumes only the
frozen, bias-free `saber train-expansion` / `evaluate-expansion` CLI and the
6/14 prepared inputs. Does NOT touch source code.

Sweep: {Derks_2022 (main), Specht_2019 (aux)} x holdout {50%,80%,90%} x
{saber, no-PLM} x seeds (default 42,123,2024). Configs clone the canonical 6/14
expansion settings; only project.outdir/name, expansion.target_fraction, the
no-PLM toggle, and the seed differ. Multi-seed so the coherence / degradation
panels carry error bars.

Relocated from outputs/proteome-wide/scripts/ -> scripts/proteome_wide/ (code
lives with the repo; run artifacts stay under outputs/proteome-wide/).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from saber.paths import outputs_root  # noqa: E402

OUT = outputs_root() / "proteome-wide"
CONFIGS = OUT / "configs"
COMMANDS = OUT / "commands"
SLURM = OUT / "slurm"
RUNS = OUT / "runs"
RESULTS = OUT / "results"
LOGS = OUT / "logs" / "slurm"
PREP_DEFAULT = outputs_root() / "paper_benchmark_20260614_full" / "prepared_inputs"

DEFAULT_SEEDS = (42, 123, 2024)
DATASETS_DEFAULT = {
    "Derks_2022": "Derks_2022/Derks_2022.h5ad",
    "Specht_2019": "Specht_2019/Specht_2019.h5ad",
}
FRACTIONS = [("50", 0.5), ("80", 0.8), ("90", 0.9)]
CONDITIONS = ("saber", "noplm")

# Canonical 6/14 expansion training/model settings (bias-free: decoder z@W@e.T).
BASE_YAML = """\
project:
  name: {name}
  outdir: {outdir}
  seed: {seed}
data:
  synthetic: false
  path: {data_path}
  protein_features_path: {feat_path}
  protein_features_ids_path: ''
  protein_ids_path: ''
  batch_key: batch
  cell_type_key: cell_type
  protein_id_field: ''
  scale_mode: none
  allow_random_protein_features: {allow_random}
missing:
  enabled: true
  total_fraction: 0.5
  mnar_fraction: 0.5
  threshold_quantile: 0.1
  threshold_scale: 0.3
  seed: {seed}
model:
  latent_dim: 16
  ae_h1: 128
  ae_h2: 64
  protein_feature_dim: 1280
  protein_proj_dim: 256
  protein_proj_hidden_dim: 256
  protein_proj_dropout: 0.1
  bilinear_with_w: true
  trainable_protein_features: {trainable}
training:
  epochs: 300
  batch_size: 32
  lr: 0.0005
  weight_decay: 0.01
  val_frac: 0.05
  device: {device}
  lr_mult_protein_features: 1.0
  lr_mult_protein_encoder: 1.0
evaluation:
  imputation: true
expansion:
  target_fraction: {target_fraction}
  seed: {seed}
  min_target_observed: 0
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--prep-dir", default=str(PREP_DEFAULT), help="prepared_inputs dir (dataset h5ad + features)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--trainable-features", action="store_true",
                    help="fine-tune the protein features (default: frozen, the benchmark protocol)")
    ap.add_argument("--out-suffix", default="",
                    help="suffix for the output tree, appended to the proteome-wide name")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    prep = Path(args.prep_dir)

    global OUT, CONFIGS, COMMANDS, SLURM, RUNS, RESULTS, LOGS
    OUT = outputs_root() / f"proteome-wide{args.out_suffix}"
    CONFIGS = OUT / "configs"
    COMMANDS = OUT / "commands"
    SLURM = OUT / "slurm"
    RUNS = OUT / "runs"
    RESULTS = OUT / "results"
    LOGS = OUT / "logs" / "slurm"

    for d in (CONFIGS, COMMANDS, SLURM, RUNS, RESULTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)

    datasets = {ds: prep / rel for ds, rel in DATASETS_DEFAULT.items()}
    names: list[str] = []
    cmd_lines: list[str] = []
    for seed in seeds:
        for ds, h5ad in datasets.items():
            feat = prep / f"{ds}/{ds}_features.npz"
            for frac_str, frac in FRACTIONS:
                for cond in CONDITIONS:
                    name = f"{ds}_holdout{frac_str}_{cond}_seed{seed}"
                    names.append(name)
                    cfg_path = CONFIGS / f"{name}.yaml"
                    cfg_path.write_text(
                        BASE_YAML.format(
                            name=name,
                            outdir=str(RUNS),
                            data_path=str(h5ad),
                            feat_path=str(feat) if cond == "saber" else "''",
                            allow_random="false" if cond == "saber" else "true",
                            target_fraction=frac,
                            seed=seed,
                            device=args.device,
                            trainable="true" if args.trainable_features else "false",
                        )
                    )
                    manifest = RUNS / name / "artifact_manifest.json"
                    eval_out = RESULTS / name
                    cmd = (
                        f"{sys.executable} -m saber.cli train-expansion --config {cfg_path} "
                        f"&& {sys.executable} -m saber.cli evaluate-expansion "
                        f"--manifest {manifest} --outdir {eval_out}"
                    )
                    cmd_lines.append(cmd)

    (COMMANDS / "train_eval_commands.txt").write_text("\n".join(cmd_lines) + "\n")

    sbatch = SLURM / "run_sequential.sbatch"
    sbatch.write_text(
        """\
#!/usr/bin/env bash
#SBATCH --job-name=saber_proteomewide
#SBATCH --partition=stat
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output={logs}/saber_proteomewide_%j.out
#SBATCH --error={logs}/saber_proteomewide_%j.err
#SBATCH --gres=gpu:1
set -euo pipefail
CMD_FILE={cmds}
cd {repo}
i=0
while IFS= read -r CMD; do
  i=$((i + 1))
  [[ -z "${{CMD}}" ]] && continue
  echo "[proteome-wide] job ${{i}}/{n}: ${{CMD}}"
  eval "${{CMD}}"
done < "${{CMD_FILE}}"
echo "[proteome-wide] all {n} runs finished"
""".format(logs=str(LOGS), cmds=str(COMMANDS / "train_eval_commands.txt"), repo=str(REPO), n=len(cmd_lines))
    )
    sbatch.chmod(0o755)

    submit = OUT / "submit_proteome_wide.sh"
    submit.write_text(
        """\
#!/usr/bin/env bash
set -euo pipefail
JOB=$(sbatch --parsable {sbatch})
echo "${{JOB}}" > {out}/run_jobid.txt
echo "Submitted proteome-wide sequential job: ${{JOB}}"
squeue -j ${{JOB}}
""".format(sbatch=str(sbatch), out=str(OUT))
    )
    submit.chmod(0o755)

    print(f"Wrote {len(names)} configs to {CONFIGS}")
    print(f"Wrote command file ({len(cmd_lines)} lines) to {COMMANDS/'train_eval_commands.txt'}")
    print(f"Wrote sbatch: {sbatch}")
    print(f"Wrote submit: {submit}")
    print(f"Seeds: {seeds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
