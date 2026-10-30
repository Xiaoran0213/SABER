# SABER

SABER: sequence-anchored bilinear expression reconstruction for imputing and expanding mass-spectrometry single-cell proteomes

SABER reconstructs each protein abundance as a bilinear interaction between a
learned cell state and a protein embedding derived from ESM-2. One fitted model
returns imputed expression, cell embeddings, protein embeddings, and zero-shot
predictions for proteins held out of training.

This repository contains the implementation accompanying the manuscript.

## Install

```bash
conda env create -f environment.yml
conda activate saber
pip install -e ".[dev,figures]"
```

## Configure the data roots

Processed matrices, ESM-2 caches, and run outputs are not stored in this
repository. Copy `env.example` to `.saber_env` (auto-loaded, gitignored) or
export the variables:

```bash
export SABER_DATA_ROOT=/path/to/proteomics
export SABER_PLM_ROOT=/path/to/precomputed_plm
```

Optional: `SABER_EXTERNAL_REPOS` (upstream clones of the external
baselines; the adapters themselves ship under `external_adapters/`, see
[external_adapters/README.md](external_adapters/README.md)),
`SABER_OUTPUTS_ROOT` (default `./outputs`), `SABER_FIGURES_ROOT` (default
`./figures`). Full contract: [docs/DATA.md](docs/DATA.md).

## Regenerate the figures

From the shipped tables, no training (Figs. 3, 4, S1-S3, S7-S9, S11-S13):

```bash
python scripts/plot_all.py --run results
```

Full campaigns (liver + Brunner case studies, missingness grid, proteome-wide,
prior ablation): [docs/REPRODUCE.md](docs/REPRODUCE.md) gives every chain,
per figure, with [docs/FIGURE_MAP.md](docs/FIGURE_MAP.md) as the panel-level
map.

## Train

```bash
saber train --config configs/templates/imputation_base.yaml
saber train-expansion --config configs/templates/expansion_base.yaml
```

Case studies use `configs/case_study/` + `scripts/case_study/`:

- Liver zonation (Fig. 2, 5a-d): `run_case_study.py` (expansion tiers incl.
  the axis-holdout variants, then cellembed), then the metrics scripts.
- Brunner cell cycle (Fig. 5e-h, S10): the axis-holdout chain
  `realimpute_axisholdout.py --dataset brunner` → `brunner_metrics_axisholdout.py`
  → `scripts/figures/fig5ef_brunner_axisholdout.py`.


## Tests

```bash
pytest -q
```

These check model shapes, the frozen-prior contract, config rules, evaluation
helpers, and the environment-variable path contract. They are not manuscript
results.

## Repository layout

```text
saber/                 model, training, data, evaluation, CLI, portable paths
configs/benchmarks/    dataset registry (7 manuscript datasets)
configs/templates/     imputation and expansion YAML
configs/case_study/    liver + Brunner configs (per-seed variants are generated)
scripts/               campaign drivers, case-study chains, figure scripts
scripts/figures/       manuscript figure scripts (common.py = shared loaders)
external_adapters/     external-baseline adapters (PIMMS, omicsGMF, scPROTEIN, kNN)
results/source_data/   shipped evaluation tables (figures without retraining)
docs/                  data contract, figure map, reproduction guide
tests/                 unit tests
```
