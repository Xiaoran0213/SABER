# Reproduction guide

Every manuscript figure regenerates from this repository. Two levels:

- **Tier 1 — figures from the shipped tables (no training).** The repository
  ships the collected evaluation tables in `results/source_data/`
  (`source_metrics_long.csv` = the 10-seed frozen-feature benchmark campaign;
  `source_metrics_grid_long.csv` = the missingness-grid arm;
  `fusion_frozen_long.csv` = the fused-prior arm; `case_study/` = the
  liver/Brunner metric tables). Most panels render directly from these.
- **Tier 2 — full retraining.** Re-run the training campaigns with the scripts
  under `scripts/` and regenerate everything from your own outputs.

Configure the roots first (see `env.example` / `docs/DATA.md`): either export
`SABER_DATA_ROOT`, `SABER_PLM_ROOT`, ... or copy `env.example` to `.saber_env`.

## Tier 1 — regenerate figures from shipped tables

```bash
# benchmark panels (Figs. 3, 4, S1-S3, S7-S9, S11-S13) from the shipped tables
python scripts/plot_all.py --run results

# liver dataset overview (Fig. S4); needs SABER_DATA_ROOT
python scripts/figures/figS4_liver_overview.py
```

The species-only LFQ control rows (Figs. 4 / S9) are already merged into the
shipped `source_metrics_long.csv`; regenerating them from manifests is a
Tier-2 step (see below).

`--run results` resolves to `results/source_data/*.csv` in place of a campaign
tree. Figures + per-panel source CSVs land under `./figures/` (override with
`SABER_FIGURES_ROOT`). Panels that additionally read run artifacts (Fig. 4d
per-condition-pair details; the case-study panels that read `expansion.npz` /
`matrices.npz`) need Tier-2 outputs — the per-panel map below says which.

Notes:
- Fig. 4a-c read the long CSV; panel 4d re-evaluates LFQ manifests per seed
  and therefore needs the campaign tree.
- `supp_figures_v3.py` is the current SI batch renderer. Its long tables come
  from the three campaign runs (`final`, `grid`, `h0709`) and it reuses the
  per-panel source CSVs written by the other figure scripts; run
  `plot_all.py` first when regenerating the full SI suite.

## Tier 2 — full campaigns

### 0. Data + caches

```bash
# one-time: materialize the benchmark slice (copies h5ad + annotations)
saber prepare-benchmark-data

# one-time: per-protein ESM-2 caches (GPU; other PLMs for the Table S5 ablation)
python scripts/extract_plm_cache.py --plm esm2_650m --dataset Derks_2022
bash scripts/run_all_plm_cache.sh          # all datasets x sequence PLMs
```

### 1. Main benchmark campaign (Figs. 3, 4, S1-S3, S7-S9, S11-S13; Table S6)

```bash
python scripts/run_all.py --tag review --seeds 42,123,2024,2025,7,2026,2027,2028,2029,2030 \
       --missingness-grid --sequence-holdout        # SLURM farm: train -> externals -> collect
python scripts/plot_all.py --run paper_benchmark_<date>_review \
       --grid-run paper_benchmark_<date>_review_grid
```

- `run_all.py` materializes one job per (seed × dataset × task × condition)
  and submits a farm; the collect stage writes
  `<run>/source_data/source_metrics_long.csv`.
- The missingness-grid arm (`--missingness-grid`) and the sequence-cluster
  hold-out arm (`--sequence-holdout`) go into a sibling `<run>_grid` tree.
  Point `--grid-run` at it for the S2/S3 panels.
- The sequence arm needs MMseqs2 clusters first:

```bash
python scripts/expansion_cluster_holdout/extract_sequences.py
python scripts/expansion_cluster_holdout/cluster_sequences.py   # needs mmseqs
```

- Fig. 4's "Species only" bars come from one extra step after collect:

```bash
python scripts/figures/lfq_species_only.py --run <run> --merge
```

- External baselines (PIMMS-DAE/VAE, omicsGMF, scProtein) run inside the farm
  via the adapters in `external_adapters/` (upstream clones +
  `SABER_EXTERNAL_REPOS` required; see `external_adapters/README.md`).

### 2. Proteome-wide arm (Figs. S5, S6; GO panels of S8)

```bash
python scripts/proteome_wide/make_campaign.py          # materialize + sbatches
# ... run the farm, then:
python scripts/proteome_wide/go_coherence.py           # GO enrichment per run
python scripts/collect_proteome_wide.py                # -> source_proteome_wide_long.csv
python scripts/proteome_wide/plot_coherence.py         # Fig. S6
python scripts/proteome_wide/plot_degradation.py       # Fig. S5 companion
```

### 3. Liver-zonation case study (Figs. 2, 5a-d, S4; Notes 5.x)

Inputs: `$SABER_DATA_ROOT/easyscp_liver/prepared/` (h5ad + ESM-2 features +
marker tiers — built during dataset onboarding; see `docs/DATA.md`).

```bash
# (a) expansion runs: 3 tiers x {saber, noplm} x 10 seeds, incl. the
#     tier-union-axis variants (canonicalax / zonatedax)
python scripts/case_study/run_case_study.py --kind expansion

# (b) zonation metrics (Fig. 2a-c)  [default = axis-holdout protocol]
python scripts/case_study/zonation_expansion_metrics.py

# (c) Fig. 2 panels (2a-c + marker trajectories)
python scripts/figures/fig2_v2_expansion.py

# (d) real-imputation chain (Fig. 5a)
python scripts/case_study/realimpute_axisholdout.py --dataset liver
python scripts/case_study/run_externals_reduced.py liver       # omicsGMF/PIMMS, 3 seeds
python scripts/case_study/run_impute_externals.py              # full-matrix externals (Fig. 5b-d)
python scripts/case_study/liver_axisholdout_metrics.py

# (e) protein/cell embedding chains (Fig. 5b-d; Note 5 cell-emb null)
python scripts/case_study/run_case_study.py --kind cellembed   # 10 seeds x {saber, noplm}
python scripts/case_study/proteinemb_metrics.py
python scripts/case_study/cellembed_zonation_metrics.py
python scripts/figures/fig5_liver_casestudy.py                 # panels 5a-d
python scripts/figures/figS4_liver_overview.py                 # Fig. S4
```


### 4. Brunner cell-cycle case study (Figs. 5e-h, S10)

```bash
python scripts/case_study/build_brunner_inputs.py    # one-time: prepared h5ad + markers
python scripts/case_study/realimpute_axisholdout.py --dataset brunner   # 10 seeds + zero-shot
python scripts/case_study/run_externals_reduced.py brunner
python scripts/case_study/brunner_metrics_axisholdout.py
python scripts/figures/fig5ef_brunner_axisholdout.py      # panels 5e-h
python scripts/figures/figS11_brunner_per_substrate.py    # Fig. S10 (per-substrate deltas)
```

### 5. Prior-choice ablation (Table S5)

```bash
python scripts/build_scgpt_gene_cache.py             # scGPT gene prior cache
python scripts/run_prior_ablation.py materialize   # generate plan + jobs
bash <plan_root>/submit_prior_ablation.sh            # SLURM
python scripts/run_prior_ablation.py collect   --root <plan_root>
python scripts/run_prior_ablation.py compare --root <plan_root>
```

SaProt additionally needs `SABER_SAPROT_TOKENS` (foldseek structure-token TSV;
see `docs/DATA.md`).

## Frozen prior

Cached ESM-2 650M embeddings are precomputed once
(`model.trainable_protein_features` defaults to `false`); the ESM-2
transformer itself is never updated.
