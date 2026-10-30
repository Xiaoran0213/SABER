# Figure map

Numbering follows the current manuscript (`paper.tex` / `supplementary.tex`).
"Tier" = which inputs a panel needs: **tables** = shipped
`results/source_data/` CSVs; **runs** = campaign/case-study outputs
(`$SABER_OUTPUTS_ROOT`). See `docs/REPRODUCE.md` for the full chains.

## Main figures

| Panel | Content | Script | Tier |
|---|---|---|---|
| Fig. 1 | Architecture | static illustration (not generated) | — |
| Fig. 2a-c | Liver expansion: per-protein Pearson / polarity / violin by marker tier | `scripts/case_study/zonation_expansion_metrics.py` → `scripts/figures/fig2_v2_expansion.py` | runs |
| Fig. 2 (traj.) | CYP1A2 / PCK1 gate trajectories + zonated violin panels | same script (same invocation) | runs |
| Fig. 3 | Imputation under injected missingness | `scripts/figures/imputation_figures.py` | tables (Fig 3a-b, f-i need run artifacts for the expression maps) |
| Fig. 4 | LFQ fold-change recovery | `scripts/figures/lfq_figures.py` (+ `lfq_species_only.py --merge` for the "Species only" bars) | tables (4d needs runs) |
| Fig. 5a-d | Liver real imputation + protein embedding | `scripts/figures/fig5_liver_casestudy.py` (metrics from `scripts/case_study/{liver_axisholdout,proteinemb,cellembed}_*.py`) | runs |
| Fig. 5e-h | Brunner cell-cycle imputation | `scripts/case_study/realimpute_axisholdout.py --dataset brunner` → `brunner_metrics_axisholdout.py` → `scripts/figures/fig5ef_brunner_axisholdout.py` | runs |

## Supplementary figures

| Panel | Content | Script | Tier |
|---|---|---|---|
| Fig. S1 | Sequence-prior contribution at 50/80% missingness | `scripts/figures/imputation_figures.py` | tables |
| Fig. S2 | Missingness × MNAR grid | `scripts/figures/missingness_grid_figures.py --csv <grid-run>/source_data/source_metrics_long.csv` | tables (grid) |
| Fig. S3 | Sequence-cluster (MMseqs2) hold-out | `scripts/expansion_cluster_holdout/{extract_sequences,cluster_sequences}.py` → `scripts/figures/expansion_holdout_figures.py --csv ...` | runs |
| Fig. S4 | Liver dataset overview | `scripts/figures/figS4_liver_overview.py` | data root |
| Fig. S5 | Proteome-wide expansion degradation | `scripts/proteome_wide/{make_campaign,go_coherence}.py` + `scripts/collect_proteome_wide.py` → `scripts/figures/expansion_figures.py` / `proteome_wide/plot_degradation.py` | runs |
| Fig. S6 | GO coherence of expansion | `scripts/proteome_wide/plot_coherence.py` | runs |
| Fig. S7 | Cell-embedding ARI/iLISI summary | `scripts/figures/representation_figures.py` | tables |
| Fig. S8 | Random hold-out expansion (+ fused-prior bars) | `scripts/figures/expansion_figures.py` | tables |
| Fig. S9 | LFQ species-only control | `scripts/figures/lfq_species_only.py` (rows already merged in the shipped long CSV) | tables |
| Fig. S10 | Brunner per-substrate deltas | `scripts/figures/figS11_brunner_per_substrate.py` (filename keeps the historical S11 label) | runs |
| Fig. S11 | Imputation vs fraction, Spearman | `scripts/figures/imputation_figures.py` | tables |
| Fig. S12 | Imputation at 80%, alternative metrics | `scripts/figures/imputation_figures.py` | tables |
| Fig. S13 | Observed vs missing reconstruction fidelity | `scripts/figures/observed_reconstruction_figures.py` | runs |

## Tables

| Table | Content | Producer |
|---|---|---|
| S1-S4 | datasets / coverage / hyperparameters / software | manuscript (sources: registry YAML, `saber/`, environment) |
| S5 | prior-choice ablation | `scripts/run_prior_ablation.py` + `scripts/build_scgpt_gene_cache.py` |
| S6 | paired Wilcoxon statistics | per-figure source CSVs (e.g. `figures/benchmark/source_data/figures/imputation/`) |

## Batch rendering

- `scripts/plot_all.py --run <run>` — the benchmark panel groups
  (Figs. 3, 4, S2, S3, S7, S8 + efficiency); `--run results` reads the shipped
  tables.
- `scripts/figures/supp_figures_v3.py` — the re-rendered SI suite (S1-S3,
  S5-S13 under the `supp_v3` naming: file `figS<k>` = SI number k-1 for k ≥ 6;
  see the script docstring). It reuses the per-panel source CSVs written by
  the scripts above, so run `plot_all.py` first for a full SI regeneration.

Figures and their per-panel source CSVs are written under
`$SABER_FIGURES_ROOT` (default `./figures/`): `benchmark/` for the campaign
panels, `fig2/`, `fig5/`, `supp/` for the case-study and SI panels.
