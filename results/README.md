# results/

Shipped evaluation tables so every manuscript figure can be regenerated
without retraining (`python scripts/plot_all.py --run results`). Not model
weights — those stay in the campaign output trees.

| File | Content |
|---|---|
| `source_data/source_metrics_long.csv` | collected 10-seed frozen-feature benchmark campaign (Figs. 3, 4, S1, S7, S8, S11-S13) |
| `source_data/lfq_species_only_rows.csv` | species-only LFQ control rows (Figs. 4, S9) |
| `source_data/source_metrics_grid_long.csv` | missingness-grid arm (Fig. S2) |
| `source_data/source_proteome_wide_long.csv` | proteome-wide expansion + GO arm (Figs. S5, S6, S8 GO panels) |
| `source_data/fusion_frozen_long.csv` | fused ESM-2∥scGPT prior arm (Fig. S8 fused bars, Table S5) |
| `source_data/case_study/` | liver + Brunner metric tables (Figs. 2, 5, S10) |

`artifact_path` columns are relative to the campaign outputs root
(`seeds/seed<N>/runs/...`). Historical rows for datasets that are not part of
the manuscript (Dou 2019, Derks 2023) may be present; figure scripts filter
them out.
