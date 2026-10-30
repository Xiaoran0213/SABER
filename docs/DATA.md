# Data

SABER does not download data at training time. It reads a cells-by-proteins
`.h5ad` matrix and a precomputed protein-feature cache. Configure the roots
with environment variables (or `.saber_env`, see `env.example` at the
repository root); `saber/paths.py` is the single resolution point.

## Environment variables

| Variable | Required | What it points to |
|---|---|---|
| `SABER_DATA_ROOT` | yes | Proteomics data root: `anndata/` matrices, case-study folders (`easyscp_liver/`, `brunner2022/`), `Pretrained_embeddings/` FASTAs |
| `SABER_PLM_ROOT` | yes | Precomputed PLM caches, one subdir per model (`esm2_650m/`, ...) |
| `SABER_BENCHMARK_ROOT` | no | Benchmark slice copy (default: `$SABER_DATA_ROOT/saber_benchmark`) |
| `SABER_EXT_METHODS_ROOT` | no | External-baseline adapters (default: the repository's `external_adapters/`; see [external_adapters/README.md](../external_adapters/README.md)) |
| `SABER_EXTERNAL_REPOS` | no | Upstream clones of the external baselines (PIMMS, omicsGMF, scPROTEIN); needed only when running external baselines |
| `SABER_PROTAE_ROOT` | no | Legacy annotation source for `saber prepare-benchmark-data` when re-materializing the benchmark slice from the original evaluation tree |
| `SABER_OUTPUTS_ROOT` | no | Where training runs are written (default `./outputs`) |
| `SABER_FIGURES_ROOT` | no | Where figures are rendered (default `./figures`) |
| `SABER_HF_CACHE` | no | HuggingFace weight cache for rebuilding PLM features (default `$HF_HOME`) |
| `SABER_SAPROT_TOKENS` | no | Foldseek structure-token TSV for the SaProt prior (Table S5 ablation only) |

## Manuscript datasets

| Dataset | Accession / source | Role |
|---|---|---|
| Specht 2021 (SCoPE2) | Slavov Lab / SPDB | Benchmark: imputation, embeddings, expansion |
| Derks 2022 (plexDIA) | Slavov Lab / SPDB | Benchmark: imputation, embeddings, expansion |
| Huffman 2023 | Slavov Lab / SPDB | Benchmark: imputation, expansion |
| Woo 2021 | Slavov Lab / SPDB | Benchmark: imputation, expansion |
| PXD071205 | PRIDE | LFQ spike-in fold-change benchmark (Fig. 4) |
| EasySCP liver | PRIDE PXD070185 (+ Zenodo 10.5281/zenodo.19627167) | Liver-zonation case study (Fig. 2, 5a-d, S4) |
| Brunner 2022 | PRIDE PXD024043 | Cell-cycle case study (Fig. 5e-h, S10) |

The registry in `configs/benchmarks/saber_benchmark.yaml` lists exactly these
seven (`saber prepare-benchmark-data` materializes the slice). Dou 2019 and
Derks 2023 are not part of the manuscript; rows for them may still appear in
historical long CSVs and are filtered out by the figure scripts.

### Case-study input folders

- `$SABER_DATA_ROOT/easyscp_liver/` — `prepared/EasySCP_liver.h5ad` (608
  hepatocytes × 5,223 proteins, UniProt-AC var names, GSVA hepatocyte
  spatial-status score `hss_score`, FACS `gate` A-H), `prepared/
  EasySCP_liver_features.npz` (ESM-2 650M), `markers/` (canonical / hss /
  zonated tier tables + `*_targets.txt` hold-out lists + `axis_targets.txt`).
  These are built from the Zenodo/PRIDE download during dataset onboarding.
- `$SABER_DATA_ROOT/brunner2022/` — raw DIANN output
  (`DIANN1.8_SingleCells_CellCycle/`), `brunner_esm2_650m.npy`,
  `brunner_derisk2_results.csv` (AC↔gene + degron flags),
  `brunner_cc_score.npy`. `scripts/case_study/build_brunner_inputs.py`
  rebuilds `prepared/` and `markers/` from these.

## Input contract

- `adata.X`: cells × proteins. Finite entries are observed.
- `adata.obs[batch_key]`, `adata.obs[cell_type_key]`: optional labels.
- Protein features: `.npz` with `features` and `protein_ids`, or a matching
  `.npy` pair. Rows are aligned to the AnnData protein order.

See `saber/data.py` and `scripts/extract_plm_cache.py`.

## External baselines

The adapters for PIMMS, omicsGMF, scPROTEIN, and the kNN baseline ship
with the repository under `external_adapters/` — nothing to configure for
SABER itself. The upstream method implementations are not redistributed;
clone them into one directory, install their environments, and export
`SABER_EXTERNAL_REPOS` pointing at it (full instructions and pinned
versions: [external_adapters/README.md](../external_adapters/README.md)).
External baselines are optional and used only for the comparison rows of
the benchmark; `saber list-external-methods --check` verifies the setup.

## Shipped tables (`results/source_data/`)

| File | Content |
|---|---|
| `source_metrics_long.csv` | the 10-seed frozen-feature benchmark campaign (Figs. 3, 4, S1, S7, S8, S11-S13) |
| `source_metrics_grid_long.csv` | the missingness-grid arm (Fig. S2) |
| `lfq_species_only_rows.csv` | species-only LFQ control rows (Figs. 4, S9) |
| `source_proteome_wide_long.csv` | proteome-wide expansion + GO arm (Figs. S5, S6, S8 GO panels) |
| `fusion_frozen_long.csv` | fused ESM-2∥scGPT prior arm (Fig. S8 fused bars, Table S5) |
| `case_study/*.csv` | liver + Brunner metric tables (Figs. 2, 5, S10) |

`artifact_path` columns are relative to the campaign outputs root
(`seeds/seed<N>/runs/...`).
