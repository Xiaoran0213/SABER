# External-baseline adapters

Adapters for the external methods compared against SABER in the manuscript.
Each adapter is a thin command-line wrapper (CSV in, CSV out) called by
`saber.external_methods`; the upstream method implementations are **not**
redistributed here; clone them separately (see the table below).

SABER itself runs without any of this. The adapters are only needed to
reproduce the external-baseline rows of the benchmark (Tier 2, see
[docs/REPRODUCE.md](../docs/REPRODUCE.md)).

## Methods

| Baseline | Adapter | Upstream repository | Pinned commit | Capabilities |
|---|---|---|---|---|
| PIMMS (DAE/VAE) | `pimms/` | https://github.com/RasmussenLab/pimms | `327a431a` (2025-02-03) | imputation, LFQ |
| omicsGMF | `omicsgmf/` | https://github.com/statOmics/omicsGMF | `95d6026` (2025-10-16) | imputation, cell/protein embedding, LFQ |
| scPROTEIN | `scprotein/` | https://github.com/TencentAILabHealthcare/scPROTEIN | pip `scprotein==1.1.1` | cell embedding |
| kNN (k=5, Pearson) | `knn/` | self-contained (numpy only) | - | imputation, LFQ |

Cell-mean and protein-mean baselines are implemented directly in the
`saber` package (no adapter needed).

## Setup

1. Create the conda environments (PIMMS and omicsGMF ship an
   `environment.yml`; kNN runs in the main `saber` environment):

   ```bash
   conda env create -f external_adapters/pimms/environment.yml
   conda env create -f external_adapters/omicsgmf/environment.yml
   conda env create -f external_adapters/scprotein/environment.yml
   ```

2. Clone the upstream repositories into one directory and, where needed,
   install them into the matching environment:

   ```bash
   mkdir external_repos && cd external_repos
   git clone https://github.com/RasmussenLab/pimms.git        # pip install pimms-learn already in env
   git clone https://github.com/statOmics/omicsGMF.git
   conda run -n r_omicsgmf Rscript -e 'devtools::install("omicsGMF")'
   # scPROTEIN is installed from PyPI (scprotein==1.1.1) by its environment.yml
   ```

3. Point SABER at the clones (the adapters themselves are found in this
   directory automatically):

   ```bash
   export SABER_EXTERNAL_REPOS=$PWD/external_repos
   ```

   The harness passes the repository directory to every adapter as
   `--repo` and as the `EXTERNAL_REPO` environment variable; adapters use
   it only as a fallback import path.

4. Check the installation:

   ```bash
   saber list-external-methods --check
   ```

## Interface contract

`saber.external_methods.build_external_command` calls

```
[conda run -n <env>] python adapter.py|Rscript adapter.R \
    --in <matrix.csv> --out <prediction.csv> \
    [--repo <upstream clone>] [--config spec.json] [--model DAE|VAE] [--seed N]
```

Input is a headerless cells x proteins CSV with NaN for missing entries;
the output must be the imputed matrix of the same shape (omicsGMF
additionally writes a `<out>.protein.csv` sidecar with protein
embeddings). `spec.json` declares the conda environment and dependencies.

## Version note

The manuscript benchmark rows were produced with the upstream checkouts
available at the time of the campaigns. The omicsGMF clone was re-fetched
on 2025-10-16, after the benchmark campaigns completed; re-running
omicsGMF from a fresh clone may therefore deviate slightly from the
shipped tables. The PIMMS and scPROTEIN versions match the environments
recorded above.
