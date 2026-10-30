from __future__ import annotations

import csv
from pathlib import Path

from saber.paper_figures import make_paper_figures


def _write_source(path: Path) -> None:
    rows = [
        {
            "task": "imputation",
            "dataset": "Specht_2019",
            "method": "saber",
            "condition": "",
            "metric": "pearson_missing_mean",
            "value": "0.82",
            "missing_fraction": "0.4",
            "mnar_fraction": "0.5",
        },
        {
            "task": "imputation",
            "dataset": "Specht_2019",
            "method": "protein_mean",
            "condition": "",
            "metric": "pearson_missing_mean",
            "value": "0.55",
            "missing_fraction": "0.4",
            "mnar_fraction": "0.5",
        },
        {
            "task": "cell_embedding",
            "dataset": "Specht_2019",
            "method": "saber",
            "condition": "",
            "metric": "ARI_cell",
            "value": "0.61",
        },
        {
            "task": "cell_embedding",
            "dataset": "Specht_2019",
            "method": "saber",
            "condition": "",
            "metric": "NMI_cell",
            "value": "0.72",
        },
        {
            "task": "cell_embedding",
            "dataset": "Specht_2019",
            "method": "saber",
            "condition": "",
            "metric": "PS_cell",
            "value": "0.80",
        },
        {
            "task": "cell_embedding",
            "dataset": "Specht_2019",
            "method": "saber",
            "condition": "",
            "metric": "iLISI",
            "value": "1.40",
        },
        {
            "task": "cell_embedding",
            "dataset": "Specht_2019",
            "method": "saber",
            "condition": "",
            "metric": "kBET_reject_rate",
            "value": "0.20",
        },
        {
            "task": "protein_embedding",
            "dataset": "Specht_2019",
            "method": "saber",
            "condition": "",
            "metric": "hit_at_5",
            "value": "0.44",
            "protein_embedding_layer": "layer1",
        },
        {
            "task": "protein_embedding",
            "dataset": "Specht_2019",
            "method": "saber",
            "condition": "",
            "metric": "hit_at_10",
            "value": "0.58",
            "protein_embedding_layer": "layer1",
        },
        {
            "task": "protein_embedding",
            "dataset": "Specht_2019",
            "method": "saber",
            "condition": "",
            "metric": "string.edge_auroc",
            "value": "0.76",
            "protein_embedding_layer": "layer2_3",
        },
        {
            "task": "proteome_expansion",
            "dataset": "Specht_2019",
            "method": "saber",
            "condition": "",
            "metric": "pearson_target_mean",
            "value": "0.63",
            "target_fraction": "0.1",
        },
        {
            "task": "lfq",
            "dataset": "PXD071205_LFQ",
            "method": "saber",
            "condition": "",
            "metric": "mean_abs_error_yeast_ecoli",
            "value": "0.35",
            "lfq_subset": "has_missing",
        },
        {
            "task": "lfq",
            "dataset": "PXD071205_LFQ",
            "method": "saber",
            "condition": "",
            "metric": "pearson_log2fc",
            "value": "0.91",
            "lfq_subset": "has_missing",
        },
    ]
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_make_paper_figures_from_source_data(tmp_path: Path) -> None:
    source = tmp_path / "source_metrics_long.csv"
    _write_source(source)
    result = make_paper_figures(source, tmp_path / "figures", formats=("png",))
    assert Path(result["manifest"]).exists()
    assert not result["warnings"]
    files = [file for group in result["figures"].values() for file in group["files"]]
    assert files
    assert all(Path(file).exists() for file in files)
    assert (tmp_path / "figures" / "source_data" / "paper_imputation_pearson.csv").exists()
