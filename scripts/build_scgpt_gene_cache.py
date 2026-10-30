"""Build the scgpt_gene PLM cache for SABER (all NON_LFQ datasets).

gene-level prior from scGPT: each protein is mapped to its gene symbol (GN=
field of the UniProt proteome FASTAs), then assigned scGPT's INPUT embedding
(``encoder.embedding.weight`` row, 512d) for that gene. This mirrors how scGPT
embeddings are commonly used as static gene features. Row order is byte-aligned
with the esm2_650m cache protein_ids so ``prepare_aligned_inputs`` keeps the
same proteins -> fair prior ablation.

Usage:
    conda run -n saber python scripts/build_scgpt_gene_cache.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saber.paths import data_root, plm_root  # noqa: E402

PLM_ROOT = plm_root()
SCGPT_DIR = PLM_ROOT / "scgpt_gene"
FASTA_DIR = data_root() / "Pretrained_embeddings"
# cache_name per saber.full_benchmark.DATASET_PLM_CACHE_NAMES (NON_LFQ subset)
DATASETS = {
    "Specht_2019": "Specht_2019",
    "Derks_2022": "Derks_2022_3ct",
    "Derks_2023": "Derks_2023_filt",
    "Dou_2019": "Dou_2019_filt",
    "Huffman_2023_melanoma_HEK": "Huffman_mel_HEK_filt",
    "Woo_2021": "Woo_2021_filt",
}

GN_RE = re.compile(r"GN=([^\s]+)")


def build_uniprot_to_gene(fastas: list[Path]) -> dict[str, str]:
    m: dict[str, str] = {}
    for fasta in fastas:
        for line in fasta.open():
            if line.startswith(">"):
                acc = line.split("|")[1] if "|" in line else ""
                gn = GN_RE.search(line)
                if acc and gn:
                    m[acc] = gn.group(1)
    return m


def main(dry_run: bool = False) -> None:
    vocab = json.load(open(SCGPT_DIR / "vocab.json"))
    emb = np.load(SCGPT_DIR / "scgpt_gene_encoder.npy")  # (n_vocab, 512)
    pad_idx = vocab["<pad>"]
    pad_vec = emb[pad_idx]
    assert emb.shape[1] == 512

    u2g = build_uniprot_to_gene(
        [FASTA_DIR / "UP000005640_9606.fasta", FASTA_DIR / "UP000000589_10090.fasta"])

    out_dir = PLM_ROOT / "scgpt_gene"
    report = []
    for dataset, cache_name in DATASETS.items():
        ids_path = PLM_ROOT / "esm2_650m" / f"{cache_name}_protein_ids.npy"
        ref_ids = np.load(ids_path, allow_pickle=True)
        rows = np.empty((len(ref_ids), 512), dtype=np.float32)
        n_exact = n_upper = n_pad = n_nomap = 0
        pad_ids, nomap_ids = [], []
        genes_used = []
        for i, acc in enumerate(ref_ids):
            acc = str(acc)
            g = u2g.get(acc)
            if g is None:
                rows[i] = pad_vec; n_nomap += 1
                nomap_ids.append(acc); genes_used.append("<pad>")
                continue
            if g in vocab:
                rows[i] = emb[vocab[g]]; n_exact += 1
            elif g.upper() in vocab:  # mouse Titlecase -> HGNC ALLCAPS
                rows[i] = emb[vocab[g.upper()]]; n_upper += 1
            else:
                rows[i] = pad_vec; n_pad += 1
                pad_ids.append(acc)
            genes_used.append(g if g in vocab else (g.upper() if g.upper() in vocab else "<pad>"))

        cov = (n_exact + n_upper) / len(ref_ids)
        report.append({"dataset": dataset, "cache_name": cache_name,
                       "n_proteins": len(ref_ids), "exact": n_exact, "upper": n_upper,
                       "pad_no_vocab": n_pad, "no_gene_map": n_nomap,
                       "coverage": round(cov, 4),
                       "pad_ids": pad_ids, "nomap_ids": nomap_ids})
        print(f"{cache_name}: n={len(ref_ids)} exact={n_exact} upper={n_upper} "
              f"pad={n_pad} nomap={n_nomap} cov={cov:.2%}")
        if not dry_run:
            np.save(out_dir / f"{cache_name}_embeddings.npy", rows)
            np.save(out_dir / f"{cache_name}_protein_ids.npy",
                    np.array([str(a) for a in ref_ids], dtype=object))
            meta = {
                "plm_key": "scgpt_gene", "model": "scGPT whole-human (cellxgene census)",
                "embedding": "input gene_encoder (encoder.embedding.weight) row, 512d",
                "source_repo": "huggingface VirtualCell2025/scGPT_human (mirror of bowang-lab/scGPT_human)",
                "dim": 512,
                "vocab_size": len(vocab),
                "protein_to_gene": "UniProt FASTA GN= field (UP000005640_9606 + UP000000589_10090); "
                                   "mouse gene symbols upper-cased to match HGNC vocab",
                "fallback": "proteins whose gene is absent from vocab get the <pad> row",
                "pooling": "none (gene-level embedding; one gene per protein)",
                "dataset": dataset, "cache_name": cache_name,
                "row_order": "aligned to esm2_650m cache protein_ids",
                "coverage": report[-1],
            }
            (out_dir / f"{cache_name}_meta.json").write_text(json.dumps(meta, indent=2))
    if not dry_run:
        (out_dir / "BUILD_REPORT_scgpt.json").write_text(json.dumps(report, indent=2))
        print("wrote cache to", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    main(dry_run=args.dry_run)
