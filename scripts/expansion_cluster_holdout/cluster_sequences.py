#!/usr/bin/env python3
"""Run MMseqs2 sequence-identity clustering for the §2 cluster-holdout datasets.

For each human dataset x identity threshold, produces a TSV mapping
accession -> cluster_representative (UniRef-style: --min-seq-id + 0.8 mutual coverage).
This is the field-standard "sequence-cluster" definition (CD-HIT/MMseqs2), replacing
the degenerate ESM-2 cosine clustering.

Requires mmseqs2 (env: `mmseqs`). Run AFTER extract_sequences.py.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FASTA_DIR = REPO / "outputs" / "expansion_cluster_holdout" / "fasta"
OUT_DIR = REPO / "outputs" / "expansion_cluster_holdout" / "clusters"
DATASETS = ["Derks_2022", "Specht_2019", "Huffman_2023_melanoma_HEK"]
THRESHOLDS = [0.30, 0.50, 0.70]  # sequence identity
COVERAGE = 0.80  # mutual (bidirectional) coverage, --cov-mode 0


def run_mmseqs(fasta: Path, out_prefix: Path, min_id: float, mmseqs: str, threads: int) -> Path:
    tmp = out_prefix.parent / f"tmp_{out_prefix.name}"
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = [
        mmseqs, "easy-cluster", str(fasta), str(out_prefix), str(tmp),
        "--min-seq-id", str(min_id), "-c", str(COVERAGE), "--cov-mode", "0",
        "--threads", str(threads), "-s", "7.5",
    ]
    subprocess.run(cmd, check=True)
    tsv = Path(str(out_prefix) + "_cluster.tsv")
    return tsv


def summarize(tsv: Path) -> dict:
    clusters = {}
    with tsv.open() as f:
        for line in f:
            rep, member = line.strip().split("\t")[:2]
            clusters.setdefault(rep, []).append(member)
    sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    return {
        "n_clusters": len(clusters),
        "n_sequences": sum(sizes),
        "largest": sizes[0] if sizes else 0,
        "singletons": sum(1 for s in sizes if s == 1),
        "top5": sizes[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mmseqs", default="mmseqs")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        fa = FASTA_DIR / f"{ds}.fasta"
        if not fa.exists() or fa.stat().st_size == 0:
            print(f"[{ds}] SKIP (no FASTA)")
            continue
        for tid in THRESHOLDS:
            tag = f"id{int(round(tid*100)):02d}"
            prefix = OUT_DIR / f"{ds}_{tag}"
            tsv = run_mmseqs(fa, prefix, tid, args.mmseqs, args.threads)
            s = summarize(tsv)
            print(f"[{ds} {tag}] clusters={s['n_clusters']:4d} seqs={s['n_sequences']:4d} "
                  f"largest={s['largest']:3d} singletons={s['singletons']:4d} top5={s['top5']}")


if __name__ == "__main__":
    main()
