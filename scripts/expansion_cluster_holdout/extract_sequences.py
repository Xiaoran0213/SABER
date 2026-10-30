#!/usr/bin/env python3
"""Extract per-dataset protein FASTA from the human proteome, by UniProt accession.

Reads protein accessions from each dataset's prepared h5ad (var_names) and pulls
sequences from the human reference proteome FASTA. Writes <ds>.fasta for mmseqs2.
Reports coverage (how many dataset proteins are found in the reference).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import anndata as ad

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from saber.paths import data_root, outputs_root  # noqa: E402

PREP = outputs_root() / "paper_benchmark_20260709_5seed" / "seeds" / "seed42" / "prepared_inputs"
HUMAN_FASTA = data_root() / "Pretrained_embeddings" / "UP000005640_9606.fasta"
OUT = outputs_root() / "expansion_cluster_holdout" / "fasta"


def parse_fasta_accessions(fasta: Path):
    """Yield (accession, header, [seq_lines]) for each record. accession = 2nd pipe field if present."""
    acc = header = None
    seq = []
    with fasta.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if acc is not None:
                    yield acc, header, "".join(seq)
                header = line[1:]
                # formats: "sp|P12345|NAME ..." / "tr|A0A024..|..." / ">A6NCW0 ..."
                toks = header.split()
                first = toks[0]
                parts = first.split("|")
                acc = parts[1] if len(parts) >= 3 else parts[0]
                seq = []
            else:
                seq.append(line)
    if acc is not None:
        yield acc, header, "".join(seq)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="Derks_2022,Specht_2019,Huffman_2023_melanoma_HEK,Dou_2019,Woo_2021")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"loading human proteome {HUMAN_FASTA.name} ...", flush=True)
    ref = {}
    for acc, header, seq in parse_fasta_accessions(HUMAN_FASTA):
        if acc not in ref:  # keep first
            ref[acc] = (header, seq)
    print(f"  reference records: {len(ref)}")

    for ds in args.datasets.split(","):
        ds = ds.strip()
        h5 = PREP / ds / f"{ds}.h5ad"
        if not h5.exists():
            print(f"[{ds}] SKIP (no prepared h5ad: {h5})")
            continue
        a = ad.read_h5ad(h5)
        ids = [str(x) for x in a.var_names]
        found = [x for x in ids if x in ref]
        miss = [x for x in ids if x not in ref]
        out_fa = OUT / f"{ds}.fasta"
        with out_fa.open("w") as f:
            for x in found:
                h, s = ref[x]
                f.write(f">{x} {h}\n{s}\n")
        print(f"[{ds}] {len(ids)} proteins | {len(found)} seqs found ({100*len(found)/len(ids):.1f}%) | {len(miss)} missing -> {out_fa}")
        if miss[:5]:
            print(f"        missing examples: {miss[:5]}")


if __name__ == "__main__":
    main()
