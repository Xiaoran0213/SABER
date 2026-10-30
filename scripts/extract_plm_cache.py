#!/usr/bin/env python
"""
Fresh, self-contained per-protein PLM embedding extractor for SABER benchmarks.

Produces caches in the SAME on-disk format as the existing ESM-2 650M caches:
    <out_root>/<plm_key>/<cache_name>_embeddings.npy   float32 [N_proteins, D]
    <out_root>/<plm_key>/<cache_name>_protein_ids.npy  object  [N_proteins]  (UniProt AC)
    <out_root>/<plm_key>/<cache_name>_meta.json         build report

Design choices (agreed with user, 2026-07-09):
  * No dependency on protae4 / ProtAE_4.0. Pure HuggingFace `transformers`.
  * All PLMs use the SAME pooling recipe so caches are mutually comparable:
      attention-mask-weighted mean over RESIDUE positions (special tokens dropped).
  * protein_ids are UNIQUE (var first-occurrence order); duplicate var_names in
    an .h5ad collapse to one row (downstream prepare.py merges duplicates anyway).
  * Proteins with no sequence in the local FASTA are DROPPED and listed in meta
    (matches prepare.py `drop_missing_proteins=True`).
  * Long sequences are truncated to each model's max length; truncated proteins
    are counted and reported.

Example:
  python extract_plm_cache.py --plm prot_t5_xl --dataset Specht_2019 \
      --out-root $SABER_PLM_ROOT --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# PLM specs (dim/family/hf_id/max_len/pooling) live in a single registry in the
# saber package so the extractor and the prior-ablation driver stay in sync.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saber.plms import PLM_REGISTRY as PLMS  # noqa: E402
from saber.paths import data_root, hf_cache, plm_root  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths / registries
# --------------------------------------------------------------------------- #
DATA_ROOT = data_root() / "anndata"
FASTA = {
    "human": data_root() / "Pretrained_embeddings" / "UP000005640_9606.fasta",
    "mouse": data_root() / "Pretrained_embeddings" / "UP000000589_10090.fasta",
}
HF_CACHE = hf_cache()

# cache_name matches saber.full_benchmark.DATASET_PLM_CACHE_NAMES (drop-in).
DATASETS: dict[str, dict[str, str]] = {
    "Specht_2019":              {"h5ad": "Specht_2019.h5ad",                                  "cache_name": "Specht_2019",          "species": "human"},
    "Derks_2022":               {"h5ad": "Derks_2022_3ct.h5ad",                               "cache_name": "Derks_2022_3ct",       "species": "human"},
    "Huffman_2023_melanoma_HEK":{"h5ad": "filtered_plm/Huffman_2023_melanoma_HEK_plmfiltered.h5ad", "cache_name": "Huffman_mel_HEK_filt", "species": "human"},
    "Woo_2021":                 {"h5ad": "filtered_plm/Woo_2021_plmfiltered.h5ad",            "cache_name": "Woo_2021_filt",        "species": "mouse"},
    "EasySCP_liver":            {"h5ad": "EasySCP_liver.h5ad",                                 "cache_name": "EasySCP_liver",        "species": "mouse"},
    "Brunner2022":              {"h5ad": "Brunner_2022.h5ad",                                  "cache_name": "Brunner2022",          "species": "human"},
    # NOTE: PXD071205_LFQ (Fig. 4) is not extractable here: it mixes human,
    # yeast and E. coli proteins, and its cache (PXD071205_Full_All) ships
    # prebuilt in $SABER_PLM_ROOT. The campaign consumes that cache directly.
}

# PLM specs are imported above as `PLMS` from saber.plms.PLM_REGISTRY (single
# source of truth). To add a PLM, edit saber/plms.py, then run this extractor
# once per dataset to build its cache.


# --------------------------------------------------------------------------- #
# FASTA + h5ad
# --------------------------------------------------------------------------- #
def load_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur is not None:
                    seqs[cur] = "".join(buf)
                parts = line[1:].split("|")
                cur = parts[1] if len(parts) > 1 else line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if cur is not None:
        seqs[cur] = "".join(buf)
    return seqs


def read_var_protein_ids(h5ad_path: Path) -> list[str]:
    import anndata as ad
    adata = ad.read_h5ad(h5ad_path)
    return [str(x) for x in adata.var_names]


# --------------------------------------------------------------------------- #
# Model loading + encoding
# --------------------------------------------------------------------------- #
def _to(seq: str, spec: dict[str, Any]) -> str:
    """Apply family-specific pre-tokenization (space join / rare-AA map)."""
    if spec.get("rare_aa_map"):
        # ProtBert can't tokenise ambiguous/rare residues; map to X.
        keep = set("ACDEFGHIKLMNPQRSTVWY")
        seq = "".join(c if c in keep else "X" for c in seq)
    if spec.get("space_join"):
        return " ".join(list(seq))
    return seq


def load_model(spec: dict[str, Any], device: str):
    """Return (model, tokenizer, family). Raises with a clear message on failure."""
    import torch
    from transformers import AutoTokenizer

    hf_id = spec["hf_id"]
    family = spec["family"]

    # T5/Ankh use sentencepiece; the fast-tokenizer converter imports
    # sentencepiece_model_pb2 which breaks on new protobuf. ProtT5's slow
    # tokenizer works; Ankh-Large has no slow vocab, so it needs fast (works
    # under PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python).
    if family == "t5":
        use_fast = False
    elif family == "ankh":
        use_fast = True
    else:
        use_fast = True
    tok = AutoTokenizer.from_pretrained(hf_id, cache_dir=str(HF_CACHE), use_fast=use_fast)
    if family in ("t5", "ankh"):
        from transformers import T5EncoderModel
        # fp16 is fine for ProtT5-XL but overflows ankh-large's activations
        # (-> all-NaN embeddings); keep ankh in fp32 on GPU.
        base = T5EncoderModel.from_pretrained(hf_id, cache_dir=str(HF_CACHE))
        if device.startswith("cuda") and family == "t5":
            base = base.half()
        model = base.to(device).eval()
    elif family == "esm2":
        from transformers import EsmModel
        model = EsmModel.from_pretrained(hf_id, cache_dir=str(HF_CACHE)).to(device).eval()
    elif family == "saprot":
        # SaProt checkpoints ARE ESM-2 architectures with a pair-token vocab
        # (AA+3Di). The HF repo ships ONLY pytorch_model.bin, which this
        # transformers version refuses (torch<2.6 CVE gate) — so load manually
        # from a safetensors copy written next to the snapshot (one-time
        # conversion from the .bin; tied lm_head.decoder dropped).
        import torch
        from safetensors.torch import load_file, save_file
        from transformers import EsmConfig, EsmModel
        repo_dir = hf_id.replace("/", "--")
        cands = list(HF_CACHE.glob(f"hub/models--{repo_dir}/snapshots/*")) or \
            list(HF_CACHE.glob(f"models--{repo_dir}/snapshots/*"))
        if not cands:
            raise FileNotFoundError(
                f"saprot snapshot not found under {HF_CACHE} "
                f"(hub/models--{repo_dir}/snapshots/*)")
        snap = cands[0]
        st = snap / "model.safetensors"
        if not st.exists():
            sd_bin = torch.load(snap / "pytorch_model.bin", map_location="cpu", weights_only=True)
            sd_bin = {k: v.contiguous() for k, v in sd_bin.items() if k != "lm_head.decoder.weight"}
            save_file(sd_bin, str(st), metadata={"format": "pt"})
        model = EsmModel(EsmConfig.from_pretrained(hf_id, cache_dir=str(HF_CACHE)))
        sd = load_file(str(st))
        model.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("esm.")},
                              strict=False)  # pooler/contact heads unused
        model = model.to(device).eval()
    elif family == "esmc":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(hf_id, cache_dir=str(HF_CACHE)).to(device).eval()
    elif family == "bert":
        from transformers import BertModel
        model = BertModel.from_pretrained(hf_id, cache_dir=str(HF_CACHE)).to(device).eval()
    else:
        raise ValueError(f"Unknown family {family}")
    return model, tok, family


def _residue_mask(attn_mask: "torch.Tensor", specials: tuple[str, ...]) -> "torch.Tensor":
    """attention_mask with the BOS/CLS (first) and EOS/SEP (last) real tokens zeroed."""
    import torch
    mask = attn_mask.clone().bool()
    if not mask.any():
        return mask
    lengths = mask.sum(dim=1)                       # real-token count per row
    rows = torch.arange(mask.shape[0], device=mask.device)
    if any(s in {"bos", "cls"} for s in specials):
        mask[rows, 0] = False                       # drop leading special
    if any(s in {"eos", "sep"} for s in specials):
        last_idx = lengths - 1
        mask[rows, last_idx] = False                # drop trailing special
    return mask


def encode_dataset(model, tok, family: str, spec: dict[str, Any],
                   ids: list[str], seqs_map: dict[str, str],
                   device: str, batch_tokens: int) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Mean-pool per protein over residues. Returns (features, kept_ids, stats)."""
    import torch

    kept_ids: list[str] = []
    records: list[tuple[str, str]] = []  # (id, prepared_seq_string)
    n_trunc = 0
    is_pair = family == "saprot"  # each residue = 2 chars (AA + 3Di)
    n_specials = len(spec["drop_specials"])
    budget = spec["max_len"] - n_specials
    for pid in ids:
        seq = seqs_map.get(pid)
        if not seq:
            continue  # missing -> dropped (reported by caller)
        n_units = len(seq) // 2 if is_pair else len(seq)
        if n_units > budget:
            seq = seq[: 2 * budget] if is_pair else seq[:budget]
            n_trunc += 1
        records.append((pid, _to(seq, spec)))
        kept_ids.append(pid)

    specials = spec["drop_specials"]
    out = np.empty((len(records), spec["dim"]), dtype=np.float32)
    pos = 0
    n_batches = 0

    # Generic HF path (all families, incl. Ankh via T5EncoderModel):
    # pack by approximate token budget, mask-weighted mean over residues.
    i = 0
    while i < len(records):
        batch: list[tuple[str, str]] = []
        cur_max = 1
        while i < len(records):
            unit = 2 if is_pair else 1
            L = len(records[i][1].replace(" ", "")) // unit + n_specials
            next_max = max(cur_max, L)
            if batch and next_max * (len(batch) + 1) > batch_tokens:
                break
            batch.append(records[i]); cur_max = next_max; i += 1
        pids_b = [b[0] for b in batch]; strs_b = [b[1] for b in batch]
        enc = tok(strs_b, padding=True, truncation=True, max_length=spec["max_len"], return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            emb = model(**enc).last_hidden_state
        rmask = _residue_mask(enc["attention_mask"], specials)
        denom = rmask.sum(dim=1).clamp(min=1).unsqueeze(-1).float()
        pooled = (emb.float() * rmask.unsqueeze(-1).float()).sum(dim=1) / denom
        out[pos:pos + pooled.shape[0]] = pooled.cpu().numpy()
        pos += pooled.shape[0]; n_batches += 1

    # Guard: fail loudly rather than silently write an all-NaN cache (e.g. ankh
    # overflowing in fp16 produced all-NaN embeddings before this check existed).
    n_nan = int(np.isnan(out).sum())
    if n_nan:
        raise RuntimeError(
            f"[extract] {family}: {n_nan}/{out.size} NaN embeddings "
            f"({100*n_nan/out.size:.1f}%) -- model/precision problem; refusing to write a NaN cache."
        )
    return out, kept_ids, {"truncated": n_trunc, "batches": n_batches}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plm", required=True, choices=list(PLMS))
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--out-root", default=str(plm_root()))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-tokens", type=int, default=8192, help="approx max tokens per batch")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    spec = PLMS[args.plm]
    ds = DATASETS[args.dataset]
    h5ad = DATA_ROOT / ds["h5ad"]
    cache_name = ds["cache_name"]
    out_dir = Path(args.out_root) / args.plm
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / f"{cache_name}_embeddings.npy"
    ids_path = out_dir / f"{cache_name}_protein_ids.npy"
    meta_path = out_dir / f"{cache_name}_meta.json"

    if emb_path.exists() and not args.overwrite:
        print(f"[skip] {emb_path} exists (use --overwrite)")
        return

    t0 = time.time()
    # 1. sequences
    if spec["family"] == "saprot":
        # SaProt takes foldseek AA+3Di pairs (structure-derived, so AA/3Di are
        # aligned by construction); the FASTA proteome is not used.
        tokens_value = spec.get("foldseek_tokens") or os.environ.get("SABER_SAPROT_TOKENS", "")
        if not tokens_value:
            raise SystemExit(
                "SaProt extraction needs a foldseek structure-token TSV. Export "
                "SABER_SAPROT_TOKENS=/path/to/foldseek_tokens.tsv (acc\tAA\t3DI; "
                "built with `foldseek structure2seq` against AlphaFold DB PDBs — "
                "see docs/DATA.md)."
            )
        tokens_path = Path(tokens_value)
        print(f"[saprot] loading foldseek tokens {tokens_path} ...")
        seqs_map = {}
        with tokens_path.open() as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                acc, aa, ss = parts
                if len(aa) != len(ss):
                    continue
                seqs_map[acc] = "".join(a + b for a, b in zip(aa, ss))
        print(f"[saprot] {len(seqs_map)} proteins with AA+3Di tokens")
    else:
        print(f"[fasta] loading {ds['species']} proteome ...")
        seqs_map = load_fasta(FASTA[ds["species"]])
        print(f"[fasta] {len(seqs_map)} entries")

    # 2. protein ids from h5ad in var order (duplicates PRESERVED to match the
    #    existing ESM-2 650M cache format; downstream prepare.py takes the first
    #    occurrence of each AC). Proteins with no sequence are dropped (below).
    var_ids = read_var_protein_ids(h5ad)
    dup_acs = sorted({pid for pid in var_ids if var_ids.count(pid) > 1})
    n_dup_positions = len(var_ids) - len(set(var_ids))
    unique_var_ids = list(dict.fromkeys(var_ids))           # preserve order, unique
    missing = [pid for pid in unique_var_ids if pid not in seqs_map]

    # 3. model + encode
    print(f"[model] loading {args.plm} ({spec['hf_id']}) ...")
    model, tok, family = load_model(spec, args.device)
    print(f"[model] loaded family={family}")

    features, kept_ids, stats = encode_dataset(
        model, tok, family, spec, var_ids, seqs_map, args.device, args.batch_tokens
    )
    assert features.shape == (len(kept_ids), spec["dim"]), (features.shape, spec["dim"])

    # 4. save
    np.save(emb_path, features.astype(np.float32))
    np.save(ids_path, np.asarray(kept_ids, dtype=object))
    dt = time.time() - t0

    meta = {
        "plm_key": args.plm, "hf_id": spec["hf_id"], "family": spec["family"],
        "dim": int(spec["dim"]),
        "pooling": "attention_mask_weighted_mean_over_residues_specials_dropped",
        "max_length": int(spec["max_len"]),
        "truncation": "residues truncated to max_length (count below)",
        "duplicate_policy": "var order preserved, duplicates kept (downstream prepare.py takes first occurrence)",
        "missing_policy": "proteins with no FASTA sequence are dropped and listed below",
        "dataset": args.dataset, "cache_name": cache_name, "species": ds["species"],
        "fasta_source": str(FASTA[ds["species"]]),
        "h5ad_source": str(h5ad),
        "n_var_positions": len(var_ids),
        "n_unique_var_proteins": len(unique_var_ids),
        "n_duplicate_var_positions": n_dup_positions,
        "duplicate_protein_ids": dup_acs,
        "n_missing_no_sequence": len(missing),
        "missing_protein_ids": missing,
        "n_kept_in_cache": len(kept_ids),
        "n_truncated": int(stats["truncated"]),
        "n_batches": int(stats["batches"]),
        "features_shape": list(features.shape),
        "features_dtype": "float32",
        "device": args.device,
        "elapsed_sec": round(dt, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[done] {args.plm}/{cache_name}: kept {len(kept_ids)}/{len(var_ids)} var-pos "
          f"(unique {len(unique_var_ids)}, dup-pos {n_dup_positions}, missing {len(missing)}, "
          f"trunc {stats['truncated']}) shape={features.shape} in {dt:.1f}s")
    print(f"[done] wrote {emb_path}\n        {ids_path}\n        {meta_path}")


if __name__ == "__main__":
    main()
