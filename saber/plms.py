"""Central registry of the protein language models (PLMs) used in SABER.

Every PLM that can serve as SABER's protein-encoder input prior is registered
here exactly once. Two consumers read this registry:

* ``scripts/extract_plm_cache.py``  -- extracts per-protein mean-pooled caches
  (needs ``hf_id``, ``family``, ``max_len``, ``space_join``, ``drop_specials``,
  optional ``rare_aa_map``).
* ``scripts/run_prior_ablation.py`` -- the prior-choice ablation driver (needs
  ``dim`` and ``family``; treats ``REFERENCE_PLM`` as the paired-comparison
  anchor but trains every PLM fresh).

Adding a new PLM therefore means editing ONE place (this file), then (once) its
cache via ``extract_plm_cache.py``. Both consumers pick it up automatically.

Pooling recipe is identical across PLMs (attention-mask-weighted mean over
residue positions, special tokens dropped) so caches are mutually comparable.
"""
from __future__ import annotations

from typing import Any

# The PLM used as the SABER default and as the anchor in prior-choice
# comparisons. Trained fresh like every other PLM in the ablation.
REFERENCE_PLM = "esm2_650m"

# label -> spec. Fields:
#   dim          embedding dimensionality (SABER model.protein_feature_dim)
#   family       loader family: esm2 | esmc | t5 | bert | ankh
#   hf_id        HuggingFace model id
#   max_len      max input tokens (specials inclusive); longer seqs truncated
#   space_join   pre-tokenize by space-joining residues (ProtT5/ProtBERT)
#   drop_specials  leading/trailing special tokens to drop before mean pooling
#                  (bos/cls = leading, eos/sep = trailing)
#   rare_aa_map  map non-standard residues to 'X' before tokenizing (ProtBERT)
PLM_REGISTRY: dict[str, dict[str, Any]] = {
    "esm2_650m": {
        "dim": 1280, "family": "esm2",
        "hf_id": "facebook/esm2_t33_650M_UR50D", "max_len": 1022,
        "space_join": False, "drop_specials": ("bos", "eos"),
    },
    "esmc_600m": {
        "dim": 1152, "family": "esmc",
        "hf_id": "biohub/ESMC-600M", "max_len": 2048,
        "space_join": False, "drop_specials": ("cls", "eos"),
    },
    "esm2_150m": {
        "dim": 640, "family": "esm2",
        "hf_id": "facebook/esm2_t30_150M_UR50D", "max_len": 1022,
        "space_join": False, "drop_specials": ("bos", "eos"),
    },
    "prot_t5_xl": {
        "dim": 1024, "family": "t5",
        "hf_id": "Rostlab/prot_t5_xl_uniref50", "max_len": 2048,
        "space_join": True, "drop_specials": ("eos",),
    },
    "prot_bert": {
        "dim": 1024, "family": "bert",
        "hf_id": "Rostlab/prot_bert", "max_len": 1022,
        "space_join": True, "drop_specials": ("cls", "sep"),
        "rare_aa_map": True,
    },
    "ankh_large": {
        # Ankh-Large = T5 encoder (hidden 1536). Loaded encoder-only via
        # T5EncoderModel + fast tokenizer. Takes RAW sequences; appends </eos>.
        "dim": 1536, "family": "ankh",
        "hf_id": "ElnaggarLab/ankh-large", "max_len": 2048,
        "space_join": False, "drop_specials": ("eos",),
    },
    "saprot_af2": {
        # SaProt-650M (AF2 version): ESM-2 architecture over AA+FoldSeek-3Di
        # PAIR tokens ("MdEvQk..." = AA uppercase + 3Di lowercase per residue;
        # specials cls/eos; the vocabulary has no plain-AA tokens).
        # The most structure-informed protein prior available; tests whether
        # structural tokens change SABER downstream vs sequence-only ESM-2.
        # Sequences come from foldseek structure2token on AlphaFold DB PDBs —
        # extract_plm_cache.py reads the TSV at foldseek_tokens (acc -> "AA 3DI")
        # and INTERLEAVES the two, so cache coverage = structures available.
        "dim": 1280, "family": "saprot",
        "hf_id": "westlake-repl/SaProt_650M_AF2", "max_len": 1022,
        "space_join": False, "drop_specials": ("cls", "eos"),
        # Foldseek structure-token TSV (acc -> "AA 3DI"); produced by
        # `foldseek structure2seq` on AlphaFold DB PDBs. Configure via
        # SABER_SAPROT_TOKENS (see docs/DATA.md); the SaProt prior is optional.
        "foldseek_tokens": "",
    },
    "scgpt_gene": {
        # NOT a PLM: scGPT (whole-human) input gene embedding, 512d. Proteins are
        # mapped to gene symbols (UniProt GN=; mouse upper-cased) and take the
        # gene's encoder.embedding.weight row. Cache built by
        # scripts/build_scgpt_gene_cache.py -- extract_plm_cache.py can NOT build
        # this one (no sequences involved). Included to test a gene-level
        # expression-trained prior against sequence PLMs.
        "dim": 512, "family": "gene",
        "hf_id": "bowang-lab/scGPT_human", "max_len": 0,
        "space_join": False, "drop_specials": (),
        "prebuilt_cache_only": True,
    },
    "fusion_esm2_scgpt": {
        # CONCAT fusion prior: [esm2_650m 1280d | scgpt_gene 512d] = 1792d, rows
        # protein-aligned. Tests whether sequence + gene-coexpression priors carry
        # complementary information through SABER. Prebuilt cache only.
        "dim": 1792, "family": "fusion",
        "hf_id": "", "max_len": 0,
        "space_join": False, "drop_specials": (),
        "prebuilt_cache_only": True,
    },
    "fusion_esm2_scgpt_shuf": {
        # CONTROL for fusion_esm2_scgpt: identical, but the scgpt half is
        # row-permuted (gene identity destroyed, marginals kept). Same dim.
        "dim": 1792, "family": "fusion",
        "hf_id": "", "max_len": 0,
        "space_join": False, "drop_specials": (),
        "prebuilt_cache_only": True,
    },
}



