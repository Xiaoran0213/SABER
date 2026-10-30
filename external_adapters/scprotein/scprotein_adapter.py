#!/usr/bin/env python3
"""
scPROTEIN Stage2 adapter.

Reads a cells x proteins matrix (CSV), builds graph, trains Stage2 (stage1=False),
and writes the embedding to CSV.

Example:
  conda run -n scPROTEIN python external/scprotein_adapter.py \\
    --input X_embed_input.csv --out scprotein_embedding.csv
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

try:
    from scprotein import Encoder, Model, scPROTEIN_learning, graph_generation
    from scprotein.utils import setup_seed
except Exception as exc:  # pragma: no cover
    print(f"[error] scprotein package import failed: {exc}", file=sys.stderr)
    sys.exit(1)


def load_matrix(path: str, sep: str = ",") -> np.ndarray:
    return np.loadtxt(path, delimiter=sep, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Run scPROTEIN Stage2 to get embeddings.")
    parser.add_argument("--input", required=True, help="Input CSV (cells x proteins)")
    parser.add_argument("--out", required=True, help="Output CSV path for embedding")
    parser.add_argument("--sep", default=",", help="CSV delimiter")
    parser.add_argument("--threshold", type=float, default=0.15, help="Graph threshold")
    parser.add_argument("--feature-preprocess", type=int, default=1, help="Feature preprocess flag (1/0)")
    parser.add_argument("--num-hidden", type=int, default=400, help="Hidden dimension")
    parser.add_argument("--num-proj-hidden", type=int, default=256, help="Projection head dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="GCN layers")
    parser.add_argument("--num-protos", type=int, default=2, help="Number of prototypes")
    parser.add_argument("--drop-edge-rate-1", type=float, default=0.2)
    parser.add_argument("--drop-edge-rate-2", type=float, default=0.4)
    parser.add_argument("--drop-feature-rate-1", type=float, default=0.4)
    parser.add_argument("--drop-feature-rate-2", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--num-changed-edges", type=int, default=10, help="Topology denoising add/remove edges")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda:0 ...")
    args = parser.parse_args()

    setup_seed(args.seed)

    X = load_matrix(args.input, sep=args.sep)

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data = graph_generation(X, args.threshold, bool(args.feature_preprocess)).to(device)
    encoder = Encoder(data.num_features, args.num_hidden, nn.PReLU(), k=args.num_layers).to(device)
    model = Model(encoder, args.num_hidden, args.num_proj_hidden, args.tau).to(device)

    runner = scPROTEIN_learning(
        model,
        device,
        data,
        args.drop_feature_rate_1,
        args.drop_feature_rate_2,
        args.drop_edge_rate_1,
        args.drop_edge_rate_2,
        args.learning_rate,
        args.weight_decay,
        args.num_protos,
        False,  # topology_denoising
        args.num_epochs,
        args.alpha,
        args.num_changed_edges,
        args.seed,
    )
    runner.train()
    emb = runner.embedding_generation()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savetxt(args.out, emb, delimiter=",")


if __name__ == "__main__":
    main()

