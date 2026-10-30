import argparse
import os
import shlex
import subprocess
import sys
import json
import pandas as pd
import numpy as np


def _load_spec(config_path: str | None, self_dir: str) -> dict | None:
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    for cand in (os.path.join(self_dir, "spec.json"), os.path.join(self_dir, "spec.yaml")):
        if os.path.exists(cand):
            with open(cand, "r") as f:
                try:
                    return json.load(f)
                except Exception:
                    return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", dest="out", required=True)
    parser.add_argument("--repo", dest="repo", required=False, default="")
    parser.add_argument("--config", dest="config", required=False, default=None)
    parser.add_argument("--model", dest="model", required=False, default="DAE")
    parser.add_argument("--seed", dest="seed", type=int, required=False, default=42)
    args = parser.parse_args()

    self_dir = os.path.dirname(os.path.abspath(__file__))
    # Prefer direct Python API to avoid depending on external CLIs
    repo = args.repo or ""
    if repo and os.path.isdir(repo):
        sys.path.insert(0, repo)
    # Importers (lazy per model below)
    try:
        from pimmslearn.sklearn.ae_transformer import AETransformer
        from pimmslearn.sklearn.cf_transformer import CollaborativeFilteringTransformer
    except Exception as e:
        sys.stderr.write("Failed to import pimmslearn. Ensure repo path or pip install pimms-learn.\n")
        sys.stderr.write(str(e) + "\n")
        return 2

    # Seeds for reproducibility (best-effort)
    try:
        import random as _py_random
        _py_random.seed(int(args.seed))
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(int(args.seed))
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(int(args.seed))
    except Exception:
        pass

    # Load input matrix (CSV without headers)
    # Handle relative paths by converting to absolute
    inp_path = os.path.abspath(args.inp)
    if not os.path.exists(inp_path):
        sys.stderr.write(f"Input file not found: {inp_path}\n")
        return 1
    X = np.loadtxt(inp_path, delimiter=',')
    in_dir = os.path.dirname(inp_path)
    # Read optional row/column ID sidecars to preserve the original column
    # order for later alignment (all-missing columns would otherwise be lost)
    rows_path = os.path.join(in_dir, "rows.txt")
    cols_path = os.path.join(in_dir, "cols.txt")
    if os.path.exists(rows_path):
        with open(rows_path, "r") as f:
            row_ids = [line.strip() for line in f if line.strip()]
    else:
        row_ids = [f"r{i}" for i in range(X.shape[0])]
    if os.path.exists(cols_path):
        with open(cols_path, "r") as f:
            col_ids = [line.strip() for line in f if line.strip()]
    else:
        col_ids = [f"p{j}" for j in range(X.shape[1])]
    # Normalize model choice
    model_name = str(args.model).strip().upper()
    if model_name not in {"DAE", "VAE", "CF"}:
        model_name = "DAE"
    # CF expects long-format Series input with named axes
    if model_name == "CF":
        # Build a wide DataFrame and stack it into a Series
        df_wide = pd.DataFrame(X, index=row_ids, columns=col_ids)
        index_name = "SampleID"
        column_name = "protein"
        value_name = "intensity"
        df_wide.index.name = index_name
        df_wide.columns.name = column_name
        series = df_wide.stack()
        series.name = value_name
        # Train the collaborative-filtering model
        cf = CollaborativeFilteringTransformer(
            target_column=value_name,
            sample_column=index_name,
            item_column=column_name,
            n_factors=30,
            batch_size=4096,
        )
        cf.fit(series, cuda=False, epochs_max=10)
        df_imputed = cf.transform(series).unstack()
        # Force alignment back to the original column order
        df_imputed = df_imputed.reindex(index=row_ids, columns=col_ids)
    else:
        # DAE/VAE go through AETransformer (wide DataFrame)
        df = pd.DataFrame(X, index=row_ids, columns=col_ids)
        model = AETransformer(
            model=model_name,
            hidden_layers=[256],
            latent_dim=16,
            batch_size=64,
        )
        # AETransformer.fit takes no seed argument; only epochs/cuda are set here
        model.fit(df, cuda=False, epochs_max=10)
        df_imputed = model.transform(df)
        # pimms may drop all-missing columns internally; restore and realign them
        missing_cols = [c for c in col_ids if c not in df_imputed.columns]
        if missing_cols:
            for c in missing_cols:
                df_imputed[c] = np.nan
        df_imputed = df_imputed.reindex(index=row_ids, columns=col_ids)
    # Save numeric CSV (no header/index)
    out_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    # Save as a numeric matrix (no header/index)
    mat = df_imputed.to_numpy(dtype=np.float32)
    np.savetxt(out_path, mat, delimiter=',')
    return 0


if __name__ == "__main__":
    sys.exit(main())


