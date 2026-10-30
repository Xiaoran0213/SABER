#!/usr/bin/env python
"""Externals (omicsGMF / PIMMS-DAE) on the AXIS-HOLDOUT reduced matrices.

Sources the matrix from the axis-holdout matrices.npz (observed_reduced) so
every method trains without the axis proteins — apples-to-apples with SABER's
axis-holdout runs. 3 seeds [42,123,7] matching the realimpute protocol;
merges results back into matrices.npz. Needs SABER_PROTAE_ROOT (adapters).

Run:  python scripts/case_study/run_externals_reduced.py liver
      python scripts/case_study/run_externals_reduced.py brunner
"""
import sys, json, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from saber.external_methods import run_external_for_manifest
from saber.paths import outputs_root

SEEDS = [42, 123, 7]

ap = argparse.ArgumentParser()
ap.add_argument("dataset", choices=["liver", "brunner"])
ap.add_argument("methods", nargs="*", default=["omicsgmf", "pimms_dae"])
ap.add_argument("--out", default="", help="override the axis-holdout output directory")
args = ap.parse_args()

OUT = Path(args.out) if args.out else outputs_root() / (
    "easyscp_realimpute_axisholdout" if args.dataset == "liver" else "brunner_realimpute_axisholdout")
EXTWD = OUT / "externals"; EXTWD.mkdir(parents=True, exist_ok=True)

with np.load(OUT / "matrices.npz") as raw:
    Z = {k: np.array(raw[k]) for k in raw.files}
X = np.asarray(Z["observed_reduced"], dtype=np.float32)   # raw scale, NaN = real missing
n_cells, n_prot = X.shape
observed_mask = np.isfinite(X).astype(np.uint8)
x_input = np.where(observed_mask.astype(bool), X, 0.0).astype(np.float32)
npz = EXTWD / "imputation.npz"
np.savez_compressed(npz, x_input=x_input, observed_mask=observed_mask, x_true=X)
(EXTWD / "manifest.json").write_text(json.dumps({"artifacts": {"imputation": str(npz)}}))
print(f"{args.dataset}: reduced matrix {X.shape}, observed {observed_mask.mean():.2f}")

results = {}
for m in args.methods:
    for s in SEEDS:
        wd = EXTWD / f"wd_{m}_seed{s}"; wd.mkdir(parents=True, exist_ok=True)
        print(f"=== {m} seed {s} ===")
        try:
            r = run_external_for_manifest(EXTWD / "manifest.json", m, "imputation", wd, seed=s, timeout_sec=7200)
            exp_csv = wd / f"{m}_imputed.expanded.csv"
            raw_csv = wd / f"{m}_imputed.csv"
            csv = exp_csv if exp_csv.exists() else raw_csv
            if csv.exists():
                M = np.loadtxt(csv, delimiter=",") if csv == raw_csv else np.loadtxt(csv, delimiter=",", skiprows=1)
                if M.shape[1] == n_prot + 1:
                    M = M[:, 1:]
                if M.shape[0] != n_cells:      # externals may emit proteins x cells
                    M = M.T
                assert M.shape == (n_cells, n_prot), f"shape {M.shape} vs {(n_cells, n_prot)}"
                results[f"{m}_seed{s}"] = M.astype(np.float32)
                print(f"  ok, shape {M.shape}")
        except Exception as e:
            print(f"  ERR {m} seed{s}: {e}")

for m, M in results.items():
    Z[m] = M
np.savez_compressed(OUT / "matrices.npz", **Z)
print("merged:", list(results.keys()))
