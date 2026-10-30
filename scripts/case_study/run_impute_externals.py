#!/usr/bin/env python
"""Run external imputers (PIMMS, omicsGMF) on the EasySCP REAL observed matrix
(no artificial injection), via the manifest pipeline. Adds their imputed complete
matrices to matrices.npz so reanalyze_realimpute.py picks them up."""
import sys, json, argparse
from pathlib import Path
import numpy as np, anndata as ad
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from saber.external_methods import run_external_for_manifest
from saber.paths import data_root, outputs_root

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--suffix", default="", help="run-tree suffix, appended to run names")
_ap.add_argument("--out", default="", help="override the output run dir")
_ap.add_argument("methods", nargs="*", default=["omicsgmf", "pimms_dae"])
_args = _ap.parse_args()
SUFFIX = _args.suffix

BASE = data_root() / "easyscp_liver"
PREP = BASE / "prepared" / "EasySCP_liver.h5ad"
OUT = Path(_args.out) if _args.out else outputs_root() / f"easyscp_realimpute_pilot{SUFFIX}"
EXTWD = OUT / "externals"; EXTWD.mkdir(parents=True, exist_ok=True)

a = ad.read_h5ad(PREP)
X = np.asarray(a.X, dtype=np.float32)                 # cells x proteins, NaN=real missing
observed_mask = np.isfinite(X).astype(np.uint8)
x_input = np.where(observed_mask.astype(bool), X, 0.0).astype(np.float32)
npz = EXTWD / "imputation.npz"
np.savez_compressed(npz, x_input=x_input, observed_mask=observed_mask, x_true=X.astype(np.float32))
manifest = {"artifacts": {"imputation": str(npz)}}
(EXTWD / "manifest.json").write_text(json.dumps(manifest))

methods = _args.methods
seeds = [42, 123, 7]
results = {}
for m in methods:
    for s in seeds:
        wd = EXTWD / f"wd_{m}_seed{s}"; wd.mkdir(parents=True, exist_ok=True)
        print(f"=== running {m} seed {s} ===")
        try:
            r = run_external_for_manifest(EXTWD / "manifest.json", m, "imputation", wd, seed=s, timeout_sec=3600)
            exp_csv = wd / f"{m}_imputed.expanded.csv"
            raw_csv = wd / f"{m}_imputed.csv"
            csv = exp_csv if exp_csv.exists() else raw_csv
            print(f"  ok={r.get('ok')} | csv={csv.name} exists={csv.exists()}")
            if csv.exists():
                M = np.loadtxt(csv, delimiter=",") if csv == raw_csv else np.loadtxt(csv, delimiter=",", skiprows=1)
                if M.shape[1] == x_input.shape[1] + 1:
                    M = M[:, 1:]
                print(f"  loaded shape {M.shape}")
                results[f"{m}_seed{s}"] = M.astype(np.float32)
        except Exception as e:
            print(f"  ERR {m} seed{s}: {e}")

# merge into matrices.npz
Z = dict(np.load(OUT / "matrices.npz"))
for m, M in results.items():
    Z[m] = M
# non-per-seed aliases (mean over seeds) so downstream readers that expect a single
# matrix per method (e.g. make_fig6_v5 panel_c reads Z["omicsgmf"]/Z["pimms_dae"]) work.
for m in sorted({k.rsplit("_seed", 1)[0] for k in results}):
    seed_mats = [Z[k] for k in Z if k.startswith(f"{m}_seed")]
    if seed_mats:
        Z[m] = np.mean(seed_mats, axis=0)
np.savez_compressed(OUT / "matrices.npz", **Z)
print("merged externals into matrices.npz:", list(results.keys()))
# omicsGMF protein factor (deterministic across seeds) as a standalone npy for
# proteinemb_metrics.py / fig5_liver_casestudy.py
sidecar = EXTWD / "wd_omicsgmf_seed42" / "omicsgmf_imputed.csv.protein.csv"
if sidecar.is_file():
    np.save(OUT / "omicsgmf_protein_embedding.npy",
            np.loadtxt(sidecar, delimiter=",").astype(np.float32))
    print("omicsgmf_protein_embedding.npy saved from sidecar")
print("matrices.npz keys:", list(np.load(OUT/'matrices.npz').files))
