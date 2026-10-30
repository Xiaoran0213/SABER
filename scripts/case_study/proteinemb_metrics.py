#!/usr/bin/env python
"""Protein-embedding CV/PV zonation separation: 10-seed SABER vs raw ESM-2 vs omicsGMF.

Case-study protein-level analog of the main benchmark (Fig. 5b-c). The omicsGMF
protein factor is read from the externals sidecar written by
run_externals_reduced.py / run_impute_externals.py
(``externals/wd_omicsgmf_seed42/omicsgmf_imputed.csv.protein.csv``; omicsGMF's
protein factor is deterministic across seeds), with the legacy
``omicsgmf_protein_embedding.npy`` accepted as a fallback.

Writes $SABER_OUTPUTS_ROOT/easyscp_expansion_logs/proteinemb_metrics_10seed{suffix}.csv
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from saber.paths import data_root, outputs_root  # noqa: E402

_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--suffix", default="", help="run-tree suffix, appended to run names")
_p.add_argument("--omicsgmf", default="", help="explicit path to the omicsGMF protein factor (npy or .protein.csv sidecar)")
_p.add_argument("--out", default="", help="output CSV path")
args = _p.parse_args()
SUFFIX = args.suffix

BASE = data_root() / "easyscp_liver"
RUNS = outputs_root()

a = ad.read_h5ad(BASE / "prepared" / "EasySCP_liver.h5ad"); acs = list(a.var_names)
rf = np.load(BASE / "prepared" / "EasySCP_liver_features.npz")
raw = rf["features"] if "features" in rf else rf["embeddings"]


def _load_omicsgmf(n_proteins: int) -> np.ndarray:
    """Resolve the omicsGMF protein factor over the FULL matrix (n_proteins rows).

    Candidates: explicit path > full-matrix externals sidecar/npy written by
    run_impute_externals.py > legacy pilot npy. Axis-holdout sidecars are not
    used: they cover only the reduced (axis-excluded) matrix.
    """
    cands = []
    if args.omicsgmf:
        cands.append(Path(args.omicsgmf))
    cands.append(RUNS / f"easyscp_realimpute_pilot{SUFFIX}" / "externals"
                 / "wd_omicsgmf_seed42" / "omicsgmf_imputed.csv.protein.csv")
    cands.append(RUNS / f"easyscp_realimpute_pilot{SUFFIX}" / "omicsgmf_protein_embedding.npy")
    for c in cands:
        if c.is_file():
            M = np.load(c) if c.suffix == ".npy" else np.loadtxt(c, delimiter=",")
            if M.shape[0] != n_proteins:
                raise SystemExit(
                    f"omicsGMF factor {c} has {M.shape[0]} rows, expected {n_proteins} "
                    "(full-matrix run required; axis-holdout sidecars are reduced). "
                    "Run scripts/case_study/run_impute_externals.py first."
                )
            return M
    raise SystemExit(
        "omicsGMF protein factor (full matrix) not found. Run "
        "scripts/case_study/run_impute_externals.py first "
        "(external-baseline setup: external_adapters/README.md), "
        "or pass --omicsgmf <path>."
    )


omg = _load_omicsgmf(len(acs))
# SABER protein embeddings across 10 seeds
saber = []
for s in [42, 123, 7, 2024, 2025, 0, 1, 2, 3, 4]:
    suff = "" if s == 42 else f"_seed{s}"
    f = sorted(glob.glob(str(RUNS / f"easyscp_cellembed_saber{suff}{SUFFIX}" / "*" / "artifacts" / "embeddings.npz")))
    if f: saber.append(np.load(f[-1])["protein_embeddings__saber"])
print("SABER seeds:", len(saber), "| raw ESM-2:", raw.shape, "| omicsGMF:", omg.shape)


def coh_gap(M, idx, y):
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9); S = Mn[idx] @ Mn[idx].T
    w = []; b = []
    for aa in range(len(idx)):
        for bb in range(aa + 1, len(idx)):
            (w if y[aa] == y[bb] else b).append(S[aa, bb])
    return float(np.mean(w) - np.mean(b))


def cv_auc(M, idx, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=0); p = np.zeros(len(y)); Mz = M[idx]
    for tr, te in skf.split(Mz, y):
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(Mz[tr], y[tr]); p[te] = lr.predict_proba(Mz[te])[:, 1]
    return roc_auc_score(y, p)


rows = []
for tier in ["canonical", "hss_markers", "zonated"]:
    zm = dict(zip(pd.read_csv(BASE / "markers" / f"{tier}_table.csv")["AC"],
                  pd.read_csv(BASE / "markers" / f"{tier}_table.csv")["zone"]))
    idx = np.array([i for i, ac in enumerate(acs) if zm.get(ac) in ("CV", "PV")])
    y = np.array([0 if zm[acs[i]] == "CV" else 1 for i in idx])
    if len(idx) < 8: continue
    sab_auc = [cv_auc(E, idx, y) for E in saber]; sab_gap = [coh_gap(E, idx, y) for E in saber]
    res = {"tier": tier, "n": len(idx),
           "SABER_AUC": f"{np.mean(sab_auc):.3f}±{np.std(sab_auc):.3f}",
           "SABER_gap": f"{np.mean(sab_gap):+.3f}±{np.std(sab_gap):.3f}",
           "rawESM2_AUC": f"{cv_auc(raw, idx, y):.3f}", "rawESM2_gap": f"{coh_gap(raw, idx, y):+.3f}",
           "omicsGMF_AUC": f"{cv_auc(omg, idx, y):.3f}", "omicsGMF_gap": f"{coh_gap(omg, idx, y):+.3f}"}
    rows.append(res); print(res)
out = Path(args.out) if args.out else RUNS / "easyscp_expansion_logs" / f"proteinemb_metrics_10seed{SUFFIX}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print("saved ->", out)
