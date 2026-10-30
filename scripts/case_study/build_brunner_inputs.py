#!/usr/bin/env python
"""Build SABER-aligned inputs for Brunner 2022 cell-cycle SCP (HeLa, 434 cells).

Reads the DIANN pg_matrix + cached ESM-2 embeddings + derisk2 annotation table
under $SABER_DATA_ROOT/brunner2022 (see docs/DATA.md) and writes
prepared/{Brunner2022.h5ad,Brunner2022_features.npz} plus the marker tiers.
"""
import argparse, sys
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from saber.paths import data_root  # noqa: E402

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--brunner-dir", default="", help="brunner2022 data dir (default: $SABER_DATA_ROOT/brunner2022)")
_args = _ap.parse_args()

B = Path(_args.brunner_dir) if _args.brunner_dir else data_root() / "brunner2022"
D = B / "DIANN1.8_SingleCells_CellCycle"
OUT = B / "prepared"; OUT.mkdir(exist_ok=True)

pg = pd.read_csv(D / "20210919_DIANN_SingleCellOutput.pg_matrix.tsv", sep="\t")
runs = [c for c in pg.columns if c.endswith(".d")]
M = pg[runs].values.astype(float)            # 2501 proteins x 434 cells
ac = pg["Protein.Group"].values
ac_base = np.array([a.split("-")[0] for a in ac])
genes = pg["Genes"].fillna("").values
# batch from run-name date
batch = np.array([r.split("\\")[-1].split("_")[0] for r in runs])  # 20201009 etc.

# ESM-2 cache (2140 proteins with FASTA seq), aligned via derisk2 results csv
emb = np.load(B / "brunner_esm2_650m.npy")              # (2140,1280)
dr2 = pd.read_csv(B / "brunner_derisk2_results.csv")    # AC_base,gene,osc_corr,... (2140, aligned to emb)
assert len(dr2) == emb.shape[0], (len(dr2), emb.shape)
# dedup by AC_base (isoform suffix stripping created duplicates) — keep first occurrence
dup_mask = ~dr2.AC_base.duplicated(keep="first")
print(f"dedup: {len(dr2)} -> {dup_mask.sum()} unique AC_base")
dr2 = dr2[dup_mask].reset_index(drop=True)
emb = emb[dup_mask.values]
keep_ac = dr2.AC_base.values
# protein-row mask: which rows of M correspond to the 2140 ESM proteins
ac_to_row = {}
for i, a in enumerate(ac_base):
    ac_to_row.setdefault(a, i)
rows = np.array([ac_to_row[a] for a in keep_ac])
Mr = M[rows, :]                                            # 2140 x 434
genr = genes[rows]
X = np.where(Mr > 0, np.log1p(Mr), np.nan).T.astype(np.float32)   # cells x proteins, NaN missing

cc = np.load(B / "brunner_cc_score.npy")                 # (434,)
obs = pd.DataFrame({
    "cell_cycle_score": cc.astype(np.float32),
    "batch": batch.astype(str),
    "cell_type": np.array(["HeLa"] * len(runs)),
})
var = pd.DataFrame({"gene": genr}, index=keep_ac.astype(str))
a = ad.AnnData(X=X, obs=obs, var=var)
a.obs["batch"] = a.obs["batch"].astype("category")
a.obs["cell_type"] = a.obs["cell_type"].astype("category")
a.write_h5ad(OUT / "Brunner2022.h5ad")
print(f"h5ad: {a.shape} (cells x proteins), missing {np.mean(~np.isfinite(X))*100:.1f}%, batches={sorted(set(batch))}")
print("obs:", list(a.obs.columns), "| var_names sample:", list(a.var_names[:3]))

# features cache aligned to var_names
np.savez(OUT / "Brunner2022_features.npz",
         features=emb.astype(np.float32), protein_ids=keep_ac.astype(str))
print(f"features: {emb.shape}, saved Brunner2022_features.npz")

# marker tiers from osc_corr (in dr2, aligned to keep_ac)
MK = B / "markers"; MK.mkdir(exist_ok=True)
dr2["abs_osc"] = dr2.osc_corr.abs()
# tier1: top-150 oscillators; tier2: bottom-150 non-oscillators (sanity floor)
top = dr2.sort_values("abs_osc", ascending=False).head(150)
bot = dr2.sort_values("abs_osc", ascending=True).head(150)
# write target lists (AC_base)
pd.DataFrame({"AC": top.AC_base, "gene": top.gene, "osc_corr": top.osc_corr}).to_csv(MK/"oscillating_table.csv", index=False)
pd.DataFrame({"AC": bot.AC_base, "gene": bot.gene, "osc_corr": bot.osc_corr}).to_csv(MK/"nonoscillating_table.csv", index=False)
np.savetxt(MK/"oscillating_targets.txt", top.AC_base.values, fmt="%s")
np.savetxt(MK/"nonoscillating_targets.txt", bot.AC_base.values, fmt="%s")
# a larger oscillating tier: top-300
top300 = dr2.sort_values("abs_osc", ascending=False).head(300)
np.savetxt(MK/"oscillating300_targets.txt", top300.AC_base.values, fmt="%s")
pd.DataFrame({"AC": top300.AC_base, "gene": top300.gene, "osc_corr": top300.osc_corr}).to_csv(MK/"oscillating300_table.csv", index=False)
print(f"markers: oscillating150 ({len(top)}), nonoscillating150 ({len(bot)}), oscillating300 ({len(top300)})")
print("DONE.")
