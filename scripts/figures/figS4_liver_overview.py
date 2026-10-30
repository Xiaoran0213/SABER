#!/usr/bin/env python
"""Supplementary Fig. S4 — liver-zonation dataset overview (EasySCP, PXD070185).

Cells ordered along the central-portal axis (hepatocyte spatial-status score,
grouped into FACS gates A-H, CV to PV) against three protein blocks: CV-zone,
non-zonated, PV-zone (zone labels from the zonated marker tier). Reads only the
prepared h5ad + marker tables; writes figS4_liver_overview.{png,pdf} and the
source CSV (gate means per protein) under $SABER_FIGURES_ROOT/supp/.

Run: python scripts/figures/figS4_liver_overview.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "figures"))

from saber.paths import data_root, figures_root  # noqa: E402
from style import get_plt  # noqa: E402

plt = get_plt()

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--liver-dir", default="", help="easyscp_liver data dir (default: $SABER_DATA_ROOT/easyscp_liver)")
_ap.add_argument("--n-non-zonated", type=int, default=300,
                 help="how many non-zonated proteins to show (fixed-seed subsample)")
_ap.add_argument("--out", default="", help="figure output dir (default: $SABER_FIGURES_ROOT/supp)")
_args = _ap.parse_args()

BASE = Path(_args.liver_dir) if _args.liver_dir else data_root() / "easyscp_liver"
OUT = Path(_args.out) if _args.out else figures_root() / "supp"
OUT.mkdir(parents=True, exist_ok=True)

GATES = list("ABCDEFGH")

a = ad.read_h5ad(BASE / "prepared" / "EasySCP_liver.h5ad")
X = np.asarray(a.X, dtype=float)                      # cells x proteins (z-scored)
acs = list(a.var_names)
gate = np.asarray(a.obs["gate"], str)
hss = np.asarray(a.obs["hss_score"], float)

zt = pd.read_csv(BASE / "markers" / "zonated_table.csv")
zone_of = dict(zip(zt["AC"].dropna().astype(str), zt["zone"]))
cv = [ac for ac in acs if zone_of.get(ac) == "CV"]
pv = [ac for ac in acs if zone_of.get(ac) == "PV"]
nonzone = [ac for ac in acs if ac not in zone_of]
rng = np.random.default_rng(0)
show_nonzone = sorted(rng.choice(nonzone, size=min(_args.n_non_zonated, len(nonzone)), replace=False))
print(f"proteins: CV {len(cv)} | non-zonated {len(show_nonzone)}/{len(nonzone)} | PV {len(pv)}")

# cells: gate blocks A->H, ordered by HSS inside each gate (CV first overall)
order, gate_vec = [], []
for g in GATES:
    idx = np.where(gate == g)[0]
    idx = idx[np.argsort(-hss[idx])]                  # high HSS (CV-like) first
    order.extend(idx.tolist())
    gate_vec.extend([g] * len(idx))
order = np.array(order)
Xo = X[order]

col = {ac: i for i, ac in enumerate(acs)}
blocks = [("CV-zone", cv, "#B22222"), ("non-zonated", show_nonzone, "0.25"), ("PV-zone", pv, "#1F4E9C")]
gap = 3
widths = [len(b[1]) for b in blocks]
canvas = np.full((Xo.shape[0], sum(widths) + gap * (len(blocks) - 1)), np.nan)
pos = 0
for _, cols, _c in blocks:
    canvas[:, pos:pos + len(cols)] = Xo[:, [col[ac] for ac in cols]]
    pos += len(cols) + gap

fig, ax = plt.subplots(figsize=(7.2, 4.2))
im = ax.imshow(np.clip(canvas, -2, 2), aspect="auto", cmap="RdBu_r",
               vmin=-2, vmax=2, interpolation="nearest")
im.cmap.set_bad("white")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

# left axis: gate letters + CV/PV orientation
bounds = np.cumsum([0] + [int((gate == g).sum()) for g in GATES])
for k, g in enumerate(GATES):
    ax.text(-0.035, (bounds[k] + bounds[k + 1]) / 2 / canvas.shape[0], g,
            transform=ax.transAxes, ha="right", va="center", fontsize=8)
ax.annotate("", xy=(-0.02, 0.0), xytext=(-0.02, 1.0), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0))
ax.text(-0.045, 1.0, "CV", transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, fontweight="bold", color="#B22222")
ax.text(-0.045, 0.0, "PV", transform=ax.transAxes, ha="right", va="top",
        fontsize=9, fontweight="bold", color="#1F4E9C")

# block titles
pos = 0
for (title, cols, color), w in zip(blocks, widths):
    ax.text((pos + w / 2) / canvas.shape[1], 1.02, title, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9,
            fontweight="bold" if title != "non-zonated" else "normal", color=color)
    pos += w + gap

cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                  ticks=[-2, -1, 0, 1, 2])
cb.set_label("normalized expression", fontsize=8)
cb.ax.tick_params(labelsize=7)

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"figS4_liver_overview.{ext}", dpi=300, bbox_inches="tight")
# source data: gate-mean expression per shown protein
rows_src = []
pos = 0
for title, cols, _c in blocks:
    sub = canvas[:, pos:pos + len(cols)]
    for j, ac in enumerate(cols):
        for g in GATES:
            m = (np.asarray(gate_vec) == g)
            rows_src.append(dict(block=title, AC=ac, gate=g,
                                 mean=float(np.nanmean(sub[m, j]))))
    pos += len(cols) + gap
pd.DataFrame(rows_src).to_csv(OUT / "figS4_liver_overview_source.csv", index=False)
plt.close(fig)
print("panel ->", OUT / "figS4_liver_overview.png")
