"""Collect proteome-wide results into a long-form table, same access
pattern as source_metrics_long.csv. Puts the proteome-wide sub-experiment INSIDE
the expansion framework (one data source for the expansion figure, instead of
the figure reaching into outputs/proteome-wide/ raw).

Outputs (default):
  $SABER_OUTPUTS_ROOT/proteome-wide/source_data/source_proteome_wide_long.csv
(--out / --campaign-run override; the expansion figure scripts take the table
via --proteome-wide-csv or read it from a campaign run's source_data/).

Two row types:
  task=proteome_wide_degradation : per (dataset, fraction, condition, seed, method)
                                   pearson_target_mean  [from each run's expansion_metrics.csv]
  task=proteome_wide_go          : per (dataset, fraction, method, cell_type)
                                   n_sig_GO_BP  [from go/coherence_summary_*_agg.csv]
Per-term GO ora_details stay file-based (non-scalar) — see expansion_figures.py.

Run:  python scripts/collect_proteome_wide.py [--campaign-run <run>]
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saber.paths import outputs_root  # noqa: E402

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--tree-suffix", default="",
                 help="campaign tree suffix, appended to the proteome-wide tree name")
_ap.add_argument("--campaign-run", default="",
                 help="write into this campaign run's source_data/ instead of the default location")
_ap.add_argument("--out", default="", help="explicit output CSV path (wins over --campaign-run)")
_args = _ap.parse_args()

PW = outputs_root() / f"proteome-wide{_args.tree_suffix}" / "results"
if _args.out:
    OUT = Path(_args.out)
elif _args.campaign_run:
    OUT = outputs_root() / _args.campaign_run / "source_data" / "source_proteome_wide_long.csv"
else:
    OUT = outputs_root() / "proteome-wide" / "source_data" / "source_proteome_wide_long.csv"

FRAC_MAP = {"50": 0.5, "80": 0.8, "90": 0.9}


def _collect_degradation():
    rows = []
    for d in glob.glob(str(PW / "*_seed*")):
        name = Path(d).name  # <ds>_holdout<X>_<cond>_seed<N>
        m = re.match(r"(.+?)_holdout(\d+?)_(saber|noplm)_seed(\d+)$", name)
        if not m:
            continue
        ds, xtag, cond, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
        frac = FRAC_MAP.get(xtag)
        if frac is None:
            continue
        f = Path(d) / "expansion_metrics.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        for _, r in t.iterrows():
            v = r.get("pearson_target_mean")
            if pd.notna(v):
                rows.append(dict(task="proteome_wide_degradation", dataset=ds, method=r["method"],
                                 condition=cond, seed=seed, target_fraction=frac, cell_type="",
                                 metric="pearson_target_mean", value=float(v)))
    return pd.DataFrame(rows)


def _collect_go():
    rows = []
    for f in glob.glob(str(PW / "go" / "coherence_summary_*_holdout*_agg.csv")):
        m = re.match(r"coherence_summary_(.+?)_holdout(\d+?)_agg\.csv$", Path(f).name)
        if not m:
            continue
        ds, xtag = m.group(1), m.group(2)
        frac = FRAC_MAP.get(xtag)
        if frac is None:
            continue
        t = pd.read_csv(f)
        for _, r in t.iterrows():
            rows.append(dict(task="proteome_wide_go", dataset=ds, method=r["method"],
                             condition=r.get("method", ""), seed="", target_fraction=frac,
                             cell_type=r["cell_type"], metric="n_sig_GO_BP",
                             value=float(r["n_sig_GO_BP_mean"])))
    return pd.DataFrame(rows)


def main():
    deg = _collect_degradation()
    go = _collect_go()
    df = pd.concat([deg, go], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} rows -> {OUT}")
    print(df.groupby(["task", "dataset"]).size().to_string())


if __name__ == "__main__":
    main()
