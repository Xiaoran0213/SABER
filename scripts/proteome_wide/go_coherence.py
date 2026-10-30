#!/usr/bin/env python3
"""SABER proteome-wide biological-coherence analysis (layer b).

For a given (dataset, holdout fraction, seed) this loads the paired saber +
no-PLM expansion runs and asks: do SABER's predicted cell-type-specific held-out
proteins enrich to that cell type's known biology? GO BP + Reactome ORA
(hypergeometric, BH-FDR), background = all GO-annotated proteins in the dataset.

Methods compared per cell type:
  saber, noplm, plm_ridge, plm_knn -- predicted expression (cells x targets)
  oracle   -- GROUND-TRUTH held-out expression (x_true_target); upper bound
  cell_mean -- per-cell mean broadcast (null specificity control)

Multi-seed: writes per-seed coherence_summary_{ds}_holdout{frac}_seed{S}.csv,
then aggregates across seeds (mean±sd of n_sig / oracle_spearman / gold_recall)
into coherence_summary_{ds}_holdout{frac}_agg.csv for plotting.

Relocated from outputs/proteome-wide/scripts/ -> scripts/proteome_wide/.
Read-only over trained artifacts + annotation JSONs; writes under
outputs/proteome-wide/results/go/.
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from saber.paths import outputs_root  # noqa: E402
OUT = outputs_root() / "proteome-wide"
RUNS = OUT / "runs"
RESULTS = OUT / "results" / "go"
PREP_DEFAULT = outputs_root() / "paper_benchmark_20260614_full" / "prepared_inputs"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from saber.paths import benchmark_root  # noqa: E402

BENCH_DEFAULT = benchmark_root() / "external_annotations"
GOLD = Path(__file__).resolve().parent / "marker_gold_standard.json"
DATASETS_H5AD = {
    "Derks_2022": "Derks_2022/Derks_2022.h5ad",
    "Specht_2019": "Specht_2019/Specht_2019.h5ad",
}
DEFAULT_SEEDS = (42, 123, 2024)

NUMERIC_FIELDS = ("n_sig_GO_BP", "n_sig_Reactome", "oracle_spearman", "oracle_jaccard_top50", "gold_recall_topN")


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def load_annotations(ext_dir: Path):
    go_raw = json.loads((ext_dir / "go_annotations.json").read_text())
    react_raw = json.loads((ext_dir / "reactome_pathways.json").read_text())
    go_terms, react_terms = {}, {}
    for a in go_raw:
        tid = a.get("go_id", "")
        if tid and a.get("ontology", "BP") == "BP":
            go_terms[tid] = (a.get("go_name", tid), set(a.get("proteins", [])))
    for a in react_raw:
        tid = a.get("pathway_id", "")
        if tid:
            react_terms[tid] = (a.get("pathway_name", tid), set(a.get("proteins", [])))
    universe = set()
    for _, prots in go_terms.values():
        universe |= prots
    for _, prots in react_terms.values():
        universe |= prots
    return go_terms, react_terms, universe


def ora(query: list[str], terms: dict, universe: set[str]):
    query_set = set(query) & universe
    n = len(query_set)
    N = len(universe)
    rows = []
    if n == 0:
        return rows
    from scipy.stats import hypergeom

    for tid, (name, members) in terms.items():
        K = len(members & universe)
        if K == 0:
            continue
        k = len(members & query_set)
        if k == 0:
            continue
        p = float(hypergeom.sf(k - 1, N, K, n))
        rows.append({"term_id": tid, "term_name": name, "k": k, "K": K, "n": n, "N": N, "p": p})
    if not rows:
        return rows
    adj = bh_fdr(np.array([r["p"] for r in rows]))
    for r, a in zip(rows, adj):
        r["fdr"] = float(a)
    rows.sort(key=lambda r: r["p"])
    return rows


def marker_scores(matrix: np.ndarray, in_mask: np.ndarray):
    mat = np.asarray(matrix, dtype=float)
    ingroup = mat[in_mask]
    outgroup = mat[~in_mask]
    with np.errstate(all="ignore"):
        g_in = np.nanmean(ingroup, axis=0) if ingroup.size else np.full(mat.shape[1], np.nan)
        g_out = np.nanmean(outgroup, axis=0) if outgroup.size else np.full(mat.shape[1], np.nan)
    score = g_in - g_out
    return np.where(np.isfinite(score), score, -np.inf)


def cell_mean_matrix(x_true_train: np.ndarray) -> np.ndarray:
    x = np.asarray(x_true_train, dtype=float)
    finite = np.isfinite(x)
    sums = np.where(finite, x, 0.0).sum(axis=1, dtype=float)
    counts = finite.sum(axis=1, dtype=float)
    mean = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return np.broadcast_to(mean[:, None], (mean.shape[0], 1)).astype(np.float32)


def load_cell_types(h5ad_path: Path) -> np.ndarray:
    import anndata as ad

    return ad.read_h5ad(h5ad_path).obs["cell_type"].astype(str).to_numpy()


def load_run(run_dir: Path, want_refs: bool):
    exp = np.load(run_dir / "artifacts/expansion.npz")
    lab = np.load(run_dir / "artifacts/labels.npz", allow_pickle=True)
    out = {
        "target_protein_ids": [str(x) for x in lab["target_protein_ids"]],
        "x_true_target": exp["x_true_target"].astype(np.float32),
        "eval_target_mask": exp["eval_target_mask"].astype(bool),
        "saber": exp["pred_target__saber"].astype(np.float32),
        "x_true_train": exp["x_true_train"].astype(np.float32),
    }
    if want_refs:
        for key, name in [("pred_target__plm_ridge", "plm_ridge"), ("pred_target__plm_knn", "plm_knn")]:
            if key in exp.files:
                out[name] = exp[key].astype(np.float32)
    return out


def analyze(dataset: str, frac: int, seed: int, topn: int, prep: Path, bench: Path) -> str | None:
    """Run ORA coherence for one (dataset, fraction, seed). Returns summary CSV path."""
    saber_run = RUNS / f"{dataset}_holdout{frac}_saber_seed{seed}"
    noplm_run = RUNS / f"{dataset}_holdout{frac}_noplm_seed{seed}"
    if not (saber_run / "artifacts/expansion.npz").exists():
        print(f"[skip] {dataset} holdout{frac} seed{seed}: saber run not found", file=sys.stderr)
        return None
    go_terms, react_terms, universe = load_annotations(bench / dataset)
    gold = json.loads(GOLD.read_text())["cell_types"]
    cell_types = load_cell_types(prep / DATASETS_H5AD[dataset])
    ct_present = sorted(set(cell_types.tolist()))

    saber = load_run(saber_run, want_refs=True)
    noplm = load_run(noplm_run, want_refs=False) if (noplm_run / "artifacts/expansion.npz").exists() else None
    tpids = saber["target_protein_ids"]
    n_targets = len(tpids)
    assert saber["x_true_target"].shape[0] == cell_types.shape[0], "cell count mismatch"

    methods = {"saber": saber["saber"], "oracle": np.where(saber["eval_target_mask"], saber["x_true_target"], np.nan)}
    if "plm_ridge" in saber:
        methods["plm_ridge"] = saber["plm_ridge"]
    if "plm_knn" in saber:
        methods["plm_knn"] = saber["plm_knn"]
    if noplm is not None:
        methods["noplm"] = noplm["saber"]
    methods["cell_mean"] = np.repeat(cell_mean_matrix(saber["x_true_train"]), n_targets, axis=1)

    gold_eligible = {c: set(gold[c]["markers"]) & set(tpids) for c in ct_present if c in gold}
    summary_rows = []
    out_dir = RESULTS / f"{dataset}_holdout{frac}_seed{seed}"
    (out_dir / "ora_details").mkdir(parents=True, exist_ok=True)

    oracle_scores_by_ct = {c: marker_scores(methods["oracle"], cell_types == c) for c in ct_present}
    from scipy.stats import spearmanr

    for method, mat in methods.items():
        for c in ct_present:
            mask = cell_types == c
            score = marker_scores(mat, mask)
            top_idx = np.argsort(-score)[:topn]
            top_proteins = [tpids[i] for i in top_idx]
            ora_go = ora(top_proteins, go_terms, universe)
            ora_re = ora(top_proteins, react_terms, universe)
            for src, rows in (("go", ora_go), ("reactome", ora_re)):
                rec_path = out_dir / "ora_details" / f"{method}__{c}__{src}.csv"
                with rec_path.open("w") as f:
                    f.write("term_id,term_name,k,K,n,N,p,fdr\n")
                    for r in rows[:30]:
                        nm = r["term_name"].replace('"', "'")
                        f.write(f'{r["term_id"]},"{nm}",{r["k"]},{r["K"]},{r["n"]},{r["N"]},{r["p"]:.3e},{r["fdr"]:.3e}\n')
            n_sig_go = int(np.sum(np.array([r["fdr"] for r in ora_go]) < 0.05)) if ora_go else 0
            n_sig_re = int(np.sum(np.array([r["fdr"] for r in ora_re]) < 0.05)) if ora_re else 0
            top_go = ora_go[0] if ora_go else {"term_name": "", "fdr": float("nan")}
            top_re = ora_re[0] if ora_re else {"term_name": "", "fdr": float("nan")}
            valid = np.isfinite(score) & np.isfinite(oracle_scores_by_ct[c])
            sp = float(spearmanr(score[valid], oracle_scores_by_ct[c][valid]).correlation) if valid.sum() > 5 else float("nan")
            top_oracle = set(np.argsort(-oracle_scores_by_ct[c])[:topn].tolist())
            jac = len(top_oracle & set(top_idx.tolist())) / max(1, len(top_oracle | set(top_idx.tolist())))
            elig = gold_eligible.get(c, set())
            gold_recall = (len(elig & set(top_proteins)) / len(elig)) if elig else float("nan")
            summary_rows.append({
                "dataset": dataset, "fraction": frac, "seed": seed, "method": method, "cell_type": c,
                "lineage": gold.get(c, {}).get("lineage", ""),
                "n_sig_GO_BP": n_sig_go, "n_sig_Reactome": n_sig_re,
                "top_GO_term": top_go["term_name"], "top_GO_fdr": f'{top_go["fdr"]:.3e}',
                "top_Reactome_term": top_re["term_name"], "top_Reactome_fdr": f'{top_re["fdr"]:.3e}',
                "oracle_spearman": f'{sp:.3f}', "oracle_jaccard_top50": f'{jac:.3f}',
                "gold_n_eligible": (len(elig) if elig else 0), "gold_recall_topN": f'{gold_recall:.3f}',
            })

    sum_path = RESULTS / f"coherence_summary_{dataset}_holdout{frac}_seed{seed}.csv"
    cols = list(summary_rows[0].keys())
    with sum_path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(summary_rows)
    print(f"[done] {dataset} holdout{frac} seed{seed}: {len(summary_rows)} rows -> {sum_path.name}")
    return str(sum_path)


def aggregate(dataset: str, frac: int, seeds: list[int]) -> Path:
    """Mean±sd across seeds per (method, cell_type). Numeric fields aggregated;
    top_GO_term = the seed42 representative (stable showcase)."""
    rows_by_key: dict[tuple, list[dict]] = {}
    for s in seeds:
        p = RESULTS / f"coherence_summary_{dataset}_holdout{frac}_seed{s}.csv"
        if not p.exists():
            continue
        for r in _csv.DictReader(open(p)):
            rows_by_key.setdefault((r["method"], r["cell_type"]), []).append(r)
    out_rows = []
    for (method, ct), rows in sorted(rows_by_key.items()):
        base = dict(rows[0])
        agg = {"dataset": dataset, "fraction": frac, "method": method, "cell_type": ct, "lineage": base.get("lineage", ""),
               "n_seeds": len(rows)}
        for fld in NUMERIC_FIELDS:
            vals = [float(r[fld]) for r in rows if r.get(fld) not in ("", "nan", "None")]
            agg[f"{fld}_mean"] = f"{statistics.mean(vals):.3f}" if vals else "nan"
            agg[f"{fld}_sd"] = f"{statistics.pstdev(vals):.3f}" if len(vals) > 1 else "0.000"
        rep = next((r for r in rows if r["seed"] == "42"), rows[0])
        agg["top_GO_term"] = rep.get("top_GO_term", "")
        agg["top_GO_fdr"] = rep.get("top_GO_fdr", "")
        out_rows.append(agg)
    out_path = RESULTS / f"coherence_summary_{dataset}_holdout{frac}_agg.csv"
    if out_rows:
        cols = list(out_rows[0].keys())
        with out_path.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(out_rows)
        print(f"[agg] {dataset} holdout{frac}: {len(out_rows)} rows -> {out_path.name}")
    return out_path


def main() -> int:
    global OUT, RUNS, RESULTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--fraction", default="all")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--prep-dir", default=str(PREP_DEFAULT))
    ap.add_argument("--annotations-dir", default=str(BENCH_DEFAULT))
    ap.add_argument("--out-suffix", default="", help="campaign tree suffix, appended to the proteome-wide name")
    args = ap.parse_args()

    OUT = outputs_root() / f"proteome-wide{args.out_suffix}"
    RUNS = OUT / "runs"
    RESULTS = OUT / "results" / "go"
    RESULTS.mkdir(parents=True, exist_ok=True)
    datasets = list(DATASETS_H5AD) if args.dataset == "all" else [args.dataset]
    fracs = [50, 80, 90] if args.fraction == "all" else [int(args.fraction)]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    prep, bench = Path(args.prep_dir), Path(args.annotations_dir)

    n = 0
    for ds in datasets:
        for fr in fracs:
            for s in seeds:
                if analyze(ds, fr, s, args.topn, prep, bench) is not None:
                    n += 1
            aggregate(ds, fr, seeds)
    print(f"[go_coherence] analyzed {n} (dataset,fraction,seed) combos; seeds={seeds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
