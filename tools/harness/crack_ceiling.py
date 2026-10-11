#!/usr/bin/env python3
"""Ceilings of the §13.4 Level-1 crack metrics for load-independent cuts.

The Rankine crack oracle (crack_oracle.py) cracks the solid once per load
case: bending in both transverse directions and torsion (fixed), plus
surface impacts at locations drawn at random (`--impact-seed`). Level-1
fragments are unions of analysis cells and are chosen without the load
cases. This script bounds what any such Level-1 can score:

* Recall@τ upper bound: the recall of ALL analysis-cell boundaries. Every
  Level-1 interface set is a subset of them, and recall is monotone in the
  interface set, so no Level-1 partition of the analysis cells can exceed
  it (for any segmentation method, load-aware or not).
* Weak-region Spearman ceiling: for each impact draw r, the Spearman
  correlation between the oracle's per-analysis-cell crack density O_r and
  the leave-one-out mean of the other draws (the best load-independent
  predictor: it knows the impact distribution but not the draw). Also
  reported: the correlation between independent draws, and our Level-1 cut
  density.

Usage: crack_ceiling.py --asset X.asset.json --cracks s1.npz s2.npz ...
(each .npz is `<cache>/<asset>.cracks.npz` from one crack_oracle.py run with
a different --impact-seed).
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crack_oracle import sample_interfaces  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--cracks", nargs="+", required=True)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    a = json.load(open(args.asset))
    cv = sorted(c["mass"]["volume"] for c in a["cells"])
    cell_d = (6 * cv[len(cv) // 2] / np.pi) ** (1 / 3)
    tau = 0.25 * cell_d
    ac = np.array([c["analysis_cell"] for c in a["cells"]])
    nac = int(ac.max()) + 1
    vol = np.bincount(ac, weights=[c["mass"]["volume"] for c in a["cells"]], minlength=nac)
    cf = a["hierarchy"]["cell_fragment"]
    ids_ac, ids_l1 = [], []
    for it in a["interfaces"]:
        b = it["cells"][1]
        if not isinstance(b, dict) or "Cell" not in b:
            continue
        x, y = it["cells"][0], b["Cell"]
        if ac[x] != ac[y]:
            ids_ac.append(it["id"])
        if cf[1][x] != cf[1][y]:
            ids_l1.append(it["id"])
    rng = np.random.default_rng(3)
    sp = tau / 3.0
    s_ac = sample_interfaces(a, ids_ac, sp, rng)
    s_l1 = sample_interfaces(a, ids_l1, sp, rng)
    t_ac, t_l1 = cKDTree(s_ac), cKDTree(s_l1)
    cent = np.array([c["mass"]["com"] for c in a["cells"]])
    tc = cKDTree(cent)
    l1_density = np.bincount(ac[tc.query(s_l1)[1]], minlength=nac) / np.maximum(vol, 1e-300)
    draws = []
    for path in args.cracks:
        z = np.load(path)
        cr = np.vstack([z[k] for k in z.files if len(z[k])])
        dens = np.bincount(ac[tc.query(cr)[1]], minlength=nac) / np.maximum(vol, 1e-300)
        draws.append({
            "file": os.path.basename(path),
            "density": dens,
            "recall_all_ac": float((t_ac.query(cr)[0] <= tau).mean()),
            "recall_l1": float((t_l1.query(cr)[0] <= tau).mean()),
            "cells_with_cracks": int((dens > 0).sum()),
        })
    rows = []
    for r, d in enumerate(draws):
        others = [x["density"] for k, x in enumerate(draws) if k != r]
        loo = np.mean(others, axis=0) if others else np.zeros(nac)
        pair = [spearmanr(d["density"], o).correlation for o in others]
        rows.append({
            "file": d["file"],
            "recall_upper_bound": d["recall_all_ac"],
            "recall_ours_l1": d["recall_l1"],
            "cells_with_cracks": d["cells_with_cracks"],
            "spearman_best_load_independent": float(spearmanr(d["density"], loo).correlation),
            "spearman_between_draws_mean": float(np.nanmean(pair)) if pair else float("nan"),
            "spearman_ours_l1": float(spearmanr(d["density"], l1_density).correlation),
        })
    out = {
        "asset": a["meta"]["name"],
        "tau": tau,
        "analysis_cells": nac,
        "draws": rows,
        "summary": {
            k: {"mean": float(np.nanmean([x[k] for x in rows])), "max": float(np.nanmax([x[k] for x in rows]))}
            for k in ("recall_upper_bound", "recall_ours_l1", "spearman_best_load_independent", "spearman_between_draws_mean", "spearman_ours_l1")
        },
    }
    if args.json:
        json.dump(out, open(args.json, "w"), indent=2)
    print(f"### Level-1 crack-metric ceilings: {out['asset']} ({len(rows)} impact draws, {nac} analysis cells, τ = {tau:.4f} m)\n")
    print("| Draw | Cells with cracks | Recall, all analysis-cell boundaries (upper bound) | Recall, our L1 | Spearman, best load-independent | Spearman, between draws | Spearman, our L1 |")
    print("|---|---|---|---|---|---|---|")
    for x in rows:
        print(f"| {x['file']} | {x['cells_with_cracks']} | {x['recall_upper_bound']:.3f} | {x['recall_ours_l1']:.3f} | {x['spearman_best_load_independent']:.3f} | {x['spearman_between_draws_mean']:.3f} | {x['spearman_ours_l1']:.3f} |")
    s = out["summary"]
    print(f"| **mean / max** | | {s['recall_upper_bound']['mean']:.3f} / {s['recall_upper_bound']['max']:.3f} | {s['recall_ours_l1']['mean']:.3f} / {s['recall_ours_l1']['max']:.3f} | {s['spearman_best_load_independent']['mean']:.3f} / {s['spearman_best_load_independent']['max']:.3f} | {s['spearman_between_draws_mean']['mean']:.3f} / {s['spearman_between_draws_mean']['max']:.3f} | {s['spearman_ours_l1']['mean']:.3f} / {s['spearman_ours_l1']['max']:.3f} |")


if __name__ == "__main__":
    main()
