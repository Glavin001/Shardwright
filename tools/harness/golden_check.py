#!/usr/bin/env python3
"""Regression check of oracle metrics against a committed baseline.

Reads the harness outputs of one asset (bond_fidelity.json, crack_oracle.json
in the oracle cache directory) and the bake report, and compares them with
benchmarks/golden/<asset>/expected.json:

  * hard gates must all pass (report.json);
  * every tracked metric must not be worse than its baseline by more than
    max(rel_tol × baseline, abs_tol) (lower-is-better metrics: errors;
    higher-is-better: recall, F-score);
  * spec targets (§13) are reported alongside, and enforced with --strict.

  golden_check.py --asset rc_column --cache DIR --report out/rc_column.report.json
  golden_check.py ... --update      # rewrite the baseline from this run
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.normpath(os.path.join(HERE, "..", "..", "benchmarks", "golden"))
REL_TOL, ABS_TOL = 0.05, 0.005

# metric -> (higher_is_better, spec target or None)
SPEC = {
    "bond.L3.raw_p50.": (False, 0.10),
    "bond.L3.raw_p95.": (False, 0.30),
    "bond.L3.stiffness.": (False, 0.10),
    "bond.L3.modal_max": (False, 0.10),
    "crack.L3.recall": (True, 0.80),
    "crack.L1.f": (True, None),
    "crack.weak_region_spearman": (True, 0.60),
}


def collect(cache, asset):
    m = {}
    p = os.path.join(cache, f"{asset}.bond_fidelity.json")
    if os.path.exists(p):
        b = json.load(open(p))
        for lv in b["levels"]:
            L = f"L{lv['level']}"
            for c in lv["cases"]:
                m[f"bond.{L}.raw_p50.{c['case']}"] = c["raw_vector_err_p50"]
                m[f"bond.{L}.raw_p95.{c['case']}"] = c["raw_vector_err_p95"]
                m[f"bond.{L}.stiffness.{c['case']}"] = c["stiffness_err"]
            if lv.get("modal_err"):
                m[f"bond.{L}.modal_max"] = max(lv["modal_err"])
    p = os.path.join(cache, f"{asset}.crack_oracle.json")
    if os.path.exists(p):
        c = json.load(open(p))
        for L, v in c.get("levels", {}).items():
            m[f"crack.{L}.recall"] = v["recall"]
            m[f"crack.{L}.f"] = v["f"]
        if "weak_region_spearman" in c:
            m["crack.weak_region_spearman"] = c["weak_region_spearman"]
    return m


def spec_of(name):
    for prefix, v in SPEC.items():
        if name.startswith(prefix):
            return v
    return (name.split(".")[-1] in ("recall", "f") or name.endswith("spearman"), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--strict", action="store_true", help="also fail on unmet spec targets")
    args = ap.parse_args()
    metrics = collect(args.cache, args.asset)
    exp_path = os.path.join(GOLDEN, args.asset, "expected.json")
    failures = []
    if args.report:
        r = json.load(open(args.report))
        bad = [g["name"] for g in r["scorecard"]["gates"] if g["status"] == "Fail"]
        if bad:
            failures.append(f"hard gates failed: {', '.join(bad)}")
    if args.update:
        os.makedirs(os.path.dirname(exp_path), exist_ok=True)
        json.dump({"rel_tol": REL_TOL, "abs_tol": ABS_TOL, "metrics": metrics}, open(exp_path, "w"), indent=2, sort_keys=True)
        print(f"baseline written: {exp_path} ({len(metrics)} metrics)")
        return 0
    if not os.path.exists(exp_path):
        print(f"no baseline at {exp_path}; run with --update")
        return 1
    exp = json.load(open(exp_path))
    rt, at = exp.get("rel_tol", REL_TOL), exp.get("abs_tol", ABS_TOL)
    print(f"| Metric | Baseline | Now | Spec target | Status |\n|---|---|---|---|---|")
    for name, base in sorted(exp["metrics"].items()):
        hib, target = spec_of(name)
        now = metrics.get(name)
        if now is None:
            failures.append(f"{name}: missing")
            print(f"| {name} | {base:.4f} | — | | MISSING |")
            continue
        slack = max(rt * abs(base), at)
        worse = (now < base - slack) if hib else (now > base + slack)
        meets = target is None or (now >= target if hib else now <= target)
        status = "REGRESSED" if worse else ("ok" if meets else "ok (spec target unmet)")
        if worse:
            failures.append(f"{name}: {base:.4f} -> {now:.4f}")
        if args.strict and not meets:
            failures.append(f"{name}: {now:.4f} misses spec target {target}")
        print(f"| {name} | {base:.4f} | {now:.4f} | {'' if target is None else ('≥ ' if hib else '≤ ') + str(target)} | {status} |")
    if failures:
        print("\nFAIL:\n- " + "\n- ".join(failures))
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
