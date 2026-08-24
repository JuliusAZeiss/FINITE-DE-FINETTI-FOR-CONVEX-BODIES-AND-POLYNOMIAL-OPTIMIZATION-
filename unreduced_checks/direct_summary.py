"""Collect the unreduced ("direct") level values p^(n) of direct_hierarchy.py
(explicit Kronecker LP) and direct_hierarchy_tree.py (complete facet family
with intermediate contractions) and compare them with the reduced
pipeline's results.json.  Writes direct_results.json and prints a
side-by-side table (also as Markdown, direct_results_table.md).

All paths are resolved relative to this file: the raw run files are searched
in --runs-dir (default: direct_check_runs/ next to this script if it exists,
else the script directory), results.json in cwd / script dir / parent (or
--ref).  Records whose HiGHS status is not "Optimal" are skipped with a
warning, and nothing is written if no record was found.

    python3 direct_summary.py [--runs-dir DIR] [--ref results.json]
                              [--out direct_results.json]
                              [--table direct_results_table.md]
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ORDER = ["classical bit", "square gbit", "pentagon", "pentagon+ineq"]
FILES_EXPLICIT = ["direct_results_classical.json", "direct_results_square.json",
                  "direct_results_pentagon.json", "direct_results_explicit.json"]
SLUGS = {"classical_bit": "classical bit", "square_gbit": "square gbit",
         "pentagon": "pentagon", "pentagon_ineq": "pentagon+ineq"}


def find_file(name, explicit=None):
    cands = [Path(explicit)] if explicit else \
        [Path.cwd() / name, HERE / name, HERE.parent / name]
    for c in cands:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"{name!r} not found in {[str(c) for c in cands]}")


def optimal(rec, fn):
    st = rec.get("status", rec.get("final_status", "Optimal"))
    if st != "Optimal":
        print(f"WARNING: {fn}: status {st!r} is not Optimal -- skipped",
              file=sys.stderr)
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=str, default=None)
    ap.add_argument("--ref", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--table", type=str, default=None)
    args = ap.parse_args()
    runs = Path(args.runs_dir) if args.runs_dir else \
        (HERE / "direct_check_runs" if (HERE / "direct_check_runs").is_dir()
         else HERE)
    out_path = Path(args.out) if args.out else HERE / "direct_results.json"
    table_path = Path(args.table) if args.table else \
        HERE / "direct_results_table.md"
    ref_path = find_file("results.json", args.ref)
    ref = json.load(open(ref_path))
    out = {name: {} for name in ORDER}
    n_found = 0

    # 1. explicit Kronecker LPs
    for fn in FILES_EXPLICIT:
        p = runs / fn
        if not p.is_file():
            continue
        d = json.load(open(p))
        for name, levels in d.items():
            if name.startswith("_"):            # provenance block
                continue
            for n, rec in levels.items():
                if not optimal(rec, p.name):
                    continue
                rec = dict(rec); rec["method"] = "explicit Kronecker LP"
                rec["source_record"] = os.path.relpath(p, HERE)
                out[name][n] = rec
                n_found += 1

    # 2. tree LPs (complete facet family, intermediate contractions)
    for fn in sorted(glob.glob(str(runs / "direct_tree_*_n*.json"))):
        if fn.endswith("_Y.json"):
            continue
        d = json.load(open(fn))
        if not optimal(d, fn):
            continue
        name = SLUGS[Path(fn).name[len("direct_tree_"):].rsplit("_n", 1)[0]]
        n = str(d["n"])
        n_found += 1
        if n not in out[name]:
            d = dict(d); d["method"] = "tree LP (complete facet family)"
            d["source_record"] = os.path.relpath(fn, HERE)
            out[name][n] = d
        else:
            out[name][n]["tree_LP_value"] = d["value"]
            out[name][n]["tree_LP_difference"] = d["difference"]
            out[name][n]["tree_LP_record"] = os.path.relpath(fn, HERE)

    if n_found == 0:
        sys.exit(f"no run files found in {runs} -- nothing written "
                 f"(use --runs-dir)")
    out["_provenance"] = dict(runs_dir=os.path.relpath(runs, HERE),
                              reference_file=os.path.relpath(ref_path, HERE),
                              n_records=n_found,
                              note="floating-point LP optima; residuals "
                                   "checked against stated tolerances")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=float)

    # ---- table ---------------------------------------------------------
    lines = []
    lines.append("| K_B | n | p^(n) unreduced | p^(n) reduced (results.json) | difference | formulation | LP size (vars / rows / nnz) | solve time |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name in ORDER:
        for n in sorted(out[name], key=int):
            r = out[name][n]
            pref = ref[name]["levels"][n]["value"]
            meth = r["method"]
            if meth.startswith("explicit"):
                size = f"{r['n_vars']:,} / {r['n_facet_rows'] + r['n_GA_rows'] + r['n_FB_rows'] + r['n_perm_rows'] + 1:,} / {r['nnz']:.1e}"
                tsec = r["solve_seconds"]
            elif meth.startswith("tree"):
                size = f"{r['n_vars_Y'] + r['n_vars_aux']:,} / {r['n_rows']:,} / {r['nnz']:.1e}"
                tsec = r["solve_seconds"]
            else:
                raise ValueError(f"unknown formulation {meth!r}")
            lines.append(f"| {name} | {n} | {r['value']:.10f} | {pref:.10f} | "
                         f"{r['value'] - pref:+.1e} | {meth} | {size} | {tsec:.0f} s |")
    table = "\n".join(lines)
    print(table)
    with open(table_path, "w") as f:
        f.write(table + "\n")
    print(f"written {out_path} and {table_path} ({n_found} records from {runs})")


if __name__ == "__main__":
    main()
