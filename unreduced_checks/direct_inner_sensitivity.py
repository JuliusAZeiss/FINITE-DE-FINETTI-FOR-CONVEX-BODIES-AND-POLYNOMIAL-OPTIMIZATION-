"""
direct_inner_sensitivity.py -- how much does the rounded value L^(n) depend
on WHICH optimizer of the unreduced level-n LP is rounded?
(observation (e) of the paper's numerics section)

The optimal face of the level-n LP is typically degenerate, so different
solver routes return different optimizers of the same LP, and the
measurement-based rounding of direct_inner_rounding -- a nonlinear map
(conditioning = division by p(z), productisation, max over outcomes) --
gives different values.  This script rounds, for n = 1, 2, the optimizers
returned by

    explicit-simplex   direct_hierarchy.py, HiGHS default (dual simplex,
                       presolve on) -- the optimizer of the direct check,
    explicit-ipm+xover explicit Kronecker LP, interior point + crossover
                       (a vertex of the optimal face),
    explicit-ipm       explicit Kronecker LP, interior point WITHOUT
                       crossover ("uncrossed IPM solution", tolerance
                       1e-10; such points tend to lie in the relative
                       interior of the optimal face, but this is not
                       verified here),
    tree-ipm           complete facet family with intermediate contraction
                       variables (direct_hierarchy_tree.py), ipm, no
                       crossover, as in the direct-check runs,

and, for each, the dihedral group average of the optimizer.  Every rounded
candidate is verified feasible within the stated tolerance (1e-7), the best
one is additionally certified rigorously (direct_inner_certify.py), and
every route optimizer is saved with its SHA-256 to
direct_optimizers/sensitivity_<exp>_n<n>_<route>_Y.npy, so that each value
is attached to a concrete, hashed Y.

Which optimizers are rounded.  HiGHS is deterministic only within one
build/platform: the same LP returns other vertices of the optimal face on
another machine.  By default the script therefore LOADS the saved route
optimizers (the ones behind the ranges quoted in observation (e) of the
paper) from direct_optimizers/ and rounds those; with --resolve the four
solver routes are run afresh on this machine (the new optimizers overwrite
the saved ones, so the rounded values may change).

Usage:  python3 direct_inner_sensitivity.py              (load saved Y's, ~10 s)
        python3 direct_inner_sensitivity.py --resolve    (re-solve, ~30 s)
Output: direct_inner_sensitivity.json, direct_inner_sensitivity.md (next to
        this file)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import direct_hierarchy as dh  # noqa: E402
import direct_hierarchy_tree as dt  # noqa: E402
from direct_inner_rounding import direct_inner_rounding, group_average, fmt_down  # noqa: E402
from direct_inner_certify import exact_instance_for  # noqa: E402

TOL_CANDIDATE = 1e-7


ROUTE_LABELS = ["explicit-simplex", "explicit-ipm+xover", "explicit-ipm",
                "tree-ipm"]


def saved_routes(D: dh.DirectData, n: int, opt_dir: Path):
    """Load the saved route optimizers (None if any file is missing)."""
    out = []
    LP = dh.build_level(D, n)
    for label in ROUTE_LABELS:
        stem = opt_dir / f"sensitivity_{dh.slug(D.name)}_n{n}_{label}"
        if not stem.with_name(stem.name + "_Y.npy").is_file():
            return None
        Y, meta = dh.load_optimizer(stem)          # verifies the SHA-256
        if len(Y) != LP["n_vars"]:
            raise RuntimeError(f"{stem}: wrong length")
        val = float(LP["c"] @ Y)
        if abs(val - meta["value"]) > 1e-12:
            raise RuntimeError(f"{stem}: objective {val!r} != recorded "
                               f"{meta['value']!r}")
        sol = dict(status=meta.get("status"),
                   solver_options=meta.get("solver_options"),
                   loaded_from=stem.with_name(stem.name + "_Y.npy").name,
                   recorded_environment=meta.get("environment"))
        out.append((label, Y, val, LP["c"], sol))
    return out


def routes(D: dh.DirectData, n: int):
    """Solve the level-n LP by the four solver routes (fresh optimizers)."""
    out = []
    # explicit Kronecker LP
    for label, solver, presolve, xover in [
            ("explicit-simplex", "choose", "choose", "on"),
            ("explicit-ipm+xover", "ipm", "choose", "on"),
            ("explicit-ipm", "ipm", "choose", "off")]:
        LP = dh.build_level(D, n)
        if xover == "on":
            sol = dh.solve_lp(LP["c"], LP["blocks"], solver=solver,
                              presolve=presolve)
        else:
            # same LP, ipm without crossover: rebuild via the tree solver's
            # generic HiGHS front end (it accepts crossover / ipm_tol)
            import gc
            from scipy import sparse
            mats, los, ups = [], [], []
            for name, A, lo, up in LP["blocks"]:
                A = A() if callable(A) else A
                mats.append(sparse.csr_matrix(A))
                los.append(np.asarray(lo, float))
                ups.append(np.asarray(up, float))
            A = sparse.vstack(mats, format="csr")
            lo, up = np.concatenate(los), np.concatenate(ups)
            del mats; gc.collect()
            sol = dt.solve(LP["c"], A, lo, up, presolve=presolve,
                           solver="ipm", verbose=False, crossover="off",
                           ipm_tol=1e-10)            # raises unless Optimal
            sol["value"] = float(LP["c"] @ sol["x"])
        out.append((label, sol["x"], sol["value"], LP["c"], sol))
    # tree formulation, ipm without crossover (as in direct_check_runs)
    cost, A, lo, up, info = dt.build_tree_lp(D, n, verbose=False)
    sol = dt.solve(cost, A, lo, up, solver="ipm", verbose=False,
                   crossover="off", ipm_tol=1e-10)
    NY = info["n_vars_Y"]
    out.append(("tree-ipm", sol["x"][:NY], float(cost[:NY] @ sol["x"][:NY]),
                cost[:NY], sol))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=2)
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--ref", type=str, default=None)
    ap.add_argument("--optimizers-dir", type=str, default=None)
    ap.add_argument("--no-certify", action="store_true")
    ap.add_argument("--resolve", action="store_true",
                    help="re-solve the four routes instead of loading the "
                         "saved optimizers (overwrites them)")
    args = ap.parse_args()
    ref_path = dh.find_file("results.json", args.ref)
    ref = json.load(open(ref_path))
    opt_dir = Path(args.optimizers_dir) if args.optimizers_dir else \
        HERE / "direct_optimizers"
    env = dh.environment()
    src_sha = {f: dh.sha256_file(HERE / f) for f in
               ("direct_inner_sensitivity.py", "direct_inner_rounding.py",
                "direct_hierarchy.py", "direct_hierarchy_tree.py")}
    res = {"_provenance": dict(environment=env, source_sha256=src_sha,
                               reduced_pipeline_cache=dict(
                                   file=os.path.relpath(ref_path, HERE),
                                   sha256=dh.sha256_file(ref_path)),
                               candidate_tolerance=TOL_CANDIDATE,
                               optimizers_dir=os.path.relpath(opt_dir, HERE))}
    lines = ["| K_B | n | route | p^(n) | min facet | S_n viol. | "
             "invariance residual | L^(n) raw (m*) | rigorous lower bound | "
             "L^(n) group-averaged (m*) | L^(n) results.json cache | Y sha256 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for prob, D in dh.make_experiments():
        if args.only and args.only not in D.name:
            continue
        res[D.name] = {}
        inst = None if args.no_certify else exact_instance_for(D)
        for n in range(1, args.nmax + 1):
            res[D.name][str(n)] = {}
            iref = ref[prob.name]["inner"][str(n)]["best"]
            gam = ref[prob.name]["gamma_star_exact"]
            loaded = None if args.resolve else saved_routes(D, n, opt_dir)
            if loaded is None:
                print(f"[{D.name} n={n}] solving the four routes afresh",
                      flush=True)
            for label, Y, val, c, sol in (loaded or routes(D, n)):
                t0 = time.time()
                chk = dh.verify_solution(D, n, Y, None)
                flags = dh.check_flags(chk)
                Yg = group_average(D, n, Y)
                inv = float(np.abs(Yg - Y).max())
                rd = direct_inner_rounding(D, n, Y, all_assignments=False,
                                           tol=TOL_CANDIDATE)
                rdg = direct_inner_rounding(D, n, Yg, all_assignments=False,
                                            tol=TOL_CANDIDATE)
                cert = inst.certify(rd["best_xi"], rd["best_N"]) \
                    if inst else None
                certg = inst.certify(rdg["best_xi"], rdg["best_N"]) \
                    if inst else None
                if loaded is None:
                    prov = dh.save_optimizer(
                        opt_dir / f"sensitivity_{dh.slug(D.name)}_n{n}_{label}",
                        Y, dict(experiment=D.name, n=n, route=label,
                                value=val, status=sol.get("status"),
                                solver_options=sol.get("solver_options"),
                                checks=chk, **flags, environment=env,
                                source_sha256=src_sha))
                else:
                    prov = dict(Y_file=sol["loaded_from"],
                                Y_sha256=dh.sha256_array(Y))
                if not (rd["best"] <= gam + 1e-9 and rdg["best"] <= gam + 1e-9
                        and val >= gam - 1e-9):
                    raise RuntimeError(f"{D.name} n={n} {label}: sandwich "
                                       f"L^(n) <= gamma* <= p^(n) violated")
                rec = dict(route=label, outer_value=val,
                           optimizer_origin=("loaded" if loaded is not None
                                             else "re-solved here"),
                           gamma_star_exact=gam,
                           status=sol.get("status"),
                           Y_file=prov["Y_file"], Y_sha256=prov["Y_sha256"],
                           min_facet=chk["min_facet"],
                           max_perm_violation=chk["max_perm_violation"],
                           max_FB_all_slots=chk["max_FB_all_slots"],
                           checks_pass=flags["all_checks_pass"],
                           invariance_residual=inv,
                           L_raw=rd["best"], m_raw=rd["best_m"],
                           per_m_raw={str(m): rd["per_m"][m]["best"]
                                      for m in rd["per_m"]},
                           avg_raw={str(m): rd["per_m"][m]["avg"]
                                    for m in rd["per_m"]},
                           min_slack_raw=rd["min_slack"],
                           min_outcome_probability_raw=
                           rd["min_outcome_probability"],
                           skipped_positive_mass_raw=
                           rd["skipped_positive_mass"],
                           L_raw_certified=(cert or {}).get("L_certified"),
                           L_raw_exact=(((cert or {}).get("exact") or {})
                                        .get("value_exact")
                                        if ((cert or {}).get("exact") or {})
                                        .get("value_exact_is_nice") else None),
                           L_raw_exact_full=((cert or {}).get("exact")),
                           L_avg=rdg["best"], m_avg=rdg["best_m"],
                           per_m_avg={str(m): rdg["per_m"][m]["best"]
                                      for m in rdg["per_m"]},
                           min_slack_avg=rdg["min_slack"],
                           L_avg_certified=(certg or {}).get("L_certified"),
                           results_json_L=iref, seconds=time.time() - t0)
                res[D.name][str(n)][label] = rec
                cs = "-" if cert is None else (
                    f"= {rec['L_raw_exact']} (exact)" if rec["L_raw_exact"]
                    else f">= {fmt_down(cert['L_certified'])}")
                lines.append(
                    f"| {D.name} | {n} | {label} | {val:.10f} | "
                    f"{chk['min_facet']:.1e} | {chk['max_perm_violation']:.1e}"
                    f" | {inv:.1e} | {rd['best']:.10f} ({rd['best_m']}) | {cs} | "
                    f"{rdg['best']:.10f} ({rdg['best_m']}) | {iref:.10f} | "
                    f"{prov['Y_sha256'][:12]} |")
                print(lines[-1], flush=True)
    # ---- ranges quoted in observation (e) of the paper -------------------
    # per experiment and level: [min, max] of the rounded values over the
    # four routes, raw and group-averaged optimizers alike
    lines += ["", "Range of the rounded value L^(n) over the four solver "
              "routes (raw and group-averaged optimizers), the spread "
              "quoted in observation (e) of the paper:", "",
              "| K_B | n | min L^(n) | max L^(n) | p^(n) | gamma* (exact) |",
              "|---|---|---|---|---|---|"]
    res["_ranges"] = {}
    for name, levels in res.items():
        if name.startswith("_"):
            continue
        res["_ranges"][name] = {}
        for n, recs in levels.items():
            vals = [r["L_raw"] for r in recs.values()] + \
                   [r["L_avg"] for r in recs.values()]
            p = max(r["outer_value"] for r in recs.values())
            gam = next(iter(recs.values()))["gamma_star_exact"]
            res["_ranges"][name][n] = dict(min=min(vals), max=max(vals),
                                           outer=p, gamma_star_exact=gam)
            lines.append(f"| {name} | {n} | {min(vals):.10f} | "
                         f"{max(vals):.10f} | {p:.10f} | {gam:.10f} |")
    with open(HERE / "direct_inner_sensitivity.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    with open(HERE / "direct_inner_sensitivity.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-14:]))
    print("written direct_inner_sensitivity.json / .md")


if __name__ == "__main__":
    main()
