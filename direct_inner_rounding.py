"""
direct_inner_rounding.py -- measurement-based rounding (the inner sequence
of Thm. "Convergent inner sequence") applied to the optimizers of the
UNREDUCED level-n LPs of direct_hierarchy.py
===========================================================================

Companion to direct_hierarchy.py / direct_check_note.md for

    J. A. Zeiss, G. Kossmann, R. Schwonnek, M. Plavala,
    "Finite de Finetti for convex bodies and polynomial optimization",
    arXiv:2601.15184, Sec. "Numerical illustration" (inner values L^(n)).

The reproduction pipeline rounds the optimizer returned by the reduced LP
(multiset basis, ker F_B parametrisation, dihedral orbit averaging) with
`gpt_definetti_games.inner_rounding`.  Here the SAME
rounding scheme is carried out on the optimizer Y of the literal,
unreduced level-n LP (full tensor in R^{d_A} (x) (R^{d_B})^{(x) n}), and
nothing is symmetrised, averaged or parametrised away:

    E_k := psi_k / s,  k = 1..|ext C_mm^*|,  with  sum_k psi_k = s * t
           (the symmetric informationally complete measurement on K_mm),
    for m = 0, ..., n-1:
        measure copies 1..m with {E_k}, keep copy m+1, marginalise the
        remaining n-1-m copies with the unit t:
            X_z := (id_A (x) E_{z_1} (x) ... (x) E_{z_m} (x) id_B (x) t (x)
                    ... (x) t)(Y)                       in  R^{d_A x d_B},
            p(z) := (u_A (x) t)(X_z)                     (outcome probability),
            x_{A|z} := (id_A (x) t)(X_z) / p(z),          (assemblage)
            x_{B|z} := (u_A (x) id_B)(X_z) / p(z),        (multimeter)
            Val(z)  := < W, x_{A|z} (x) x_{B|z} >,
    L^(n)   := max_m max_z Val(z),
    P(w_m)  := sum_z p(z) Val(z)   (value of the de Finetti inner point
               w^(n)_m = sum_z p(z) x_{A|z} (x) x_{B|z} for the given m).

All ORDERED outcome tuples z in {1..|ext C_mm^*|}^m are enumerated (no
multiset collapse; sum_z p(z) = 1 is checked), and -- optionally -- all
assignments of the n copies to the roles "measured / kept / marginalised"
are enumerated as well (they are equivalent by permutation invariance,
which the unreduced optimizer satisfies only up to solver tolerance, so
the spread over assignments is reported).  Every rounded candidate is
checked to be feasible for the ORIGINAL constrained problem in the
unreduced description:  Phi x_A >= 0 (all extreme rays of C_asm^*),
Psi x_B >= 0 (all extreme rays of C_mm^*), F_B x_B = 0 (multimeter
normalisation), u_A x_A = 1, t x_B = 1, and G_A x_A >= 0 (4th experiment).

Additionally the script
  * rounds the dihedral-group average of the unreduced optimizer (a
    group-invariant optimizer with the same objective value -- invariant,
    but NOT thereby unique: the invariant optimal face is degenerate too),
  * maps the unreduced optimizer to the reduced (multiset, ker F_B) basis
    and hands it to the library's `inner_rounding` as an independent
    implementation check (enforced to agree within 1e-9),
  * compares with the inner values of results.json (reduced pipeline; this
    is a comparison against the cached file, whose SHA-256 is recorded) and
    with the exact optimum gamma* (field gamma_star_exact of results.json),
  * turns the best candidate of every entry into a RIGOROUS lower bound
    (direct_inner_certify.py: interval arithmetic on an eps-mixed strategy,
    plus an exact rational check for the bit / square instances).

Which optimizer is rounded:  by default the script LOADS the optimizer
saved by direct_hierarchy.py
(`direct_optimizers/direct_explicit_<exp>_n<n>_Y.npy` + side-car JSON with
SHA-256, solver options, environment) and records its hash; it re-solves
only if no saved optimizer exists (or with --resolve), and then saves the
new one.  The rounded value L^(n)(Y) is a property of the selected Y: the
optimal face of the level-n LP is degenerate and the rounding map is
nonlinear, so different optimizers of the same LP round to different
values (observation (e) of the paper; see direct_inner_sensitivity.py).
Every value reported here is therefore attached to a hashed optimizer file.

Usage:
    python3 direct_inner_rounding.py                # n = 1, 2 (default, ~1 s)
    python3 direct_inner_rounding.py --nmax 3       # also n = 3 (seconds with
                                                    #  the saved optimizers)
    python3 direct_inner_rounding.py --resolve      # ignore saved Y's (the
                                                    #  n = 3 pentagon LPs take
                                                    #  ~20-30 min and ~3.5 GB)
Outputs: direct_inner_results.json, direct_inner_table.md (next to this
file unless --out/--table are given)
"""

from __future__ import annotations

import argparse
import itertools
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

import gpt_definetti_games as lib  # noqa: E402
import direct_hierarchy as dh  # noqa: E402
from direct_inner_certify import exact_instance_for  # noqa: E402

# tolerances of the enforced checks (explicit, recorded in the output)
TOL_OUTER_VS_CACHE = 1e-9     # |p^(n) unreduced - results.json| (obs. <=1e-12)
TOL_DIRECT_VS_LIB = 1e-9      # |direct rounding - library rounding| (<=1e-10)
TOL_VALUE_RECOMP = 1e-10      # best value recomputed from states/effects
TOL_CANDIDATE = 1e-8          # feasibility slack of every rounded candidate
TOL_PSUM = 1e-8               # |sum_z p(z) - 1|
P_FLOOR = 1e-12               # branches with p(z) <= P_FLOOR are not rounded
P_NEG_TOL = 1e-8              # p(z) < -P_NEG_TOL is an error, not skipped


def fmt_down(x: float, digits: int = 10) -> str:
    """Decimal string of a lower bound rounded DOWN (never up)."""
    import math
    return f"{math.floor(x * 10 ** digits) / 10 ** digits:.{digits}f}"


# ---------------------------------------------------------------------------
# 1.  The rounding on the unreduced tensor
# ---------------------------------------------------------------------------

def sic_measurement(D: dh.DirectData):
    """E_k = psi_k / s with sum_k psi_k = s t (checked)."""
    sums = D.Psi.sum(axis=0)
    s = float(sums @ D.t) / float(D.t @ D.t)
    if not np.allclose(sums, s * D.t, atol=1e-12):
        raise ValueError("extreme rays of C_mm^* do not form a symmetric IC "
                         "measurement")
    return D.Psi / s, s


def role_assignments(n: int, m: int, all_assignments: bool):
    """Tuples (measured, kept, marginalised) of B-slot indices (0-based)."""
    if not all_assignments:
        return [(tuple(range(m)), m, tuple(range(m + 1, n)))]
    out = []
    for kept in range(n):
        rest = [j for j in range(n) if j != kept]
        for meas in itertools.combinations(rest, m):
            marg = tuple(j for j in rest if j not in meas)
            out.append((meas, kept, marg))
    return out


def candidate_checks(D: dh.DirectData, xi: np.ndarray, N: np.ndarray):
    """Feasibility of the product strategy (xi, N) for the ORIGINAL
    constrained problem, in the unreduced description.  Returns the
    minimum slack over all inequality rows and the max |equality residual|."""
    slack = min(float((D.Phi @ xi).min()), float((D.Psi @ N).min()))
    if D.GA is not None:
        slack = min(slack, float((D.GA @ xi).min()))
    eq = max(float(abs(D.uA @ xi - 1.0)), float(abs(D.t @ N - 1.0)),
             float(np.abs(D.F @ N).max()))
    return slack, eq


def direct_inner_rounding(D: dh.DirectData, n: int, Y: np.ndarray,
                          all_assignments: bool = False,
                          tol: float = TOL_CANDIDATE,
                          p_floor: float = P_FLOOR,
                          p_neg_tol: float = P_NEG_TOL):
    """Measurement-based rounding of the unreduced level-n extension Y.

    Returns a dict with, per m: best value, best (ordered) outcome, the
    p(z)-weighted average value (value of w^(n)_m), the number of enumerated
    outcomes, sum_z p(z), the minimum outcome probability, the number of
    skipped branches (p <= p_floor) and their total positive mass, the
    worst feasibility slack / equality residual, and (if all_assignments)
    the spread of the best value over the role assignments; plus the
    overall L^(n) and the best candidate strategy.

    Outcome probabilities: p(z) = (u_A (x) E_{z_1} .. (x) t ..)(Y) is a
    nonnegative combination of product-facet values of Y, so p(z) >=
    -(const) * |min facet| whenever Y passed the outer checks; a branch
    with p(z) < -p_neg_tol is therefore an ERROR (raised), a branch with
    |p(z)| <= p_floor is skipped (its contribution to the average is
    bounded by p_floor * max|Val| and is recorded as skipped mass).
    """
    dA, dB = D.dA, D.dB
    T = Y.reshape((dA,) + (dB,) * n)
    E, s = sic_measurement(D)
    nPsi = len(E)
    out = {"n": n, "s": s, "n_effects": nPsi, "per_m": {},
           "best": -np.inf, "best_m": None, "best_z": None,
           "best_assignment": None, "best_xi": None, "best_N": None,
           "min_slack": np.inf, "max_eq_residual": 0.0,
           "min_outcome_probability": np.inf, "n_skipped_branches": 0,
           "skipped_positive_mass": 0.0,
           "tolerances": dict(candidate_slack=tol, p_floor=p_floor,
                              p_negative=p_neg_tol, p_sum=TOL_PSUM)}
    for m in range(n):
        rec_m = {"assignments": []}
        for meas, kept, marg in role_assignments(n, m, all_assignments):
            # --- marginalise the unmeasured, unkept copies with t ---------
            V = T
            axes = {j: 1 + j for j in range(n)}      # slot -> current axis
            for j in sorted(marg, reverse=True):
                V = np.tensordot(V, D.t, axes=([axes[j]], [0]))
                for jj in axes:
                    if axes[jj] > axes[j]:
                        axes[jj] -= 1
                del axes[j]
            # V now has axes (A, remaining slots in increasing order)
            meas_axes = [axes[j] for j in meas]
            kept_axis = axes[kept]
            best, best_z, avg, psum, cnt = -np.inf, None, 0.0, 0.0, 0
            worst_slack, worst_eq = np.inf, 0.0
            pmin, n_skip, skip_mass = np.inf, 0, 0.0
            best_pair = None
            for z in itertools.product(range(nPsi), repeat=m):
                X = V
                # contract the measured slots (highest axis first, so the
                # lower axes keep their positions)
                order = sorted(range(m), key=lambda q: -meas_axes[q])
                ax = {q: meas_axes[q] for q in range(m)}
                ka = kept_axis
                for q in order:
                    X = np.tensordot(X, E[z[q]], axes=([ax[q]], [0]))
                    if ka > ax[q]:
                        ka -= 1
                    for qq in ax:
                        if ax[qq] > ax[q]:
                            ax[qq] -= 1
                if X.shape != (dA, dB):
                    raise RuntimeError(f"contraction shape {X.shape}")
                p = float(D.uA @ X @ D.t)
                psum += p
                cnt += 1
                pmin = min(pmin, p)
                if p < -p_neg_tol:
                    raise RuntimeError(f"negative outcome probability "
                                       f"p(z)={p:.3e} at m={m}, z={z}")
                if p <= p_floor:
                    n_skip += 1
                    skip_mass += max(p, 0.0)
                    continue
                xi = (X @ D.t) / p
                N = (D.uA @ X) / p
                slack, eq = candidate_checks(D, xi, N)
                worst_slack = min(worst_slack, slack)
                worst_eq = max(worst_eq, eq)
                if not (slack > -tol and eq < tol):
                    raise RuntimeError(f"rounded candidate infeasible at the "
                                       f"stated tolerance {tol:.0e} (slack "
                                       f"{slack:.3e}, eq {eq:.3e})")
                v = float(xi @ D.W @ N)
                avg += p * v
                if v > best:
                    best, best_z, best_pair = v, z, (xi.copy(), N.copy())
            if not abs(psum - 1.0) < TOL_PSUM:
                raise RuntimeError(f"sum_z p(z) = {psum!r} != 1")
            rec_m["assignments"].append(
                {"measured": list(meas), "kept": kept, "marginalised":
                 list(marg), "best": best, "best_z": list(best_z),
                 "avg": avg / psum, "n_outcomes": cnt, "p_sum": psum,
                 "min_outcome_probability": pmin,
                 "n_skipped_branches": n_skip,
                 "skipped_positive_mass": skip_mass,
                 "min_slack": worst_slack, "max_eq_residual": worst_eq})
            out["min_slack"] = min(out["min_slack"], worst_slack)
            out["max_eq_residual"] = max(out["max_eq_residual"], worst_eq)
            out["min_outcome_probability"] = min(
                out["min_outcome_probability"], pmin)
            out["n_skipped_branches"] += n_skip
            out["skipped_positive_mass"] += skip_mass
            if best > out["best"]:
                out.update(best=best, best_m=m, best_z=list(best_z),
                           best_assignment=(list(meas), kept, list(marg)),
                           best_xi=best_pair[0], best_N=best_pair[1])
        bests = [a["best"] for a in rec_m["assignments"]]
        avgs = [a["avg"] for a in rec_m["assignments"]]
        rec_m.update(best=max(bests), best_spread=max(bests) - min(bests),
                     avg=avgs[int(np.argmax(bests))],
                     avg_spread=max(avgs) - min(avgs),
                     best_z=rec_m["assignments"][int(np.argmax(bests))]
                     ["best_z"])
        out["per_m"][m] = rec_m
    return out


# ---------------------------------------------------------------------------
# 2.  Dihedral (resp. Z_2) group average of an unreduced extension
# ---------------------------------------------------------------------------

def symmetry_matrices(D: dh.DirectData):
    """Linear actions (M_A on R^{d_A}, M_B on R^{d_B}) of the symmetry
    group of K_B under which the level-n data are invariant:
    states rho -> R rho, effects e -> R^{-T} e."""
    K = D.K
    dP = D.dP
    Rs = []
    if dP == 3:                                     # regular polygon
        mpoly = len(K.gens)
        ang = 2 * np.pi / mpoly
        for k in range(mpoly):
            c, sn = np.cos(k * ang), np.sin(k * ang)
            R = np.array([[1, 0, 0], [0, c, -sn], [0, sn, c]])
            Rs.append(R)
            Rs.append(R @ np.diag([1.0, 1.0, -1.0]))
    elif dP == 2:                                   # classical bit: Z_2
        Rs.append(np.eye(2))
        Rs.append(np.array([[1.0, 0.0], [1.0, -1.0]]))  # swaps vertices
    else:
        raise ValueError
    mats = []
    for R in Rs:
        # sanity: R permutes the vertices, R^{-T} the dual rays, fixes u,tau
        Rinvt = np.linalg.inv(R).T
        ok = all(np.min(np.abs(K.gens - R @ gv).max(axis=1)) < 1e-9
                 for gv in K.gens)
        ok &= all(np.min(np.abs(K.dual - Rinvt @ h).max(axis=1)) < 1e-9
                  for h in K.dual)
        ok &= bool(np.allclose(Rinvt @ K.u, K.u)
                   and np.allclose(R @ K.tau, K.tau))
        if not ok:
            raise RuntimeError("symmetry matrix does not preserve K")
        MA = np.kron(np.eye(3), R)                  # xi index alpha*dP + i
        MB = np.kron(Rinvt, np.eye(3))              # N  index i*3 + beta
        mats.append((MA, MB))
    return mats


def group_average(D: dh.DirectData, n: int, Y: np.ndarray) -> np.ndarray:
    dA, dB = D.dA, D.dB
    T = Y.reshape((dA,) + (dB,) * n)
    acc = np.zeros_like(T)
    mats = symmetry_matrices(D)
    for MA, MB in mats:
        S = np.tensordot(MA, T, axes=([1], [0]))            # A axis
        for j in range(n):
            S = np.moveaxis(np.tensordot(MB, S, axes=([1], [1 + j])), 0,
                            1 + j)
        acc += S
    return (acc / len(mats)).reshape(-1)


# ---------------------------------------------------------------------------
# 3.  Map to the reduced basis and reuse the library's inner_rounding
# ---------------------------------------------------------------------------

def to_reduced_symmetric(prob: lib.GameProblem, n: int,
                         Y: np.ndarray) -> np.ndarray:
    """Unreduced Y (dA, dB^n) -> library value form (dA, Sym^n(dBr)):
    B slots restricted to the ker F_B coordinates (Bred is a coordinate
    selection here) and one representative ordering per multiset."""
    dA, dB, dBr = prob.dA, prob.dB, prob.dBr
    Bred = prob.Bred
    # coordinate selection: each column of Bred is a unit vector
    sel = []
    for c in Bred.T:
        nz = np.nonzero(np.abs(c) > 0)[0]
        if not (len(nz) == 1 and abs(c[nz[0]] - 1.0) < 1e-12):
            raise RuntimeError("Bred is not a coordinate selection")
        sel.append(int(nz[0]))
    T = Y.reshape((dA,) + (dB,) * n)
    for j in range(n):
        T = np.take(T, sel, axis=1 + j)
    sym = lib.SymSpace(dBr, n)
    out = np.empty((dA, sym.sizes[n]))
    for k, M in enumerate(sym.msets[n]):
        out[:, k] = T[(slice(None),) + tuple(M)]
    return out.reshape(-1)


# ---------------------------------------------------------------------------
# 4.  Driver
# ---------------------------------------------------------------------------

def solve_direct(D: dh.DirectData, n: int, solver="choose",
                 presolve="choose"):
    LP = dh.build_level(D, n)
    sol = dh.solve_lp(LP["c"], LP["blocks"], solver=solver,
                      presolve=presolve)
    chk = dh.verify_solution(D, n, sol["x"], LP)
    return sol, chk, LP


def get_optimizer(D: dh.DirectData, n: int, opt_dir: Path, resolve: bool,
                  env: dict, src_sha: dict):
    """Return (Y, value, checks, LP, provenance).  Loads the optimizer saved
    by direct_hierarchy.py unless --resolve or no file; a fresh solve is
    saved (so that it can be loaded next time) and flagged as such."""
    stem = opt_dir / f"direct_explicit_{dh.slug(D.name)}_n{n}"
    LP = dh.build_level(D, n)           # needed for the objective c anyway
    npy = stem.with_name(stem.name + "_Y.npy")
    if npy.is_file() and not resolve:
        Y, meta = dh.load_optimizer(stem)
        if len(Y) != LP["n_vars"]:
            raise RuntimeError(f"{npy}: wrong length")
        chk = dh.verify_solution(D, n, Y, LP)
        value = float(LP["c"] @ Y)
        if abs(value - meta["value"]) > 1e-12:
            raise RuntimeError(f"{npy}: objective {value!r} != recorded "
                               f"{meta['value']!r}")
        prov = dict(origin="loaded", Y_file=npy.name, Y_dir=opt_dir.name,
                    Y_sha256=meta["Y_sha256"],
                    outer_record=meta)
        return Y, value, chk, LP, prov
    sol = dh.solve_lp(LP["c"], LP["blocks"])
    chk = dh.verify_solution(D, n, sol["x"], LP)
    flags = dh.check_flags(chk)
    saved = dh.save_optimizer(
        stem, sol["x"], dict(experiment=D.name, n=n,
                             method="explicit Kronecker LP",
                             value=sol["value"], status=sol["status"],
                             solver="choose", presolve="choose",
                             solver_options=sol["solver_options"],
                             simplex_iterations=sol["simplex_iterations"],
                             ipm_iterations=sol["ipm_iterations"],
                             checks=chk, **flags, environment=env,
                             source_sha256=src_sha,
                             written_by="direct_inner_rounding.py "
                                        "(re-solve)"))
    prov = dict(origin="re-solved here and saved", Y_file=saved["Y_file"],
                Y_dir=opt_dir.name, Y_sha256=saved["Y_sha256"],
                outer_record=dict(value=sol["value"], status=sol["status"],
                                  solver_options=sol["solver_options"],
                                  simplex_iterations=sol["simplex_iterations"],
                                  ipm_iterations=sol["ipm_iterations"],
                                  checks=chk, **flags, environment=env))
    return sol["x"], sol["value"], chk, LP, prov


def strategy_value_from_effects(prob: lib.GameProblem, xi, N) -> float:
    """Independent evaluation of the winning probability from the explicit
    states rho~_{a|x} and effects n_{b|y} (no W involved)."""
    dP = prob.dP
    Xi = xi.reshape(3, dP)                 # rows: rho_bar, rho~_{1|0}, rho~_{1|1}
    Nm = N.reshape(dP, 3)                  # cols: sum_b n_{b|y}, n_{1|0}, n_{1|1}
    val = 0.0
    for a, b, x, y in itertools.product(range(2), repeat=4):
        rho = Xi[1 + x] if a == 1 else Xi[0] - Xi[1 + x]
        eff = Nm[:, 1 + y] if b == 1 else Nm[:, 0] - Nm[:, 1 + y]
        val += prob.g[a, b, x, y] * float(eff @ rho)
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=2)
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--out", type=str, default=None,
                    help="default: direct_inner_results.json next to this "
                         "file")
    ap.add_argument("--table", type=str, default=None,
                    help="default: direct_inner_table.md next to this file")
    ap.add_argument("--ref", type=str, default=None,
                    help="results.json of the reduced pipeline (cache)")
    ap.add_argument("--optimizers-dir", type=str, default=None,
                    help="directory with the saved outer optimizers "
                         "(default: direct_optimizers/ next to this file)")
    ap.add_argument("--resolve", action="store_true",
                    help="re-solve the outer LPs instead of loading the "
                         "saved optimizers (the new ones are saved)")
    ap.add_argument("--no-all-assignments", action="store_true")
    ap.add_argument("--no-certify", action="store_true",
                    help="skip the interval-arithmetic certification")
    args = ap.parse_args()
    all_asg = not args.no_all_assignments
    out_path = Path(args.out) if args.out else HERE / "direct_inner_results.json"
    table_path = Path(args.table) if args.table else HERE / "direct_inner_table.md"
    opt_dir = Path(args.optimizers_dir) if args.optimizers_dir else \
        HERE / "direct_optimizers"

    ref_path = dh.find_file("results.json", args.ref)
    ref = json.load(open(ref_path))
    env = dh.environment()
    src_sha = {f: dh.sha256_file(HERE / f) for f in
               ("direct_inner_rounding.py", "direct_hierarchy.py",
                "direct_inner_certify.py", "gpt_definetti_games.py")
               if (HERE / f).is_file()}
    exps = dh.make_experiments()
    results = {"_provenance": dict(
        environment=env, source_sha256=src_sha,
        reduced_pipeline_cache=dict(file=os.path.relpath(ref_path, HERE),
                                    sha256=dh.sha256_file(ref_path),
                                    mtime=ref_path.stat().st_mtime,
                                    note="the reduced-pipeline values are "
                                         "read from this cache, not "
                                         "recomputed here"),
        optimizers_dir=os.path.relpath(opt_dir, HERE),
        tolerances=dict(outer_vs_cache=TOL_OUTER_VS_CACHE,
                        direct_vs_library_rounding=TOL_DIRECT_VS_LIB,
                        value_recomputation=TOL_VALUE_RECOMP,
                        candidate_slack=TOL_CANDIDATE, p_sum=TOL_PSUM,
                        p_floor=P_FLOOR, p_negative=P_NEG_TOL),
        note="L^(n)(Y) is the rounded value of the hashed optimizer Y named "
             "in each record; it depends on Y (degenerate optimal face). "
             "Floating-point values are verified feasible within the stated "
             "tolerances; L_certified is a rigorous lower bound (interval "
             "arithmetic), value_exact an exact rational value where the "
             "instance is rational.")}
    for prob, D in exps:
        if args.only and args.only not in D.name:
            continue
        print(f"== {D.name} ==", flush=True)
        results[D.name] = {}
        inst = None if args.no_certify else exact_instance_for(D)
        for n in range(1, args.nmax + 1):
            t0 = time.time()
            Y, p_val, chk, LP, prov = get_optimizer(D, n, opt_dir,
                                                    args.resolve, env,
                                                    src_sha)
            pref = ref[prob.name]["levels"][str(n)]["value"]
            iref = ref[prob.name]["inner"][str(n)]
            gam = ref[prob.name]["gamma_star_exact"]
            print(f"  n={n}: unreduced LP  p^(n) = {p_val:.12f}  "
                  f"(reduced {pref:.12f}, diff {p_val-pref:+.1e}); "
                  f"optimizer {prov['origin']} [{Path(prov['Y_file']).name}, "
                  f"sha256 {prov['Y_sha256'][:12]}...]; checks: min facet "
                  f"{chk['min_facet']:.1e}, F_B all slots "
                  f"{chk['max_FB_all_slots']:.1e}, S_n "
                  f"{chk['max_perm_violation']:.1e}", flush=True)
            checks = {}
            checks["outer_vs_cache"] = dict(
                difference=p_val - pref, tol=TOL_OUTER_VS_CACHE,
                passed=bool(abs(p_val - pref) <= TOL_OUTER_VS_CACHE))
            if not checks["outer_vs_cache"]["passed"]:
                raise RuntimeError(f"unreduced p^({n}) = {p_val!r} differs "
                                   f"from the cache {pref!r}")

            # (a) rounding of the raw unreduced optimizer
            rd = direct_inner_rounding(D, n, Y, all_assignments=all_asg)
            checks["sandwich_exact_optimum"] = dict(
                gamma_star_exact=gam, L_minus_gamma=rd["best"] - gam,
                p_minus_gamma=p_val - gam, tol=1e-9,
                passed=bool(rd["best"] <= gam + 1e-9 and p_val >= gam - 1e-9))
            if not checks["sandwich_exact_optimum"]["passed"]:
                raise RuntimeError(f"sandwich L^({n}) <= gamma* <= p^({n}) "
                                   f"violated for {D.name}")
            # independent evaluation of the best candidate
            v_eff = strategy_value_from_effects(prob, rd["best_xi"],
                                                rd["best_N"])
            v_lib = prob.value(rd["best_xi"], rd["best_N"])
            checks["best_value_recomputed"] = dict(
                from_effects=v_eff - rd["best"], library=v_lib - rd["best"],
                tol=TOL_VALUE_RECOMP,
                passed=bool(abs(v_eff - rd["best"]) < TOL_VALUE_RECOMP
                            and abs(v_lib - rd["best"]) < TOL_VALUE_RECOMP))
            if not checks["best_value_recomputed"]["passed"]:
                raise RuntimeError("best candidate value does not recompute")

            # (b) rounding of the group-averaged unreduced optimizer
            Yg = group_average(D, n, Y)
            chk_g = dh.verify_solution(D, n, Yg, LP)
            val_g = float(LP["c"] @ Yg)
            rd_g = direct_inner_rounding(D, n, Yg, all_assignments=all_asg)
            # invariance of the average itself
            inv_res = float(np.abs(group_average(D, n, Yg) - Yg).max())

            # (c) library inner_rounding on the reduced image (both Y's)
            rd_lib = lib.inner_rounding(prob, n, to_reduced_symmetric(
                prob, n, Y))
            rd_lib_g = lib.inner_rounding(prob, n, to_reduced_symmetric(
                prob, n, Yg))
            checks["direct_vs_library_rounding"] = dict(
                raw=rd_lib["best"] - rd["best"],
                group_averaged=rd_lib_g["best"] - rd_g["best"],
                tol=TOL_DIRECT_VS_LIB,
                passed=bool(abs(rd_lib["best"] - rd["best"]) <= TOL_DIRECT_VS_LIB
                            and abs(rd_lib_g["best"] - rd_g["best"])
                            <= TOL_DIRECT_VS_LIB))
            if not checks["direct_vs_library_rounding"]["passed"]:
                raise RuntimeError("direct and library rounding disagree")

            # (d) rigorous certification of the best candidates
            cert = cert_g = None
            if inst is not None:
                cert = inst.certify(rd["best_xi"], rd["best_N"])
                cert_g = inst.certify(rd_g["best_xi"], rd_g["best_N"])
                for c in (cert, cert_g):
                    if not c.get("certified"):
                        raise RuntimeError(f"certification failed: {c}")

            def per_m_table(r):
                return {str(m): {"best": r["per_m"][m]["best"],
                                 "avg": r["per_m"][m]["avg"],
                                 "best_z": r["per_m"][m]["best_z"],
                                 "best_spread_over_assignments":
                                 r["per_m"][m]["best_spread"],
                                 "avg_spread_over_assignments":
                                 r["per_m"][m]["avg_spread"],
                                 "n_outcomes":
                                 r["per_m"][m]["assignments"][0]
                                 ["n_outcomes"]}
                        for m in r["per_m"]}

            def cert_table(c):
                if c is None:
                    return None
                out = {k: c[k] for k in ("certified", "method", "eps",
                                         "L_certified", "L_certified_str",
                                         "min_slack_lower_bound",
                                         "equality_coordinates_changed_by")}
                if "exact" in c:
                    out["exact"] = c["exact"]
                return out

            rec = {
                "n": n,
                "outer_value_unreduced": p_val,
                "outer_value_reduced_cache": pref,
                "outer_checks": chk,
                "optimizer": prov,
                "sic_scale_s": rd["s"], "n_effects": rd["n_effects"],
                "enforced_checks": checks,
                "raw_optimizer": {
                    "L_n": rd["best"], "best_m": rd["best_m"],
                    "best_z": rd["best_z"],
                    "best_assignment": rd["best_assignment"],
                    "avg_at_best_m": rd["per_m"][rd["best_m"]]["avg"],
                    "per_m": per_m_table(rd),
                    "min_slack": rd["min_slack"],
                    "max_eq_residual": rd["max_eq_residual"],
                    "min_outcome_probability": rd["min_outcome_probability"],
                    "n_skipped_branches": rd["n_skipped_branches"],
                    "skipped_positive_mass": rd["skipped_positive_mass"],
                    "tolerances": rd["tolerances"],
                    "best_xi": rd["best_xi"].tolist(),
                    "best_N": rd["best_N"].tolist(),
                    "value_from_effects": v_eff,
                    "certificate": cert_table(cert),
                    "library_inner_rounding_on_reduced_image": {
                        "best": rd_lib["best"], "best_m": rd_lib["best_m"],
                        "per_m_best": {str(m): d["best"] for m, d in
                                       rd_lib["per_m"].items()},
                        "per_m_avg": {str(m): d["avg"] for m, d in
                                      rd_lib["per_m"].items()}}},
                "group_averaged_optimizer": {
                    "outer_value": val_g,
                    "checks": chk_g,
                    "invariance_residual": inv_res,
                    "L_n": rd_g["best"], "best_m": rd_g["best_m"],
                    "best_z": rd_g["best_z"],
                    "avg_at_best_m": rd_g["per_m"][rd_g["best_m"]]["avg"],
                    "per_m": per_m_table(rd_g),
                    "min_slack": rd_g["min_slack"],
                    "max_eq_residual": rd_g["max_eq_residual"],
                    "min_outcome_probability": rd_g["min_outcome_probability"],
                    "n_skipped_branches": rd_g["n_skipped_branches"],
                    "skipped_positive_mass": rd_g["skipped_positive_mass"],
                    "best_xi": rd_g["best_xi"].tolist(),
                    "best_N": rd_g["best_N"].tolist(),
                    "certificate": cert_table(cert_g),
                    "library_inner_rounding_on_reduced_image": {
                        "best": rd_lib_g["best"],
                        "best_m": rd_lib_g["best_m"],
                        "per_m_best": {str(m): d["best"] for m, d in
                                       rd_lib_g["per_m"].items()},
                        "per_m_avg": {str(m): d["avg"] for m, d in
                                      rd_lib_g["per_m"].items()}}},
                "reduced_pipeline_results_json": iref,
                "gamma_star_exact": gam,
                "seconds": time.time() - t0,
            }
            results[D.name][str(n)] = rec
            cs = (f"; certified >= {fmt_down(cert['L_certified'])}"
                  + (f" (exact {cert['exact']['value_exact']})"
                     if cert and cert.get("exact", {}).get("value_exact_is_nice")
                     else "")) if cert else ""
            print(f"        rounding of the raw unreduced optimizer:      "
                  f"L^({n}) = {rd['best']:.10f}  (m*={rd['best_m']}, "
                  f"z*={rd['best_z']}), per-m best "
                  f"{[round(rd['per_m'][m]['best'], 10) for m in rd['per_m']]}"
                  f", per-m avg "
                  f"{[round(rd['per_m'][m]['avg'], 10) for m in rd['per_m']]}"
                  f", min slack {rd['min_slack']:.1e}, eq res "
                  f"{rd['max_eq_residual']:.1e}, min p(z) "
                  f"{rd['min_outcome_probability']:.1e}, skipped "
                  f"{rd['n_skipped_branches']} (mass "
                  f"{rd['skipped_positive_mass']:.1e}), spread over "
                  f"assignments "
                  f"{max(rd['per_m'][m]['best_spread'] for m in rd['per_m']):.1e}"
                  f"; library on reduced image: {rd_lib['best']:.10f}{cs}",
                  flush=True)
            cs = (f"; certified >= {fmt_down(cert_g['L_certified'])}"
                  + (f" (exact {cert_g['exact']['value_exact']})"
                     if cert_g and cert_g.get("exact", {}).get("value_exact_is_nice")
                     else "")) if cert_g else ""
            print(f"        rounding of the group-averaged optimizer:      "
                  f"L^({n}) = {rd_g['best']:.10f}  (m*={rd_g['best_m']}), "
                  f"per-m best "
                  f"{[round(rd_g['per_m'][m]['best'], 10) for m in rd_g['per_m']]}"
                  f", per-m avg "
                  f"{[round(rd_g['per_m'][m]['avg'], 10) for m in rd_g['per_m']]}"
                  f"; objective {val_g:.12f}, min facet "
                  f"{chk_g['min_facet']:.1e}; library on reduced image: "
                  f"{rd_lib_g['best']:.10f}{cs}", flush=True)
            print(f"        reduced pipeline (results.json cache):         "
                  f"L^({n}) = {iref['best']:.10f}  (m*={iref['best_m']}), "
                  f"per-m best {iref['per_m_best']}", flush=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=1, default=float)

    # ---- markdown table --------------------------------------------------
    lines = ["| K_B | n | p^(n) unreduced | L^(n)(Y): raw unreduced optimizer "
             "Y (m*) | rigorous lower bound from Y | L^(n): group-averaged "
             "Y (m*) | L^(n): reduced pipeline, results.json cache (m*) | "
             "gamma* (exact) | optimizer file (sha256) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, levels in results.items():
        if name.startswith("_"):
            continue
        for n, r in levels.items():
            a, g, i = (r["raw_optimizer"], r["group_averaged_optimizer"],
                       r["reduced_pipeline_results_json"])
            c = a.get("certificate")
            if c is None:
                cs = "-"
            elif c.get("exact") and c["exact"].get("value_exact_is_nice"):
                cs = f"= {c['exact']['value_exact']} (exact)"
            else:
                cs = f">= {fmt_down(c['L_certified'])}"
            lines.append(
                f"| {name} | {n} | {r['outer_value_unreduced']:.10f} | "
                f"{a['L_n']:.10f} ({a['best_m']}) | {cs} | "
                f"{g['L_n']:.10f} ({g['best_m']}) | "
                f"{i['best']:.10f} ({i['best_m']}) | {r['gamma_star_exact']:.10f} | "
                f"{Path(r['optimizer']['Y_file']).name} "
                f"({r['optimizer']['Y_sha256'][:12]}) |")
    with open(table_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"written {out_path}, {table_path}")


if __name__ == "__main__":
    main()
