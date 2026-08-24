"""
direct_hierarchy_tree.py -- the unreduced level-n LP with the COMPLETE facet
family, written with named intermediate contractions.

Same problem as direct_hierarchy.py (full tensor variable Y, permutation
equalities, normalisation, F_B, all facets and all conic G_A rows).  The
only difference is how the exponentially many rows

    (phi (x) psi_{k_1} (x) ... (x) psi_{k_n})(Y) >= 0

are written down.  Instead of one explicit Kronecker row per tuple (whose
nonzero count grows like 4^(n+1)), the contraction is spelled out slot by
slot with auxiliary variables

    Z_{phi}             = (phi (x) id^{n})(Y)              in R^{d_B^n}
    Z_{phi,k_1}         = (psi_{k_1} (x) id^{n-1})(Z_{phi})  in R^{d_B^{n-1}}
    ...
    Z_{phi,k_1..k_{n-1}} in R^{d_B}
    psi_{k_n}( Z_{phi,k_1..k_{n-1}} ) >= 0                 for all k_n,

i.e. exactly the nested sums one would type into a modelling language.
Eliminating the Z's returns the explicit rows, so this LP has the same
feasible Y's and the same optimum; nothing is symmetrised, averaged or
parametrised away, and EVERY facet is present.  Its size is polynomially
larger in the number of rows but its nonzero count is only ~5 per row,
which is what makes n = 4 for the polygon theories fit into a few GB.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

import highspy

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from direct_hierarchy import (DirectData, kron_chain, make_experiments,  # noqa: E402
                              transposition_rows, verify_solution,
                              find_file, save_optimizer, environment,
                              sha256_file, slug, solver_options,
                              check_flags, FEAS_TOL, EQ_TOL,
                              _json_default as dh_json_default)


def build_tree_lp(D: DirectData, n: int, verbose=True):
    dA, dB = D.dA, D.dB
    N = dA * dB ** n
    heads = D.Phi if D.GA is None else np.vstack([D.Phi, D.GA])
    nH, nPsi = len(heads), len(D.Psi)
    inf = np.inf

    # ---- Y-only rows ------------------------------------------------------
    c = kron_chain([D.W.reshape(1, -1)] + [D.t] * (n - 1)).toarray().ravel()
    A_norm = kron_chain([D.uA] + [D.t] * n)
    I_A = sparse.identity(dA, format="csr")
    I_B = sparse.identity(dB, format="csr")
    A_F = kron_chain([I_A, D.F] + [I_B] * (n - 1))
    A_F = A_F[np.nonzero(np.diff(A_F.indptr) > 0)[0]]
    A_perm = transposition_rows(dA, dB, n)

    # ---- variable layout:  Y | Z^{(1)} | Z^{(2)} | ... | Z^{(n)} ---------
    # level l (1..n): one block per (head, k_1..k_{l-1}) prefix, of size
    # d_B^{n-l+1}; block index = prefix rank in lexicographic order
    n_prefix = [nH * nPsi ** (l - 1) for l in range(1, n + 1)]
    blk = [dB ** (n - l + 1) for l in range(1, n + 1)]
    offs = [N]
    for l in range(n):
        offs.append(offs[-1] + n_prefix[l] * blk[l])
    n_vars = offs[-1]

    blocks = []                      # CSR blocks (built directly, no COO)
    lo_list, up_list = [], []
    n_rows_tree = 0

    def emit_block(col_arrays, val_arrays, nrows, lo, up):
        """Rows r = 0..nrows-1 with entries (col_arrays[q][r], val_arrays[q][r])
        for q = 0..Q-1: build the CSR block directly."""
        nonlocal n_rows_tree
        Q = len(col_arrays)
        indices = np.stack(col_arrays, axis=1).astype(np.int32).ravel()
        data = np.stack(val_arrays, axis=1).astype(np.float64).ravel()
        indptr = np.arange(0, Q * nrows + 1, Q, dtype=np.int64)
        blocks.append(sparse.csr_matrix((data, indices, indptr),
                                        shape=(nrows, n_vars)))
        lo_list.append(np.full(nrows, lo)); up_list.append(np.full(nrows, up))
        n_rows_tree += nrows

    # level 1:  Z_{h}[j] - sum_alpha head_h[alpha] Y[alpha, j] = 0
    #   for all h and all multi-indices j in [d_B]^n
    M = dB ** n
    for hh in range(nH):
        base = offs[0] + hh * blk[0]
        nz = np.nonzero(heads[hh])[0]
        rr = np.arange(M)
        cols = [base + rr]; vals = [np.ones(M)]
        for al in nz:
            cols.append(al * M + rr); vals.append(np.full(M, -heads[hh][al]))
        emit_block(cols, vals, M, 0.0, 0.0)
    # levels 2..n:  Z_{p,k}[j'] - sum_i psi_k[i] Z_p[i, j'] = 0
    #   (vectorised over all prefixes p of the previous level)
    for l in range(2, n + 1):
        M_in, M_out = blk[l - 2], blk[l - 1]           # d_B^{n-l+2}, d_B^{n-l+1}
        n_in = n_prefix[l - 2]
        pp = np.repeat(np.arange(n_in), M_out)          # prefix of each row
        rr = np.tile(np.arange(M_out), n_in)            # j' of each row
        for k in range(nPsi):
            nz = np.nonzero(D.Psi[k])[0]
            base_out = offs[l - 1] + (pp * nPsi + k) * M_out + rr
            cols = [base_out]; vals = [np.ones(n_in * M_out)]
            for i in nz:
                cols.append(offs[l - 2] + pp * M_in + i * M_out + rr)
                vals.append(np.full(n_in * M_out, -D.Psi[k][i]))
            emit_block(cols, vals, n_in * M_out, 0.0, 0.0)
    n_eq_tree = n_rows_tree
    # facet rows:  psi_k( Z_{p} ) >= 0  for every level-n prefix p and k
    M_in = blk[n - 1]                                  # = d_B
    n_in = n_prefix[n - 1]
    for k in range(nPsi):
        nz = np.nonzero(D.Psi[k])[0]
        pp = np.arange(n_in)
        cols = []; vals = []
        for i in nz:
            cols.append(offs[n - 1] + pp * M_in + i)
            vals.append(np.full(n_in, D.Psi[k][i]))
        emit_block(cols, vals, n_in, 0.0, inf)
    n_facet_rows = n_rows_tree - n_eq_tree

    A_tree = sparse.vstack(blocks, format="csr"); del blocks
    lo_tree = np.concatenate(lo_list); up_tree = np.concatenate(up_list)

    def padY(A):
        A = sparse.csr_matrix(A)
        return sparse.hstack([A, sparse.csr_matrix((A.shape[0],
                                                    n_vars - N))],
                             format="csr")

    A = sparse.vstack([padY(A_norm), padY(A_F), padY(A_perm), A_tree],
                      format="csr")
    lo = np.concatenate([[1.0], np.zeros(A_F.shape[0] + A_perm.shape[0]),
                         lo_tree])
    up = np.concatenate([[1.0], np.zeros(A_F.shape[0] + A_perm.shape[0]),
                         up_tree])
    cost = np.zeros(n_vars); cost[:N] = c
    info = dict(n_vars_Y=N, n_vars_aux=n_vars - N, n_rows=A.shape[0],
                n_perm=A_perm.shape[0], n_FB=A_F.shape[0],
                n_tree_eq=n_eq_tree, n_facet_rows=n_facet_rows,
                nnz=int(A.nnz))
    if verbose:
        print(f"    [n={n}] tree LP: {n_vars:,} vars ({N:,} Y + "
              f"{n_vars-N:,} aux), {A.shape[0]:,} rows "
              f"({n_facet_rows:,} facet rows), nnz {A.nnz:,}", flush=True)
    return cost, A, lo, up, info


def solve(cost, A, lo, up, presolve="choose", solver="choose",
          time_limit=None, verbose=True, box=None, n_Y=None,
          crossover="on", ipm_tol=None, require_optimal=True):
    """box: optional temporary bounds |Y_i| <= box on the Y variables and
    |Z| <= 100*box on the auxiliary ones (checked inactive afterwards).

    Raises RuntimeError unless HiGHS reports model status Optimal (a
    time-limited, infeasible or otherwise unsuccessful solve is *not* a
    hierarchy value; `require_optimal=False` returns it anyway, flagged).
    """
    import gc
    n_vars = A.shape[1]
    inf = highspy.kHighsInf
    col_lo, col_up = np.full(n_vars, -inf), np.full(n_vars, inf)
    if box is not None:
        col_lo[:n_Y], col_up[:n_Y] = -box, box
        col_lo[n_Y:], col_up[n_Y:] = -100.0 * box, 100.0 * box
    A = A.tocsc(); A.sort_indices()
    up = np.where(np.isinf(up), inf, up)
    h = highspy.Highs()
    h.setOptionValue("output_flag", verbose)
    if time_limit:
        h.setOptionValue("time_limit", float(time_limit))
    if presolve != "choose":
        h.setOptionValue("presolve", presolve)
    if solver != "choose":
        h.setOptionValue("solver", solver)
        if solver == "ipm":
            # crossover "off": the interior-point optimum is returned as is
            # (accurate to ipm_tol); this avoids HiGHS' simplex clean-up,
            # whose memory footprint exceeds the machine at n = 4
            h.setOptionValue("run_crossover", crossover)
            if ipm_tol is not None:
                h.setOptionValue("ipm_optimality_tolerance", float(ipm_tol))
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = n_vars, A.shape[0]
    lp.col_cost_ = -cost
    lp.col_lower_, lp.col_upper_ = col_lo, col_up
    lp.row_lower_, lp.row_upper_ = lo, up
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32, copy=False)
    lp.a_matrix_.index_ = A.indices.astype(np.int32, copy=False)
    lp.a_matrix_.value_ = A.data.astype(np.float64, copy=False)
    del A; gc.collect()
    st = h.passModel(lp)
    if st not in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning):
        raise RuntimeError(f"HiGHS passModel returned {st}")
    del lp; gc.collect()
    opts = solver_options(h)
    t0 = time.time()
    h.run()
    st = h.getModelStatus()
    status = h.modelStatusToString(st)
    optimal = st == highspy.HighsModelStatus.kOptimal
    if not optimal and require_optimal:
        raise RuntimeError(f"HiGHS status {status} (not optimal) -- no "
                           f"result is saved or reported")
    x = np.array(h.getSolution().col_value)
    info = h.getInfo()
    return dict(status=status, optimal=optimal, x=x,
                value=float(cost @ x), seconds=time.time() - t0,
                simplex_iterations=int(info.simplex_iteration_count),
                ipm_iterations=int(info.ipm_iteration_count),
                solver_options=opts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=str, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--presolve", type=str, default="choose")
    ap.add_argument("--solver", type=str, default="choose")
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--box", type=float, default=None)
    ap.add_argument("--crossover", type=str, default="on")
    ap.add_argument("--ipm-tol", type=float, default=None)
    ap.add_argument("--out", type=str, default=None,
                    help="result JSON (default: direct_tree_<exp>_n<n>.json "
                         "in the current directory); the optimizer goes to "
                         "<out stem>_Y.npy/.json")
    ap.add_argument("--ref", type=str, default=None,
                    help="results.json of the reduced pipeline")
    ap.add_argument("--feas-tol", type=float, default=FEAS_TOL)
    ap.add_argument("--eq-tol", type=float, default=EQ_TOL)
    args = ap.parse_args()
    ref_path = find_file("results.json", args.ref)
    ref = json.load(open(ref_path))
    env = environment()
    src_sha = {"direct_hierarchy_tree.py": sha256_file(__file__),
               "direct_hierarchy.py": sha256_file(HERE / "direct_hierarchy.py")}
    for prob, D in make_experiments():
        if D.name != args.exp:
            continue
        print(f"== {D.name}, n={args.n} (unreduced, complete facet family, "
              f"intermediate contractions as variables)", flush=True)
        t0 = time.time()
        cost, A, lo, up, info = build_tree_lp(D, args.n)
        t_build = time.time() - t0
        N = info["n_vars_Y"]
        # raises RuntimeError unless the model status is Optimal: a
        # time-limited or failed solve never becomes a hierarchy record
        sol = solve(cost, A, lo, up, presolve=args.presolve,
                    solver=args.solver, time_limit=args.time_limit,
                    box=args.box, n_Y=N, crossover=args.crossover,
                    ipm_tol=args.ipm_tol, require_optimal=True)
        Y = sol["x"][:N]
        if args.box is not None:
            mx = float(np.abs(Y).max()); mz = float(np.abs(sol["x"][N:]).max())
            print(f"  box check: max|Y| = {mx:.6f} (box {args.box}), "
                  f"max|Z| = {mz:.6f} (bound {100*args.box})")
            if not (mx < args.box - 1e-6 and mz < 100 * args.box - 1e-6):
                raise RuntimeError("temporary bound active -- rerun with a "
                                   "larger --box")
        chk = verify_solution(D, args.n, Y, None)
        flags = check_flags(chk, args.feas_tol, args.eq_tol)
        pref = ref[prob.name]["levels"][str(args.n)]["value"]
        out = Path(args.out) if args.out else \
            Path.cwd() / f"direct_tree_{slug(D.name)}_n{args.n}.json"
        stem = out.with_suffix("")
        prov = save_optimizer(
            stem, Y, dict(experiment=D.name, n=args.n,
                          method="tree LP (complete facet family)",
                          value=sol["value"], status=sol["status"],
                          solver=args.solver, presolve=args.presolve,
                          crossover=args.crossover, ipm_tol=args.ipm_tol,
                          box=args.box, solver_options=sol["solver_options"],
                          simplex_iterations=sol["simplex_iterations"],
                          ipm_iterations=sol["ipm_iterations"],
                          checks=chk, **flags, environment=env,
                          source_sha256=src_sha))
        res = dict(n=args.n, value=sol["value"], reduced_value=pref,
                   difference=sol["value"] - pref, status=sol["status"],
                   box=args.box, solver=args.solver, presolve=args.presolve,
                   crossover=args.crossover, ipm_tol=args.ipm_tol,
                   solver_options=sol["solver_options"],
                   simplex_iterations=sol["simplex_iterations"],
                   ipm_iterations=sol["ipm_iterations"],
                   build_seconds=t_build, solve_seconds=sol["seconds"],
                   Y_file=prov["Y_file"], Y_sha256=prov["Y_sha256"],
                   environment=env, source_sha256=src_sha,
                   reference_file=ref_path.name,
                   reference_sha256=sha256_file(ref_path),
                   **info, **chk, **flags)
        print(f"  p^({args.n}) direct = {sol['value']:.10f}   reduced = "
              f"{pref:.10f}   diff = {res['difference']:+.2e}   "
              f"[{sol['status']}, build {t_build:.0f}s, solve "
              f"{sol['seconds']:.0f}s]")
        print("  checks:", chk, "->",
              "all pass" if flags["all_checks_pass"] else "CHECK FAILED",
              f"(tol {args.feas_tol:.0e}/{args.eq_tol:.0e})")
        with open(out, "w") as f:
            json.dump(res, f, indent=1, default=dh_json_default)
        print(f"written {out} and {prov['Y_file']}")


if __name__ == "__main__":
    main()
