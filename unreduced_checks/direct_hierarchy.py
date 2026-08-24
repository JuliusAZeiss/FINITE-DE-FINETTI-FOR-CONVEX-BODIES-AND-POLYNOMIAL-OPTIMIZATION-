"""
direct_hierarchy.py -- the level-n LP of the hierarchy WITHOUT any reduction
==========================================================================

Companion check for the numerics of

    J. A. Zeiss, G. Kossmann, R. Schwonnek, M. Plavala,
    "Finite de Finetti for convex bodies and polynomial optimization",
    arXiv:2601.15184, Sec. 6.

The reproduction pipeline (gpt_definetti_games.py / reproduce.py) solves
every level n of the hierarchy after three exact reductions (multiset
basis for the permutation symmetry, ker F_B parametrisation of the local
equalities, dihedral orbit averaging).  This file solves the *same*
level-n program with none of them:

    variables     the full extension  Y  in  R^{d_A} (x) (R^{d_B})^{(x) n}
                  (one free real per tensor entry, d_A * d_B^n of them)
    objective     max  < W (x) t (x) ... (x) t , Y >
    constraints   (i)   (phi (x) psi_{k_1} (x) ... (x) psi_{k_n})(Y) >= 0
                        for EVERY phi in ext(C_asm^*) and EVERY ordered
                        tuple (k_1, ..., k_n) of extreme rays of C_mm^*
                        [membership in C_asm (x)^ C_mm^{(x)^ n}]
                  (ii)  Y = (id_A (x) S_sigma) Y   for the adjacent
                        transpositions sigma = (j j+1), j = 1..n-1, which
                        generate S_n     [permutation invariance]
                  (iii) (u_A (x) t^{(x) n})(Y) = 1   [normalisation]
                  (iv)  (id_A (x) F_B (x) id^{(x)(n-1)})(Y) = 0
                        [local equality (2b): multimeter normalisation]
                  (v)   (G_A (x) psi_{k_1} (x) ... (x) psi_{k_n})(Y) >= 0
                        for every ordered tuple  [local inequality (2c),
                        imposed in the conic sense; 4th experiment only]

Every row is built literally as a (sparse) Kronecker product of the
defining vectors and the whole matrix is handed to HiGHS.  Nothing is
symmetrised, averaged, or parametrised away.  For the polygon theories
the explicit facet matrix has |ext C_asm^*| |ext C_mm^*|^n rows with
~4.2^(n+1) nonzeros each, so on a machine with a few GB of RAM this
literal formulation reaches n <= 3 (n <= 5 for the classical bit); the
figures are printed by `size_table()`.

The problem data (W, u_A, t, F_B, G_A, the extreme rays) are rebuilt here
from the definitions and checked to coincide with the library's data, so
that both routes provably solve the same instance.

Provenance (added 2026-08-19 after the code audit): every returned
optimizer Y is saved to `<save-dir>/direct_explicit_<exp>_n<n>_Y.npy`
together with a side-car JSON (`..._Y.json`: SHA-256 of the array, the
objective, HiGHS model status, solver options, iteration counts, the
a-posteriori residuals with explicit pass/fail flags and tolerances, the
environment and the SHA-256 of this source file).  The inner-rounding
script loads these files instead of re-solving.  All input files are
resolved relative to this file's directory (or explicitly via --ref), so
the documented commands work from any working directory.

Usage:
    python3 direct_hierarchy.py                 # all feasible levels
    python3 direct_hierarchy.py --max-nnz 2e8   # memory guard (default 1.5e8)
    python3 direct_hierarchy.py --nmax 2 --out direct_results_n2.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse

import highspy

# --- file resolution: this file's directory first, then its parent --------
HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gpt_definetti_games as lib  # noqa: E402


def find_file(name: str, explicit: str | None = None) -> Path:
    """Resolve an input file: an explicit path if given, else the current
    directory, this file's directory, its parent."""
    cands = [Path(explicit)] if explicit else \
        [Path.cwd() / name, HERE / name, HERE.parent / name]
    for c in cands:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"{name!r} not found; looked in "
                            f"{[str(c) for c in cands]} (use --ref)")


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_array(Y: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(Y, dtype=np.float64).tobytes()).hexdigest()


def environment() -> dict:
    return dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, highspy=highspy.Highs().version(),
                platform=platform.platform(), machine=platform.machine())


def slug(name: str) -> str:
    return name.replace(" ", "_").replace("+", "_")


def save_optimizer(path_stem, Y: np.ndarray, meta: dict) -> dict:
    """Write <stem>_Y.npy and <stem>_Y.json; return the provenance dict."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    npy = path_stem.with_name(path_stem.name + "_Y.npy")
    np.save(npy, np.asarray(Y, dtype=np.float64))
    # the file is referred to by name (relative to its directory), so the
    # record stays valid when the folder is moved
    prov = dict(Y_file=npy.name, Y_sha256=sha256_array(Y), **meta)
    with open(path_stem.with_name(path_stem.name + "_Y.json"), "w") as f:
        json.dump(prov, f, indent=1, default=_json_default)
    return prov


def _json_default(o):
    """json.dump helper: numpy scalars -> float, inf -> string."""
    try:
        f = float(o)
    except (TypeError, ValueError):
        return str(o)
    return f if np.isfinite(f) else str(f)


def _finite(d: dict) -> dict:
    """Replace +-inf values (e.g. HiGHS time_limit) by strings for JSON."""
    return {k: (v if not (isinstance(v, float) and not np.isfinite(v))
                else str(v)) for k, v in d.items()}


def load_optimizer(path_stem):
    """Load <stem>_Y.npy + side-car; verify the SHA-256; return (Y, meta)."""
    path_stem = Path(path_stem)
    npy = path_stem.with_name(path_stem.name + "_Y.npy")
    side = path_stem.with_name(path_stem.name + "_Y.json")
    Y = np.load(npy)
    meta = json.load(open(side))
    if sha256_array(Y) != meta["Y_sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {npy}")
    return Y, meta


def _check(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# 1.  Problem data, rebuilt from the definitions
# ---------------------------------------------------------------------------

def cs_data():
    """CS^1_{2,2} in coordinates (column sum, M[a=1|x=0], M[a=1|x=1]).
    Returns (vertices, extreme effects eps^{(x)}_a, unit)."""
    verts = np.array([[1.0, a, b] for a in (0, 1) for b in (0, 1)])
    eps = {}
    for x in (0, 1):
        e1 = np.zeros(3); e1[1 + x] = 1.0            # eps^{(x)}_1 = M[1|x]
        eps[(x, 1)] = e1
        eps[(x, 0)] = np.array([1.0, 0.0, 0.0]) - e1  # eps^{(x)}_0 = 1 - M[1|x]
    E = np.array([eps[(0, 0)], eps[(0, 1)], eps[(1, 0)], eps[(1, 1)]])
    return verts, E, eps, np.array([1.0, 0.0, 0.0])


class DirectData:
    """All static data of the game problem, in the conventions of the paper.

    K   : local state space (Cone: gens = vertices, dual = extreme effects,
          u = order unit, tau = interior reference state)
    g   : g[a,b,x,y] = pi(x,y) V(a,b,x,y)
    GA  : optional local inequality rows on the assemblage (covectors)
    """

    def __init__(self, K, g, GA=None, name=""):
        self.K, self.g, self.name = K, g, name
        _, E, eps, uS = cs_data()
        dP = K.gens.shape[1]
        self.dP, self.dA, self.dB = dP, 3 * dP, dP * 3

        # ext(C_asm^*) = ext(S^*) (x) ext(P^*):   C_asm = S (x)^ P
        self.Phi = np.array([np.kron(e, h) for e in E for h in K.dual])
        # ext(C_mm^*)  = ext(P) (x) ext(S^*):     C_mm  = P^* (x)^ S
        self.Psi = np.array([np.kron(gv, e) for gv in K.gens for e in E])
        # order unit of C_asm and base functional of C_mm
        self.uA = np.kron(uS, K.u)
        self.t = np.kron(K.tau, uS)
        # multimeter normalisation as a homogeneous linear map F_B:
        #   F_B(N)_l = N[l,0] - u_l * tau(N[:,0])      (=0  <=>  N[:,0] || u)
        F = np.zeros((dP, self.dB))
        for l in range(dP):
            for i in range(dP):
                F[l, 3 * i] = (1.0 if i == l else 0.0) - K.u[l] * K.tau[i]
        self.F = F
        # bilinear objective  sum_{abxy} g_abxy  n_{b|y}( rho~_{a|x} )
        #   rho~_{a|x} = (eps^x_a (x) id)(xi),  n_{b|y} = (id (x) eps^y_b)(N)
        W = np.zeros((self.dA, self.dB))
        for a, b, x, y in itertools.product(range(2), repeat=4):
            if g[a, b, x, y] == 0.0:
                continue
            for al in range(3):
                for be in range(3):
                    coef = g[a, b, x, y] * eps[(x, a)][al] * eps[(y, b)][be]
                    if coef != 0.0:
                        for i in range(dP):
                            W[al * dP + i, i * 3 + be] += coef
        self.W = W
        self.GA = None if GA is None else np.atleast_2d(np.asarray(GA, float))
        # numerical zeros of the polygon construction (e.g. cos(pi/2) ~ 6e-17)
        # are set to exact zeros; this changes the instance by < 1e-15 and
        # keeps the explicit matrices as sparse as they mathematically are
        for name in ("Phi", "Psi", "uA", "t", "F", "W", "GA"):
            arr = getattr(self, name)
            if arr is not None:
                arr = np.array(arr, dtype=float)
                arr[np.abs(arr) < 1e-13] = 0.0
                setattr(self, name, arr)

    # ---- consistency with the library (same instance!) -------------------
    def assert_same_as_library(self, prob: lib.GameProblem, tol=1e-12):
        """Raises ValueError (not `assert`, so it survives `python -O`)."""
        def same_rows(A, B):
            A = np.asarray(A); B = np.asarray(B)
            if A.shape != B.shape:
                return False
            return all(np.min(np.abs(B - a).max(axis=1)) < tol for a in A) \
                and all(np.min(np.abs(A - b).max(axis=1)) < tol for b in B)
        _check(same_rows(self.Phi, prob.DR_A), "ext C_asm^* differ")
        _check(same_rows(self.Psi, prob.DR_B), "ext C_mm^* differ")
        _check(np.allclose(self.uA, prob.uA, atol=tol), "u_A differs")
        _check(np.allclose(self.t, prob.t, atol=tol), "t differs")
        _check(np.allclose(self.F, prob.F_rows, atol=tol), "F_B differs")
        _check(np.allclose(self.W, prob.W, atol=tol), "W differs")
        if self.GA is None:
            _check(prob.GA_rows is None, "G_A differs")
        else:
            _check(np.allclose(self.GA, prob.GA_rows, atol=tol),
                   "G_A differs")


# ---------------------------------------------------------------------------
# 2.  The literal level-n LP
# ---------------------------------------------------------------------------

def _as_sparse(M):
    if sparse.issparse(M):
        return sparse.csr_matrix(M)
    return sparse.csr_matrix(np.atleast_2d(np.asarray(M, dtype=float)))


def kron_chain(mats):
    """Sparse Kronecker product  mats[0] (x) mats[1] (x) ... (row-major)."""
    out = _as_sparse(mats[0])
    for M in mats[1:]:
        out = sparse.kron(out, _as_sparse(M), format="csr")
    return out


def transposition_rows(dA: int, dB: int, n: int, all_pairs: bool = False):
    """Equality rows  Y[alpha, i_1..i_n] - Y[alpha, i_sigma(1)..i_sigma(n)] = 0
    for the adjacent transpositions sigma = (j j+1) (or all transpositions if
    all_pairs).  Each equation is listed once."""
    N = dA * dB ** n
    idx = np.arange(N).reshape((dA,) + (dB,) * n)
    pairs = [(j, j + 1) for j in range(n - 1)] if not all_pairs else \
        [(j, k) for j in range(n) for k in range(j + 1, n)]
    rows_i, rows_j = [], []
    for j, k in pairs:
        perm = np.swapaxes(idx, 1 + j, 1 + k).ravel()
        src = np.arange(N)
        m = src < perm
        rows_i.append(src[m]); rows_j.append(perm[m])
    if not rows_i:
        return sparse.csr_matrix((0, N))
    ri = np.concatenate(rows_i); rj = np.concatenate(rows_j)
    R = len(ri)
    data = np.concatenate([np.ones(R), -np.ones(R)])
    rows = np.concatenate([np.arange(R), np.arange(R)])
    cols = np.concatenate([ri, rj])
    return sparse.csr_matrix((data, (rows, cols)), shape=(R, N))


def build_level(D: DirectData, n: int, all_pairs: bool = False,
                fB_all_slots: bool = False):
    """Assemble objective and constraint blocks of the literal level-n LP.

    Returns dict with c (objective, maximise) and a list `blocks` of
    (name, sparse matrix, lower, upper) row blocks, plus bookkeeping."""
    dA, dB = D.dA, D.dB
    N = dA * dB ** n
    I_B = sparse.identity(dB, format="csr")
    I_A = sparse.identity(dA, format="csr")
    inf = np.inf

    # objective:  W on (A, B_1), t on B_2..B_n
    c = kron_chain([D.W.reshape(1, -1)] + [D.t] * (n - 1)).toarray().ravel()

    blocks = []
    # (iii) normalisation
    A_norm = kron_chain([D.uA] + [D.t] * n)
    blocks.append(("normalisation", A_norm, np.ones(1), np.ones(1)))
    # (iv) local equality F_B on the first slot (or on every slot)
    slots = range(n) if fB_all_slots else [0]
    F_blocks = []
    for s in slots:
        mats = [I_A] + [I_B] * s + [D.F] + [I_B] * (n - 1 - s)
        F_blocks.append(kron_chain(mats))
    A_F = sparse.vstack(F_blocks, format="csr")
    nz = np.diff(A_F.indptr) > 0                    # drop identically-0 rows
    A_F = A_F[np.nonzero(nz)[0]]
    blocks.append(("F_B", A_F, np.zeros(A_F.shape[0]), np.zeros(A_F.shape[0])))
    # (ii) permutation invariance
    A_perm = transposition_rows(dA, dB, n, all_pairs=all_pairs)
    blocks.append(("permutation", A_perm, np.zeros(A_perm.shape[0]),
                   np.zeros(A_perm.shape[0])))
    # (v) local inequality G_A in the conic sense
    n_GA = 0
    if D.GA is not None:
        n_GA = len(D.GA) * len(D.Psi) ** n
        blocks.append(("G_A", (lambda: kron_chain([D.GA] + [D.Psi] * n)),
                       np.zeros(n_GA), np.full(n_GA, inf)))
    # (i) all product facets, ordered tuples:  Phi (x) Psi (x) ... (x) Psi
    #     (built literally, one head phi at a time so that at most one
    #      chunk of the explicit matrix lives outside the solver)
    n_chunk = len(D.Psi) ** n
    for hh in range(len(D.Phi)):
        blocks.append((f"facets[phi_{hh}]",
                       (lambda hh=hh: kron_chain([D.Phi[hh]] + [D.Psi] * n)),
                       np.zeros(n_chunk), np.full(n_chunk, inf)))
    n_facets = len(D.Phi) * n_chunk
    return dict(c=c, blocks=blocks, n_vars=N, n_facets=n_facets,
                n_GA=n_GA, n_F=A_F.shape[0], n_perm=A_perm.shape[0])


def solve_lp(c, blocks, solver="choose", verbose=False, time_limit=None,
             free_blocks=True, presolve="choose"):
    """max c.x  s.t.  lo <= A x <= up  (row blocks),  x free   (HiGHS).
    Row blocks (matrices or zero-argument callables producing them) are
    materialised one after the other, stacked into one column-wise matrix
    and passed to HiGHS in a single call (passModel), which keeps the peak
    memory low (HiGHS holds one copy of the matrix; our copies are freed
    before the solve)."""
    import gc
    N = len(c)
    inf = highspy.kHighsInf
    mats, los, ups = [], [], []
    for k in range(len(blocks)):
        name, A, lo, up = blocks[k]
        if callable(A):
            A = A()
        A = sparse.csr_matrix(A)
        _check(A.shape[0] == len(lo) == len(up), f"block {name}: shape")
        mats.append(A); los.append(np.asarray(lo, float))
        ups.append(np.where(np.isinf(up), inf, np.asarray(up, float)))
        if free_blocks:
            blocks[k] = (name, None, None, None)
    A = sparse.vstack(mats, format="csr"); del mats; gc.collect()
    lo = np.concatenate(los); up = np.concatenate(ups)
    A = A.tocsc(); gc.collect()
    A.sort_indices()
    nnz_total = int(A.nnz)
    n_rows = A.shape[0]
    h = highspy.Highs()
    if not verbose:
        h.silent()
    if time_limit is not None:
        h.setOptionValue("time_limit", float(time_limit))
    if solver != "choose":
        h.setOptionValue("solver", solver)
        if solver == "ipm":
            h.setOptionValue("run_crossover", "on")
    if presolve != "choose":
        h.setOptionValue("presolve", presolve)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = N, n_rows
    lp.col_cost_ = -np.asarray(c, float)
    lp.col_lower_, lp.col_upper_ = np.full(N, -inf), np.full(N, inf)
    lp.row_lower_, lp.row_upper_ = lo, up
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)
    del A; gc.collect()
    st = h.passModel(lp)
    if st not in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning):
        raise RuntimeError(f"HiGHS passModel returned {st}")
    del lp; gc.collect()
    opts = solver_options(h)
    t0 = time.time()
    h.run()
    st = h.getModelStatus()
    # only an optimal solve is a hierarchy value; anything else is an error
    if st != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"HiGHS status {h.modelStatusToString(st)} "
                           f"(not optimal) -- no result")
    x = np.array(h.getSolution().col_value)
    info = h.getInfo()
    out = dict(value=float(c @ x), x=x, seconds=time.time() - t0,
               simplex_iterations=int(info.simplex_iteration_count),
               ipm_iterations=int(info.ipm_iteration_count),
               status=h.modelStatusToString(st), nnz=nnz_total,
               solver_options=opts)
    del h
    return out


def solver_options(h) -> dict:
    """The HiGHS options that determine the returned optimizer."""
    out = {}
    for name in ("solver", "presolve", "run_crossover", "time_limit",
                 "primal_feasibility_tolerance",
                 "dual_feasibility_tolerance", "ipm_optimality_tolerance",
                 "simplex_strategy", "random_seed", "threads"):
        try:
            v = h.getOptionValue(name)
            out[name] = v[1] if isinstance(v, tuple) else v
        except Exception:            # option unknown in this HiGHS version
            pass
    return _finite(out)


# a-posteriori tolerances (explicit, recorded in every result record)
FEAS_TOL = 1e-8      # product facets and G_A rows:  value >= -FEAS_TOL
EQ_TOL = 1e-8        # |norm - 1|, F_B on every slot, permutation residuals


def check_flags(chk: dict, feas_tol: float = FEAS_TOL,
                eq_tol: float = EQ_TOL) -> dict:
    """Turn the residuals of verify_solution into pass/fail flags."""
    flags = dict(tolerances=dict(feasibility=feas_tol, equality=eq_tol),
                 facets_pass=bool(chk["min_facet"] >= -feas_tol),
                 equalities_pass=bool(chk["norm_residual"] <= eq_tol
                                      and chk["max_FB_all_slots"] <= eq_tol
                                      and chk["max_perm_violation"] <= eq_tol))
    if "min_GA" in chk:
        flags["GA_pass"] = bool(chk["min_GA"] >= -feas_tol)
    flags["all_checks_pass"] = all(v for k, v in flags.items()
                                   if k.endswith("_pass"))
    return flags


# ---------------------------------------------------------------------------
# 3.  A-posteriori verification of a returned extension
# ---------------------------------------------------------------------------

def all_products(heads: np.ndarray, Psi: np.ndarray, T: np.ndarray, n: int):
    """Values (h (x) psi_{k_1} (x) ... (x) psi_{k_n})(Y) for ALL heads h
    (rows of `heads`) and ALL ordered tuples, computed by contracting the
    tensor T = Y.reshape(dA, dB, .., dB) slot by slot; returns an array of
    shape (len(heads), nPsi, ..., nPsi)."""
    V = np.tensordot(heads, T, axes=([1], [0]))          # (nH, dB, ..., dB)
    for _ in range(n):
        # contract the leading remaining B-slot with all psi_k; the new
        # k-axis is appended, so after n steps the axes are (nH, k_1..k_n)
        V = np.tensordot(V, Psi, axes=([1], [1]))
    return V


def verify_solution(D: DirectData, n: int, Y: np.ndarray, LP: dict,
                    max_perms: int = 720):
    """Independent checks on the returned Y (no LP involved):
    every product facet, every G_A row, normalisation, F_B on EVERY slot,
    invariance under ALL n! permutations (or a sample of max_perms).
    The facet values are recomputed by tensor contraction, i.e. without
    holding the explicit facet matrix in memory."""
    dA, dB = D.dA, D.dB
    T = Y.reshape((dA,) + (dB,) * n)
    out = {}
    out["min_facet"] = float(all_products(D.Phi, D.Psi, T, n).min())
    if D.GA is not None:
        out["min_GA"] = float(all_products(D.GA, D.Psi, T, n).min())
    norm = np.tensordot(D.uA, T, axes=([0], [0]))
    for _ in range(n):
        norm = np.tensordot(D.t, norm, axes=([0], [0]))
    out["norm_residual"] = float(abs(norm - 1.0))
    # F_B on every slot
    worst = 0.0
    for s in range(n):
        Z = np.tensordot(D.F, T, axes=([1], [1 + s]))     # (dP, dA, dB^{n-1})
        worst = max(worst, float(np.abs(Z).max()))
    out["max_FB_all_slots"] = worst
    # full permutation invariance
    perms = list(itertools.permutations(range(n)))
    if len(perms) > max_perms:
        rng = np.random.default_rng(0)
        perms = [perms[i] for i in rng.choice(len(perms), max_perms,
                                              replace=False)]
    worst = 0.0
    for p in perms:
        Tp = np.transpose(T, (0,) + tuple(1 + i for i in p))
        worst = max(worst, float(np.abs(Tp - T).max()))
    out["max_perm_violation"] = worst
    out["n_perms_checked"] = len(perms)
    return out


# ---------------------------------------------------------------------------
# 4.  Experiments (identical to reproduce.py) and driver
# ---------------------------------------------------------------------------

def make_experiments():
    pent = lib.pentagon_selfdual()
    plain = lib.GameProblem(pent, lib.chsh_coefficients(), name="pentagon")
    GA = np.kron(lib.eps_functional(0, 1), pent.u) - 0.7 * plain.uA
    ineq = lib.GameProblem(pent, lib.chsh_coefficients(),
                           name="pentagon+ineq", GA_rows=GA[None, :],
                           GA_desc="q_{1|x=0} >= 0.7")
    exps = [
        lib.GameProblem(lib.classical_bit(), lib.chsh_coefficients(),
                        name="classical bit"),
        lib.GameProblem(lib.square_gbit(), lib.chsh_coefficients(),
                        name="square gbit"),
        plain, ineq,
    ]
    out = []
    for prob in exps:
        D = DirectData(prob.K, prob.g, GA=prob.GA_rows, name=prob.name)
        D.assert_same_as_library(prob)
        out.append((prob, D))
    return out


def predicted_nnz(D: DirectData, n: int) -> float:
    """Exact nonzero count of the explicit facet matrix (before solver)."""
    nphi = np.count_nonzero(D.Phi)
    npsi = np.count_nonzero(D.Psi)
    nnz = nphi * npsi ** n
    if D.GA is not None:
        nnz += np.count_nonzero(D.GA) * npsi ** n
    return nnz


def size_table(exps, nmax=6):
    print("explicit facet matrix (product facets + conic G_A rows): "
          "rows x cols, nonzeros (no reduction)")
    for prob, D in exps:
        for n in range(1, nmax + 1):
            rows = len(D.Phi) * len(D.Psi) ** n
            if D.GA is not None:            # the G_A rows are counted too
                rows += len(D.GA) * len(D.Psi) ** n
            print(f"  {D.name:15s} n={n}: {rows:>12,d} x {D.dA*D.dB**n:>9,d}"
                  f"   nnz = {predicted_nnz(D, n):.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-nnz", type=float, default=1.5e8,
                    help="skip levels whose explicit facet matrix has more "
                         "nonzeros than this")
    ap.add_argument("--nmax", type=int, default=5)
    ap.add_argument("--only", type=str, default=None,
                    help="substring filter on experiment name")
    ap.add_argument("--all-pairs", action="store_true",
                    help="impose ALL transpositions (default: adjacent)")
    ap.add_argument("--fB-all-slots", action="store_true",
                    help="impose F_B on every slot (default: first slot)")
    ap.add_argument("--solver", type=str, default="choose")
    ap.add_argument("--presolve", type=str, default="choose",
                    help='HiGHS presolve option ("on"/"off"/"choose")')
    ap.add_argument("--lean-above", type=float, default=4e7,
                    help="above this many nonzeros use the memory-lean "
                         "solver settings (IPM + crossover, presolve off)")
    ap.add_argument("--out", type=str, default=None,
                    help="result file (default: direct_results.json next "
                         "to this script)")
    ap.add_argument("--ref", type=str, default=None,
                    help="results.json of the reduced pipeline (default: "
                         "searched in cwd, script dir, parent)")
    ap.add_argument("--save-dir", type=str, default=None,
                    help="directory for the optimizers Y (default: "
                         "direct_optimizers/ next to this script)")
    ap.add_argument("--feas-tol", type=float, default=FEAS_TOL)
    ap.add_argument("--eq-tol", type=float, default=EQ_TOL)
    ap.add_argument("--sizes", action="store_true")
    args = ap.parse_args()

    exps = make_experiments()
    if args.sizes:
        size_table(exps)
        return
    ref_path = find_file("results.json", args.ref)
    ref = json.load(open(ref_path))
    out_path = Path(args.out) if args.out else HERE / "direct_results.json"
    save_dir = Path(args.save_dir) if args.save_dir else \
        HERE / "direct_optimizers"
    env = environment()
    src_sha = sha256_file(__file__)
    provenance = dict(source_file=Path(__file__).name,
                      source_sha256=src_sha, environment=env,
                      reference_file=ref_path.name,
                      reference_sha256=sha256_file(ref_path),
                      optimizer_dir=save_dir.name,
                      tolerances=dict(feasibility=args.feas_tol,
                                      equality=args.eq_tol),
                      note="values are floating-point LP optima (HiGHS); "
                           "residuals are checked against the stated "
                           "tolerances, not certified exactly")
    results = {"_provenance": provenance}
    for prob, D in exps:
        if args.only and args.only not in D.name:
            continue
        print(f"== {D.name} ==  d_A={D.dA}, d_B={D.dB}, "
              f"|ext C_asm^*|={len(D.Phi)}, |ext C_mm^*|={len(D.Psi)}",
              flush=True)
        results[D.name] = {}
        for n in range(1, args.nmax + 1):
            pn = predicted_nnz(D, n)
            if pn > args.max_nnz:
                print(f"  n={n}: explicit facet matrix would have "
                      f"{pn:.1e} nonzeros > --max-nnz; skipped", flush=True)
                continue
            t0 = time.time()
            LP = build_level(D, n, all_pairs=args.all_pairs,
                             fB_all_slots=args.fB_all_slots)
            t_build = time.time() - t0
            solver, presolve = args.solver, args.presolve
            if pn > args.lean_above and solver == "choose" \
                    and presolve == "choose":
                # HiGHS presolve keeps several working copies of the matrix
                # (~120 bytes/nnz); IPM+crossover without presolve needs
                # ~50 bytes/nnz and gives the same vertex optimum
                solver, presolve = "ipm", "off"
                print(f"  n={n}: {pn:.1e} nonzeros -> memory-lean solver "
                      f"settings (ipm + crossover, presolve off)", flush=True)
            sol = solve_lp(LP["c"], LP["blocks"], solver=solver,
                           presolve=presolve)
            chk = verify_solution(D, n, sol["x"], LP)
            flags = check_flags(chk, args.feas_tol, args.eq_tol)
            pref = ref[prob.name]["levels"][str(n)]["value"]
            rec = dict(n=n, value=sol["value"], reduced_value=pref,
                       difference=sol["value"] - pref,
                       status=sol["status"],
                       solver=solver, presolve=presolve,
                       solver_options=sol["solver_options"],
                       n_vars=LP["n_vars"], n_facet_rows=LP["n_facets"],
                       n_GA_rows=LP["n_GA"], n_FB_rows=LP["n_F"],
                       n_perm_rows=LP["n_perm"], nnz=int(sol["nnz"]),
                       build_seconds=t_build, solve_seconds=sol["seconds"],
                       simplex_iterations=sol["simplex_iterations"],
                       ipm_iterations=sol["ipm_iterations"], **chk, **flags)
            # save the optimizer itself (the rounding depends on it)
            prov = save_optimizer(
                save_dir / f"direct_explicit_{slug(D.name)}_n{n}",
                sol["x"], dict(experiment=D.name, n=n, method="explicit "
                               "Kronecker LP", value=sol["value"],
                               status=sol["status"], solver=solver,
                               presolve=presolve,
                               solver_options=sol["solver_options"],
                               simplex_iterations=sol["simplex_iterations"],
                               ipm_iterations=sol["ipm_iterations"],
                               checks=chk, **flags, environment=env,
                               source_sha256=src_sha))
            rec["Y_file"] = prov["Y_file"]; rec["Y_sha256"] = prov["Y_sha256"]
            if not flags["all_checks_pass"]:
                print(f"  n={n}: WARNING -- a-posteriori check failed at the "
                      f"stated tolerances: {flags}", flush=True)
            results[D.name][str(n)] = rec
            print(f"  n={n}: p^(n) direct = {sol['value']:.10f}   "
                  f"reduced = {pref:.10f}   diff = {rec['difference']:+.2e}"
                  f"\n        vars={LP['n_vars']:,}  facet rows="
                  f"{LP['n_facets']:,}  GA rows={LP['n_GA']:,}  "
                  f"F_B rows={LP['n_F']:,}  perm rows={LP['n_perm']:,}  "
                  f"nnz={sol['nnz']:.2e}\n        build {t_build:.1f}s, "
                  f"solve {sol['seconds']:.1f}s "
                  f"(simplex it {sol['simplex_iterations']}, ipm it "
                  f"{sol['ipm_iterations']});  checks: min facet "
                  f"{chk['min_facet']:.1e}, |norm-1| "
                  f"{chk['norm_residual']:.1e}, F_B all slots "
                  f"{chk['max_FB_all_slots']:.1e}, S_n invariance "
                  f"{chk['max_perm_violation']:.1e} "
                  f"({chk['n_perms_checked']} perms)"
                  + (f", min G_A {chk['min_GA']:.1e}" if 'min_GA' in chk
                     else "")
                  + f"  [{'all checks pass' if flags['all_checks_pass'] else 'CHECK FAILED'}"
                  f" at tol {args.feas_tol:.0e}/{args.eq_tol:.0e}; Y saved "
                  f"-> {Path(rec['Y_file']).name}]", flush=True)
            del LP, sol
            with open(out_path, "w") as f:
                json.dump(results, f, indent=1, default=_json_default)
    print(f"written {out_path}")


if __name__ == "__main__":
    main()
