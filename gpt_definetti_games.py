"""
gpt_definetti_games.py -- the hierarchy and the rounding scheme of

    J. A. Zeiss, G. Kossmann, R. Schwonnek, M. Plavala,
    "Finite de Finetti for convex bodies and polynomial optimization",
    arXiv:2601.15184,

applied to the assemblage value gamma*(G) of a two-player non-local game
over a polytopic state space (the paper's Section "Games" and Section
"Numerical illustration: non-local games over polygon theories").

--------------------------------------------------------------------------
Mathematical setup
--------------------------------------------------------------------------
Fix a state space K_B (a polytope, the base of a cone P in R^{d_P}) and the
game G = (V, pi) with |A| = |B| = 2 answers and |X| = |Y| = 2 questions.
The assemblage value of the game is the *bilinear* problem

    gamma*(G) = sup  sum_{abxy} g_{abxy}  n_{b|y}( rho~_{a|x} )
      over    xi  in  CS^1_{2,2} (x)^  K_B         (Alice's assemblage)
              N   in  A(K_B)^+  (x)^  CS^+_{2,2}   (Bob's multimeter)
      s.t.    (id (x) 1_CS)(N) = 1_{K_B},          (local eq. constraint F_B)

with g_{abxy} = pi(x,y) V(a,b,x,y).  This is an instance of the paper's
polynomial optimization problem with local constraints, with the two convex
bodies

    K_asm := base of  C_A := S (x)^ P        (S = CS cone, a square cone)
    K_mm  := slice of C_B := P* (x)^ S       cut out by F_B and t(N) = 1.

Level n of the hierarchy Sigma^n (theorem "Convergence of the hierarchy")
relaxes the product xi (x) N to a permutation-invariant extension
Y in C_A (x)^ C_B^{(x)^ n} (maximal tensor product), normalized, with the
local constraint F_B imposed on every copy:

    p^(n) = max  < W (x) t^{(x)(n-1)}, Y >
      s.t.  Y in C_A (x)^ C_B^{(x)^ n}           (facet inequalities)
            (u_A (x) t^{(x) n})(Y) = 1           (normalization)
            (id_A (x) F_B (x) id^{n-1})(Y) = 0   (local equalities, (2b))
            (G_A (x) id_{B^n})(Y) >= 0           (local inequalities, (2c))

The rounding scheme of the theorem "Convergent inner sequence" turns an
optimizer of level n into feasible product strategies (inner_rounding).

--------------------------------------------------------------------------
Why everything is a *linear program* here
--------------------------------------------------------------------------
For polyhedral cones the dual of the n-fold maximal tensor product is the
minimal tensor product of the duals, whose extreme rays are the products of
the extreme rays of the factors.  Hence membership in C_A (x)^ C_B^{(x)^n}
is a finite list of linear inequalities

    ( phi (x) psi_{k_1} (x) ... (x) psi_{k_n} )(Y)  >=  0,

phi in ext(C_A^*), psi_k in ext(C_B^*).  Permutation invariance lets us
store Y in the symmetric basis (one coordinate per multiset) and reduces the
inequality list to multisets (k_1 <= ... <= k_n); for the polygon theories
the dihedral symmetry of K_B reduces it further to orbit averages
(hierarchy_level_symfull).  Every reported level value is the optimum of
the LP with the COMPLETE (reduced) facet list -- no cutting planes, no
truncation.  See implementation_notes.pdf for the details.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

# =========================================================================
# 1.  Cones:  polygon state spaces and the CS ("column-stochastic") square
# =========================================================================


@dataclass
class Cone:
    """A polyhedral cone C in R^d with base K = {x in C : u(x) = 1}.

    gens : (n_g, d)  extreme rays, scaled so that u(g) = 1  (vertices of K)
    dual : (n_h, d)  extreme rays of the dual cone C^* (facet normals of C),
                     scaled so that  max_j  <h, g_j> = 1  (extremal effects)
    u    : (d,)      order unit / normalization functional, u(g_j) = 1
    tau  : (d,)      a distinguished state in relint(K)
    name : str
    """

    name: str
    gens: np.ndarray
    dual: np.ndarray
    u: np.ndarray
    tau: np.ndarray

    @property
    def dim(self) -> int:
        return self.gens.shape[1]

    def check(self, tol: float = 1e-10) -> None:
        assert np.allclose(self.gens @ self.u, 1.0, atol=tol), self.name
        M = self.gens @ self.dual.T                      # (n_g, n_h)
        assert M.min() > -tol, f"{self.name}: dual rays not positive"
        assert np.allclose(M.max(axis=0), 1.0, atol=1e-8), \
            f"{self.name}: dual rays not effect-normalized"
        assert (self.dual @ self.tau).min() > tol, \
            f"{self.name}: tau not in relint"


def polygon_cone(m: int, radius: float, phase: float = 0.0,
                 name: str | None = None) -> Cone:
    """Regular m-gon state space in R^3:  gens (1, r cos, r sin)."""
    ang = 2.0 * np.pi * np.arange(m) / m + phase
    gens = np.stack([np.ones(m), radius * np.cos(ang), radius * np.sin(ang)],
                    axis=1)
    # facet between consecutive vertices -> dual ray via cross product
    dual = []
    for j in range(m):
        h = np.cross(gens[j], gens[(j + 1) % m])
        if h @ np.array([1.0, 0.0, 0.0]) < 0:
            h = -h
        h = h / (gens @ h).max()                         # effect-normalize
        dual.append(h)
    c = Cone(name or f"{m}-gon", gens, np.array(dual),
             np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    c.check()
    return c


def pentagon_selfdual() -> Cone:
    """Regular pentagon with circumradius r_5 = 1/sqrt(cos(pi/5)).

    With this radius the pentagon is *strongly* self-dual (odd polygon):
    the extreme rays of the dual cone, normalized by their first
    coordinate, coincide with the vertices of the pentagon itself (no
    rotation; for even polygons the dual is the polygon rotated by pi/m).
    This is the normalization of Janotta et al., NJP 13, 063024 (2011).
    """
    r = 1.0 / np.sqrt(np.cos(np.pi / 5.0))
    return polygon_cone(5, r, 0.0, "pentagon")


def square_gbit() -> Cone:
    """The 'boxworld' gbit: square with vertices (1, +-1, +-1)."""
    return polygon_cone(4, np.sqrt(2.0), np.pi / 4.0, "square")


def classical_bit() -> Cone:
    """1-simplex (classical bit) in R^2."""
    gens = np.array([[1.0, 0.0], [1.0, 1.0]])
    dual = np.array([[1.0, -1.0], [0.0, 1.0]])
    c = Cone("classical bit", gens, dual,
             np.array([1.0, 0.0]), np.array([1.0, 0.5]))
    c.check()
    return c


def cs_square() -> Cone:
    """CS^1_{2,2}: column-stochastic 2x2 matrices, coordinatized in R^3 as
    (c0, c1, c2) = (column sum, M[a=1|x=0], M[a=1|x=1]).  A square cone.

    Extremal effects are the evaluation functionals eps^{(x)}_a.
    """
    gens = np.array([[1.0, a, b] for a in (0.0, 1.0) for b in (0.0, 1.0)])
    dual = np.array([
        [1.0, -1.0, 0.0],    # eps^{(0)}_0
        [0.0, 1.0, 0.0],     # eps^{(0)}_1
        [1.0, 0.0, -1.0],    # eps^{(1)}_0
        [0.0, 0.0, 1.0],     # eps^{(1)}_1
    ])
    c = Cone("CS(2,2)", gens, dual,
             np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.5, 0.5]))
    c.check()
    return c


def eps_functional(x: int, a: int) -> np.ndarray:
    """eps^{(x)}_a in the CS coordinates above (x, a in {0,1})."""
    e = np.zeros(3)
    if a == 1:
        e[1 + x] = 1.0
    else:
        e[0] = 1.0
        e[1 + x] = -1.0
    return e


# =========================================================================
# 2.  Games and the derived convex bodies K_asm, K_mm
# =========================================================================


def chsh_coefficients() -> np.ndarray:
    """g[a,b,x,y] = pi(x,y) V(a,b,x,y) for CHSH: win iff a XOR b = x AND y."""
    g = np.zeros((2, 2, 2, 2))
    for a, b, x, y in itertools.product(range(2), repeat=4):
        g[a, b, x, y] = 0.25 * float((a ^ b) == (x & y))
    return g


@dataclass
class GameProblem:
    """All static data of the level-n LPs for one (state space, game) pair.

    Conventions (see module docstring):
      xi  (Alice/assemblage)  in R^{d_A}, d_A = 3*d_P, index (alpha, i) ->
          alpha*d_P + i.  Rows of the 3 x d_P matrix Xi:
              Xi[0,:] = rho_bar,  Xi[1,:] = rho~_{a=1|x=0},
              Xi[2,:] = rho~_{a=1|x=1}.
      N   (Bob/multimeter)   in R^{d_B}, d_B = d_P*3, index (i, beta) ->
          i*3 + beta.  Columns of the d_P x 3 matrix N:
              N[:,0] = sum_b n_{b|y} (must equal 1_K),
              N[:,1] = n_{b=1|y=0},  N[:,2] = n_{b=1|y=1}.
    """

    K: Cone                       # local state space K_B
    g: np.ndarray                 # game coefficients g[a,b,x,y]
    name: str = ""
    # local inequality constraints G_A (rows: covectors on xi), item (2c):
    GA_rows: np.ndarray | None = None
    GA_desc: str = ""

    S: Cone = field(init=False)
    dP: int = field(init=False)
    dA: int = field(init=False)
    dB: int = field(init=False)
    DR_A: np.ndarray = field(init=False)   # ext(C_A^*), rows
    DR_B: np.ndarray = field(init=False)   # ext(C_B^*), rows
    uA: np.ndarray = field(init=False)     # unit of C_A
    t: np.ndarray = field(init=False)      # base functional of C_B
    F_rows: np.ndarray = field(init=False)  # F_B, rows: covectors on N
    W: np.ndarray = field(init=False)      # bilinear objective, (d_A, d_B)

    def __post_init__(self) -> None:
        self.S = cs_square()
        K, S = self.K, self.S
        self.dP = K.dim
        self.dA = 3 * self.dP
        self.dB = self.dP * 3

        # C_A = S (x)^ P :  ext(C_A^*) = ext(S^*) (x) ext(P^*)
        self.DR_A = np.array([np.kron(e, h)
                              for e in S.dual for h in K.dual])
        # C_B = P^* (x)^ S :  ext(C_B^*) = ext(P) (x) ext(S^*)
        self.DR_B = np.array([np.kron(gv, e)
                              for gv in K.gens for e in S.dual])
        self.uA = np.kron(S.u, K.u)
        self.t = np.kron(K.tau, S.u)

        # F_B(N)[l] = N[l,0] - u_P[l] * (tau_P . N[:,0])   (linear!)
        F = np.zeros((self.dP, self.dB))
        for l in range(self.dP):
            for i in range(self.dP):
                F[l, i * 3 + 0] = (1.0 if i == l else 0.0) \
                    - K.u[l] * K.tau[i]
        self.F_rows = F

        # bilinear objective  <W, xi (x) N> = sum g_abxy eps_a^x Xi N eps_b^y
        C = np.zeros((3, 3))
        for a, b, x, y in itertools.product(range(2), repeat=4):
            C += self.g[a, b, x, y] * np.outer(eps_functional(x, a),
                                               eps_functional(y, b))
        W = np.zeros((self.dA, self.dB))
        for al in range(3):
            for i in range(self.dP):
                for be in range(3):
                    W[al * self.dP + i, i * 3 + be] = C[al, be]
        self.W = W

        # ---- reduced B-slot subspace  V0 = ker(F_B) ----------------------
        # Permutation invariance + the local equality (2b) force every
        # B-slot of the extension into V0 = {N : N[:,0] in span(u_P)}, of
        # dimension 1 + 2 d_P.  Working directly in V0 removes all F-rows
        # from the LP and shrinks the symmetric variable space.
        cols = [np.kron(K.u, np.array([1.0, 0.0, 0.0]))]
        for be in (1, 2):
            for i in range(self.dP):
                e = np.zeros(self.dP); e[i] = 1.0
                eb = np.zeros(3); eb[be] = 1.0
                cols.append(np.kron(e, eb))
        self.Bred = np.array(cols).T                     # (dB, dBr)
        self.dBr = self.Bred.shape[1]
        self.DR_B_red = self.DR_B @ self.Bred
        self.t_red = self.t @ self.Bred
        self.W_red = self.W @ self.Bred

    # ---------- helpers to build / read explicit strategies -------------

    def N_from_effects(self, n10: np.ndarray, n11: np.ndarray) -> np.ndarray:
        """Multimeter vector from the two 'answer 1' effects (d_P each)."""
        N = np.zeros((self.dP, 3))
        N[:, 0] = self.K.u
        N[:, 1] = n10
        N[:, 2] = n11
        return N.reshape(-1)

    def value(self, xi: np.ndarray, N: np.ndarray) -> float:
        """Strategy value  <W, xi (x) N>  of a product pair (assemblage xi,
        multimeter N)."""
        return float(xi @ self.W @ N)


# =========================================================================
# 3.  Symmetric-tensor machinery for the level-n LP
# =========================================================================


class SymSpace:
    """Multiset bookkeeping for Sym^t(R^{dB}), t = 0..n.

    Y is stored in "value form": Y[alpha, M] is the common value of the
    tensor entry Y_{alpha, i_1..i_n} for any ordering of the multiset M.
    Single-slot contraction with a covector f is then simply
        (f .1 Y)[alpha, M'] = sum_i f_i Y[alpha, M' + {i}] .
    """

    def __init__(self, dB: int, n: int):
        self.dB, self.n = dB, n
        self.msets = [list(itertools.combinations_with_replacement(
            range(dB), t)) for t in range(n + 1)]
        self.index = [{m: k for k, m in enumerate(ms)} for ms in self.msets]
        self.sizes = [len(ms) for ms in self.msets]
        # ADD[t][m', i] = index in msets[t] of sorted(m' + (i,)),  t = 1..n
        self.ADD = [None]
        for t in range(1, n + 1):
            A = np.empty((self.sizes[t - 1], dB), dtype=np.int64)
            for k, mp in enumerate(self.msets[t - 1]):
                for i in range(dB):
                    A[k, i] = self.index[t][tuple(sorted(mp + (i,)))]
            self.ADD.append(A)

    def contract(self, v: np.ndarray, f: np.ndarray, t: int) -> np.ndarray:
        """One-slot contraction Sym^t -> Sym^{t-1} (v has trailing dim S_t)."""
        return v[..., self.ADD[t]] @ f

    def lift_covector(self, w: np.ndarray, f: np.ndarray,
                      t: int) -> np.ndarray:
        """Transpose of `contract`: covector over (.., S_{t-1}) ->
        covector over (.., S_t), i.e. w' with <w', Y> = <w, f .1 Y>."""
        out = np.zeros(w.shape[:-1] + (self.sizes[t],))
        A = self.ADD[t]
        for i in range(self.dB):
            np.add.at(out, (..., A[:, i]), w * f[i])
        return out


def build_covector(sym: SymSpace, head: np.ndarray,
                   fs: list[np.ndarray]) -> np.ndarray:
    """Covector on (d_A, S_n) representing  head (x) f_1 (x) ... (x) f_k
    (x) [remaining slots must be filled by further lifts]; here we require
    k = n and `head` a covector on d_A (S_0 case) or (d_A, d_B) (S_1)."""
    if head.ndim == 1:
        w = head[:, None]                      # (d_A, S_0)
        t0 = 0
    else:
        w = head                               # (d_A, S_1) = (d_A, d_B)
        t0 = 1
    for k, f in enumerate(fs):
        w = sym.lift_covector(w, f, t0 + 1 + k)
    return w.reshape(-1)


# =========================================================================
# 4.  The level-n LP with the complete facet list
# =========================================================================


@dataclass
class LevelResult:
    """Summary of one solved level.

    value         the LP optimum p^(n)
    n_vars        number of LP variables after the reductions
    n_eq          number of equality rows (1: the normalization)
    n_cuts        number of inequality rows (facets, resp. facet orbits)
    lp_solves     number of LP solves (1: the complete list is solved once)
    max_violation 0.0 by construction: every facet (orbit) is a row of the
                  LP, so the optimizer satisfies all of them to the solver's
                  feasibility tolerance
    seconds       wall-clock time of the solve
    certified_by  "full-LP" (plain complete list) or "sym-full-LP"
                  (complete list, dihedral orbit averaged)
    """
    n: int
    value: float
    n_vars: int
    n_eq: int
    n_cuts: int
    lp_solves: int
    max_violation: float
    seconds: float
    done: bool = True
    certified_by: str = "full-LP"


def hierarchy_level_full(prob: GameProblem, n: int, box: float = 50.0,
                         verbose: bool = True, time_limit: float = 1e30,
                         crossover: bool = False,
                         basis_file: str | None = None,
                         prev_seconds: float = 0.0,
                         return_solution: bool = False):
    """Solve the level-n LP with the *complete* facet list (multiset
    reduction only, no orbit averaging; used for the classical bit).
    Rows factor as  (head (x) p_K)  with  p_K  the symmetric covector of
    the ray multiset K; all p_K are computed by a prefix-sharing DFS, and
    rows are assembled sparsely.  Feasible because the reduced-slot rows
    are only ~5-10% dense.

    Returns (LevelResult, active_keys[, x]) where active_keys are the
    facets with (near-)zero slack at the optimum and x (with
    return_solution=True) is the optimizer in the multiset basis, the
    input of inner_rounding.
    """
    t_start = time.time()
    import highspy
    dA, dBr = prob.dA, prob.dBr
    sym = SymSpace(dBr, n)
    Sn = sym.sizes[n]
    nvars = dA * Sn
    nPsi = len(prob.DR_B_red)

    heads = [prob.DR_A[k] for k in range(len(prob.DR_A))]
    if prob.GA_rows is not None:
        heads += [prob.GA_rows[k] for k in range(len(prob.GA_rows))]
    heads = np.array(heads)
    nH = len(heads)

    c_obj = build_covector(sym, prob.W_red, [prob.t_red] * (n - 1))
    w_norm = build_covector(sym, prob.uA, [prob.t_red] * n)

    # ---- all symmetric ray covectors p_K by DFS with prefix sharing ------
    msets = list(itertools.combinations_with_replacement(range(nPsi), n))
    pvec: dict[tuple, np.ndarray] = {}

    def dfs(prefix: tuple, w: np.ndarray):
        t = len(prefix)
        if t == n:
            pvec[prefix] = w
            return
        last = prefix[-1] if prefix else 0
        for k in range(last, nPsi):
            dfs(prefix + (k,),
                sym.lift_covector(w, prob.DR_B_red[k], t + 1))

    dfs((), np.ones(1))

    # ---- sparse assembly:  row(hk, K) = kron(head_hk, p_K) ---------------
    data, indices, indptr = [], [], [0]
    keys = []
    nnz = 0
    for hk in range(nH):
        hnz = np.nonzero(np.abs(heads[hk]) > 1e-14)[0]
        hval = heads[hk][hnz]
        for K in msets:
            p = pvec[K]
            pnz = np.nonzero(np.abs(p) > 1e-14)[0]
            pval = p[pnz]
            colblock = (hnz[:, None] * Sn + pnz[None, :]).ravel()
            valblock = (hval[:, None] * pval[None, :]).ravel()
            indices.append(colblock)
            data.append(valblock)
            nnz += len(colblock)
            indptr.append(nnz)
            keys.append((hk, K))
    A = sparse.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.array(indptr)),
        shape=(len(keys), nvars))
    n_rows, n_nnz = A.shape[0], A.nnz
    if verbose:
        print(f"    [n={n}] full LP: {n_rows} facet rows, "
              f"{n_nnz/1e6:.1f}M nnz, {nvars} vars "
              f"(built in {time.time()-t_start:.0f}s)", flush=True)

    import os
    h = highspy.Highs()
    h.silent()
    INF = highspy.kHighsInf
    h.setOptionValue("time_limit", float(time_limit))
    resume = basis_file is not None and os.path.exists(basis_file)
    # dual simplex: memory-lean (the 3 GB container cannot hold IPM's
    # workspace on ~40M nnz) and its basis can be checkpointed across runs
    h.setOptionValue("solver", "simplex")
    if resume:
        h.setOptionValue("presolve", "off")
    h.addVars(nvars, np.full(nvars, -box), np.full(nvars, box))
    h.changeColsCost(nvars, np.arange(nvars, dtype=np.int32), -c_obj)
    Anorm = sparse.csr_matrix(w_norm[None, :])
    h.addRows(1, np.array([1.0]), np.array([1.0]), Anorm.nnz,
              Anorm.indptr.astype(np.int32),
              Anorm.indices.astype(np.int32), Anorm.data)
    h.addRows(A.shape[0], np.zeros(A.shape[0]), np.full(A.shape[0], INF),
              A.nnz, A.indptr.astype(np.int32),
              A.indices.astype(np.int32), A.data)
    del A, data, indices, pvec          # free before HiGHS works
    import gc; gc.collect()
    if resume:
        h.readBasis(basis_file)
        if verbose:
            print(f"    [n={n}] resuming simplex from saved basis",
                  flush=True)
    h.run()
    status = h.getModelStatus()
    if status == highspy.HighsModelStatus.kTimeLimit:
        # simplex/ipm hit the chunk limit: persist the basis and resume
        # in the next invocation (IPM first run falls back to simplex)
        if basis_file is not None:
            h.writeBasis(basis_file)
        res = LevelResult(n=n, value=float("nan"), n_vars=nvars, n_eq=1,
                          n_cuts=n_rows, lp_solves=1,
                          max_violation=float("nan"),
                          seconds=prev_seconds + time.time() - t_start,
                          done=False, certified_by="full-LP")
        return res, []
    if status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"full LP level {n}: status {status}")
    sol = h.getSolution()
    x = np.array(sol.col_value)
    value = float(c_obj @ x)
    rowval = np.array(sol.row_value)[1:]
    active = [keys[i] for i in np.nonzero(rowval < 1e-6)[0]]
    assert np.abs(x).max() < box - 1e-6
    res = LevelResult(n=n, value=value, n_vars=nvars, n_eq=1,
                      n_cuts=n_rows, lp_solves=1,
                      max_violation=0.0,
                      seconds=prev_seconds + time.time() - t_start,
                      done=True, certified_by="full-LP")
    if return_solution:
        return res, active, x
    return res, active


def dihedral_actions(prob: GameProblem):
    """Permutations induced by the dihedral symmetry group of the polygon
    K_B on (a) its dual rays h_j  and (b) its vertices g_i.  Every group
    element g maps facet rows to facet rows by relabeling:
        row(hk, K)  ->  row(g.hk, g.K),
    where g acts on the pentagon-dual index inside hk and on the vertex
    indices inside the ray labels of K.  Returns (head_perms, ray_perms).
    """
    K = prob.K
    m = len(K.gens)
    ang = 2 * np.pi / m
    mats = []
    for k in range(m):
        c, s = np.cos(k * ang), np.sin(k * ang)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        mats.append(R)
        F = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])   # reflection
        mats.append(R @ F)

    def match(vecs, images):
        perm = []
        for v in images:
            d = np.linalg.norm(vecs - v, axis=1)
            j = int(np.argmin(d))
            assert d[j] < 1e-9
            perm.append(j)
        return np.array(perm)

    head_perms, ray_perms = [], []
    nS = len(prob.S.dual)                    # CS-dual index block size
    for R in mats:
        pg = match(K.gens, K.gens @ R.T)     # vertex index permutation
        ph = match(K.dual, K.dual @ np.linalg.inv(R))  # dual-ray perm
        # heads: index hk = s * |H_K| + j  (or the extra G_A heads, fixed)
        hp = np.arange(len(prob.DR_A) + (0 if prob.GA_rows is None
                                         else len(prob.GA_rows)))
        nHK = len(K.dual)
        for s in range(nS):
            for j in range(nHK):
                hp[s * nHK + j] = s * nHK + ph[j]
        head_perms.append(hp)
        # rays: index k = i * nS + e
        rp = np.arange(len(prob.DR_B))
        for i in range(m):
            for e in range(nS):
                rp[i * nS + e] = pg[i] * nS + e
        ray_perms.append(rp)
    return head_perms, ray_perms


def hierarchy_level_symfull(prob: GameProblem, n: int, box: float = 50.0,
                            verbose: bool = True,
                            dump_to: str | None = None,
                            load_from: str | None = None,
                            return_solution: bool = False):
    """Level-n LP with the complete facet list, orbit-averaged over the
    dihedral symmetry of K_B.  Exact:  the objective, the normalization
    and the feasible cone are invariant, so replacing each facet by its
    orbit average and afterwards projecting any optimizer with the group
    average preserves feasibility and the optimal value.  The averaged
    rows span the relevant invariant subspace, in which the LP is solved
    (a few hundred variables)."""
    t_start = time.time()
    if load_from is not None:                 # solve a pre-built model
        d = np.load(load_from)
        A_red, c_red, n_red = d["A"], d["c"], d["w"]
        d_inv, n_orb = A_red.shape[1], A_red.shape[0]
        orbit_reps = [None] * n_orb
        res, active, zsol = _solve_reduced(n, A_red, c_red, n_red,
                                           orbit_reps, t_start,
                                           float(d["prev"]), verbose)
        if return_solution:
            return res, active, (zsol @ d["B"]) / d["D"]
        return res, active
    dA, dBr = prob.dA, prob.dBr
    sym = SymSpace(dBr, n)
    Sn = sym.sizes[n]
    nvars = dA * Sn
    nPsi = len(prob.DR_B_red)

    heads = [prob.DR_A[k] for k in range(len(prob.DR_A))]
    if prob.GA_rows is not None:
        heads += [prob.GA_rows[k] for k in range(len(prob.GA_rows))]
    heads = np.array(heads)
    nH = len(heads)

    c_obj = build_covector(sym, prob.W_red, [prob.t_red] * (n - 1))
    w_norm = build_covector(sym, prob.uA, [prob.t_red] * n)

    # sqrt-multiplicity rescaling: in coordinates  Yhat = D Y  with
    # D = diag(sqrt(n!/prod m_i!)) the induced action of the dihedral
    # group is orthogonal, so the span of the (invariant) averaged rows
    # coincides with the fixed subspace of the action and the
    # reconstructed optimizer  Y = D^{-1} Bhat^T z  is exactly
    # group-invariant -- hence feasible for every *individual* facet,
    # which the rounding scheme requires.
    from math import factorial
    mults = np.empty(Sn)
    for idx, M in enumerate(sym.msets[n]):
        c: dict[int, int] = {}
        for k in M:
            c[k] = c.get(k, 0) + 1
        r = factorial(n)
        for v in c.values():
            r //= factorial(v)
        mults[idx] = r
    Dsqrt = np.tile(np.sqrt(mults), dA)
    c_obj_h = c_obj / Dsqrt
    w_norm_h = w_norm / Dsqrt

    head_perms, ray_perms = dihedral_actions(prob)
    G = len(head_perms)

    # ---- all symmetric ray covectors p_K by DFS with prefix sharing ------
    pvec: dict[tuple, np.ndarray] = {}

    def dfs(prefix: tuple, w: np.ndarray):
        t = len(prefix)
        if t == n:
            pvec[prefix] = w
            return
        last = prefix[-1] if prefix else 0
        for k in range(last, nPsi):
            dfs(prefix + (k,),
                sym.lift_covector(w, prob.DR_B_red[k], t + 1))

    dfs((), np.ones(1))

    # ---- orbit representatives -------------------------------------------
    msets = list(itertools.combinations_with_replacement(range(nPsi), n))
    seen: set[tuple] = set()
    reps: list[tuple] = []
    orbits: list[list[tuple]] = []
    for hk in range(nH):
        for K in msets:
            if (hk, K) in seen:
                continue
            orb = {(int(hp[hk]), tuple(sorted(rp[list(K)])))
                   for hp, rp in zip(head_perms, ray_perms)}
            seen |= orb
            reps.append((hk, K))
            orbits.append(sorted(orb))
    if verbose:
        print(f"    [n={n}] sym-full LP: {len(reps)} orbit rows "
              f"(of {nH*len(msets)} facets); streaming projection ...",
              flush=True)

    # ---- streaming block Gram-Schmidt ------------------------------------
    # Invariant-subspace basis grows as new directions appear; since each
    # new b is orthogonal to the earlier basis, rows projected earlier
    # have exactly zero coefficient on later b's, so one pass yields both
    # the basis and the projected constraint matrix A_red.
    cap = 4000
    Bmat = np.zeros((cap, nvars))
    d_inv = 0

    def add_dir(v: np.ndarray) -> None:
        nonlocal d_inv
        assert d_inv < cap
        Bmat[d_inv] = v
        d_inv += 1

    def project_block(V: np.ndarray) -> np.ndarray:
        """Project rows of V; append new directions; return coefficients
        (padded later)."""
        nonlocal d_inv
        C = V @ Bmat[:d_inv].T
        R = V - C @ Bmat[:d_inv]
        R -= (R @ Bmat[:d_inv].T) @ Bmat[:d_inv]        # reorthogonalize
        nrm = np.linalg.norm(R, axis=1)
        scale = np.maximum(1.0, np.linalg.norm(V, axis=1))
        extra = []
        for i in np.nonzero(nrm > 1e-9 * scale)[0]:
            r = R[i] - (Bmat[:d_inv] @ R[i]) @ Bmat[:d_inv]
            nr = np.linalg.norm(r)
            if nr > 1e-10 * scale[i]:
                add_dir(r / nr)
        if d_inv > C.shape[1]:
            C2 = V @ Bmat[C.shape[1]:d_inv].T
            C = np.hstack([C, C2])
        return C

    cvec = project_block(np.vstack([c_obj_h, w_norm_h]))
    c_red0, n_red0 = cvec[0], cvec[1]

    blocks = []
    BS = 512
    t_blk = time.time()
    for b0 in range(0, len(reps), BS):
        block_reps = orbits[b0:b0 + BS]
        V = np.zeros((len(block_reps), nvars))
        for i, orb in enumerate(block_reps):
            for hk2, K2 in orb:
                V[i] += np.kron(heads[hk2], pvec[K2])
            V[i] /= len(orb)
        blocks.append(project_block(V / Dsqrt))
    del pvec
    A_red = np.zeros((len(reps), d_inv))
    r0 = 0
    for Cb in blocks:
        A_red[r0:r0 + Cb.shape[0], :Cb.shape[1]] = Cb
        r0 += Cb.shape[0]
    del blocks
    c_red = np.zeros(d_inv); c_red[:len(c_red0)] = c_red0
    n_red = np.zeros(d_inv); n_red[:len(n_red0)] = n_red0
    orbit_reps = reps
    if verbose:
        print(f"    [n={n}] invariant dimension d_inv = {d_inv} "
              f"(projection {time.time()-t_blk:.0f}s)", flush=True)

    if dump_to is not None:
        np.savez(dump_to, A=A_red, c=c_red, w=n_red,
                 B=Bmat[:d_inv], D=Dsqrt, prev=time.time() - t_start)
        if verbose:
            print(f"    [n={n}] reduced model dumped to {dump_to}",
                  flush=True)
        return None, None
    res, active, zsol = _solve_reduced(n, A_red, c_red, n_red, orbit_reps,
                                       t_start, 0.0, verbose)
    if return_solution:
        return res, active, (zsol @ Bmat[:d_inv]) / Dsqrt
    return res, active


def _solve_reduced(n, A_red, c_red, n_red, orbit_reps, t_start,
                   prev_seconds, verbose):
    import highspy
    d_inv = A_red.shape[1]
    h = highspy.Highs()
    h.silent()
    INF = highspy.kHighsInf
    if len(orbit_reps) > 30000:
        h.setOptionValue("solver", "ipm")     # tall dense reduced LP
        h.setOptionValue("run_crossover", "on")
    h.addVars(d_inv, np.full(d_inv, -INF), np.full(d_inv, INF))
    h.changeColsCost(d_inv, np.arange(d_inv, dtype=np.int32), -c_red)
    An = sparse.csr_matrix(n_red[None, :])
    h.addRows(1, np.array([1.0]), np.array([1.0]), An.nnz,
              An.indptr.astype(np.int32), An.indices.astype(np.int32),
              An.data)
    Ar = sparse.csr_matrix(A_red)
    h.addRows(Ar.shape[0], np.zeros(Ar.shape[0]),
              np.full(Ar.shape[0], INF), Ar.nnz,
              Ar.indptr.astype(np.int32), Ar.indices.astype(np.int32),
              Ar.data)
    h.run()
    status = h.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"sym-full LP level {n}: status {status}")
    sol = h.getSolution()
    z = np.array(sol.col_value)
    value = float(c_red @ z)
    rowval = np.array(sol.row_value)[1:]
    active = [orbit_reps[i] for i in np.nonzero(rowval < 1e-7)[0]]
    res = LevelResult(n=n, value=value, n_vars=d_inv, n_eq=1,
                      n_cuts=Ar.shape[0], lp_solves=1, max_violation=0.0,
                      seconds=prev_seconds + time.time() - t_start,
                      done=True, certified_by="sym-full-LP")
    return res, active, z


# =========================================================================
# 5.  The rounding scheme (inner sequence)
# =========================================================================


def inner_rounding(prob: GameProblem, n: int, Y: np.ndarray,
                   tol: float = 1e-6):
    """Feasible product strategies from a level-n optimizer (the rounding
    scheme of the theorem "Convergent inner sequence", maximization
    version: the best conditional product is reported).

    Measure m of the n multimeter copies of the extension Y with the
    symmetric informationally complete measurement { e_k = psi_k / s },
    where psi_k are the extreme rays of C_mm^* and  sum_k psi_k = s t;
    condition on the outcome multiset z, marginalize the remaining
    n-1-m copies with the unit t, and productize the conditional pair
    (xi_z, N_z).  Every candidate is feasible for the *original*
    constrained problem: cone membership survives contraction with
    positive functionals, the multimeter normalization F_B holds because
    the slots are parametrized inside ker F_B, and the local inequality
    G_A survives conditioning by item (2c) of Sigma^n.  The reported
    L^(n) = max_{m, z} value(xi_z, N_z) is therefore a lower bound on
    gamma^*(G) attained by an explicit feasible strategy.

    NOTE: the level-n LP typically has many optimizers (degenerate optimal
    face) and the rounding map is nonlinear, so L^(n) depends on WHICH
    optimizer Y is rounded (observation (e) of the paper's numerics
    section).  The value is a property of the pair (level, Y).

    Returns a dict with per-m statistics (best value and outcome, the
    p(z)-weighted average value of the conditional products, i.e. the
    value of the de Finetti inner point), the overall best value, the
    best m and outcome, and the worst feasibility slack `min_slack` over
    all candidates (the candidates are verified feasible to `tol`).
    """
    from math import factorial

    sym = SymSpace(prob.dBr, n)
    Ymat = Y.reshape(prob.dA, sym.sizes[n])
    rays = prob.DR_B_red
    nPsi = len(rays)
    sums = rays.sum(axis=0)
    s = float(sums @ prob.t_red) / float(prob.t_red @ prob.t_red)
    assert np.allclose(sums, s * prob.t_red, atol=1e-9), \
        "extreme rays do not form a symmetric IC measurement"
    E = rays / s                                    # IC effects, sum = t

    def multinom(z):
        c = {}
        for k in z:
            c[k] = c.get(k, 0) + 1
        r = factorial(len(z))
        for v in c.values():
            r //= factorial(v)
        return r

    out = {"per_m": {}, "best": -np.inf, "best_m": None, "best_z": None,
           "min_slack": np.inf}
    for m in range(n):
        # marginalize the n-1-m unmeasured copies
        Ym = Ymat
        for t_lvl in range(n, m + 1, -1):
            Ym = sym.contract(Ym, prob.t_red, t_lvl)   # -> (dA, S_{m+1})
        best, best_z, avg, wsum = -np.inf, None, 0.0, 0.0

        def visit(z, V):
            nonlocal best, best_z, avg, wsum
            t_lvl = m + 1 - len(z)
            if len(z) == m:
                X = V                                   # (dA, dBr)
                pn = float(prob.uA @ X @ prob.t_red)
                w = multinom(z) * pn
                if pn <= 1e-12:
                    return
                xi = (X @ prob.t_red) / pn
                N = (prob.uA @ X) / pn
                slack = min(float((prob.DR_A @ xi).min()),
                            float((rays @ N).min()))
                if prob.GA_rows is not None:
                    slack = min(slack, float((prob.GA_rows @ xi).min()))
                out["min_slack"] = min(out["min_slack"], slack)
                assert slack > -tol, f"rounded point infeasible ({slack})"
                v = float(xi @ prob.W_red @ N)
                avg += w * v
                wsum += w
                if v > best:
                    best, best_z = v, z
                return
            last = z[-1] if z else 0
            for k in range(last, nPsi):
                visit(z + (k,), sym.contract(V, E[k], t_lvl))

        visit((), Ym)
        out["per_m"][m] = {"best": best, "best_z": list(best_z),
                           "avg": avg / wsum, "weight_sum": wsum}
        assert abs(wsum - 1.0) < 1e-7, wsum
        if best > out["best"]:
            out["best"], out["best_m"] = best, m
            out["best_z"] = list(best_z)
    return out


if __name__ == "__main__":
    # smoke test: classical bit, CHSH, levels 1-2 with rounding (~1 s)
    prob = GameProblem(classical_bit(), chsh_coefficients(),
                       name="classical bit / CHSH")
    for n in (1, 2):
        res, _, Y = hierarchy_level_full(prob, n, verbose=False,
                                         return_solution=True)
        rd = inner_rounding(prob, n, Y)
        print(f"n={n}: p^(n) = {res.value:.10f}   L^(n) = {rd['best']:.10f}"
              f"   (m* = {rd['best_m']}, min slack {rd['min_slack']:.1e})")
