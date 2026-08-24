"""
exact_game_value.py  --  the exact reference values gamma*(CHSH) of the
numerics section, computed directly (bypassing the hierarchy and the
rounding scheme) and certified in exact arithmetic.

    J. A. Zeiss, G. Kossmann, R. Schwonnek, M. Plavala,
    "Finite de Finetti for convex bodies and polynomial optimization",
    arXiv:2601.15184, Sec. "Numerical illustration", paragraph
    "Exact reference values".

Computed for the CHSH game (winning-probability convention, pi uniform) and
the three local state spaces of the paper (classical bit, square gbit,
strongly self-dual regular pentagon), plus the constrained pentagon
experiment q_{1|0} >= 0.7:

  gamma*(G)  = max over  xi in K_asm = CS^1_{2,2} (x)^ K_B  (assemblages)
               and N in K_mm = E(K_B) x E(K_B)  (complete multimeters)
               of the bilinear strategy value
             = the quantity bracketed by p^(n) and L^(n) in Table 1.

Why this is a finite computation.  Both bodies are polytopes and the
objective is bilinear, so the optimum is attained at a pair of vertices.
K_mm is affinely isomorphic to E(K_B) x E(K_B), so its vertices are the
pairs of extreme effects (4, 6, 12 extreme effects for bit, gbit,
pentagon), enumerated exactly; for every such pair the remaining
optimisation over the assemblage polytope K_asm is a linear program.  Each
LP is solved in floating point (HiGHS dual simplex) and its optimum is then
*certified exactly*: the optimal vertex is recomputed from its active
facets and an exact dual multiplier vector (Farkas certificate) is
recomputed from the LP basis, both in exact arithmetic over Q(sqrt5)
(Q for the bit and the square).  The maximum over the finitely many
certified LP values is the exact game value, reported as a closed form
a + b*sqrt5.  Multimeter pairs related by the affine symmetry group
Aut(K_B) give the same LP value, so by default only orbit representatives
are solved (--full solves every pair; the results agree).

Independent cross-checks performed by the script:
  * all vertices of K_asm are enumerated with the double description
    method (pycddlib; exact GMP arithmetic for the rational state spaces,
    floating point + exact re-derivation for the pentagon) and the maximum
    over vertex pairs is recomputed exactly;
  * the pentagon is realised over Q(sqrt5) by an affine image of the
    library's regular pentagon (affine equivalence verified numerically),
    and the float LP pipeline is re-run in the library's own coordinates
    (gpt_definetti_games.GameProblem: its W matrix and facet lists);
  * the results are compared against results.json (outer p^(n), inner
    L^(n)); the sandwich L^(n) <= gamma* <= p^(n) must hold.

Usage:   python3 exact_game_value.py [--full] [--no-vertices]
Runtime: a few seconds (--full: ~10 s).
Dependencies: numpy, scipy (HiGHS), pycddlib >= 3.0 (only for the vertex
enumeration; --no-vertices runs without it); gpt_definetti_games.py and
results.json next to this script for the cross-checks (optional).
Outputs: exact_game_values.json and exact_game_values.md (next to this
script unless --out is given).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:                 # find gpt_definetti_games.py
    sys.path.insert(0, HERE)

# =========================================================================
# 1.  Exact arithmetic in Q(sqrt5)
# =========================================================================


class Q5:
    """Numbers a + b*sqrt(5) with a, b in Q (exact)."""
    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a = a if isinstance(a, Fraction) else Fraction(a)
        self.b = b if isinstance(b, Fraction) else Fraction(b)

    @staticmethod
    def of(x):
        return x if isinstance(x, Q5) else Q5(x, 0)

    def __float__(self):
        return float(self.a) + float(self.b) * math.sqrt(5.0)

    def __repr__(self):
        if self.b == 0:
            return f"{self.a}"
        if self.a == 0:
            return f"{self.b}*sqrt5"
        sgn = "+" if self.b > 0 else "-"
        return f"{self.a} {sgn} {abs(self.b)}*sqrt5"

    def latex(self):
        """Closed form (a + b sqrt5)/den as a LaTeX string."""
        def fr(q):
            return f"{q.numerator}" if q.denominator == 1 \
                else f"\\tfrac{{{q.numerator}}}{{{q.denominator}}}"
        if self.b == 0:
            return fr(self.a)
        if self.a == 0:
            pre = "" if self.b == 1 else ("-" if self.b == -1 else fr(self.b))
            return pre + "\\sqrt5"
        den = math.lcm(self.a.denominator, self.b.denominator)
        A, B = int(self.a * den), int(self.b * den)
        sB = "+" if B > 0 else "-"
        Bs = "" if abs(B) == 1 else f"{abs(B)}"
        inner = f"{A} {sB} {Bs}\\sqrt5"
        return inner if den == 1 else f"\\tfrac{{{inner}}}{{{den}}}"

    def is_zero(self):
        return self.a == 0 and self.b == 0

    def __add__(self, o):
        o = Q5.of(o)
        return Q5(self.a + o.a, self.b + o.b)
    __radd__ = __add__

    def __neg__(self):
        return Q5(-self.a, -self.b)

    def __sub__(self, o):
        o = Q5.of(o)
        return Q5(self.a - o.a, self.b - o.b)

    def __rsub__(self, o):
        return Q5.of(o) - self

    def __mul__(self, o):
        o = Q5.of(o)
        return Q5(self.a * o.a + 5 * self.b * o.b, self.a * o.b + self.b * o.a)
    __rmul__ = __mul__

    def inv(self):
        d = self.a * self.a - 5 * self.b * self.b
        if d == 0:
            raise ZeroDivisionError("division by zero in Q(sqrt5)")
        return Q5(self.a / d, -self.b / d)

    def __truediv__(self, o):
        return self * Q5.of(o).inv()

    def __rtruediv__(self, o):
        return Q5.of(o) * self.inv()

    def sign(self):
        a, b = self.a, self.b
        if b == 0:
            return (a > 0) - (a < 0)
        if a == 0:
            return (b > 0) - (b < 0)
        if a > 0 and b > 0:
            return 1
        if a < 0 and b < 0:
            return -1
        if a > 0:                       # b < 0
            return 1 if a * a > 5 * b * b else -1
        return 1 if 5 * b * b > a * a else -1      # a < 0 < b

    def __eq__(self, o):
        o = Q5.of(o)
        return self.a == o.a and self.b == o.b

    def __hash__(self):
        return hash((self.a, self.b))

    def __lt__(self, o):
        return (self - Q5.of(o)).sign() < 0

    def __le__(self, o):
        return (self - Q5.of(o)).sign() <= 0

    def __gt__(self, o):
        return (self - Q5.of(o)).sign() > 0

    def __ge__(self, o):
        return (self - Q5.of(o)).sign() >= 0


SQRT5 = Q5(0, 1)


def qvec(xs):
    return [Q5.of(x) for x in xs]


def qdot(u, v):
    s = Q5(0)
    for x, y in zip(u, v):
        s = s + x * y
    return s


def qfloat(v):
    return np.array([float(x) for x in v])


def vkey(v):
    return tuple((x.a, x.b) for x in v)


def _rref(rows, n, rhs=None):
    """Row reduction over Q(sqrt5).  Returns (M, pivot columns, rank);
    if rhs is given it is appended as the last column of M."""
    m = len(rows)
    M = [list(r) + ([b] if rhs is not None else []) for r, b in
         zip(rows, rhs if rhs is not None else [None] * m)]
    piv, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, m) if not M[i][c].is_zero()), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = M[r][c].inv()
        M[r] = [x * inv for x in M[r]]
        for i in range(m):
            if i != r and not M[i][c].is_zero():
                f = M[i][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == m:
            break
    return M, piv, r


def exact_solve(rows, rhs):
    """Solve rows.z = rhs exactly (free variables := 0).  Returns (z, rank);
    raises ValueError if inconsistent."""
    n = len(rows[0])
    M, piv, r = _rref(rows, n, rhs)
    for i in range(r, len(rows)):
        if not M[i][n].is_zero():
            raise ValueError("inconsistent exact system")
    z = [Q5(0)] * n
    for i, c in enumerate(piv):
        z[c] = M[i][n]
    return z, r


def exact_rank(rows):
    return _rref(rows, len(rows[0]))[2]


def nullspace(rows, n):
    """Basis of {h : rows.h = 0} over Q(sqrt5)."""
    M, piv, r = _rref(rows, n)
    basis = []
    for f in [c for c in range(n) if c not in piv]:
        h = [Q5(0)] * n
        h[f] = Q5(1)
        for i, c in enumerate(piv):
            h[c] = -M[i][f]
        basis.append(h)
    return basis


# =========================================================================
# 2.  Exact state spaces: vertices, facets, extreme effects, symmetries
# =========================================================================


class ExactStateSpace:
    """Polytopic state space K in R^d (u = first coordinate).

    gens   : vertices g_j (Q5 vectors, first coordinate 1)
    facets : extreme rays h of the dual cone, effect-normalised
             (max_j h(g_j) = 1), h(K) >= 0
    ext_E  : vertices of the effect body E(K) = {e : 0 <= e(g_j) <= 1}
    auts   : linear automorphisms S of the cone permuting the vertices
    """

    def __init__(self, name, gens):
        self.name = name
        self.gens = [qvec(g) for g in gens]
        self.d = len(self.gens[0])
        self.u = [Q5(1)] + [Q5(0)] * (self.d - 1)
        self.facets = self._facets()
        self.ext_E = self._extreme_effects()
        self.auts = self._automorphisms()

    def _facets(self):
        d, gens = self.d, self.gens
        out, seen = [], set()
        for sub in itertools.combinations(range(len(gens)), d - 1):
            null = nullspace([gens[j] for j in sub], d)
            if len(null) != 1:
                continue
            h = null[0]
            vals = [qdot(h, g) for g in gens]
            if all(v <= 0 for v in vals):
                h, vals = [-x for x in h], [-v for v in vals]
            if not all(v >= 0 for v in vals):
                continue
            mx = max(vals)
            h = [x / mx for x in h]
            if vkey(h) not in seen:
                seen.add(vkey(h))
                out.append(h)
        return out

    def _extreme_effects(self):
        d, gens = self.d, self.gens
        planes = [(g, Q5(0)) for g in gens] + [(g, Q5(1)) for g in gens]
        out, seen = [], set()
        for sub in itertools.combinations(range(len(planes)), d):
            try:
                e, rk = exact_solve([planes[j][0] for j in sub],
                                    [planes[j][1] for j in sub])
            except ValueError:
                continue
            if rk < d:
                continue
            vals = [qdot(e, g) for g in gens]
            if all(v >= 0 and v <= 1 for v in vals) and vkey(e) not in seen:
                seen.add(vkey(e))
                out.append(e)
        return out

    def _automorphisms(self):
        """All linear maps S with S g_j = g_{pi(j)} for a permutation pi
        (matrices as lists of rows)."""
        d, gens = self.d, self.gens
        # d vertices spanning R^d
        base = None
        for sub in itertools.combinations(range(len(gens)), d):
            if exact_rank([gens[j] for j in sub]) == d:
                base = sub
                break
        G = [gens[j] for j in base]                     # rows g_j
        auts = []
        for pi in itertools.permutations(range(len(gens))):
            # S^T solves  G S^T = G_pi   (row-wise:  g_j S^T = g_pi(j))
            Gpi = [gens[pi[j]] for j in base]
            ST = []
            for col in range(d):
                z, _ = exact_solve(G, [row[col] for row in Gpi])
                ST.append(z)
            # column k of S^T is row k of S
            S = ST
            ok = all(vkey([qdot(S[i], gens[j]) for i in range(d)]) == vkey(gens[pi[j]])
                     for j in range(len(gens)))
            if ok:
                auts.append(S)
        return auts

    def act_effect(self, e, S):
        """e o S  (row vector times matrix)."""
        return [sum((e[i] * S[i][k] for i in range(self.d)), Q5(0)) for k in range(self.d)]

    def effect_index(self):
        return {vkey(e): i for i, e in enumerate(self.ext_E)}

    def pair_orbits(self):
        """Orbit representatives of pairs (e0, e1) in ext_E^2 under the
        simultaneous action e -> e o S; returns list of (i0, i1, orbit size)."""
        idx = self.effect_index()
        perms = []
        for S in self.auts:
            perms.append([idx[vkey(self.act_effect(e, S))] for e in self.ext_E])
        # the maps must form a group (closure) for the orbits to be classes
        pset = {tuple(p) for p in perms}
        for p in perms:
            for q in perms:
                assert tuple(p[i] for i in q) in pset, "automorphism list not closed"
        n = len(self.ext_E)
        seen, reps = set(), []
        for i0 in range(n):
            for i1 in range(n):
                if (i0, i1) in seen:
                    continue
                orbit = {(p[i0], p[i1]) for p in perms}
                seen |= orbit
                reps.append((i0, i1, len(orbit)))
        return reps

    def check(self):
        for h in self.facets:
            assert all(qdot(h, g) >= 0 for g in self.gens)
            assert max(qdot(h, g) for g in self.gens) == 1
        for e in self.ext_E:
            assert all(qdot(e, g) >= 0 and qdot(e, g) <= 1 for g in self.gens)
        for g in self.gens:           # facet list complete: each vertex is a vertex
            tight = [h for h in self.facets if qdot(h, g).is_zero()]
            assert exact_rank(tight) == self.d - 1, "facet list incomplete"
        # automorphisms permute facets and extreme effects
        fk = {vkey(h) for h in self.facets}
        ek = {vkey(e) for e in self.ext_E}
        for S in self.auts:
            assert {vkey(self.act_effect(h, S)) for h in self.facets} == fk
            assert {vkey(self.act_effect(e, S)) for e in self.ext_E} == ek


def exact_classical_bit():
    return ExactStateSpace("classical bit", [[1, 0], [1, 1]])


def exact_square_gbit():
    return ExactStateSpace("square gbit", [[1, 1, 1], [1, -1, 1], [1, -1, -1], [1, 1, -1]])


def exact_pentagon():
    """Affine image of the regular pentagon (cos 72k, sin 72k) under
    diag(1, 1/sin 72): all coordinates lie in Q(sqrt5).  Game values are
    affine invariants of K_B, so this realisation gives the same values as
    the library's self-dual regular pentagon (pentagon_selfdual)."""
    c1 = (SQRT5 - 1) / 4            # cos 72
    c2 = -(SQRT5 + 1) / 4           # cos 144
    s2 = (SQRT5 - 1) / 2            # sin 144 / sin 72 = 2 cos 72
    return ExactStateSpace("pentagon", [[1, 1, 0], [1, c1, 1], [1, c2, s2],
                                        [1, c2, -s2], [1, c1, -1]])


def chsh_g():
    g = {}
    for a, b, x, y in itertools.product(range(2), repeat=4):
        g[(a, b, x, y)] = Fraction(1, 4) if ((a ^ b) == (x & y)) else Fraction(0)
    return g


# =========================================================================
# 3.  Certified LPs:  max c.xi  s.t.  A xi >= 0,  u.xi = 1
# =========================================================================


class CertifiedLP:
    """Polytope {xi : A xi >= 0, u.xi = 1} with exact data; float LP
    (HiGHS dual simplex, no presolve -> basic optimal solution) followed by
    an exact primal vertex and an exact Farkas/dual certificate."""

    def __init__(self, A_rows, u_row, tag=""):
        self.A, self.u, self.n = A_rows, u_row, len(u_row)
        self.Af = np.array([qfloat(r) for r in A_rows])
        self.uf = qfloat(u_row)
        self.tag = tag
        self.stats = {"lps": 0, "fallbacks": 0}

    def solve(self, c):
        cf = qfloat(c)
        res = linprog(-cf, A_ub=-self.Af, b_ub=np.zeros(len(self.Af)),
                      A_eq=self.uf[None, :], b_eq=np.array([1.0]),
                      bounds=(None, None), method="highs-ds",
                      options={"presolve": False})
        self.stats["lps"] += 1
        if res.status != 0:
            raise RuntimeError(f"LP failed ({self.tag}): {res.message}")
        xf = res.x
        slack = self.Af @ xf
        tol = 1e-8 * max(1.0, np.abs(xf).max())
        active = [j for j in range(len(self.A)) if abs(slack[j]) <= tol]
        # exact primal vertex from the active facets
        xi, rk = exact_solve([self.A[j] for j in active] + [self.u],
                             [Q5(0)] * len(active) + [Q5(1)])
        if rk < self.n:
            raise RuntimeError(f"float optimum not a vertex ({self.tag}): rank {rk}<{self.n}")
        if any(qdot(a, xi) < 0 for a in self.A):
            raise RuntimeError(f"exact vertex infeasible ({self.tag})")
        assert qdot(self.u, xi) == 1
        val = qdot(c, xi)
        # exact dual certificate  mu*u - sum_j y_j a_j = c,  y >= 0
        marg = res.ineqlin.marginals
        support = [j for j in active if abs(marg[j]) > 1e-9]
        y, mu = self._dual(c, support)
        if y is None:
            self.stats["fallbacks"] += 1
            for k in range(min(len(active), self.n - 1), 0, -1):
                for sub in itertools.combinations(active, k):
                    y, mu = self._dual(c, list(sub))
                    if y is not None:
                        break
                if y is not None:
                    break
        if y is None:
            raise RuntimeError(f"no exact dual certificate ({self.tag})")
        assert mu == val, "exact primal and dual values differ"
        return {"value": val, "xi": xi, "dual": y, "mu": mu,
                "float_value": float(-res.fun), "active": active}

    def _dual(self, c, support):
        cols = [self.u] + [[-x for x in self.A[j]] for j in support]
        rows = [[cols[k][i] for k in range(len(cols))] for i in range(self.n)]
        try:
            z, _ = exact_solve(rows, c)
        except ValueError:
            return None, None
        mu, y = z[0], z[1:]
        if any(v < 0 for v in y):
            return None, None
        for i in range(self.n):            # verify the identity exactly
            lhs = mu * self.u[i] - sum((y[k] * self.A[j][i] for k, j in enumerate(support)), Q5(0))
            if not (lhs - c[i]).is_zero():
                return None, None
        return {j: v for j, v in zip(support, y)}, mu


# =========================================================================
# 4.  The game-value LP over the assemblage polytope
# =========================================================================


def place(vec, block, nblocks, d):
    out = [Q5(0)] * (nblocks * d)
    for i, x in enumerate(vec):
        out[block * d + i] = x
    return out


def vadd(*vs):
    out = list(vs[0])
    for v in vs[1:]:
        out = [a + b for a, b in zip(out, v)]
    return out


def vscale(s, v):
    return [Q5.of(s) * x for x in v]


def assemblage_polytope(W, ineq_q10=None):
    """K_asm, coordinates xi = (rho_bar, rho~_{1|0}, rho~_{1|1}) in R^{3d}:
    h(rho~_{1|x}) >= 0, h(rho_bar - rho~_{1|x}) >= 0 for all facets h;
    optionally q_{1|0} - ineq_q10 >= 0;  normalisation u(rho_bar) = 1."""
    d = W.d
    rows, labels = [], []
    for x in (0, 1):
        for k, h in enumerate(W.facets):
            rows.append(place(h, 1 + x, 3, d))
            labels.append(f"h{k}(rho~_{{1|{x}}})>=0")
            rows.append(vadd(place(h, 0, 3, d), vscale(-1, place(h, 1 + x, 3, d))))
            labels.append(f"h{k}(rho~_{{0|{x}}})>=0")
    if ineq_q10 is not None:
        rows.append(vadd(place(W.u, 1, 3, d), vscale(-Q5.of(ineq_q10), place(W.u, 0, 3, d))))
        labels.append(f"q_{{1|0}}>={ineq_q10}")
    return rows, labels, place(W.u, 0, 3, d)


def assemblage_objective(W, g, e0, e1):
    """c.xi = sum_abxy g_abxy n_{b|y}(rho~_{a|x}) with n_{1|y} = e_y."""
    d = W.d
    n = {(1, 0): e0, (1, 1): e1,
         (0, 0): [a - b for a, b in zip(W.u, e0)],
         (0, 1): [a - b for a, b in zip(W.u, e1)]}
    c = [Q5(0)] * (3 * d)
    for (a, b, x, y), coef in g.items():
        if coef == 0:
            continue
        v = place(n[(b, y)], 1 + x, 3, d)
        if a == 0:
            v = vadd(place(n[(b, y)], 0, 3, d), vscale(-1, v))
        c = vadd(c, vscale(coef, v))
    return c


# =========================================================================
# 5.  Vertex enumeration (double description) + exact re-derivation
# =========================================================================


def enumerate_vertices(rows, u_row, exact_rational=False):
    """Vertices of {x : rows.x >= 0, u.x = 1} via pycddlib.  GMP exact
    arithmetic if all data are rational; otherwise floating point followed
    by exact re-derivation of every vertex from its active facets."""
    import cdd
    n = len(u_row)
    if exact_rational:
        import cdd.gmp as cddx
        arr = [[Fraction(0)] + [r[i].a for i in range(n)] for r in rows]
        arr.append([Fraction(-1)] + [u_row[i].a for i in range(n)])
        mat = cddx.matrix_from_array(arr, lin_set={len(arr) - 1},
                                     rep_type=cddx.RepType.INEQUALITY)
        gen = cddx.copy_generators(cddx.polyhedron_from_matrix(mat))
        verts = []
        for row in gen.array:
            assert row[0] == 1, "unbounded?"
            verts.append([Q5(Fraction(v)) for v in row[1:]])
        return verts, "cdd-gmp"
    arr = [[0.0] + list(qfloat(r)) for r in rows]
    arr.append([-1.0] + list(qfloat(u_row)))
    mat = cdd.matrix_from_array(arr, lin_set={len(arr) - 1},
                                rep_type=cdd.RepType.INEQUALITY)
    gen = cdd.copy_generators(cdd.polyhedron_from_matrix(mat))
    Af = np.array([qfloat(r) for r in rows])
    verts, seen = [], set()
    for row in gen.array:
        assert abs(row[0] - 1.0) < 1e-9, "unbounded?"
        xf = np.array(row[1:])
        sl = Af @ xf
        active = [j for j in range(len(rows)) if abs(sl[j]) <= 1e-7]
        xi, rk = exact_solve([rows[j] for j in active] + [u_row],
                             [Q5(0)] * len(active) + [Q5(1)])
        assert rk == n, "float vertex does not determine an exact vertex"
        assert all(qdot(r, xi) >= 0 for r in rows), "exact vertex infeasible"
        if vkey(xi) not in seen:
            seen.add(vkey(xi))
            verts.append(xi)
    return verts, "cdd-float+exact"


# =========================================================================
# 6.  Drivers
# =========================================================================


def run_gamma(W, g, ineq=None, full=False):
    rows, labels, u_row = assemblage_polytope(W, ineq)
    lp = CertifiedLP(rows, u_row, tag=f"gamma*/{W.name}")
    pairs = [(i0, i1, 1) for i0 in range(len(W.ext_E)) for i1 in range(len(W.ext_E))] \
        if full else W.pair_orbits()
    t0 = time.time()
    best, best_list, values = None, [], []
    for i0, i1, osz in pairs:
        r = lp.solve(assemblage_objective(W, g, W.ext_E[i0], W.ext_E[i1]))
        values.append((r["value"], i0, i1, osz))
        if best is None or r["value"] > best:
            best, best_list = r["value"], [(i0, i1, osz, r)]
        elif r["value"] == best:
            best_list.append((i0, i1, osz, r))
    return {"value": best, "maximisers": best_list, "all": values, "rows": rows,
            "labels": labels, "u_row": u_row, "seconds": time.time() - t0,
            "n_lps": len(pairs), "n_tuples_total": len(W.ext_E) ** 2,
            "n_maximising_tuples": sum(m[2] for m in best_list),
            "lp_stats": lp.stats}


def effect_name(W, e):
    """Readable tag for an extreme effect: 0, 1, or its values on vertices."""
    vals = [qdot(e, gg) for gg in W.gens]
    if all(v.is_zero() for v in vals):
        return "0"
    if all(v == 1 for v in vals):
        return "1"
    return "(" + ", ".join(repr(v) for v in vals) + ")"


def describe_assemblage(W, xi):
    d = W.d
    rb, r10, r11 = xi[0:d], xi[d:2 * d], xi[2 * d:3 * d]
    out = {"rho_bar": [repr(x) for x in rb],
           "rho~_{1|0}": [repr(x) for x in r10], "rho~_{1|1}": [repr(x) for x in r11],
           "q_{1|0}": repr(qdot(W.u, r10)), "q_{1|1}": repr(qdot(W.u, r11))}
    # barycentric description of the conditional states where possible
    def bary(v):
        q = qdot(W.u, v)
        if q.is_zero():
            return "0"
        s = [x / q for x in v]
        for j, gg in enumerate(W.gens):
            if vkey(s) == vkey(gg):
                return f"{q!r} * g{j}"
        return f"{q!r} * {[repr(x) for x in s]}"
    out["decomposition"] = {
        "rho~_{0|0}": bary([a - b for a, b in zip(rb, r10)]), "rho~_{1|0}": bary(r10),
        "rho~_{0|1}": bary([a - b for a, b in zip(rb, r11)]), "rho~_{1|1}": bary(r11)}
    return out


def library_crosscheck(name, ineq):
    """Float pipeline in the library's own coordinates (GameProblem.W,
    DR_A, DR_B): enumerate ext E(K_B) numerically, LP per pair, max."""
    try:
        import gpt_definetti_games as lib
    except ImportError:
        return None
    K = {"classical bit": lib.classical_bit, "square gbit": lib.square_gbit,
         "pentagon": lib.pentagon_selfdual, "pentagon+ineq": lib.pentagon_selfdual}[name]()
    prob = lib.GameProblem(K, lib.chsh_coefficients(), name=name)
    if ineq is not None:
        GA = np.kron(lib.eps_functional(0, 1), K.u) - float(ineq) * prob.uA
        prob = lib.GameProblem(K, lib.chsh_coefficients(), name=name, GA_rows=GA[None, :])
    # extreme effects numerically: intersections of d planes e.g_j in {0,1}
    d = K.dim
    planes = [(gg, 0.0) for gg in K.gens] + [(gg, 1.0) for gg in K.gens]
    effs = []
    for sub in itertools.combinations(range(len(planes)), d):
        M = np.array([planes[j][0] for j in sub])
        b = np.array([planes[j][1] for j in sub])
        if abs(np.linalg.det(M)) < 1e-10:
            continue
        e = np.linalg.solve(M, b)
        vals = K.gens @ e
        if vals.min() > -1e-9 and vals.max() < 1 + 1e-9:
            if not any(np.allclose(e, f, atol=1e-9) for f in effs):
                effs.append(e)
    A_ub = [-prob.DR_A]
    if prob.GA_rows is not None:
        A_ub.append(-prob.GA_rows)
    A_ub = np.vstack(A_ub)
    best = -np.inf
    for e0 in effs:
        for e1 in effs:
            N = prob.N_from_effects(e0, e1)
            c = prob.W @ N
            res = linprog(-c, A_ub=A_ub, b_ub=np.zeros(len(A_ub)),
                          A_eq=prob.uA[None, :], b_eq=np.array([1.0]),
                          bounds=(None, None), method="highs")
            assert res.status == 0
            best = max(best, -res.fun)
    return {"n_extreme_effects": len(effs), "gamma_star_float": best}


def affine_equivalence_check(W):
    """Affine map from the library's regular pentagon onto W.gens (residual)."""
    try:
        import gpt_definetti_games as lib
    except ImportError:
        return None
    K = lib.pentagon_selfdual()
    X = K.gens                               # (5, 3), first coord 1
    Y = np.array([qfloat(gv) for gv in W.gens])
    T, *_ = np.linalg.lstsq(X, Y, rcond=None)  # X T = Y
    return float(np.abs(X @ T - Y).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="solve every multimeter pair (no symmetry reduction)")
    ap.add_argument("--no-vertices", action="store_true",
                    help="skip the vertex-enumeration cross-check (no pycddlib needed)")
    ap.add_argument("--out", default=os.path.join(HERE, "exact_game_values.json"))
    ap.add_argument("--results", default=os.path.join(HERE, "results.json"),
                    help="results.json of reproduce.py for the sandwich check")
    args = ap.parse_args()

    g = chsh_g()
    spaces = {w.name: w for w in [exact_classical_bit(), exact_square_gbit(), exact_pentagon()]}
    for W in spaces.values():
        W.check()
        print(f"[{W.name}] d={W.d}: {len(W.gens)} vertices, {len(W.facets)} facets, "
              f"{len(W.ext_E)} extreme effects, |Aut|={len(W.auts)}, "
              f"{len(W.pair_orbits())} orbits of multimeter pairs (of {len(W.ext_E)**2})")
    res_aff = affine_equivalence_check(spaces["pentagon"])
    if res_aff is not None:
        print(f"[pentagon] affine image of the library's regular pentagon: residual {res_aff:.1e}")

    experiments = [("classical bit", "classical bit", None), ("square gbit", "square gbit", None),
                   ("pentagon", "pentagon", None), ("pentagon+ineq", "pentagon", Fraction(7, 10))]
    try:
        ref = json.load(open(args.results))
    except Exception:
        ref = None

    out = {"meta": {"game": "CHSH, pi uniform, winning-probability convention",
                    "method": "extreme multimeters x certified LPs over the state polytope; "
                              "exact arithmetic in Q(sqrt5); symmetry reduction "
                              + ("off" if args.full else "on"),
                    "pentagon_affine_residual": res_aff},
           "experiments": {}}
    md = ["| experiment | gamma* (exact) | closed form | p^(5) | L^(5) | #certified LPs | #maximising pairs | K_asm vertices |",
          "|---|---|---|---|---|---|---|---|"]

    for name, wname, ineq in experiments:
        W = spaces[wname]
        print(f"\n=== {name} ===")
        G = run_gamma(W, g, ineq, full=args.full)
        gam = G["value"]
        print(f"  gamma*  = {float(gam):.12f} = {gam!r}   [{G['n_lps']} certified LPs "
              f"({G['n_tuples_total']} tuples), {G['seconds']:.1f}s, "
              f"{G['lp_stats']['fallbacks']} dual fallbacks]; attained by "
              f"{G['n_maximising_tuples']} multimeter tuples")
        ent = {"gamma_star": {"value_float": float(gam), "closed_form": repr(gam),
                              "latex": gam.latex(), "n_certified_lps": G["n_lps"],
                              "n_multimeter_tuples": G["n_tuples_total"],
                              "n_maximising_tuples": G["n_maximising_tuples"],
                              "seconds": G["seconds"],
                              "distinct_lp_values_top": sorted({float(v[0]) for v in G["all"]}, reverse=True)[:6]}}
        mx = []
        for i0, i1, osz, r in G["maximisers"]:
            mx.append({"n_{1|0}": [repr(x) for x in W.ext_E[i0]], "n_{1|0}_on_vertices": effect_name(W, W.ext_E[i0]),
                       "n_{1|1}": [repr(x) for x in W.ext_E[i1]], "n_{1|1}_on_vertices": effect_name(W, W.ext_E[i1]),
                       "orbit_size": osz, "assemblage": describe_assemblage(W, r["xi"]),
                       "dual_certificate": {G["labels"][j]: repr(v) for j, v in r["dual"].items()}})
        ent["gamma_star"]["maximisers"] = mx
        ex = mx[0]
        print(f"  example optimal strategy: n_{{1|0}} = {ex['n_{1|0}_on_vertices']} on vertices, "
              f"n_{{1|1}} = {ex['n_{1|1}_on_vertices']};  assemblage: {ex['assemblage']['decomposition']}")

        nv_asm = None
        if not args.no_vertices:
            t0 = time.time()
            verts, how = enumerate_vertices(G["rows"], G["u_row"], exact_rational=(wname != "pentagon"))
            vbest = None
            for e0 in W.ext_E:
                for e1 in W.ext_E:
                    c = assemblage_objective(W, g, e0, e1)
                    for v in verts:
                        val = qdot(c, v)
                        if vbest is None or val > vbest:
                            vbest = val
            ok = (vbest == gam)
            nv_asm = len(verts)
            print(f"  K_asm: {len(verts)} vertices ({how}); exact max over vertex pairs "
                  f"{'agrees' if ok else 'DISAGREES'} ({time.time()-t0:.1f}s)")
            ent["K_asm_vertices"] = {"count": len(verts), "method": how,
                                     "max_over_vertex_pairs": float(vbest), "agrees": bool(ok)}
            assert ok

        lc = library_crosscheck(name, ineq)
        if lc is not None:
            lc["difference_to_exact"] = lc["gamma_star_float"] - float(gam)
            ent["library_coordinates_crosscheck"] = lc
            print(f"  library coordinates (float, {lc['n_extreme_effects']} effects): gamma* = "
                  f"{lc['gamma_star_float']:.12f}  (diff {lc['difference_to_exact']:+.1e})")
            assert abs(lc["difference_to_exact"]) < 1e-9

        p5 = L5 = None
        if ref is not None and name in ref:
            R = ref[name]
            p = {int(k): v["value"] for k, v in R["levels"].items()}
            Ln = {int(k): v["best"] for k, v in R["inner"].items()}
            gf = float(gam)
            ent["comparison_results_json"] = {
                "outer_gaps_p_minus_gamma": {n: p[n] - gf for n in sorted(p)},
                "inner_gaps_gamma_minus_L": {n: gf - Ln[n] for n in sorted(Ln)}}
            p5, L5 = p[max(p)], Ln[max(Ln)]
            print("  results.json:  p^(n)-gamma*: "
                  + ", ".join(f"{p[n]-gf:+.2e}" for n in sorted(p))
                  + ";  gamma*-L^(n): " + ", ".join(f"{gf-Ln[n]:+.2e}" for n in sorted(Ln)))
            if not (all(p[n] >= gf - 1e-9 for n in p) and all(Ln[n] <= gf + 1e-9 for n in Ln)):
                raise RuntimeError(f"{name}: sandwich L^(n) <= gamma* <= p^(n) violated")
        out["experiments"][name] = ent
        md.append(f"| {name} | {float(gam):.10f} | {gam.latex()} | "
                  f"{'' if p5 is None else f'{p5:.10f}'} | "
                  f"{'' if L5 is None else f'{L5:.10f}'} | {G['n_lps']} | "
                  f"{G['n_maximising_tuples']} | {nv_asm if nv_asm is not None else ''} |")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    with open(args.out.replace(".json", ".md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nwritten {args.out} and {args.out.replace('.json', '.md')}")


if __name__ == "__main__":
    main()
