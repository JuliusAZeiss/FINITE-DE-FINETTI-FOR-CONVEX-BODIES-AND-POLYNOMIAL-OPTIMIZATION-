"""
direct_inner_certify.py -- rigorous validation of a rounded product strategy
(xi, N) as a lower bound on gamma^*  (added 2026-08-19 after the code audit)
===========================================================================

The rounded candidates of direct_inner_rounding.py are floating-point
vectors that satisfy the constraints of the ORIGINAL problem to ~1e-13.
That is a numerical feasibility check, not a proof.  This module turns the
best candidate into a rigorous lower bound:

  (1) the instance is rebuilt in interval arithmetic (mpmath.iv) from its
      definition -- the CS square, the regular m-gon with radius r and phase,
      the CHSH coefficients, the G_A row -- not from the float data (the
      float data are checked to lie inside the intervals, +-1e-13);
  (2) the equality constraints are imposed exactly: they pin coordinates,
      u_A xi = xi[0] = 1  and  N[:,0] = u_K  (<=> F_B N = 0, t N = 1); the
      float candidate satisfies them to ~1e-16, we set them exactly;
  (3) the candidate is mixed with a strictly interior product strategy
      (xi_0, N_0):  xi_eps = (1-eps) xi + eps xi_0,  N_eps likewise;
  (4) EVERY inequality constraint of the original problem (all extreme rays
      of C_asm^* and C_mm^*, G_A) is evaluated on (xi_eps, N_eps) in interval
      arithmetic and required to be strictly positive, and the game value
      <W, xi_eps (x) N_eps> is evaluated as an interval.

The lower endpoint of the value interval is then a rigorous lower bound on
gamma^*: (xi_eps, N_eps) is an exactly feasible product strategy of the
exact instance.  The price is O(eps) (default eps = 1e-9).

For instances with rational data (classical bit, square gbit) the
candidate is additionally rationalised (limit_denominator) and checked
EXACTLY with fractions (slacks >= 0, no eps); if that succeeds the exact
rational value is reported (e.g. 3/4 and 1 at n = 2).

The normalisation of the dual rays is irrelevant here (the inequalities are
homogeneous), so only their direction is reconstructed.
"""

from __future__ import annotations

from fractions import Fraction
import itertools

import numpy as np
import mpmath
from mpmath import iv, mp

EPS_FUNCTIONALS = {(0, 0): (1, -1, 0), (0, 1): (0, 1, 0),
                   (1, 0): (1, 0, -1), (1, 1): (0, 0, 1)}   # eps^{(x)}_a
U_S = (1, 0, 0)


def _q(x):
    """exact rational / float / interval -> interval (floats are exact)."""
    if isinstance(x, Fraction):
        return iv.mpf(x.numerator) / iv.mpf(x.denominator)
    return iv.mpf(x)


def _lo(x):
    return mp.mpf(x.a)


def _hi(x):
    return mp.mpf(x.b)


class ExactInstance:
    """The game instance in interval arithmetic (and, if rational, exactly).

    kind: "classical bit" | ("polygon", m, radius_expr, phase_expr)
    radius_expr / phase_expr: callables taking the iv context, e.g.
        lambda iv: 1/iv.sqrt(iv.cos(iv.pi/5)),  lambda iv: iv.pi/4
    rational_gens / rational_dual: exact Fraction data if the instance is
        rational (bit, square), else None.
    ga_threshold: Fraction or None (constraint  q_{1|x=0} >= threshold).
    """

    def __init__(self, name, kind, m=None, radius_expr=None, phase_expr=None,
                 rational_gens=None, rational_dual=None, ga_threshold=None,
                 dps=60):
        self.name = name
        iv.dps = dps
        mp.dps = dps
        self.rational = rational_gens is not None
        if kind == "classical bit":
            gens = [[1, 0], [1, 1]]
            dual = [[1, -1], [0, 1]]
            self.u = [1, 0]
            self.tau = [1, Fraction(1, 2)]
        else:                                   # regular polygon in R^3
            r = radius_expr(iv)
            ph = phase_expr(iv)
            gens = []
            for j in range(m):
                ang = 2 * iv.pi * j / m + ph
                gens.append([iv.mpf(1), r * iv.cos(ang), r * iv.sin(ang)])
            dual = []
            for j in range(m):
                g1, g2 = gens[j], gens[(j + 1) % m]
                h = [g1[1] * g2[2] - g1[2] * g2[1],
                     g1[2] * g2[0] - g1[0] * g2[2],
                     g1[0] * g2[1] - g1[1] * g2[0]]
                # direction such that h(tau) = h[0] > 0
                if _hi(h[0]) < 0:
                    h = [-c for c in h]
                elif not _lo(h[0]) > 0:
                    raise RuntimeError("sign of a dual ray undetermined")
                dual.append(h)
            self.u = [1, 0, 0]
            self.tau = [1, 0, 0]
        self.gens_iv = [[_q(c) for c in g] for g in gens]
        self.dual_iv = [[_q(c) for c in h] for h in dual]
        self.gens_q = rational_gens
        self.dual_q = rational_dual
        self.dP = len(self.u)
        self.dA = 3 * self.dP
        self.dB = self.dP * 3
        self.ga_threshold = ga_threshold
        # CHSH bilinear form C[al,be] = sum g_abxy eps^x_a[al] eps^y_b[be]
        C = [[Fraction(0)] * 3 for _ in range(3)]
        for a, b, x, y in itertools.product(range(2), repeat=4):
            if (a ^ b) == (x & y):
                ea, eb = EPS_FUNCTIONALS[(x, a)], EPS_FUNCTIONALS[(y, b)]
                for al in range(3):
                    for be in range(3):
                        C[al][be] += Fraction(1, 4) * ea[al] * eb[be]
        self.C = C                                   # exact

    # ---- rows of the original problem (as lists of intervals) -----------
    def phi_rows(self):
        """ext C_asm^*  =  e (x) h   (index alpha*dP + i)."""
        for e in EPS_FUNCTIONALS.values():
            for h in self.dual_iv:
                yield [_q(e[al]) * h[i] for al in range(3)
                       for i in range(self.dP)]

    def psi_rows(self):
        """ext C_mm^*  =  g (x) e   (index i*3 + beta)."""
        for g in self.gens_iv:
            for e in EPS_FUNCTIONALS.values():
                yield [g[i] * _q(e[be]) for i in range(self.dP)
                       for be in range(3)]

    def ga_row(self):
        if self.ga_threshold is None:
            return None
        row = [iv.mpf(0)] * self.dA
        row[1 * self.dP + 0] += iv.mpf(1)              # q_{1|x=0}: eps(0,1) (x) u
        for i in range(self.dP):
            row[0 * self.dP + i] -= _q(self.ga_threshold) * _q(self.u[i])
        return row

    # ---- exact-equality embedding and interior strategy -----------------
    def embed(self, xi, N):
        """Float candidate -> interval vectors with the equalities exact."""
        xi_iv = [iv.mpf(float(v)) for v in xi]
        N_iv = [iv.mpf(float(v)) for v in N]
        eq_change = max(abs(float(xi[0]) - 1.0),
                        max(abs(float(N[i * 3]) - float(self.u[i]))
                            for i in range(self.dP)))
        xi_iv[0] = iv.mpf(1)
        for i in range(self.dP):
            N_iv[i * 3] = _q(self.u[i])
        return xi_iv, N_iv, eq_change

    def interior(self):
        s0 = [Fraction(1), Fraction(1, 2), Fraction(1, 2)]
        if self.ga_threshold is not None:
            s0[1] = Fraction(4, 5)                      # q_{1|x=0} = 0.8 > 0.7
            if not s0[1] > self.ga_threshold:
                raise RuntimeError("interior point violates G_A")
        k0 = self.tau
        xi0 = [_q(s0[al]) * _q(k0[i]) for al in range(3)
               for i in range(self.dP)]
        s1 = [Fraction(1), Fraction(1, 2), Fraction(1, 2)]
        N0 = [_q(self.u[i]) * _q(s1[be]) for i in range(self.dP)
              for be in range(3)]
        return xi0, N0

    def value(self, xi, N):
        """<W, xi (x) N> = sum_{al,i,be} C[al,be] xi[al*dP+i] N[i*3+be]."""
        tot = iv.mpf(0)
        for al in range(3):
            for be in range(3):
                c = self.C[al][be]
                if c == 0:
                    continue
                s = iv.mpf(0)
                for i in range(self.dP):
                    s += xi[al * self.dP + i] * N[i * 3 + be]
                tot += _q(c) * s
        return tot

    # ---- the certificate -------------------------------------------------
    def certify(self, xi, N, eps=1e-9, eps_max=1e-3):
        """Rigorous lower bound from the mixed strategy; returns a dict."""
        xi_iv, N_iv, eq_change = self.embed(xi, N)
        xi0, N0 = self.interior()
        rows_A = list(self.phi_rows())
        rows_B = list(self.psi_rows())
        ga = self.ga_row()
        while True:
            e = _q(Fraction(eps).limit_denominator(10 ** 12))
            om = iv.mpf(1) - e
            xe = [om * a + e * b for a, b in zip(xi_iv, xi0)]
            Ne = [om * a + e * b for a, b in zip(N_iv, N0)]
            slacks = [sum(r * v for r, v in zip(row, xe)) for row in rows_A]
            slacks += [sum(r * v for r, v in zip(row, Ne)) for row in rows_B]
            if ga is not None:
                slacks.append(sum(r * v for r, v in zip(ga, xe)))
            min_lo = min(_lo(s) for s in slacks)
            if min_lo > 0:
                break
            eps *= 10
            if eps > eps_max:
                return dict(certified=False, eps=eps / 10,
                            min_slack_lower_bound=float(min_lo))
        val = self.value(xe, Ne)
        lo = _lo(val)
        f = float(lo)
        if mp.mpf(f) > lo:                     # round the float DOWN
            f = float(np.nextafter(f, -np.inf))
        out = dict(certified=True, method="interval arithmetic (mpmath.iv, "
                   f"{iv.dps} digits) on the eps-mixed strategy",
                   eps=eps, L_certified=f,
                   L_certified_str=mpmath.nstr(lo, 20),
                   value_interval_width=float(_hi(val) - lo),
                   min_slack_lower_bound=float(min_lo),
                   equality_coordinates_changed_by=eq_change)
        if self.rational:
            out["exact"] = self.certify_exact(xi, N)
        return out

    def certify_exact(self, xi, N, max_den=10 ** 6):
        """Rational instance only: rationalise the candidate and check it
        exactly (no eps); returns the exact value if feasible."""
        xq = [Fraction(float(v)).limit_denominator(max_den) for v in xi]
        Nq = [Fraction(float(v)).limit_denominator(max_den) for v in N]
        xq[0] = Fraction(1)
        for i in range(self.dP):
            Nq[i * 3] = Fraction(self.u[i])
        dev = max([abs(float(a) - float(b)) for a, b in zip(xq, xi)] +
                  [abs(float(a) - float(b)) for a, b in zip(Nq, N)])
        ok = True
        for e in EPS_FUNCTIONALS.values():
            for h in self.dual_q:
                s = sum(Fraction(e[al]) * Fraction(h[i]) * xq[al * self.dP + i]
                        for al in range(3) for i in range(self.dP))
                ok &= s >= 0
        for g in self.gens_q:
            for e in EPS_FUNCTIONALS.values():
                s = sum(Fraction(g[i]) * Fraction(e[be]) * Nq[i * 3 + be]
                        for i in range(self.dP) for be in range(3))
                ok &= s >= 0
        if self.ga_threshold is not None:
            ok &= xq[self.dP] - self.ga_threshold * sum(
                Fraction(self.u[i]) * xq[i] for i in range(self.dP)) >= 0
        val = sum(self.C[al][be] * xq[al * self.dP + i] * Nq[i * 3 + be]
                  for al in range(3) for be in range(3)
                  for i in range(self.dP))
        return dict(rationalised=True, max_denominator=max_den,
                    rationalisation_distance=dev, feasible_exactly=bool(ok),
                    value_exact=str(val) if ok else None,
                    value_exact_float=float(val) if ok else None,
                    # "nice": the candidate was a genuinely rational strategy
                    # (small denominators), not an arbitrary float that was
                    # merely rationalised
                    value_exact_is_nice=bool(ok and val.denominator <= max_den
                                             and dev < 1e-9),
                    xi_exact=[str(v) for v in xq] if ok else None,
                    N_exact=[str(v) for v in Nq] if ok else None)


def exact_instance_for(D) -> ExactInstance:
    """Build the ExactInstance matching a DirectData object of
    direct_hierarchy.py (by name) and check the float data against it."""
    name = D.name
    thr = None if D.GA is None else Fraction(7, 10)
    if name == "classical bit":
        inst = ExactInstance(name, "classical bit",
                             rational_gens=[[1, 0], [1, 1]],
                             rational_dual=[[1, -1], [0, 1]])
    elif name == "square gbit":
        sq_gens = [[1, 1, 1], [1, -1, 1], [1, -1, -1], [1, 1, -1]]
        # cross products of consecutive vertices, direction h[0] > 0
        sq_dual = []
        for j in range(4):
            g1, g2 = sq_gens[j], sq_gens[(j + 1) % 4]
            h = [g1[1] * g2[2] - g1[2] * g2[1], g1[2] * g2[0] - g1[0] * g2[2],
                 g1[0] * g2[1] - g1[1] * g2[0]]
            if h[0] < 0:
                h = [-c for c in h]
            sq_dual.append(h)
        inst = ExactInstance(name, "polygon", m=4,
                             radius_expr=lambda iv: iv.sqrt(iv.mpf(2)),
                             phase_expr=lambda iv: iv.pi / 4,
                             rational_gens=sq_gens, rational_dual=sq_dual)
    elif name in ("pentagon", "pentagon+ineq"):
        inst = ExactInstance(name, "polygon", m=5,
                             radius_expr=lambda iv: 1 / iv.sqrt(iv.cos(iv.pi / 5)),
                             phase_expr=lambda iv: iv.mpf(0),
                             ga_threshold=thr)
    else:
        raise ValueError(name)
    # sanity: the float rays of direct_hierarchy.py and the interval rays
    # describe the same cones (same directions up to positive scale, 1e-12)
    def same_directions(rows_iv, rows_f, tol=1e-12):
        def unit(v):
            v = np.asarray(v, float)
            return v / np.abs(v).max()
        mids = [unit([float((_lo(c) + _hi(c)) / 2) for c in r])
                for r in rows_iv]
        for rf in rows_f:
            u = unit(rf)
            if min(np.abs(m - u).max() for m in mids) > tol:
                return False
        return len(mids) == len(rows_f)
    if not same_directions(list(inst.phi_rows()), list(D.Phi)):
        raise RuntimeError(f"{name}: float ext C_asm^* differs from the "
                           f"interval instance")
    if not same_directions(list(inst.psi_rows()), list(D.Psi)):
        raise RuntimeError(f"{name}: float ext C_mm^* differs from the "
                           f"interval instance")
    return inst
