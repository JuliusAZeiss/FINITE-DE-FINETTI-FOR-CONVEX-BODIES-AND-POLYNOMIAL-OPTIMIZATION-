"""Level-1 diagnostics for observation (a) of the numerics section.

Re-solves the level-1 LP (K_asm (x)^ K_mm) of the four CHSH experiments with
gpt_definetti_games.py and contracts the optimizer with the Choi tensor
chi_{A(K_B)^*}, i.e. computes the induced table p(a,b|x,y).  Expected output:

  * classical bit : p^(1) = 1, the table is the PR box (nonnegative,
                    normalized, no-signalling), value = no-signalling value;
  * square gbit   : p^(1) = 3/2, entries down to -1/4;
  * pentagon      : p^(1) = 3/sqrt(5), entries down to (1-3/sqrt5)/2;
  * pentagon+ineq : p^(1) = 1.1733..., negative entries as well.

All tables are normalized and no-signalling; only the classical bit (a
simplex, whose cone is simplicial) gives nonnegative entries -- for the
square gbit and the pentagon the level-1 optimizer lives in the maximal
tensor product K_asm (x)^ K_mm, on which the Choi contraction is not
positive, so p^(1) exceeds the algebraic maximum 1 of a winning probability.

Usage:  python3 level1_pseudocorrelations.py        (~1 s)
"""
import itertools, json
import numpy as np
from gpt_definetti_games import (GameProblem, chsh_coefficients, classical_bit,
                                 eps_functional, hierarchy_level_full,
                                 hierarchy_level_symfull, pentagon_selfdual,
                                 square_gbit)

def experiments():
    pent = pentagon_selfdual()
    plain = GameProblem(pent, chsh_coefficients(), name="pentagon")
    GA = np.kron(eps_functional(0, 1), pent.u) - 0.7 * plain.uA
    ineq = GameProblem(pent, chsh_coefficients(), name="pentagon+ineq",
                       GA_rows=GA[None, :], GA_desc="q_{1|x=0} >= 0.7")
    return [(GameProblem(classical_bit(), chsh_coefficients(), name="classical bit"), hierarchy_level_full),
            (GameProblem(square_gbit(), chsh_coefficients(), name="square gbit"), hierarchy_level_symfull),
            (plain, hierarchy_level_symfull), (ineq, hierarchy_level_symfull)]

def pseudo_correlations(prob, Yfull):
    dP = prob.dP
    # contraction over the K_B index i (Choi pairing): M[alpha, beta] = sum_i Y[(alpha,i),(i,beta)]
    M = np.zeros((3, 3))
    for al in range(3):
        for be in range(3):
            M[al, be] = sum(Yfull[al*dP + i, i*3 + be] for i in range(dP))
    p = np.zeros((2, 2, 2, 2))
    for a, b, x, y in itertools.product(range(2), repeat=4):
        p[a, b, x, y] = eps_functional(x, a) @ M @ eps_functional(y, b)
    return p

g = chsh_coefficients()
for prob, solver in experiments():
    res, _, x = solver(prob, 1, verbose=False, return_solution=True)
    Yred = x.reshape(prob.dA, prob.dBr)
    Yfull = Yred @ prob.Bred.T
    p = pseudo_correlations(prob, Yfull)
    val = float((g * p).sum())
    print(f"== {prob.name}: p^(1) = {res.value:.10f}  (from pseudo-correlations: {val:.10f})")
    print(f"   min entry = {p.min():+.6f}, max entry = {p.max():+.6f}")
    for x_, y_ in itertools.product(range(2), repeat=2):
        row = "  ".join(f"p({a}{b}|{x_}{y_})={p[a,b,x_,y_]:+.4f}" for a, b in itertools.product(range(2), repeat=2))
        print("   ", row, f"  sum={p[:,:,x_,y_].sum():+.4f}")
    # no-signalling check of the marginals
    pa = p.sum(axis=1)  # a,x,y
    pb = p.sum(axis=0)  # b,x,y
    nsA = max(abs(pa[a,x_,0]-pa[a,x_,1]) for a in range(2) for x_ in range(2))
    nsB = max(abs(pb[b,0,y_]-pb[b,1,y_]) for b in range(2) for y_ in range(2))
    print(f"   no-signalling violation: Alice {nsA:.2e}, Bob {nsB:.2e}; CHSH S = {4*sum(g[a,b,x_,y_]*p[a,b,x_,y_]*(1) for a,b,x_,y_ in itertools.product(range(2),repeat=4))*2-4:.4f}")
