"""Reproduce every number of the numerics section (outer levels p^(n) and
rounded inner values L^(n) for n = 1..5, the four CHSH experiments).

    python3 reproduce.py                 # ~15-40 min on one CPU core
    python3 reproduce.py --nmax 3        # quick run (seconds)
    python3 reproduce.py --out my.json   # do not overwrite the shipped file

Pipeline per experiment (classical bit, square gbit, pentagon, pentagon with
the local inequality q_{1|0} >= 0.7):
  1. hierarchy levels p^(1..nmax) of the theorem "Convergence of the
     hierarchy", each solved as an *exact* linear program with the complete
     facet list:
       - classical bit: plain full facet list (hierarchy_level_full),
       - square gbit / pentagon (+ inequality variant): dihedral
         orbit-averaged full list (hierarchy_level_symfull);
  2. the rounding scheme of the theorem "Convergent inner sequence" applied
     to the optimizer returned by the LP solver (inner_rounding) -> L^(n).

Both sequences are checked against the exact optima gamma*(CHSH) of
exact_game_value.py (closed forms below): L^(n) - tol <= gamma* <= p^(n) + tol.

Outputs: results.json (default; the shipped file is the run behind Table 1 of
the paper -- pass --out to keep it) and hierarchy_convergence.{pdf,png}.

Reproducibility: the outer values p^(n) are LP optima and reproduce to
~1e-12 on any machine.  The inner values L^(n) depend on WHICH optimizer of
the (degenerate) optimal face the solver returns -- observation (e) of the
paper -- so they can differ between HiGHS versions/platforms; every value is
nevertheless a valid lower bound attained by an explicit feasible strategy.
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import numpy as np
import scipy

from gpt_definetti_games import (GameProblem, chsh_coefficients,
                                 classical_bit, eps_functional,
                                 hierarchy_level_full,
                                 hierarchy_level_symfull, inner_rounding,
                                 pentagon_selfdual, square_gbit)

S5 = np.sqrt(5.0)
# exact optima gamma*(CHSH), winning-probability convention
# (exact_game_value.py: certified closed forms over Q(sqrt5))
EXACT = {
    "classical bit": 3.0 / 4.0,
    "square gbit": 1.0,
    "pentagon": (2.0 * S5 - 1.0) / 4.0,
    "pentagon+ineq": 3.0 * (9.0 + S5) / 40.0,
}
TOL = 1e-8           # LP solver tolerance scale


def make_experiments():
    pent = pentagon_selfdual()
    plain = GameProblem(pent, chsh_coefficients(), name="pentagon")
    GA = np.kron(eps_functional(0, 1), pent.u) - 0.7 * plain.uA
    ineq = GameProblem(pent, chsh_coefficients(), name="pentagon+ineq",
                       GA_rows=GA[None, :], GA_desc="q_{1|x=0} >= 0.7")
    return [
        (GameProblem(classical_bit(), chsh_coefficients(),
                     name="classical bit"), hierarchy_level_full),
        (GameProblem(square_gbit(), chsh_coefficients(),
                     name="square gbit"), hierarchy_level_symfull),
        (plain, hierarchy_level_symfull),
        (ineq, hierarchy_level_symfull),
    ]


def environment():
    import highspy
    return dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, highspy=highspy.Highs().version(),
                platform=platform.platform(), machine=platform.machine(),
                date=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=5)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    levels = list(range(1, args.nmax + 1))

    st = {"_provenance": dict(
        script="reproduce.py", environment=environment(),
        reference="exact gamma*(CHSH) of exact_game_value.py",
        note="p^(n): LP optima of the complete (reduced) facet description; "
             "L^(n): rounding of the optimizer returned by the LP solver "
             "(optimizer dependent, see observation (e) of the paper)")}
    for prob, solver in make_experiments():
        print(f"== {prob.name} ==")
        ex = EXACT[prob.name]
        ent = {"GA_desc": prob.GA_desc, "gamma_star_exact": ex,
               "levels": {}, "inner": {}}
        for n in levels:
            res, _, Y = solver(prob, n, verbose=False, return_solution=True)
            ent["levels"][str(n)] = res.__dict__
            rd = inner_rounding(prob, n, Y)
            ent["inner"][str(n)] = {
                "best": rd["best"], "best_m": rd["best_m"],
                "avg_at_best_m": rd["per_m"][rd["best_m"]]["avg"],
                "per_m_best": {str(m): d["best"]
                               for m, d in rd["per_m"].items()},
                "min_slack": rd["min_slack"]}
            print(f"  p^({n}) = {res.value:.9f}   L^({n}) = "
                  f"{rd['best']:.9f} (m*={rd['best_m']})   "
                  f"gamma* = {ex:.9f}   [{res.certified_by}, "
                  f"{res.seconds:.0f}s]")
            # the sandwich  L^(n) <= gamma* <= p^(n)  (up to LP tolerance)
            if not res.value >= ex - TOL:
                raise RuntimeError(f"{prob.name} n={n}: outer value "
                                   f"{res.value!r} below gamma* {ex!r}")
            if not rd["best"] <= ex + TOL:
                raise RuntimeError(f"{prob.name} n={n}: inner value "
                                   f"{rd['best']!r} above gamma* {ex!r}")
            if not rd["min_slack"] > -1e-6:
                raise RuntimeError(f"{prob.name} n={n}: rounded candidate "
                                   f"infeasible (slack {rd['min_slack']})")
        st[prob.name] = ent
    with open(args.out, "w") as f:
        json.dump(st, f, default=float, indent=1)
    print(f"{args.out} written")
    if not args.no_figure and args.out == "results.json" and args.nmax == 5:
        import make_figure  # noqa: F401  (renders the figure on import)


if __name__ == "__main__":
    main()
