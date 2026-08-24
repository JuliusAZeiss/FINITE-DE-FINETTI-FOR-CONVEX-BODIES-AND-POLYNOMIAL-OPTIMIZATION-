# Cross-checks without any reduction, and the optimizer dependence of L^(n)

Companion folder to Sections 8–9 of `../implementation_notes.pdf`. The
scripts import `gpt_definetti_games.py` and read `results.json` from the
parent folder; all paths are resolved relative to the scripts, so the commands
work from any directory.

## What is here

**The level-n LPs with none of the reductions** (no multiset basis, no
ker F_B parametrisation, no orbit averaging): the full extension Y as one free
variable per tensor entry, all product facets over ordered tuples, permutation
invariance as explicit equality rows, normalisation, F_B on the first copy and
the conic G_A rows of the constrained experiment.

| script | formulation | reach |
|---|---|---|
| `direct_hierarchy.py` | (A) explicit Kronecker matrix | n ≤ 3 (polygons), n ≤ 5 (bit); ~1 h in total |
| `direct_hierarchy_tree.py` | (B) complete facet family with intermediate contraction variables | pentagon n = 4 (~2.6 h, ~5 GB) |
| `direct_summary.py` | merges the run records of `direct_check_runs/` into `direct_results.json` + `direct_results_table.md` | seconds |

Result (`direct_results_table.md`): every unreduced level value agrees with the
reduced pipeline's `results.json` to within 1e-12 — the reductions are exact.

**The rounding of the unreduced optimizers**, and the dependence of the rounded
value on *which* optimizer is rounded (observation (e) of the paper):

| script | what it does | time |
|---|---|---|
| `direct_inner_rounding.py --nmax 3` | rounds the stored optimizers of (A) in the full tensor picture, verifies every candidate, cross-checks against the library's `inner_rounding`, certifies the best candidates rigorously; writes `direct_inner_results.json` + `direct_inner_table.md` | seconds |
| `direct_inner_certify.py` | (module) rigorous lower bounds: interval arithmetic on an ε-mixed strategy (mpmath, 60 digits) and exact rational checks for the bit and the square gbit | — |
| `direct_inner_sensitivity.py` | rounds the optimizers returned by four solver routes for the same level-2 LP (dual simplex, IPM with/without crossover, tree formulation), raw and group-averaged; writes `direct_inner_sensitivity.json` + `.md`, including the ranges quoted in observation (e) | seconds |

## Stored optimizers

`direct_optimizers/` holds every optimizer Y behind the shipped numbers as
`*_Y.npy` with a side-car `*_Y.json` (SHA-256 of the array, objective, HiGHS
status and options, iteration counts, a-posteriori residuals with pass/fail
flags, software environment, source hash); `direct_check_runs/` holds the raw
run records of (A) and (B), including the pentagon n = 4 tensor.

`direct_inner_rounding.py` and `direct_inner_sensitivity.py` **load these files
by default** and therefore reproduce the shipped tables digit by digit on any
machine. With `--resolve` they re-solve the LPs instead: HiGHS is deterministic
only within one build/platform, so the returned optimizers — and with them the
rounded values — may then differ, while the outer values stay the same.
Re-solving overwrites the stored optimizers.

Note on provenance hashes: the raw run records of `direct_check_runs/`
and the side-car files record the SHA-256 of the `results.json` they were
compared with at the time of the run; that file has since been trimmed to its
level and rounding fields (a heuristic lower-bound field, equal to the exact
optimum in all four experiments, was removed), so the recorded hash refers to the original file, while all
level values compared are unchanged. `direct_results.json`,
`direct_inner_results.json` and `direct_inner_sensitivity.json` were
regenerated from the stored records and refer to the shipped `results.json`.

## Dependencies

`numpy`, `scipy`, `highspy` (≥ 1.15) and `mpmath` (for the certificates;
`--no-certify` skips them), plus `../gpt_definetti_games.py` and
`../results.json`.
