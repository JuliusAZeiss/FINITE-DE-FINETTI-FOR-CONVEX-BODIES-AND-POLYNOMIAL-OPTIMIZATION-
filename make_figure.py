"""Figure of the numerics section: outer levels p^(n), inner rounded values
L^(n) and the exact optimum gamma* for the four CHSH experiments.

Reads results.json (hierarchy levels + inner rounding, written by
reproduce.py) next to this script.  The exact reference values gamma* are
the closed forms of the direct computation (exact_game_value.py); if
exact_game_values.json is present next to this script the closed forms are
cross-checked against it.

    python3 make_figure.py      ->  hierarchy_convergence.{pdf,png}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif", "font.size": 9.5,
    "mathtext.fontset": "cm", "axes.linewidth": 0.7,
    "xtick.direction": "in", "ytick.direction": "in",
})

HERE = os.path.dirname(os.path.abspath(__file__))
st = json.load(open(os.path.join(HERE, "results.json")))
ns = np.arange(1, 6)

# exact gamma*(CHSH) (winning-probability convention), closed forms
S5 = np.sqrt(5.0)
EXACT = {
    "classical bit": 3.0 / 4.0,
    "square gbit": 1.0,
    "pentagon": (2.0 * S5 - 1.0) / 4.0,
    "pentagon+ineq": 3.0 * (9.0 + S5) / 40.0,
}
if os.path.exists(os.path.join(HERE, "exact_game_values.json")):
    dg = json.load(open(os.path.join(HERE, "exact_game_values.json")))
    for key, val in EXACT.items():
        ref = dg["experiments"][key]["gamma_star"]["value_float"]
        assert abs(ref - val) < 1e-12, (key, ref, val)
for key, val in EXACT.items():          # results.json carries the same values
    if "gamma_star_exact" in st[key]:
        assert abs(st[key]["gamma_star_exact"] - val) < 1e-12, key

# colour-blind-safe categorical palette (validated), fixed order;
# marker shapes give a second encoding
series = [
    ("classical bit",  "classical bit",  "#4a3aa7", "o"),
    ("square gbit",    "square gbit",    "#2a78d6", "s"),
    ("pentagon",       "pentagon",       "#d95926", "D"),
    ("pentagon+ineq",  r"pentagon, $q_{1|0}\geq 0.7$", "#199e70", "^"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.75))

# ---- left: the certified sandwich ---------------------------------------
for key, lbl, col, mk in series:
    v = [st[key]["levels"][str(n)]["value"] for n in ns]
    w = [st[key]["inner"][str(n)]["best"] for n in ns]
    ax1.axhline(EXACT[key], color=col, lw=0.8, ls=":", alpha=0.85, zorder=1)
    ax1.plot(ns, v, marker=mk, ms=4, lw=1.1, color=col, label=lbl, zorder=3)
    ax1.plot(ns, w, marker=mk, ms=4, lw=1.0, color=col, ls="--",
             mfc="white", zorder=3)
ax1.set_xlabel(r"hierarchy level $n$")
ax1.set_ylabel(r"game value")
ax1.set_xticks(ns)
leg1 = ax1.legend(frameon=False, fontsize=7, loc="upper right",
                  handlelength=1.6)
style_handles = [
    Line2D([], [], color="0.3", lw=1.1, marker="o", ms=4,
           label=r"outer $p^{(n)}$"),
    Line2D([], [], color="0.3", lw=1.0, ls="--", marker="o", ms=4,
           mfc="white", label=r"inner $L^{(n)}$"),
    Line2D([], [], color="0.3", lw=0.8, ls=":", label=r"exact $\gamma^*$"),
]
ax1.add_artist(leg1)
ax1.legend(handles=style_handles, frameon=False, fontsize=7,
           loc="lower center", bbox_to_anchor=(0.56, -0.01), ncol=1,
           handlelength=1.8)
ax1.set_ylim(0.28, 1.55)

# ---- right: gaps to the exact optimum (pentagon experiments) ------------
for key, lbl, col, mk in series[2:]:
    ex = EXACT[key]
    go = np.array([st[key]["levels"][str(n)]["value"] for n in ns]) - ex
    gi = ex - np.array([st[key]["inner"][str(n)]["best"] for n in ns])
    assert go.min() > 0 and gi.min() > 0      # exact value inside bracket
    ax2.loglog(ns, go, marker=mk, ms=4, lw=1.1, color=col, label=lbl)
    ax2.loglog(ns, gi, marker=mk, ms=4, lw=1.0, ls="--", color=col,
               mfc="white")
nn = np.linspace(1, 5, 50)
ax2.loglog(nn, 0.474 / np.sqrt(nn), "--", color="0.6", lw=0.9,
           label=r"$\propto n^{-1/2}$ (Thm.)")
ax2.loglog(nn, 0.474 / nn, "-.", color="0.35", lw=0.9,
           label=r"$\propto n^{-1}$")
ax2.set_xlabel(r"hierarchy level $n$")
ax2.set_ylabel(r"gap to exact $\gamma^*$")
ax2.set_xticks(ns); ax2.set_xticklabels([str(n) for n in ns])
ax2.legend(frameon=False, fontsize=7, loc="lower left", handlelength=1.8)

fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(HERE, "hierarchy_convergence.pdf"))
fig.savefig(os.path.join(HERE, "hierarchy_convergence.png"), dpi=220)
print("figure written: hierarchy_convergence.pdf/.png")
