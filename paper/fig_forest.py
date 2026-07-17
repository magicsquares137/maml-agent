#!/usr/bin/env python3
"""Fig 3: forest plot of paired ΔTGC with 95% bootstrap CIs.
Every merge/joint contrast crosses 0 (null); RL-vs-base excludes 0 (the one real
effect). Reads paper/stats.json (run build_stats.py first)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
stats = json.load(open(os.path.join(HERE, "stats.json")))
forest = stats["forest"]

# order: base contrasts at top (the real effect), merge/joint nulls below
forest = sorted(forest, key=lambda f: (not f["label"].endswith("base*"), f["point"]))
labels = [f["label"] for f in forest]
pts    = [f["point"] for f in forest]
los    = [f["lo"] for f in forest]
his    = [f["hi"] for f in forest]
y      = list(range(len(forest)))

from matplotlib.lines import Line2D
fig, ax = plt.subplots(figsize=(5.6, 3.4))
for yi, f in zip(y, forest):
    real = not f["crosses0"]
    col = "#27ae60" if real else "#7f8c8d"
    ax.plot([f["lo"], f["hi"]], [yi, yi], "-", color=col, lw=2.2)
    ax.plot(f["point"], yi, "o", color=col, ms=7,
            markeredgecolor="white", markeredgewidth=0.8, zorder=3)
ax.axvline(0, color="#c0392b", ls="--", lw=1.2)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel(r"$\Delta$ TGC (percentage points), 95% CI")
ax.set_title("Only RL-over-base clears the noise wall", fontsize=10.5)
legend_elems = [
    Line2D([0], [0], color="#27ae60", lw=2.2, marker="o", ms=6, label="excludes 0 (real effect)"),
    Line2D([0], [0], color="#7f8c8d", lw=2.2, marker="o", ms=6, label="crosses 0 (indistinguishable)"),
]
ax.legend(handles=legend_elems, loc="upper right", fontsize=8, framealpha=0.92)
ax.margins(y=0.08)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, "figures", f"fig3_forest.{ext}"), dpi=200, bbox_inches="tight")
print("wrote figures/fig3_forest.{png,pdf}")
