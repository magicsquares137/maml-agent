#!/usr/bin/env python3
"""Fig 5 (future work): interference grid I(s1,s2)=cos(tau_d1(s1),tau_d2(s2)) over
training checkpoints. Interference is low ONLY in the undertrained corner and rises
with both indices toward the capability peak (5,5) -- so a mergeability-aware
objective has no high-capability/low-interference cell to select. Reads
paper/data/interference_grid.csv."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.reader(open(os.path.join(HERE, "data", "interference_grid.csv"))))
iters = [int(x) for x in rows[0][1:]]
G = np.array([[float(v) for v in r[1:]] for r in rows[1:]])

fig, ax = plt.subplots(figsize=(4.8, 4.1))
im = ax.imshow(G, origin="lower", cmap="viridis", aspect="equal",
               extent=[0.5, 9.5, 0.5, 9.5])
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label(r"interference  $\cos(\tau_{d1}(s_1),\tau_{d2}(s_2))$", fontsize=8)

# mark the reward-peak / capability-max pair (5,5) = the merge actually evaluated
ax.plot(5, 5, "o", ms=13, mfc="none", mec="white", mew=2.2)
ax.annotate("reward peak (5,5)\n= merge evaluated\n= argmax J ($\\gamma$ small)",
            xy=(5, 5), xytext=(5.4, 2.4), color="white", fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color="white", lw=1.1))
# mark the degenerate low-interference pick (5,1)
ax.plot(1, 5, "s", ms=10, mfc="none", mec="red", mew=2.0)
ax.annotate("(5,1): objective's\n$\\gamma\\!\\geq\\!2$ pick =\ndiscard diff-2 training",
            xy=(1, 5), xytext=(1.3, 7.4), color="red", fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color="red", lw=1.1))

ax.set_xticks(iters); ax.set_yticks(iters)
ax.set_xlabel(r"diff-2 checkpoint $s_2$")
ax.set_ylabel(r"diff-1 checkpoint $s_1$")
ax.set_title("Interference is low only where a specialist is undertrained", fontsize=9.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, "figures", f"fig5_jgrid.{ext}"), dpi=200, bbox_inches="tight")
print("wrote figures/fig5_jgrid.{png,pdf}")
