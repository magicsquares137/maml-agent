#!/usr/bin/env python3
"""Fig 1 (hero): developmental task-vector geometry over RL training.
cosine(tau_d1, tau_d2) rises 0.002 -> 0.10 and saturates, while |tau| grows ~4x
and sign agreement stays at chance. Reads paper/data/trajectory.csv."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "data", "trajectory.csv"))))
it   = [int(r["iter"]) for r in rows]
cos  = [float(r["cosine"]) for r in rows]
n1   = [float(r["tau_d1"]) for r in rows]
n2   = [float(r["tau_d2"]) for r in rows]
sign = [float(r["sign_agree_pct"]) for r in rows]

fig, ax = plt.subplots(figsize=(5.4, 3.6))

# hero: cosine (left axis)
ax.plot(it, cos, "o-", color="#c0392b", lw=2.2, ms=6, label=r"cos$(\tau_{d1},\tau_{d2})$", zorder=3)
ax.set_xlabel("RL iteration")
ax.set_ylabel(r"cosine similarity", color="#c0392b")
ax.tick_params(axis="y", labelcolor="#c0392b")
ax.set_ylim(0, 0.13)
ax.set_xticks(it)
ax.axhline(0, color="0.8", lw=0.8, zorder=0)
ax.annotate("saturates ~0.10\n(still ~84°, near-orthogonal)",
            xy=(9, 0.103), xytext=(5.2, 0.118), fontsize=8, color="#c0392b",
            ha="left", va="top")

# magnitude growth (right axis)
ax2 = ax.twinx()
ax2.plot(it, n1, "s--", color="#2c3e50", lw=1.3, ms=4, alpha=0.75, label=r"$|\tau_{d1}|$")
ax2.plot(it, n2, "^--", color="#7f8c8d", lw=1.3, ms=4, alpha=0.75, label=r"$|\tau_{d2}|$")
ax2.set_ylabel(r"task-vector norm $|\tau|$", color="#2c3e50")
ax2.tick_params(axis="y", labelcolor="#2c3e50")
ax2.set_ylim(0, 6)

# chance-line annotation for sign agreement (no 3rd axis; keep clean)
ax.text(1.05, 0.006, "sign agreement flat at chance (50-52%) throughout",
        fontsize=7.5, color="0.4", style="italic")

# merged legend
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8, framealpha=0.9)

ax.set_title("Task-vector geometry develops over RL training", fontsize=10.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, "figures", f"fig1_trajectory.{ext}"), dpi=200, bbox_inches="tight")
print("wrote figures/fig1_trajectory.{png,pdf}")
