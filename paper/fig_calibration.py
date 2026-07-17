#!/usr/bin/env python3
"""Fig 4: floor/ceiling calibration of the cross-specialist cosine.
Shows the measured 0.07-0.10 sits an order of magnitude above the random null
(spectrum-matched) and an order of magnitude below same-run self-alignment.
Refutes "it's just a LoRA construction artifact." Reads paper/calibration.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
c = json.load(open(os.path.join(HERE, "calibration.json")))

GRAY, RED, BLUE = "#7f8c8d", "#c0392b", "#2471a3"
fig, ax = plt.subplots(figsize=(6.2, 2.9))

# row 0: null (spectrum-matched) band + mean
ns = c["floor_spectrum"]
ax.barh(0, ns["hi"] - ns["lo"], left=ns["lo"], height=0.34, color=GRAY, alpha=0.35,
        edgecolor=GRAY, zorder=1)
ax.plot(ns["mean"], 0, "o", color=GRAY, ms=7, zorder=3)
ax.annotate(f"random null (spectrum-matched)\nmean {ns['mean']:+.4f}, 95% [{ns['lo']:+.4f}, {ns['hi']:+.4f}]",
            xy=(ns["hi"], 0), xytext=(0.16, 0), fontsize=8, va="center", color="#4d4d4d")

# row 1: cross-specialist iter5 -> iter9
m5, m9 = c["measured"]["iter5"], c["measured"]["iter9"]
ax.plot([m5, m9], [1, 1], "-", color=RED, lw=1.4, alpha=0.6, zorder=2)
ax.plot(m5, 1, "o", color=RED, ms=7, mfc="white", mec=RED, mew=1.8, zorder=3)
ax.plot(m9, 1, "o", color=RED, ms=8, zorder=3)
ax.annotate(f"cross-specialist  cos($\\tau_{{d1}},\\tau_{{d2}}$)\niter5 {m5:.3f} $\\rightarrow$ iter9 {m9:.3f}",
            xy=(m9, 1), xytext=(0.16, 1), fontsize=8, va="center", color=RED)

# row 2: same-run self (ceiling), plus early iter1->9 as lighter
cd1, cd2, early = c["ceiling"]["d1_5v9"], c["ceiling"]["d2_5v9"], c["ceiling"]["d1_1v9"]
ax.plot(early, 2, "o", color=BLUE, ms=6, alpha=0.4, zorder=3)
ax.plot([cd1, cd2], [2, 2], "-", color=BLUE, lw=1.4, alpha=0.6, zorder=2)
ax.plot(cd1, 2, "o", color=BLUE, ms=8, zorder=3)
ax.plot(cd2, 2, "o", color=BLUE, ms=8, zorder=3)
ax.annotate(f"same-run self (ceiling)\niter1$\\rightarrow$9 {early:.2f};  iter5$\\rightarrow$9 {cd1:.2f}/{cd2:.2f}",
            xy=(cd2, 2), xytext=(cd2 + 0.02, 2), fontsize=8, va="center", color=BLUE)

ax.axvline(0, color="0.8", lw=0.8, zorder=0)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["null", "measured", "ceiling"], fontsize=9)
ax.set_ylim(-0.6, 2.6)
ax.set_xlim(-0.03, 1.02)
ax.set_xlabel(r"cosine similarity")
ax.set_title("Cross-specialist alignment: real, but far from floor or ceiling", fontsize=10.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, "figures", f"fig4_calibration.{ext}"), dpi=200, bbox_inches="tight")
print("wrote figures/fig4_calibration.{png,pdf}")
