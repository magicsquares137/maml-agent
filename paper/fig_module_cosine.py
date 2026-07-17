#!/usr/bin/env python3
"""Fig 2: near-orthogonality is uniform across module types.
Per module-type cosine(tau_d1, tau_d2); global cosine as reference line.
Reads paper/data/module_cosine.csv."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GLOBAL_COS = 0.061
rows = list(csv.DictReader(open(os.path.join(HERE, "data", "module_cosine.csv"))))
rows.sort(key=lambda r: float(r["cosine"]))
mods = [r["module"] for r in rows]
cos  = [float(r["cosine"]) for r in rows]

fig, ax = plt.subplots(figsize=(5.2, 3.2))
bars = ax.barh(mods, cos, color="#3498db", edgecolor="#21618c", height=0.66)
ax.axvline(GLOBAL_COS, color="#c0392b", ls="--", lw=1.4, label=f"global cos = {GLOBAL_COS:.3f}")
ax.axvline(0, color="0.7", lw=0.8)
for b, c in zip(bars, cos):
    ax.text(c + 0.003, b.get_y() + b.get_height() / 2, f"{c:.3f}",
            va="center", fontsize=8, color="#21618c")
ax.set_xlabel(r"cosine$(\tau_{d1}, \tau_{d2})$")
ax.set_xlim(0, 0.16)
ax.set_title("Specialist task vectors are near-orthogonal in every module", fontsize=10)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, "figures", f"fig2_module_cosine.{ext}"), dpi=200, bbox_inches="tight")
print("wrote figures/fig2_module_cosine.{png,pdf}")
