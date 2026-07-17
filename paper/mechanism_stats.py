#!/usr/bin/env python3
"""Continuous-metric + mechanism analysis for the paper's secondary section.
All from per-task result JSONs (no GPU). Writes paper/mechanism.json.

Produces:
  - TGC vs mean partial-credit score per model  (the metric-reversal table)
  - paired McNemar (binary) vs Wilcoxon (continuous) for key contrasts
  - super-additivity test: does joint exceed per-task max(specialists)?  (emergence)
  - averaging test: are merges closer to per-task avg or max of specialists?
"""
import json, os
from statistics import mean
from math import comb
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
AR = os.environ.get("APPWORLD_ROOT", "/home/smcclendon/Documents/github/appworld/appworld-rl")
F = {"diff-1": "results_diff_1_iter5.json", "diff-2": "results_diff_2_iter5.json",
     "joint": "results_joint_iter9.json", "TIES": "results_merged_ties.json",
     "RAM+": "results_merged_ram.json"}

SC = {k: {x["task_id"]: float(x["score"]) for x in json.load(open(os.path.join(ROOT, v)))["results"]} for k, v in F.items()}
SU = {k: {x["task_id"]: bool(x["success"]) for x in json.load(open(os.path.join(ROOT, v)))["results"]} for k, v in F.items()}
KS = sorted(set.intersection(*[set(SC[k]) for k in F]))
sc = lambda k: [SC[k][t] for t in KS]
su = lambda k: [SU[k][t] for t in KS]


def mcnemar(a, b):
    A, B = su(a), su(b)
    bo = sum(1 for x, y in zip(A, B) if x and not y)
    co = sum(1 for x, y in zip(A, B) if y and not x)
    n = bo + co
    if not n: return 1.0
    k = min(bo, co)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) * 0.5 ** n)


def wilcoxon(a, b):
    try: return stats.wilcoxon(sc(a), sc(b)).pvalue
    except Exception: return float("nan")


out = {}

# --- per-model TGC + mean score ---
out["per_model"] = {k: {"tgc": 100 * mean(su(k)), "score": mean(sc(k))} for k in F}

# --- key contrasts: binary vs continuous ---
contrasts = [("joint", "RAM+"), ("joint", "TIES"), ("RAM+", "TIES"),
             ("diff-2", "RAM+"), ("diff-2", "TIES"), ("joint", "diff-2")]
out["contrasts"] = [{"a": a, "b": b,
                     "dscore": mean(x - y for x, y in zip(sc(a), sc(b))),
                     "mcnemar_p": mcnemar(a, b), "wilcoxon_p": wilcoxon(a, b)} for a, b in contrasts]

# --- super-additivity (emergence) ---
d1, d2, jt = sc("diff-1"), sc("diff-2"), sc("joint")
pmax = [max(a, b) for a, b in zip(d1, d2)]
pmin = [min(a, b) for a, b in zip(d1, d2)]
pavg = [(a + b) / 2 for a, b in zip(d1, d2)]
m = 0.10
out["emergence"] = {
    "mean_joint": mean(jt), "mean_pertask_max": mean(pmax), "mean_pertask_avg": mean(pavg),
    "joint_above_max": sum(1 for j, x in zip(jt, pmax) if j > x + m),
    "joint_below_min": sum(1 for j, x in zip(jt, pmin) if j < x - m),
    "joint_within": sum(1 for j, lo, hi in zip(jt, pmin, pmax) if lo - m <= j <= hi + m),
    "n": len(KS), "margin": m,
}

# --- averaging test ---
out["averaging"] = {}
for mk in ["TIES", "RAM+"]:
    mv = sc(mk)
    out["averaging"][mk] = {"mad_avg": mean(abs(a - b) for a, b in zip(mv, pavg)),
                            "mad_max": mean(abs(a - b) for a, b in zip(mv, pmax))}

with open(os.path.join(HERE, "mechanism.json"), "w") as f:
    json.dump(out, f, indent=1)

# --- print ---
print("=== per-model: TGC vs mean partial-credit score ===")
for k in ["diff-1", "diff-2", "joint", "TIES", "RAM+"]:
    print(f"  {k:8s} TGC={out['per_model'][k]['tgc']:5.1f}%   score={out['per_model'][k]['score']:.3f}")
print("\n=== metric reversal: diff-2 vs merges ===")
print(f"  TGC:   diff-2={out['per_model']['diff-2']['tgc']:.1f}  TIES={out['per_model']['TIES']['tgc']:.1f}  RAM+={out['per_model']['RAM+']['tgc']:.1f}  (merge > diff-2)")
print(f"  score: diff-2={out['per_model']['diff-2']['score']:.3f}  TIES={out['per_model']['TIES']['score']:.3f}  RAM+={out['per_model']['RAM+']['score']:.3f}  (merge < diff-2)")
print("\n=== contrasts: McNemar(binary) vs Wilcoxon(continuous) ===")
for c in out["contrasts"]:
    print(f"  {c['a']:7s} vs {c['b']:7s}  Δscore={c['dscore']:+.4f}  McNemar={c['mcnemar_p']:.3f}  Wilcoxon={c['wilcoxon_p']:.3f}")
e = out["emergence"]
print(f"\n=== emergence: mean(joint)={e['mean_joint']:.3f} vs mean(per-task max)={e['mean_pertask_max']:.3f} -> {'EXCEEDS' if e['mean_joint']>e['mean_pertask_max'] else 'BELOW best-of (NO emergence)'}")
print(f"  joint within [min,max]: {e['joint_within']}/{e['n']}   above max: {e['joint_above_max']}   below min: {e['joint_below_min']}")
for mk, v in out["averaging"].items():
    print(f"  {mk}: |merge-avg|={v['mad_avg']:.3f} < |merge-max|={v['mad_max']:.3f} -> merge tracks AVERAGE")
print("\nwrote paper/mechanism.json")
