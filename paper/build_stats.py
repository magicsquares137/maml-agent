#!/usr/bin/env python3
"""Build all paper statistics from the per-task eval result JSONs.

Outputs:
  - Prints Table 1 (per-difficulty TGC + SGC, with bootstrap 95% CI on agg TGC)
  - Prints Table 2 (pairwise McNemar exact p-values, paired on task_id)
  - Writes paper/stats.json  (consumed by fig_forest.py)

Paired tests use the intersection of task_ids across the two models.
Base model has NO per-task file (numbers are from the official external report),
so base contrasts are done UNPAIRED (two-proportion, Fisher-exact) and flagged.
"""
import json, os, sys
from collections import defaultdict
from math import comb, sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AR = os.environ.get("APPWORLD_ROOT", "/home/smcclendon/Documents/github/appworld/appworld-rl")

# model label -> results file (base is external, no file)
FILES = {
    "diff-1 spec": "results_diff_1_iter5.json",
    "diff-2 spec": "results_diff_2_iter5.json",
    "joint":       "results_joint_iter9.json",
    "TIES":        "results_merged_ties.json",
    "RAM+":        "results_merged_ram.json",
}
# base from the official Text Evaluation Report (aggregate + per-difficulty TGC%)
BASE = {"agg": (11, 168), "d": {1: 15.8, 2: 2.1, 3: 1.6}, "agg_pct": 6.6}
N_BOOT = 20000
SEED = 100  # fixed; Math.random-free reproducibility


def difficulty(tid):
    try:
        return json.load(open(f"{AR}/data/tasks/{tid}/ground_truth/metadata.json")).get("difficulty")
    except Exception:
        return None


def scenario(tid):
    return "_".join(tid.split("_")[:-1])


def load(label):
    d = json.load(open(os.path.join(ROOT, FILES[label])))
    return {x["task_id"]: bool(x["success"]) for x in d["results"]}


def tgc(succ):
    n = len(succ); s = sum(succ.values())
    return s, n, 100.0 * s / n


def sgc(succ):
    sc = defaultdict(list)
    for tid, ok in succ.items():
        sc[scenario(tid)].append(ok)
    g = sum(1 for v in sc.values() if all(v))
    return g, len(sc), 100.0 * g / len(sc)


def per_difficulty(succ):
    by = defaultdict(list)
    for tid, ok in succ.items():
        by[difficulty(tid)].append(ok)
    return {k: 100.0 * sum(v) / len(v) for k, v in by.items()}


# ---- deterministic PRNG (no Math.random / numpy default seed drift) ----
class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF
    def randint(self, n):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s % n


def bootstrap_ci_tgc(succ, n_boot=N_BOOT, seed=SEED):
    vals = list(succ.values())
    n = len(vals)
    rng = LCG(seed)
    means = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += vals[rng.randint(n)]
        means.append(100.0 * s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return lo, hi


def bootstrap_ci_delta(a, b, n_boot=N_BOOT, seed=SEED):
    """Paired ΔTGC (a - b) bootstrap CI over shared task_ids."""
    keys = sorted(set(a) & set(b))
    da = [a[k] for k in keys]
    db = [b[k] for k in keys]
    n = len(keys)
    rng = LCG(seed)
    diffs = []
    for _ in range(n_boot):
        sa = sb = 0
        for _ in range(n):
            i = rng.randint(n)
            sa += da[i]; sb += db[i]
        diffs.append(100.0 * (sa - sb) / n)
    diffs.sort()
    point = 100.0 * (sum(da) - sum(db)) / n
    return point, diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def mcnemar_exact(a, b):
    """Two-sided exact McNemar on shared task_ids. Returns (b_only, c_only, p)."""
    keys = set(a) & set(b)
    b_only = sum(1 for k in keys if a[k] and not b[k])   # a wins
    c_only = sum(1 for k in keys if b[k] and not a[k])   # b wins
    n = b_only + c_only
    if n == 0:
        return b_only, c_only, 1.0
    k = min(b_only, c_only)
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    p = min(1.0, 2.0 * tail)
    return b_only, c_only, p


def fisher_2prop(s1, n1, s2, n2):
    """Two-sided Fisher exact on a 2x2 (unpaired). Returns p."""
    a, b_ = s1, n1 - s1
    c, d = s2, n2 - s2
    def logfact(x):
        from math import lgamma
        return lgamma(x + 1)
    def lp(a, b_, c, d):
        r1, r2 = a + b_, c + d
        c1, c2 = a + c, b_ + d
        tot = a + b_ + c + d
        return (logfact(r1) + logfact(r2) + logfact(c1) + logfact(c2)
                - logfact(tot) - logfact(a) - logfact(b_) - logfact(c) - logfact(d))
    from math import exp
    p0 = lp(a, b_, c, d)
    r1 = a + b_
    c1 = a + c
    tot = a + b_ + c + d
    p = 0.0
    for aa in range(max(0, c1 - (tot - r1)), min(r1, c1) + 1):
        bb = r1 - aa; cc = c1 - aa; dd = (tot - r1) - cc
        if bb < 0 or cc < 0 or dd < 0:
            continue
        lpp = lp(aa, bb, cc, dd)
        if lpp <= p0 + 1e-9:
            p += exp(lpp)
    return min(1.0, p)


def main():
    succ = {lab: load(lab) for lab in FILES}

    # ---------- Table 1 ----------
    print("\n=== TABLE 1: per-difficulty TGC (%) + agg TGC [95% CI] + SGC ===")
    print(f"{'model':14s} {'d1':>6} {'d2':>6} {'d3':>6} {'AggTGC':>8} {'95% CI':>16} {'SGC':>6}")
    print(f"{'base':14s} {BASE['d'][1]:6.1f} {BASE['d'][2]:6.1f} {BASE['d'][3]:6.1f} "
          f"{BASE['agg_pct']:8.1f} {'(external)':>16} {0.0:6.1f}")
    table1 = {"base": {"d": BASE["d"], "agg": BASE["agg_pct"], "sgc": 0.0, "external": True}}
    for lab in FILES:
        s, n, agg = tgc(succ[lab])
        pd = per_difficulty(succ[lab])
        g, gn, sg = sgc(succ[lab])
        lo, hi = bootstrap_ci_tgc(succ[lab])
        print(f"{lab:14s} {pd.get(1,0):6.1f} {pd.get(2,0):6.1f} {pd.get(3,0):6.1f} "
              f"{agg:8.1f} {f'[{lo:.1f}, {hi:.1f}]':>16} {sg:6.1f}")
        table1[lab] = {"d": pd, "agg": agg, "ci": [lo, hi], "sgc": sg,
                       "tgc_count": [s, n]}

    # ---------- Table 2 ----------
    print("\n=== TABLE 2: pairwise significance ===")
    print(f"{'contrast':26s} {'A-only':>7} {'B-only':>7} {'test':>10} {'p':>8}")
    pairs = [("joint", "RAM+"), ("RAM+", "TIES"), ("joint", "TIES"),
             ("diff-1 spec", "diff-2 spec"), ("joint", "diff-2 spec")]
    table2 = []
    for a, b in pairs:
        bo, co, p = mcnemar_exact(succ[a], succ[b])
        print(f"{f'{a} vs {b}':26s} {bo:7d} {co:7d} {'McNemar':>10} {p:8.3f}")
        table2.append({"a": a, "b": b, "a_only": bo, "b_only": co, "test": "mcnemar", "p": p})
    # base contrasts: UNPAIRED Fisher (no per-task base file)
    for lab in ["RAM+", "joint", "diff-2 spec"]:
        s, n, _ = tgc(succ[lab])
        p = fisher_2prop(s, n, BASE["agg"][0], BASE["agg"][1])
        print(f"{f'{lab} vs base':26s} {s:7d} {BASE['agg'][0]:7d} {'Fisher*':>10} {p:8.4f}")
        table2.append({"a": lab, "b": "base", "a_succ": [s, n],
                       "b_succ": list(BASE["agg"]), "test": "fisher_unpaired", "p": p})
    print("  * base has no per-task file (external report) -> unpaired Fisher exact")

    # ---------- forest data (paired ΔTGC CIs) ----------
    print("\n=== FOREST: paired ΔTGC (%) with 95% CI ===")
    forest = []
    contrasts = [("joint", "RAM+"), ("RAM+", "TIES"), ("joint", "TIES"),
                 ("joint", "diff-1 spec"), ("joint", "diff-2 spec")]
    for a, b in contrasts:
        pt, lo, hi = bootstrap_ci_delta(succ[a], succ[b])
        cross0 = lo <= 0 <= hi
        print(f"{f'{a} - {b}':26s} {pt:+6.1f}  [{lo:+.1f}, {hi:+.1f}]  {'(crosses 0)' if cross0 else '** excludes 0 **'}")
        forest.append({"label": f"{a} − {b}", "point": pt, "lo": lo, "hi": hi, "crosses0": cross0})
    # RL - base as unpaired aggregate delta (approx CI via normal on unpaired props)
    for lab in ["RAM+", "joint"]:
        s, n, agg = tgc(succ[lab])
        p1 = s / n; p0 = BASE["agg"][0] / BASE["agg"][1]
        se = sqrt(p1 * (1 - p1) / n + p0 * (1 - p0) / BASE["agg"][1])
        pt = 100 * (p1 - p0)
        lo, hi = pt - 196 * se, pt + 196 * se
        print(f"{f'{lab} - base (unpaired)':26s} {pt:+6.1f}  [{lo:+.1f}, {hi:+.1f}]")
        forest.append({"label": f"{lab} − base*", "point": pt, "lo": lo, "hi": hi,
                       "crosses0": lo <= 0 <= hi, "unpaired": True})

    out = {"table1": table1, "table2": table2, "forest": forest,
           "n_boot": N_BOOT, "seed": SEED}
    with open(os.path.join(HERE, "stats.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {os.path.join(HERE, 'stats.json')}")


if __name__ == "__main__":
    main()
