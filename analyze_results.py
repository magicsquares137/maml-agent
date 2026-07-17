#!/usr/bin/env python3
"""Per-difficulty TGC/SGC breakdown for eval result JSONs.
Reads each task's difficulty from $APPWORLD_ROOT/data/tasks/<tid>/ground_truth/metadata.json.
Usage: python analyze_results.py results_<model>.json [results_<model2>.json ...]
"""
import json, os, sys
from collections import defaultdict

AR = os.environ.get("APPWORLD_ROOT", "/home/smcclendon/Documents/github/appworld/appworld-rl")

def difficulty(tid):
    try:
        return json.load(open(f"{AR}/data/tasks/{tid}/ground_truth/metadata.json")).get("difficulty")
    except Exception:
        return None

def scenario(tid):
    return "_".join(tid.split("_")[:-1])

for path in sys.argv[1:]:
    try:
        d = json.load(open(path))
    except Exception as e:
        print(f"{path}: (not available: {e})"); continue
    r = d["results"]
    n = len(r); s = sum(1 for x in r if x["success"])
    # SGC (aggregate): scenario counts only if all its tasks succeed
    sc = defaultdict(list)
    for x in r: sc[scenario(x["task_id"])].append(x["success"])
    sgc = sum(1 for v in sc.values() if all(v))
    print(f"\n=== {path} ===")
    print(f"  AGGREGATE: TGC {s}/{n} = {100*s/n:.1f}%  |  SGC {sgc}/{len(sc)} = {100*sgc/len(sc):.1f}%  |  avg_score {d['avg_score']:.3f}  |  errored {d.get('errored',0)}")
    bydiff = defaultdict(list)
    for x in r: bydiff[difficulty(x["task_id"])].append(x["success"])
    for diff in sorted(bydiff, key=lambda z: (z is None, z)):
        v = bydiff[diff]; sc2 = sum(v)
        print(f"    difficulty_{diff}: TGC {sc2}/{len(v)} = {100*sc2/len(v):.1f}%")
