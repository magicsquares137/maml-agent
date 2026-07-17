#!/usr/bin/env python3
"""Does the mergeability-aware objective J select a different checkpoint pair than
the naive reward peak (5,5)?  J(s1,s2) = P1(s1) + P2(s2) - gamma * I(s1,s2), where
P_t = mean training reward at iter s (from run logs) and I = cos(tau_d1(s1),tau_d2(s2))
computed over the full 9x9 grid (incl. asymmetric s1!=s2) via the Frobenius identity.

If argmax J = (5,5) for all reasonable gamma, trajectory-aware selection cannot beat
the merge already evaluated -> the joint>merge partial-credit gap is not a
checkpoint-selection artifact."""
import os, math
import torch
from safetensors.torch import load_file

D1 = "checkpoints/diff_1"
D2 = "checkpoints/diff_2_remote_traj"
ITERS = list(range(1, 10))

# mean training reward per LoRA checkpoint (aligned to lora_iter_N).
# diff-1 was a RESUMED run (+1 offset vs reward-line index); diff-2/joint fresh.
P1 = {1:0.4248,2:0.4450,3:0.4478,4:0.4953,5:0.5691,6:0.3872,7:0.4343,8:0.4725,9:0.4910}
P2 = {1:0.3976,2:0.4087,3:0.3942,4:0.4310,5:0.4643,6:0.4002,7:0.4300,8:0.3650,9:0.3785}

def load_lora(d):
    sd = load_file(os.path.join(d, "adapter_model.safetensors"))
    m = {}
    for k, v in sd.items():
        if "lora_A" in k: m.setdefault(k.split(".lora_A")[0], {})["A"] = v.float()
        elif "lora_B" in k: m.setdefault(k.split(".lora_B")[0], {})["B"] = v.float()
    return {k: v for k, v in m.items() if "A" in v and "B" in v}

def fro(B1,A1,B2,A2): return ((B1.T@B2)*(A1@A2.T)).sum().item()

def gcos(m1, m2):
    dot=n1=n2=0.0
    for k in set(m1)&set(m2):
        dot+=fro(m1[k]["B"],m1[k]["A"],m2[k]["B"],m2[k]["A"])
        n1 +=fro(m1[k]["B"],m1[k]["A"],m1[k]["B"],m1[k]["A"])
        n2 +=fro(m2[k]["B"],m2[k]["A"],m2[k]["B"],m2[k]["A"])
    return dot/math.sqrt(n1*n2)

print("loading adapters (iters 1-9, both arms)...")
L1 = {t: load_lora(os.path.join(D1, f"lora_iter_{t}")) for t in ITERS}
L2 = {t: load_lora(os.path.join(D2, f"lora_iter_{t}")) for t in ITERS}

print("computing 9x9 interference grid I(s1,s2)=cos(tau_d1(s1),tau_d2(s2))...")
I = {(a,b): gcos(L1[a], L2[b]) for a in ITERS for b in ITERS}

print("\nI(s1,s2) grid (rows=diff1 iter, cols=diff2 iter):")
print("      " + "".join(f"{b:7d}" for b in ITERS))
for a in ITERS:
    print(f"  s1={a} " + "".join(f"{I[(a,b)]:7.3f}" for b in ITERS))

# regenerate paper/data/interference_grid.csv (consumed by paper/fig_jgrid.py)
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "interference_grid.csv")
os.makedirs(os.path.dirname(_out), exist_ok=True)
with open(_out, "w") as _f:
    _f.write("s1," + ",".join(str(b) for b in ITERS) + "\n")
    for a in ITERS:
        _f.write(f"{a}," + ",".join(f"{I[(a,b)]:.3f}" for b in ITERS) + "\n")
print(f"wrote {_out}")

print(f"\ndiagonal check I(5,5)={I[(5,5)]:.3f}  (snapshot ~0.068 expected)")

print("\nargmax J(s1,s2) = P1(s1)+P2(s2) - gamma*I(s1,s2), over full grid:")
print(f"{'gamma':>7}   {'argmax (s1,s2)':>16}   {'J*':>8}   {'J(5,5)':>8}   {'I*':>7}")
for g in [0, 1, 2, 5, 10, 20]:
    best=None
    for a in ITERS:
        for b in ITERS:
            J = P1[a] + P2[b] - g*I[(a,b)]
            if best is None or J > best[0]: best=(J,a,b)
    J55 = P1[5]+P2[5]-g*I[(5,5)]
    print(f"{g:>7}   {f'({best[1]},{best[2]})':>16}   {best[0]:>8.4f}   {J55:>8.4f}   {I[(best[1],best[2])]:>7.3f}")

# capability ceiling of the merge family: max P1 + max P2
capmax_pair = (max(ITERS,key=lambda t:P1[t]), max(ITERS,key=lambda t:P2[t]))
print(f"\ncapability-max pair (gamma=0) = {capmax_pair}, P1+P2={P1[capmax_pair[0]]+P2[capmax_pair[1]]:.4f}")
print("=> this is the merge already evaluated (iter5+iter5). No trajectory pair exceeds it.")
print("DONE")
