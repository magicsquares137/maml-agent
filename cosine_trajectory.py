import os, math
import torch
from safetensors.torch import load_file

D1 = "checkpoints/diff_1"
D2 = "checkpoints/diff_2_remote_traj"

def load_lora(adapter_dir):
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    mods = {}
    for k, v in sd.items():
        if "lora_A" in k:
            base = k.split(".lora_A")[0]
            mods.setdefault(base, {})["A"] = v.float()
        elif "lora_B" in k:
            base = k.split(".lora_B")[0]
            mods.setdefault(base, {})["B"] = v.float()
    return mods

def stats(m1, m2):
    dot = n1 = n2 = 0.0
    sign_agree = sign_tot = 0
    for k in set(m1) & set(m2):
        if not ({"A","B"} <= set(m1[k])) or not ({"A","B"} <= set(m2[k])): continue
        dW1 = (m1[k]["B"] @ m1[k]["A"]).flatten()
        dW2 = (m2[k]["B"] @ m2[k]["A"]).flatten()
        dot += torch.dot(dW1, dW2).item(); n1 += torch.dot(dW1, dW1).item(); n2 += torch.dot(dW2, dW2).item()
        thr = 1e-5
        bm = (dW1.abs() > thr) & (dW2.abs() > thr)
        if bool(bm.any()):
            sign_agree += int((torch.sign(dW1[bm]) == torch.sign(dW2[bm])).sum()); sign_tot += int(bm.sum())
    cos = dot / math.sqrt(n1 * n2)
    sa = 100 * sign_agree / sign_tot if sign_tot else float("nan")
    return cos, math.sqrt(n1), math.sqrt(n2), sa

print("iter   cosine    |tau_d1|   |tau_d2|   sign_agree%")
rows = []
for t in range(1, 11):
    p1 = os.path.join(D1, f"lora_iter_{t}"); p2 = os.path.join(D2, f"lora_iter_{t}")
    if not (os.path.isdir(p1) and os.path.isdir(p2)): continue
    c, nn1, nn2, sa = stats(load_lora(p1), load_lora(p2))
    print(f"{t:4d}   {c:+.4f}   {nn1:8.3f}   {nn2:8.3f}   {sa:6.1f}")
    rows.append((t, c, nn1, nn2, sa))

# regenerate paper/data/trajectory.csv (consumed by paper/fig_trajectory.py)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "data", "trajectory.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    f.write("iter,cosine,tau_d1,tau_d2,sign_agree_pct\n")
    for t, c, nn1, nn2, sa in rows:
        f.write(f"{t},{c:.4f},{nn1:.3f},{nn2:.3f},{sa:.1f}\n")
print(f"wrote {out}")
print("DONE")
