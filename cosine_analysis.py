import os, re, math, statistics
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM

BASE = "Qwen/Qwen3-8B"
M1 = "/media/smcclendon/Expansion/maml_merged/diff_1_iter5"
M2 = "/media/smcclendon/Expansion/maml_merged/diff_2_iter5"
THR = 1e-5

def load_sd(p):
    m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16)
    d = {k: v.clone() for k, v in m.state_dict().items()}
    del m
    return d

print("loading base...", flush=True);  b  = load_sd(BASE)
print("loading diff1...", flush=True); d1 = load_sd(M1)
print("loading diff2...", flush=True); d2 = load_sd(M2)

def modtype(k):
    m = re.search(r'(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|embed_tokens|lm_head|input_layernorm|post_attention_layernorm|norm)', k)
    return m.group(1) if m else 'other'

gdot = gn1 = gn2 = 0.0
changed1 = changed2 = both = 0
sign_agree = sign_total = 0
per = []
bymod = defaultdict(lambda: [0.0, 0.0, 0.0])

for k in b:
    if k not in d1 or k not in d2: continue
    t1 = (d1[k].float() - b[k].float()).flatten()
    t2 = (d2[k].float() - b[k].float()).flatten()
    dot = torch.dot(t1, t2).item(); n1 = torch.dot(t1, t1).item(); n2 = torch.dot(t2, t2).item()
    gdot += dot; gn1 += n1; gn2 += n2
    mt = modtype(k); bymod[mt][0] += dot; bymod[mt][1] += n1; bymod[mt][2] += n2
    if n1 > 0 and n2 > 0: per.append(dot / math.sqrt(n1 * n2))
    m1 = t1.abs() > THR; m2 = t2.abs() > THR; bm = m1 & m2
    changed1 += int(m1.sum()); changed2 += int(m2.sum()); both += int(bm.sum())
    if bool(bm.any()):
        sign_agree += int((torch.sign(t1[bm]) == torch.sign(t2[bm])).sum()); sign_total += int(bm.sum())

gcos = gdot / math.sqrt(gn1 * gn2)
print(f"\n=== GLOBAL cosine(tau_diff1, tau_diff2) = {gcos:.4f} ===")
print(f"changed(diff1)={changed1}  changed(diff2)={changed2}  overlap={both}")
print(f"sign agreement within overlap: {sign_agree}/{sign_total} = {100*sign_agree/sign_total:.1f}%")
cv = sorted(per)
print(f"per-tensor cosine: mean={statistics.mean(cv):.3f}  median={statistics.median(cv):.3f}  min={min(cv):.3f}  max={max(cv):.3f}  (n={len(cv)})")
print("\nper module-type (aggregated cosine):")
projmods = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
csv_rows = []
for mt, (dt, n1, n2) in sorted(bymod.items()):
    if n1 > 0 and n2 > 0:
        cval = dt / math.sqrt(n1 * n2)
        print(f"  {mt:24s}: cos={cval:+.3f}")
        if mt in projmods:
            csv_rows.append((mt, cval))

# regenerate paper/data/module_cosine.csv (consumed by paper/fig_module_cosine.py)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "data", "module_cosine.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    f.write("module,cosine\n")
    for mt, cval in sorted(csv_rows, key=lambda r: -r[1]):
        f.write(f"{mt},{cval:.3f}\n")
print(f"wrote {out}")
print("\nDONE")
