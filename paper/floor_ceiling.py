#!/usr/bin/env python3
"""Floor + ceiling calibration for the cross-specialist cosine measurement.

Brackets the measurement to refute "0.06 is a LoRA construction artifact":
  FLOOR (random)  : cosine of two INDEPENDENT random rank-r adapters       -> ~0, but
                    ultra-tight variance (random low-rank overstates eff. dim);
                    do NOT quote sigma against this null.
  FLOOR (spectrum): random directions carrying each trained tau's SINGULAR
                    VALUE profile -> honest null variance (this is the one to cite).
  CEILING         : same-run tau at two iters (iter5 vs iter9)              -> ~0.8
  MEASURED        : cross-specialist cosine(tau_d1, tau_d2)                 -> 0.07-0.10

Frobenius identity <B1 A1, B2 A2>_F = sum((B1^T B2)*(A1 A2^T)) keeps all work r x r.
Singular values of dW=B@A via s = sqrt(eig((A A^T)(B^T B))) -- exact, r x r. No GPU.
Writes paper/calibration.json for fig_calibration.py.
"""
import os, math, json
import torch
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
D1 = "checkpoints/diff_1"
D2 = "checkpoints/diff_2_remote_traj"
N_RANDOM = 500
N_SPECTRUM = 200
SEED = 100


def load_lora(adapter_dir):
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    mods = {}
    for k, v in sd.items():
        if "lora_A" in k:
            mods.setdefault(k.split(".lora_A")[0], {})["A"] = v.float()
        elif "lora_B" in k:
            mods.setdefault(k.split(".lora_B")[0], {})["B"] = v.float()
    return {k: v for k, v in mods.items() if "A" in v and "B" in v}


def fro_dot(B1, A1, B2, A2):
    return ((B1.T @ B2) * (A1 @ A2.T)).sum().item()


def global_cos(m1, m2):
    dot = n1 = n2 = 0.0
    for k in set(m1) & set(m2):
        B1, A1 = m1[k]["B"], m1[k]["A"]
        B2, A2 = m2[k]["B"], m2[k]["A"]
        dot += fro_dot(B1, A1, B2, A2)
        n1 += fro_dot(B1, A1, B1, A1)
        n2 += fro_dot(B2, A2, B2, A2)
    return dot / math.sqrt(n1 * n2)


def singular_values(B, A):
    # nonzero singular values of dW = B@A  ==  sqrt(eig((A A^T)(B^T B)))
    M = A @ A.T
    K = B.T @ B
    ev = torch.linalg.eigvals(M @ K).real.clamp(min=0.0)
    return torch.sqrt(ev)


def orthonormal(n, r, gen):
    Q, _ = torch.linalg.qr(torch.randn(n, r, generator=gen))
    return Q  # n x r, orthonormal columns


def rand_adapter(shapes, gen):
    return {k: {"B": torch.randn(*bs, generator=gen),
                "A": torch.randn(*as_, generator=gen)}
            for k, (bs, as_) in shapes.items()}


def spectrum_adapter(spec, gen):
    # random directions U,V carrying trained singular values s -> B=U diag(s), A=V^T
    out = {}
    for k, (s, out_dim, in_dim) in spec.items():
        r = s.shape[0]
        U = orthonormal(out_dim, r, gen)
        V = orthonormal(in_dim, r, gen)
        out[k] = {"B": U * s.unsqueeze(0), "A": V.T}
    return out


def stats(vals):
    t = torch.tensor(vals)
    return {"mean": t.mean().item(), "std": t.std().item(),
            "lo": t.quantile(0.025).item(), "hi": t.quantile(0.975).item(),
            "absmax": t.abs().max().item()}


def main():
    print("loading real adapters (iter1/5/9)...")
    d1_1 = load_lora(os.path.join(D1, "lora_iter_1"))
    d1_5 = load_lora(os.path.join(D1, "lora_iter_5"))
    d1_9 = load_lora(os.path.join(D1, "lora_iter_9"))
    d2_5 = load_lora(os.path.join(D2, "lora_iter_5"))
    d2_9 = load_lora(os.path.join(D2, "lora_iter_9"))

    meas5, meas9 = global_cos(d1_5, d2_5), global_cos(d1_9, d2_9)
    ceil_d1 = global_cos(d1_5, d1_9)
    ceil_d2 = global_cos(d2_5, d2_9)
    early = global_cos(d1_1, d1_9)
    print("\n=== MEASURED (cross-specialist) ===")
    print(f"  iter5 = {meas5:+.4f}   iter9 = {meas9:+.4f}")
    print("=== CEILING (same-run self-alignment) ===")
    print(f"  d1@5-vs-@9 = {ceil_d1:+.4f}   d2@5-vs-@9 = {ceil_d2:+.4f}   d1@1-vs-@9 = {early:+.4f}")

    shapes = {k: (v["B"].shape, v["A"].shape) for k, v in d1_9.items()}
    gen = torch.Generator().manual_seed(SEED)

    print(f"\n=== FLOOR (random rank-16, {N_RANDOM} draws) ===")
    rnd = [global_cos(rand_adapter(shapes, gen), rand_adapter(shapes, gen)) for _ in range(N_RANDOM)]
    sr = stats(rnd)
    print(f"  mean={sr['mean']:+.6f}  std={sr['std']:.6f}  95%=[{sr['lo']:+.6f},{sr['hi']:+.6f}]  |max|={sr['absmax']:.6f}")
    print("  (variance pathologically tight -> do NOT quote sigma against this)")

    # spectrum for each specialist, per module
    print(f"\n=== FLOOR (spectrum-matched, {N_SPECTRUM} draws) -- the honest null ===")
    spec1 = {k: (singular_values(v["B"], v["A"]), v["B"].shape[0], v["A"].shape[1]) for k, v in d1_9.items()}
    spec2 = {k: (singular_values(v["B"], v["A"]), v["B"].shape[0], v["A"].shape[1]) for k, v in d2_9.items()}
    spm = [global_cos(spectrum_adapter(spec1, gen), spectrum_adapter(spec2, gen)) for _ in range(N_SPECTRUM)]
    ss = stats(spm)
    print(f"  mean={ss['mean']:+.5f}  std={ss['std']:.5f}  95%=[{ss['lo']:+.5f},{ss['hi']:+.5f}]  |max|={ss['absmax']:.5f}")
    z = (meas9 - ss["mean"]) / ss["std"] if ss["std"] else float("inf")
    z1 = (0.0015 - ss["mean"]) / ss["std"] if ss["std"] else float("inf")
    print(f"  => cross-spec 0.10 is {z:.1f} sigma above the spectrum-matched null")
    print(f"     iter-1 (0.0015) is {z1:.1f} sigma from null  (>~inside null => 'starts at the floor' is honest)")

    out = {
        "measured": {"iter5": meas5, "iter9": meas9},
        "ceiling": {"d1_5v9": ceil_d1, "d2_5v9": ceil_d2, "d1_1v9": early},
        "floor_random": sr,
        "floor_spectrum": ss,
        "iter1_crossspec": 0.0015,
        "n_random": N_RANDOM, "n_spectrum": N_SPECTRUM, "seed": SEED,
    }
    with open(os.path.join(HERE, "calibration.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {os.path.join(HERE, 'calibration.json')}\nDONE")


if __name__ == "__main__":
    main()
