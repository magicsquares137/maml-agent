"""
Standalone PPO-LOOP training step (the torch/transformers/peft half).

Run as a SUBPROCESS from the orchestrator (main_alt_H.py). This process must
NEVER import appworld — appworld's native stack and torch crash (segfault in GC)
when co-resident in one process. Keeping training isolated here is what makes
the whole pipeline runnable.

Invoke with the vllm_env python, e.g.:
  vllm_env/bin/python -m ppo_baseline.train_step \
      --base-model Qwen/Qwen3-8B \
      --rollouts /tmp/rollouts.pkl \
      --lora-in   ./checkpoints/diff_1/lora_iter_2   (or empty for fresh) \
      --lora-out  ./checkpoints/diff_1/lora_iter_3 \
      --optimizer-in  ./checkpoints/diff_1/lora_iter_2/optimizer.pt  (or empty) \
      --optimizer-out ./checkpoints/diff_1/lora_iter_3/optimizer.pt \
      --metrics-out   ./checkpoints/diff_1/lora_iter_3/train_metrics.json \
      --epsilon 0.2 --lr 5e-5 --n-epochs 3 --batch-size 3 \
      --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
      --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

Rollouts pickle format: list of dicts
  {"advantage": float,
   "messages": [ {"tokenized_input": [int, ...],
                  "log_probs": [(token_str, old_logprob, token_id), ...]}, ... ]}
"""
import argparse
import json
import os
import pickle
import random
from pathlib import Path

# Reduce allocator fragmentation (helps the long-sequence forward fit 16GB).
# Must be set before torch initializes the CUDA allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel


def build_policy_model(args):
    """Load frozen base + (fresh or existing) LoRA, ready for training."""
    print("\n🏋️  Initializing policy model for training...", flush=True)

    from_pretrained_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # CRITICAL: device_map="auto" greedily fills GPU 0 to capacity (~15GB of
    # weights) and barely uses GPU 1, leaving GPU 0 with NO room for activations
    # -> every training forward OOMs. Cap per-GPU weight budget so the ~16GB of
    # weights split evenly and each card keeps headroom for activations/logits.
    if not args.load_4bit and args.max_memory_per_gpu:
        n = torch.cuda.device_count()
        from_pretrained_kwargs["max_memory"] = {
            i: args.max_memory_per_gpu for i in range(n)
        }
        print(f"   max_memory per GPU: {args.max_memory_per_gpu} across {n} GPUs", flush=True)

    if args.load_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError(f"--load-4bit requires bitsandbytes installed: {e}")
        from_pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("   Loading base in 4-bit (QLoRA)", flush=True)
    else:
        print("   Loading base in bf16 (device_map=auto shards across visible GPUs)", flush=True)

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **from_pretrained_kwargs)

    # Freeze base
    for param in base_model.parameters():
        param.requires_grad = False

    base_model.config.use_cache = False
    if args.load_4bit:
        from peft import prepare_model_for_kbit_training
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    lora_in = args.lora_in.strip() if args.lora_in else ""
    if lora_in and Path(lora_in).exists():
        print(f"   Loading existing LoRA from: {lora_in}", flush=True)
        policy_model = PeftModel.from_pretrained(base_model, lora_in, is_trainable=True)
    else:
        print("   Creating fresh LoRA", flush=True)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[m for m in args.target_modules.split(",") if m],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(base_model, lora_config)

    # Enable gradient checkpointing on the FINAL (PEFT) model with
    # use_reentrant=False. Calling it on the base model before the PEFT wrap (and
    # with accelerate device-map hooks present) silently no-ops, so the forward
    # stored every layer's activations -> +8GB for a short sequence -> OOM.
    # enable_input_require_grads is also required (frozen base + checkpointing
    # otherwise zeros the LoRA grads).
    if not args.load_4bit:
        policy_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    policy_model.enable_input_require_grads()
    # HF gradient checkpointing only engages when the module is in TRAINING mode;
    # in eval mode it stores every activation -> OOM. Force train mode.
    policy_model.train()

    gc_on = getattr(policy_model.base_model.model.model, "gradient_checkpointing", None)
    print(f"   gradient_checkpointing={gc_on}, training={policy_model.training}", flush=True)

    return policy_model


def message_token_count(msg):
    pi = msg.get("tokenized_input")
    od = msg.get("log_probs")
    return len(od) if (pi and od) else 0


def message_loss(policy_model, msg, advantage, epsilon, scale, max_prompt_tokens):
    """Forward ONE assistant message and return its scaled scalar PPO loss
    (with grad). Memory-bounded two ways: (1) logits_to_keep -> logits only for
    the OUTPUT positions instead of the full prompt x 151k vocab; (2) prompt
    truncated to its last `max_prompt_tokens` tokens so grad-checkpoint
    activations stay bounded even when the full ReAct prompt is ~30k tokens.
    We only train on output tokens (the prompt is left-context), and output
    positions are indexed from the end so truncating the front doesn't shift them.
    """
    prompt_token_ids = msg["tokenized_input"]
    if max_prompt_tokens and len(prompt_token_ids) > max_prompt_tokens:
        prompt_token_ids = prompt_token_ids[-max_prompt_tokens:]
    output_token_data = msg["log_probs"]  # [(token_str, old_logprob, token_id), ...]
    output_token_ids = [tid for _, _, tid in output_token_data]
    O = len(output_token_ids)

    full_ids = torch.tensor(
        [list(prompt_token_ids) + output_token_ids],
        device=policy_model.get_input_embeddings().weight.device,
    )

    # logits_to_keep=O+1 -> logits only for the last O+1 positions. With prompt
    # length P, kept index k maps to original position (P-1)+k, so output token j
    # (predicted by logits at position P+j-1) is exactly kept index j.
    outputs = policy_model(full_ids, logits_to_keep=O + 1)
    logits = outputs.logits[0]                       # [O+1, vocab]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    dev = log_probs.device
    adv_t = torch.tensor(advantage, device=dev, dtype=torch.float32)

    token_losses = []
    for j, (_tok, old_logprob, token_id) in enumerate(output_token_data):
        new_logprob = log_probs[j, token_id]
        old_logprob_t = torch.tensor(old_logprob, device=dev, dtype=torch.float32)
        ratio = torch.exp(new_logprob - old_logprob_t)
        clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
        token_losses.append(-torch.min(ratio * adv_t, clipped * adv_t))

    msg_loss = torch.stack(token_losses).sum() * scale
    return msg_loss


def shuffled_batchify(data, batch_size):
    idx = list(range(len(data)))
    random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield [data[j] for j in idx[i:i + batch_size]]


def main():
    ap = argparse.ArgumentParser(description="PPO-LOOP training step (isolated torch process)")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--rollouts", required=True, help="pickle of [{advantage, messages}]")
    ap.add_argument("--lora-in", default="")
    ap.add_argument("--lora-out", required=True)
    ap.add_argument("--optimizer-in", default="")
    ap.add_argument("--optimizer-out", default="")
    ap.add_argument("--metrics-out", default="")
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--n-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--max-prompt-tokens", type=int, default=6000,
                    help="Truncate each message prompt to its last N tokens to bound "
                         "training memory (output tokens are always kept).")
    ap.add_argument("--max-memory-per-gpu", type=str, default="9GiB",
                    help="Per-GPU weight budget for device_map so GPU 0 isn't packed "
                         "full of weights (leaves headroom for activations). '' to disable.")
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    with open(args.rollouts, "rb") as f:
        train_data = pickle.load(f)
    # Keep only episodes that actually carry trainable tokens.
    train_data = [d for d in train_data if d.get("messages")]
    print(f"   Loaded {len(train_data)} trainable episodes from {args.rollouts}", flush=True)

    policy_model = build_policy_model(args)
    loss_device = policy_model.get_input_embeddings().weight.device

    trainable = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    opt_in = args.optimizer_in.strip() if args.optimizer_in else ""
    if opt_in and Path(opt_in).exists():
        try:
            optimizer.load_state_dict(torch.load(opt_in, map_location="cpu"))
            print(f"   Resumed optimizer state from {opt_in}", flush=True)
        except Exception as e:
            print(f"   ⚠️  Could not load optimizer state ({e}); starting fresh", flush=True)

    total_epoch_loss = 0.0
    skipped_oom = 0
    for epoch in range(args.n_epochs):
        epoch_loss, num_batches = 0.0, 0
        for minibatch in shuffled_batchify(train_data, args.batch_size):
            # Paper: drop low-advantage rollouts (|A| < 0.01) before the gradient.
            valid = [ep for ep in minibatch if abs(ep["advantage"]) >= 0.01]
            # Total output tokens in this minibatch -> denominator for the mean.
            N = sum(message_token_count(m) for ep in valid for m in ep["messages"])
            if N == 0:
                continue

            optimizer.zero_grad()
            batch_loss, did_step = 0.0, False
            # Backward PER MESSAGE so only one forward's graph is alive at a time
            # (the whole-minibatch graph is what OOM'd before).
            for ep in valid:
                for m in ep["messages"]:
                    if message_token_count(m) == 0:
                        continue
                    try:
                        l = message_loss(policy_model, m, ep["advantage"], args.epsilon,
                                         1.0 / N, args.max_prompt_tokens)
                        l.backward()
                        batch_loss += l.item()
                        did_step = True
                    except torch.OutOfMemoryError:
                        # Skip just this one oversized message; keep the grads
                        # already accumulated from this batch's other messages.
                        skipped_oom += 1
                        torch.cuda.empty_cache()
                        continue
            if did_step:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                epoch_loss += batch_loss
                num_batches += 1
                if num_batches % 10 == 0:
                    peak = torch.cuda.max_memory_allocated() / 1024**3
                    print(f"   [e{epoch+1}] batch {num_batches}: "
                          f"running_avg_loss={epoch_loss/num_batches:.6f}, "
                          f"peak_gpu={peak:.1f}GiB", flush=True)
        avg = epoch_loss / num_batches if num_batches else 0.0
        total_epoch_loss += avg
        print(f"   Epoch {epoch+1}/{args.n_epochs}, Avg Loss: {avg:.6f}", flush=True)

    avg_loss = total_epoch_loss / args.n_epochs if args.n_epochs else 0.0
    if skipped_oom:
        print(f"   ⚠️  Skipped {skipped_oom} oversized message(s) due to OOM", flush=True)

    # Save adapter + optimizer + metrics
    Path(args.lora_out).mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(args.lora_out)
    print(f"💾 Saved LoRA to {args.lora_out}", flush=True)

    opt_out = args.optimizer_out.strip() if args.optimizer_out else str(Path(args.lora_out) / "optimizer.pt")
    torch.save(optimizer.state_dict(), opt_out)

    if args.metrics_out:
        with open(args.metrics_out, "w") as f:
            json.dump({"avg_loss": avg_loss, "num_episodes": len(train_data)}, f)
    print(f"AVG_LOSS={avg_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
