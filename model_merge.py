#!/usr/bin/env python3
"""
Merge specialist LoRA adapters with base model and push to HuggingFace
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

HF_TOKEN = "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"  
HF_ORG = "arkitekt-ai"
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Specialists to merge - (checkpoint_dir, lora_iteration, output_name)
SPECIALISTS = [
    ("checkpoints/specialist_d1_take2", 3, "qwen-coder-7b-specialist-d1"),
    ("checkpoints/specialist_d2", 3, "qwen-coder-7b-specialist-d2"),
]

# =============================================================================
# MAIN
# =============================================================================

def merge_and_push(checkpoint_dir: str, lora_iter: int, output_name: str):
    """Merge a LoRA adapter with base model and push to HF"""
    
    lora_path = os.path.join(checkpoint_dir, f"lora_iter_{lora_iter}")
    local_output = f"models/{output_name}"
    hf_repo = f"{HF_ORG}/{output_name}"
    
    print(f"\n{'='*60}")
    print(f"Processing: {output_name}")
    print(f"  LoRA path: {lora_path}")
    print(f"  HF repo: {hf_repo}")
    print(f"{'='*60}")
    
    # Verify LoRA exists
    if not os.path.exists(lora_path):
        print(f"❌ LoRA not found at {lora_path}")
        return False
    
    # Load base model
    print(f"\n📦 Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # Load and merge LoRA
    print(f"🔗 Loading LoRA from {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("🔀 Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    
    # Save locally
    print(f"💾 Saving merged model to {local_output}")
    os.makedirs(local_output, exist_ok=True)
    merged_model.save_pretrained(local_output)
    tokenizer.save_pretrained(local_output)
    
    # Push to HuggingFace
    print(f"🚀 Pushing to HuggingFace: {hf_repo}")
    merged_model.push_to_hub(
        hf_repo,
        token=HF_TOKEN,
        private=False,  # Set to True if you want private repos
    )
    tokenizer.push_to_hub(
        hf_repo,
        token=HF_TOKEN,
    )
    
    print(f"✅ Successfully pushed to https://huggingface.co/{hf_repo}")
    
    # Clear GPU memory
    del model, merged_model, base_model
    torch.cuda.empty_cache()
    
    return True


def main():
    # Login to HuggingFace
    print("🔐 Logging in to HuggingFace...")
    login(token=HF_TOKEN)
    
    # Process each specialist
    results = []
    for checkpoint_dir, lora_iter, output_name in SPECIALISTS:
        success = merge_and_push(checkpoint_dir, lora_iter, output_name)
        results.append((output_name, success))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    print("\n🎉 Done!")


if __name__ == "__main__":
    main()