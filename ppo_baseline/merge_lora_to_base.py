# merge_lora_to_base.py
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_to_base(lora_path: str, base_model: str, output_path: str):
    """
    Merge a LoRA adapter with base model
    """
    print(f"\n🔀 Merging LoRA with base model")
    print(f"   LoRA: {lora_path}")
    print(f"   Base: {base_model}")
    print(f"   Output: {output_path}")
    
    # Load base model on CPU: an 8B in bf16 (~16GB) doesn't fit one 16GB card,
    # and during training the GPUs are busy anyway. Merging is one-time, CPU is fine.
    print("   Loading base model (CPU)...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    
    # Load LoRA
    print("   Loading LoRA...")
    model = PeftModel.from_pretrained(base, lora_path)
    
    # Merge
    print("   Merging...")
    merged = model.merge_and_unload()
    
    # Save
    print("   Saving merged model...")
    merged.save_pretrained(output_path)
    
    # Copy tokenizer
    print("   Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ Merged model saved to {output_path}")
    
    # Cleanup
    del base, model, merged
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA with base model")
    parser.add_argument("--lora", type=str, required=True,
                       help="Path to LoRA adapter")
    parser.add_argument("--base", type=str,
                       default="Qwen/Qwen3-8B",
                       help="Base model name or path")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for merged model")
    args = parser.parse_args()
    
    # Verify LoRA exists
    if not Path(args.lora).exists():
        raise FileNotFoundError(f"LoRA not found: {args.lora}")
    
    merge_lora_to_base(args.lora, args.base, args.output)

if __name__ == "__main__":
    main()