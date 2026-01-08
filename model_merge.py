from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# Load and merge LoRA
model = PeftModel.from_pretrained(base_model, "checkpoints/lora_iter_3")
merged_model = model.merge_and_unload()

# Save as new model
merged_model.save_pretrained("models/qwen-ppo-iter3-merged")

# Also copy tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
tokenizer.save_pretrained("models/qwen-ppo-iter3-merged")

print("✅ Merged model saved to models/qwen-ppo-iter3-merged")