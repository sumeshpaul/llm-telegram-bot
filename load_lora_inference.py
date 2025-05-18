from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# 🔧 Setup
MODEL_DIR = "./final_lora_model"

print("🔄 Loading PEFT config...")
config = PeftConfig.from_pretrained(MODEL_DIR)

print(f"🔄 Loading base model: {config.base_model_name_or_path}")
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("🔄 Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token

# ✅ Set a test prompt
prompt = """### Instruction:
Explain what a blockchain is in simple terms.

### Response:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

print("🔍 Raw output tokens:\n", outputs[0])
decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

print("\n🧠 Raw Decoded Output:\n", decoded)

# Extract cleaned response
response = decoded.split("### Response:")[-1].strip()
print("\n✅ Final Cleaned Response:\n", response)
