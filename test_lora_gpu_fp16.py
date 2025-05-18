import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print("🧠 Starting test with LoRA on GPU (no quantization)")

model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load base model in float16 (no quantization)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Load your fine-tuned LoRA model if available
try:
    model = PeftModel.from_pretrained(model, "./final_lora_model_v2")
    print("✅ Loaded LoRA model overlay")
except Exception as e:
    print(f"⚠️ LoRA model not loaded: {e}")

# Tokenize
inputs = tokenizer("What is the capital of France?", return_tensors="pt").to("cuda")

# Inference
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=32)
    print("📝 Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
