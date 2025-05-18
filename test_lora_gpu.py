from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
import torch

print("🧠 Starting test with LoRA + bitsandbytes on GPU")

model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

# ✅ Prepare model for 8-bit LoRA tuning
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# ✅ Dummy forward pass to initialize BnB internal states
_ = model(tokenizer("Hello world!", return_tensors="pt").to("cuda"))

# Apply LoRA
try:
    model = PeftModel.from_pretrained(model, "./final_lora_model_v2")
    print("✅ Loaded LoRA model overlay")
except Exception as e:
    print(f"⚠️ LoRA model not loaded: {e}")

# Optional: Torch compile for RTX 5080
torch._dynamo.config.suppress_errors = True
model = torch.compile(model, backend="inductor", mode="max-autotune")
print("⚙️ Model compiled with torch.compile + inductor")

# Run inference
inputs = tokenizer("What is the capital of France?", return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=32)
    print("📝 Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
