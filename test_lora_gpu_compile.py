import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

print("🧠 Starting test with LoRA + bitsandbytes on GPU")

model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load model in 8bit with device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Apply LoRA if available
try:
    model = PeftModel.from_pretrained(model, "./final_lora_model_v2")
    print("✅ Loaded LoRA model overlay")
except Exception as e:
    print(f"⚠️ LoRA model not loaded: {e}")

# Compile model for CUDA sm_120 compatibility (RTX 5080)
torch._dynamo.config.suppress_errors = True  # optional, suppress tracing errors
model = torch.compile(model, backend="inductor", mode="max-autotune")
print("⚙️ Model compiled with torch.compile + inductor")

# Tokenize input
inputs = tokenizer("What is the capital of France?", return_tensors="pt").to("cuda")

# Inference
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=32)
    print("📝 Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
