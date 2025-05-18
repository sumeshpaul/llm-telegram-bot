from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import os

# Load model directory
MODEL_DIR = "final_lora_model"

# Load PEFT config once
print("🔄 Loading PEFT config...")
config = PeftConfig.from_pretrained(MODEL_DIR)

# Load base model
print(f"🔄 Loading base model: {config.base_model_name_or_path}")
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Attach LoRA adapter
print("🔄 Attaching LoRA adapter...")
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

# Load tokenizer
print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# FastAPI app
app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    user_id: str

@app.get("/")
async def root():
    return {"message": "✅ LLM API is running"}

@app.post("/predict")
async def predict(request: PromptRequest):
    prompt = request.prompt
    try:
        print(f"📩 Prompt received: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.5,
                top_p=0.85,
                do_sample=True,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"✅ Response generated.")
        return {"response": decoded}
    except Exception as e:
        print(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Only needed if running this file directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("llm_fastapi_server:app", host="0.0.0.0", port=8000)
