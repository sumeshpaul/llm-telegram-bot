from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import uvicorn
import os

MODEL_DIR = "./final_lora_model"

print("🔄 Loading PEFT config...")
config = PeftConfig.from_pretrained(MODEL_DIR)

print(f"🔄 Loading base model: {config.base_model_name_or_path}")
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("🔄 Attaching LoRA adapter...")
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    user_id: str

@app.get("/")
async def root():
    return {"message": "✅ LLM API is running"}

@app.post("/predict")
async def predict(request: PromptRequest):
    try:
        prompt = request.prompt.strip()
        user_id = request.user_id

        # Format the prompt
        full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded.split("### Response:")[-1].strip()

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

        # Return text after last ### Response:
        if "### Response:" in decoded:
            response = decoded.split("### Response:")[-1].strip()
        else:
            response = decoded.strip()

        return {
            "prompt": prompt,
            "response": response,
            "predicted_label": "n/a"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("start_llm_api:app", host="0.0.0.0", port=8000, reload=False)
