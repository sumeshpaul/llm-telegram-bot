import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

app = FastAPI()

MODEL_DIR = "./final_lora_model"
BASE_MODEL = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"

print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("🔄 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("🧠 Loading LoRA model...")
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(req: ChatRequest):
    input_ids = tokenizer(req.prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=200, temperature=0.7)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"response": response}
