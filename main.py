import os
import datetime
from typing import List
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from llama_cpp import Llama

# === Load environment variables ===
load_dotenv(dotenv_path="/home/sam/.env")
API_TOKEN = os.getenv("LLM_API_TOKEN")

# === Initialize LLM model ===
llm_model = Llama(
    model_path="/mnt/mymodels/Nous-Hermes-2-Mistral-7B-DPO.Q4_K_M.gguf",
    n_ctx=16384,
    chat_format="chatml",
    f16_kv=True,
    n_threads=12
)

app = FastAPI()

# === DuckDuckGo Fallback Search ===
def fallback_search(query: str) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region="wt-wt", safesearch="Moderate", max_results=5)

            for r in results:
                body = r.get("body", "").lower()
                if "donald trump" in body:
                    return f"(As of {now}) Donald Trump is the current President of the United States."
                elif "joe biden" in body:
                    return f"(As of {now}) Joe Biden is the current President of the United States."

            for r in results:
                return f"(As of {now}) {r.get('title', '')}: {r.get('body', '')}"
    except Exception as e:
        return f"(As of {now}) Could not fetch real-time information. Error: {str(e)}"

    return f"(As of {now}) No relevant information found."

# === Request Models ===
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 4096

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 4096

# === /generate Endpoint ===
@app.post("/generate")
def generate(req: GenerateRequest, authorization: str = Header(None)):
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        output = llm_model(prompt=req.prompt, max_tokens=req.max_tokens, stop=["<|im_end|>"])
        reply = output["choices"][0]["text"].strip()

        # Check for outdated info
        lower_reply = reply.lower()
        if (
            not reply or
            "joe biden" in lower_reply or
            "as of 2021" in lower_reply or
            "i don't know" in lower_reply
        ):
            raise ValueError("Outdated or weak response")
    except Exception:
        reply = fallback_search(req.prompt)

    return {"response": reply}
# === /chat Endpoint ===
@app.post("/chat")
def chat(req: ChatRequest, authorization: str = Header(None)):
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        output = llm_model.create_chat_completion(
            messages=[msg.dict() for msg in req.messages],
            max_tokens=req.max_tokens,
            temperature=0.7,
            stop=["<|im_end|>"]
        )
        reply = output["choices"][0]["message"]["content"].strip()
        if not reply or "i don't know" in reply.lower():
            raise ValueError("LLM gave weak chat reply")
    except Exception:
        query = req.messages[-1].content
        reply = fallback_search(query)
    return {"response": reply}
