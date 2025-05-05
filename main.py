import os
import re
import datetime
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
from duckduckgo_search import DDGS
from dotenv import load_dotenv

# === Load environment ===
load_dotenv()
API_TOKEN = os.getenv("LLM_API_TOKEN")

# === Initialize FastAPI ===
app = FastAPI()

# === Request Models ===
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 1024

# === LLM Model Setup ===
llm = Llama(
    model_path="/mnt/mymodels/Nous-Hermes-2-Mistral-7B-DPO.Q5_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=8192,
    n_threads=12,
    f16_kv=True,
    use_mlock=True,
    verbose=True,
)

def llm_model(prompt, max_tokens=4096, stop=None):
    output = llm(
        prompt=prompt,
        max_tokens=max_tokens,
        stop=stop,
        echo=False,
        temperature=0.7,
    )
    return output

def llm_model_create_chat_completion(messages, max_tokens=2048, stop=None, temperature=0.7):
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
    )
    return output

def trim_chat_history(messages, limit=4096):
    total_tokens = 0
    trimmed = []
    for msg in reversed(messages):
        tokens = len(msg.content.split())
        if total_tokens + tokens > limit:
            break
        trimmed.insert(0, msg)
        total_tokens += tokens
    return trimmed

# === Prompt Formatter ===
def format_prompt(prompt: str) -> str:
    date = datetime.datetime.now().strftime("%A, %d %B %Y")
    return f"You are a helpful assistant.\nAs of {date}, answer the following question clearly and helpfully.\n\nQ: {prompt}\nA:"

# === Weak Response Detector ===
def is_weak_response(reply_text: str) -> bool:
    return (
        not reply_text
        or "i don't know" in reply_text.lower()
        or re.search(r"https?://", reply_text)
        or "wikipedia" in reply_text.lower()
        or re.search(r"\\b(19\\d{2}|20[0-2]\\d)\\b", reply_text)
        or len(reply_text.strip()) < 30
    )

# === Fallback Search ===
def fallback_search(query: str) -> str:
    now = datetime.datetime.now()
    date_str = now.strftime("%A, %d %B %Y")
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region="wt-wt", safesearch="Moderate", max_results=5)
            for r in results:
                snippet = r.get("body", "").strip()
                title = r.get("title", "").strip()
                if snippet:
                    return f"(As of {date_str})\n🔎 {title}:\n{snippet}"
    except Exception as e:
        return f"(As of {date_str})\n⚠️ Live search failed: {str(e)}"

# === /generate endpoint ===
@app.post("/generate")
def generate(req: GenerateRequest, authorization: str = Header(None)):
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        prompt = format_prompt(req.prompt)
        output = llm_model(
            prompt=prompt,
            max_tokens=req.max_tokens,
            stop=["\n\n", "<|im_end|>"]
        )
        reply = output["choices"][0]["text"].strip()

        if is_weak_response(reply):
            raise ValueError("Weak LLM response")

    except Exception:
        reply = fallback_search(req.prompt)

    return {"response": reply}

# === /chat endpoint ===
@app.post("/chat")
def chat(req: ChatRequest, authorization: str = Header(None)):
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        trimmed = trim_chat_history(req.messages)

        # Insert system message
        trimmed.insert(0, ChatMessage(
            role="system",
            content=f"You are a helpful assistant. Today's date is {datetime.datetime.now().strftime('%A, %d %B %Y')}."
        ))

        output = llm_model_create_chat_completion(
            messages=[msg.dict() for msg in trimmed],
            max_tokens=req.max_tokens,
            temperature=0.7,
            stop=["<|im_end|>"]
        )
        reply = output["choices"][0]["message"]["content"].strip()

        if is_weak_response(reply):
            raise ValueError("Weak chat response")

    except Exception:
        last_user_message = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        reply = fallback_search(last_user_message)

    return {"response": reply}
