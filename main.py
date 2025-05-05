from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os
import re
import datetime
from duckduckgo_search import DDGS
from dotenv import load_dotenv

# === Load env ===
load_dotenv()
API_TOKEN = os.getenv("LLM_API_TOKEN")

# === FastAPI init ===
app = FastAPI()

# === Models ===
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256

class ChatRequest(BaseModel):
    messages: list
    max_tokens: int = 1024

# === Import your LLM handler ===
import re
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

llm = Llama(
    model_path="/mnt/mymodels/Nous-Hermes-2-Mistral-7B-DPO.Q5_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    n_threads=8,
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
        tokens = len(msg.get("content", "").split())
        if total_tokens + tokens > limit:
            break
        trimmed.insert(0, msg)
        total_tokens += tokens
    return trimmed

# (Update this import to reflect your actual LLM loader module)

# === Fallback search ===
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
                    return f"_(As of {date_str})_\n🔎 *{title}*:\n{snippet}"
    except Exception as e:
        return f"_(As of {date_str})_\n⚠️ Live search failed: {str(e)}"

# === /generate ===
@app.post("/generate")
def generate(req: GenerateRequest, authorization: str = Header(None)):
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        import datetime
        today = datetime.datetime.now().strftime("As of %A, %d %B %Y")
        full_prompt = f"{today}\n\n{req.prompt}"

        output = llm_model(
            prompt=full_prompt,
            max_tokens=req.max_tokens,
            stop=["\n\n", "<|im_end|>"]
        )
            reply = output["choices"][0]["text"].strip()
        if not reply:
            raise ValueError("Empty LLM response")

    except Exception:
        reply = fallback_search(req.prompt)

    return {"response": reply}

# === /chat ===
@app.post("/chat")
def chat(req: ChatRequest, authorization: str = Header(None)):
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        output = llm_model_create_chat_completion(
            messages=trim_chat_history([msg for msg in req.messages]),
            max_tokens=req.max_tokens,
            temperature=0.7,
            stop=["<|im_end|>"]
        )
        reply = output["choices"][0]["message"]["content"].strip()

        if (
            not reply
            or "i don't know" in reply.lower()
            or re.search(r"https?://", reply)
            or "wikipedia" in reply.lower()
            or re.search(r"\b(19\d{2}|20[0-2]\d)\b", reply)
            or len(reply.split()) < 5
        ):
            raise ValueError("Weak chat response")

    except Exception:
        last_user_message = req.messages[-1]['content']
        reply = fallback_search(last_user_message)

    return {"response": reply}
