# ✅ Final start_combined.py with Improved Weather Logic, Error Logging, Aliases, Templates, Memory

from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from uvicorn import Config, Server
from dotenv import load_dotenv
import torch
import asyncio
import aiohttp
import sqlite3
import os
import re
import pickle
import faiss
import nest_asyncio
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer

from utils.weather import get_weather_for_city
from utils.wikipedia_fallback import get_wikipedia_weather_summary, get_wikipedia_summary
from utils.search import fallback_search
from utils.triggers import is_weather_related, is_fallback_needed

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
nest_asyncio.apply()
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
API_KEY = os.getenv("API_KEY", "DESKNAV-AI-2025-SECURE")
LLM_MODEL_PATH = "/mnt/models/final_lora_model"
BASE_MODEL_PATH = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
DB_PATH = "./data/query_logs.db"
UPLOAD_DIR = Path("./rag_data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
user_histories = {}
user_rag_map = {}
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("🔄 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
print("🔄 Attaching LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LLM_MODEL_PATH, device_map="auto")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

city_aliases = {
    "UAE": "Dubai", "Kerala": "Thiruvananthapuram", "India": "New Delhi",
    "KSA": "Riyadh", "USA": "Washington", "UK": "London"
}

TEMPLATE_KNOWLEDGE = {
    "what is penne pasta?": """Penne is a type of cylindrical pasta with angled ends, resembling the nib of a fountain pen. It is one of the most popular pasta types in Italian cuisine.

Penne originated in Italy and was patented in 1865 by Giovanni Battista Capurro, a pasta maker from Genoa. His invention allowed pasta to be cut diagonally without crushing it.

Penne pairs well with chunky sauces like arrabbiata, pesto, or creamy Alfredo. Its ridges help hold sauces, making it ideal for baked dishes and pasta salads."""
}

def is_vague_prompt(prompt: str) -> bool:
    vague_patterns = ["what is it", "what is it used for", "can you share", "tell me more", "how about that"]
    return any(prompt.strip().lower().startswith(p) for p in vague_patterns)

def rewrite_vague_prompt(prompt: str, user_id: str) -> str:
    if user_id in user_histories and user_histories[user_id]:
        last_q = user_histories[user_id][-1]["user"]
        topic = last_q.strip(" ?.")
        return f"{prompt.strip().capitalize()} about {topic}?"
    return prompt

def check_cached_response(user_id: str, prompt: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT response, source FROM queries
        WHERE user_id = ? AND prompt = ?
        ORDER BY timestamp DESC LIMIT 1
    """, (user_id, prompt.strip()))
    row = cur.fetchone()
    conn.close()
    if row:
        return row[0], row[1]
    return None, None

def format_response(answer, source, fallback_used=False, reason=None):
    if fallback_used:
        import re
        answer = re.sub(r'(?<!\n)\s*\n\s*(?!\n)', ' ', answer).strip()
        paragraphs = re.split(r'(?<=\.)\s+(?=[A-Z])', answer)
        if len(paragraphs) < 2:
            answer = f"Here is what I could find:\n\n{answer.strip()}\n\nLet me know if you'd like more historical or usage details."
        else:
            answer = "\n\n".join(paragraphs[:3])
        if not answer.endswith(('.', '!', '?')):
            answer += '.'
    meta = f"\n\n### Fallback Used:\n{fallback_used}\n### Reason:\n{reason}" if fallback_used else ""
    return f"### Answer:\n{answer}\n\n### Source:\n{source}\n\n### Timestamp:\n{datetime.utcnow().isoformat()} UTC{meta}"

def log_query(user_id, prompt, response, source):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, user_id TEXT, prompt TEXT, response TEXT, source TEXT
        )
    """)
    cur.execute("INSERT INTO queries (timestamp, user_id, prompt, response, source) VALUES (?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), user_id, prompt, response, source))
    conn.commit()
    conn.close()

def update_user_memory(user_id, prompt, answer):
    if user_id not in user_histories:
        user_histories[user_id] = []
    user_histories[user_id].append({"user": prompt, "bot": answer})

def generate_response(history):
    text = "".join(f"### Instruction:\n{m['user']}\n\n### Response:\n{m['bot']}\n\n" for m in history[:-1])
    text += f"### Instruction:\n{history[-1]['user']}\n\n### Response:\n"
    tokens = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**tokens, max_new_tokens=1024, temperature=0.7, top_p=0.9, do_sample=True)
    return tokenizer.decode(out[0], skip_special_tokens=True).split("### Response:")[-1].strip()

def load_user_index(user_id):
    if user_id in user_rag_map:
        return user_rag_map[user_id]
    try:
        index = faiss.read_index(f"./rag_data/generated/{user_id}/faiss.index")
        with open(f"./rag_data/generated/{user_id}/metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        user_rag_map[user_id] = (index, meta)
        return index, meta
    except:
        return None, []

def query_vector_store(prompt, index, metadata, top_k=3):
    vec = embedder.encode([prompt])
    D, I = index.search(vec, top_k)
    return [metadata[i] for i in I[0] if i < len(metadata)]

class PromptRequest(BaseModel):
    prompt: str
    user_id: str

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(request: PromptRequest):
    prompt, user_id = request.prompt.strip(), request.user_id

    if is_vague_prompt(prompt):
        prompt = rewrite_vague_prompt(prompt, user_id)

    cached, cached_source = check_cached_response(user_id, prompt)
    if cached:
        return {"response": cached}

    template_answer = TEMPLATE_KNOWLEDGE.get(prompt.lower())
    if template_answer:
        log_query(user_id, prompt, template_answer, "template")
        return {"response": format_response(template_answer, "template")}

    if is_weather_related(prompt):
        city_match = re.search(r"\b(?:weather(?:\s+in)?|in)\s+([A-Za-z\s]+)", prompt, re.IGNORECASE)
        city = city_match.group(1).strip().title() if city_match else None
        if city in city_aliases:
            city = city_aliases[city]
        try:
            result = await get_weather_for_city(city)
            log_query(user_id, prompt, result, f"weather:{city}")
            return {"response": format_response(result, f"weather:{city}")}
        except Exception as e:
            print(f"[OpenWeather ERROR] {e}")
            try:
                summary = await get_wikipedia_weather_summary(city)
                log_query(user_id, prompt, summary, f"wiki-weather:{city}")
                return {"response": format_response(summary, f"wiki-weather:{city}", True, "OpenWeatherMap failed")}
            except:
                error = f"⚠️ No weather or Wikipedia info for '{city}'."
                log_query(user_id, prompt, error, "weather-fail")
                return {"response": format_response(error, "weather-fail", True, "Wiki fallback failed")}

    if user_id not in user_histories:
        user_histories[user_id] = []
    if any(x in prompt.lower() for x in ["upload", "pdf", "summarize"]):
        index, meta = load_user_index(user_id)
        if not index:
            msg = "⚠️ Upload required."
            log_query(user_id, prompt, msg, "no-index")
            return {"response": format_response(msg, "no-index")}
        context = "\n\n".join(query_vector_store(prompt, index, meta))
        prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"

    hist = user_histories[user_id][-5:]
    hist.append({"user": prompt, "bot": ""})
    try:
        answer = generate_response(hist)
        if is_fallback_needed(prompt, answer):
            try:
                fallback = await fallback_search(prompt)
                if not fallback or len(fallback.strip()) < 40 or not any(p in fallback for p in ".!?\n"):
                    raise Exception("DuckDuckGo insufficient or vague")
                log_query(user_id, prompt, fallback, "fallback")
                return {"response": format_response(fallback, "fallback", True, "LLM failed")}
            except:
                try:
                    summary = await get_wikipedia_summary(prompt)
                    log_query(user_id, prompt, summary, "wiki-fallback")
                    return {"response": format_response(summary, "wiki-fallback", True, "DuckDuckGo failed")}
                except:
                    msg = "⚠️ All fallback methods failed."
                    log_query(user_id, prompt, msg, "ultimate-fail")
                    return {"response": format_response(msg, "ultimate-fail", True, "Final fallback failed")}
        hist[-1]["bot"] = answer
        user_histories[user_id] = hist
        update_user_memory(user_id, prompt, answer)
        log_query(user_id, prompt, answer, "llm")
        return {"response": format_response(answer, "llm")}
    except Exception as e:
        error = f"❌ LLM Error: {e}"
        log_query(user_id, prompt, error, "error")
        return {"response": format_response(error, "error")}

@app.get("/")
def root():
    return {"message": "✅ LLM API live"}

@app.get("/queries")
def get_queries(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, timestamp, user_id, prompt, response, source FROM queries ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "timestamp": r[1], "user_id": r[2], "prompt": r[3], "response": r[4], "source": r[5]} for r in rows]

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post("http://localhost:8000/predict", json={"prompt": update.message.text, "user_id": uid}, headers={"x-api-key": API_KEY}) as resp:
                data = await resp.json()
                reply = data.get("response", "🤖 No response.")
        except Exception as e:
            reply = f"⚠️ Telegram error: {e}"
    await update.message.reply_text(reply)

async def run_telegram_bot():
    bot = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    await bot.run_polling()

async def run_fastapi():
    config = Config(app=app, host="0.0.0.0", port=8000, log_level="info")
    server = Server(config)
    await server.serve()

async def main():
    fastapi_task = asyncio.create_task(run_fastapi())
    if TELEGRAM_TOKEN:
        telegram_task = asyncio.create_task(run_telegram_bot())
        await asyncio.gather(fastapi_task, telegram_task)
    else:
        await fastapi_task

if __name__ == "__main__":
    asyncio.run(main())
