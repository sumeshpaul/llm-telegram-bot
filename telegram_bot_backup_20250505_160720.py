import os
import logging
import requests
import re
from datetime import datetime
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from telegram import Update, constants
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# === Load Environment ===
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
LLM_API_URL = os.getenv("LLM_API_URL", "http://127.0.0.1:8000/generate")
LLM_API_TOKEN = os.getenv("LLM_API_TOKEN")

# === User History ===
user_histories = {}

# === Helper: Split Long Messages ===
def split_message(text, limit=4000):
    return [text[i:i + limit] for i in range(0, len(text), limit)]

# === Fallback Logic ===
def fallback_search(query: str) -> str:
    try:
        now = datetime.now()
        formatted_date = now.strftime("%A, %d %B %Y")
        with DDGS() as ddgs:
            results = ddgs.text(query, region="wt-wt", safesearch="Moderate", max_results=5)
            if not results:
                return "⚠️ No relevant real-time info found."

            top_result = results[0]
            title = top_result.get("title", "Source")
            snippet = top_result.get("body", "No summary available.").strip()
            return f"_As of {formatted_date}_\n🔎 *{title}*:\n{snippet}"
    except Exception as e:
        return f"⚠️ Real-time fallback failed: {str(e)}"

# === /start ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Hello! I’m your assistant. Ask me anything!")

# === Handle Messages ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_input = update.message.text.strip()

    if user_id not in user_histories:
        user_histories[user_id] = []

    user_histories[user_id].append({"role": "user", "content": user_input})
    if len(user_histories[user_id]) > 20:
        user_histories[user_id] = user_histories[user_id][-20:]

    # === LLM Call ===
    try:
        response = requests.post(
            LLM_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LLM_API_TOKEN}"
            },
            json={"prompt": user_input, "max_tokens": 1024},
            timeout=15
        )
        response.raise_for_status()
        reply_text = response.json().get("response", "").strip()

        # Mirror main.py logic — fallback only if reply is totally empty
        if not reply_text:
            raise ValueError("Empty LLM response")

    except Exception as e:
        logging.warning(f"[LLM fallback triggered] {e}")
        reply_text = fallback_search(user_input)

    for chunk in split_message(reply_text):
        await update.message.reply_text(chunk, parse_mode=constants.ParseMode.MARKDOWN)

    user_histories[user_id].append({"role": "assistant", "content": reply_text})

# === Main ===
def main():
    logging.basicConfig(level=logging.INFO)
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logging.info("✅ Telegram bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
