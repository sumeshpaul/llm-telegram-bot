import os
import logging
import datetime
import requests
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

# === Load Environment Variables ===
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_TOKEN = os.getenv("LLM_API_TOKEN")

# === User history (in memory) ===
user_histories = {}

# === Fallback search ===
def fallback_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region="wt-wt", safesearch="Moderate", max_results=3)
            for r in results:
                body = r.get("body", "")
                if body:
                    return body.strip()
        return "⚠️ No live fallback results found."
    except Exception as e:
        return f"⚠️ Fallback failed: {str(e)}"

# === Utility to split long messages ===
def split_message(text, limit=4000):
    return [text[i:i + limit] for i in range(0, len(text), limit)]

# === /start command ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Hello! I’m ready. Ask me anything.")

# === Main handler ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_input = update.message.text.strip()
    date_str = datetime.datetime.now().strftime("As of %A, %d %B %Y")

    if user_id not in user_histories:
        user_histories[user_id] = []

    user_histories[user_id].append({"role": "user", "content": user_input})
    if len(user_histories[user_id]) > 20:
        user_histories[user_id] = user_histories[user_id][-20:]

    try:
        response = requests.post(
            LLM_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LLM_API_TOKEN}"
            },
            json={"prompt": f"({date_str})\n{user_input}", "max_tokens": 2048},
            timeout=15
        )
        response.raise_for_status()
        reply_text = response.json().get("response", "").strip()

        if (
            not reply_text
            or "i don't know" in reply_text.lower()
            or "wikipedia" in reply_text.lower()
            or "https://en.wikipedia.org" in reply_text.lower()
        ):
            raise ValueError("Weak or generic LLM reply")

    except Exception as e:
        logging.warning(f"[LLM fallback] {e}")
        reply_text = fallback_search(user_input)

    for chunk in split_message(reply_text):
        await update.message.reply_text(chunk, parse_mode=constants.ParseMode.MARKDOWN)

    user_histories[user_id].append({"role": "assistant", "content": reply_text})

# === Main entry ===
def main():
    logging.basicConfig(level=logging.INFO)
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logging.info("✅ Telegram bot is polling...")
    app.run_polling()

if __name__ == "__main__":
    main()
