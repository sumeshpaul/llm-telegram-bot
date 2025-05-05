import os
import logging
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from duckduckgo_search import DDGS

# Load token from .env
load_dotenv(dotenv_path="/home/sam/.env")

# Debugging print
import sys
print(f"TELEGRAM_TOKEN: {os.getenv('TELEGRAM_TOKEN')}", file=sys.stderr)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
LLM_API_TOKEN = os.getenv("LLM_API_TOKEN", "supersecuretoken123")
LLM_API_URL = "http://localhost:8000/chat"

# Fail fast if missing
if not TELEGRAM_TOKEN:
    print("❌ TELEGRAM_TOKEN is missing in .env")
    sys.exit(1)
if not LLM_API_TOKEN:
    print("❌ LLM_API_TOKEN is missing in .env")
    sys.exit(1)

# Logging config
logging.basicConfig(level=logging.INFO)

# In-memory chat history
user_histories = {}  # {user_id: [{"role": ..., "content": ...}, ...]}

# ✅ Split long replies to fit Telegram's 4096 char limit
def split_message(text, max_length=4000):
    lines = text.split('\n')
    chunks, current = [], ''
    for line in lines:
        if len(current) + len(line) + 1 < max_length:
            current += line + '\n'
        else:
            chunks.append(current.strip())
            current = line + '\n'
    if current:
        chunks.append(current.strip())
    return chunks

# 🔍 DuckDuckGo fallback search
def get_search_snippet(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region='wt-wt', safesearch='Moderate', max_results=3)
            if results:
                top = results[0]
                return f"🔎 *Live Result:*\n{top['title']}\n{top['body']}\n{top['href']}"
            else:
                return "❗ No live search results found."
    except Exception as e:
        logging.error(f"Search failed: {e}")
        return f"❗ Search failed: {e}"

# 📌 /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! I’m your AI Assistant. Ask me anything.")

# 💬 Main handler for user messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_input = update.message.text.strip()

    # Initialize history
    if user_id not in user_histories:
        user_histories[user_id] = []

    # Add user message to history
    user_histories[user_id].append({"role": "user", "content": user_input})
    if len(user_histories[user_id]) > 20:
        user_histories[user_id] = user_histories[user_id][-20:]

    # 🎯 Call LLM API
    try:
        response = requests.post(
            LLM_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LLM_API_TOKEN}"
            },
            json={"messages": user_histories[user_id]}
        )
        response.raise_for_status()
        reply_text = response.json().get("response", "No response from model.")
    except Exception as e:
        logging.error(f"LLM error: {e}")
        reply_text = "Sorry, something went wrong with the AI."

    # 🤖 Fallback to DuckDuckGo if reply is short or unclear
    if len(reply_text.strip()) < 50 or "I don't know" in reply_text.lower():
        logging.info("Fallback: using DuckDuckGo search.")
        search_result = get_search_snippet(user_input)
        reply_text += f"\n\n{search_result}"

    # ✂️ Send reply in chunks
    for chunk in split_message(reply_text):
        await update.message.reply_text(chunk)

    # Add assistant response to history
    user_histories[user_id].append({"role": "assistant", "content": reply_text})

# 🚀 Main entry point
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logging.info("Bot started and polling...")
    app.run_polling()

if __name__ == "__main__":
    main()
