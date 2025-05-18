import os
import asyncio
import logging
import aiohttp
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
import nest_asyncio

# Apply patch to allow nested event loops (for VSCode and Jupyter)
nest_asyncio.apply()
load_dotenv()

# Load environment variable for Telegram bot token
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
LLM_API_URL = "http://localhost:8000/predict"

# Logging config
logging.basicConfig(level=logging.INFO)

# Handler for user messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_prompt = update.message.text.strip()
    user_id = str(update.effective_user.id)

    # Construct prompt as expected by your model
    full_prompt = f"### Instruction:\n{user_prompt}\n\n### Response:\n"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                LLM_API_URL,
                json={"prompt": full_prompt, "user_id": user_id},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    answer = data.get("response", "No response.")
                else:
                    answer = f"❌ LLM server error: Status {response.status}"
    except Exception as e:
        answer = f"⚠️ Failed to reach LLM server:\n{str(e)}"

    await update.message.reply_text(answer)

# Start polling
async def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
