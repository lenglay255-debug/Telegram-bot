import asyncio
import logging
from google import genai
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
import os

# ==================================
# CONFIGURATION & API KEYS
# ==================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ប្រើប្រាស់ Model ថ្មី និងស្តង់ដាររបស់ Google GenAI SDK
MODEL = "gemini-2.0-flash"

client = genai.Client(api_key=GEMINI_API_KEY)

logging.basicConfig(level=logging.INFO)

# Store chat sessions and locks in memory
sessions = {}
locks = {}

SYSTEM_PROMPT = """
You are a smart, friendly Telegram AI assistant.

You can:
- Answer general questions
- Speak Khmer and English
- Help with schoolwork
- Solve mathematics
- Explain programming
- Write and debug Python code
- Translate languages
- Write stories and essays

Always respond in the user's language.
Be accurate, helpful and friendly.
If you do not know something, say so.
"""

def get_lock(user_id):
    if user_id not in locks:
        locks[user_id] = asyncio.Lock()
    return locks[user_id]


def get_session(user_id):
    if user_id not in sessions:
        sessions[user_id] = client.chats.create(
            model=MODEL,
            config={
                "system_instruction": SYSTEM_PROMPT
            }
        )
    return sessions[user_id]


def generate_answer(user_id, question):
    chat = get_session(user_id)
    response = chat.send_message(question)
    return response.text or "No response generated."


async def send_answer(message, answer):
    while answer:
        chunk = answer[:4000]
        answer = answer[4000:]
        await message.reply_text(chunk)


# ==================================
# COMMANDS (/start, /reset, /help)
# ==================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Welcome to Gemini AI Bot!\n\n"
        "🇰🇭 សួស្តី! ខ្ញុំជា Gemini AI Bot។\n\n"
        "អ្នកអាចសួរខ្ញុំអំពី៖\n"
        "📚 ការសិក្សា\n"
        "💻 Programming\n"
        "🧮 Mathematics\n"
        "🌍 Translation\n"
        "✍️ Writing\n"
        "🧠 General Knowledge\n\n"
        "Commands:\n"
        "/start - Start bot\n"
        "/reset - Clear memory\n"
        "/help - Help"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    async with get_lock(user_id):
        sessions.pop(user_id, None)

    await update.message.reply_text("🔄 Memory cleared successfully!")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Gemini AI Bot\n\n"
        "Send me any text question.\n"
        "I support Khmer and English.\n\n"
        "/reset - Start a new conversation."
    )


# ==================================
# MESSAGE HANDLER
# ==================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    question = update.message.text

    async with get_lock(user_id):
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        try:
            answer = await asyncio.to_thread(
                generate_answer,
                user_id,
                question
            )

        except Exception as error:
            logging.exception("Gemini error occurred")
            error_text = str(error).lower()

            if "429" in error_text:
                answer = "⚠️ Gemini API rate limit reached. Please try again later."
            elif "404" in error_text:
                answer = "⚠️ Gemini model not available. Please check the model name."
            elif "403" in error_text or "api key" in error_text:
                answer = "⚠️ API key error. Please check your Gemini API key."
            else:
                answer = f"❌ AI connection error: {str(error)[:100]}"

        await send_answer(update.message, answer)


# ==================================
# ERROR HANDLER (Fixes the warning)
# ==================================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logging.error("Exception while handling an update:", exc_info=context.error)


# ==================================
# MAIN BOT INITIALIZATION
# ==================================

def main():
    if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
        print("❌ Please set TELEGRAM_TOKEN and GEMINI_API_KEY environment variables!")
        return

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Register error handler to avoid missing error handler logs
    app.add_error_handler(error_handler)

    print("==============================")
    print("🤖 GEMINI TELEGRAM BOT")
    print("✅ Starting bot...")
    print("==============================")

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
