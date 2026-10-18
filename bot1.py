
import os
import asyncio
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from google import genai
from google.genai import types
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set GEMINI_MODEL in Render if needed.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

SYSTEM_PROMPT = """
You are a helpful and intelligent Telegram AI bot.
Answer questions in Khmer or English.
Always reply in the user's language.
Help with programming, schoolwork, mathematics,
translation, writing, and general questions.
Explain clearly and accurately.
"""

sessions = {}
locks = {}

client = None


# HTTP server for Render
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Telegram bot service running")

    def log_message(self, format, *args):
        pass


def start_health_server():
    port = int(os.getenv("PORT", "10000"))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logging.info("HTTP server listening on port %s", port)
    server.serve_forever()


def get_lock(user_id):
    if user_id not in locks:
        locks[user_id] = asyncio.Lock()
    return locks[user_id]


def generate_answer(user_id, question):
    if user_id not in sessions:
        sessions[user_id] = client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            )
        )

    response = sessions[user_id].send_message(question)
    return response.text or "No text response generated."


async def send_long_message(message, text):
    for i in range(0, len(text), 4000):
        await message.reply_text(text[i:i + 4000])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Welcome to Gemini AI Bot!\n\n"
        "🇰🇭 សួស្តី! ខ្ញុំអាចជួយអ្នកបាន។\n\n"
        "📚 Education\n"
        "💻 Programming\n"
        "🧮 Mathematics\n"
        "🌍 Translation\n"
        "✍️ Writing\n\n"
        "Send me a question!\n\n"
        "/reset - Clear conversation\n"
        "/help - Show help"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    async with get_lock(user_id):
        sessions.pop(user_id, None)

    await update.message.reply_text("✅ Memory cleared!")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Gemini AI Assistant\n\n"
        "Ask me anything in Khmer or English.\n"
        "/start - Start\n"
        "/reset - Clear memory\n"
        "/help - Help"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    question = update.message.text

    async with get_lock(user_id):
        try:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.TYPING
            )

            answer = await asyncio.to_thread(
                generate_answer, user_id, question
            )

        except Exception as error:
            logging.exception("Gemini request failed")
            error_text = str(error).lower()

            if "429" in error_text:
                answer = "⚠️ API quota exceeded. Try again later."
            elif "404" in error_text:
                answer = (
                    "⚠️ Gemini model unavailable.\n"
                    "Check GEMINI_MODEL in Render."
                )
            elif "403" in error_text or "api key" in error_text:
                answer = "⚠️ Gemini API key or permission error."
            else:
                answer = "❌ AI error. Check Render logs."

        await send_long_message(update.message, answer)


async def error_handler(update, context):
    logging.error(
        "Telegram error",
        exc_info=(
            type(context.error),
            context.error,
            context.error.__traceback__
        ) if context.error else None
    )


def main():
    global client

    if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
        raise RuntimeError(
            "Missing TELEGRAM_TOKEN or GEMINI_API_KEY"
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    threading.Thread(
        target=start_health_server,
        daemon=True
    ).start()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )
    app.add_error_handler(error_handler)

    logging.info("Starting Telegram bot with model %s", MODEL)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
