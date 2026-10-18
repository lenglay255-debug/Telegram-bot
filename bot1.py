
import os
import io
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

# =========================
# CONFIGURATION
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

TOKEN = os.getenv("TELEGRAM_TOKEN")
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not TOKEN or not API_KEY:
    raise RuntimeError("Missing Telegram or Gemini API key")

client = genai.Client(api_key=API_KEY)

SYSTEM_PROMPT = """
You are an intelligent educational Telegram AI assistant.

You can:
- Answer in Khmer and English
- Analyze photos and handwritten exercises
- Solve mathematics step by step
- Explain physics and chemistry
- Help with homework and programming
- Translate languages
- Explain difficult concepts simply

Rules:
1. Always respond in the user's language.
2. For math, show formulas and solution steps.
3. For images, carefully read the question.
4. If image text is unclear, ask for a clearer photo.
5. Never invent missing information.
6. Be friendly, accurate, and educational.
"""

sessions = {}
locks = {}

# =========================
# RENDER HEALTH SERVER
# =========================

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Gemini Telegram Bot Running")

    def log_message(self, *args):
        pass


def health_server():
    port = int(os.getenv("PORT", "10000"))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    server.serve_forever()


# =========================
# GEMINI FUNCTIONS
# =========================

def get_lock(user_id):
    if user_id not in locks:
        locks[user_id] = asyncio.Lock()
    return locks[user_id]


def get_chat(user_id):
    if user_id not in sessions:
        sessions[user_id] = client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            )
        )
    return sessions[user_id]


def ask_gemini(user_id, question, image_bytes=None):
    chat = get_chat(user_id)

    if image_bytes is not None:
        content = [
            types.Part.from_text(text=question),
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        ]
    else:
        content = question

    response = chat.send_message(content)
    return response.text or "No response generated."


async def send_answer(message, answer):
    for i in range(0, len(answer), 4000):
        await message.reply_text(answer[i:i + 4000])


def format_error(error):
    logging.error("Gemini error: %s", error)
    msg = str(error).lower()

    if "429" in msg:
        return "⚠️ Gemini quota exceeded. Try again later."
    if "404" in msg:
        return "⚠️ Model unavailable. Check GEMINI_MODEL."
    if "403" in msg or "api key" in msg:
        return "⚠️ API key or permission error."
    return "❌ AI error. Please check Render logs."


# =========================
# BOT COMMANDS
# =========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 សួស្តី! ខ្ញុំជា Gemini AI Bot!\n\n"
        "📸 ផ្ញើរូបភាពលំហាត់មកខ្ញុំ\n"
        "🧮 ដោះស្រាយគណិតវិទ្យា\n"
        "📚 ជួយការសិក្សា\n"
        "💻 Programming\n"
        "🌍 បកប្រែភាសា\n\n"
        "អ្នកអាចផ្ញើសំណួរ ឬរូបភាពបាន!\n\n"
        "/reset - Clear memory\n"
        "/help - Help"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    async with get_lock(user_id):
        sessions.pop(user_id, None)

    await update.message.reply_text(
        "✅ បានលុបប្រវត្តិសន្ទនារួចរាល់!"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Gemini AI Bot\n\n"
        "📝 Send a text question\n"
        "📸 Send a photo of homework\n"
        "🧮 Ask for step-by-step solutions\n\n"
        "/reset - New conversation"
    )


# =========================
# TEXT HANDLER
# =========================

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    question = update.message.text

    async with get_lock(user_id):
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING
        )

        try:
            answer = await asyncio.to_thread(
                ask_gemini, user_id, question
            )
        except Exception as error:
            logging.exception("Text processing failed")
            answer = format_error(error)

        await send_answer(update.message, answer)


# =========================
# PHOTO HANDLER
# =========================

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    question = (
        update.message.caption
        or "សូមអានរូបភាពនេះ ហើយដោះស្រាយលំហាត់ "
           "ដោយពន្យល់លម្អិតជាជំហានៗជាភាសាខ្មែរ។"
    )

    async with get_lock(user_id):
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING
        )

        try:
            photo = update.message.photo[-1]
            file = await photo.get_file()

            image_data = await file.download_as_bytearray()
            image_bytes = bytes(image_data)

            answer = await asyncio.to_thread(
                ask_gemini,
                user_id,
                question,
                image_bytes
            )

        except Exception as error:
            logging.exception("Photo processing failed")
            answer = format_error(error)

        await send_answer(update.message, answer)


# =========================
# ERROR HANDLER
# =========================

async def error_handler(update, context):
    logging.error(
        "Telegram error: %s",
        context.error,
        exc_info=context.error
    )


# =========================
# START BOT
# =========================

def main():
    threading.Thread(
        target=health_server,
        daemon=True
    ).start()

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("help", help_command))

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_text
        )
    )

    app.add_handler(
        MessageHandler(
            filters.PHOTO,
            handle_photo
        )
    )

    app.add_error_handler(error_handler)

    logging.info("Gemini AI Bot starting: %s", MODEL)

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
