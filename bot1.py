
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

# ==========================
# CONFIG
# ==========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

TOKEN = os.getenv("TELEGRAM_TOKEN")
API_KEY = os.getenv("GEMINI_API_KEY")
PREFERRED_MODEL = os.getenv(
    "GEMINI_MODEL", "gemini-2.5-flash"
)

if not TOKEN or not API_KEY:
    raise RuntimeError(
        "Missing TELEGRAM_TOKEN or GEMINI_API_KEY"
    )

client = genai.Client(api_key=API_KEY)

SYSTEM_PROMPT = """
You are a smart educational Telegram AI assistant.

Your capabilities:
- Answer questions in Khmer and English
- Understand images and handwritten homework
- Solve mathematics step by step
- Explain physics and chemistry
- Help with programming
- Translate languages
- Explain school exercises clearly

Rules:
1. Always reply in the user's language.
2. For mathematics, show each solution step.
3. Read images carefully.
4. If an image is unclear, ask for a better photo.
5. Never invent information.
6. Be friendly, helpful, and accurate.
"""

sessions = {}
locks = {}
available_models = []
active_model = None


# ==========================
# RENDER HTTP SERVER
# ==========================

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header(
            "Content-Type", "text/plain"
        )
        self.end_headers()
        self.wfile.write(b"Telegram AI Bot Running")

    def log_message(self, *args):
        pass


def run_health_server():
    port = int(os.getenv("PORT", "10000"))
    server = HTTPServer(
        ("0.0.0.0", port), HealthHandler
    )
    logging.info("HTTP server on port %s", port)
    server.serve_forever()


# ==========================
# GEMINI MODEL DISCOVERY
# ==========================

def discover_models():
    global available_models

    names = []

    for model in client.models.list():
        actions = getattr(
            model, "supported_actions", None
        )

        if actions is not None:
            if "generateContent" not in actions:
                continue

        name = model.name.replace("models/", "")

        # Prefer general-purpose text/image models
        if (
            "gemini" in name.lower()
            and "embedding" not in name.lower()
            and "tts" not in name.lower()
            and "image" not in name.lower()
            and "live" not in name.lower()
        ):
            names.append(name)

    def priority(name):
        if name == PREFERRED_MODEL:
            return (0, name)
        if "flash-lite" in name:
            return (1, name)
        if "flash" in name:
            return (2, name)
        return (3, name)

    available_models = sorted(
        set(names), key=priority
    )

    logging.info(
        "Available models: %s", available_models
    )

    return available_models


def get_lock(user_id):
    if user_id not in locks:
        locks[user_id] = asyncio.Lock()
    return locks[user_id]


def create_chat(model):
    return client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        )
    )


def is_model_not_found(error):
    text = str(error).lower()
    return (
        "404" in text
        or "not_found" in text
        or "not found" in text
    )


def ask_ai(user_id, question, image_bytes=None):
    global active_model

    if not available_models:
        raise RuntimeError(
            "No Gemini generation models available"
        )

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

    # Keep the user's existing conversation
    saved = sessions.get(user_id)

    candidates = list(available_models)

    if saved:
        old_model = saved["model"]
        if old_model in candidates:
            candidates.remove(old_model)
            candidates.insert(0, old_model)
    elif active_model in candidates:
        candidates.remove(active_model)
        candidates.insert(0, active_model)

    last_error = None

    for model in candidates:
        try:
            if saved and saved["model"] == model:
                chat = saved["chat"]
            else:
                chat = create_chat(model)

            response = chat.send_message(content)

            sessions[user_id] = {
                "model": model,
                "chat": chat
            }

            active_model = model

            return (
                response.text
                or "⚠️ AI did not return a text answer."
            )

        except Exception as error:
            last_error = error

            if is_model_not_found(error):
                logging.warning(
                    "Model %s unavailable: %s",
                    model, error
                )
                continue

            # Do not retry quota/key/network errors
            # on other models automatically.
            raise

    raise RuntimeError(
        "All discovered models failed: "
        + str(last_error)
    )


def error_message(error):
    text = str(error).lower()

    if "429" in text:
        return (
            "⚠️ Gemini API quota exceeded.\n"
            "សូមរង់ចាំ ឬពិនិត្យ API quota។"
        )

    if "403" in text or "api key" in text:
        return (
            "⚠️ API Key ឬ Permission មានបញ្ហា។"
        )

    if (
        "404" in text
        or "no gemini" in text
        or "all discovered models" in text
    ):
        return (
            "⚠️ មិនអាចប្រើ Gemini Model បាន។\n"
            "សូមពិនិត្យ Render Logs។"
        )

    return (
        "❌ AI មានបញ្ហា។\n"
        "សូមពិនិត្យ Render Logs។"
    )


async def send_answer(message, answer):
    for i in range(0, len(answer), 4000):
        await message.reply_text(
            answer[i:i + 4000]
        )


# ==========================
# BOT COMMANDS
# ==========================

async def start(update: Update, context):
    await update.message.reply_text(
        "🤖 សួស្តី! ខ្ញុំជា Gemini AI Bot!\n\n"
        "📸 ផ្ញើរូបភាពលំហាត់\n"
        "🧮 ដោះស្រាយគណិតវិទ្យា\n"
        "📚 ជួយការសិក្សា\n"
        "💻 Programming\n"
        "🌍 បកប្រែភាសា\n"
        "🧠 សន្ទនាជាមួយ AI\n\n"
        "/models - Show Gemini models\n"
        "/reset - Clear conversation\n"
        "/help - Help"
    )


async def reset(update: Update, context):
    user_id = update.effective_user.id

    async with get_lock(user_id):
        sessions.pop(user_id, None)

    await update.message.reply_text(
        "✅ បានសម្អាតការសន្ទនារួចរាល់!"
    )


async def help_command(update: Update, context):
    await update.message.reply_text(
        "🤖 Gemini AI Assistant\n\n"
        "Send text or photos.\n"
        "I can explain homework in Khmer.\n\n"
        "/models - Available models\n"
        "/reset - Clear conversation"
    )


async def models_command(update: Update, context):
    try:
        models = await asyncio.to_thread(
            discover_models
        )

        if not models:
            await update.message.reply_text(
                "⚠️ No models found."
            )
            return

        names = "\n".join(
            "• " + name for name in models[:15]
        )

        await update.message.reply_text(
            "🤖 Available Gemini Models:\n\n"
            + names
            + "\n\nSelected: "
            + str(active_model or models[0])
        )

    except Exception as error:
        logging.exception("Model list error")
        await update.message.reply_text(
            error_message(error)
        )


# ==========================
# TEXT MESSAGE
# ==========================

async def handle_text(update: Update, context):
    user_id = update.effective_user.id
    question = update.message.text

    async with get_lock(user_id):
        try:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.TYPING
            )

            answer = await asyncio.to_thread(
                ask_ai, user_id, question
            )

        except Exception as error:
            logging.exception("Text AI error")
            answer = error_message(error)

        await send_answer(update.message, answer)


# ==========================
# PHOTO MESSAGE
# ==========================

async def handle_photo(update: Update, context):
    user_id = update.effective_user.id

    question = (
        update.message.caption
        or "សូមមើលរូបភាពនេះ និងដោះស្រាយ"
           "លំហាត់ជាជំហានៗជាភាសាខ្មែរ។"
    )

    async with get_lock(user_id):
        try:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.TYPING
            )

            photo = update.message.photo[-1]
            file = await photo.get_file()

            data = await file.download_as_bytearray()
            image_bytes = bytes(data)

            answer = await asyncio.to_thread(
                ask_ai,
                user_id,
                question,
                image_bytes
            )

        except Exception as error:
            logging.exception("Photo AI error")
            answer = error_message(error)

        await send_answer(update.message, answer)


# ==========================
# TELEGRAM ERROR HANDLER
# ==========================

async def error_handler(update, context):
    logging.error(
        "Telegram error: %s",
        context.error,
        exc_info=context.error
    )


# ==========================
# MAIN
# ==========================

def main():
    try:
        models = discover_models()

        if not models:
            logging.warning(
                "No Gemini generation models found"
            )

    except Exception:
        logging.exception(
            "Failed to discover Gemini models"
        )

    threading.Thread(
        target=run_health_server,
        daemon=True
    ).start()

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(
        CommandHandler("models", models_command)
    )

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

    logging.info("Telegram Gemini Bot starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
