
import os
import re
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

# =========================================
# CONFIG
# =========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

TOKEN = os.getenv("TELEGRAM_TOKEN")
API_KEY = os.getenv("GEMINI_API_KEY")

PREFERRED_MODEL = os.getenv(
    "GEMINI_MODEL", "gemini-2.5-flash"
).replace("models/", "").strip()

if not TOKEN or not API_KEY:
    raise RuntimeError(
        "Missing TELEGRAM_TOKEN or GEMINI_API_KEY"
    )

client = genai.Client(api_key=API_KEY)

# Only text/multimodal reasoning models.
# Never select image-generation models.
FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

available_models = []
sessions = {}
user_locks = {}
active_model = None

# =========================================
# AI INSTRUCTIONS
# =========================================

SYSTEM_PROMPT = """
You are a highly capable Khmer AI tutor.

Your priorities are:
1. Correct reasoning and accurate answers.
2. Understanding the user's actual question.
3. Solving mathematics step by step.
4. Explaining clearly in Khmer when appropriate.
5. Checking results before presenting them.

You specialize in mathematics, physics,
chemistry, programming and logical reasoning.

For difficult problems:
- Identify the known information.
- Select an appropriate method.
- Work carefully through calculations.
- Check signs, units and assumptions.
- Explain the important steps.
- Give a clearly labeled final answer.
- Admit uncertainty rather than guessing.

For image-based homework:
- Read the problem carefully.
- Never invent unreadable numbers.
- Ask for a clearer image if necessary.
- Solve and explain the problem.

MATH FORMATTING FOR TELEGRAM:

Telegram does not render LaTeX.

Do not use dollar signs, LaTeX delimiters,
\\frac, \\sqrt, \\boxed, or LaTeX environments.

Use readable Unicode notation:
x² + 5x + 6 = 0
√25 = 5
(a + b) / (c + d)
x ≥ 0
π ≈ 3.14159
x × y

Put important equations on separate lines.
Use clear Khmer explanations.

For programming questions, provide working code.

Do not disclose hidden internal reasoning.
Give useful solution steps and verification.
"""

# =========================================
# RENDER HEALTH SERVER
# =========================================

class HealthHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header(
            "Content-Type", "text/plain"
        )
        self.end_headers()
        self.wfile.write(b"Gemini Bot Running")

    def log_message(self, *args):
        pass


def health_server():
    port = int(os.getenv("PORT", "10000"))
    server = HTTPServer(
        ("0.0.0.0", port),
        HealthHandler
    )
    server.serve_forever()


# =========================================
# SAFE MODEL SELECTION
# =========================================

def is_safe_model(name):
    name = name.lower()

    if not name.startswith("gemini-"):
        return False

    blocked = [
        "image",
        "imagen",
        "embedding",
        "tts",
        "audio",
        "live",
        "video",
        "veo",
        "computer-use",
        "robotics",
    ]

    if any(word in name for word in blocked):
        return False

    # Explicit allowlist avoids choosing
    # unsupported or image-generation models.
    return name in (
        PREFERRED_MODEL.lower(),
        *[m.lower() for m in FALLBACK_MODELS]
    )


def discover_models():
    global available_models

    found = []

    for item in client.models.list():
        name = (
            getattr(item, "name", "") or ""
        ).replace("models/", "", 1)

        actions = getattr(
            item, "supported_actions", None
        )

        if actions is not None:
            if "generateContent" not in actions:
                continue

        if is_safe_model(name):
            found.append(name)

    preferred_order = [
        PREFERRED_MODEL,
        *FALLBACK_MODELS
    ]

    result = []

    for model in preferred_order:
        if model in found and model not in result:
            result.append(model)

    available_models = result

    logging.info(
        "Safe Gemini models: %s",
        available_models
    )

    return available_models


# =========================================
# USER SESSIONS
# =========================================

def get_lock(user_id):
    if user_id not in user_locks:
        user_locks[user_id] = asyncio.Lock()
    return user_locks[user_id]


def create_chat(model):
    return client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2
        )
    )


def error_code(error):
    code = getattr(error, "code", None)

    try:
        if code is not None:
            return int(code)
    except (TypeError, ValueError):
        pass

    message = str(error).lower()

    if "429" in message or "resource_exhausted" in message:
        return 429

    if "404" in message or "not_found" in message:
        return 404

    return None


# =========================================
# GEMINI AI REQUEST
# =========================================

def ask_ai(user_id, question, image_bytes=None):
    global active_model

    if not available_models:
        discover_models()

    if not available_models:
        raise RuntimeError(
            "No supported Gemini text models found"
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

    saved = sessions.get(user_id)
    candidates = list(available_models)

    if saved and saved["model"] in candidates:
        chosen = saved["model"]
        candidates.remove(chosen)
        candidates.insert(0, chosen)

    last_error = None

    for model in candidates:
        try:
            if saved and saved["model"] == model:
                chat = saved["chat"]
            else:
                chat = create_chat(model)

            response = chat.send_message(content)

            answer = (
                response.text
                or "AI មិនបានផ្តល់ចម្លើយទេ។"
            )

            sessions[user_id] = {
                "model": model,
                "chat": chat
            }

            active_model = model

            return answer

        except Exception as error:
            last_error = error
            code = error_code(error)

            if code in (404, 429):
                logging.warning(
                    "Model %s failed with %s",
                    model,
                    code
                )
                continue

            raise

    raise RuntimeError(
        "No model currently available. "
        + str(last_error)
    )


# =========================================
# CLEAN MATHEMATICS
# =========================================

SUPERSCRIPTS = str.maketrans({
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "n": "ⁿ",
})


def clean_answer(text):
    if not text:
        return ""

    text = text.replace("$$", "")
    text = text.replace("$", "")

    for delimiter in (
        r"\(",
        r"\)",
        r"\[",
        r"\]",
    ):
        text = text.replace(delimiter, "")

    # Convert fractions, including nested ones
    fraction_pattern = re.compile(
        r"\\(?:frac|dfrac|tfrac)"
        r"\{([^{}]*)\}\{([^{}]*)\}"
    )

    for _ in range(10):
        updated = fraction_pattern.sub(
            lambda m: (
                "(" + m.group(1) + ")/("
                + m.group(2) + ")"
            ),
            text
        )
        if updated == text:
            break
        text = updated

    # Convert simple square roots
    text = re.sub(
        r"\\sqrt\{([^{}]+)\}",
        lambda m: "√(" + m.group(1) + ")",
        text
    )

    # Convert common math symbols
    replacements = {
        r"\times": "×",
        r"\cdot": "·",
        r"\div": "÷",
        r"\pm": "±",
        r"\mp": "∓",
        r"\pi": "π",
        r"\alpha": "α",
        r"\beta": "β",
        r"\theta": "θ",
        r"\infty": "∞",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\neq": "≠",
        r"\approx": "≈",
        r"\rightarrow": "→",
        r"\Rightarrow": "⇒",
        r"\sum": "Σ",
        r"\int": "∫",
    }

    for old, new in sorted(
        replacements.items(),
        key=lambda x: len(x[0]),
        reverse=True
    ):
        text = text.replace(old, new)

    text = re.sub(
        r"\\(?:text|mathrm|mathbf|boxed)"
        r"\{([^{}]*)\}",
        r"\1",
        text
    )

    for command in (
        r"\displaystyle",
        r"\left",
        r"\right",
        r"\quad",
        r"\qquad",
        r"\,",
        r"\;",
        r"\!",
    ):
        text = text.replace(command, " ")

    text = re.sub(
        r"\^\{([0-9n+-]+)\}",
        lambda m: m.group(1).translate(
            SUPERSCRIPTS
        ),
        text
    )

    text = re.sub(
        r"\^([0-9]+)",
        lambda m: m.group(1).translate(
            SUPERSCRIPTS
        ),
        text
    )

    text = text.replace("**", "")
    text = text.replace("$$", "")
    text = text.replace("$", "")

    return text.strip()


# =========================================
# TELEGRAM MESSAGE SENDING
# =========================================

async def send_answer(message, answer):
    text = clean_answer(answer)

    if not text:
        text = "AI មិនបានផ្តល់ចម្លើយទេ។"

    while text:
        if len(text) <= 3900:
            await message.reply_text(text)
            return

        cut = text.rfind("\n", 0, 3900)

        if cut < 1800:
            cut = 3900

        await message.reply_text(
            text[:cut].strip()
        )

        text = text[cut:].strip()


# =========================================
# ERROR MESSAGES
# =========================================

def format_error(error):
    message = str(error).lower()

    if (
        "429" in message
        or "resource_exhausted" in message
    ):
        return (
            "⚠️ Gemini API Quota មិនគ្រប់គ្រាន់។\n"
            "Model ដែលអនុញ្ញាតទាំងអស់ "
            "មិនអាចឆ្លើយបាននៅពេលនេះ។\n"
            "សូមពិនិត្យ Google AI Studio Quota។"
        )

    if "404" in message:
        return (
            "⚠️ Gemini Model មិនអាចប្រើបាន។\n"
            "សូមវាយ /models ដើម្បីពិនិត្យ។"
        )

    if "403" in message or "401" in message:
        return (
            "⚠️ Gemini API Key ឬ Permission "
            "មានបញ្ហា។"
        )

    if "no supported gemini" in message:
        return (
            "⚠️ មិនមាន Gemini Model "
            "ដែលអាចប្រើបានក្នុងបញ្ជីសុវត្ថិភាព។\n"
            "សូមពិនិត្យ /models និង API Key។"
        )

    return (
        "❌ AI មានបញ្ហាក្នុងការឆ្លើយ។\n"
        "សូមពិនិត្យ Render Logs។"
    )


# =========================================
# COMMANDS
# =========================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "🤖 សួស្តី! ខ្ញុំជា Gemini AI Tutor\n\n"
        "🧠 គិត និងវិភាគ\n"
        "🧮 ដោះស្រាយគណិតវិទ្យា\n"
        "📸 អានរូបភាពលំហាត់\n"
        "📚 ពន្យល់ជាភាសាខ្មែរ\n"
        "💻 Programming\n\n"
        "/reset - ចាប់ផ្តើមថ្មី\n"
        "/models - មើល Models\n"
        "/help - ជំនួយ"
    )


async def reset(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    user_id = update.effective_user.id

    async with get_lock(user_id):
        sessions.pop(user_id, None)

    await update.message.reply_text(
        "✅ បានចាប់ផ្តើមការសន្ទនាថ្មី!"
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "ផ្ញើសំណួរ ឬរូបភាពលំហាត់មកបាន។\n"
        "Bot នឹងព្យាយាមដោះស្រាយ "
        "និងពន្យល់ជាជំហានៗ។\n\n"
        "/reset - Clear conversation\n"
        "/models - Check Gemini models"
    )


async def models_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    try:
        found = await asyncio.to_thread(
            discover_models
        )

        if not found:
            await update.message.reply_text(
                "⚠️ No supported models found."
            )
            return

        await update.message.reply_text(
            "🤖 Supported Gemini Models:\n\n"
            + "\n".join(
                "• " + model for model in found
            )
            + "\n\nPreferred: "
            + PREFERRED_MODEL
            + "\nLast successful: "
            + str(active_model or "None")
        )

    except Exception as error:
        logging.exception("Model list error")
        await update.message.reply_text(
            format_error(error)
        )


# =========================================
# TEXT HANDLER
# =========================================

async def handle_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    user_id = update.effective_user.id

    async with get_lock(user_id):
        try:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.TYPING
            )

            answer = await asyncio.to_thread(
                ask_ai,
                user_id,
                update.message.text
            )

        except Exception as error:
            logging.exception("Text request failed")
            answer = format_error(error)

        await send_answer(update.message, answer)


# =========================================
# PHOTO HANDLER
# =========================================

async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    user_id = update.effective_user.id

    question = (
        update.message.caption
        or (
            "សូមអានលំហាត់ក្នុងរូបភាពនេះ។ "
            "ដោះស្រាយដោយប្រុងប្រយ័ត្ន "
            "ពន្យល់ជាជំហានៗជាភាសាខ្មែរ "
            "ហើយពិនិត្យចម្លើយចុងក្រោយ។"
        )
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

            answer = await asyncio.to_thread(
                ask_ai,
                user_id,
                question,
                bytes(data)
            )

        except Exception as error:
            logging.exception("Photo request failed")
            answer = format_error(error)

        await send_answer(update.message, answer)


# =========================================
# GLOBAL ERROR HANDLER
# =========================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):
    logging.error(
        "Telegram error: %s",
        context.error,
        exc_info=context.error
    )


# =========================================
# MAIN
# =========================================

def main():
    try:
        discover_models()
    except Exception:
        logging.exception(
            "Startup model discovery failed"
        )

    threading.Thread(
        target=health_server,
        daemon=True
    ).start()

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("models", models_command))

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

    logging.info(
        "Starting Gemini AI Tutor..."
    )

    app.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()
