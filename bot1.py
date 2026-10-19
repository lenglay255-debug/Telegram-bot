
import os
import re
import asyncio
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from google import genai
from google.genai import types

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import Conflict, NetworkError, TimedOut
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)


# ============================================
# CONFIGURATION
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Prevent full Telegram Bot URLs (including token)
# from appearing in ordinary HTTP logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

PREFERRED_MODEL = (
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    .strip()
    .removeprefix("models/")
)

if not TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN is missing")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

# A Telegram token must not contain whitespace.
if any(character.isspace() for character in TOKEN):
    raise RuntimeError(
        "Invalid TELEGRAM_TOKEN: contains whitespace "
        "or newline. Fix Render Environment."
    )

if not re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+", TOKEN):
    raise RuntimeError(
        "Invalid TELEGRAM_TOKEN format. "
        "Get a fresh token from BotFather."
    )


# ============================================
# GEMINI SETTINGS
# ============================================

# Strict allowlist: never select image generation
# models such as gemini-3-pro-image.
SAFE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

if PREFERRED_MODEL not in SAFE_MODELS:
    logging.warning(
        "GEMINI_MODEL %s is not allowed. "
        "Using gemini-2.5-flash.",
        PREFERRED_MODEL
    )
    PREFERRED_MODEL = SAFE_MODELS[0]

MODEL_ORDER = [
    PREFERRED_MODEL,
    *[m for m in SAFE_MODELS if m != PREFERRED_MODEL]
]

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(
        timeout=60000
    )
)

available_models = MODEL_ORDER.copy()
sessions = {}
user_locks = {}
last_successful_model = None

# Avoid retrying models known to be temporarily
# unavailable or over quota.
model_cooldowns = {}

GEMINI_TIMEOUT_SECONDS = 75


# ============================================
# AI SYSTEM PROMPT
# ============================================

SYSTEM_PROMPT = """
You are an intelligent, careful AI tutor.

You understand Khmer and English.
Answer in the language of the user.

Your main tasks:
- Solve mathematics accurately.
- Explain physics and chemistry.
- Analyze homework photographs.
- Help with programming.
- Answer general questions.
- Explain difficult concepts clearly.

For mathematics:
1. Read the problem carefully.
2. Identify known information.
3. Select the correct method.
4. Show useful step-by-step calculations.
5. Check the result before answering.
6. Give a clear final answer.

For image questions:
- Read the image carefully.
- Do not invent unreadable information.
- Ask for a clearer picture if necessary.
- Explain the solution in Khmer when
  the user speaks Khmer.

IMPORTANT TELEGRAM FORMATTING:

Telegram plain text does not render LaTeX.

Never write LaTeX commands such as:
\\frac, \\sqrt, \\boxed, \\begin, \\end.

Do not use dollar-sign math delimiters.

Use readable Unicode mathematics:
x² + 5x + 6 = 0
√25 = 5
(a + b) / (c + d)
x ≥ 0
π ≈ 3.14159

Keep important equations on separate lines.

Provide accurate explanations rather than
unnecessary lengthy responses.

Do not reveal hidden internal reasoning.
Provide clear solution steps and verification.
"""


# ============================================
# RENDER HEALTH SERVER
# ============================================

class HealthHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        # HTTP liveness means the Python process
        # and web thread are running, not that
        # Telegram or Gemini is healthy.
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "text/plain; charset=utf-8"
        )
        self.end_headers()
        self.wfile.write(b"Bot process alive")

    def log_message(self, format, *args):
        pass


def start_health_server():
    port = int(os.getenv("PORT", "10000"))

    server = ThreadingHTTPServer(
        ("0.0.0.0", port),
        HealthHandler
    )

    logging.info(
        "Render health server listening on %s",
        port
    )

    server.serve_forever()


# ============================================
# MODEL DISCOVERY
# ============================================

def discover_models():
    global available_models

    try:
        discovered = set()

        for item in client.models.list():
            name = (
                getattr(item, "name", "") or ""
            ).removeprefix("models/")

            actions = getattr(
                item, "supported_actions", None
            )

            if actions is not None:
                if "generateContent" not in actions:
                    continue

            if name in SAFE_MODELS:
                discovered.add(name)

        result = [
            name for name in MODEL_ORDER
            if name in discovered
        ]

        if result:
            available_models = result
        else:
            logging.warning(
                "Model list returned no allowed models. "
                "Keeping configured candidates."
            )

    except Exception:
        logging.exception(
            "Model discovery failed; "
            "using configured candidates"
        )

    logging.info(
        "Gemini candidates: %s",
        available_models
    )

    return available_models


# ============================================
# SESSION MANAGEMENT
# ============================================

def get_user_lock(user_id):
    if user_id not in user_locks:
        user_locks[user_id] = asyncio.Lock()

    return user_locks[user_id]


def create_chat(model):
    return client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=4096
        )
    )


def get_error_code(error):
    code = getattr(error, "code", None)

    try:
        if code is not None:
            return int(code)
    except (ValueError, TypeError):
        pass

    message = str(error).lower()

    if "429" in message or "resource_exhausted" in message:
        return 429

    if "404" in message or "not_found" in message:
        return 404

    if "503" in message or "unavailable" in message:
        return 503

    if "500" in message or "internal" in message:
        return 500

    return None


# ============================================
# GEMINI REQUEST
# ============================================

def ask_gemini(user_id, question, image_bytes=None):
    global last_successful_model

    content = question

    if image_bytes is not None:
        content = [
            types.Part.from_text(text=question),
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        ]

    saved = sessions.get(user_id)

    candidates = list(available_models)

    if saved and saved["model"] in candidates:
        selected = saved["model"]
        candidates.remove(selected)
        candidates.insert(0, selected)

    last_error = None
    attempted = 0

    for model in candidates:

        # Skip temporarily blocked models.
        if time.monotonic() < model_cooldowns.get(model, 0):
            continue

        attempted += 1

        try:
            logging.info(
                "Gemini request started | user=%s | model=%s",
                user_id,
                model
            )

            if saved and saved["model"] == model:
                chat = saved["chat"]
            else:
                chat = create_chat(model)

            response = chat.send_message(content)

            answer = response.text

            if not answer:
                answer = (
                    "AI មិនបានផ្តល់ចម្លើយទេ។ "
                    "សូមសាកល្បងសំណួរម្តងទៀត។"
                )

            sessions[user_id] = {
                "model": model,
                "chat": chat
            }

            last_successful_model = model

            logging.info(
                "Gemini request succeeded | model=%s",
                model
            )

            return answer

        except Exception as error:
            last_error = error
            code = get_error_code(error)

            logging.warning(
                "Gemini failed | model=%s | code=%s",
                model,
                code
            )

            if code == 429:
                # Avoid repeatedly hitting a quota limit.
                model_cooldowns[model] = (
                    time.monotonic() + 300
                )
                continue

            if code == 404:
                model_cooldowns[model] = (
                    time.monotonic() + 3600
                )
                continue

            if code in (500, 503):
                model_cooldowns[model] = (
                    time.monotonic() + 30
                )
                continue

            raise

    if attempted == 0:
        raise RuntimeError(
            "ALL_MODELS_COOLDOWN"
        )

    if last_error:
        raise last_error

    raise RuntimeError(
        "No Gemini model available"
    )


# ============================================
# CLEAN ANSWERS
# ============================================

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

    # Remove math delimiters.
    text = text.replace("$$", "")
    text = text.replace("$", "")

    for item in (
        r"\(",
        r"\)",
        r"\[",
        r"\]"
    ):
        text = text.replace(item, "")

    # Fractions.
    fraction_pattern = re.compile(
        r"\\(?:frac|dfrac|tfrac)"
        r"\{([^{}]+)\}\{([^{}]+)\}"
    )

    for _ in range(8):
        new_text = fraction_pattern.sub(
            lambda m: (
                "(" + m.group(1) + ")/("
                + m.group(2) + ")"
            ),
            text
        )

        if new_text == text:
            break

        text = new_text

    # Square roots.
    text = re.sub(
        r"\\sqrt\{([^{}]+)\}",
        lambda m: "√(" + m.group(1) + ")",
        text
    )

    replacements = {
        r"\times": "×",
        r"\cdot": "·",
        r"\div": "÷",
        r"\pm": "±",
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

    return text.strip()


# ============================================
# TELEGRAM RESPONSE
# ============================================

async def send_answer(message, answer):
    text = clean_answer(answer)

    if not text:
        text = "AI មិនមានចម្លើយទេ។"

    while text:
        if len(text) <= 3900:
            await message.reply_text(text)
            return

        split_at = text.rfind("\n", 0, 3900)

        if split_at < 1500:
            split_at = 3900

        await message.reply_text(
            text[:split_at].strip()
        )

        text = text[split_at:].strip()


# ============================================
# FRIENDLY ERROR MESSAGES
# ============================================

def friendly_error(error):
    message = str(error).lower()
    code = get_error_code(error)

    if "all_models_cooldown" in message:
        return (
            "⚠️ Gemini Models កំពុងមាន "
            "Rate Limit ឬ Quota។\n"
            "សូមរង់ចាំបន្តិច ហើយសាកល្បងវិញ។"
        )

    if code == 429:
        return (
            "⚠️ Gemini API អស់ Quota "
            "ឬលើស Rate Limit។\n"
            "សូមពិនិត្យ Google AI Studio។"
        )

    if code == 404:
        return (
            "⚠️ Gemini Model មិនអាចប្រើបាន។\n"
            "សូមវាយ /models។"
        )

    if code in (500, 503):
        return (
            "⚠️ Gemini Server កំពុងមានបញ្ហា។\n"
            "សូមសាកល្បងម្តងទៀត។"
        )

    if isinstance(error, asyncio.TimeoutError):
        return (
            "⏳ Gemini ឆ្លើយយឺតពេក។\n"
            "សូមសាកល្បងម្តងទៀត។"
        )

    return (
        "❌ មានបញ្ហាក្នុងការភ្ជាប់ AI។\n"
        "សូមពិនិត្យ Render Logs។"
    )


# ============================================
# TELEGRAM COMMANDS
# ============================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "🤖 សួស្តី! ខ្ញុំជា Gemini AI Tutor\n\n"
        "🧠 វិភាគ និងពន្យល់\n"
        "🧮 ដោះស្រាយគណិតវិទ្យា\n"
        "📸 អានរូបភាពលំហាត់\n"
        "📚 ពន្យល់ជាភាសាខ្មែរ\n"
        "💻 Programming\n\n"
        "/ping - ពិនិត្យ Bot\n"
        "/models - មើល Gemini Models\n"
        "/reset - ចាប់ផ្តើមសន្ទនាថ្មី\n"
        "/help - ជំនួយ"
    )


async def ping(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "🏓 Pong! Telegram Bot កំពុងដំណើរការ។"
    )


async def reset(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    user_id = update.effective_user.id

    async with get_user_lock(user_id):
        sessions.pop(user_id, None)

    await update.message.reply_text(
        "✅ បានចាប់ផ្តើមសន្ទនាថ្មី!"
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "📚 Gemini AI Tutor\n\n"
        "ផ្ញើសំណួរ ឬរូបភាពលំហាត់។\n"
        "Bot នឹងព្យាយាមពន្យល់ជំហានៗ។\n\n"
        "/ping - Check Telegram\n"
        "/models - Check Models\n"
        "/reset - Reset conversation"
    )


async def models_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    found = await asyncio.to_thread(
        discover_models
    )

    await update.message.reply_text(
        "🤖 Gemini Models:\n\n"
        + "\n".join("• " + m for m in found)
        + "\n\nPreferred: " + PREFERRED_MODEL
        + "\nLast successful: "
        + str(last_successful_model or "None")
    )


# ============================================
# TEXT MESSAGE HANDLER
# ============================================

async def handle_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    user_id = update.effective_user.id

    logging.info(
        "Text received | user=%s",
        user_id
    )

    async with get_user_lock(user_id):
        try:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.TYPING
            )

            answer = await asyncio.wait_for(
                asyncio.to_thread(
                    ask_gemini,
                    user_id,
                    update.message.text
                ),
                timeout=GEMINI_TIMEOUT_SECONDS
            )

        except Exception as error:
            logging.exception(
                "Text processing failed"
            )
            answer = friendly_error(error)

        await send_answer(update.message, answer)


# ============================================
# PHOTO MESSAGE HANDLER
# ============================================

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
            "និងពន្យល់ជំហានៗជាភាសាខ្មែរ។"
        )
    )

    logging.info(
        "Photo received | user=%s",
        user_id
    )

    async with get_user_lock(user_id):
        try:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.TYPING
            )

            photo = update.message.photo[-1]
            file = await photo.get_file()

            data = await file.download_as_bytearray()

            answer = await asyncio.wait_for(
                asyncio.to_thread(
                    ask_gemini,
                    user_id,
                    question,
                    bytes(data)
                ),
                timeout=GEMINI_TIMEOUT_SECONDS
            )

        except Exception as error:
            logging.exception(
                "Photo processing failed"
            )
            answer = friendly_error(error)

        await send_answer(update.message, answer)


# ============================================
# TELEGRAM ERROR HANDLER
# ============================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):
    error = context.error

    if isinstance(error, Conflict):
        logging.error(
            "Telegram 409 Conflict: "
            "Another bot instance may be using "
            "the same token."
        )

    elif isinstance(error, (NetworkError, TimedOut)):
        logging.error(
            "Telegram network problem: %s",
            type(error).__name__
        )

    else:
        logging.error(
            "Telegram update error: %s",
            type(error).__name__,
            exc_info=error
        )


# ============================================
# MAIN
# ============================================

def main():
    logging.info("Starting Gemini Telegram Bot")

    # Do not block startup on Gemini model discovery.
    threading.Thread(
        target=discover_models,
        daemon=True
    ).start()

    threading.Thread(
        target=start_health_server,
        daemon=True
    ).start()

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .connect_timeout(20)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(20)
        .get_updates_connect_timeout(20)
        .get_updates_read_timeout(30)
        .build()
    )

    app.add_handler(
        CommandHandler("start", start)
    )

    app.add_handler(
        CommandHandler("ping", ping)
    )

    app.add_handler(
        CommandHandler("reset", reset)
    )

    app.add_handler(
        CommandHandler("help", help_command)
    )

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

    try:
        logging.info(
            "Starting Telegram long polling"
        )

        app.run_polling(
            drop_pending_updates=True,
            poll_interval=1.0,
            timeout=10,
            bootstrap_retries=-1
        )

    except KeyboardInterrupt:
        logging.info("Bot interrupted")

    except Exception:
        logging.exception(
            "Telegram polling stopped unexpectedly"
        )
        raise

    finally:
        logging.warning(
            "Telegram application has stopped"
        )


if __name__ == "__main__":
    main()
