
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

# ==========================================
# CONFIGURATION
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

TOKEN = os.getenv("TELEGRAM_TOKEN")
API_KEY = os.getenv("GEMINI_API_KEY")

PREFERRED_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

if not TOKEN or not API_KEY:
    raise RuntimeError(
        "Please set TELEGRAM_TOKEN and GEMINI_API_KEY"
    )

client = genai.Client(api_key=API_KEY)

sessions = {}
user_locks = {}
available_models = []
active_model = None

# ==========================================
# INTELLIGENT AI SYSTEM PROMPT
# ==========================================

SYSTEM_PROMPT = """
You are an advanced, intelligent AI tutor.

Your main priority is:
ACCURACY, DEEP REASONING, LOGICAL THINKING,
AND STEP-BY-STEP PROBLEM SOLVING.

You are especially skilled at:
- Mathematics
- Physics
- Chemistry
- Biology
- Programming
- Logical reasoning
- Image-based homework
- Khmer education

LANGUAGE:
- Answer Khmer questions in natural Khmer.
- Answer English questions in English.
- Explain difficult ideas clearly.
- Use appropriate educational terminology.

REASONING INSTRUCTIONS:

1. Understand the problem carefully before answering.
2. Identify the given information and the goal.
3. Choose the correct mathematical or scientific method.
4. Work through the necessary calculations carefully.
5. Check calculations, signs, units, and assumptions.
6. Explain the important solution steps clearly.
7. Give the final answer separately.
8. If uncertain, say what is uncertain.
9. Never invent facts or missing values.
10. For difficult problems, prioritize correctness
    over speed or unnecessary verbosity.

MATHEMATICS FORMATTING:

IMPORTANT:
Telegram does not automatically render LaTeX.

Never use:
- Dollar signs around mathematics
- Double dollar signs
- LaTeX display delimiters
- \\frac
- \\sqrt
- \\begin
- \\end
- \\boxed
- \\left
- \\right
- \\displaystyle
- \\mathrm
- \\text

Instead use readable Unicode mathematics:

x² + 5x + 6 = 0
√25 = 5
a × b
a ÷ b
x ≤ 10
x ≥ 2
x ≠ 0
π ≈ 3.14159

For fractions, use:
(a + b) / (c + d)

For powers, use:
x², x³, x⁴ or x^(n)

For square roots, use:
√(x + 2)

For equations, put each important step
on its own line.

Example:

📚 លំហាត់៖
ដោះស្រាយសមីការ៖
x² - 5x + 6 = 0

✏️ ជំហានទី ១៖ បំបែកជាកត្តា

(x - 2)(x - 3) = 0

✏️ ជំហានទី ២៖

x - 2 = 0  ឬ  x - 3 = 0

ដូច្នេះ៖
x = 2  ឬ  x = 3

✅ ចម្លើយ៖ x = 2 ឬ x = 3

IMAGE ANALYSIS:

When a user sends a homework image:
- Read the image carefully.
- Identify the actual question.
- Do not guess unclear numbers or symbols.
- Solve the problem logically.
- Explain each important step.
- Verify the final result.
- If unreadable, request a clearer image.

PROGRAMMING:
- Provide correct, runnable code.
- Explain important logic when helpful.
- Help debug errors carefully.

Do not reveal private internal reasoning.
Instead, give a clear, useful explanation
of the solution steps and verification.

Be helpful, accurate, and patient.
"""

# ==========================================
# HEALTH SERVER FOR RENDER
# ==========================================

class HealthHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "text/plain; charset=utf-8"
        )
        self.end_headers()
        self.wfile.write(b"Gemini AI Bot is running")

    def log_message(self, *args):
        pass


def start_health_server():
    port = int(os.getenv("PORT", "10000"))

    server = HTTPServer(
        ("0.0.0.0", port),
        HealthHandler
    )

    logging.info("Health server started on %s", port)
    server.serve_forever()


# ==========================================
# GEMINI MODEL DISCOVERY
# ==========================================

def discover_models():
    global available_models

    found = []

    for item in client.models.list():

        name = getattr(item, "name", "") or ""

        if not name:
            continue

        name = name.replace("models/", "", 1)
        lower = name.lower()

        if not lower.startswith("gemini"):
            continue

        actions = getattr(
            item,
            "supported_actions",
            None
        )

        if actions is not None:
            if "generateContent" not in actions:
                continue

        excluded = [
            "embedding",
            "tts",
            "live",
            "image-generation",
            "native-audio",
            "computer-use",
        ]

        if any(x in lower for x in excluded):
            continue

        found.append(name)

    def model_priority(name):

        if name == PREFERRED_MODEL:
            return (0, name)

        if "pro" in name:
            return (1, name)

        if "flash" in name and "lite" not in name:
            return (2, name)

        if "flash-lite" in name:
            return (3, name)

        return (4, name)

    available_models = sorted(
        set(found),
        key=model_priority
    )

    logging.info(
        "Available models: %s",
        available_models
    )

    return available_models


# ==========================================
# USER SESSIONS
# ==========================================

def get_user_lock(user_id):

    if user_id not in user_locks:
        user_locks[user_id] = asyncio.Lock()

    return user_locks[user_id]


def create_chat(model):

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
    )

    return client.chats.create(
        model=model,
        config=config
    )


def is_model_not_found(error):

    message = str(error).lower()

    return (
        "404" in message
        or "not_found" in message
        or "not found" in message
    )


# ==========================================
# ASK GEMINI AI
# ==========================================

def ask_gemini(user_id, question, image_bytes=None):

    global active_model

    if not available_models:
        discover_models()

    if not available_models:
        raise RuntimeError(
            "No available Gemini models found"
        )

    if image_bytes is not None:

        content = [
            types.Part.from_text(
                text=question
            ),
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        ]

    else:
        content = question

    saved_session = sessions.get(user_id)

    candidates = list(available_models)

    if saved_session:
        saved_model = saved_session["model"]

        if saved_model in candidates:
            candidates.remove(saved_model)
            candidates.insert(0, saved_model)

    elif active_model in candidates:
        candidates.remove(active_model)
        candidates.insert(0, active_model)

    last_error = None

    for model in candidates:

        try:

            if (
                saved_session
                and saved_session["model"] == model
            ):
                chat = saved_session["chat"]
            else:
                chat = create_chat(model)

            response = chat.send_message(content)

            answer = response.text

            if not answer:
                answer = (
                    "សូមអភ័យទោស "
                    "AI មិនបានផ្តល់ចម្លើយទេ។"
                )

            sessions[user_id] = {
                "model": model,
                "chat": chat
            }

            active_model = model

            return answer

        except Exception as error:

            last_error = error

            if is_model_not_found(error):

                logging.warning(
                    "Model not available: %s",
                    model
                )

                continue

            raise

    raise RuntimeError(
        "All Gemini models failed: "
        + str(last_error)
    )


# ==========================================
# LATEX AND MATH CLEANER
# ==========================================

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
    "-": "⁻",
    "+": "⁺",
    "n": "ⁿ",
    "i": "ⁱ",
})


def replace_fractions(text):
    """
    Convert simple and nested LaTeX fractions.
    """

    pattern = re.compile(
        r"\\(?:dfrac|tfrac|frac)\s*"
        r"\{([^{}]*)\}\s*\{([^{}]*)\}"
    )

    for _ in range(10):

        new_text = pattern.sub(
            lambda match: (
                "(" + match.group(1) + ") / "
                "(" + match.group(2) + ")"
            ),
            text
        )

        if new_text == text:
            break

        text = new_text

    return text


def replace_roots(text):

    text = re.sub(
        r"\\sqrt\s*\[([^\]]+)\]\s*\{([^{}]+)\}",
        lambda m: (
            "(" + m.group(2) + ")^(1/"
            + m.group(1) + ")"
        ),
        text
    )

    text = re.sub(
        r"\\sqrt\s*\{([^{}]+)\}",
        lambda m: "√(" + m.group(1) + ")",
        text
    )

    return text


def replace_powers(text):

    def convert_power(match):

        power = match.group(1)

        if all(
            character in SUPERSCRIPTS
            for character in power
        ):
            return power.translate(SUPERSCRIPTS)

        return "^(" + power + ")"

    text = re.sub(
        r"\^\{([^{}]+)\}",
        convert_power,
        text
    )

    text = re.sub(
        r"\^([0-9]+)",
        lambda m: m.group(1).translate(SUPERSCRIPTS),
        text
    )

    return text


def clean_answer(text):

    if not text:
        return ""

    # Remove math delimiters
    text = text.replace("$$", "")
    text = text.replace("$", "")

    text = text.replace(r"\(", "")
    text = text.replace(r"\)", "")
    text = text.replace(r"\[", "")
    text = text.replace(r"\]", "")

    # Remove common LaTeX formatting commands
    for command in [
        r"\displaystyle",
        r"\textstyle",
        r"\left",
        r"\right",
        r"\!",
        r"\,",
        r"\;",
        r"\quad",
        r"\qquad",
    ]:
        text = text.replace(command, " ")

    # Convert fractions and roots
    text = replace_fractions(text)
    text = replace_roots(text)

    # Replace math symbols
    replacements = {
        r"\times": "×",
        r"\cdot": "·",
        r"\div": "÷",
        r"\pm": "±",
        r"\mp": "∓",
        r"\pi": "π",
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\delta": "δ",
        r"\theta": "θ",
        r"\lambda": "λ",
        r"\mu": "μ",
        r"\sigma": "σ",
        r"\omega": "ω",
        r"\infty": "∞",
        r"\leq": "≤",
        r"\le": "≤",
        r"\geq": "≥",
        r"\ge": "≥",
        r"\neq": "≠",
        r"\ne": "≠",
        r"\approx": "≈",
        r"\equiv": "≡",
        r"\rightarrow": "→",
        r"\Rightarrow": "⇒",
        r"\Leftrightarrow": "⇔",
        r"\in": "∈",
        r"\notin": "∉",
        r"\subseteq": "⊆",
        r"\cup": "∪",
        r"\cap": "∩",
        r"\sum": "Σ",
        r"\prod": "Π",
        r"\int": "∫",
        r"\partial": "∂",
        r"\nabla": "∇",
        r"\degree": "°",
    }

    # Longer commands first
    for old, new in sorted(
        replacements.items(),
        key=lambda x: len(x[0]),
        reverse=True
    ):
        text = text.replace(old, new)

    # Convert simple LaTeX text wrappers
    text = re.sub(
        r"\\(?:text|mathrm|mathbf|mathit|operatorname)"
        r"\{([^{}]*)\}",
        r"\1",
        text
    )

    # Convert powers
    text = replace_powers(text)

    # Remove remaining math delimiters
    text = text.replace("$$", "")
    text = text.replace("$", "")

    # Remove Markdown emphasis markers,
    # but preserve code blocks and underscores.
    text = text.replace("**", "")

    # Clean unnecessary whitespace
    text = re.sub(
        r"[ \t]{3,}",
        "  ",
        text
    )

    text = re.sub(
        r"\n{4,}",
        "\n\n\n",
        text
    )

    return text.strip()


# ==========================================
# SEND LONG TELEGRAM MESSAGES
# ==========================================

async def send_long_message(message, text):

    text = clean_answer(text)

    if not text:
        text = "AI មិនបានផ្តល់ចម្លើយទេ។"

    max_length = 3900

    while text:

        if len(text) <= max_length:
            await message.reply_text(text)
            break

        cut = text.rfind("\n", 0, max_length)

        if cut < max_length // 2:
            cut = max_length

        part = text[:cut].strip()

        if part:
            await message.reply_text(part)

        text = text[cut:].strip()


# ==========================================
# ERROR HANDLING
# ==========================================

def format_error(error):

    message = str(error).lower()

    if "429" in message:
        return (
            "⚠️ Gemini API អស់ Quota "
            "ឬមានសំណើច្រើនពេក។\n"
            "សូមរង់ចាំបន្តិច ហើយសាកល្បងម្តងទៀត។"
        )

    if "401" in message or "403" in message:
        return (
            "⚠️ Gemini API Key ឬ Permission "
            "មានបញ្ហា។"
        )

    if "404" in message or "model" in message:
        return (
            "⚠️ Gemini Model មិនអាចប្រើបាន។\n"
            "សូមពិនិត្យ /models។"
        )

    if "timeout" in message:
        return (
            "⚠️ AI ឆ្លើយយឺតពេក។ "
            "សូមសាកល្បងម្តងទៀត។"
        )

    return (
        "❌ មានបញ្ហាក្នុងការភ្ជាប់ Gemini AI។\n"
        "សូមពិនិត្យ Render Logs។"
    )


# ==========================================
# TELEGRAM COMMANDS
# ==========================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    text = (
        "🤖 សួស្តី! ខ្ញុំជា Gemini AI Tutor\n\n"
        "🧠 គិត និងវិភាគបញ្ហាស្មុគស្មាញ\n"
        "🧮 ដោះស្រាយគណិតវិទ្យា\n"
        "📸 អានរូបភាពលំហាត់\n"
        "📚 ពន្យល់ជាភាសាខ្មែរ\n"
        "🔬 រូបវិទ្យា និងគីមីវិទ្យា\n"
        "💻 Programming\n"
        "🌍 បកប្រែភាសា\n\n"
        "ផ្ញើសំណួរ ឬរូបភាពមកបាន!\n\n"
        "/reset - ចាប់ផ្តើមការសន្ទនាថ្មី\n"
        "/models - មើល Gemini Models\n"
        "/help - ជំនួយ"
    )

    await update.message.reply_text(text)


async def reset(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = update.effective_user.id

    async with get_user_lock(user_id):
        sessions.pop(user_id, None)

    await update.message.reply_text(
        "✅ បានសម្អាតប្រវត្តិសន្ទនារួចរាល់!"
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📚 Gemini AI Tutor\n\n"
        "សួរសំណួរគណិតវិទ្យា "
        "ឬផ្ញើរូបភាពលំហាត់។\n\n"
        "Bot នឹងព្យាយាមវិភាគ "
        "ដោះស្រាយជាជំហានៗ "
        "និងពិនិត្យចម្លើយ។\n\n"
        "/reset - New conversation\n"
        "/models - Available models"
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
                "⚠️ មិនមាន Gemini Model ដែលអាចប្រើបាន។"
            )
            return

        names = "\n".join(
            "• " + model
            for model in found[:20]
        )

        selected = active_model or found[0]

        await update.message.reply_text(
            "🤖 Available Gemini Models:\n\n"
            + names
            + "\n\nCurrently selected:\n"
            + selected
        )

    except Exception as error:

        logging.exception("Models command failed")

        await update.message.reply_text(
            format_error(error)
        )


# ==========================================
# TEXT MESSAGES
# ==========================================

async def handle_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = update.effective_user.id
    question = update.message.text

    async with get_user_lock(user_id):

        try:

            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.TYPING
            )

            answer = await asyncio.to_thread(
                ask_gemini,
                user_id,
                question
            )

        except Exception as error:

            logging.exception("Text message error")
            answer = format_error(error)

        await send_long_message(
            update.message,
            answer
        )


# ==========================================
# PHOTO MESSAGES
# ==========================================

async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = update.effective_user.id

    question = (
        update.message.caption
        or (
            "សូមអានលំហាត់ក្នុងរូបភាពនេះឱ្យបានច្បាស់។ "
            "វិភាគបញ្ហា ហើយដោះស្រាយជាជំហានៗ។ "
            "ពន្យល់ជាភាសាខ្មែរ "
            "និងពិនិត្យចម្លើយចុងក្រោយ។ "
            "សូមប្រើរូបមន្តងាយអាន "
            "ដោយគ្មានសញ្ញា LaTeX។"
        )
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

            image_bytes = bytes(data)

            answer = await asyncio.to_thread(
                ask_gemini,
                user_id,
                question,
                image_bytes
            )

        except Exception as error:

            logging.exception("Photo message error")
            answer = format_error(error)

        await send_long_message(
            update.message,
            answer
        )


# ==========================================
# GLOBAL ERROR HANDLER
# ==========================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):

    logging.error(
        "Telegram error: %s",
        context.error,
        exc_info=context.error
    )


# ==========================================
# MAIN
# ==========================================

def main():

    try:
        discover_models()

    except Exception:
        logging.exception(
            "Initial model discovery failed"
        )

    threading.Thread(
        target=start_health_server,
        daemon=True
    ).start()

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .build()
    )

    app.add_handler(
        CommandHandler("start", start)
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

    logging.info(
        "Gemini AI Tutor is starting..."
    )

    app.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()
