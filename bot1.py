
import os
import re
import io
import asyncio
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# ====================================
# SETTINGS
# ====================================

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

sessions = {}
locks = {}
models = []
active_model = None

# ====================================
# SYSTEM PROMPT
# ====================================

SYSTEM_PROMPT = r"""
You are a friendly, intelligent Khmer AI tutor.

You can:
- Understand Khmer and English
- Analyze photographs of homework
- Solve mathematics step by step
- Explain physics and chemistry
- Help with programming and translation
- Answer general questions

Always answer in the user's language.

IMPORTANT MATHEMATICS RULES:

For mathematics, produce a clear step-by-step
explanation and typeset equations.

Every mathematical equation that should appear
in the rendered image MUST be written on its
own line using this exact format:

MATH: <valid matplotlib mathtext expression>

Example:

📚 លំហាត់៖ ដោះស្រាយសមីការ

MATH: x^2-5x+6=0

✏️ ជំហានទី ១៖ បំបែកជាកត្តា

MATH: (x-2)(x-3)=0

✏️ ជំហានទី ២៖ រកតម្លៃ x

MATH: x=2 \quad \mathrm{or} \quad x=3

✅ ចម្លើយចុងក្រោយ៖

MATH: \boxed{x=2 \quad \mathrm{or} \quad x=3}

MATH RULES:
1. Use valid Matplotlib mathtext syntax.
2. Do not put $ delimiters in MATH lines.
3. Use \frac{a}{b} for fractions.
4. Use \sqrt{x} for square roots.
5. Use x^2 for powers.
6. Use \times for multiplication.
7. Use \pi for pi.
8. Use \leq and \geq for inequalities.
9. Keep Khmer text outside MATH lines.
10. Use only Latin characters and supported
    mathematical symbols in MATH lines.
11. Do not use \begin, \end, \text,
    \displaystyle, \dfrac, \left, or \right.
12. Do not use markdown code fences
    around mathematical equations.
13. Keep each equation reasonably short.
14. Verify calculations before answering.
15. Explain every important step in Khmer
    for Khmer-speaking users.

For ordinary questions, respond naturally.
For programming requests, provide useful code.
Do not use MATH lines unless equations are needed.

When analyzing images, read carefully.
If the image is unclear, ask for a clearer one.
Never invent missing information.
"""

# ====================================
# RENDER HEALTH SERVER
# ====================================

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header(
            "Content-Type", "text/plain"
        )
        self.end_headers()
        self.wfile.write(b"AI Bot Running")

    def log_message(self, *args):
        pass


def health_server():
    port = int(os.getenv("PORT", "10000"))
    server = HTTPServer(
        ("0.0.0.0", port),
        HealthHandler
    )
    logging.info("Health server on port %s", port)
    server.serve_forever()


# ====================================
# MODEL DISCOVERY
# ====================================

def discover_models():
    global models

    found = []

    for item in client.models.list():
        actions = getattr(
            item, "supported_actions", None
        )

        if actions is not None:
            if "generateContent" not in actions:
                continue

        name = item.name.replace("models/", "")
        lower = name.lower()

        if "gemini" not in lower:
            continue

        excluded = [
            "embedding",
            "tts",
            "live",
            "image-generation",
            "native-audio",
        ]

        if any(word in lower for word in excluded):
            continue

        found.append(name)

    def priority(name):
        if name == PREFERRED_MODEL:
            return (0, name)
        if "flash-lite" in name:
            return (1, name)
        if "flash" in name:
            return (2, name)
        return (3, name)

    models = sorted(set(found), key=priority)

    logging.info("Gemini models: %s", models)
    return models


# ====================================
# SESSION MANAGEMENT
# ====================================

def get_lock(user_id):
    if user_id not in locks:
        locks[user_id] = asyncio.Lock()
    return locks[user_id]


def create_chat(model):
    return client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2
        )
    )


def is_model_missing(error):
    message = str(error).lower()
    return (
        "404" in message
        or "not_found" in message
        or "not found" in message
    )


# ====================================
# ASK GEMINI
# ====================================

def ask_ai(user_id, question, image_bytes=None):
    global active_model

    if not models:
        discover_models()

    if not models:
        raise RuntimeError(
            "No Gemini models available"
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
    candidates = list(models)

    if saved and saved["model"] in candidates:
        selected = saved["model"]
        candidates.remove(selected)
        candidates.insert(0, selected)

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
                or "AI មិនបានផ្តល់ចម្លើយទេ។"
            )

        except Exception as error:
            last_error = error

            if is_model_missing(error):
                logging.warning(
                    "Model unavailable: %s", model
                )
                continue

            raise

    raise RuntimeError(
        "All models unavailable: "
        + str(last_error)
    )


# ====================================
# MATH EQUATION RENDERER
# ====================================

def render_math(equations):
    """
    Render equations as a clean PNG.
    Uses Matplotlib mathtext.
    """

    if not equations:
        return None

    equations = equations[:35]

    valid = []

    # Validate equations individually.
    for equation in equations:
        equation = equation.strip().strip("$")

        if not equation:
            continue

        if len(equation) > 250:
            continue

        try:
            test_fig = plt.figure(figsize=(8, 1))
            test_fig.text(
                0.05, 0.5,
                "$" + equation + "$",
                fontsize=16
            )

            buffer = io.BytesIO()
            test_fig.savefig(
                buffer,
                format="png",
                dpi=60
            )
            buffer.close()
            plt.close(test_fig)

            valid.append(equation)

        except Exception:
            logging.warning(
                "Invalid equation: %s", equation
            )
            plt.close("all")

    if not valid:
        return None

    # Divide into pages for long solutions.
    images = []

    for start in range(0, len(valid), 8):
        page = valid[start:start + 8]

        height = max(3, len(page) * 1.25 + 1)

        fig = plt.figure(
            figsize=(10, height),
            facecolor="white"
        )

        fig.text(
            0.07, 0.96,
            "Mathematics Solution",
            fontsize=19,
            weight="bold",
            color="#17365D",
            va="top"
        )

        for index, equation in enumerate(page):
            y = 0.86 - index * (0.76 / max(len(page), 1))

            try:
                fig.text(
                    0.09,
                    y,
                    "$" + equation + "$",
                    fontsize=21,
                    color="#172B4D",
                    va="center"
                )
            except Exception:
                logging.exception(
                    "Equation drawing failed"
                )

        output = io.BytesIO()

        try:
            fig.savefig(
                output,
                format="png",
                dpi=160,
                bbox_inches="tight",
                pad_inches=0.35,
                facecolor="white"
            )

            output.seek(0)
            output.name = "math_solution.png"
            images.append(output)

        finally:
            plt.close(fig)

    return images


# ====================================
# PARSE AI ANSWER
# ====================================

def parse_answer(answer):
    text_lines = []
    equations = []

    for line in answer.splitlines():
        stripped = line.strip()

        if stripped.upper().startswith("MATH:"):
            equation = stripped[5:].strip()

            if equation:
                equations.append(equation)

                # Keep a readable fallback.
                text_lines.append(
                    "🔢 រូបមន្តទី "
                    + str(len(equations))
                    + "៖ មើលក្នុងរូបភាព"
                )
        else:
            text_lines.append(line)

    explanation = "\n".join(text_lines).strip()

    explanation = explanation.replace("**", "")

    return explanation, equations


# ====================================
# SEND AI ANSWER
# ====================================

async def send_answer(message, answer):
    explanation, equations = parse_answer(answer)

    if explanation:
        for i in range(0, len(explanation), 3900):
            await message.reply_text(
                explanation[i:i + 3900]
            )

    if equations:
        try:
            images = await asyncio.to_thread(
                render_math, equations
            )

            if images:
                for img in images:
                    try:
                        await message.reply_photo(
                            photo=img,
                            caption="📐 រូបមន្តគណិតវិទ្យា"
                        )
                    finally:
                        img.close()
            else:
                # Fallback when rendering fails.
                await message.reply_text(
                    "⚠️ មិនអាចបង្កើតរូបភាពបាន។\n\n"
                    + "\n".join(equations)[:3500]
                )

        except Exception:
            logging.exception("Math rendering error")
            await message.reply_text(
                "⚠️ មិនអាចបង្ហាញរូបភាពបាន។\n\n"
                + "\n".join(equations)[:3500]
            )


# ====================================
# ERROR MESSAGES
# ====================================

def format_error(error):
    message = str(error).lower()

    if "429" in message:
        return (
            "⚠️ Gemini API អស់ Quota។\n"
            "សូមសាកល្បងពេលក្រោយ។"
        )

    if "403" in message or "api key" in message:
        return "⚠️ API Key ឬ Permission មានបញ្ហា។"

    if "404" in message or "model" in message:
        return (
            "⚠️ Gemini Model មិនអាចប្រើបាន។\n"
            "សូមពិនិត្យ /models។"
        )

    return (
        "❌ មានបញ្ហាក្នុងការភ្ជាប់ AI។\n"
        "សូមពិនិត្យ Render Logs។"
    )


# ====================================
# TELEGRAM COMMANDS
# ====================================

async def start(update: Update, context):
    await update.message.reply_text(
        "🤖 សួស្តី! ខ្ញុំជា Gemini AI Bot!\n\n"
        "📸 មើលរូបភាពលំហាត់\n"
        "🧮 ដោះស្រាយគណិតវិទ្យា\n"
        "✨ បង្កើតរូបភាពរូបមន្តស្អាត\n"
        "📚 ពន្យល់ជាភាសាខ្មែរ\n"
        "💻 Programming\n"
        "🌍 Translation\n\n"
        "ផ្ញើសំណួរ ឬរូបភាពមកបាន!\n\n"
        "/reset - Clear memory\n"
        "/models - Gemini models\n"
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
        "📚 AI Math Tutor\n\n"
        "ផ្ញើសំណួរគណិតវិទ្យា ឬរូបភាព។\n"
        "Bot នឹងពន្យល់ជាភាសាខ្មែរ "
        "និងបង្កើតរូបមន្តជារូបភាព។\n\n"
        "/reset - New conversation\n"
        "/models - Available models"
    )


async def models_command(update: Update, context):
    try:
        found = await asyncio.to_thread(
            discover_models
        )

        if not found:
            await update.message.reply_text(
                "⚠️ No models found."
            )
            return

        names = "\n".join(
            "• " + name for name in found[:15]
        )

        await update.message.reply_text(
            "🤖 Available Gemini Models:\n\n"
            + names
            + "\n\nSelected: "
            + str(active_model or found[0])
        )

    except Exception as error:
        logging.exception("Model list failed")
        await update.message.reply_text(
            format_error(error)
        )


# ====================================
# TEXT HANDLER
# ====================================

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
            logging.exception("Text error")
            answer = format_error(error)

        await send_answer(update.message, answer)


# ====================================
# PHOTO HANDLER
# ====================================

async def handle_photo(update: Update, context):
    user_id = update.effective_user.id

    question = (
        update.message.caption
        or "សូមអានលំហាត់ក្នុងរូបភាពនេះ "
           "ហើយដោះស្រាយជាជំហានៗ។ "
           "សូមបង្ហាញរូបមន្តគណិតវិទ្យា "
           "ជាទម្រង់ MATH: ដើម្បីបង្កើតរូបភាព។"
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
            logging.exception("Photo error")
            answer = format_error(error)

        await send_answer(update.message, answer)


# ====================================
# TELEGRAM ERROR HANDLER
# ====================================

async def error_handler(update, context):
    logging.error(
        "Telegram error: %s",
        context.error,
        exc_info=context.error
    )


# ====================================
# MAIN
# ====================================

def main():
    try:
        discover_models()
    except Exception:
        logging.exception(
            "Model discovery failed at startup"
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

    logging.info("AI Math Bot starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
