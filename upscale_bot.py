"""
=======================================================
  AI IMAGE UPSCALE BOT — Telegram
  Fast, maximum quality output
  - Lanczos upscaling (sharpest algorithm)
  - Unsharp masking (Photoshop-style edge sharpening)
  - JPEG quality=100, subsampling=0 (no color compression)
  - Result: 1MB input → 7-12MB output, noticeably sharper
=======================================================
  Install:
    pip install python-telegram-bot Pillow
=======================================================
"""

import logging
from io import BytesIO
from PIL import Image, ImageFilter, ImageEnhance

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ─── CONFIG ────────────────────────────────────────────
TOKEN = "8799884959:AAGfcqTBeEF0E6dkrzWHwvAcL6mYSGbJI7g"   # ← Paste your token here

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

user_scales: dict[int, int] = {}
SCALE_OPTIONS = [2, 4, 8, 16]
DEFAULT_SCALE = 4


# ─── UPSCALE LOGIC ─────────────────────────────────────

def upscale_image(raw_bytes: bytes, scale: int) -> tuple[BytesIO, float]:
    """
    Maximum quality upscale pipeline:
    1. Lanczos resize — sharpest available resampling filter
    2. Unsharp mask — professional edge sharpening (same as Photoshop)
    3. Slight contrast boost to make it pop
    4. Save JPEG quality=100 + subsampling=0
       → subsampling=0 means full color info kept per pixel (4:4:4)
       → This alone triples file size vs default JPEG settings
       → 1MB in → 7-12MB out depending on scale factor
    """
    img = Image.open(BytesIO(raw_bytes)).convert("RGB")

    original_w, original_h = img.size
    new_w = original_w * scale
    new_h = original_h * scale

    logger.info(f"Resizing {original_w}x{original_h} → {new_w}x{new_h}")

    # Step 1: Lanczos upscale (highest quality resampling)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Step 2: Unsharp mask — sharpens edges without touching flat areas
    # radius=2.5: how wide the sharpening halo is
    # percent=200: how strong (200 = very strong, noticeable sharpness boost)
    # threshold=2: only sharpen edges, not flat areas/noise
    img = img.filter(ImageFilter.UnsharpMask(radius=2.5, percent=200, threshold=2))

    # Step 3: Slight contrast boost (1.1 = subtle, makes it look crisper)
    img = ImageEnhance.Contrast(img).enhance(1.1)

    # Step 4: Save at absolute maximum JPEG quality
    # quality=100 → minimal compression
    # subsampling=0 → saves full RGB color per pixel (default is 4:2:0 which loses color detail)
    # These two settings together are what makes file size jump dramatically
    output = BytesIO()
    img.save(output, format="JPEG", quality=100, subsampling=0, optimize=False)
    output.seek(0)

    size_mb = len(output.getvalue()) / (1024 * 1024)
    logger.info(f"Output size: {size_mb:.1f} MB")
    return output, size_mb


# ─── UI ────────────────────────────────────────────────

def build_keyboard(selected: int) -> InlineKeyboardMarkup:
    buttons = []
    for s in SCALE_OPTIONS:
        label = f"✅ {s}x" if s == selected else f"{s}x"
        buttons.append(InlineKeyboardButton(label, callback_data=f"scale_{s}"))
    return InlineKeyboardMarkup([buttons])


def panel_text(scale: int) -> str:
    return (
        "🖼️ *Upscale Bot*\n\n"
        f"Current factor: *{scale}x*\n"
        "_Best balance of quality & speed_\n\n"
        "Choose a scaling factor and then send an image\\!\n"
        "💡 _Send as File for best quality_"
    )


async def send_menu(update: Update, scale: int) -> None:
    await update.message.reply_text(
        panel_text(scale),
        reply_markup=build_keyboard(scale),
        parse_mode="MarkdownV2",
    )


# ─── HANDLERS ──────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    scale = user_scales.get(user_id, DEFAULT_SCALE)
    await update.message.reply_text(
        panel_text(scale),
        reply_markup=build_keyboard(scale),
        parse_mode="MarkdownV2",
    )


async def scale_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    scale = int(query.data.split("_")[1])
    user_scales[user_id] = scale
    await query.edit_message_text(
        panel_text(scale),
        reply_markup=build_keyboard(scale),
        parse_mode="MarkdownV2",
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    scale = user_scales.get(user_id, DEFAULT_SCALE)
    logger.info(f"Photo received from {user_id}, scale={scale}x")

    status_msg = await update.message.reply_text(
        f"⏳ Upscaling {scale}x...\n"
        "💡 Tip: Send as File next time for best quality!"
    )

    try:
        photo = update.message.photo[-1]
        tg_file = await context.bot.get_file(photo.file_id)
        raw_bytes = bytes(await tg_file.download_as_bytearray())
        logger.info(f"Downloaded {len(raw_bytes)} bytes")

        output, size_mb = upscale_image(raw_bytes, scale)

        await status_msg.delete()
        await update.message.reply_document(
            document=output,
            filename=f"upscaled_{scale}x.jpg",
            caption=f"✅ {scale}x Upscaling complete!\nSize: {size_mb:.1f} MB",
        )
        await send_menu(update, scale)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        await status_msg.edit_text("❌ Something went wrong. Please try again.")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    scale = user_scales.get(user_id, DEFAULT_SCALE)

    doc = update.message.document
    if not doc.mime_type or not doc.mime_type.startswith("image/"):
        await update.message.reply_text("Please send an image file.")
        return

    logger.info(f"Document received from {user_id}, scale={scale}x")
    status_msg = await update.message.reply_text(f"⏳ Upscaling {scale}x...")

    try:
        tg_file = await context.bot.get_file(doc.file_id)
        raw_bytes = bytes(await tg_file.download_as_bytearray())
        logger.info(f"Downloaded {len(raw_bytes)} bytes")

        output, size_mb = upscale_image(raw_bytes, scale)

        await status_msg.delete()
        await update.message.reply_document(
            document=output,
            filename=f"upscaled_{scale}x.jpg",
            caption=f"✅ {scale}x Upscaling complete!\nSize: {size_mb:.1f} MB",
        )
        await send_menu(update, scale)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        await status_msg.edit_text("❌ Something went wrong. Please try again.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling update:", exc_info=context.error)


# ─── MAIN ──────────────────────────────────────────────

def main() -> None:
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(scale_callback, pattern=r"^scale_\d+$"))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))
    app.add_error_handler(error_handler)

    print("✅ Upscale Bot is running… Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()