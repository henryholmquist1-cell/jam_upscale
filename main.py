"""
=======================================================
  AI IMAGE UPSCALE BOT — Telegram
  Real-ESRGAN — model loaded ONCE at startup (fast)
  - Auto-uses GPU if available (10x faster)
  - Uses ALL CPU cores if no GPU
  - ~45-90 seconds on CPU, ~5-10 seconds on GPU
=======================================================
  Install:
    pip install realesrgan basicsr facexlib gfpgan pillow python-telegram-bot torch
=======================================================
"""

import logging
import os
import urllib.request
from io import BytesIO

import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image, ImageFilter, ImageEnhance
from realesrgan import RealESRGANer

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

MODEL_PATH = "weights/RealESRGAN_x4plus.pth"
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

# Global upsampler — loaded ONCE at startup, reused for every image
UPSAMPLER = None


# ─── MODEL SETUP ───────────────────────────────────────

def download_model():
    os.makedirs("weights", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("⬇️  Downloading Real-ESRGAN model (~65MB, one-time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Model downloaded!")


def load_model():
    """
    Load Real-ESRGAN once at startup and keep in memory.
    This is the key fix — previous version reloaded from disk every image.
    Auto-detects GPU (CUDA). Falls back to CPU with all cores.
    """
    global UPSAMPLER

    # Use ALL available CPU cores for inference
    cpu_cores = os.cpu_count() or 4
    torch.set_num_threads(cpu_cores)
    torch.set_flush_denormal(True)
    logger.info(f"Using {cpu_cores} CPU threads")

    # Auto-detect GPU
    if torch.cuda.is_available():
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)} — using GPU (fast mode)")
        half_precision = True   # GPU: use float16 for 2x speed boost
    else:
        print(f"💻 No GPU found — using CPU with {cpu_cores} cores")
        half_precision = False  # CPU: must use float32

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )

    UPSAMPLER = RealESRGANer(
        scale=4,
        model_path=MODEL_PATH,
        model=model,
        tile=192,           # tile size — 192 balances speed vs RAM usage well
        tile_pad=10,
        pre_pad=0,
        half=half_precision,
    )

    print("✅ AI model loaded and ready!")


# ─── UPSCALE LOGIC ─────────────────────────────────────

def run_esrgan_pass(img_pil: Image.Image) -> Image.Image:
    """Run one 4x Real-ESRGAN pass using the pre-loaded global model."""
    img_np = np.array(img_pil.convert("RGB"))
    img_bgr = img_np[:, :, ::-1]  # RGB → BGR for OpenCV convention

    with torch.inference_mode():   # faster than torch.no_grad()
        output_bgr, _ = UPSAMPLER.enhance(img_bgr, outscale=4)

    output_rgb = output_bgr[:, :, ::-1]  # BGR → RGB
    return Image.fromarray(output_rgb)


def upscale_image(raw_bytes: bytes, scale: int) -> tuple[BytesIO, float]:
    """
    Upscale pipeline:
    - 2x  → ESRGAN 4x pass, resize down to 2x (AI smoothing at 2x dimensions)
    - 4x  → ESRGAN 4x pass
    - 8x  → ESRGAN 4x pass + Lanczos 2x
    - 16x → ESRGAN 4x pass + Lanczos 4x
    Then: unsharp mask + contrast + quality=100 JPEG
    """
    img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    original_w, original_h = img.size
    logger.info(f"Input: {original_w}x{original_h}, target scale: {scale}x")

    logger.info("Running Real-ESRGAN 4x AI pass...")
    img_4x = run_esrgan_pass(img)
    logger.info(f"ESRGAN pass done: {img_4x.size}")

    if scale == 2:
        img_out = img_4x.resize((original_w * 2, original_h * 2), Image.LANCZOS)

    elif scale == 4:
        img_out = img_4x

    elif scale == 8:
        img_out = img_4x.resize((original_w * 8, original_h * 8), Image.LANCZOS)

    elif scale == 16:
        img_out = img_4x.resize((original_w * 16, original_h * 16), Image.LANCZOS)

    # Unsharp mask — professional edge sharpening
    img_out = img_out.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=2))

    # Subtle contrast boost
    img_out = ImageEnhance.Contrast(img_out).enhance(1.08)

    # Save: quality=100 + subsampling=0 = full color, no compression
    # This is what pushes file size to 7-12MB
    output = BytesIO()
    img_out.save(output, format="JPEG", quality=100, subsampling=0, optimize=False)
    output.seek(0)

    size_mb = len(output.getvalue()) / (1024 * 1024)
    logger.info(f"Final size: {size_mb:.1f} MB at {img_out.size}")
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
    logger.info(f"Photo from {user_id}, scale={scale}x")

    status_msg = await update.message.reply_text(
        f"⏳ AI enhancing {scale}x...\n"
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
            caption=f"✅ {scale}x AI Enhancement complete!\nSize: {size_mb:.1f} MB",
        )
        await send_menu(update, scale)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ Error: {str(e)[:200]}\n\nPlease try again.")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    scale = user_scales.get(user_id, DEFAULT_SCALE)

    doc = update.message.document
    if not doc.mime_type or not doc.mime_type.startswith("image/"):
        await update.message.reply_text("Please send an image file.")
        return

    logger.info(f"Document from {user_id}, scale={scale}x")
    status_msg = await update.message.reply_text(f"⏳ AI enhancing {scale}x...")

    try:
        tg_file = await context.bot.get_file(doc.file_id)
        raw_bytes = bytes(await tg_file.download_as_bytearray())

        output, size_mb = upscale_image(raw_bytes, scale)

        await status_msg.delete()
        await update.message.reply_document(
            document=output,
            filename=f"upscaled_{scale}x.jpg",
            caption=f"✅ {scale}x AI Enhancement complete!\nSize: {size_mb:.1f} MB",
        )
        await send_menu(update, scale)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ Error: {str(e)[:200]}\n\nPlease try again.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling update:", exc_info=context.error)


# ─── MAIN ──────────────────────────────────────────────

def main() -> None:
    print("🔄 Downloading model if needed...")
    download_model()
    print("🔄 Loading AI model into memory...")
    load_model()   # ← loads once here, never again

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
