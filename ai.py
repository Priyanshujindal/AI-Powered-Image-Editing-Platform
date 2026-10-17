"""GenAI & Image Processing integration for PixelPrompt AI.

Three Processing Engines:
  1. DIRECT PHOTO ENGINE (PIL, OpenCV, Rembg / U2-Net)
     Modifies the ACTUAL uploaded photo/UI pixels directly, with zero hallucinations:
       - Light Mode / Dark Mode UI transformation (LAB-space luminosity inversion)
       - Invert colors
       - Brightness & Contrast adjustments
       - Background removal (White studio backdrop or Transparent PNG)
       - Portrait Mode Bokeh (AI subject isolation + Gaussian background blur)
       - Color & Lighting: B&W, Vibrant Pop, Vintage Film, Golden Hour, Cyberpunk Neon, Sepia, Enhance
       - Artistic Transforms: Pencil Sketch, Comic / Cartoon
       - Geometry: Rotate 90°, Flip Horizontal (Mirror)
     Speed: ~0.1 - 0.3s (instant local execution).

  2. INTELLIGENT AI PLANNER (Gemini 3.8 Flash)
     Understands natural language requests (e.g. "make it to the light mode",
     "blur the background", "darken the scene", "sketch effect") and maps them
     directly to pixel operations on your actual photo.

  3. GEMINI NATIVE IMAGE EDITOR (gemini-3.1-flash-image)
     Used for edits requiring content-level understanding (text changes, object manipulation,
     creative additions) - sends the actual image + prompt to Gemini for AI editing.
"""
import io
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from google import genai
from google.genai import types

import prompts

TEXT_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]
TEXT_MODEL = TEXT_MODELS[0]  # Default for caption/embed (non-critical)
EMBED_MODEL = "gemini-embedding-001"

# Lazy-loaded fast rembg session (u2netp is ~4.5MB, ultra-fast ~0.15s)
_rembg_session = None


def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        try:
            from rembg import new_session
            _rembg_session = new_session("u2netp")
        except Exception:
            _rembg_session = None
    return _rembg_session


def has_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def _client():
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _call_text_model(contents, config):
    """Calls Gemini text model with automatic fallback through multiple models.
    Each model has its own separate free-tier quota, so if one is exhausted
    we transparently try the next.
    """
    client = _client()
    last_error = None
    for model in TEXT_MODELS:
        try:
            r = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return r
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                last_error = e
                continue  # Try next model
            raise  # Non-quota error, raise immediately
    # All models exhausted
    if last_error:
        raise last_error
    raise RuntimeError("No text models available.")


def _extract_text(res) -> str:
    """Safely extracts text parts from a Gemini response, ignoring non-text thoughts."""
    if not res.candidates or not res.candidates[0].content or not res.candidates[0].content.parts:
        return ""
    text_parts = [p.text for p in res.candidates[0].content.parts if hasattr(p, "text") and p.text]
    return "".join(text_parts).strip()


# ─────────────────────────────────────────────────────────────────────
# DIRECT PHOTO ENGINE: Pixel-accurate transforms on the user's actual photo
# ─────────────────────────────────────────────────────────────────────

def to_light_mode(img: Image.Image) -> Image.Image:
    """Converts a dark mode UI or photo to light mode by inverting luminosity in LAB space,
    preserving colors, buttons, and icons while turning dark backgrounds bright and white text dark.
    """
    arr = np.array(img.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_inv = 255 - l
    lab_inv = cv2.merge([l_inv, a, b])
    rgb_inv = cv2.cvtColor(lab_inv, cv2.COLOR_LAB2RGB)
    out = Image.fromarray(rgb_inv)
    return ImageEnhance.Contrast(out).enhance(1.08)


def to_dark_mode(img: Image.Image) -> Image.Image:
    """Converts a light mode UI or photo to dark mode."""
    return to_light_mode(img)


def invert_colors(img: Image.Image) -> Image.Image:
    """Full RGB color inversion."""
    return ImageOps.invert(img.convert("RGB"))


def adjust_brightness(img: Image.Image, factor: float = 1.3) -> Image.Image:
    """Adjusts image brightness."""
    return ImageEnhance.Brightness(img.convert("RGB")).enhance(factor)


def adjust_contrast(img: Image.Image, factor: float = 1.3) -> Image.Image:
    """Adjusts image contrast."""
    return ImageEnhance.Contrast(img.convert("RGB")).enhance(factor)


def remove_background(img: Image.Image, transparent: bool = False, replace_color: tuple = (255, 255, 255)) -> Image.Image:
    """Removes background from the actual uploaded image using AI segmentation (rembg)."""
    from rembg import remove
    session = _get_rembg_session()
    rgba = remove(img.convert("RGB"), session=session) if session else remove(img.convert("RGB"))

    if transparent:
        return rgba

    # Composite onto solid backdrop (e.g. clean studio white)
    bg = Image.new("RGBA", rgba.size, replace_color + (255,))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg.convert("RGB")


def portrait_bokeh(img: Image.Image, blur_radius: int = 14) -> Image.Image:
    """Portrait Mode: Keeps the foreground subject sharp while blurring the background smoothly."""
    from rembg import remove
    session = _get_rembg_session()
    rgb = img.convert("RGB")
    rgba = remove(rgb, session=session) if session else remove(rgb)
    mask = rgba.split()[3]

    # Blur only the background
    blurred_bg = rgb.filter(ImageFilter.GaussianBlur(blur_radius))
    return Image.composite(rgb, blurred_bg, mask)


def pencil_sketch(img: Image.Image) -> Image.Image:
    """Turns the actual photo into an authentic graphite pencil sketch."""
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    blurred = cv2.GaussianBlur(inv, (21, 21), sigmaX=0, sigmaY=0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    return Image.fromarray(sketch).convert("RGB")


def cartoon_art(img: Image.Image) -> Image.Image:
    """Transforms the actual photo into comic/cel-shaded cartoon art."""
    arr = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    color = cv2.bilateralFilter(bgr, d=9, sigmaColor=250, sigmaSpace=250)
    cartoon_bgr = cv2.bitwise_and(color, color, mask=edges)
    cartoon_rgb = cv2.cvtColor(cartoon_bgr, cv2.COLOR_BGR2RGB)
    out = Image.fromarray(cartoon_rgb)
    return ImageEnhance.Color(out).enhance(1.2)


def black_and_white(img: Image.Image) -> Image.Image:
    """Rich, high-contrast monochrome conversion."""
    gray = ImageOps.grayscale(img.convert("RGB"))
    return ImageEnhance.Contrast(gray.convert("RGB")).enhance(1.25)


def vibrant_colors(img: Image.Image) -> Image.Image:
    """Boosts color saturation and vibrancy naturally."""
    rgb = img.convert("RGB")
    sat = ImageEnhance.Color(rgb).enhance(1.65)
    return ImageEnhance.Contrast(sat).enhance(1.08)


def enhance_quality(img: Image.Image) -> Image.Image:
    """Smart auto-enhancement: balanced sharpness, clarity, contrast, and brightness."""
    rgb = img.convert("RGB")
    sharp = ImageEnhance.Sharpness(rgb).enhance(1.35)
    contr = ImageEnhance.Contrast(sharp).enhance(1.12)
    return ImageEnhance.Brightness(contr).enhance(1.03)


def vintage_film(img: Image.Image) -> Image.Image:
    """35mm classic warm film aesthetic with faded blacks and analog warmth."""
    rgb = img.convert("RGB")
    r, g, b = rgb.split()
    r = r.point(lambda i: min(255, int(i * 1.12 + 12)))
    b = b.point(lambda i: int(i * 0.88))
    vintage = Image.merge("RGB", (r, g, b))
    return ImageEnhance.Contrast(vintage).enhance(0.92)


def golden_hour(img: Image.Image) -> Image.Image:
    """Simulates warm golden hour sunlight with amber highlights and gentle glow."""
    rgb = img.convert("RGB")
    r, g, b = rgb.split()
    r = r.point(lambda i: min(255, int(i * 1.20 + 15)))
    g = g.point(lambda i: min(255, int(i * 1.06 + 6)))
    b = b.point(lambda i: int(i * 0.80))
    warm = Image.merge("RGB", (r, g, b))
    return ImageEnhance.Brightness(warm).enhance(1.04)


def cyberpunk_neon(img: Image.Image) -> Image.Image:
    """Cyberpunk aesthetic with neon cyan and electric magenta split-toning."""
    rgb = img.convert("RGB")
    r, g, b = rgb.split()
    r = r.point(lambda i: min(255, int(i * 1.18 + 20)))
    b = b.point(lambda i: min(255, int(i * 1.28 + 28)))
    g = g.point(lambda i: int(i * 0.92))
    neon = Image.merge("RGB", (r, g, b))
    return ImageEnhance.Contrast(neon).enhance(1.2)


def sepia_warmth(img: Image.Image) -> Image.Image:
    """Classic warm antique sepia tones."""
    gray = ImageOps.grayscale(img.convert("RGB"))
    return ImageOps.colorize(gray, "#2b1900", "#f5ebd9")


def rotate_clockwise(img: Image.Image) -> Image.Image:
    """Rotates the image 90 degrees clockwise."""
    return img.rotate(-90, expand=True)


def flip_horizontal(img: Image.Image) -> Image.Image:
    """Flips the image horizontally (mirror effect)."""
    return ImageOps.mirror(img)


# Named color -> OpenCV HSV hue (0-179 range)
_COLOR_HUES = {
    "red": 0, "dark red": 0, "maroon": 0,
    "orange": 15, "dark orange": 15,
    "yellow": 30, "gold": 25,
    "lime": 45, "light green": 50,
    "green": 60, "dark green": 60,
    "teal": 75, "turquoise": 80,
    "cyan": 90, "light blue": 100, "sky blue": 100,
    "azure": 105,
    "blue": 120, "dark blue": 120, "navy": 120,
    "purple": 135, "violet": 135, "indigo": 130,
    "magenta": 150, "fuchsia": 150,
    "pink": 165, "light pink": 168, "rose": 168,
    "white": -1, "black": -2, "gray": -3, "grey": -3,
}


def recolor_region(img: Image.Image, from_color: str, to_color: str, tolerance: int = 25) -> Image.Image:
    """Replaces one color with another in the actual image using HSV hue replacement.
    Only affects pixels that match the source color; everything else is untouched.
    Sets hue directly to the target value for accurate color reproduction.
    """
    from_color = from_color.lower().strip()
    to_color = to_color.lower().strip()

    src_hue = _COLOR_HUES.get(from_color)
    tgt_hue = _COLOR_HUES.get(to_color)
    if src_hue is None or tgt_hue is None:
        return img  # unknown color name

    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Build mask for pixels matching the source color hue
    # Use a wider tolerance band to catch real-world color variations
    lo = src_hue - tolerance
    hi = src_hue + tolerance

    if lo < 0:
        # Source wraps around 0 (e.g. red)
        mask_h = (h <= hi) | (h >= (180 + lo))
    elif hi > 179:
        mask_h = (h >= lo) | (h <= (hi - 180))
    else:
        mask_h = (h >= lo) & (h <= hi)

    mask_s = s > 35  # needs some saturation (not gray/white/black)
    mask_v = v > 35  # not too dark
    mask = mask_h & mask_s & mask_v

    # SET the hue directly to the target value (not shift).
    # This avoids drift from the source hue not matching the ideal mapping.
    h_new = h.copy()
    h_new[mask] = np.uint8(tgt_hue)

    hsv_new = cv2.merge([h_new, s, v])
    rgb_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2RGB)
    return Image.fromarray(rgb_new)


# ─────────────────────────────────────────────────────────────────────
# REGION & TEXT MANIPULATION ENGINE (Local PIL + Gemini Vision)
# ─────────────────────────────────────────────────────────────────────

def _swap_regions_in_image(img: Image.Image, box1_norm: list, box2_norm: list) -> Image.Image | None:
    """Swaps two rectangular regions in the image given their normalized [ymin, xmin, ymax, xmax] coordinates (0-1000).
    Works on buttons, text blocks, cards, or any visual elements with 100% pixel fidelity.
    """
    try:
        W, H = img.size
        y1_a = max(0, min(H, int(box1_norm[0] / 1000.0 * H)))
        x1_a = max(0, min(W, int(box1_norm[1] / 1000.0 * W)))
        y2_a = max(0, min(H, int(box1_norm[2] / 1000.0 * H)))
        x2_a = max(0, min(W, int(box1_norm[3] / 1000.0 * W)))

        y1_b = max(0, min(H, int(box2_norm[0] / 1000.0 * H)))
        x1_b = max(0, min(W, int(box2_norm[1] / 1000.0 * W)))
        y2_b = max(0, min(H, int(box2_norm[2] / 1000.0 * H)))
        x2_b = max(0, min(W, int(box2_norm[3] / 1000.0 * W)))

        w1, h1 = x2_a - x1_a, y2_a - y1_a
        w2, h2 = x2_b - x1_b, y2_b - y1_b

        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return None

        crop1 = img.crop((x1_a, y1_a, x2_a, y2_a))
        crop2 = img.crop((x1_b, y1_b, x2_b, y2_b))

        # Resize crops if necessary to match the destination bounding box
        if crop2.size != (w1, h1):
            crop2 = crop2.resize((w1, h1), Image.LANCZOS)
        if crop1.size != (w2, h2):
            crop1 = crop1.resize((w2, h2), Image.LANCZOS)

        result = img.copy()
        result.paste(crop2, (x1_a, y1_a))
        result.paste(crop1, (x1_b, y1_b))
        return result
    except Exception:
        return None


def _replace_text_in_image(img: Image.Image, old_text: str, new_text: str) -> Image.Image | None:
    """Uses Gemini text model to locate text in the image, then uses PIL to
    paint over the old text with matching background color and render new text.
    Works entirely locally with zero image generation quota needed.
    """
    if not has_key():
        return None

    try:
        from PIL import ImageDraw, ImageFont

        small = img.convert("RGB").copy()
        small.thumbnail((1024, 1024))

        prompt = f"""Find the text "{old_text}" (or closest match) in this image.
Return ONLY valid JSON:
{{
  "found": true,
  "box_2d": [ymin, xmin, ymax, xmax],
  "text_color": [r, g, b]
}}
Note: box_2d MUST be normalized 0-1000 coordinates [ymin, xmin, ymax, xmax] tightly bounding the text."""

        r = _call_text_model(
            contents=[small, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        data = json.loads(_extract_text(r))
        if not data.get("found", False) or "box_2d" not in data:
            return None

        box = data["box_2d"]
        W, H = img.size
        y1 = max(0, int(box[0] / 1000.0 * H))
        x1 = max(0, int(box[1] / 1000.0 * W))
        y2 = min(H, int(box[2] / 1000.0 * H))
        x2 = min(W, int(box[3] / 1000.0 * W))

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return None

        # Sample background color from border just outside the text box
        np_img = np.array(img.convert("RGB"))
        pad = max(2, int(bh * 0.15))
        sample_y1 = max(0, y1 - pad)
        sample_y2 = min(H, y2 + pad)
        sample_x1 = max(0, x1 - pad)
        sample_x2 = min(W, x2 + pad)

        border_pixels = []
        if sample_y1 < y1:
            border_pixels.append(np_img[sample_y1:y1, sample_x1:sample_x2])
        if sample_y2 > y2:
            border_pixels.append(np_img[y2:sample_y2, sample_x1:sample_x2])
        if sample_x1 < x1:
            border_pixels.append(np_img[sample_y1:sample_y2, sample_x1:x1])
        if sample_x2 > x2:
            border_pixels.append(np_img[sample_y1:sample_y2, x2:sample_x2])

        if border_pixels:
            border_concat = np.concatenate([p.reshape(-1, 3) for p in border_pixels if p.size > 0], axis=0)
            bg_color = tuple(int(c) for c in np.median(border_concat, axis=0))
        else:
            bg_color = (255, 255, 255)

        text_color = tuple(data.get("text_color", [30, 30, 30]))

        result = img.convert("RGB").copy()
        draw = ImageDraw.Draw(result)

        # Erase old text with exact background fill
        draw.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], fill=bg_color)

        # Estimate font size and load font
        font_size = max(10, int(bh * 0.85))
        font = None
        for font_name in ["arial.ttf", "Arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "Helvetica.ttf"]:
            try:
                font = ImageFont.truetype(font_name, font_size)
                break
            except Exception:
                pass
        if font is None:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

        # Adjust font size if text is too wide or tall
        bbox = draw.textbbox((0, 0), new_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw > bw or th > bh:
            ratio = min(bw / max(tw, 1), bh / max(th, 1))
            adjusted_size = max(8, int(font_size * ratio * 0.95))
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", adjusted_size)
            except Exception:
                pass
            bbox = draw.textbbox((0, 0), new_text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Draw vertically centered in the original box
        draw_y = y1 + (bh - th) // 2
        draw.text((x1, draw_y), new_text, fill=text_color, font=font)

        return result
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# INTELLIGENT ROUTER & GEMINI PLANNER
# ─────────────────────────────────────────────────────────────────────

def _mentions_color(text: str) -> bool:
    """Check if the text mentions a known color name (used to avoid false-positive theme triggers)."""
    color_words = set(_COLOR_HUES.keys()) - {"white", "black", "gray", "grey"}
    for c in color_words:
        if c in text:
            return True
    return False


def match_direct_edit(task: str, img: Image.Image) -> Tuple[Image.Image, str] | None:
    """Fast keyword matching for instantaneous execution without calling LLM."""
    t = task.lower().strip()

    # ── Never trigger whole-image filters if user is asking to CREATE / ADD / GENERATE ──
    is_creation = any(
        w in t for w in [
            "create", "generate", "draw a", "draw an", "add a", "add an",
            "insert", "put a", "put an", "place a", "place an", "paint a", "paint an",
            "character", "make a character", "spawn", "render a", "render an",
        ]
    )
    if is_creation:
        return None

    # ── Skip theme/brightness keywords if the prompt is about changing colors ──
    # e.g. "change green to light blue" should NOT trigger light mode
    is_color_change = ("change" in t or "replace" in t or "swap" in t) and _mentions_color(t)

    # Light Mode / Dark Mode UI & Invert (only if NOT a color change request)
    if not is_color_change:
        if any(k in t for k in ["light mode", "light theme", "switch to light mode", "switch to light theme"]):
            return to_light_mode(img), "☀️ Converted to Light Mode (edited your actual image)."
        # "to light" and "make it light" only match if no color word follows
        if ("to light" in t or "make it light" in t) and not _mentions_color(t):
            return to_light_mode(img), "☀️ Converted to Light Mode (edited your actual image)."

        if any(k in t for k in ["dark mode", "dark theme", "switch to dark mode", "switch to dark theme"]):
            return to_dark_mode(img), "🌙 Converted to Dark Mode (edited your actual image)."
        if ("to dark" in t or "make it dark" in t) and not _mentions_color(t):
            return to_dark_mode(img), "🌙 Converted to Dark Mode (edited your actual image)."

    if "invert" in t and ("color" in t or "colours" in t or "negative" in t):
        return invert_colors(img), "🔄 Inverted colors (edited your actual image)."

    # Brightness & Exposure (only if NOT a color change request)
    if not is_color_change:
        if any(k in t for k in ["brighter", "brighten", "more light", "increase brightness", "lighter"]):
            return adjust_brightness(img, 1.35), "💡 Increased brightness (edited your actual image)."

        if any(k in t for k in ["darker", "dimmer", "decrease brightness", "dim"]):
            return adjust_brightness(img, 0.72), "🌑 Decreased brightness (edited your actual image)."

    # Background Removal & Bokeh
    if "transparent" in t and ("background" in t or "bg" in t or "cutout" in t):
        return remove_background(img, transparent=True), "✂️ Removed background with transparent cutout (edited your actual photo)."

    if "portrait" in t or "bokeh" in t or "blur background" in t or "blurred background" in t or "blur the background" in t or "blur bg" in t:
        return portrait_bokeh(img), "📸 Applied Portrait Mode bokeh blur to background (edited your actual photo)."

    if "remove" in t and ("background" in t or "bg" in t) or "white background" in t:
        return remove_background(img, transparent=False), "✂️ Isolated subject on clean white background (edited your actual photo)."

    # Artistic Stylizations (style transfer on existing photo)
    if any(k in t for k in ["pencil sketch", "graphite sketch", "sketch style", "line art style", "pencil drawing style", "convert to sketch", "turn into sketch", "make it a sketch"]):
        return pencil_sketch(img), "✏️ Converted to pencil sketch (edited your actual photo)."

    if any(k in t for k in ["cartoon style", "comic style", "anime style", "cartoon art", "cel shaded", "pop art style", "make it cartoon", "convert to cartoon", "convert to anime", "turn into cartoon", "turn into anime", "cartoon comic cel shaded art"]):
        return cartoon_art(img), "🎭 Converted to cartoon/comic art (edited your actual photo)."

    # Color & Tone
    if any(k in t for k in ["black and white", "black & white", "b&w", "monochrome", "grayscale", "greyscale"]):
        return black_and_white(img), "🖤 Converted to black & white (edited your actual photo)."

    if any(k in t for k in ["vibrant", "saturation", "colorful", "color pop", "vivid", "rich colors"]):
        return vibrant_colors(img), "🎨 Enhanced colors and saturation (edited your actual photo)."

    if any(k in t for k in ["vintage", "film", "retro", "35mm", "polaroid", "nostalgic", "grain"]):
        return vintage_film(img), "🎞️ Applied vintage 35mm film grade (edited your actual photo)."

    if any(k in t for k in ["golden hour", "sunset", "warm glow", "amber", "golden light"]):
        return golden_hour(img), "🌅 Applied golden hour warm sunlight grade (edited your actual photo)."

    if any(k in t for k in ["cyberpunk", "neon", "matrix", "sci-fi", "cool tone", "futuristic"]):
        return cyberpunk_neon(img), "🌆 Applied cyberpunk neon color grade (edited your actual photo)."

    if any(k in t for k in ["sepia", "antique", "old photo", "brown tone"]):
        return sepia_warmth(img), "📜 Converted to antique sepia (edited your actual photo)."

    if any(k in t for k in ["enhance", "clarity", "sharpen", "hdr", "fix quality", "improve quality", "hd", "clear"]):
        return enhance_quality(img), "✨ Auto-enhanced clarity, contrast, and sharpness (edited your actual photo)."

    if any(k in t for k in ["contrast", "high contrast"]):
        return adjust_contrast(img, 1.3), "🌓 Increased contrast (edited your actual photo)."

    # Geometric
    if any(k in t for k in ["rotate", "turn 90", "turn right", "clockwise"]):
        return rotate_clockwise(img), "🔄 Rotated 90° clockwise (edited your actual photo)."

    if any(k in t for k in ["flip", "mirror", "horizontal flip"]):
        return flip_horizontal(img), "🪞 Mirrored horizontally (edited your actual photo)."

    return None


def _ai_plan_edit(task: str, img: Image.Image) -> Tuple[Image.Image, str] | None:
    """Uses Gemini Flash vision to interpret natural language requests and apply direct pixel edits."""
    if not has_key():
        return None

    planner_prompt = f"""You are an expert image processing planner.
Analyze the user's uploaded image and edit instruction.
Determine how to apply this edit directly to their actual image.

User Request: "{task}"

Available Actions:
- "swap_elements": Swap the positions/places of two visual elements, buttons, texts, cards, or sections in the image (e.g. "swap the google and github buttons", "swap their places", "exchange X and Y"). Requires "element1_box": [ymin, xmin, ymax, xmax] and "element2_box": [ymin, xmin, ymax, xmax] in normalized 0-1000 coordinates.
- "text_replace": Replace specific text in the image. Requires "replacements": [{{"old_text": "...", "new_text": "..."}}] list. Use this when the user wants to change/edit words, numbers, emails, or titles visible in the image.
- "recolor": Change one color to another in the image (e.g. "change green button to blue", "make red text yellow"). Requires "from_color" and "to_color".
- "light_mode": Convert dark UI or dark photo to light mode (light background, dark text/elements)
- "dark_mode": Convert light UI or photo to dark mode
- "brightness": Make brighter or darker
- "contrast": Increase or decrease contrast
- "invert": Invert image colors
- "remove_bg_white": Remove background and replace with white
- "remove_bg_transparent": Remove background with transparent PNG
- "portrait_bokeh": Blur background, keep subject sharp
- "pencil_sketch": Convert existing photo into graphite pencil sketch. Use ONLY for style transfer.
- "cartoon": Stylize the existing photo as cel-shaded cartoon art. Use ONLY for style transfer on an existing subject. DO NOT use when creating or adding characters, objects, or drawings.
- "black_and_white": High contrast black and white
- "vibrant": Boost saturation and color punch
- "enhance": Auto balance exposure, clarity, sharpness
- "vintage": 35mm film warm faded grade
- "golden_hour": Warm sunset amber sunlight
- "cyberpunk": Cool cyan and neon magenta grade
- "sepia": Warm antique sepia tone
- "rotate": Rotate 90 degrees
- "flip": Mirror horizontally
- "ai_edit": For creating or adding new visual objects, characters, or drawings (e.g. "create an anime character", "add a hat", "draw a dragon"). Never use for UI, text, buttons, or photo adjustments.

Respond ONLY with valid JSON:
{{
  "action": "<action_name>",
  "element1_box": [ymin, xmin, ymax, xmax],
  "element2_box": [ymin, xmin, ymax, xmax],
  "replacements": [{{"old_text": "...", "new_text": "..."}}],
  "from_color": "green",
  "to_color": "blue",
  "brightness_factor": 1.3,
  "description": "<one short sentence explaining what was done>"
}}"""

    try:
        small = img.convert("RGB").copy()
        small.thumbnail((768, 768))
        r = _call_text_model(
            contents=[small, planner_prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1),
        )
        plan_text = _extract_text(r)
        data = json.loads(plan_text)
        action = data.get("action", "").lower()
        desc = data.get("description", "Applied AI edit.")

        if action in ("swap_elements", "swap_text", "swap"):
            box1 = data.get("element1_box")
            box2 = data.get("element2_box")
            if box1 and box2 and len(box1) == 4 and len(box2) == 4:
                result = _swap_regions_in_image(img, box1, box2)
                if result is not None:
                    return result, f"🔄 {desc} (edited your actual image)"

        elif action == "light_mode":
            return to_light_mode(img), f"☀️ {desc} (edited your actual image)"
        elif action == "dark_mode":
            return to_dark_mode(img), f"🌙 {desc} (edited your actual image)"
        elif action == "brightness":
            factor = float(data.get("brightness_factor", 1.3))
            return adjust_brightness(img, factor), f"💡 {desc} (edited your actual image)"
        elif action == "contrast":
            return adjust_contrast(img, 1.3), f"🌓 {desc} (edited your actual image)"
        elif action == "invert":
            return invert_colors(img), f"🔄 {desc} (edited your actual image)"
        elif action == "remove_bg_white":
            return remove_background(img, transparent=False), f"✂️ {desc} (edited your actual image)"
        elif action == "remove_bg_transparent":
            return remove_background(img, transparent=True), f"✂️ {desc} (edited your actual image)"
        elif action == "portrait_bokeh":
            return portrait_bokeh(img), f"📸 {desc} (edited your actual image)"
        elif action == "pencil_sketch":
            return pencil_sketch(img), f"✏️ {desc} (edited your actual image)"
        elif action == "cartoon":
            return cartoon_art(img), f"🎭 {desc} (edited your actual image)"
        elif action == "black_and_white":
            return black_and_white(img), f"🖤 {desc} (edited your actual image)"
        elif action == "vibrant":
            return vibrant_colors(img), f"🎨 {desc} (edited your actual image)"
        elif action == "enhance":
            return enhance_quality(img), f"✨ {desc} (edited your actual image)"
        elif action == "vintage":
            return vintage_film(img), f"🎞️ {desc} (edited your actual image)"
        elif action == "golden_hour":
            return golden_hour(img), f"🌅 {desc} (edited your actual image)"
        elif action == "cyberpunk":
            return cyberpunk_neon(img), f"🌆 {desc} (edited your actual image)"
        elif action == "sepia":
            return sepia_warmth(img), f"📜 {desc} (edited your actual image)"
        elif action == "rotate":
            return rotate_clockwise(img), f"🔄 {desc} (edited your actual image)"
        elif action == "flip":
            return flip_horizontal(img), f"🪞 {desc} (edited your actual image)"
        elif action == "recolor":
            fc = data.get("from_color", "green")
            tc = data.get("to_color", "blue")
            result = recolor_region(img, fc, tc)
            return result, f"🎨 {desc} (edited your actual image)"
        elif action == "text_replace":
            replacements = data.get("replacements") or []
            if not replacements:
                ot = data.get("old_text", "")
                nt = data.get("new_text", "")
                if ot and nt:
                    replacements = [{"old_text": ot, "new_text": nt}]

            if replacements:
                curr = img
                for rep in replacements:
                    ot = rep.get("old_text", "")
                    nt = rep.get("new_text", "")
                    if ot and nt:
                        res = _replace_text_in_image(curr, ot, nt)
                        if res is not None:
                            curr = res
                if curr is not img:
                    return curr, f"✏️ {desc} (replaced text in your actual image)"
            return None
        elif action == "ai_edit":
            return None  # Fall through to Gemini image editor
    except Exception:
        pass

    # If the user prompt specifically asks to swap elements or places, try a targeted swap detection
    t_low = task.lower()
    if "swap" in t_low or "exchange" in t_low or ("change" in t_low and "to" in t_low and "and" in t_low):
        try:
            swap_prompt = f"""Find the two visual elements (buttons, text boxes, or cards) to swap for request: "{task}".
Return ONLY valid JSON:
{{
  "found": true,
  "element1_box": [ymin, xmin, ymax, xmax],
  "element2_box": [ymin, xmin, ymax, xmax],
  "description": "Swapped the requested elements"
}}
Note: coordinates must be normalized 0-1000."""
            small = img.convert("RGB").copy()
            small.thumbnail((768, 768))
            r = _call_text_model(
                contents=[small, swap_prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1),
            )
            swap_data = json.loads(_extract_text(r))
            if swap_data.get("found", False) and "element1_box" in swap_data and "element2_box" in swap_data:
                res = _swap_regions_in_image(img, swap_data["element1_box"], swap_data["element2_box"])
                if res is not None:
                    desc = swap_data.get("description", "Swapped elements in your actual image.")
                    return res, f"🔄 {desc} (edited your actual image)"
        except Exception:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────
# GEMINI NATIVE IMAGE EDITOR (sends actual image + prompt for AI editing)
# ─────────────────────────────────────────────────────────────────────

IMAGE_EDIT_MODEL = "gemini-3.1-flash-image"


def _gemini_image_edit(img: Image.Image, task: str) -> Image.Image | None:
    """Uses Gemini's native image editing model to edit the actual uploaded image.
    Sends the real image + the user's edit instruction and receives back an edited image.
    Returns None if rate-limited (so caller can fall back to Pollinations).
    """
    if not has_key():
        return None

    try:
        # Prepare the image (keep reasonable resolution for the API)
        edit_img = img.convert("RGB").copy()
        edit_img.thumbnail((1024, 1024))

        system_instruction = (
            "You are a precise image editor. The user has uploaded an image and wants you to "
            "edit it according to their instructions. Make ONLY the requested change — preserve "
            "everything else in the image exactly as it is. Do not change the layout, style, "
            "colors, or any other elements unless explicitly asked. Return the edited image."
        )

        r = _client().models.generate_content(
            model=IMAGE_EDIT_MODEL,
            contents=[
                edit_img,
                f"Edit this image: {task}",
            ],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                system_instruction=system_instruction,
                temperature=0.2,
            ),
        )

        # Extract the edited image from the response
        if r.candidates and r.candidates[0].content and r.candidates[0].content.parts:
            for part in r.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    edited = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                    # Resize back to original dimensions
                    if edited.size != img.size:
                        edited = edited.resize(img.size, Image.LANCZOS)
                    return edited
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
            return None  # Rate limited — fall back to Pollinations
        raise

    return None  # No image returned


# ─────────────────────────────────────────────────────────────────────
# POLLINATIONS IMAGE-TO-IMAGE (sends actual image for editing, free tier)
# ─────────────────────────────────────────────────────────────────────

def _upload_temp_image(img: Image.Image) -> str | None:
    """Uploads an image to a temporary free host (catbox.moe) and returns the public URL.
    This is needed because Pollinations' free img2img endpoint requires a public image URL.
    """
    buf = io.BytesIO()
    # Use JPEG for smaller upload size, keep quality reasonable
    rgb = img.convert("RGB")
    rgb.thumbnail((1024, 1024))
    rgb.save(buf, format="JPEG", quality=80)
    buf.seek(0)
    image_data = buf.read()

    boundary = f"----PixelPromptBoundary{random.randint(100000, 999999)}"
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += b"Content-Disposition: form-data; name=\"reqtype\"\r\n\r\nfileupload\r\n"
    body += f"--{boundary}\r\n".encode()
    body += b"Content-Disposition: form-data; name=\"fileToUpload\"; filename=\"edit.jpg\"\r\n"
    body += b"Content-Type: image/jpeg\r\n\r\n"
    body += image_data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    try:
        req = urllib.request.Request(
            "https://catbox.moe/user/api.php",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "Mozilla/5.0 PixelPromptAI/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            url = resp.read().decode().strip()
            if url.startswith("http"):
                return url
    except Exception:
        pass

    return None


def _pollinations_img2img(prompt: str, image_url: str, w: int, h: int) -> Image.Image:
    """Calls Pollinations API with the user's actual image for image-to-image editing."""
    seed = random.randint(1, 999999)
    encoded_prompt = urllib.parse.quote(prompt)
    encoded_image = urllib.parse.quote(image_url)
    url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?width={w}&height={h}&nologo=true&seed={seed}"
        f"&image={encoded_image}"
    )

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 PixelPromptAI/1.0"},
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        except urllib.error.HTTPError as e:
            if e.code == 402:
                body = e.read().decode("utf-8", errors="ignore")
                match = re.search(r"Retry after (\d+)s", body)
                wait = min(int(match.group(1)) + 2, 50) if match else 10
                if attempt < max_attempts - 1:
                    time.sleep(wait)
                    continue
            raise RuntimeError(
                "Free image generation rate limit reached. Please wait ~45 seconds and try again."
            )
        except Exception as e:
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Image editing engine failed: {e}")
            time.sleep(3)

    raise RuntimeError("Image editing timed out.")


def _pollinations_generate(prompt: str, w: int, h: int) -> Image.Image:
    """Calls Pollinations API for text-to-image generation (no input image)."""
    seed = random.randint(1, 999999)
    encoded = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width={w}&height={h}&nologo=true&seed={seed}"

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 PixelPromptAI/1.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        except urllib.error.HTTPError as e:
            if e.code == 402:
                body = e.read().decode("utf-8", errors="ignore")
                match = re.search(r"Retry after (\d+)s", body)
                wait = min(int(match.group(1)) + 2, 50) if match else 10
                if attempt < max_attempts - 1:
                    time.sleep(wait)
                    continue
            raise RuntimeError(
                "Free image generation rate limit reached. Please wait ~45 seconds and try again."
            )
        except Exception as e:
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Generative engine failed: {e}")
            time.sleep(3)

    raise RuntimeError("Image generation timed out.")


# ─────────────────────────────────────────────────────────────────────
# MASTER DISPATCHER: edit_image
# ─────────────────────────────────────────────────────────────────────

def edit_image(img: Image.Image, task: str) -> Tuple[Image.Image, str]:
    """Primary entry point for all edits.

    1. Checks fast keyword match (0ms latency direct edit).
    2. Checks Gemini 3.8 Flash AI planner (interprets natural language to direct edit).
    3. Tries Gemini native image editor (sends actual image + prompt).
    4. Falls back to Pollinations img2img (uploads your image, sends it with the prompt).
    """
    # ── 1. Fast direct keyword check ──
    direct = match_direct_edit(task, img)
    if direct is not None:
        return direct

    # ── 2. Gemini 3.8 Flash AI Planner ──
    planned = _ai_plan_edit(task, img)
    if planned is not None:
        return planned

    # ── 3. Gemini Native Image Editing (best quality — sends your actual image) ──
    gemini_result = _gemini_image_edit(img, task)
    if gemini_result is not None:
        note = (
            "🎨 AI Image Edit: Gemini edited your actual image based on your prompt. "
            "(Tip: For instant edits, try 'light mode', 'dark mode', 'sketch', 'remove background', etc.)"
        )
        return gemini_result, note

    # ── Guard: Do NOT use generative diffusion for UI, text, buttons, or swapping ──
    t_lower = task.lower()
    is_ui_or_text_edit = any(
        w in t_lower
        for w in [
            "swap", "exchange", "text", "replace", "font", "button", "email",
            "mail", "input", "letter", "word", "title", "label", "heading",
            "box", "card", "section", "change to", "change the",
        ]
    )
    if is_ui_or_text_edit:
        raise RuntimeError(
            "Could not detect the requested text or elements in your image. "
            "Please ensure the image is clear and phrase your request simply (e.g. 'swap the Google and GitHub buttons' or 'replace text X with Y')."
        )

    # ── 4. Pollinations img2img (uploads your actual image + sends prompt) ──
    orig_w, orig_h = img.size
    aspect = orig_w / max(orig_h, 1)
    if aspect >= 1:
        w, h = 512, max(320, int(512 / aspect) // 64 * 64)
    else:
        h, w = 512, max(320, int(512 * aspect) // 64 * 64)

    # For creation requests (e.g. "create an anime character on the page", "draw a dragon"),
    # generate directly for speed and quality:
    is_creation = any(
        w in t_lower
        for w in [
            "create", "generate", "draw a", "draw an", "character", "make a character",
            "paint a", "paint an", "render a", "render an"
        ]
    )
    if is_creation:
        out = _pollinations_generate(task, w, h)
        note = "🎨 AI Generation: Created anime/drawing from your prompt."
        return out, note

    # Try to upload the actual image for img2img editing
    hosted_url = _upload_temp_image(img)
    if hosted_url:
        out = _pollinations_img2img(task, hosted_url, w, h)
        note = (
            "🎨 AI Image Edit (Pollinations): Edited your actual image based on your prompt. "
            "(Gemini image quota exceeded — using Pollinations with your real image as reference.)"
        )
        return out, note

    # Last resort: text-to-image without input image
    out = _pollinations_generate(task, w, h)
    note = (
        "🎨 AI Generation (text-to-image fallback): Could not upload your image for editing. "
        "Generated a new image from your prompt description. Results may differ from the original."
    )
    return out, note


# ─────────────────────────────────────────────────────────────────────
# Multimodal Search & Indexing (Gemini free tier)
# ─────────────────────────────────────────────────────────────────────

def caption_image(img: Image.Image) -> dict:
    if not has_key():
        return {"caption": "Uploaded image", "tags": ["image"]}
    try:
        small = img.convert("RGB").copy()
        small.thumbnail((1024, 1024))
        r = _call_text_model(
            contents=[small, prompts.CAPTION_PROMPT],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        text = _extract_text(r)
        data = json.loads(text)
        return {"caption": data.get("caption", ""), "tags": data.get("tags", [])}
    except Exception:
        return {"caption": "Uploaded image", "tags": ["image"]}


def embed(text: str, query: bool = False) -> list[float]:
    if not has_key():
        return []
    try:
        r = _client().models.embed_content(
            model=EMBED_MODEL,
            contents=[text],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
            ),
        )
        return r.embeddings[0].values
    except Exception:
        return []


def search(query: str, images: list[dict], top_k: int = 12) -> list[dict]:
    """Semantic search over caption embeddings with keyword fallback."""
    q = query.lower().split()
    scored = []
    qv = np.array(embed(query, query=True)) if has_key() else None
    for im in images:
        text = f"{im.get('caption','')} {im.get('tags','')}".lower()
        kw = sum(w in text for w in q) / max(len(q), 1)
        sem = 0.0
        if qv is not None and len(qv) > 0 and im.get("embedding"):
            try:
                v = np.array(json.loads(im["embedding"]))
                norm_prod = np.linalg.norm(qv) * np.linalg.norm(v)
                if norm_prod > 0:
                    sem = float(qv @ v / norm_prod)
            except Exception:
                pass
        scored.append((0.8 * sem + 0.2 * kw, im))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored or scored[0][0] <= 0:
        return []
    best = scored[0][0]
    return [im for s, im in scored[:top_k] if s >= best - 0.12 and s > 0]
