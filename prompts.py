"""Prompt templates & Preset definitions for PixelPrompt AI."""

# One-click photo presets that edit your actual uploaded photo.
# All presets run instantly on the real image pixels.
PRESETS = {
    "✂️ White background": "Remove background and place subject on clean white studio backdrop",
    "✂️ Transparent PNG": "Remove background with transparent cutout",
    "📸 Portrait Bokeh": "Apply portrait mode bokeh blur to background, keep subject sharp",
    "✨ Enhance clarity": "Auto enhance clarity, exposure balance, and sharpness",
    "🎨 Vibrant colors": "Boost color saturation and vibrancy naturally",
    "🖤 Black & white": "Convert to rich contrast black and white monochrome",
    "✏️ Pencil sketch": "Convert photo into hand-drawn graphite pencil sketch",
    "🎭 Comic cartoon": "Transform photo into cel-shaded cartoon art",
    "🌅 Golden hour": "Apply warm golden hour sunset glow",
    "🎞️ Vintage film": "Apply classic vintage 35mm film grade with soft contrast",
    "🌆 Cyberpunk neon": "Apply cool cyan and neon magenta color grade",
    "📜 Antique sepia": "Convert to warm nostalgic antique sepia tone",
    "🔄 Rotate 90°": "Rotate 90 degrees clockwise",
    "🪞 Flip mirror": "Flip horizontal mirror image",
}

# Prompt interpreter: Gemini turns a user request into an explicit instruction.
REFINE_SYSTEM = """You rewrite casual image-editing requests into one precise, unambiguous instruction
for an image-editing model. Keep the user's intent exactly. Make location, object and style explicit
(e.g. 'the person on the left' -> 'the person standing at the left edge of the frame').
If the request is vague, choose the most reasonable interpretation. Do not invent extra edits.
Reply with the instruction only: one or two sentences, no preamble."""

# Captioning prompt (returns JSON) used for search indexing.
CAPTION_PROMPT = """Describe this image for a photo-search index. Return JSON with keys:
"caption": 1-2 factual sentences covering main subjects, setting, colors, mood.
"tags": list of 6-12 lowercase keywords (objects, scene, activity, colors, style)."""
