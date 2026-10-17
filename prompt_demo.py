"""Prompt Handling & Planning Pipeline Demo for PixelPrompt AI.

This script demonstrates how PixelPrompt AI handles user prompts:
1. Direct keyword & intent matching (0ms fast path for presets/theme changes).
2. Multimodal LLM Vision Planning (Gemini Flash analyzes image + freeform text).
3. Structured action dispatch (text replacement, element swapping, filters, generation).
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import json
import toml
from PIL import Image, ImageDraw

# Attempt to load credentials
try:
    secrets = toml.load(".streamlit/secrets.toml")
    if "GEMINI_API_KEY" in secrets:
        os.environ["GEMINI_API_KEY"] = secrets["GEMINI_API_KEY"]
except Exception:
    pass

import ai


def demo_prompt_handling(prompt: str, image: Image.Image):
    print("=" * 60)
    print(f"INPUT PROMPT: \"{prompt}\"")
    print(f"IMAGE SIZE: {image.size}")
    print("=" * 60)

    # Step 1: Direct keyword detection
    direct = ai.match_direct_edit(prompt, image)
    if direct is not None:
        result_img, note = direct
        print(f"⚡ FAST-PATH ROUTED: Direct keyword match triggered.")
        print(f"  Result Note: {note}")
        return result_img, note

    # Step 2: Multimodal LLM + Vision Planner
    print(f"🧠 MULTIMODAL VISION PLANNER: Calling Gemini Flash fallback chain...")
    planned = ai._ai_plan_edit(prompt, image)
    if planned is not None:
        result_img, note = planned
        print(f"🎯 PLANNER EXECUTED: Successfully parsed and applied.")
        print(f"  Result Note: {note}")
        return result_img, note

    # Step 3: Master Dispatcher (Image Edit / Generative Fallback)
    print(f"🎨 MASTER DISPATCHER: Evaluating generative / content edit...")
    result_img, note = ai.edit_image(image, prompt)
    print(f"✨ COMPLETED: {note}")
    return result_img, note


if __name__ == "__main__":
    # Create a mock image with buttons for demonstration
    img = Image.new("RGB", (500, 300), (245, 247, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 60, 440, 110], fill=(255, 255, 255), outline=(210, 214, 220), width=2)
    draw.text((120, 75), "Continue with Google", fill=(30, 30, 30))
    draw.rectangle([60, 140, 440, 190], fill=(36, 41, 46), outline=(36, 41, 46), width=2)
    draw.text((120, 155), "Continue to GitHub", fill=(255, 255, 255))

    sample_prompts = [
        "make it to the light mode",
        "change continue with google to continue to github and continue to github to continue to google , swap their places",
        "create a anime character on the page",
    ]

    for p in sample_prompts:
        try:
            demo_prompt_handling(p, img)
        except Exception as e:
            print(f"Error handling prompt '{p}': {e}")
        print()
