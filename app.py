import os
from pathlib import Path

import streamlit as st
from PIL import Image

import ai
import prompts
import storage

st.set_page_config(page_title="PixelPrompt AI", page_icon="🪄", layout="wide")
storage.init()

try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

st.session_state.setdefault("page", "library")
st.session_state.setdefault("selected", None)
st.session_state.setdefault("last_note", None)


def go(page, image_id=None):
    st.session_state.page, st.session_state.selected = page, image_id
    st.session_state["last_note"] = None
    st.rerun()


def index_image(image_id: str):
    """Caption with vision model, embed caption for search."""
    rec = storage.get_image(image_id)
    img = Image.open(rec["path"])
    info = ai.caption_image(img)
    vec = ai.embed(f"{info['caption']} Tags: {', '.join(info['tags'])}")
    storage.set_caption(image_id, info["caption"], info["tags"], vec)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🪄 PixelPrompt AI")
    st.caption("AI Photo Editor & Intelligent Visual Studio")
    if not ai.has_key():
        key = st.text_input("Gemini API key", type="password",
                            help="Get one free at aistudio.google.com")
        if key:
            os.environ["GEMINI_API_KEY"] = key
            st.rerun()
    else:
        st.success("✨ Gemini connected (Vision & Search)")

    st.markdown("---")
    st.markdown("**Active Engines:**")
    st.markdown("⚡ **Direct Photo Engine**: Real-pixel edits (Light/Dark Mode, Rembg, OpenCV, PIL)")
    st.markdown("🧠 **AI Planner**: Maps natural language prompts to photo operations")
    st.markdown("🎨 **Gemini Image Editor**: AI edits your actual image (text changes, object edits, creative mods)")
    st.markdown("---")

    if st.button("📚 Library", use_container_width=True):
        go("library")
    st.divider()
    up = st.file_uploader("Upload images", type=["png", "jpg", "jpeg", "webp"],
                          accept_multiple_files=True)
    if up and st.button("Add to library", type="primary", use_container_width=True):
        bar = st.progress(0.0)
        for i, f in enumerate(up):
            iid = storage.add_image(f.name, Image.open(f))
            if ai.has_key():
                try:
                    index_image(iid)
                except Exception as e:
                    st.warning(f"{f.name}: captioning failed ({e})")
            bar.progress((i + 1) / len(up))
        st.rerun()

# ---------------- Library ----------------
if st.session_state.page == "library":
    st.header("Photo Library")
    query = st.text_input("🔎 Search by description", placeholder="e.g. show me photos with a beach or login screens")
    images = storage.list_images()
    if query and images:
        with st.spinner("Searching..."):
            try:
                images = ai.search(query, images)
            except Exception as e:
                st.error(f"Search failed: {e}")
                images = []
        st.caption(f"{len(images)} result(s)")
    if not images:
        st.info("No images yet. Upload some from the sidebar." if not query else "No matches.")
    cols = st.columns(4)
    for i, im in enumerate(images):
        with cols[i % 4]:
            st.image(im["thumb"], use_container_width=True)
            st.caption(im["caption"] or im["name"])
            if st.button("Open Studio", key=f"open_{im['id']}", use_container_width=True):
                go("detail", im["id"])

# ---------------- Detail / Studio ----------------
else:
    rec = storage.get_image(st.session_state.selected)
    if not rec:
        go("library")
    versions = storage.list_versions(rec["id"])
    cur = next(v for v in versions if v["id"] == rec["current_version"])

    head, delcol = st.columns([6, 1])
    head.header(rec["name"])
    if delcol.button("🗑 Delete"):
        storage.delete_image(rec["id"])
        go("library")
    if rec["caption"]:
        st.caption(f"🤖 {rec['caption']}")

    if st.session_state.get("last_note"):
        st.info(st.session_state["last_note"])

    left, right = st.columns([3, 2])
    with left:
        st.image(rec["path"], caption=f"Version {cur['version_no']}: {cur['prompt']}",
                 use_container_width=True)
        st.download_button("⬇ Download PNG", Path(rec["path"]).read_bytes(),
                           file_name=f"{Path(rec['name']).stem}_v{cur['version_no']}.png",
                           mime="image/png")

    def run_edit(task: str, label: str):
        src = Image.open(rec["path"])
        try:
            with st.spinner("Processing edit on your photo..."):
                out, note = ai.edit_image(src, task)
                storage.add_version(rec["id"], out, label, cur["id"])
                try:
                    index_image(rec["id"])
                except Exception:
                    pass
            st.session_state["last_note"] = note
            st.rerun()
        except Exception as e:
            msg = str(e)
            if "402" in msg or "FREE_LIMIT" in msg or "rate limit" in msg.lower():
                st.warning("⏳ Free rate limit reached. Please wait ~45s and try again, or use any direct photo edit.")
            else:
                st.error(f"Edit failed: {e}")

    with right:
        st.subheader("🛠️ Studio Photo Tools")
        st.caption("Edits your **actual uploaded photo/UI** directly.")

        tab_theme, tab_cut, tab_art = st.tabs(["🎨 Theme & Color", "✂️ Cutout & Bokeh", "🖌️ Art & Transform"])

        with tab_theme:
            c1, c2 = st.columns(2)
            if c1.button("☀️ Light Mode UI", use_container_width=True):
                run_edit("make it to the light mode", "Light Mode")
            if c2.button("🌙 Dark Mode UI", use_container_width=True):
                run_edit("make it to the dark mode", "Dark Mode")

            if c1.button("💡 Brighten", use_container_width=True):
                run_edit("increase brightness", "Brighten")
            if c2.button("🌑 Darken", use_container_width=True):
                run_edit("dim decrease brightness", "Darken")

            if c1.button("✨ Enhance Clarity", use_container_width=True):
                run_edit("enhance clarity sharpness contrast", "Enhance clarity")
            if c2.button("🎨 Vibrant Colors", use_container_width=True):
                run_edit("vibrant rich colorful saturation", "Vibrant colors")

            if c1.button("🖤 Black & White", use_container_width=True):
                run_edit("black and white high contrast", "Black & white")
            if c2.button("🌅 Golden Hour Sunset", use_container_width=True):
                run_edit("golden hour warm amber sunlight", "Golden hour")

            if c1.button("🎞️ Vintage 35mm Film", use_container_width=True):
                run_edit("vintage 35mm film warm faded", "Vintage film")
            if c2.button("🌆 Cyberpunk Neon", use_container_width=True):
                run_edit("cyberpunk cool neon magenta grade", "Cyberpunk neon")

            if c1.button("📜 Antique Sepia", use_container_width=True):
                run_edit("antique sepia tone warmth", "Antique sepia")
            if c2.button("🔄 Invert Colors", use_container_width=True):
                run_edit("invert colors", "Invert colors")

        with tab_cut:
            c1, c2 = st.columns(2)
            if c1.button("✂️ White Studio Backdrop", use_container_width=True):
                run_edit("remove background and replace with white", "White background")
            if c2.button("✂️ Transparent PNG Cutout", use_container_width=True):
                run_edit("transparent background cutout", "Transparent PNG")
            if st.button("📸 Portrait Mode (Bokeh Blur)", use_container_width=True):
                run_edit("portrait bokeh background blur", "Portrait Bokeh")

        with tab_art:
            c1, c2 = st.columns(2)
            if c1.button("✏️ Pencil Sketch", use_container_width=True):
                run_edit("pencil sketch graphite drawing", "Pencil sketch")
            if c2.button("🎭 Comic Cartoon Art", use_container_width=True):
                run_edit("cartoon comic cel shaded art", "Comic cartoon")
            if c1.button("🔄 Rotate 90°", use_container_width=True):
                run_edit("rotate 90 clockwise", "Rotate 90°")
            if c2.button("🪞 Flip Mirror", use_container_width=True):
                run_edit("flip horizontal mirror", "Flip mirror")

        st.divider()
        st.subheader("💬 Custom Edit Prompt")
        st.caption("Type any edit. The AI planner analyzes your request and applies it directly to your image.")

        with st.form("custom_edit_form"):
            txt = st.text_area("Instruction", placeholder="e.g. 'make it to the light mode', 'blur background', 'pencil sketch', 'brighten by 30%'")
            if st.form_submit_button("✨ Apply Edit", type="primary") and txt.strip():
                run_edit(txt.strip(), txt.strip())

    st.divider()
    st.subheader("Version History")
    hc = st.columns(5)
    for i, v in enumerate(versions):
        with hc[i % 5]:
            st.image(v["thumb"], use_container_width=True)
            tag = " ✅ current" if v["id"] == cur["id"] else ""
            st.markdown(f"**v{v['version_no']}**{tag}")
            st.caption(f"{v['created_at'].replace('T', ' ')}\n\n{v['prompt'][:90]}")
            if v["id"] != cur["id"] and st.button("Restore", key=f"r{v['id']}"):
                storage.set_current(rec["id"], v["id"])
                st.session_state["last_note"] = None
                st.rerun()
