import os
from pathlib import Path

import streamlit as st
from PIL import Image

import ai
import indexer
import prompts
import search as search_mod
import storage
from vectorstore import VectorStore

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
st.session_state.setdefault("q", "")


@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore(model=ai.EMBED_MODEL)


store = get_store()


def _caption(img):
    return ai.caption_image(img, strict=True)


def _embed(text):
    return ai.embed(text, strict=True)


def _embed_query(text):
    return ai.embed(text, query=True)   # non-strict: search degrades to keywords on failure


def go(page, image_id=None):
    st.session_state.page, st.session_state.selected = page, image_id
    st.session_state["last_note"] = None
    st.rerun()


def index_image(image_id: str, recaption: bool = True) -> bool:
    """Caption + embed one image; the outcome is stored on the image (never raises)."""
    return indexer.index_image(image_id, store, _caption, _embed, recaption=recaption)


# One free pass per session: drop orphan vectors, migrate Week-2 embeddings (no API calls).
if not st.session_state.get("synced"):
    indexer.sync(store, _caption, _embed, call_api=False)
    st.session_state["synced"] = True


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
        added = skipped = failed = 0
        for i, f in enumerate(up):
            try:
                img = Image.open(f)
                dup = storage.find_duplicate(storage.prepare_upload(img))
                if dup:
                    st.warning(f"{f.name} is already in your library as '{dup['name']}': skipped.")
                    skipped += 1
                else:
                    iid = storage.add_image(f.name, img)
                    added += 1
                    if ai.has_key() and not index_image(iid):
                        st.warning(f"{f.name}: added, but search indexing failed. Use Rebuild index to retry.")
            except storage.StorageError as e:
                st.error(f"{f.name}: {e}")
                failed += 1
            except Exception as e:
                st.error(f"{f.name}: unexpected error ({e})")
                failed += 1
            bar.progress((i + 1) / len(up))
        st.session_state["flash"] = f"Added {added}, skipped {skipped} duplicate(s), {failed} failed."
        st.rerun()

    # ---- search index health ----
    st.divider()
    st.markdown("**🔎 Search index**")
    stats = storage.index_stats()
    if stats["total"]:
        st.progress(stats["ok"] / stats["total"], text=f"{stats['ok']}/{stats['total']} images searchable")
        if stats["failed"] or stats["pending"]:
            st.caption(f"{stats['failed']} failed · {stats['pending']} pending")
    if st.button("♻️ Rebuild index", use_container_width=True, disabled=not ai.has_key(),
                 help="Captions and embeds images that are missing from the index or failed earlier."):
        bar = st.progress(0.0, text="Indexing...")
        res = indexer.sync(store, _caption, _embed,
                           progress=lambda f, n: bar.progress(f, text=f"Indexing {n}"))
        st.session_state["flash"] = f"Index rebuilt: {res['indexed']} indexed, {res['failed']} failed."
        st.rerun()

# ---------------- Library ----------------
def _set_query(text):
    st.session_state["q"] = text


def _top_tags(images, n=6):
    counts = {}
    for im in images:
        for t in search_mod._tags(im):
            counts[t] = counts.get(t, 0) + 1
    return [t for t, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n]]


if st.session_state.page == "library":
    st.header("Photo Library")
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    all_images = storage.list_images()
    query = st.text_input("🔎 Search by description", key="q",
                          placeholder="e.g. sunset, mountain, show me the ones with a login form")
    if not query and all_images:
        tags = _top_tags(all_images)
        if tags:
            st.caption("Try:")
            tcols = st.columns(len(tags))
            for c, t in zip(tcols, tags):
                c.button(t, key=f"tag_{t}", on_click=_set_query, args=(t,), use_container_width=True)

    hits, mode, by_id = [], "hybrid", {}
    images = all_images
    if query.strip() and all_images:
        with st.spinner("Searching..."):
            try:
                hits, mode = search_mod.search(query, all_images, store, _embed_query)
            except Exception as e:
                st.error(f"Search failed: {e}")
        images = [h.image for h in hits]
        by_id = {h.image["id"]: h for h in hits}
        label = "semantic + keyword" if mode == "hybrid" else "keyword only (embeddings unavailable)"
        st.caption(f"{len(hits)} of {len(all_images)} image(s) match · {label}")
        if mode == "keyword" and ai.has_key():
            st.warning("Semantic search is temporarily unavailable (rate limit or network). Showing keyword matches.")
        unindexed = storage.index_stats()
        if unindexed["failed"] or unindexed["pending"]:
            st.info(f"{unindexed['failed'] + unindexed['pending']} image(s) are not in the search index yet. "
                    "Use **Rebuild index** in the sidebar.")
    if not images:
        st.info("No images yet. Upload some from the sidebar." if not query.strip() else
                "No matches. Try a different word, or describe the scene (e.g. 'a person at a beach').")
    cols = st.columns(4)
    for i, im in enumerate(images):
        with cols[i % 4]:
            st.image(im["thumb"], use_container_width=True)
            if query.strip() and im["id"] in by_id:
                h = by_id[im["id"]]
                st.progress(min(h.score, 1.0), text=f"relevance {h.score:.0%}")
            badge = {"failed": " ⚠️", "pending": " ⏳"}.get(im.get("index_status"), "")
            st.caption((im["caption"] or im["name"]) + badge)
            if st.button("Open Studio", key=f"open_{im['id']}", use_container_width=True):
                go("detail", im["id"])

# ---------------- Detail / Studio ----------------
else:
    rec = storage.get_image(st.session_state.selected)
    if not rec:
        go("library")
    versions = storage.list_versions(rec["id"])
    cur = next((v for v in versions if v["id"] == rec["current_version"]), versions[0])

    head, delcol = st.columns([6, 1])
    head.header(rec["name"])
    if delcol.button("🗑 Delete"):
        indexer.remove_image(rec["id"], store)
        go("library")
    if rec["caption"]:
        st.caption(f"🤖 {rec['caption']}")
        tags = search_mod._tags(rec)
        if tags:
            st.markdown(" ".join(f"`{t}`" for t in tags))
    status = rec.get("index_status") or "pending"
    if status != "ok":
        st.warning(f"Search index status: **{status}**" + (f" ({rec['index_error']})" if rec.get("index_error") else ""))
    ic1, ic2 = st.columns([1, 1])
    if ic1.button("🔄 Re-caption & re-index", disabled=not ai.has_key()):
        if index_image(rec["id"]):
            st.session_state["last_note"] = "Caption and search index refreshed."
        else:
            st.session_state["last_note"] = "Indexing failed; try again in a minute."
        st.rerun()
    with ic2.popover("✏️ Rename"):
        new_name = st.text_input("New name", value=rec["name"], key="rename_box")
        if st.button("Save name"):
            try:
                storage.rename_image(rec["id"], new_name)
                st.rerun()
            except storage.StorageError as e:
                st.error(str(e))

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
                storage.add_version(rec["id"], out, label, cur["id"], note=note or "")
                try:
                    index_image(rec["id"])
                except Exception:
                    pass
            st.session_state["last_note"] = note
            st.rerun()
        except storage.StorageError as e:
            st.error(str(e))
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
            dims = f" · {v['width']}×{v['height']}" if v.get("width") else ""
            st.caption(f"{v['created_at'].replace('T', ' ')}{dims}\n\n{v['prompt'][:90]}")
            if v["id"] != cur["id"] and st.button("Restore", key=f"r{v['id']}"):
                try:
                    storage.set_current(rec["id"], v["id"])
                except storage.StorageError as e:
                    st.error(str(e))
                st.session_state["last_note"] = None
                st.rerun()
