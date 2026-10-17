"""Indexing pipeline: image -> caption + tags (vision LLM) -> embedding -> vector store.

Each step records its outcome in SQLite (index_status/index_error) so a failed
caption or a rate-limited embedding is visible in the UI and can be retried with
"Rebuild index", instead of silently producing an unsearchable image.
"""
import json
import logging

from PIL import Image

import storage
from vectorstore import VectorStore

log = logging.getLogger("pixelprompt.indexer")
FALLBACK_CAPTION = "Uploaded image"


def index_text(rec: dict, caption: str, tags: list) -> str:
    """The text that gets embedded: caption + tags + filename give the model more to match on."""
    return f"{caption}\nTags: {', '.join(tags)}\nFile: {rec.get('name', '')}"


def index_image(image_id: str, store: VectorStore, caption_fn, embed_fn, recaption: bool = True) -> bool:
    """Caption (optional) and embed one image. Returns True on success; never raises."""
    rec = storage.get_image(image_id)
    if not rec:
        return False
    try:
        caption, tags = rec.get("caption") or "", json.loads(rec.get("tags") or "[]")
        if recaption or not caption or caption == FALLBACK_CAPTION:
            with Image.open(rec["path"]) as im:
                info = caption_fn(im)          # strict: raises on API failure
            caption, tags = info["caption"], info["tags"]
            if not caption:
                raise ValueError("Model returned an empty caption")
            storage.set_caption(image_id, caption, tags)
        vec = embed_fn(index_text(rec, caption, tags))   # strict: raises on API failure
        store.add(image_id, vec, text=caption)
        storage.set_index_status(image_id, "ok")
        return True
    except Exception as e:
        log.warning("indexing %s failed: %s", image_id, e)
        storage.set_index_status(image_id, "failed", f"{type(e).__name__}: {e}")
        return False


def remove_image(image_id: str, store: VectorStore):
    """Delete image + its vector together so search never returns ghosts."""
    storage.delete_image(image_id)
    store.delete(image_id)


def sync(store: VectorStore, caption_fn, embed_fn, progress=None, call_api: bool = True) -> dict:
    """Bring the vector index in line with the library.

    1. Drop vectors for images that no longer exist.
    2. Migrate Week-2 embeddings (stored inside SQLite) into the vector store for free.
    3. (Re)index anything that is missing, failed, or still has the placeholder caption.
    """
    images = storage.list_images()
    live = {im["id"] for im in images}
    for orphan in store.ids() - live:
        store.delete(orphan)

    todo = []
    for im in images:
        if store.has(im["id"]) and im.get("index_status") == "ok":
            continue
        legacy = im.get("embedding")
        good_caption = im.get("caption") and im["caption"] != FALLBACK_CAPTION
        if legacy and good_caption and not store.has(im["id"]):
            try:
                store.add(im["id"], json.loads(legacy), text=im["caption"])
                storage.set_index_status(im["id"], "ok")
                continue
            except Exception:
                pass
        if store.has(im["id"]) and good_caption:
            storage.set_index_status(im["id"], "ok")
            continue
        todo.append(im)

    if not call_api:   # free pass only: cleanup + legacy migration, no quota used
        return {"checked": len(images), "indexed": 0, "failed": 0, "pending": len(todo)}
    done = failed = 0
    for i, im in enumerate(todo):
        ok = index_image(im["id"], store, caption_fn, embed_fn, recaption=not (im.get("caption") and im["caption"] != FALLBACK_CAPTION))
        done += ok
        failed += (not ok)
        if progress:
            progress((i + 1) / len(todo), im["name"])
    return {"checked": len(images), "indexed": done, "failed": failed, "pending": 0}
