import json

from PIL import Image

import indexer
from vectorstore import VectorStore


def add(storage, name="a.png", color=(10, 20, 30)):
    return storage.add_image(name, Image.new("RGB", (40, 40), color))


def caption_ok(_img):
    return {"caption": "A red square", "tags": ["red", "square"]}


def embed_ok(text):
    return [1.0, float(len(text) % 7), 0.5]


def test_index_success_marks_ok_and_stores_vector(storage, tmp_path):
    store = VectorStore(tmp_path / "v.json", model="m")
    iid = add(storage)
    assert indexer.index_image(iid, store, caption_ok, embed_ok)
    row = storage.get_image(iid)
    assert row["index_status"] == "ok" and row["caption"] == "A red square" and store.has(iid)


def test_failure_is_recorded_not_raised(storage, tmp_path):
    store = VectorStore(tmp_path / "v.json", model="m")
    iid = add(storage)

    def boom(_):
        raise RuntimeError("429 quota")

    assert indexer.index_image(iid, store, boom, embed_ok) is False
    row = storage.get_image(iid)
    assert row["index_status"] == "failed" and "429" in row["index_error"] and not store.has(iid)


def test_sync_retries_failed_and_cleans_orphans(storage, tmp_path):
    store = VectorStore(tmp_path / "v.json", model="m")
    a, b = add(storage, "a", (1, 1, 1)), add(storage, "b", (2, 2, 2))
    store.add("ghost", [1, 1, 1])                        # vector for a deleted image
    res = indexer.sync(store, caption_ok, embed_ok)
    assert res["indexed"] == 2 and res["failed"] == 0
    assert not store.has("ghost") and store.has(a) and store.has(b)
    assert indexer.sync(store, caption_ok, embed_ok)["indexed"] == 0   # nothing left to do


def test_sync_migrates_week2_embeddings_without_api_calls(storage, tmp_path):
    store = VectorStore(tmp_path / "v.json", model="m")
    iid = add(storage)
    storage.set_caption(iid, "A cat on a sofa", ["cat"], [0.3, 0.2, 0.1])   # Week-2 style row

    def forbidden(*_):
        raise AssertionError("API must not be called")

    res = indexer.sync(store, forbidden, forbidden, call_api=False)
    assert store.has(iid) and storage.get_image(iid)["index_status"] == "ok" and res["pending"] == 0


def test_placeholder_caption_is_reindexed(storage, tmp_path):
    store = VectorStore(tmp_path / "v.json", model="m")
    iid = add(storage)
    storage.set_caption(iid, indexer.FALLBACK_CAPTION, ["image"], [0.1, 0.1, 0.1])
    indexer.sync(store, caption_ok, embed_ok)
    assert storage.get_image(iid)["caption"] == "A red square"


def test_remove_image_deletes_vector(storage, tmp_path):
    store = VectorStore(tmp_path / "v.json", model="m")
    iid = add(storage)
    indexer.index_image(iid, store, caption_ok, embed_ok)
    indexer.remove_image(iid, store)
    assert storage.get_image(iid) is None and not store.has(iid)
