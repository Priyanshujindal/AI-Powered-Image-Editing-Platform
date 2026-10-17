import json

import search as S
from vectorstore import VectorStore

# Tiny fake "embedding space": axis 0 = beach/sea, 1 = mountain, 2 = ui/screen
VOCAB = {"beach": 0, "sea": 0, "sunset": 0, "ocean": 0, "mountain": 1, "snow": 1, "peak": 1,
         "login": 2, "screen": 2, "form": 2, "ui": 2}


def fake_embed(text):
    v = [0.05, 0.05, 0.05]
    for w in S.tokenize(text):
        if w in VOCAB:
            v[VOCAB[w]] += 1
    return v


def make_lib(tmp_path):
    imgs = [
        {"id": "beach", "name": "IMG_1.jpg", "caption": "Sunset over the ocean with a sandy beach", "tags": json.dumps(["beach", "sunset"])},
        {"id": "mtn", "name": "IMG_2.jpg", "caption": "Snowy mountain peak at dawn", "tags": json.dumps(["mountain", "snow"])},
        {"id": "ui", "name": "shot.png", "caption": "Login screen with a form", "tags": json.dumps(["login", "ui"])},
    ]
    store = VectorStore(tmp_path / "v.json", model="fake")
    for im in imgs:
        store.add(im["id"], fake_embed(im["caption"] + " " + " ".join(json.loads(im["tags"]))))
    return imgs, store


def ids(hits):
    return [h.image["id"] for h in hits]


def test_semantic_match_beats_unrelated(tmp_path):
    imgs, store = make_lib(tmp_path)
    hits, mode = S.search("sea", imgs, store, fake_embed)   # word 'sea' is in no caption
    assert mode == "hybrid"
    assert ids(hits)[0] == "beach"           # found purely by meaning
    assert "mtn" not in ids(hits)


def test_natural_language_query_strips_filler_words(tmp_path):
    imgs, store = make_lib(tmp_path)
    hits, _ = S.search("show me the ones with a mountain", imgs, store, fake_embed)
    assert ids(hits) == ["mtn"]
    assert "mountain" in hits[0].matched


def test_keyword_fallback_when_embeddings_fail(tmp_path):
    imgs, store = make_lib(tmp_path)

    def broken(_):
        raise RuntimeError("429 quota")

    S._query_cache.clear()
    hits, mode = S.search("login", imgs, store, broken)
    assert mode == "keyword" and ids(hits) == ["ui"]


def test_query_embedding_is_cached(tmp_path):
    imgs, store = make_lib(tmp_path)
    calls = []

    def counting(t):
        calls.append(t)
        return fake_embed(t)

    S._query_cache.clear()
    S.search("beach", imgs, store, counting)
    S.search("Beach ", imgs, store, counting)
    assert len(calls) == 1


def test_empty_inputs(tmp_path):
    imgs, store = make_lib(tmp_path)
    assert S.search("", imgs, store, fake_embed)[0] == []
    assert S.search("beach", [], store, fake_embed)[0] == []
