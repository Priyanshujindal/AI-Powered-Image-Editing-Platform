"""Retrieval logic: hybrid semantic + keyword search over image captions.

score = 0.75 * semantic (cosine of query vs caption embedding, rescaled)
      + 0.25 * keyword  (fraction of query terms found in caption/tags/filename)

Design notes
* The embedding function is injected, so this module is testable offline.
* If embedding fails (no key, quota, network) we degrade to keyword-only search and
  report mode="keyword" so the UI can say so, instead of showing an error.
* Results below a relevance floor are dropped, and so is anything far behind the
  best hit, so "mountain" doesn't return the whole library.
"""
import json
import re
from dataclasses import dataclass, field

STOP = {"a", "an", "the", "of", "in", "on", "with", "and", "or", "to", "show", "me",
        "photos", "photo", "images", "image", "pictures", "picture", "ones", "that",
        "have", "has", "find", "all", "my", "some", "any", "is", "are", "at", "for"}

SEM_WEIGHT, KW_WEIGHT = 0.75, 0.25
SEM_FLOOR = 0.45     # cosine below this is treated as "unrelated"
BEST_MARGIN = 0.15   # keep hits within this distance of the top score
_query_cache: dict[str, list[float]] = {}


@dataclass
class Hit:
    image: dict
    score: float
    semantic: float = 0.0
    keyword: float = 0.0
    matched: list[str] = field(default_factory=list)


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in STOP]


def _tags(image: dict) -> list[str]:
    raw = image.get("tags") or "[]"
    try:
        return [str(t).lower() for t in (raw if isinstance(raw, list) else json.loads(raw))]
    except (json.JSONDecodeError, TypeError):
        return []


def keyword_score(terms: list[str], image: dict) -> tuple[float, list[str]]:
    if not terms:
        return 0.0, []
    haystack = " ".join([image.get("caption") or "", " ".join(_tags(image)),
                         image.get("name") or ""]).lower()
    words = set(re.findall(r"[a-z0-9]+", haystack))
    matched = [t for t in terms if t in words or any(w.startswith(t) for w in words if len(t) > 3)]
    return len(matched) / len(terms), matched


def embed_query(query: str, embed_fn) -> list[float]:
    """Embed once per distinct query (Streamlit reruns the script on every click)."""
    key = query.strip().lower()
    if key not in _query_cache:
        vec = embed_fn(query)
        if not vec:
            return []
        if len(_query_cache) > 256:
            _query_cache.clear()
        _query_cache[key] = vec
    return _query_cache[key]


def search(query: str, images: list[dict], store, embed_fn, top_k: int = 24):
    """Returns (hits, mode) where mode is 'hybrid' or 'keyword'."""
    query = (query or "").strip()
    if not query or not images:
        return [], "hybrid"
    by_id = {im["id"]: im for im in images}
    terms = tokenize(query) or query.lower().split()

    qvec = []
    try:
        qvec = embed_query(query, embed_fn)
    except Exception:
        qvec = []
    mode = "hybrid" if qvec else "keyword"

    sem = {}
    if qvec:
        for iid, cos in store.query(qvec, top_k=len(images)):
            sem[iid] = cos

    hits = []
    for iid, im in by_id.items():
        kw, matched = keyword_score(terms, im)
        cos = sem.get(iid, 0.0)
        s = max(0.0, (cos - SEM_FLOOR) / (1 - SEM_FLOOR)) if cos else 0.0  # rescale floor..1 -> 0..1
        score = SEM_WEIGHT * s + KW_WEIGHT * kw if mode == "hybrid" else kw
        if score > 0:
            hits.append(Hit(im, round(score, 4), round(cos, 4), round(kw, 4), matched))
    hits.sort(key=lambda h: h.score, reverse=True)
    if not hits:
        return [], mode
    best = hits[0].score
    return [h for h in hits[:top_k] if h.score >= best - BEST_MARGIN], mode
