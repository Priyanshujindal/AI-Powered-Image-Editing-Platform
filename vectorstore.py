"""Lightweight JSON-backed vector store (no extra services, no native deps).

Layout of data/vectors.json:
{
  "version": 1,
  "model": "gemini-embedding-001",
  "items": { "<image_id>": {"vector": [...], "text": "...", "updated_at": "..."} }
}

* Cosine similarity is computed with NumPy over one stacked matrix (fast for <=50k images).
* Writes are atomic (temp file + os.replace) and guarded by a lock, so a crash or a
  Streamlit rerun mid-write can never leave a half-written index.
* A corrupt file is quarantined (renamed *.corrupt) and the store starts empty,
  because the index can always be rebuilt from captions.
* To scale up, replace this class with FAISS/Chroma: only add/delete/query are used.
"""
import json
import os
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path("data") / "vectors.json"


class VectorStore:
    def __init__(self, path: Path | str = DEFAULT_PATH, model: str = ""):
        self.path = Path(path)
        self.model = model
        self._lock = threading.RLock()
        self._items: dict[str, dict] = {}
        self._matrix = None  # cached (ids, normalised matrix)
        self._load()

    # ---------- persistence ----------
    def _load(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.model and data.get("model") not in (None, "", self.model):
                # Different embedding model => old vectors are not comparable.
                self._items = {}
                return
            self._items = data.get("items", {})
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            try:
                os.replace(self.path, self.path.with_suffix(".corrupt"))
            except OSError:
                pass
            self._items = {}

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        payload = {"version": 1, "model": self.model, "items": self._items}
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.path)
        self._matrix = None

    # ---------- public API ----------
    def add(self, image_id: str, vector, text: str = ""):
        v = [float(x) for x in vector]
        if not v:
            raise ValueError("empty vector")
        with self._lock:
            self._items[image_id] = {
                "vector": v, "text": text,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._save()

    def delete(self, image_id: str):
        with self._lock:
            if self._items.pop(image_id, None) is not None:
                self._save()

    def has(self, image_id: str) -> bool:
        return image_id in self._items

    def ids(self) -> set[str]:
        return set(self._items)

    def __len__(self):
        return len(self._items)

    def _build(self):
        if self._matrix is None:
            ids, rows = [], []
            dim = None
            for k, it in self._items.items():
                v = np.asarray(it["vector"], dtype=np.float32)
                if dim is None:
                    dim = v.shape[0]
                if v.shape[0] != dim:  # skip malformed rows rather than crash
                    continue
                n = np.linalg.norm(v)
                if n > 0:
                    ids.append(k)
                    rows.append(v / n)
            self._matrix = (ids, np.vstack(rows) if rows else np.zeros((0, dim or 1), np.float32))
        return self._matrix

    def query(self, vector, top_k: int = 12) -> list[tuple[str, float]]:
        """Return [(image_id, cosine_similarity)] best first."""
        q = np.asarray(vector, dtype=np.float32)
        n = np.linalg.norm(q)
        with self._lock:
            ids, mat = self._build()
            if not ids or n == 0 or mat.shape[1] != q.shape[0]:
                return []
            sims = mat @ (q / n)
        order = np.argsort(-sims)[:top_k]
        return [(ids[i], float(sims[i])) for i in order]
