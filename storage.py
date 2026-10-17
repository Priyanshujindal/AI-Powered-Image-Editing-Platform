"""Local storage: SQLite for metadata/versions, filesystem for image files.

Week 3 upgrades
* Connections are always closed (sqlite3's `with` only commits) and use WAL + foreign keys.
* Versioned schema migrations: existing databases from Week 2 are upgraded in place.
* Versions record the engine note, pixel size and file size; parent_id keeps the edit lineage.
* Index health per image (index_status / index_error) so failures are visible and retryable.
* Upload validation (decode check, EXIF rotation, size cap) and duplicate detection by content hash.
* Files are written before DB rows and cleaned up if the insert fails, so no orphans.
"""
import hashlib
import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

log = logging.getLogger("pixelprompt.storage")

DATA = Path("data")
IMG_DIR, THUMB_DIR = DATA / "images", DATA / "thumbs"
for _p in (IMG_DIR, THUMB_DIR):
    _p.mkdir(parents=True, exist_ok=True)

MAX_SIDE = 6000          # px; larger uploads are scaled down to protect memory
SCHEMA_VERSION = 2


class StorageError(Exception):
    """Raised for user-facing storage problems (bad file, missing image...)."""


@contextmanager
def _db():
    c = sqlite3.connect(DATA / "app.db", timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def _fix_paths(row: dict) -> dict:
    """Paths are stored as POSIX strings; rows written on Windows (backslash paths) are normalised on read
    so the same database works on Linux / Streamlit Cloud."""
    for k in ("path", "thumb"):
        if row.get(k):
            row[k] = str(row[k]).replace("\\", "/")
    return row


def _columns(c, table):
    return {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}


def init():
    with _db() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript("""
        CREATE TABLE IF NOT EXISTS images(
            id TEXT PRIMARY KEY, name TEXT, caption TEXT DEFAULT '', tags TEXT DEFAULT '[]',
            embedding TEXT, current_version TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS versions(
            id TEXT PRIMARY KEY, image_id TEXT REFERENCES images(id) ON DELETE CASCADE,
            version_no INTEGER, path TEXT, thumb TEXT, prompt TEXT, parent_id TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
        """)
        # ---- migrations (idempotent: add only what is missing) ----
        img_cols, ver_cols = _columns(c, "images"), _columns(c, "versions")
        for col, ddl in {
            "index_status": "TEXT DEFAULT 'pending'",   # pending | ok | failed
            "index_error": "TEXT",
            "content_hash": "TEXT",
            "updated_at": "TEXT",
        }.items():
            if col not in img_cols:
                c.execute(f"ALTER TABLE images ADD COLUMN {col} {ddl}")
        for col, ddl in {"note": "TEXT", "width": "INTEGER", "height": "INTEGER",
                         "size_bytes": "INTEGER"}.items():
            if col not in ver_cols:
                c.execute(f"ALTER TABLE versions ADD COLUMN {col} {ddl}")
        c.execute("CREATE INDEX IF NOT EXISTS idx_versions_image ON versions(image_id, version_no)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_images_hash ON images(content_hash)")
        c.execute("INSERT OR REPLACE INTO meta VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))


# ---------------------------------------------------------------- files
def _normalise(img: Image.Image) -> Image.Image:
    return img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") else img.convert("RGB")


def _save_files(img: Image.Image, image_id: str):
    vid = uuid.uuid4().hex[:10]
    img = _normalise(img)
    path, thumb = IMG_DIR / f"{image_id}_{vid}.png", THUMB_DIR / f"{image_id}_{vid}.jpg"
    try:
        img.save(path)
        t = img.convert("RGB")
        t.thumbnail((360, 360))
        t.save(thumb, quality=85)
    except Exception as e:
        path.unlink(missing_ok=True)
        thumb.unlink(missing_ok=True)
        raise StorageError(f"Could not save image: {e}") from e
    return vid, path, thumb, img.size


def prepare_upload(img: Image.Image) -> Image.Image:
    """Validate + normalise an uploaded image (EXIF rotation, size cap)."""
    try:
        img.load()
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        raise StorageError(f"Not a readable image: {e}") from e
    if max(img.size) > MAX_SIDE:
        img = img.copy()
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def content_hash(img: Image.Image) -> str:
    im = _normalise(img)
    h = hashlib.sha256(f"{im.size}{im.mode}".encode())
    h.update(im.tobytes())
    return h.hexdigest()


def find_duplicate(img: Image.Image):
    h = content_hash(img)
    with _db() as c:
        r = c.execute("SELECT id, name FROM images WHERE content_hash=?", (h,)).fetchone()
    return dict(r) if r else None


# ---------------------------------------------------------------- images / versions
def add_image(name: str, img: Image.Image) -> str:
    img = prepare_upload(img)
    image_id = uuid.uuid4().hex[:10]
    now = datetime.now().isoformat(timespec="seconds")
    with _db() as c:
        c.execute("INSERT INTO images(id,name,current_version,created_at,updated_at,content_hash,index_status)"
                  " VALUES(?,?,?,?,?,?, 'pending')",
                  (image_id, name, None, now, now, content_hash(img)))
    try:
        add_version(image_id, img, "Original upload", None, note="Original")
    except Exception:
        with _db() as c:
            c.execute("DELETE FROM images WHERE id=?", (image_id,))
        raise
    return image_id


def add_version(image_id: str, img: Image.Image, prompt: str, parent_id, note: str = "") -> str:
    if not get_image(image_id):
        raise StorageError("Image no longer exists.")
    vid, path, thumb, (w, h) = _save_files(img, image_id)
    try:
        with _db() as c:
            n = c.execute("SELECT COALESCE(MAX(version_no),0)+1 FROM versions WHERE image_id=?",
                          (image_id,)).fetchone()[0]
            now = datetime.now().isoformat(timespec="seconds")
            c.execute("INSERT INTO versions(id,image_id,version_no,path,thumb,prompt,parent_id,created_at,"
                      "note,width,height,size_bytes) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                      (vid, image_id, n, path.as_posix(), thumb.as_posix(), prompt, parent_id, now,
                       note or "", w, h, path.stat().st_size))
            c.execute("UPDATE images SET current_version=?, updated_at=? WHERE id=?", (vid, now, image_id))
    except Exception:
        path.unlink(missing_ok=True)
        thumb.unlink(missing_ok=True)
        raise
    return vid


def set_current(image_id: str, version_id: str):
    with _db() as c:
        ok = c.execute("SELECT 1 FROM versions WHERE id=? AND image_id=?", (version_id, image_id)).fetchone()
        if not ok:
            raise StorageError("That version does not belong to this image.")
        c.execute("UPDATE images SET current_version=?, updated_at=? WHERE id=?",
                  (version_id, datetime.now().isoformat(timespec="seconds"), image_id))


def rename_image(image_id: str, name: str):
    name = (name or "").strip()
    if not name:
        raise StorageError("Name cannot be empty.")
    with _db() as c:
        c.execute("UPDATE images SET name=? WHERE id=?", (name[:200], image_id))


def set_caption(image_id: str, caption: str, tags: list, embedding=None):
    with _db() as c:
        c.execute("UPDATE images SET caption=?, tags=?, embedding=COALESCE(?, embedding) WHERE id=?",
                  (caption, json.dumps(tags), json.dumps(embedding) if embedding else None, image_id))


def set_index_status(image_id: str, status: str, error: str | None = None):
    with _db() as c:
        c.execute("UPDATE images SET index_status=?, index_error=? WHERE id=?",
                  (status, (error or "")[:500] or None, image_id))


def list_images():
    with _db() as c:
        return [_fix_paths(dict(r)) for r in c.execute("""
            SELECT i.*, v.thumb AS thumb, v.path AS path FROM images i
            LEFT JOIN versions v ON v.id = i.current_version ORDER BY i.created_at DESC""")]


def get_image(image_id: str):
    if not image_id:
        return None
    with _db() as c:
        r = c.execute("""SELECT i.*, v.path AS path FROM images i
                         LEFT JOIN versions v ON v.id = i.current_version WHERE i.id=?""",
                      (image_id,)).fetchone()
        return _fix_paths(dict(r)) if r else None


def list_versions(image_id: str):
    with _db() as c:
        return [_fix_paths(dict(r)) for r in c.execute(
            "SELECT * FROM versions WHERE image_id=? ORDER BY version_no DESC", (image_id,))]


def delete_image(image_id: str):
    versions = list_versions(image_id)
    with _db() as c:
        c.execute("DELETE FROM versions WHERE image_id=?", (image_id,))
        c.execute("DELETE FROM images WHERE id=?", (image_id,))
    for v in versions:  # remove files only after the DB commit succeeded
        Path(v["path"]).unlink(missing_ok=True)
        Path(v["thumb"]).unlink(missing_ok=True)


def index_stats() -> dict:
    with _db() as c:
        rows = c.execute("SELECT COALESCE(index_status,'pending') s, COUNT(*) n FROM images GROUP BY s").fetchall()
    out = {"ok": 0, "pending": 0, "failed": 0}
    out.update({r["s"]: r["n"] for r in rows})
    out["total"] = sum(out.values())
    return out
