"""Local storage: SQLite for metadata/versions, filesystem for image files."""
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image

DATA = Path("data")
IMG_DIR, THUMB_DIR = DATA / "images", DATA / "thumbs"
for _p in (IMG_DIR, THUMB_DIR):
    _p.mkdir(parents=True, exist_ok=True)


def _db():
    c = sqlite3.connect(DATA / "app.db")
    c.row_factory = sqlite3.Row
    return c


def init():
    with _db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS images(
            id TEXT PRIMARY KEY, name TEXT, caption TEXT DEFAULT '', tags TEXT DEFAULT '[]',
            embedding TEXT, current_version TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS versions(
            id TEXT PRIMARY KEY, image_id TEXT, version_no INTEGER, path TEXT, thumb TEXT,
            prompt TEXT, parent_id TEXT, created_at TEXT);
        """)


def _save(img: Image.Image, image_id: str):
    vid = uuid.uuid4().hex[:10]
    img = img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") else img.convert("RGB")
    path, thumb = IMG_DIR / f"{image_id}_{vid}.png", THUMB_DIR / f"{image_id}_{vid}.jpg"
    img.save(path)
    t = img.convert("RGB")
    t.thumbnail((360, 360))
    t.save(thumb, quality=85)
    return vid, str(path), str(thumb)


def add_image(name: str, img: Image.Image) -> str:
    image_id = uuid.uuid4().hex[:10]
    now = datetime.now().isoformat(timespec="seconds")
    with _db() as c:
        c.execute("INSERT INTO images(id,name,current_version,created_at) VALUES(?,?,?,?)",
                  (image_id, name, None, now))
    add_version(image_id, img, "Original upload", None)
    return image_id


def add_version(image_id: str, img: Image.Image, prompt: str, parent_id) -> str:
    vid, path, thumb = _save(img, image_id)
    with _db() as c:
        n = c.execute("SELECT COALESCE(MAX(version_no),0)+1 FROM versions WHERE image_id=?",
                      (image_id,)).fetchone()[0]
        c.execute("INSERT INTO versions VALUES(?,?,?,?,?,?,?,?)",
                  (vid, image_id, n, path, thumb, prompt, parent_id,
                   datetime.now().isoformat(timespec="seconds")))
        c.execute("UPDATE images SET current_version=? WHERE id=?", (vid, image_id))
    return vid


def set_current(image_id: str, version_id: str):
    with _db() as c:
        c.execute("UPDATE images SET current_version=? WHERE id=?", (version_id, image_id))


def set_caption(image_id: str, caption: str, tags: list, embedding: list | None):
    with _db() as c:
        c.execute("UPDATE images SET caption=?, tags=?, embedding=? WHERE id=?",
                  (caption, json.dumps(tags), json.dumps(embedding) if embedding else None, image_id))


def list_images():
    with _db() as c:
        return [dict(r) for r in c.execute("""
            SELECT i.*, v.thumb AS thumb, v.path AS path FROM images i
            LEFT JOIN versions v ON v.id = i.current_version ORDER BY i.created_at DESC""")]


def get_image(image_id: str):
    with _db() as c:
        r = c.execute("""SELECT i.*, v.path AS path FROM images i
                         LEFT JOIN versions v ON v.id = i.current_version WHERE i.id=?""",
                      (image_id,)).fetchone()
        return dict(r) if r else None


def list_versions(image_id: str):
    with _db() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM versions WHERE image_id=? ORDER BY version_no DESC", (image_id,))]


def delete_image(image_id: str):
    for v in list_versions(image_id):
        Path(v["path"]).unlink(missing_ok=True)
        Path(v["thumb"]).unlink(missing_ok=True)
    with _db() as c:
        c.execute("DELETE FROM versions WHERE image_id=?", (image_id,))
        c.execute("DELETE FROM images WHERE id=?", (image_id,))
