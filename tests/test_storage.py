import shutil
import sqlite3
from pathlib import Path

import pytest
from PIL import Image


def img(color=(200, 30, 30), size=(64, 48)):
    return Image.new("RGB", size, color)


def test_upload_creates_v1_with_metadata(storage):
    iid = storage.add_image("a.png", img())
    vs = storage.list_versions(iid)
    assert len(vs) == 1 and vs[0]["version_no"] == 1
    assert (vs[0]["width"], vs[0]["height"]) == (64, 48) and vs[0]["size_bytes"] > 0
    assert storage.get_image(iid)["index_status"] == "pending"


def test_versions_are_immutable_lineage_and_restore(storage):
    iid = storage.add_image("a.png", img())
    v1 = storage.get_image(iid)["current_version"]
    v2 = storage.add_version(iid, img((0, 0, 255)), "make it blue", v1, note="Direct engine")
    vs = storage.list_versions(iid)
    assert [v["version_no"] for v in vs] == [2, 1]
    assert vs[0]["parent_id"] == v1 and vs[0]["note"] == "Direct engine"
    assert Path(vs[1]["path"]).exists()                       # original untouched
    storage.set_current(iid, v1)
    assert storage.get_image(iid)["current_version"] == v1
    assert v2 != v1


def test_restore_rejects_foreign_version(storage):
    a, b = storage.add_image("a", img()), storage.add_image("b", img((1, 2, 3)))
    with pytest.raises(storage.StorageError):
        storage.set_current(a, storage.get_image(b)["current_version"])


def test_duplicate_detection(storage):
    storage.add_image("a.png", img())
    assert storage.find_duplicate(storage.prepare_upload(img()))["name"] == "a.png"
    assert storage.find_duplicate(storage.prepare_upload(img((9, 9, 9)))) is None


def test_failed_version_insert_leaves_no_orphan_files(storage):
    with pytest.raises(storage.StorageError):
        storage.add_version("missing", img(), "x", None)
    assert list(storage.IMG_DIR.iterdir()) == []


def test_delete_removes_files_and_rows(storage):
    iid = storage.add_image("a.png", img())
    storage.add_version(iid, img((0, 0, 1)), "x", None)
    storage.delete_image(iid)
    assert storage.get_image(iid) is None
    assert list(storage.IMG_DIR.iterdir()) == [] and list(storage.THUMB_DIR.iterdir()) == []


def test_index_stats_and_rename(storage):
    iid = storage.add_image("a.png", img())
    storage.set_index_status(iid, "failed", "boom")
    assert storage.index_stats()["failed"] == 1
    storage.rename_image(iid, "  New name ")
    assert storage.get_image(iid)["name"] == "New name"
    with pytest.raises(storage.StorageError):
        storage.rename_image(iid, "  ")


def test_oversized_upload_is_scaled(storage):
    big = Image.new("RGB", (storage.MAX_SIDE + 500, 10))
    assert max(storage.prepare_upload(big).size) == storage.MAX_SIDE


def test_migrates_week2_database(storage, tmp_path):
    """A Week-2 database (old schema, real rows) is upgraded in place without data loss."""
    old = tmp_path / "app.db"
    old.unlink()
    c = sqlite3.connect(old)
    c.executescript("""
      CREATE TABLE images(id TEXT PRIMARY KEY, name TEXT, caption TEXT DEFAULT '', tags TEXT DEFAULT '[]',
                          embedding TEXT, current_version TEXT, created_at TEXT);
      CREATE TABLE versions(id TEXT PRIMARY KEY, image_id TEXT, version_no INTEGER, path TEXT, thumb TEXT,
                            prompt TEXT, parent_id TEXT, created_at TEXT);
      INSERT INTO images VALUES('i1','old.png','A cat','["cat"]','[0.1,0.2]','v1','2026-01-01T00:00:00');
      INSERT INTO versions VALUES('v1','i1',1,'p.png','t.jpg','Original upload',NULL,'2026-01-01T00:00:00');
    """)
    c.commit(); c.close()
    storage.init(); storage.init()                      # idempotent
    row = storage.get_image("i1")
    assert row["caption"] == "A cat" and row["index_status"] == "pending"
    assert storage.list_versions("i1")[0]["note"] is None


def test_windows_paths_are_normalised_on_read(storage):
    iid = storage.add_image("a.png", img())
    vid = storage.get_image(iid)["current_version"]
    with storage._db() as c:   # simulate a row written on Windows
        c.execute("UPDATE versions SET path=?, thumb=? WHERE id=?",
                  ("data\\images\\x.png", "data\\thumbs\\x.jpg", vid))
    row, ver = storage.get_image(iid), storage.list_versions(iid)[0]
    assert row["path"] == "data/images/x.png" and ver["thumb"] == "data/thumbs/x.jpg"
    assert storage.list_images()[0]["thumb"] == "data/thumbs/x.jpg"
