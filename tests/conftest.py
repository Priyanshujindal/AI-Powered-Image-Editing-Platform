import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def storage(tmp_path, monkeypatch):
    """storage.py pointed at an empty temp directory."""
    import storage as st
    monkeypatch.setattr(st, "DATA", tmp_path)
    monkeypatch.setattr(st, "IMG_DIR", tmp_path / "images")
    monkeypatch.setattr(st, "THUMB_DIR", tmp_path / "thumbs")
    st.IMG_DIR.mkdir()
    st.THUMB_DIR.mkdir()
    st.init()
    return st
