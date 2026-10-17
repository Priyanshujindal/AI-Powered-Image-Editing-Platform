import json

from vectorstore import VectorStore


def test_add_query_delete_and_persistence(tmp_path):
    p = tmp_path / "v.json"
    s = VectorStore(p, model="m")
    s.add("a", [1, 0, 0])
    s.add("b", [0, 1, 0])
    s.add("c", [0.9, 0.1, 0])
    top = s.query([1, 0, 0], top_k=2)
    assert [i for i, _ in top] == ["a", "c"]
    assert top[0][1] > 0.99

    s2 = VectorStore(p, model="m")          # reload from disk
    assert len(s2) == 3 and s2.has("b")
    s2.delete("a")
    assert [i for i, _ in s2.query([1, 0, 0], top_k=1)] == ["c"]


def test_corrupt_file_is_quarantined(tmp_path):
    p = tmp_path / "v.json"
    p.write_text("{not json")
    s = VectorStore(p, model="m")
    assert len(s) == 0
    assert (tmp_path / "v.corrupt").exists()
    s.add("x", [1, 2])                       # still usable
    assert VectorStore(p, model="m").has("x")


def test_model_change_invalidates_old_vectors(tmp_path):
    p = tmp_path / "v.json"
    VectorStore(p, model="old").add("a", [1, 0])
    assert len(VectorStore(p, model="new")) == 0


def test_dimension_mismatch_and_zero_vectors_are_safe(tmp_path):
    s = VectorStore(tmp_path / "v.json", model="m")
    s.add("a", [1, 0, 0])
    assert s.query([1, 0]) == []             # wrong dimension
    assert s.query([0, 0, 0]) == []          # zero query
    assert json.loads((tmp_path / "v.json").read_text())["model"] == "m"
