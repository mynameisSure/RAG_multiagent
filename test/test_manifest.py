"""验证文档 manifest 能够基于文件哈希识别内容变化。"""

from pathlib import Path

from RAG_multiagent.storage.manifest import DocumentManifest


def test_manifest_tracks_hash(tmp_path: Path):
    """文件首次写入后应被记录，内容变化后应重新视为未入库。"""
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    m = DocumentManifest(tmp_path / "manifest.json")
    assert not m.is_seen(f)
    m.mark_seen(f, chunks=1)
    assert m.is_seen(f)
    f.write_text("hello world", encoding="utf-8")
    assert not m.is_seen(f)
