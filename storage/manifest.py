import hashlib
import json
from pathlib import Path
from typing import Any


class DocumentManifest:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.decoder.JSONDecodeError:
            return {}

    @staticmethod
    def fingerprint(file_path: Path) -> str:
        h = hashlib.sha1()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    def is_seen(self, file_path: Path) -> bool:
        key = str(file_path.resolve())
        return self._data.get(key, {}).get("sha256") == self.fingerprint(file_path)

    def mark_seen(self, file_path: Path, chunks: int) -> None:
        key = str(file_path.resolve())
        self._data[key] = {"sha256": self.fingerprint(file_path), "chunks": chunks}
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
