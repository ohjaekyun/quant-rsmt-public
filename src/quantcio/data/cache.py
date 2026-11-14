"""Small disk cache for fetched data and LLM responses (pickle for DataFrames, JSON otherwise)."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class DiskCache:
    def __init__(self, root: Path, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def _path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / f"{key}.pkl"

    def get(self, namespace: str, key: str, ttl_hours: float | None) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(namespace, key)
        if not path.exists():
            return None
        if ttl_hours is not None and (time.time() - path.stat().st_mtime) > ttl_hours * 3600:
            return None
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def set(self, namespace: str, key: str, value: Any) -> None:
        if not self.enabled:
            return
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # unique temp file per writer: concurrent writers of the same key must not share it
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with tmp.open("wb") as f:
                pickle.dump(value, f)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def get_or_fetch(
        self,
        namespace: str,
        key: str,
        ttl_hours: float | None,
        fetch: Callable[[], T],
    ) -> T:
        hit = self.get(namespace, key, ttl_hours)
        if hit is not None:
            return hit
        value = fetch()
        if value is not None:
            self.set(namespace, key, value)
        return value
