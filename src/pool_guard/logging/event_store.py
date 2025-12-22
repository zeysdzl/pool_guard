from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from pool_guard.logging.setup import ensure_parent_dir


class JSONLEventStore:
    def __init__(self, path: str) -> None:
        self.path = path
        ensure_parent_dir(path)
        Path(path).touch(exist_ok=True)

    def append(self, event: Dict[str, Any]) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            return
