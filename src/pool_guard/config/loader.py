from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
from dotenv import load_dotenv

from pool_guard.constants import ENV_PREFIX
from pool_guard.exceptions import ConfigError
from pool_guard.config.schema import AppConfig


def _deep_set(d: Dict[str, Any], keys: Tuple[str, ...], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _coerce_scalar(s: str) -> Any:
    sl = s.strip().lower()
    if sl in {"true", "false"}:
        return sl == "true"
    # int
    try:
        if sl.startswith("0") and len(sl) > 1 and sl[1].isdigit():
            # keep as string (could be pin/leading zeros)
            return s
        return int(sl)
    except Exception:
        pass
    # float
    try:
        return float(sl)
    except Exception:
        pass
    # yaml list/dict support via inline yaml
    if (sl.startswith("[") and sl.endswith("]")) or (sl.startswith("{") and sl.endswith("}")):
        try:
            return yaml.safe_load(s)
        except Exception:
            return s
    return s


def load_config(path: str, dotenv_path: str | None = ".env") -> AppConfig:
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)

    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {path}")

    with p.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    # Apply environment overrides: POOL_GUARD__a__b=... sets data["a"]["b"]
    for k, v in os.environ.items():
        if not k.startswith(ENV_PREFIX):
            continue
        suffix = k[len(ENV_PREFIX) :]
        if not suffix:
            continue
        keys = tuple(part.lower() for part in suffix.split("__") if part)
        if not keys:
            continue
        _deep_set(data, keys, _coerce_scalar(v))

    try:
        return AppConfig.model_validate(data)
    except Exception as e:
        raise ConfigError(f"Config validation failed: {e}") from e
