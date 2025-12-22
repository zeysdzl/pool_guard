from __future__ import annotations

import time


def now_s() -> float:
    return time.time()


def sleep_s(seconds: float) -> None:
    time.sleep(max(0.0, seconds))
