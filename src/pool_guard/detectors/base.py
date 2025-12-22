from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pool_guard.utils.images import BBox


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    conf: float
    class_id: int


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...
