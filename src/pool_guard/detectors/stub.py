from __future__ import annotations

from typing import List

import numpy as np

from pool_guard.detectors.base import Detection
from pool_guard.utils.images import BBox


class StubDetector:
    """
    Deterministic-ish stub: returns one centered person bbox every N frames if
    the image is non-empty. Good enough for CI smoke tests.
    """

    def __init__(self, person_class_id: int = 0, conf: float = 0.9) -> None:
        self.person_class_id = person_class_id
        self.conf = conf
        self._tick = 0

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        self._tick += 1
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        if self._tick % 3 != 0:
            return []
        h, w = frame_bgr.shape[:2]
        bw, bh = int(w * 0.25), int(h * 0.5)
        x1 = (w - bw) // 2
        y1 = (h - bh) // 2
        bbox = BBox(x1, y1, x1 + bw, y1 + bh)
        return [Detection(bbox=bbox, conf=self.conf, class_id=self.person_class_id)]
