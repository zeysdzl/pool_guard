from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BBox:
    # xyxy in pixels
    x1: int
    y1: int
    x2: int
    y2: int

    def clamp(self, w: int, h: int) -> BBox:
        x1 = max(0, min(self.x1, w - 1))
        y1 = max(0, min(self.y1, h - 1))
        x2 = max(0, min(self.x2, w - 1))
        y2 = max(0, min(self.y2, h - 1))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return BBox(x1, y1, x2, y2)

    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2


def crop(image: np.ndarray, bbox: BBox) -> np.ndarray:
    h, w = image.shape[:2]
    b = bbox.clamp(w, h)
    if b.width() <= 1 or b.height() <= 1:
        return image[0:1, 0:1].copy()
    return image[b.y1 : b.y2, b.x1 : b.x2].copy()
