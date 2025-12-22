from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


PointF = Tuple[float, float]
PointI = Tuple[int, int]


def point_in_polygon(point: PointI, polygon: List[PointI]) -> bool:
    """
    Ray casting algorithm.
    polygon: list of (x,y) in pixels.
    """
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        # check edge intersects ray
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
            inside = not inside
        x1, y1 = x2, y2
    return inside


@dataclass(frozen=True)
class Zone:
    zone_id: str
    polygon_norm: List[PointF]
    enabled: bool = True

    def polygon_pixels(self, frame_w: int, frame_h: int) -> List[PointI]:
        pts: List[PointI] = []
        for x, y in self.polygon_norm:
            px = int(round(x * frame_w))
            py = int(round(y * frame_h))
            pts.append((px, py))
        return pts

    def contains(self, point_px: PointI, frame_w: int, frame_h: int) -> bool:
        if not self.enabled:
            return False
        poly = self.polygon_pixels(frame_w, frame_h)
        return point_in_polygon(point_px, poly)
