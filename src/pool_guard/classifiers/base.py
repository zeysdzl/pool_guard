from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Classification:
    label: str  # "child" or "adult"
    child_score: float  # P(child)
    conf: float  # confidence proxy (can equal child_score distance)


class Classifier(Protocol):
    def classify_person_crop(self, crop_bgr: np.ndarray) -> Classification:
        ...
