from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from pool_guard.classifiers.base import Classification


StubMode = Literal["adult", "child", "alternate", "random"]


class StubClassifier:
    """
    Deterministic stub classifier for pipeline testing.

    Modes:
      - "adult": always returns adult (default; safest, no false alarms)
      - "child": always returns child (useful to test alarm + cooldown)
      - "alternate": alternates child/adult every call
      - "random": random child_score in [0,1] (stress testing)
    """

    def __init__(
        self,
        child_threshold: float = 0.5,
        mode: StubMode = "adult",
        seed: Optional[int] = 0,
    ) -> None:
        self.child_threshold = float(child_threshold)
        self.mode: StubMode = mode
        self._tick = 0
        self._rng = None
        if mode == "random":
            # Local RNG; deterministic with seed
            self._rng = np.random.default_rng(seed)

    def classify_person_crop(self, crop_bgr: np.ndarray) -> Classification:
        self._tick += 1

        if self.mode == "adult":
            score = 0.0
        elif self.mode == "child":
            score = 1.0
        elif self.mode == "alternate":
            score = 0.8 if (self._tick % 2 == 0) else 0.2
        elif self.mode == "random":
            assert self._rng is not None
            score = float(self._rng.random())
        else:
            # Defensive fallback
            score = 0.0

        label = "child" if score >= self.child_threshold else "adult"
        conf = float(min(1.0, abs(score - self.child_threshold) * 2.0))
        return Classification(label=label, child_score=float(score), conf=conf)
