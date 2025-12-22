from __future__ import annotations

from typing import Tuple

import numpy as np

from pool_guard.classifiers.base import Classification


class TFLiteRuntimeClassifier:
    """
    Uses tflite_runtime.interpreter (recommended on Raspberry Pi).
    Assumption: model outputs a single float = P(child).
    """

    def __init__(self, model_path: str, input_size: int = 160, child_threshold: float = 0.5) -> None:
        if not model_path:
            raise ValueError("model_path is required for TFLiteRuntimeClassifier")
        self.model_path = model_path
        self.input_size = input_size
        self.child_threshold = child_threshold

        from tflite_runtime.interpreter import Interpreter  # type: ignore

        self._interp = Interpreter(model_path=model_path, num_threads=2)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        import cv2

        img = cv2.resize(crop_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)  # NHWC
        return x

    def classify_person_crop(self, crop_bgr: np.ndarray) -> Classification:
        x = self._preprocess(crop_bgr)
        self._interp.set_tensor(self._in["index"], x)
        self._interp.invoke()
        y = self._interp.get_tensor(self._out["index"])
        score = float(np.squeeze(y))
        label = "child" if score >= self.child_threshold else "adult"
        conf = abs(score - self.child_threshold) * 2.0
        return Classification(label=label, child_score=score, conf=conf)
