from __future__ import annotations

import numpy as np

from pool_guard.classifiers.base import Classification


class TorchScriptClassifier:
    """
    TorchScript model inference (PC dev). Lazy-import torch.
    Assumption: model outputs logits or probability for child as single scalar.
    """

    def __init__(self, model_path: str, input_size: int = 160, child_threshold: float = 0.5) -> None:
        if not model_path:
            raise ValueError("model_path is required for TorchScriptClassifier")
        self.model_path = model_path
        self.input_size = input_size
        self.child_threshold = child_threshold

        import torch  # type: ignore

        self._torch = torch
        self._model = torch.jit.load(model_path, map_location="cpu")
        self._model.eval()

    def _preprocess(self, crop_bgr: np.ndarray) -> "np.ndarray":
        import cv2

        img = cv2.resize(crop_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # CHW
        x = np.expand_dims(x, axis=0)  # NCHW
        return x

    def classify_person_crop(self, crop_bgr: np.ndarray) -> Classification:
        x = self._preprocess(crop_bgr)
        torch = self._torch
        with torch.no_grad():
            t = torch.from_numpy(x)
            y = self._model(t)
            score = float(torch.sigmoid(y).reshape(-1)[0].item()) if y.numel() == 1 else float(y.reshape(-1)[0].item())
        label = "child" if score >= self.child_threshold else "adult"
        conf = abs(score - self.child_threshold) * 2.0
        return Classification(label=label, child_score=score, conf=conf)
