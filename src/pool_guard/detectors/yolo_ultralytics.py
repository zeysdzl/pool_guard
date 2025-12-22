from __future__ import annotations

from typing import List, Optional

import numpy as np

from pool_guard.detectors.base import Detection
from pool_guard.utils.images import BBox


class UltralyticsYOLODetector:
    """
    Wrapper around ultralytics YOLO.
    Lazy-imports ultralytics so CI can run without it.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ) -> None:
        if not model_path:
            raise ValueError("model_path is required for UltralyticsYOLODetector")
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        from ultralytics import YOLO  # type: ignore

        self._model = YOLO(model_path)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        # ultralytics expects BGR ok; returns boxes in xyxy
        res = self._model.predict(
            source=frame_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        dets: List[Detection] = []
        if not res:
            return dets

        r0 = res[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return dets

        xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
        conf = boxes.conf.cpu().numpy()  # (N,)
        cls = boxes.cls.cpu().numpy().astype(int)  # (N,)

        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            dets.append(
                Detection(
                    bbox=BBox(int(x1), int(y1), int(x2), int(y2)),
                    conf=float(c),
                    class_id=int(k),
                )
            )
        return dets
