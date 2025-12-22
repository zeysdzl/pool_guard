from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from pool_guard.utils.threading import StoppableThread
from pool_guard.utils.time import now_s, sleep_s


@dataclass
class FramePacket:
    frame: np.ndarray
    ts_s: float
    frame_idx: int


class ThreadedCapture:
    """
    Threaded video capture that keeps the latest frame (low-latency).
    Uses OpenCV VideoCapture under the hood.

    Supports:
      - RTSP/HTTP/file paths: uri: "rtsp://..." or "./video.mp4"
      - Webcam index as string: uri: "0" (or "1", "2"...)
    """

    def __init__(self, uri: str, fps_limit: float = 10.0, resize_width: Optional[int] = 640) -> None:
        self.uri = uri
        self.fps_limit = float(fps_limit)
        self.resize_width = resize_width

        self._cap = None
        self._thread: Optional[StoppableThread] = None
        self._lock = threading.Lock()
        self._latest: Optional[FramePacket] = None
        self._running = False

    def start(self) -> None:
        import cv2  # local import keeps CI lighter

        # --- Webcam support: if uri is a numeric string, treat as camera index.
        src: Union[str, int] = self.uri
        if isinstance(src, str):
            s = src.strip()
            if s.isdigit():
                src = int(s)

        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.uri}")

        self._running = True
        self._thread = StoppableThread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.stop()
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass

    def read_latest(self) -> Optional[FramePacket]:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        import cv2

        assert self._cap is not None
        frame_idx = 0
        last_emit = 0.0
        min_dt = (1.0 / self.fps_limit) if self.fps_limit and self.fps_limit > 0 else 0.0

        while self._running and self._thread and not self._thread.stopped():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                sleep_s(0.05)
                continue

            if self.resize_width is not None:
                h, w = frame.shape[:2]
                if w > 0 and w != self.resize_width:
                    scale = self.resize_width / float(w)
                    nh = max(1, int(h * scale))
                    frame = cv2.resize(frame, (self.resize_width, nh), interpolation=cv2.INTER_LINEAR)

            ts = now_s()
            if min_dt > 0 and (ts - last_emit) < min_dt:
                continue

            pkt = FramePacket(frame=frame, ts_s=ts, frame_idx=frame_idx)
            with self._lock:
                self._latest = pkt

            last_emit = ts
            frame_idx += 1
