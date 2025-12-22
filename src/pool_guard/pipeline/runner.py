from __future__ import annotations

from typing import Any

import numpy as np

from pool_guard.alarms import AlarmEvent, CompositeAlarm
from pool_guard.capture import FramePacket, ThreadedCapture
from pool_guard.classifiers import Classifier
from pool_guard.detectors import Detector
from pool_guard.logging import JSONLEventStore
from pool_guard.utils.images import crop
from pool_guard.utils.time import now_s, sleep_s
from pool_guard.zone import Zone


class PipelineRunner:
    def __init__(
        self,
        *,
        source_name: str,
        capture: ThreadedCapture,
        detector: Detector,
        classifier: Classifier,
        zones: list[Zone],
        alarm: CompositeAlarm,
        store: JSONLEventStore,
        person_class_id: int,
        inference_stride: int = 1,
        max_people_per_frame: int = 10,
        draw_debug: bool = False,
        headless: bool = True,
        window_name: str = "Pool Guard",
        logger=None,
    ) -> None:
        self.source_name = source_name
        self.capture = capture
        self.detector = detector
        self.classifier = classifier
        self.zones = zones
        self.alarm = alarm
        self.store = store
        self.person_class_id = person_class_id
        self.inference_stride = max(1, int(inference_stride))
        self.max_people_per_frame = int(max_people_per_frame)
        self.draw_debug = bool(draw_debug)
        self.headless = bool(headless)
        self.window_name = window_name
        self.logger = logger

        self._last_frame: np.ndarray | None = None
        self._last_status: dict[str, Any] = {}
        self._imshow_failed_once = False

        # Debug window init
        self._win_inited: bool = False

    def get_last_frame(self) -> np.ndarray | None:
        return self._last_frame

    def get_status(self) -> dict[str, Any]:
        return self._last_status

    # ---------- Debug drawing helpers ----------

    def _zone_polygon_norm(self, z: Zone) -> list[tuple[float, float]] | None:
        # Try common attribute names without assuming a single implementation
        for attr in ("polygon_norm", "polygon", "points", "points_norm"):
            if hasattr(z, attr):
                pts = getattr(z, attr)
                if (
                    isinstance(pts, list)
                    and pts
                    and isinstance(pts[0], (list, tuple))
                    and len(pts[0]) == 2
                ):
                    return [(float(a), float(b)) for a, b in pts]
        return None

    def _draw_zone(self, frame: np.ndarray, z: Zone, w: int, h: int, active: bool) -> None:
        import cv2

        # If Zone has its own draw(), prefer that.
        if hasattr(z, "draw") and callable(z.draw):
            try:
                # Try common signatures
                try:
                    z.draw(frame, active=active, w=w, h=h)  # type: ignore[arg-type]
                    return
                except TypeError:
                    z.draw(frame, active=active)  # type: ignore[arg-type]
                    return
            except Exception:
                # Fall back to manual drawing
                pass

        pts_norm = self._zone_polygon_norm(z)
        if not pts_norm:
            return

        pts = np.array([[int(x * w), int(y * h)] for x, y in pts_norm], dtype=np.int32).reshape(
            (-1, 1, 2)
        )
        cv2.polylines(
            frame,
            [pts],
            isClosed=True,
            color=(0, 0, 255) if active else (255, 255, 255),
            thickness=2,
        )

        # Label
        try:
            zone_id = getattr(z, "zone_id", "zone")
        except Exception:
            zone_id = "zone"
        x0, y0 = pts[0, 0, 0], pts[0, 0, 1]
        cv2.putText(
            frame,
            str(zone_id),
            (int(x0), max(20, int(y0) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255) if active else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_bbox(
        self, frame: np.ndarray, det, cls_label: str, cls_conf: float, should_alarm: bool
    ) -> None:
        import cv2

        x1, y1, x2, y2 = int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2)
        color = (0, 0, 255) if should_alarm else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{cls_label} {cls_conf:.2f} | det {det.conf:.2f}"
        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    def _maybe_show(self, frame: np.ndarray) -> bool:
        """Returns False if user requested exit (q), True otherwise."""
        if self.headless or not self.draw_debug:
            return True

        try:
            import cv2

            # Init window once (position/size)
            if not self._win_inited:
                self._win_inited = True
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, 960, 540)
                cv2.moveWindow(self.window_name, 50, 50)

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
            return True

        except Exception as e:
            # Likely opencv-python-headless or no GUI backend
            if not self._imshow_failed_once:
                self._imshow_failed_once = True
                if self.logger:
                    self.logger.warning(f"Debug window failed (disabling draw_debug). Reason: {e}")
            self.draw_debug = False
            return True

    def _cleanup_windows(self) -> None:
        if self.headless:
            return
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass

    # ---------- Main loop ----------

    def run_forever(self) -> None:
        if self.logger:
            self.logger.info("Pipeline started.")

        frame_seen = 0
        try:
            while True:
                pkt: FramePacket | None = self.capture.read_latest()
                if pkt is None:
                    sleep_s(0.01)
                    continue

                frame_seen += 1
                if frame_seen % self.inference_stride != 0:
                    continue

                frame = pkt.frame
                h, w = frame.shape[:2]

                # We'll draw on a copy if debug is enabled; otherwise keep original reference.
                debug_frame = frame.copy() if (self.draw_debug and not self.headless) else frame
                self._last_frame = debug_frame

                dets = self.detector.detect(frame)
                people = [d for d in dets if d.class_id == self.person_class_id]
                people = sorted(people, key=lambda d: d.conf, reverse=True)[
                    : self.max_people_per_frame
                ]

                alarms_fired = 0
                cooldown_hits = 0

                # Pre-draw zones (inactive by default; we can highlight active later per detection)
                if self.draw_debug and not self.headless:
                    for z in self.zones:
                        if getattr(z, "enabled", True):
                            self._draw_zone(debug_frame, z, w, h, active=False)

                for det in people:
                    person_crop = crop(frame, det.bbox)
                    cls = self.classifier.classify_person_crop(person_crop)

                    cx, cy = det.bbox.center()
                    in_zone_id: str | None = None

                    for z in self.zones:
                        if not getattr(z, "enabled", True):
                            continue
                        if z.contains((cx, cy), w, h):
                            in_zone_id = getattr(z, "zone_id", None) or "zone"
                            break

                    should_alarm = (cls.label == "child") and (in_zone_id is not None)

                    event: dict[str, Any] = {
                        "ts_s": now_s(),
                        "source": self.source_name,
                        "frame_ts_s": pkt.ts_s,
                        "frame_idx": pkt.frame_idx,
                        "zone_id": in_zone_id,
                        "person": {
                            "bbox_xyxy": [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2],
                            "det_conf": det.conf,
                            "child_score": cls.child_score,
                            "cls_label": cls.label,
                            "cls_conf": cls.conf,
                            "centroid": [cx, cy],
                        },
                        "should_alarm": should_alarm,
                    }

                    if should_alarm and in_zone_id:
                        ae = AlarmEvent(
                            ts_s=event["ts_s"],
                            source=self.source_name,
                            zone_id=in_zone_id,
                            payload=event,
                        )
                        ok = self.alarm.trigger(ae)
                        if ok:
                            alarms_fired += 1
                            event["alarm_fired"] = True
                        else:
                            cooldown_hits += 1
                            event["alarm_fired"] = False
                            event["cooldown_hit"] = True
                    else:
                        event["alarm_fired"] = False

                    self.store.append(event)

                    # Debug overlay per detection
                    if self.draw_debug and not self.headless:
                        self._draw_bbox(debug_frame, det, cls.label, cls.conf, should_alarm)

                        # Highlight the active zone (first match)
                        if in_zone_id is not None:
                            for z in self.zones:
                                if getattr(z, "zone_id", None) == in_zone_id:
                                    self._draw_zone(debug_frame, z, w, h, active=True)
                                    break

                self._last_status = {
                    "source": self.source_name,
                    "frame_idx": pkt.frame_idx,
                    "frame_ts_s": pkt.ts_s,
                    "people": len(people),
                    "alarms_fired": alarms_fired,
                    "cooldown_hits": cooldown_hits,
                    "zones": [
                        getattr(z, "zone_id", "zone")
                        for z in self.zones
                        if getattr(z, "enabled", True)
                    ],
                }

                # Show window if configured
                if not self._maybe_show(debug_frame):
                    break

        finally:
            self._cleanup_windows()
