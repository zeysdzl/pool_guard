from __future__ import annotations

import numpy as np

from pool_guard.zone.polygon import PointF


def interactive_polygon_editor(frame_bgr: np.ndarray) -> list[PointF]:
    """
    Minimal OpenCV editor:
    - Left click to add points
    - Press 's' to save/finish
    - Press 'c' to clear
    - Press 'q' to quit without saving (returns empty)
    Returns normalized points in [0,1].
    """
    import cv2

    h, w = frame_bgr.shape[:2]
    points_px: list[tuple[int, int]] = []
    win = "Pool Guard - Zone Editor"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points_px.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        img = frame_bgr.copy()
        for p in points_px:
            cv2.circle(img, p, 4, (0, 255, 0), -1)
        if len(points_px) >= 2:
            cv2.polylines(img, [np.array(points_px, dtype=np.int32)], False, (0, 255, 0), 2)

        cv2.putText(
            img,
            "LClick:add  s:save  c:clear  q:quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imshow(win, img)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("c"):
            points_px.clear()
        elif key == ord("q"):
            cv2.destroyAllWindows()
            return []
        elif key == ord("s"):
            cv2.destroyAllWindows()
            if len(points_px) < 3:
                return []
            return [(x / float(w), y / float(h)) for x, y in points_px]
