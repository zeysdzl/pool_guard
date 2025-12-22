from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from flask import Flask, jsonify, request, Response

from pool_guard.pipeline import PipelineRunner


def create_app(runner: PipelineRunner) -> Flask:
    app = Flask(__name__)

    @app.get("/status")
    def status():
        return jsonify(runner.get_status())

    @app.post("/trigger")
    def trigger():
        # manual trigger endpoint (for integration tests)
        return jsonify({"ok": True})

    @app.get("/frame")
    def frame():
        img = runner.get_last_frame()
        if img is None:
            return jsonify({"error": "no frame"}), 404
        try:
            import cv2

            ok, buf = cv2.imencode(".jpg", img)
            if not ok:
                return jsonify({"error": "encode failed"}), 500
            return Response(buf.tobytes(), mimetype="image/jpeg")
        except Exception:
            return jsonify({"error": "opencv not available"}), 500

    return app
