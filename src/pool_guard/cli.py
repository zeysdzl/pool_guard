from __future__ import annotations

import argparse
from typing import List

from pool_guard.config import load_config
from pool_guard.capture import ThreadedCapture
from pool_guard.detectors import StubDetector, UltralyticsYOLODetector
from pool_guard.classifiers import StubClassifier, TFLiteRuntimeClassifier, TorchScriptClassifier
from pool_guard.zone import Zone
from pool_guard.alarms import CompositeAlarm, RateLimiter, HTTPAlarmSink, GPIOAlarmSink
from pool_guard.logging import setup_logger, JSONLEventStore
from pool_guard.pipeline import PipelineRunner


def _build_detector(cfg, force_stub: bool):
    if force_stub or cfg.backend == "stub":
        return StubDetector(person_class_id=cfg.person_class_id, conf=max(cfg.conf_threshold, 0.9))
    if cfg.backend == "ultralytics_yolo":
        return UltralyticsYOLODetector(
            model_path=cfg.model_path or "",
            conf_threshold=cfg.conf_threshold,
            iou_threshold=cfg.iou_threshold,
            device=cfg.device,
        )
    raise ValueError(f"Unknown detector backend: {cfg.backend}")


def _build_classifier(cfg, force_stub: bool):
    if force_stub or cfg.backend == "stub":
        # default: no false alarms while model yok
        return StubClassifier(child_threshold=cfg.child_threshold, mode="adult")
    if cfg.backend == "tflite_runtime":
        return TFLiteRuntimeClassifier(
            model_path=cfg.model_path or "",
            input_size=cfg.input_size,
            child_threshold=cfg.child_threshold,
        )
    if cfg.backend == "torchscript":
        return TorchScriptClassifier(
            model_path=cfg.model_path or "",
            input_size=cfg.input_size,
            child_threshold=cfg.child_threshold,
        )
    raise ValueError(f"Unknown classifier backend: {cfg.backend}")


def _build_alarm(cfg, logger):
    sinks = []
    # HTTP (opsiyonel)
    if cfg.http.enabled:
        sinks.append(HTTPAlarmSink(url=cfg.http.url, timeout_s=cfg.http.timeout_s, headers=cfg.http.headers))
        logger.info("HTTP alarm enabled.")
    # GPIO (Windows'ta zaten init olamaz; yakalayıp kapatıyoruz)
    if cfg.gpio.enabled:
        try:
            sinks.append(
                GPIOAlarmSink(
                    buzzer_pin=cfg.gpio.buzzer_pin,
                    led_pin=cfg.gpio.led_pin,
                    active_high=cfg.gpio.active_high,
                    pulse_seconds=cfg.gpio.pulse_seconds,
                )
            )
            logger.info("GPIO alarm enabled.")
        except Exception as e:
            logger.warning(f"GPIO alarm requested but not available: {e}")

    limiter = RateLimiter(cooldown_s=cfg.cooldown_s)
    return CompositeAlarm(sinks=sinks, limiter=limiter)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pool_guard", description="Pool Guard (Edge AI Pool Safety)")
    ap.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    ap.add_argument("--dotenv", default=".env", help="Path to .env file (optional).")
    ap.add_argument("--stub", action="store_true", help="Force stub detector/classifier (CI/dev).")
    ap.add_argument("--no-api", action="store_true", help="Disable API even if config enables it.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, dotenv_path=args.dotenv)
    logger = setup_logger(cfg.logging.level)
    store = JSONLEventStore(cfg.logging.events_path)

    cap = ThreadedCapture(uri=cfg.source.uri, fps_limit=cfg.source.fps_limit, resize_width=cfg.source.resize_width)
    cap.start()

    detector = _build_detector(cfg.detector, force_stub=args.stub)
    classifier = _build_classifier(cfg.classifier, force_stub=args.stub)

    zones = [Zone(zone_id=z.zone_id, polygon_norm=z.polygon, enabled=z.enabled) for z in cfg.zones]
    alarm = _build_alarm(cfg.alarm, logger=logger)

    runner = PipelineRunner(
        source_name=cfg.source.uri,
        capture=cap,
        detector=detector,
        classifier=classifier,
        zones=zones,
        alarm=alarm,
        store=store,
        person_class_id=cfg.detector.person_class_id,
        inference_stride=cfg.pipeline.inference_stride,
        max_people_per_frame=cfg.pipeline.max_people_per_frame,
        draw_debug=cfg.pipeline.draw_debug,   # <-- EKLE
        headless=cfg.pipeline.headless, 
        logger=logger,
    )

    # API opsiyonel
    if cfg.api.enabled and not args.no_api:
        try:
            from pool_guard.api import create_app
            from waitress import serve
            import threading

            app = create_app(runner)
            t = threading.Thread(target=lambda: serve(app, host=cfg.api.host, port=cfg.api.port), daemon=True)
            t.start()
            logger.info(f"API started at http://{cfg.api.host}:{cfg.api.port}")
        except Exception as e:
            logger.warning(f"API failed to start: {e}")

    try:
        runner.run_forever()
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        cap.stop()

    return 0
