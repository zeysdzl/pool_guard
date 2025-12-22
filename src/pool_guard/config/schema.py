from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SourceConfig(BaseModel):
    kind: Literal["rtsp", "mp4"] = "rtsp"
    uri: str = Field(..., description="RTSP URL or MP4 file path")
    fps_limit: float = Field(10.0, ge=0.0, description="0 means unlimited")
    resize_width: int | None = Field(
        640, ge=64, description="Resize keeping aspect. None disables."
    )
    buffer_latest_only: bool = True


class DetectorConfig(BaseModel):
    backend: Literal["ultralytics_yolo", "stub"] = "stub"
    model_path: str | None = None
    conf_threshold: float = Field(0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    device: str = "cpu"
    person_class_id: int = 0  # for COCO


class ClassifierConfig(BaseModel):
    backend: Literal["tflite_runtime", "torchscript", "stub"] = "stub"
    model_path: str | None = None
    input_size: int = Field(160, ge=64)
    child_threshold: float = Field(0.5, ge=0.0, le=1.0)
    # label mapping assumption: output score = P(child)
    normalize: bool = True


class ZoneConfig(BaseModel):
    zone_id: str = "danger"
    # normalized polygon points: (x,y) in [0,1]
    polygon: list[tuple[float, float]] = Field(..., min_length=3)
    enabled: bool = True

    @field_validator("polygon")
    @classmethod
    def _validate_points(cls, pts: list[tuple[float, float]]):
        for x, y in pts:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("Zone polygon points must be normalized in [0,1].")
        return pts


class AlarmGPIOConfig(BaseModel):
    enabled: bool = False
    buzzer_pin: int = 18
    led_pin: int | None = 23
    active_high: bool = True
    pulse_seconds: float = Field(0.3, ge=0.05, le=5.0)


class AlarmHTTPConfig(BaseModel):
    enabled: bool = False
    url: str = ""
    timeout_s: float = Field(2.0, ge=0.2, le=10.0)
    headers: dict[str, str] = Field(default_factory=dict)


class AlarmConfig(BaseModel):
    cooldown_s: float = Field(3.0, ge=0.0)
    gpio: AlarmGPIOConfig = Field(default_factory=AlarmGPIOConfig)
    http: AlarmHTTPConfig = Field(default_factory=AlarmHTTPConfig)


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    events_path: str = "runs/events.jsonl"
    include_frames: bool = False  # keep false in MVP; storage explodes quickly


class PipelineConfig(BaseModel):
    inference_stride: int = Field(1, ge=1, description="Run inference every N frames")
    max_people_per_frame: int = Field(10, ge=1, le=50)
    draw_debug: bool = False
    headless: bool = True  # no cv2.imshow on Pi headless


class APIConfig(BaseModel):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    show_frame: bool = True


class AppConfig(BaseModel):
    source: SourceConfig
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    zones: list[ZoneConfig]
    alarm: AlarmConfig = Field(default_factory=AlarmConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    @field_validator("zones")
    @classmethod
    def _zones_nonempty(cls, z):
        if not z:
            raise ValueError("At least one zone must be configured.")
        return z
