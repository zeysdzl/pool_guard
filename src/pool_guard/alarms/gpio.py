from __future__ import annotations

import sys

from pool_guard.alarms.base import AlarmEvent
from pool_guard.utils.time import sleep_s


class GPIOAlarmSink:
    """
    Raspberry Pi GPIO sink.
    Import is platform-guarded so PC/CI won't break.
    """

    def __init__(
        self, buzzer_pin: int, led_pin: int | None, active_high: bool, pulse_seconds: float
    ) -> None:
        self.buzzer_pin = buzzer_pin
        self.led_pin = led_pin
        self.active_high = active_high
        self.pulse_seconds = pulse_seconds

        if not sys.platform.startswith("linux"):
            raise RuntimeError("GPIOAlarmSink is only supported on Linux/Raspberry Pi.")

        try:
            import RPi.GPIO as GPIO  # type: ignore
        except Exception as e:
            raise RuntimeError("RPi.GPIO not available. Install it on Raspberry Pi.") from e

        self.GPIO = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.buzzer_pin, GPIO.OUT)
        if self.led_pin is not None:
            GPIO.setup(self.led_pin, GPIO.OUT)

        self._on = GPIO.HIGH if self.active_high else GPIO.LOW
        self._off = GPIO.LOW if self.active_high else GPIO.HIGH

        GPIO.output(self.buzzer_pin, self._off)
        if self.led_pin is not None:
            GPIO.output(self.led_pin, self._off)

    def trigger(self, event: AlarmEvent) -> None:
        try:
            self.GPIO.output(self.buzzer_pin, self._on)
            if self.led_pin is not None:
                self.GPIO.output(self.led_pin, self._on)
            sleep_s(self.pulse_seconds)
        finally:
            self.GPIO.output(self.buzzer_pin, self._off)
            if self.led_pin is not None:
                self.GPIO.output(self.led_pin, self._off)

    def close(self) -> None:
        try:
            self.GPIO.cleanup()
        except Exception:
            pass
