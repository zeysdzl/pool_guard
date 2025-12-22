class PoolGuardError(Exception):
    """Base error for Pool Guard."""


class ConfigError(PoolGuardError):
    """Configuration related errors."""


class CaptureError(PoolGuardError):
    """Video capture related errors."""
