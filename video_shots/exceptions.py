"""Exceptions for video-shots module."""


class VideoShotsError(Exception):
    """Base exception for video-shots."""
    pass


class VideoFileError(VideoShotsError):
    """Video file operation error."""
    pass


class ConfigError(VideoShotsError):
    """Configuration error."""
    pass


class ProcessingError(VideoShotsError):
    """Video processing error."""
    pass
