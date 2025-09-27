"""Video Shots - модуль для створення скріншотів з відео."""

from .cli import main
from .exceptions import VideoShotsError, VideoFileError, ConfigError, ProcessingError

__version__ = "1.0.0"
__all__ = ["main", "VideoShotsError", "VideoFileError", "ConfigError", "ProcessingError"]
