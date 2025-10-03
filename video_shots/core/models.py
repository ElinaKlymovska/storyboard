"""Data models for video-shots module."""

from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path


@dataclass
class TimePoint:
    """Time point for screenshot creation."""
    seconds: float
    frame_index: int
    index: Optional[int] = None
    description: Optional[str] = None


@dataclass
class VideoInfo:
    """Video file information."""
    path: Path
    duration: float
    fps: float
    total_frames: int
    width: int
    height: int


@dataclass
class ScreenshotConfig:
    """Screenshot creation configuration."""
    output_dir: Path
    prefix: str = "shot"
    format: str = "png"
    quality: int = 95
    resize: Optional[tuple] = None


@dataclass
class ProcessingResult:
    """Video processing result."""
    success: bool
    screenshots_created: int
    errors: List[str]
    output_dir: Path
