"""Time and timecode utilities."""

import math
from typing import List

from video_shots.config.config import CYRILLIC_S_LOWER, CYRILLIC_S_UPPER, DEFAULT_FPS_FOR_TIMECODE
from video_shots.core.exceptions import VideoShotsError
from video_shots.core.models import TimePoint


def parse_time_value(value: str, *, allow_zero: bool = False) -> float:
    """Parse time in seconds from user string."""

    if value is None:
        raise VideoShotsError("Time value not specified")

    normalized = (
        value.strip()
        .replace(" ", "")
        .replace(",", ".")
        .replace(CYRILLIC_S_LOWER, "s")
        .replace(CYRILLIC_S_UPPER, "s")
        .lower()
    )

    if not normalized:
        raise VideoShotsError("Empty time value")

    if ":" in normalized:
        parts = normalized.split(":")
        if len(parts) != 3:
            raise VideoShotsError(
                f"Invalid time format '{value}'. Expected HH:MM:SS.mmm"
            )
        hours_str, minutes_str, seconds_str = parts
        try:
            hours = int(hours_str)
            minutes = int(minutes_str)
            seconds = float(seconds_str)
        except ValueError as exc:
            raise VideoShotsError(f"Failed to parse time '{value}'") from exc
        if minutes >= 60 or minutes < 0:
            raise VideoShotsError(f"Minutes out of range 0-59 in value '{value}'")
        if seconds < 0 or seconds >= 60:
            raise VideoShotsError(f"Seconds out of range 0-59 in value '{value}'")
        total_seconds = hours * 3600 + minutes * 60 + seconds
    else:
        suffix = None
        if normalized.endswith("ms"):
            suffix = "ms"
            number_part = normalized[:-2]
        elif normalized.endswith("s"):
            suffix = "s"
            number_part = normalized[:-1]
        else:
            number_part = normalized

        if not number_part:
            raise VideoShotsError(f"Invalid time format '{value}'")

        try:
            number = float(number_part)
        except ValueError as exc:
            raise VideoShotsError(f"Failed to parse number in '{value}'") from exc

        if suffix == "ms":
            total_seconds = number / 1000.0
        else:
            total_seconds = number

    if total_seconds < 0 or (total_seconds == 0 and not allow_zero):
        raise VideoShotsError("Time must be > 0")

    return total_seconds


def format_timestamp(seconds: float, *, precision: str) -> str:
    """Format time in readable format."""
    include_ms = precision == "ms"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if include_ms:
        sec_str = f"{secs:06.3f}" if hours else f"{secs:06.3f}"
    else:
        secs = int(round(secs))
        sec_str = f"{secs:02d}"

    if hours:
        if include_ms:
            return f"{hours:02d}:{minutes:02d}:{sec_str}"
        return f"{hours:02d}:{minutes:02d}:{sec_str}"
    return f"{minutes:02d}:{sec_str}"


def format_timestamp_for_name(seconds: float, *, precision: str) -> str:
    """Format time for use in file names."""
    include_ms = precision == "ms"
    minutes_total = int(seconds // 60)
    secs_float = seconds % 60
    hours, minutes = divmod(minutes_total, 60)
    secs = int(secs_float)
    millis = int(round((secs_float - secs) * 1000))

    parts = []
    if hours:
        parts.append(f"{hours:02d}")
    parts.append(f"{minutes:02d}")
    parts.append(f"{secs:02d}")
    if include_ms:
        parts.append(f"{millis:03d}")
    return "-".join(parts)


def collect_timepoints(
    start: float,
    end: float,
    step: float,
    fps: float,
    frame_count: int,
) -> List[TimePoint]:
    """Collect time points for processing."""
    if step <= 0:
        raise VideoShotsError("Interval must be > 0")

    if end < start:
        raise VideoShotsError("--end parameter must be >= --start")

    duration = frame_count / fps if fps > 0 else 0
    hard_end = min(end, duration)
    if hard_end < start:
        raise VideoShotsError("Start time exceeds video duration")

    timepoints: List[TimePoint] = []
    index = 0
    epsilon = step / 10_000 if step else 1e-9

    while True:
        current_time = start + index * step
        if current_time > hard_end + epsilon:
            break

        frame_index = int(round(current_time * fps))
        frame_index = max(0, min(frame_index, frame_count - 1))

        timepoints.append(TimePoint(index=len(timepoints) + 1, seconds=current_time, frame_index=frame_index))

        index += 1
        if index > 10_000_000:  # MAX_ITERATIONS
            raise VideoShotsError(
                "Too many iterations. Check interval or use --end/--start"
            )

    return timepoints


def parse_timecode_to_seconds(timecode: str) -> float:
    """Convert timecode HH:MM:SS:FF to seconds."""
    try:
        parts = timecode.split(':')
        if len(parts) != 4:
            raise VideoShotsError(f"Invalid timecode format '{timecode}'. Expected HH:MM:SS:FF")
        
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        frames = int(parts[3])
        
        if minutes >= 60 or seconds >= 60 or frames < 0:
            raise VideoShotsError(f"Invalid timecode values in '{timecode}'")
        
        total_seconds = hours * 3600 + minutes * 60 + seconds + frames / DEFAULT_FPS_FOR_TIMECODE
        return total_seconds
    except ValueError as exc:
        raise VideoShotsError(f"Failed to parse timecode '{timecode}': {exc}") from exc


def estimate_total_frames(duration: float, interval: float) -> int:
    """Estimate total number of frames."""
    if interval <= 0:
        return 0
    return int(math.floor(duration / interval + 1e-9)) + 1
