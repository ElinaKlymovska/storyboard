"""Video processing and screenshot creation."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Sequence, Tuple

from ..config import (
    DEFAULT_FONT_SCALE, 
    DEFAULT_FONT_THICKNESS, 
    DEFAULT_PADDING,
    PANEL_BG_COLOR,
    PANEL_ALPHA,
    TEXT_COLOR,
    DEFAULT_JPEG_QUALITY
)
from ..exceptions import VideoShotsError
from ..models import TimePoint
from ..utils import format_timestamp


def apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Apply rotation to frame."""
    if rotation == 0:
        return frame
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise VideoShotsError(f"Unsupported rotation angle: {rotation}")


def render_label_panel(
    frame: np.ndarray,
    label_lines: Sequence[str],
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = DEFAULT_FONT_SCALE,
    font_thickness: int = DEFAULT_FONT_THICKNESS,
    padding: int = DEFAULT_PADDING,
    panel_bg_color: Tuple[int, int, int] = PANEL_BG_COLOR,
    panel_alpha: float = PANEL_ALPHA,
    text_color: Tuple[int, int, int] = TEXT_COLOR,
) -> np.ndarray:
    """Add label panel to frame."""
    height, width = frame.shape[:2]

    # Calculate panel width based on text length
    max_text_width = 0
    text_height_total = 0
    line_height = 0
    for line in label_lines:
        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, font_thickness)
        max_text_width = max(max_text_width, text_width)
        line_height = max(line_height, text_height + baseline)
        text_height_total += text_height + baseline

    panel_width = max_text_width + padding * 2
    if panel_width < int(width * 0.18):
        panel_width = int(width * 0.18)

    new_width = width + panel_width
    result = np.zeros((height, new_width, 3), dtype=np.uint8)
    result[:, :width] = frame

    overlay = result.copy()
    overlay[:, width:] = panel_bg_color
    cv2.addWeighted(overlay, panel_alpha, result, 1 - panel_alpha, 0, result)

    # Place text in top part of panel
    y = padding + line_height
    x = width + padding
    for line in label_lines:
        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, font_thickness)
        cv2.putText(result, line, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        y += line_height + padding // 2

    return result


def save_frame_image(
    frame: np.ndarray,
    output_path: Path,
    image_format: str,
) -> None:
    """Save frame as image."""
    params = []
    if image_format == "jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), DEFAULT_JPEG_QUALITY]

    success = cv2.imwrite(str(output_path), frame, params)
    if not success:
        raise VideoShotsError(f"Failed to save file '{output_path}'")


def process_video_frames(
    video_path: Path,
    timepoints: List[TimePoint],
    output_dir: Path,
    prefix: str,
    image_format: str,
    rotation: int,
    time_precision: str,
) -> List[Path]:
    """Process video and create screenshots."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoShotsError(f"Failed to open video '{video_path}'")

    image_paths: List[Path] = []
    previous_frame_idx = None
    duplicates = 0

    try:
        for tp in timepoints:
            if previous_frame_idx is not None and tp.frame_index == previous_frame_idx:
                duplicates += 1
            previous_frame_idx = tp.frame_index

            cap.set(cv2.CAP_PROP_POS_FRAMES, tp.frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"⚠️ Skipped frame #{tp.frame_index}: failed to read")
                continue

            frame = apply_rotation(frame, rotation)

            time_label = format_timestamp(tp.seconds, precision=time_precision)
            label_lines = [f"Time {time_label}"]
            framed = render_label_panel(frame, label_lines)

            from ..utils import format_timestamp_for_name
            time_tag = format_timestamp_for_name(tp.seconds, precision=time_precision)
            filename = f"{prefix}_{tp.index:05d}_f{tp.frame_index:05d}_t{time_tag}.{image_format}"
            output_path = output_dir / "screenshots" / filename
            save_frame_image(framed, output_path, image_format)
            image_paths.append(output_path)

    finally:
        cap.release()

    return image_paths, duplicates
