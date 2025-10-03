"""Video processing and screenshot creation."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Sequence, Tuple
import os

from video_shots.config.config import (
    DEFAULT_FONT_SCALE, 
    DEFAULT_FONT_THICKNESS, 
    DEFAULT_PADDING,
    PANEL_BG_COLOR,
    PANEL_ALPHA,
    TEXT_COLOR,
    DEFAULT_JPEG_QUALITY
)
from video_shots.core.exceptions import VideoShotsError
from video_shots.core.models import TimePoint
from video_shots.utils.time_utils import format_timestamp


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

            from video_shots.utils.time_utils import format_timestamp_for_name
            time_tag = format_timestamp_for_name(tp.seconds, precision=time_precision)
            filename = f"{prefix}_{tp.index:05d}_f{tp.frame_index:05d}_t{time_tag}.{image_format}"
            output_path = output_dir / "screenshots" / filename
            save_frame_image(framed, output_path, image_format)
            image_paths.append(output_path)

    finally:
        cap.release()

    return image_paths, duplicates


def split_video_into_segments(
    video_path: Path,
    output_dir: Path,
    segment_duration: float = 5.0,
    prefix: str = "segment"
) -> List[Path]:
    """Split video into segments of specified duration.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save video segments
        segment_duration: Duration of each segment in seconds (default: 5.0)
        prefix: Prefix for output segment files
        
    Returns:
        List of paths to created video segments
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoShotsError(f"Failed to open video '{video_path}'")
    
    try:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Calculate frames per segment
        frames_per_segment = int(fps * segment_duration)
        
        # Create output directory if it doesn't exist
        segments_dir = output_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video codec and properties for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        segment_paths = []
        segment_num = 0
        current_frame = 0
        
        print(f"Splitting video into {segment_duration}s segments...")
        print(f"Total duration: {duration:.2f}s, FPS: {fps:.2f}")
        
        while current_frame < total_frames:
            # Calculate end frame for this segment
            end_frame = min(current_frame + frames_per_segment, total_frames)
            
            # Create output filename
            segment_filename = f"{prefix}_{segment_num:03d}_{current_frame}_{end_frame}.mp4"
            segment_path = segments_dir / segment_filename
            
            # Initialize video writer for this segment
            out = cv2.VideoWriter(str(segment_path), fourcc, fps, (frame_width, frame_height))
            
            # Set position and read frames for this segment
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            
            frames_written = 0
            for frame_idx in range(current_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frames_written += 1
            
            out.release()
            
            if frames_written > 0:
                segment_paths.append(segment_path)
                start_time = current_frame / fps
                end_time = (current_frame + frames_written) / fps
                print(f"Created segment {segment_num}: {segment_filename} ({start_time:.1f}s - {end_time:.1f}s)")
                segment_num += 1
            else:
                # Remove empty file
                if segment_path.exists():
                    os.remove(segment_path)
            
            current_frame = end_frame
            
    finally:
        cap.release()
    
    print(f"Successfully created {len(segment_paths)} segments")
    return segment_paths
