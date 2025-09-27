"""CLI interface for video-shots."""

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .config import MAX_FRAMES_WARNING
from .exceptions import VideoShotsError
from .core import process_video_frames
from .utils import collect_timepoints, estimate_total_frames, parse_time_value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="video-shots",
        description="Create video screenshots at time intervals",
    )
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument(
        "-n",
        "--interval",
        required=True,
        help="Interval between frames (e.g.: 0.5s, 200ms, 00:00:03.5)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="0",
        help="Start time (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End time (default: video duration)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames",
    )
    return parser


def resolve_output_dir(base_dir: Path, video_path: Path) -> Path:
    """Determine output directory."""
    base_name = video_path.stem
    timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = base_dir / f"{base_name}_{timestamp}"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "screenshots").mkdir(exist_ok=True)

    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI function."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    video_path = Path(args.video_path)
    if not video_path.exists():
        parser.error(f"Video file '{video_path}' does not exist")

    try:
        interval_seconds = parse_time_value(args.interval)
    except VideoShotsError as exc:
        parser.error(str(exc))

    try:
        start_seconds = parse_time_value(args.start, allow_zero=True)
    except VideoShotsError as exc:
        parser.error(f"Error in --start: {exc}")

    end_seconds = None
    if args.end is not None:
        try:
            end_seconds = parse_time_value(args.end, allow_zero=True)
        except VideoShotsError as exc:
            parser.error(f"Error in --end: {exc}")

    # Open video and get metadata
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        parser.error(f"Failed to open video '{video_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        parser.error("Unable to get video FPS")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        parser.error("Unable to determine frame count")

    duration = frame_count / fps
    if end_seconds is None:
        end_seconds = duration

    if start_seconds > duration:
        cap.release()
        parser.error("Start time exceeds video duration")

    estimated_total = estimate_total_frames(end_seconds - start_seconds, interval_seconds)
    if args.max_frames is not None and estimated_total > args.max_frames:
        cap.release()
        parser.error(
            f"Expected frame count ({estimated_total}) exceeds --max-frames"
        )

    if estimated_total > MAX_FRAMES_WARNING:
        print(
            f"⚠️ Warning: {estimated_total} screenshots will be created."
            " Consider --start/--end limits or increase interval.",
            file=sys.stderr,
        )

    theoretical_min_step = 1.0 / fps
    if interval_seconds < theoretical_min_step:
        print(
            f"ℹ️ Interval {interval_seconds:.6f}s smaller than 1/FPS ({theoretical_min_step:.6f}s)."
            " Possible frame duplication.",
            file=sys.stderr,
        )

    try:
        timepoints = collect_timepoints(start_seconds, end_seconds, interval_seconds, fps, frame_count)
    except VideoShotsError as exc:
        cap.release()
        parser.error(str(exc))

    cap.release()

    output_dir = resolve_output_dir(Path.cwd() / "output", video_path)

    time_precision = "ms" if interval_seconds < 1.0 else "sec"

    image_paths, duplicates = process_video_frames(
        video_path,
        timepoints,
        output_dir,
        prefix="shot",
        image_format="png",
        rotation=0,
        time_precision=time_precision,
    )

    print(f"Done. Saved {len(image_paths)} files to '{output_dir}'.")
    if duplicates:
        print(f"ℹ️ {duplicates} timestamps pointed to same frame (FPS limitation).")

    return 0
