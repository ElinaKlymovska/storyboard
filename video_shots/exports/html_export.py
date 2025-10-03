"""HTML export utilities."""

from pathlib import Path
from typing import List
from video_shots.core.models import TimePoint
from video_shots.utils.time_utils import format_timestamp

def create_html_storyboard(video_name: str, screenshots: List[Path], timepoints: List[TimePoint], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for shot, tp in zip(screenshots, timepoints):
        rows.append(f"<tr><td><img src='{shot.as_posix()}' width='320'></td><td>{format_timestamp(tp.seconds, precision='ms')}</td><td>{tp.frame_index}</td></tr>")
    html_content = f"""<html><head><title>{video_name} Storyboard</title></head><body><h1>{video_name} Storyboard</h1><table border='1'>{''.join(rows)}</table></body></html>"""
    output_path.write_text(html_content, encoding='utf-8')
    return output_path

def create_editing_table(*args, **kwargs) -> Path:
    return create_html_storyboard(*args, **kwargs)
