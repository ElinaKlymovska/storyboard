from __future__ import annotations

import base64
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
from notion_client import Client
from PIL import Image

import replicate

import sys
from pathlib import Path
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = PROJECT_ROOT.as_posix()
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from .config import (
    DEFAULT_ANALYSIS_PROMPT,
    SHOT_ANALYSIS_PROMPT,
    VIDEO_ANALYSIS_INTERVAL,
    VIDEO_ANALYSIS_MODEL,
)
from .models import TimePoint
from .utils import collect_timepoints
from .utils.audio_utils import get_vo_for_timestamp, parse_audio_transcript
from .core.image_analysis import analyze_image_with_replicate

SUPABASE_MODULE_PATH = PROJECT_ROOT / "supabase_storage.py"
if SUPABASE_MODULE_PATH.exists():
    spec = importlib.util.spec_from_file_location("supabase_storage", SUPABASE_MODULE_PATH)
    if spec and spec.loader:
        supabase_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(supabase_module)
        SupabaseStorageManager = supabase_module.SupabaseStorageManager
    else:
        raise ImportError("Unable to load supabase_storage module")
else:
    from supabase_storage import SupabaseStorageManager


SHOT_SEGMENT_DURATION = 5.0


@dataclass
class KeyframeRecord:
    scene_id: str
    shot_id: str
    kf_index: int
    timecode: str
    thumb_url: str
    action_note: str
    dialogue_or_vo: str
    transition: str
    status: str
    notes: str


@dataclass
class ShotRecord:
    scene_id: str
    shot_id: str
    start_tc: str
    end_tc: str
    shot_type: str
    objective_or_beat: str
    action: str
    camera: str
    audio: str
    transition_out: str
    status: str


class VideoAnalysisPipeline:
    def __init__(self, video_path: str) -> None:
        load_dotenv()
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self.video_name = self.video_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("output") / f"{self.video_name}_{timestamp}"
        self.frames_dir = self.output_dir / "keyframes"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.supabase = SupabaseStorageManager()

        notion_token = os.getenv("NOTION_API_TOKEN") or os.getenv("NOTION_TOKEN")
        if not notion_token:
            raise RuntimeError("NOTION_API_TOKEN (or NOTION_TOKEN) is not set")
        self.notion = Client(auth=notion_token)

        self.keyframes_db = os.getenv("KEYFRAMES_DB_ID")
        self.storyboard_db = os.getenv("STORYBOARD_DB_ID")
        if not self.keyframes_db or not self.storyboard_db:
            raise RuntimeError("KEYFRAMES_DB_ID and STORYBOARD_DB_ID must be configured")

        transcript_path = Path("input") / "audio" / f"{self.video_name}.txt"
        self.transcript = (
            parse_audio_transcript(transcript_path)
            if transcript_path.exists()
            else {}
        )

    def run(self) -> None:
        metadata = self._get_video_metadata()
        keyframes = self._process_keyframes(metadata)
        shots = self._process_shots(metadata)
        self._sync_keyframes_to_notion(keyframes)
        self._sync_storyboard_to_notion(shots)

    def _get_video_metadata(self) -> Dict[str, Any]:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps else 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if fps <= 0 or frame_count <= 0:
            raise RuntimeError("Unable to determine FPS or frame count")

        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "resolution": (width, height),
        }

    def _process_keyframes(self, metadata: Dict[str, Any]) -> List[KeyframeRecord]:
        fps = metadata["fps"]
        frame_count = metadata["frame_count"]
        duration = metadata["duration"]

        timepoints = collect_timepoints(
            start=0.0,
            end=duration,
            step=VIDEO_ANALYSIS_INTERVAL,
            fps=fps,
            frame_count=frame_count,
        )

        keyframes: List[KeyframeRecord] = []
        shot_counters: Dict[str, int] = defaultdict(int)

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        for tp in timepoints:
            cap.set(cv2.CAP_PROP_POS_FRAMES, tp.frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            filename = f"kf_{tp.index:05d}_t{self._format_timecode(tp.seconds).replace(':', '-')}.png"
            frame_path = self.frames_dir / filename
            pil_image.save(frame_path)

            analysis = analyze_image_with_replicate(frame_path, DEFAULT_ANALYSIS_PROMPT)
            scene_id = analysis.get("scene_id") or "scene_01"

            default_shot_id = self._shot_id_for_time(tp.seconds)
            shot_id = analysis.get("shot_id") or default_shot_id
            shot_counters[shot_id] += 1
            kf_index = analysis.get("kf_index") or shot_counters[shot_id]

            transition = self._extract_transition_value(analysis.get("transition_in_out"))
            status = (analysis.get("status") or "draft").lower()
            if status not in {"draft", "approved"}:
                status = "draft"

            supabase_result = self.supabase.upload_image_from_pil(
                pil_image,
                filename,
                folder=f"screenshots/{self.video_name}"
            )
            thumb_url = supabase_result.get("public_url") if supabase_result else ""

            dialogue = get_vo_for_timestamp(self.transcript, tp.seconds)
            if not dialogue:
                dialogue = analysis.get("dialogue_or_vo", "")

            keyframes.append(
                KeyframeRecord(
                    scene_id=scene_id,
                    shot_id=shot_id,
                    kf_index=int(kf_index),
                    timecode=self._format_timecode(tp.seconds),
                    thumb_url=thumb_url,
                    action_note=analysis.get("action_note", ""),
                    dialogue_or_vo=dialogue,
                    transition=transition,
                    status=status,
                    notes=analysis.get("notes", ""),
                )
            )

        cap.release()
        return keyframes

    def _process_shots(self, metadata: Dict[str, Any]) -> List[ShotRecord]:
        duration = metadata["duration"]
        shots: List[ShotRecord] = []

        try:
            clip = VideoFileClip(str(self.video_path))
        except Exception as exc:
            raise RuntimeError(f"Unable to load video for shot analysis: {exc}")

        start = 0.0
        shot_index = 0
        while start < duration:
            end = min(start + SHOT_SEGMENT_DURATION, duration)
            shot_index += 1
            shot_id = f"shot_{shot_index:02d}"
            analysis = self._analyze_shot_segment(clip, start, end, shot_id)

            scene_id = analysis.get("scene_id") or "scene_01"
            shot_type = analysis.get("shot_type") or "WS"
            transition_out = analysis.get("transition_out") or "cut"
            status = (analysis.get("status") or "draft").lower()
            if status not in {"draft", "approved"}:
                status = "draft"

            audio_summary = analysis.get("audio") or self._collect_audio_summary(start, end)

            shots.append(
                ShotRecord(
                    scene_id=scene_id,
                    shot_id=analysis.get("shot_id") or shot_id,
                    start_tc=self._format_timecode(start),
                    end_tc=self._format_timecode(end),
                    shot_type=shot_type,
                    objective_or_beat=analysis.get("objective_or_beat", ""),
                    action=analysis.get("action", ""),
                    camera=analysis.get("camera", ""),
                    audio=audio_summary,
                    transition_out=transition_out,
                    status=status,
                )
            )

            start += SHOT_SEGMENT_DURATION

        clip.close()
        return shots

    def _analyze_shot_segment(
        self,
        clip: VideoFileClip,
        start: float,
        end: float,
        shot_id: str,
    ) -> Dict[str, Any]:
        prompt = SHOT_ANALYSIS_PROMPT
        prompt += (
            f"\nShot ID suggestion: {shot_id}\n"
            f"Segment start: {self._format_timecode(start)}\n"
            f"Segment end: {self._format_timecode(end)}\n"
        )

        try:
            subclip = clip.subclip(start, end)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
                tmp_path = Path(tmp_video.name)
            subclip.write_videofile(
                str(tmp_path),
                codec="libx264",
                audio=True,
                verbose=False,
                logger=None,
            )
            subclip.close()

            with open(tmp_path, "rb") as fp:
                encoded = base64.b64encode(fp.read()).decode("utf-8")
            tmp_path.unlink(missing_ok=True)

            video_payload = f"data:video/mp4;base64,{encoded}"
            output = replicate.run(
                VIDEO_ANALYSIS_MODEL,
                input={
                    "video": video_payload,
                    "prompt": prompt,
                },
            )
            if hasattr(output, "__iter__") and not isinstance(output, str):
                result = "".join(str(item) for item in output)
            else:
                result = str(output)

            analysis = self._parse_json_from_text(result)
            if not isinstance(analysis, dict):
                return {}
            return analysis
        except Exception as exc:
            return {"error": str(exc)}

    def _collect_audio_summary(self, start: float, end: float) -> str:
        if not self.transcript:
            return ""
        entries = []
        for timestamp, data in self.transcript.items():
            if start - 1e-3 <= timestamp <= end + 1e-3:
                entries.append(data.get("text", ""))
        return " ".join(entries)

    def _sync_keyframes_to_notion(self, records: Iterable[KeyframeRecord]) -> None:
        self._clear_database(self.keyframes_db)
        for record in records:
            properties = {
                "scene_id": self._title_prop(record.scene_id),
                "shot_id": self._rich_text_prop(record.shot_id),
                "kf_index": {"number": record.kf_index},
                "timecode": self._rich_text_prop(record.timecode),
                "thumb_url": {"url": record.thumb_url or None},
                "action_note": self._rich_text_prop(record.action_note),
                "dialogue_or_vo": self._rich_text_prop(record.dialogue_or_vo),
                "transition_in/out": self._select_prop(record.transition),
                "status": self._select_prop(record.status),
                "notes": self._rich_text_prop(record.notes),
            }
            self.notion.pages.create(
                parent={"database_id": self.keyframes_db},
                properties=properties,
            )

    def _sync_storyboard_to_notion(self, records: Iterable[ShotRecord]) -> None:
        self._clear_database(self.storyboard_db)
        for record in records:
            properties = {
                "scene_id": self._title_prop(record.scene_id),
                "shot_id": self._rich_text_prop(record.shot_id),
                "start_tc": self._rich_text_prop(record.start_tc),
                "end_tc": self._rich_text_prop(record.end_tc),
                "shot_type": self._select_prop(record.shot_type),
                "objective/beat": self._rich_text_prop(record.objective_or_beat),
                "action": self._rich_text_prop(record.action),
                "camera": self._rich_text_prop(record.camera),
                "audio": self._rich_text_prop(record.audio),
                "transition_out": self._select_prop(record.transition_out),
                "status": self._select_prop(record.status),
            }
            self.notion.pages.create(
                parent={"database_id": self.storyboard_db},
                properties=properties,
            )

    def _clear_database(self, database_id: str) -> None:
        start_cursor: Optional[str] = None
        while True:
            response = self.notion.databases.query(database_id, start_cursor=start_cursor)
            results = response.get("results", [])
            for page in results:
                self.notion.pages.update(page["id"], archived=True)
            start_cursor = response.get("next_cursor")
            if not start_cursor:
                break

    @staticmethod
    def _shot_id_for_time(seconds: float) -> str:
        index = int(seconds // SHOT_SEGMENT_DURATION) + 1
        return f"shot_{index:02d}"

    @staticmethod
    def _format_timecode(seconds: float) -> str:
        milliseconds = int(round((seconds - math.floor(seconds)) * 1000))
        total_seconds = int(math.floor(seconds))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

    @staticmethod
    def _extract_transition_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            inbound = value.get("in")
            outbound = value.get("out")
            if inbound and outbound:
                return f"{inbound}/{outbound}"
            if inbound:
                return str(inbound)
            if outbound:
                return str(outbound)
        return "cut"

    @staticmethod
    def _parse_json_from_text(text: str) -> Any:
        try:
            json_match = json.loads(text)
            return json_match
        except Exception:
            pass
        match = None
        try:
            match = VideoAnalysisPipeline._extract_json_substring(text)
        except Exception:
            return {}
        if not match:
            return {}
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _extract_json_substring(text: str) -> Optional[str]:
        stack = []
        start = None
        for idx, char in enumerate(text):
            if char == "{":
                if not stack:
                    start = idx
                stack.append(char)
            elif char == "}":
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        return text[start : idx + 1]
        return None

    @staticmethod
    def _title_prop(value: str) -> Dict[str, Any]:
        return {"title": [{"type": "text", "text": {"content": value or "-"}}]}

    @staticmethod
    def _rich_text_prop(value: str) -> Dict[str, Any]:
        if not value:
            return {"rich_text": []}
        return {
            "rich_text": [
                {
                    "type": "text",
                    "text": {"content": value},
                }
            ]
        }

    @staticmethod
    def _select_prop(value: str) -> Dict[str, Any]:
        if not value:
            return {"select": None}
        return {"select": {"name": value}}


def main(video_path: str) -> None:
    pipeline = VideoAnalysisPipeline(video_path)
    pipeline.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run video analysis pipeline")
    parser.add_argument("video_path", help="Path to source video file")
    args = parser.parse_args()
    main(args.video_path)
