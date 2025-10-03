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

from video_shots.config.config import (
    DEFAULT_ANALYSIS_PROMPT,
    SHOT_ANALYSIS_PROMPT,
    VIDEO_ANALYSIS_INTERVAL,
    VIDEO_ANALYSIS_MODEL,
    KEYFRAME_SELECTION_MODE,
    CHANGE_SENSITIVITY,
    SIMILARITY_THRESHOLD,
    MAX_FRAMES_PER_SECOND,
    FILTER_STRATEGY,
    EVENT_CLASSIFICATION_WEIGHTS,
    MIN_SIGNIFICANCE_THRESHOLD,
    KEEP_EVENT_PRIORITIES,
)
from video_shots.core.models import TimePoint
from video_shots.utils.time_utils import collect_timepoints
from video_shots.utils.audio_utils import get_vo_for_timestamp, parse_audio_transcript
from video_shots.core.image_analysis import analyze_image_with_replicate
from video_shots.core.keyframe_selector import KeyframeSelector
from video_shots.core.audio_analyzer import AudioAnalyzer

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
    from video_shots.exports.supabase_storage import SupabaseStorageManager


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
        self.video_path = self._validate_video_path(video_path)
        self.video_name = self.video_path.stem
        self.output_dir, self.frames_dir = self._setup_output_directories()
        self.supabase = SupabaseStorageManager()
        self.notion = self._setup_notion_client()
        self.keyframes_db, self.storyboard_db = self._setup_notion_databases()
        self.transcript = self._load_transcript()
        self.audio_analysis = self._analyze_audio_track()

    def _validate_video_path(self, video_path: str) -> Path:
        """Validate and return video path."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        return path

    def _setup_output_directories(self) -> tuple[Path, Path]:
        """Setup output directories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output") / f"{self.video_name}_{timestamp}"
        frames_dir = output_dir / "keyframes"
        frames_dir.mkdir(parents=True, exist_ok=True)
        return output_dir, frames_dir

    def _setup_notion_client(self) -> Client:
        """Setup Notion client."""
        notion_token = os.getenv("NOTION_API_TOKEN") or os.getenv("NOTION_TOKEN")
        if not notion_token:
            raise RuntimeError("NOTION_API_TOKEN (or NOTION_TOKEN) is not set")
        return Client(auth=notion_token)

    def _setup_notion_databases(self) -> tuple[str, str]:
        """Setup Notion database IDs."""
        keyframes_db = os.getenv("KEYFRAMES_DB_ID")
        storyboard_db = os.getenv("STORYBOARD_DB_ID")
        if not keyframes_db or not storyboard_db:
            raise RuntimeError("KEYFRAMES_DB_ID and STORYBOARD_DB_ID must be configured")
        return keyframes_db, storyboard_db

    def _load_transcript(self) -> Dict:
        """Load audio transcript if available."""
        transcript_path = Path("input") / "audio" / f"{self.video_name}.txt"
        return (
            parse_audio_transcript(transcript_path)
            if transcript_path.exists()
            else {}
        )
    
    def _analyze_audio_track(self) -> Dict:
        """Analyze audio track from video."""
        try:
            audio_analyzer = AudioAnalyzer(self.video_path)
            audio_dir = Path("input") / "audio"
            results = audio_analyzer.analyze_full_audio_track(audio_dir)
            
            if results.get('success'):
                print(f"🎵 Audio analysis completed for: {self.video_name}")
                if results.get('transcript'):
                    print(f"📝 Generated transcript with {len(results['transcript'])} segments")
                if results.get('audio_levels'):
                    print(f"🔊 Analyzed {len(results['audio_levels'])} audio segments")
            
            return results
        except Exception as e:
            print(f"⚠️ Audio analysis failed: {e}")
            return {}

    def run(self) -> None:
        metadata = self._get_video_metadata()
        keyframes = self._process_keyframes(metadata)
        shots = self._process_shots(metadata)
        self._sync_keyframes_to_notion(keyframes)
        self._sync_storyboard_to_notion(shots)
        
        # Generate event-driven analysis report if using that mode
        if KEYFRAME_SELECTION_MODE == "event_driven":
            self._generate_event_analysis_report(metadata)
    
    def _generate_event_analysis_report(self, metadata: Dict[str, Any]) -> None:
        """Generate detailed analysis report for event-driven mode."""
        try:
            keyframe_selector = KeyframeSelector(
                mode="event_driven",
                facial_sensitivity=CHANGE_SENSITIVITY.get('facial', 0.15),
                pose_sensitivity=CHANGE_SENSITIVITY.get('pose', 0.25),
                scene_sensitivity=CHANGE_SENSITIVITY.get('scene', 0.4),
                similarity_threshold=SIMILARITY_THRESHOLD,
                filter_strategy=FILTER_STRATEGY
            )
            
            analysis_data = keyframe_selector.get_event_driven_analysis(self.video_path)
            
            # Save detailed analysis report
            report_path = self.output_dir / "event_analysis_report.json"
            
            # Create serializable report
            report = {
                "video_metadata": {
                    "duration": metadata["duration"],
                    "fps": metadata["fps"],
                    "frame_count": metadata["frame_count"],
                    "resolution": metadata["resolution"]
                },
                "analysis_summary": {
                    "total_changes_detected": analysis_data["total_changes_detected"],
                    "classified_events": analysis_data["classified_events"],
                    "filtered_events": analysis_data["filtered_events"],
                    "final_timepoints": analysis_data["final_timepoints"],
                    "compression_ratio": analysis_data["compression_ratio"]
                },
                "filtering_stats": analysis_data["filtering_stats"],
                "settings_used": {
                    "mode": KEYFRAME_SELECTION_MODE,
                    "change_sensitivity": CHANGE_SENSITIVITY,
                    "similarity_threshold": SIMILARITY_THRESHOLD,
                    "filter_strategy": FILTER_STRATEGY,
                    "min_significance_threshold": MIN_SIGNIFICANCE_THRESHOLD
                },
                "detected_changes": [
                    {
                        "timestamp": f"{event.timestamp:.3f}s",
                        "frame_index": event.frame_index,
                        "change_type": event.change_type.value,
                        "change_score": round(event.change_score, 3),
                        "confidence": round(event.confidence, 3),
                        "description": event.description
                    }
                    for event in analysis_data["change_events"]
                ],
                "classified_events": [
                    {
                        "timestamp": f"{event.original_event.timestamp:.3f}s",
                        "frame_index": event.original_event.frame_index,
                        "category": event.category.value,
                        "priority": event.priority.value,
                        "significance_score": round(event.significance_score, 3),
                        "recommended_action": event.recommended_action,
                        "description": event.original_event.description,
                        "tags": event.tags
                    }
                    for event in analysis_data["classified_events"]
                ],
                "final_selected_timepoints": [
                    {
                        "index": tp.index,
                        "timestamp": f"{tp.seconds:.3f}s",
                        "frame_index": tp.frame_index
                    }
                    for tp in analysis_data["filter_result"].selected_timepoints
                ]
            }
            
            import json
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"📊 Event analysis report saved to: {report_path}")
            print(f"📈 Detected {analysis_data['total_changes_detected']} changes, "
                  f"selected {analysis_data['final_timepoints']} keyframes "
                  f"(compression ratio: {analysis_data['compression_ratio']:.2f})")
            
        except Exception as e:
            print(f"⚠️ Failed to generate event analysis report: {e}")

    def _get_video_metadata(self) -> Dict[str, Any]:
        """Extract video metadata."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps else 0.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if fps <= 0 or frame_count <= 0:
                raise RuntimeError("Unable to determine FPS or frame count")

            return {
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "resolution": (width, height),
            }
        finally:
            cap.release()

    def _process_keyframes(self, metadata: Dict[str, Any]) -> List[KeyframeRecord]:
        """Process keyframes from video and analyze them."""
        timepoints = self._calculate_keyframe_timepoints(metadata)
        keyframes: List[KeyframeRecord] = []
        shot_counters: Dict[str, int] = defaultdict(int)

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        try:
            for tp in timepoints:
                keyframe = self._process_single_keyframe(cap, tp, shot_counters)
                if keyframe:
                    keyframes.append(keyframe)
        finally:
            cap.release()
        
        return keyframes

    def _calculate_keyframe_timepoints(self, metadata: Dict[str, Any]) -> List[TimePoint]:
        """Calculate timepoints for keyframe extraction."""
        if KEYFRAME_SELECTION_MODE == "event_driven":
            # Use event-driven keyframe selection
            keyframe_selector = KeyframeSelector(
                mode="event_driven",
                facial_sensitivity=CHANGE_SENSITIVITY.get('facial', 0.15),
                pose_sensitivity=CHANGE_SENSITIVITY.get('pose', 0.25),
                scene_sensitivity=CHANGE_SENSITIVITY.get('scene', 0.4),
                similarity_threshold=SIMILARITY_THRESHOLD,
                filter_strategy=FILTER_STRATEGY
            )
            
            # Estimate target count based on video duration and desired density
            target_keyframes = max(10, int(metadata["duration"] / 2))  # Approximately one keyframe every 2 seconds
            
            selected_timepoints = keyframe_selector.select_keyframes(
                video_path=self.video_path,
                target_count=target_keyframes
            )
            
            return selected_timepoints
        else:
            # Use traditional interval-based approach
            return collect_timepoints(
                start=0.0,
                end=metadata["duration"],
                step=VIDEO_ANALYSIS_INTERVAL,
                fps=metadata["fps"],
                frame_count=metadata["frame_count"],
            )

    def _process_single_keyframe(
        self, cap: cv2.VideoCapture, tp: TimePoint, shot_counters: Dict[str, int]
    ) -> Optional[KeyframeRecord]:
        """Process a single keyframe."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, tp.frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        filename = f"kf_{tp.index:05d}_t{self._format_timecode(tp.seconds).replace(':', '-')}.png"
        frame_path = self.frames_dir / filename
        pil_image.save(frame_path)

        analysis = analyze_image_with_replicate(frame_path, DEFAULT_ANALYSIS_PROMPT)
        return self._create_keyframe_record(analysis, tp, pil_image, filename, shot_counters)

    def _create_keyframe_record(
        self, analysis: Dict, tp: TimePoint, pil_image: Image.Image, 
        filename: str, shot_counters: Dict[str, int]
    ) -> KeyframeRecord:
        """Create keyframe record from analysis data."""
        scene_id = analysis.get("scene_id") or "scene_01"
        default_shot_id = self._shot_id_for_time(tp.seconds)
        shot_id = analysis.get("shot_id") or default_shot_id
        shot_counters[shot_id] += 1
        kf_index = analysis.get("kf_index") or shot_counters[shot_id]

        transition = self._extract_transition_value(analysis.get("transition_in_out"))
        status = self._normalize_status(analysis.get("status"))
        thumb_url = self._upload_to_supabase(pil_image, filename)
        dialogue = self._get_dialogue_for_timepoint(tp.seconds, analysis)

        return KeyframeRecord(
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

    def _normalize_status(self, status: Optional[str]) -> str:
        """Normalize status to valid values."""
        normalized = (status or "draft").lower()
        return normalized if normalized in {"draft", "approved"} else "draft"

    def _upload_to_supabase(self, pil_image: Image.Image, filename: str) -> str:
        """Upload image to Supabase and return URL."""
        supabase_result = self.supabase.upload_image_from_pil(
            pil_image, filename, folder=f"screenshots/{self.video_name}"
        )
        return supabase_result.get("public_url") if supabase_result else ""

    def _get_dialogue_for_timepoint(self, seconds: float, analysis: Dict) -> str:
        """Get dialogue for timepoint from transcript or analysis."""
        # First try existing transcript
        dialogue = get_vo_for_timestamp(self.transcript, seconds)
        
        # If no dialogue found, try audio analysis transcript
        if not dialogue and self.audio_analysis.get('transcript'):
            for segment in self.audio_analysis['transcript']:
                seg_start = segment.get('start_time', 0)
                seg_end = segment.get('end_time', 0)
                if seg_start <= seconds <= seg_end:
                    dialogue = segment.get('text', '')
                    break
        
        # Fallback to analysis
        return dialogue if dialogue else analysis.get("dialogue_or_vo", "")

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
        """Collect audio summary for a time segment."""
        entries = []
        
        # First try to get from existing transcript
        if self.transcript:
            for timestamp, data in self.transcript.items():
                if start - 1e-3 <= timestamp <= end + 1e-3:
                    entries.append(data.get("text", ""))
        
        # Then try to get from audio analysis transcript
        if not entries and self.audio_analysis.get('transcript'):
            for segment in self.audio_analysis['transcript']:
                seg_start = segment.get('start_time', 0)
                seg_end = segment.get('end_time', 0)
                if (seg_start <= end and seg_end >= start):  # Overlapping segments
                    entries.append(segment.get('text', ''))
        
        # Also add audio level information
        audio_info = []
        if self.audio_analysis.get('audio_levels'):
            for segment in self.audio_analysis['audio_levels']:
                seg_start = segment.get('start_time', 0)
                seg_end = segment.get('end_time', 0)
                if (seg_start <= end and seg_end >= start):
                    if segment.get('is_silence'):
                        audio_info.append("silence")
                    else:
                        db_level = segment.get('db_level', -60)
                        if db_level > -20:
                            audio_info.append("loud")
                        elif db_level > -40:
                            audio_info.append("normal")
                        else:
                            audio_info.append("quiet")
        
        text_summary = " ".join(entries)
        audio_summary = f"[Audio: {', '.join(set(audio_info))}]" if audio_info else ""
        
        return f"{text_summary} {audio_summary}".strip()

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
