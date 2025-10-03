"""Integrated scene detection and keyframe selection system."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import cv2

from video_shots.core.scene_detector import SceneChangeDetector, SceneChange
from video_shots.core.keyframe_selector import KeyframeSelector, FrameScore
from video_shots.core.models import TimePoint
from video_shots.utils.time_utils import format_timestamp, format_timestamp_for_name
from video_shots.core.exceptions import VideoShotsError


@dataclass
class Scene:
    """Scene with its keyframes."""
    scene_id: str
    start_time: float
    end_time: float
    duration: float
    scene_change: Optional[SceneChange]
    keyframes: List[FrameScore]
    total_candidates: int
    scene_description: str = ""


@dataclass
class SceneKeyframeResult:
    """Complete scene-keyframe analysis result."""
    video_path: Path
    video_duration: float
    total_scenes: int
    total_keyframes: int
    scenes: List[Scene]
    processing_summary: Dict


class SceneKeyframeIntegrator:
    """Integrates scene detection with keyframe selection."""
    
    def __init__(self, 
                 keyframes_per_scene: int = 3,
                 min_scene_duration: float = 2.0,
                 max_keyframes_total: int = 30):
        self.keyframes_per_scene = keyframes_per_scene
        self.min_scene_duration = min_scene_duration
        self.max_keyframes_total = max_keyframes_total
        
        self.scene_detector = SceneChangeDetector()
        self.keyframe_selector = KeyframeSelector()
    
    def process_video(self, 
                     video_path: Path, 
                     sample_interval: float = 0.5) -> SceneKeyframeResult:
        """Process video to detect scenes and select keyframes."""
        if not video_path.exists():
            raise VideoShotsError(f"Video file not found: {video_path}")
        
        print(f"🎬 Processing video: {video_path.name}")
        
        # Get video metadata
        video_duration = self._get_video_duration(video_path)
        print(f"⏱️  Video duration: {format_timestamp(video_duration, precision='s')}")
        
        # Detect scene changes
        print("🔍 Detecting scene changes...")
        scene_changes = self.scene_detector.detect_scene_changes(video_path, sample_rate=sample_interval)
        print(f"📋 Detected {len(scene_changes)} scene changes")
        
        # Create scene segments
        scenes = self._create_scene_segments(scene_changes, video_duration)
        print(f"🎭 Created {len(scenes)} scenes")
        
        # Process each scene to select keyframes
        print("🔑 Selecting keyframes for each scene...")
        for i, scene in enumerate(scenes, 1):
            print(f"   Scene {i}/{len(scenes)}: {format_timestamp(scene.start_time, precision='s')} - {format_timestamp(scene.end_time, precision='s')}")
            self._process_scene_keyframes(video_path, scene)
        
        # Calculate totals
        total_keyframes = sum(len(scene.keyframes) for scene in scenes)
        
        # Create processing summary
        processing_summary = self._create_processing_summary(scenes, scene_changes)
        
        print(f"✅ Processing complete: {total_keyframes} keyframes selected from {len(scenes)} scenes")
        
        return SceneKeyframeResult(
            video_path=video_path,
            video_duration=video_duration,
            total_scenes=len(scenes),
            total_keyframes=total_keyframes,
            scenes=scenes,
            processing_summary=processing_summary
        )
    
    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoShotsError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if fps <= 0 or frame_count <= 0:
            raise VideoShotsError("Invalid video metadata")
        
        return frame_count / fps
    
    def _create_scene_segments(self, 
                             scene_changes: List[SceneChange], 
                             video_duration: float) -> List[Scene]:
        """Create scene segments from detected changes."""
        scenes = []
        
        # First scene (from start to first change)
        if scene_changes:
            first_end = scene_changes[0].timestamp
        else:
            first_end = video_duration
        
        scenes.append(Scene(
            scene_id="scene_01",
            start_time=0.0,
            end_time=first_end,
            duration=first_end,
            scene_change=None,
            keyframes=[],
            total_candidates=0,
            scene_description="Opening scene"
        ))
        
        # Intermediate scenes
        for i, change in enumerate(scene_changes[:-1]):
            next_change = scene_changes[i + 1]
            duration = next_change.timestamp - change.timestamp
            
            # Skip very short scenes
            if duration >= self.min_scene_duration:
                scenes.append(Scene(
                    scene_id=f"scene_{len(scenes) + 1:02d}",
                    start_time=change.timestamp,
                    end_time=next_change.timestamp,
                    duration=duration,
                    scene_change=change,
                    keyframes=[],
                    total_candidates=0,
                    scene_description=f"Scene after {change.change_type} transition"
                ))
        
        # Final scene (from last change to end)
        if scene_changes:
            last_change = scene_changes[-1]
            final_duration = video_duration - last_change.timestamp
            
            if final_duration >= self.min_scene_duration:
                scenes.append(Scene(
                    scene_id=f"scene_{len(scenes) + 1:02d}",
                    start_time=last_change.timestamp,
                    end_time=video_duration,
                    duration=final_duration,
                    scene_change=last_change,
                    keyframes=[],
                    total_candidates=0,
                    scene_description=f"Final scene after {last_change.change_type} transition"
                ))
        
        return scenes
    
    def _process_scene_keyframes(self, video_path: Path, scene: Scene) -> None:
        """Process keyframes for a single scene."""
        # Generate candidate timepoints for this scene
        candidate_timepoints = self._generate_scene_timepoints(scene)
        scene.total_candidates = len(candidate_timepoints)
        
        if not candidate_timepoints:
            return
        
        # Reset keyframe selector for this scene
        self.keyframe_selector = KeyframeSelector()
        
        # Score all candidates
        frame_scores = self.keyframe_selector._score_all_frames(video_path, candidate_timepoints)
        
        # Select best keyframes for this scene
        target_count = min(self.keyframes_per_scene, len(frame_scores))
        
        if frame_scores:
            # Apply diversity filtering and select top frames
            diverse_frames = self.keyframe_selector._ensure_diversity(frame_scores, target_count)
            selected = sorted(diverse_frames, key=lambda x: x.total_score, reverse=True)
            scene.keyframes = selected[:target_count]
    
    def _generate_scene_timepoints(self, scene: Scene) -> List[TimePoint]:
        """Generate candidate timepoints within a scene."""
        # More candidates for longer scenes
        base_candidates = max(5, int(scene.duration * 2))  # 2 candidates per second
        
        timepoints = []
        
        if scene.duration <= 1.0:
            # Very short scene - just pick middle
            mid_time = scene.start_time + scene.duration / 2
            timepoints.append(TimePoint(
                seconds=mid_time,
                frame_index=int(mid_time * 30),  # Assume 30fps for frame index
                index=1
            ))
        else:
            # Distribute candidates across scene duration
            for i in range(base_candidates):
                # Use slight randomization to avoid always picking same points
                time_offset = (i + 0.5) * (scene.duration / base_candidates)
                timestamp = scene.start_time + time_offset
                
                # Don't go beyond scene end
                timestamp = min(timestamp, scene.end_time - 0.1)
                
                timepoints.append(TimePoint(
                    seconds=timestamp,
                    frame_index=int(timestamp * 30),  # Assume 30fps
                    index=i + 1
                ))
        
        return timepoints
    
    def _create_processing_summary(self, 
                                 scenes: List[Scene], 
                                 scene_changes: List[SceneChange]) -> Dict:
        """Create processing summary."""
        total_keyframes = sum(len(scene.keyframes) for scene in scenes)
        total_candidates = sum(scene.total_candidates for scene in scenes)
        
        # Scene distribution
        scene_durations = [scene.duration for scene in scenes]
        avg_scene_duration = sum(scene_durations) / len(scene_durations) if scene_durations else 0
        
        # Keyframe distribution
        keyframes_per_scene = [len(scene.keyframes) for scene in scenes]
        avg_keyframes_per_scene = sum(keyframes_per_scene) / len(keyframes_per_scene) if keyframes_per_scene else 0
        
        # Change type distribution
        change_types = {}
        for change in scene_changes:
            change_types[change.change_type] = change_types.get(change.change_type, 0) + 1
        
        return {
            "scene_analysis": {
                "total_scenes": len(scenes),
                "scene_changes_detected": len(scene_changes),
                "average_scene_duration": round(avg_scene_duration, 2),
                "shortest_scene": round(min(scene_durations), 2) if scene_durations else 0,
                "longest_scene": round(max(scene_durations), 2) if scene_durations else 0,
                "change_types": change_types
            },
            "keyframe_analysis": {
                "total_keyframes_selected": total_keyframes,
                "total_candidates_analyzed": total_candidates,
                "selection_rate": round(total_keyframes / total_candidates * 100, 1) if total_candidates > 0 else 0,
                "average_keyframes_per_scene": round(avg_keyframes_per_scene, 1),
                "keyframes_distribution": keyframes_per_scene
            },
            "quality_metrics": {
                "scenes_with_keyframes": len([s for s in scenes if s.keyframes]),
                "empty_scenes": len([s for s in scenes if not s.keyframes]),
                "total_video_coverage": round(sum(scene_durations), 2)
            }
        }


def generate_keyframe_table(result: SceneKeyframeResult, output_path: Path) -> None:
    """Generate table containing only keyframes organized by scene."""
    
    keyframe_table = {
        "video_info": {
            "filename": result.video_path.name,
            "duration": f"{format_timestamp(result.video_duration, precision='s')}",
            "total_scenes": result.total_scenes,
            "total_keyframes": result.total_keyframes
        },
        "scenes": [],
        "keyframe_summary": result.processing_summary
    }
    
    # Process each scene
    for scene in result.scenes:
        scene_data = {
            "scene_id": scene.scene_id,
            "description": scene.scene_description,
            "timing": {
                "start": f"{format_timestamp(scene.start_time, precision='s')}",
                "end": f"{format_timestamp(scene.end_time, precision='s')}",
                "duration": f"{format_timestamp(scene.duration, precision='s')}"
            },
            "scene_change": None,
            "keyframes": []
        }
        
        # Add scene change info if available
        if scene.scene_change:
            scene_data["scene_change"] = {
                "timestamp": f"{format_timestamp(scene.scene_change.timestamp, precision='s')}",
                "type": scene.scene_change.change_type,
                "score": round(scene.scene_change.change_score, 3)
            }
        
        # Add keyframes for this scene
        for kf in scene.keyframes:
            time_tag = format_timestamp_for_name(kf.time_point.seconds, precision='s')
            
            keyframe_data = {
                "keyframe_id": f"{scene.scene_id}_kf_{len(scene_data['keyframes']) + 1:02d}",
                "timecode": f"{format_timestamp(kf.time_point.seconds, precision='s')}",
                "frame_index": kf.time_point.frame_index,
                "filename_suggestion": f"{result.video_path.stem}_{scene.scene_id}_kf_{len(scene_data['keyframes']) + 1:02d}_t{time_tag}.png",
                "quality_scores": {
                    "visual_quality": round(kf.visual_quality, 2),
                    "content": round(kf.content_score, 2),
                    "uniqueness": round(kf.uniqueness, 2),
                    "motion": round(kf.motion_score, 2),
                    "composition": round(kf.composition_score, 2),
                    "total": round(kf.total_score, 2)
                },
                "selection_reason": kf.reasoning,
                "recommended_for": _categorize_keyframe_use(kf)
            }
            
            scene_data["keyframes"].append(keyframe_data)
        
        # Add statistics for this scene
        scene_data["statistics"] = {
            "keyframes_selected": len(scene.keyframes),
            "candidates_analyzed": scene.total_candidates,
            "selection_rate": f"{round(len(scene.keyframes) / scene.total_candidates * 100, 1)}%" if scene.total_candidates > 0 else "0%",
            "average_score": round(sum(kf.total_score for kf in scene.keyframes) / len(scene.keyframes), 2) if scene.keyframes else 0
        }
        
        keyframe_table["scenes"].append(scene_data)
    
    # Save the table
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(keyframe_table, f, indent=2, ensure_ascii=False)


def _categorize_keyframe_use(keyframe: FrameScore) -> List[str]:
    """Categorize keyframe for recommended use cases."""
    uses = []
    
    # Based on scores, suggest use cases
    if keyframe.visual_quality >= 7:
        uses.append("high-quality prints")
    
    if keyframe.content_score >= 8:
        uses.append("main storyline")
    
    if keyframe.composition_score >= 7:
        uses.append("artistic showcase")
    
    if keyframe.motion_score >= 6:
        uses.append("action sequences")
    
    if keyframe.uniqueness >= 8:
        uses.append("key moments")
    
    if keyframe.total_score >= 8:
        uses.append("thumbnail candidate")
    
    # Default use case
    if not uses:
        uses.append("general documentation")
    
    return uses


def generate_scene_timeline_report(result: SceneKeyframeResult, output_path: Path) -> None:
    """Generate timeline-style report showing scenes and keyframes."""
    
    timeline_report = {
        "video_timeline": {
            "video_name": result.video_path.name,
            "total_duration": f"{format_timestamp(result.video_duration, precision='s')}",
            "processing_date": str(Path().cwd() / "timestamp"),  # Current timestamp would go here
        },
        "timeline_events": [],
        "summary": result.processing_summary
    }
    
    # Create timeline events
    all_events = []
    
    # Add scene start events
    for scene in result.scenes:
        all_events.append({
            "timestamp": scene.start_time,
            "type": "scene_start",
            "data": {
                "scene_id": scene.scene_id,
                "description": scene.scene_description,
                "duration": scene.duration
            }
        })
        
        # Add scene change events
        if scene.scene_change:
            all_events.append({
                "timestamp": scene.scene_change.timestamp,
                "type": "scene_change",
                "data": {
                    "change_type": scene.scene_change.change_type,
                    "change_score": scene.scene_change.change_score,
                    "new_scene": scene.scene_id
                }
            })
        
        # Add keyframe events
        for kf in scene.keyframes:
            all_events.append({
                "timestamp": kf.time_point.seconds,
                "type": "keyframe",
                "data": {
                    "scene_id": scene.scene_id,
                    "keyframe_id": f"{scene.scene_id}_kf_{scene.keyframes.index(kf) + 1:02d}",
                    "total_score": kf.total_score,
                    "reason": kf.reasoning
                }
            })
    
    # Sort events by timestamp
    all_events.sort(key=lambda x: x["timestamp"])
    
    # Format events for timeline
    for event in all_events:
        timeline_event = {
            "timecode": format_timestamp(event["timestamp"], precision='s'),
            "timestamp_seconds": round(event["timestamp"], 3),
            "event_type": event["type"],
            "details": event["data"]
        }
        timeline_report["timeline_events"].append(timeline_event)
    
    # Save timeline report
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(timeline_report, f, indent=2, ensure_ascii=False)


def export_keyframes_csv(result: SceneKeyframeResult, output_path: Path) -> None:
    """Export keyframes to CSV format for spreadsheet applications."""
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow([
            'Scene ID', 'Keyframe ID', 'Timecode', 'Frame Index',
            'Visual Quality', 'Content Score', 'Uniqueness', 
            'Motion Score', 'Composition Score', 'Total Score',
            'Selection Reason', 'Recommended Use'
        ])
        
        # Write keyframe data
        for scene in result.scenes:
            for i, kf in enumerate(scene.keyframes, 1):
                keyframe_id = f"{scene.scene_id}_kf_{i:02d}"
                recommended_use = ", ".join(_categorize_keyframe_use(kf))
                
                writer.writerow([
                    scene.scene_id,
                    keyframe_id,
                    format_timestamp(kf.time_point.seconds, precision='s'),
                    kf.time_point.frame_index,
                    round(kf.visual_quality, 2),
                    round(kf.content_score, 2),
                    round(kf.uniqueness, 2),
                    round(kf.motion_score, 2),
                    round(kf.composition_score, 2),
                    round(kf.total_score, 2),
                    kf.reasoning,
                    recommended_use
                ])