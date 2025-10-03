"""Scene change detection for automatic keyframe selection."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from video_shots.core.models import TimePoint
from video_shots.core.exceptions import VideoShotsError


@dataclass
class SceneChange:
    """Detected scene change."""
    timestamp: float
    frame_index: int
    change_score: float
    change_type: str  # 'cut', 'fade', 'gradual'


class SceneChangeDetector:
    """Detect scene changes using multiple methods."""
    
    def __init__(self, 
                 histogram_threshold: float = 0.3,
                 edge_threshold: float = 0.4,
                 optical_flow_threshold: float = 0.5):
        self.histogram_threshold = histogram_threshold
        self.edge_threshold = edge_threshold
        self.optical_flow_threshold = optical_flow_threshold
        
    def detect_scene_changes(self, 
                           video_path: Path, 
                           sample_rate: float = 0.5) -> List[SceneChange]:
        """Detect scene changes in video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoShotsError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * sample_rate)
        
        scene_changes = []
        prev_frame = None
        prev_hist = None
        prev_edges = None
        
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    timestamp = frame_idx / fps
                    
                    if prev_frame is not None:
                        change_score, change_type = self._analyze_frame_difference(
                            prev_frame, frame, prev_hist, prev_edges
                        )
                        
                        if self._is_scene_change(change_score, change_type):
                            scene_changes.append(SceneChange(
                                timestamp=timestamp,
                                frame_index=frame_idx,
                                change_score=change_score,
                                change_type=change_type
                            ))
                    
                    prev_frame = frame.copy()
                    prev_hist = self._calculate_histogram(frame)
                    prev_edges = self._calculate_edges(frame)
                
                frame_idx += 1
                
        finally:
            cap.release()
        
        return self._filter_scene_changes(scene_changes)
    
    def _analyze_frame_difference(self, 
                                prev_frame: np.ndarray, 
                                curr_frame: np.ndarray,
                                prev_hist: np.ndarray,
                                prev_edges: np.ndarray) -> Tuple[float, str]:
        """Analyze difference between two frames."""
        # Histogram comparison
        curr_hist = self._calculate_histogram(curr_frame)
        hist_diff = 1 - cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
        
        # Edge comparison
        curr_edges = self._calculate_edges(curr_frame)
        edge_diff = np.mean(np.abs(prev_edges.astype(float) - curr_edges.astype(float))) / 255.0
        
        # Pixel difference (for cuts)
        pixel_diff = np.mean(np.abs(prev_frame.astype(float) - curr_frame.astype(float))) / 255.0
        
        # Combine scores
        change_score = (hist_diff + edge_diff + pixel_diff) / 3
        
        # Determine change type
        if pixel_diff > 0.6:
            change_type = "cut"
        elif hist_diff > 0.4 and edge_diff > 0.3:
            change_type = "fade"
        else:
            change_type = "gradual"
            
        return change_score, change_type
    
    def _calculate_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Calculate color histogram for frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], 
                           [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()
    
    def _calculate_edges(self, frame: np.ndarray) -> np.ndarray:
        """Calculate edge map for frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 50, 150)
    
    def _is_scene_change(self, change_score: float, change_type: str) -> bool:
        """Determine if change qualifies as scene change."""
        if change_type == "cut" and change_score > 0.5:
            return True
        elif change_type == "fade" and change_score > 0.4:
            return True
        elif change_type == "gradual" and change_score > 0.6:
            return True
        return False
    
    def _filter_scene_changes(self, scene_changes: List[SceneChange]) -> List[SceneChange]:
        """Filter out too-close scene changes."""
        if not scene_changes:
            return []
        
        filtered = [scene_changes[0]]
        min_interval = 2.0  # Minimum 2 seconds between scene changes
        
        for change in scene_changes[1:]:
            if change.timestamp - filtered[-1].timestamp > min_interval:
                filtered.append(change)
        
        return filtered


def create_scene_based_timepoints(video_path: Path, 
                                scene_changes: List[SceneChange],
                                frames_per_scene: int = 3) -> List[TimePoint]:
    """Create timepoints based on detected scene changes."""
    if not scene_changes:
        return []
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    timepoints = []
    
    # Add keyframes around each scene change
    for i, change in enumerate(scene_changes):
        # Add frame before scene change
        before_time = max(0, change.timestamp - 0.5)
        before_frame = int(before_time * fps)
        timepoints.append(TimePoint(
            seconds=before_time,
            frame_index=before_frame,
            index=len(timepoints) + 1
        ))
        
        # Add frame at scene change
        timepoints.append(TimePoint(
            seconds=change.timestamp,
            frame_index=change.frame_index,
            index=len(timepoints) + 1
        ))
        
        # Add frame after scene change
        after_time = change.timestamp + 0.5
        after_frame = int(after_time * fps)
        timepoints.append(TimePoint(
            seconds=after_time,
            frame_index=after_frame,
            index=len(timepoints) + 1
        ))
        
        # Add additional frames within the scene if requested
        if frames_per_scene > 3 and i < len(scene_changes) - 1:
            next_change = scene_changes[i + 1]
            scene_duration = next_change.timestamp - change.timestamp
            
            if scene_duration > 3.0:  # Only for longer scenes
                additional_frames = min(frames_per_scene - 3, int(scene_duration / 2))
                for j in range(additional_frames):
                    frame_time = change.timestamp + (j + 1) * (scene_duration / (additional_frames + 1))
                    frame_idx = int(frame_time * fps)
                    timepoints.append(TimePoint(
                        seconds=frame_time,
                        frame_index=frame_idx,
                        index=len(timepoints) + 1
                    ))
    
    # Sort by timestamp and remove duplicates
    timepoints.sort(key=lambda tp: tp.seconds)
    
    # Remove duplicates (frames too close to each other)
    filtered_timepoints = []
    min_interval = 0.2  # Minimum 200ms between frames
    
    for tp in timepoints:
        if not filtered_timepoints or tp.seconds - filtered_timepoints[-1].seconds > min_interval:
            filtered_timepoints.append(tp)
    
    return filtered_timepoints


def generate_scene_report(scene_changes: List[SceneChange], output_path: Path) -> None:
    """Generate scene detection report."""
    report = {
        "scene_detection_summary": {
            "total_scenes": len(scene_changes) + 1,  # +1 for first scene
            "scene_changes_detected": len(scene_changes),
            "change_types": {
                "cuts": len([c for c in scene_changes if c.change_type == "cut"]),
                "fades": len([c for c in scene_changes if c.change_type == "fade"]),
                "gradual": len([c for c in scene_changes if c.change_type == "gradual"])
            }
        },
        "scene_changes": [
            {
                "timestamp": f"{change.timestamp:.3f}s",
                "frame_index": change.frame_index,
                "change_score": round(change.change_score, 3),
                "change_type": change.change_type
            }
            for change in scene_changes
        ]
    }
    
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)