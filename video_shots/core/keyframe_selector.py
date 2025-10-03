"""Automatic keyframe selection system."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
import json

from video_shots.core.models import TimePoint
from video_shots.core.exceptions import VideoShotsError
from video_shots.core.micro_change_detector import MicroChangeDetector, ChangeEvent, create_timepoints_from_changes
from video_shots.core.intelligent_filter import IntelligentFilter, FilterStrategy, FilteredResult
from video_shots.core.event_classifier import EventClassifier, filter_by_classification, EventPriority, EventCategory


@dataclass
class FrameScore:
    """Score for a single frame."""
    time_point: TimePoint
    visual_quality: float  # 0-10
    content_score: float   # 0-10
    uniqueness: float      # 0-10
    motion_score: float    # 0-10
    composition_score: float  # 0-10
    total_score: float     # 0-10
    reasoning: str


class KeyframeSelector:
    """Automatic keyframe selection using multiple criteria."""
    
    def __init__(self, 
                 quality_weight: float = 0.25,
                 content_weight: float = 0.30,
                 uniqueness_weight: float = 0.20,
                 motion_weight: float = 0.15,
                 composition_weight: float = 0.10,
                 mode: str = "interval_based",  # "interval_based" or "event_driven"
                 facial_sensitivity: float = 0.15,
                 pose_sensitivity: float = 0.25,
                 scene_sensitivity: float = 0.4,
                 similarity_threshold: float = 0.85,
                 filter_strategy: str = "keep_first_discard_rest"):
        self.quality_weight = quality_weight
        self.content_weight = content_weight
        self.uniqueness_weight = uniqueness_weight
        self.motion_weight = motion_weight
        self.composition_weight = composition_weight
        self.mode = mode
        
        # History for similarity comparison
        self.frame_histograms = []
        self.previous_frame = None
        
        # Event-driven mode components
        if mode == "event_driven":
            self.micro_detector = MicroChangeDetector(
                facial_sensitivity=facial_sensitivity,
                pose_sensitivity=pose_sensitivity,
                scene_sensitivity=scene_sensitivity,
                similarity_threshold=similarity_threshold
            )
            self.event_classifier = EventClassifier()
            self.intelligent_filter = IntelligentFilter(
                similarity_threshold=similarity_threshold,
                strategy=FilterStrategy(filter_strategy)
            )
        else:
            self.micro_detector = None
            self.event_classifier = None
            self.intelligent_filter = None
        
    def select_keyframes(self, 
                        video_path: Path, 
                        timepoints: List[TimePoint] = None,
                        target_count: int = 10) -> List[TimePoint]:
        """Select best keyframes automatically."""
        if self.mode == "event_driven":
            return self._select_keyframes_event_driven(video_path, target_count)
        else:
            return self._select_keyframes_interval_based(video_path, timepoints, target_count)
    
    def _select_keyframes_event_driven(self, video_path: Path, target_count: int = 10) -> List[TimePoint]:
        """Select keyframes using event-driven micro-change detection."""
        # Detect all micro-changes in the video
        change_events = self.micro_detector.detect_changes_in_video(video_path)
        
        if not change_events:
            # Fallback to interval-based if no changes detected
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
            cap.release()
            
            # Generate minimal timepoints
            from video_shots.utils.time_utils import collect_timepoints
            timepoints = collect_timepoints(0, duration, duration / max(1, target_count), fps, int(duration * fps))
            return timepoints[:target_count]
        
        # Get video metadata
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
        
        # Classify events
        classified_events = self.event_classifier.classify_events(change_events, duration)
        
        # Filter by classification (keep high-priority events)
        filtered_events = filter_by_classification(
            classified_events,
            min_significance=0.3,
            priority_filter=[EventPriority.CRITICAL, EventPriority.HIGH, EventPriority.MEDIUM]
        )
        
        # Extract original events
        filtered_change_events = [event.original_event for event in filtered_events]
        
        # Apply intelligent filtering
        filter_result = self.intelligent_filter.filter_frames(video_path, filtered_change_events, fps)
        
        # If we have too many or too few frames, adjust
        if len(filter_result.selected_timepoints) > target_count:
            # Take top-scored events
            scored_events = [(event, event.change_score * event.confidence) 
                           for event in filtered_change_events]
            scored_events.sort(key=lambda x: x[1], reverse=True)
            selected_events = [event for event, score in scored_events[:target_count]]
            return create_timepoints_from_changes(selected_events, fps)
        elif len(filter_result.selected_timepoints) < target_count // 2:
            # Include more events by lowering thresholds
            filtered_events = filter_by_classification(
                classified_events,
                min_significance=0.2,  # Lower threshold
                priority_filter=[EventPriority.CRITICAL, EventPriority.HIGH, EventPriority.MEDIUM, EventPriority.LOW]
            )
            filtered_change_events = [event.original_event for event in filtered_events]
            filter_result = self.intelligent_filter.filter_frames(video_path, filtered_change_events, fps)
        
        return filter_result.selected_timepoints
    
    def _select_keyframes_interval_based(self, video_path: Path, timepoints: List[TimePoint], target_count: int = 10) -> List[TimePoint]:
        """Select keyframes using traditional interval-based method."""
        frame_scores = self._score_all_frames(video_path, timepoints)
        
        # Apply diversity filtering
        diverse_frames = self._ensure_diversity(frame_scores, target_count)
        
        # Sort by score and return top frames
        selected = sorted(diverse_frames, key=lambda x: x.total_score, reverse=True)
        return [frame.time_point for frame in selected[:target_count]]
    
    def get_event_driven_analysis(self, video_path: Path) -> Dict:
        """Get detailed analysis from event-driven mode for debugging/reporting."""
        if self.mode != "event_driven":
            raise VideoShotsError("Event-driven analysis only available in event_driven mode")
        
        # Get video metadata
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
        
        # Detect changes
        change_events = self.micro_detector.detect_changes_in_video(video_path)
        
        # Classify events
        classified_events = self.event_classifier.classify_events(change_events, duration)
        
        # Filter events
        filtered_events = filter_by_classification(
            classified_events,
            min_significance=0.3,
            priority_filter=[EventPriority.CRITICAL, EventPriority.HIGH, EventPriority.MEDIUM]
        )
        
        # Apply intelligent filtering
        filtered_change_events = [event.original_event for event in filtered_events]
        filter_result = self.intelligent_filter.filter_frames(video_path, filtered_change_events, fps)
        
        return {
            "total_changes_detected": len(change_events),
            "classified_events": len(classified_events),
            "filtered_events": len(filtered_events),
            "final_timepoints": len(filter_result.selected_timepoints),
            "compression_ratio": filter_result.compression_ratio,
            "filtering_stats": filter_result.filtering_stats,
            "change_events": change_events,
            "classified_events": classified_events,
            "filter_result": filter_result
        }
    
    def _score_all_frames(self, video_path: Path, timepoints: List[TimePoint]) -> List[FrameScore]:
        """Score all frames using multiple criteria."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoShotsError(f"Cannot open video: {video_path}")
        
        frame_scores = []
        
        try:
            for tp in timepoints:
                cap.set(cv2.CAP_PROP_POS_FRAMES, tp.frame_index)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                
                score = self._score_single_frame(frame, tp)
                frame_scores.append(score)
                
        finally:
            cap.release()
            
        return frame_scores
    
    def _score_single_frame(self, frame: np.ndarray, tp: TimePoint) -> FrameScore:
        """Score a single frame using all criteria."""
        visual_quality = self._assess_visual_quality(frame)
        content_score = self._assess_content(frame)
        uniqueness = self._assess_uniqueness(frame)
        motion_score = self._assess_motion(frame)
        composition_score = self._assess_composition(frame)
        
        total_score = (
            visual_quality * self.quality_weight +
            content_score * self.content_weight +
            uniqueness * self.uniqueness_weight +
            motion_score * self.motion_weight +
            composition_score * self.composition_weight
        )
        
        reasoning = self._generate_reasoning(
            visual_quality, content_score, uniqueness, motion_score, composition_score
        )
        
        return FrameScore(
            time_point=tp,
            visual_quality=visual_quality,
            content_score=content_score,
            uniqueness=uniqueness,
            motion_score=motion_score,
            composition_score=composition_score,
            total_score=total_score,
            reasoning=reasoning
        )
    
    def _assess_visual_quality(self, frame: np.ndarray) -> float:
        """Assess technical quality of frame (0-10)."""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(10, laplacian_var / 100)
        
        # Contrast (standard deviation)
        contrast_score = min(10, gray.std() / 25.5)
        
        # Brightness balance (avoid over/under exposure)
        mean_brightness = gray.mean()
        brightness_score = 10 - abs(mean_brightness - 127.5) / 12.75
        
        # Noise assessment (using bilateral filter difference)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        noise_level = np.mean(np.abs(gray.astype(float) - filtered.astype(float)))
        noise_score = max(0, 10 - noise_level / 2.55)
        
        return (sharpness_score + contrast_score + brightness_score + noise_score) / 4
    
    def _assess_content(self, frame: np.ndarray) -> float:
        """Assess content richness (0-10)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge density (more edges = more content)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_score = min(10, edge_density * 50)
        
        # Color richness (histogram spread)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        color_diversity = np.count_nonzero(hist) / hist.size
        color_score = min(10, color_diversity * 20)
        
        # Face detection bonus
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        face_score = min(10, len(faces) * 3)
        
        return (edge_score + color_score + face_score) / 3
    
    def _assess_uniqueness(self, frame: np.ndarray) -> float:
        """Assess how unique this frame is compared to previous ones (0-10)."""
        # Calculate histogram
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        if not self.frame_histograms:
            self.frame_histograms.append(hist)
            return 10.0  # First frame is unique by definition
        
        # Compare with previous frames
        max_similarity = 0
        for prev_hist in self.frame_histograms[-10:]:  # Compare with last 10 frames
            similarity = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
            max_similarity = max(max_similarity, similarity)
        
        self.frame_histograms.append(hist)
        uniqueness = (1 - max_similarity) * 10
        return max(0, uniqueness)
    
    def _assess_motion(self, frame: np.ndarray) -> float:
        """Assess motion/action in frame (0-10)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return 5.0  # Neutral score for first frame
        
        # Optical flow to detect motion
        flow = cv2.calcOpticalFlowPyrLK(
            self.previous_frame, gray, 
            np.random.randint(0, gray.shape[1], (100, 1, 2)).astype(np.float32),
            None
        )[0]
        
        if flow is not None:
            motion_magnitude = np.mean(np.linalg.norm(flow, axis=2))
            motion_score = min(10, motion_magnitude / 5)
        else:
            motion_score = 0
        
        self.previous_frame = gray
        return motion_score
    
    def _assess_composition(self, frame: np.ndarray) -> float:
        """Assess composition quality using rule of thirds (0-10)."""
        h, w = frame.shape[:2]
        
        # Rule of thirds grid
        third_h, third_w = h // 3, w // 3
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check interest points on rule of thirds lines
        interest_points = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
        
        if interest_points is None:
            return 5.0
        
        # Score based on how many interest points are near rule of thirds lines
        score = 0
        for point in interest_points:
            x, y = point[0]
            
            # Check proximity to rule of thirds lines
            near_vertical = min(abs(x - third_w), abs(x - 2*third_w)) < w * 0.1
            near_horizontal = min(abs(y - third_h), abs(y - 2*third_h)) < h * 0.1
            
            if near_vertical or near_horizontal:
                score += 1
        
        return min(10, (score / len(interest_points)) * 10)
    
    def _ensure_diversity(self, frame_scores: List[FrameScore], target_count: int) -> List[FrameScore]:
        """Ensure diversity in selected frames using clustering."""
        if len(frame_scores) <= target_count:
            return frame_scores
        
        # Sort by score first
        sorted_frames = sorted(frame_scores, key=lambda x: x.total_score, reverse=True)
        
        # Take top candidates (2x target count)
        candidates = sorted_frames[:target_count * 2]
        
        # Use temporal diversity - ensure frames are spread across time
        time_points = [f.time_point.seconds for f in candidates]
        
        # Simple time-based clustering
        if len(candidates) <= target_count:
            return candidates
        
        # Select frames spread across time while maintaining quality
        selected = []
        time_step = (max(time_points) - min(time_points)) / (target_count - 1)
        
        for i in range(target_count):
            target_time = min(time_points) + i * time_step
            
            # Find candidate closest to target time
            best_candidate = min(candidates, 
                               key=lambda f: abs(f.time_point.seconds - target_time))
            
            if best_candidate not in selected:
                selected.append(best_candidate)
                candidates.remove(best_candidate)
        
        return selected
    
    def _generate_reasoning(self, visual: float, content: float, unique: float, 
                          motion: float, composition: float) -> str:
        """Generate human-readable reasoning for frame selection."""
        reasons = []
        
        if visual >= 8:
            reasons.append("excellent visual quality")
        elif visual >= 6:
            reasons.append("good visual quality")
        
        if content >= 8:
            reasons.append("rich content")
        elif content >= 6:
            reasons.append("interesting content")
        
        if unique >= 8:
            reasons.append("very unique")
        elif unique >= 6:
            reasons.append("distinctive")
        
        if motion >= 7:
            reasons.append("dynamic action")
        elif motion >= 4:
            reasons.append("moderate motion")
        
        if composition >= 7:
            reasons.append("strong composition")
        
        if not reasons:
            return "acceptable quality frame"
        
        return "Selected for: " + ", ".join(reasons)


def save_selection_report(frame_scores: List[FrameScore], output_path: Path) -> None:
    """Save detailed selection report."""
    report = {
        "selection_summary": {
            "total_frames_analyzed": len(frame_scores),
            "average_score": sum(f.total_score for f in frame_scores) / len(frame_scores),
            "score_distribution": {
                "excellent (8-10)": len([f for f in frame_scores if f.total_score >= 8]),
                "good (6-8)": len([f for f in frame_scores if 6 <= f.total_score < 8]),
                "fair (4-6)": len([f for f in frame_scores if 4 <= f.total_score < 6]),
                "poor (0-4)": len([f for f in frame_scores if f.total_score < 4])
            }
        },
        "frames": [
            {
                "timecode": f"{f.time_point.seconds:.3f}s",
                "frame_index": f.time_point.frame_index,
                "scores": {
                    "visual_quality": round(f.visual_quality, 2),
                    "content": round(f.content_score, 2),
                    "uniqueness": round(f.uniqueness, 2),
                    "motion": round(f.motion_score, 2),
                    "composition": round(f.composition_score, 2),
                    "total": round(f.total_score, 2)
                },
                "reasoning": f.reasoning
            }
            for f in sorted(frame_scores, key=lambda x: x.total_score, reverse=True)
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)