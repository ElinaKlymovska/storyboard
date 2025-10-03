"""Micro-change detection for granular keyframe selection."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum

from video_shots.core.models import TimePoint
from video_shots.core.exceptions import VideoShotsError


class ChangeType(Enum):
    """Types of detected changes."""
    FACIAL_EXPRESSION = "facial_expression"
    POSE_CHANGE = "pose_change" 
    SCENE_COMPOSITION = "scene_composition"
    VISUAL_QUALITY = "visual_quality"
    MOTION_CHANGE = "motion_change"


@dataclass
class ChangeEvent:
    """Detected change event."""
    frame_index: int
    timestamp: float
    change_type: ChangeType
    change_score: float  # 0.0 to 1.0
    confidence: float    # 0.0 to 1.0
    description: str
    features: Dict       # Additional feature data


@dataclass
class FrameFeatures:
    """Extracted features from a frame."""
    frame_index: int
    timestamp: float
    histogram: np.ndarray
    edges: np.ndarray
    optical_flow_magnitude: float
    face_landmarks: Optional[np.ndarray]
    face_bbox: Optional[Tuple[int, int, int, int]]
    eye_aspect_ratio: Optional[float]
    mouth_aspect_ratio: Optional[float]
    head_pose: Optional[Tuple[float, float, float]]
    dominant_colors: List[Tuple[int, int, int]]
    brightness: float
    contrast: float
    sharpness: float


class MicroChangeDetector:
    """Detects micro-changes in video frames for granular keyframe selection."""
    
    def __init__(self,
                 facial_sensitivity: float = 0.15,
                 pose_sensitivity: float = 0.25,
                 scene_sensitivity: float = 0.4,
                 similarity_threshold: float = 0.85,
                 max_frames_per_second: int = 30):
        """
        Initialize micro-change detector.
        
        Args:
            facial_sensitivity: Threshold for detecting facial changes (0-1)
            pose_sensitivity: Threshold for detecting pose changes (0-1) 
            scene_sensitivity: Threshold for detecting scene changes (0-1)
            similarity_threshold: Frames more similar than this are considered same event
            max_frames_per_second: Maximum frames to process per second
        """
        self.facial_sensitivity = facial_sensitivity
        self.pose_sensitivity = pose_sensitivity
        self.scene_sensitivity = scene_sensitivity
        self.similarity_threshold = similarity_threshold
        self.max_frames_per_second = max_frames_per_second
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Feature history for comparison
        self.previous_features: Optional[FrameFeatures] = None
        self.feature_history: List[FrameFeatures] = []
        self.max_history_size = 10
        
        # Optical flow tracker
        self.previous_gray = None
        self.lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    def detect_changes_in_video(self, video_path: Path) -> List[ChangeEvent]:
        """
        Detect all micro-changes in a video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of detected change events
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoShotsError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / self.max_frames_per_second))
        
        change_events = []
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only every nth frame based on max_frames_per_second
                if frame_idx % frame_interval == 0:
                    timestamp = frame_idx / fps
                    features = self._extract_frame_features(frame, frame_idx, timestamp)
                    
                    if self.previous_features is not None:
                        events = self._detect_changes(self.previous_features, features)
                        change_events.extend(events)
                    
                    self._update_history(features)
                
                frame_idx += 1
                
        finally:
            cap.release()
        
        return self._filter_significant_changes(change_events)
    
    def _extract_frame_features(self, frame: np.ndarray, frame_index: int, timestamp: float) -> FrameFeatures:
        """Extract comprehensive features from a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Basic image features
        histogram = self._calculate_histogram(frame)
        edges = cv2.Canny(gray, 50, 150)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Optical flow
        optical_flow_magnitude = self._calculate_optical_flow(gray)
        
        # Face detection and analysis
        face_landmarks, face_bbox, eye_ratio, mouth_ratio, head_pose = self._analyze_face(frame, gray)
        
        # Color analysis
        dominant_colors = self._extract_dominant_colors(frame)
        
        return FrameFeatures(
            frame_index=frame_index,
            timestamp=timestamp,
            histogram=histogram,
            edges=edges,
            optical_flow_magnitude=optical_flow_magnitude,
            face_landmarks=face_landmarks,
            face_bbox=face_bbox,
            eye_aspect_ratio=eye_ratio,
            mouth_aspect_ratio=mouth_ratio,
            head_pose=head_pose,
            dominant_colors=dominant_colors,
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness
        )
    
    def _calculate_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Calculate color histogram for frame comparison."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60],
                           [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()
    
    def _calculate_optical_flow(self, gray: np.ndarray) -> float:
        """Calculate optical flow magnitude."""
        if self.previous_gray is None:
            self.previous_gray = gray
            return 0.0
        
        # Use sparse optical flow for efficiency
        corners = cv2.goodFeaturesToTrack(self.previous_gray, 100, 0.01, 10)
        
        if corners is not None:
            # Calculate optical flow
            new_corners, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_gray, gray, corners, None, **self.lk_params)
            
            # Calculate flow magnitude
            if status is not None:
                good_corners = corners[status == 1]
                good_new = new_corners[status == 1]
                
                if len(good_corners) > 0:
                    flow_vectors = good_new - good_corners
                    magnitude = np.mean(np.linalg.norm(flow_vectors, axis=1))
                    self.previous_gray = gray
                    return magnitude
        
        self.previous_gray = gray
        return 0.0
    
    def _analyze_face(self, frame: np.ndarray, gray: np.ndarray) -> Tuple:
        """Analyze facial features and pose."""
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None, None, None, None, None
        
        # Use largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        face_bbox = (x, y, w, h)
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]
        
        # Simple eye aspect ratio calculation
        eye_ratio = self._calculate_eye_aspect_ratio(face_roi)
        
        # Simple mouth aspect ratio calculation  
        mouth_ratio = self._calculate_mouth_aspect_ratio(face_roi)
        
        # Basic head pose estimation (placeholder)
        head_pose = self._estimate_head_pose(face_roi)
        
        # For landmarks, we'll use a simplified approach
        landmarks = self._extract_simple_landmarks(face_roi)
        
        return landmarks, face_bbox, eye_ratio, mouth_ratio, head_pose
    
    def _calculate_eye_aspect_ratio(self, face_roi: np.ndarray) -> float:
        """Calculate eye aspect ratio to detect blinking."""
        if face_roi.size == 0:
            return 0.0
        
        # Simple approach: analyze upper portion of face for eye region
        h, w = face_roi.shape
        eye_region = face_roi[int(h*0.2):int(h*0.5), :]
        
        # Use horizontal edges to estimate eye openness
        edges = cv2.Canny(eye_region, 50, 150)
        horizontal_edges = np.sum(edges, axis=1)
        
        if len(horizontal_edges) > 0:
            # Eye aspect ratio approximation
            return np.std(horizontal_edges) / (np.mean(horizontal_edges) + 1e-6)
        
        return 0.0
    
    def _calculate_mouth_aspect_ratio(self, face_roi: np.ndarray) -> float:
        """Calculate mouth aspect ratio to detect speaking."""
        if face_roi.size == 0:
            return 0.0
        
        # Simple approach: analyze lower portion of face for mouth region
        h, w = face_roi.shape
        mouth_region = face_roi[int(h*0.6):int(h*0.9), :]
        
        # Use edge detection to estimate mouth state
        edges = cv2.Canny(mouth_region, 30, 100)
        vertical_edges = np.sum(edges, axis=0)
        
        if len(vertical_edges) > 0:
            # Mouth aspect ratio approximation
            return np.std(vertical_edges) / (np.mean(vertical_edges) + 1e-6)
        
        return 0.0
    
    def _estimate_head_pose(self, face_roi: np.ndarray) -> Tuple[float, float, float]:
        """Estimate head pose (yaw, pitch, roll)."""
        if face_roi.size == 0:
            return (0.0, 0.0, 0.0)
        
        # Simplified head pose using face asymmetry
        h, w = face_roi.shape
        left_half = face_roi[:, :w//2]
        right_half = face_roi[:, w//2:]
        
        # Calculate asymmetry as rough pose indicator
        left_mean = np.mean(left_half) if left_half.size > 0 else 0
        right_mean = np.mean(right_half) if right_half.size > 0 else 0
        
        yaw = (left_mean - right_mean) / 255.0  # Normalized difference
        
        # Simplified pitch and roll (placeholders)
        pitch = 0.0
        roll = 0.0
        
        return (yaw, pitch, roll)
    
    def _extract_simple_landmarks(self, face_roi: np.ndarray) -> np.ndarray:
        """Extract simplified facial landmarks."""
        if face_roi.size == 0:
            return np.array([])
        
        # Use corner detection as simple landmarks
        corners = cv2.goodFeaturesToTrack(face_roi, 20, 0.01, 10)
        
        if corners is not None:
            return corners.reshape(-1, 2)
        
        return np.array([])
    
    def _extract_dominant_colors(self, frame: np.ndarray, k: int = 3) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from frame."""
        # Reshape frame to list of pixels
        pixels = frame.reshape(-1, 3)
        
        # Use k-means clustering to find dominant colors
        from sklearn.cluster import KMeans
        
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int)
            return [tuple(color) for color in colors]
        except:
            # Fallback to simple color analysis
            return [(int(np.mean(frame[:,:,0])), 
                    int(np.mean(frame[:,:,1])), 
                    int(np.mean(frame[:,:,2])))]
    
    def _detect_changes(self, prev_features: FrameFeatures, curr_features: FrameFeatures) -> List[ChangeEvent]:
        """Detect changes between two frames."""
        changes = []
        
        # Facial expression changes
        if prev_features.eye_aspect_ratio is not None and curr_features.eye_aspect_ratio is not None:
            eye_change = abs(prev_features.eye_aspect_ratio - curr_features.eye_aspect_ratio)
            if eye_change > self.facial_sensitivity:
                changes.append(ChangeEvent(
                    frame_index=curr_features.frame_index,
                    timestamp=curr_features.timestamp,
                    change_type=ChangeType.FACIAL_EXPRESSION,
                    change_score=min(1.0, eye_change / self.facial_sensitivity),
                    confidence=0.8,
                    description=f"Eye state change (ratio diff: {eye_change:.3f})",
                    features={'eye_ratio_diff': eye_change}
                ))
        
        if prev_features.mouth_aspect_ratio is not None and curr_features.mouth_aspect_ratio is not None:
            mouth_change = abs(prev_features.mouth_aspect_ratio - curr_features.mouth_aspect_ratio)
            if mouth_change > self.facial_sensitivity:
                changes.append(ChangeEvent(
                    frame_index=curr_features.frame_index,
                    timestamp=curr_features.timestamp,
                    change_type=ChangeType.FACIAL_EXPRESSION,
                    change_score=min(1.0, mouth_change / self.facial_sensitivity),
                    confidence=0.7,
                    description=f"Mouth state change (ratio diff: {mouth_change:.3f})",
                    features={'mouth_ratio_diff': mouth_change}
                ))
        
        # Head pose changes
        if prev_features.head_pose is not None and curr_features.head_pose is not None:
            pose_diff = abs(prev_features.head_pose[0] - curr_features.head_pose[0])  # Yaw difference
            if pose_diff > self.pose_sensitivity:
                changes.append(ChangeEvent(
                    frame_index=curr_features.frame_index,
                    timestamp=curr_features.timestamp,
                    change_type=ChangeType.POSE_CHANGE,
                    change_score=min(1.0, pose_diff / self.pose_sensitivity),
                    confidence=0.6,
                    description=f"Head pose change (yaw diff: {pose_diff:.3f})",
                    features={'head_pose_diff': pose_diff}
                ))
        
        # Scene composition changes (histogram comparison)
        hist_diff = 1 - cv2.compareHist(prev_features.histogram, curr_features.histogram, cv2.HISTCMP_CORREL)
        if hist_diff > self.scene_sensitivity:
            changes.append(ChangeEvent(
                frame_index=curr_features.frame_index,
                timestamp=curr_features.timestamp,
                change_type=ChangeType.SCENE_COMPOSITION,
                change_score=min(1.0, hist_diff / self.scene_sensitivity),
                confidence=0.9,
                description=f"Scene composition change (hist diff: {hist_diff:.3f})",
                features={'histogram_diff': hist_diff}
            ))
        
        # Motion changes
        motion_diff = abs(prev_features.optical_flow_magnitude - curr_features.optical_flow_magnitude)
        motion_threshold = 5.0  # Adjust based on testing
        if motion_diff > motion_threshold:
            changes.append(ChangeEvent(
                frame_index=curr_features.frame_index,
                timestamp=curr_features.timestamp,
                change_type=ChangeType.MOTION_CHANGE,
                change_score=min(1.0, motion_diff / motion_threshold),
                confidence=0.8,
                description=f"Motion change (flow diff: {motion_diff:.3f})",
                features={'motion_diff': motion_diff}
            ))
        
        # Visual quality changes
        brightness_diff = abs(prev_features.brightness - curr_features.brightness)
        if brightness_diff > 30:  # Significant brightness change
            changes.append(ChangeEvent(
                frame_index=curr_features.frame_index,
                timestamp=curr_features.timestamp,
                change_type=ChangeType.VISUAL_QUALITY,
                change_score=min(1.0, brightness_diff / 100),
                confidence=0.7,
                description=f"Brightness change (diff: {brightness_diff:.1f})",
                features={'brightness_diff': brightness_diff}
            ))
        
        return changes
    
    def _update_history(self, features: FrameFeatures) -> None:
        """Update feature history."""
        self.previous_features = features
        self.feature_history.append(features)
        
        # Keep only recent history
        if len(self.feature_history) > self.max_history_size:
            self.feature_history.pop(0)
    
    def _filter_significant_changes(self, change_events: List[ChangeEvent]) -> List[ChangeEvent]:
        """Filter out insignificant or redundant changes."""
        if not change_events:
            return []
        
        # Sort by timestamp
        events = sorted(change_events, key=lambda x: x.timestamp)
        
        # Remove events too close in time (within 0.1 seconds)
        filtered = [events[0]]
        min_time_gap = 0.1
        
        for event in events[1:]:
            if event.timestamp - filtered[-1].timestamp > min_time_gap:
                filtered.append(event)
            elif event.change_score > filtered[-1].change_score:
                # Replace with higher scoring event
                filtered[-1] = event
        
        # Keep only events above minimum confidence
        min_confidence = 0.5
        filtered = [e for e in filtered if e.confidence >= min_confidence]
        
        return filtered


def create_timepoints_from_changes(change_events: List[ChangeEvent], fps: float) -> List[TimePoint]:
    """Convert change events to TimePoint objects."""
    timepoints = []
    
    for i, event in enumerate(change_events):
        timepoints.append(TimePoint(
            index=i + 1,
            seconds=event.timestamp,
            frame_index=event.frame_index
        ))
    
    return timepoints