"""Intelligent frame filtering for event-driven keyframe selection."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import math

from video_shots.core.models import TimePoint
from video_shots.core.micro_change_detector import ChangeEvent, ChangeType


class FilterStrategy(Enum):
    """Frame filtering strategies."""
    KEEP_FIRST_DISCARD_REST = "keep_first_discard_rest"
    KEEP_BEST_IN_SEQUENCE = "keep_best_in_sequence" 
    ADAPTIVE_SAMPLING = "adaptive_sampling"
    QUALITY_BASED = "quality_based"


@dataclass
class FrameSequence:
    """Sequence of similar frames."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    representative_frame: int  # The frame chosen to represent this sequence
    similarity_score: float
    sequence_type: str  # "static", "gradual_change", "micro_movement"
    frame_count: int
    quality_scores: List[float]


@dataclass
class FilteredResult:
    """Result of intelligent filtering."""
    selected_timepoints: List[TimePoint]
    discarded_sequences: List[FrameSequence]
    filtering_stats: Dict
    compression_ratio: float


class IntelligentFilter:
    """
    Intelligent frame filtering that keeps first frame of similar sequences
    and discards the rest until the next significant keyframe.
    """
    
    def __init__(self,
                 similarity_threshold: float = 0.85,
                 min_sequence_length: int = 3,
                 max_sequence_duration: float = 2.0,
                 strategy: FilterStrategy = FilterStrategy.KEEP_FIRST_DISCARD_REST,
                 quality_weight: float = 0.3,
                 temporal_weight: float = 0.7):
        """
        Initialize intelligent filter.
        
        Args:
            similarity_threshold: Frames more similar than this are considered same sequence
            min_sequence_length: Minimum frames needed to form a sequence
            max_sequence_duration: Maximum duration for a single sequence (seconds)
            strategy: Filtering strategy to use
            quality_weight: Weight for quality in selection (0-1)
            temporal_weight: Weight for temporal distribution (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.min_sequence_length = min_sequence_length
        self.max_sequence_duration = max_sequence_duration
        self.strategy = strategy
        self.quality_weight = quality_weight
        self.temporal_weight = temporal_weight
        
        # For quality assessment
        self.quality_analyzer = FrameQualityAnalyzer()
    
    def filter_frames(self, 
                     video_path: Path,
                     change_events: List[ChangeEvent],
                     fps: float) -> FilteredResult:
        """
        Apply intelligent filtering to change events.
        
        Args:
            video_path: Path to video file
            change_events: Detected change events
            fps: Video frame rate
            
        Returns:
            FilteredResult with selected frames and statistics
        """
        if not change_events:
            return FilteredResult([], [], {}, 1.0)
        
        # Sort events by timestamp
        sorted_events = sorted(change_events, key=lambda x: x.timestamp)
        
        # Identify similar frame sequences
        sequences = self._identify_sequences(video_path, sorted_events, fps)
        
        # Apply filtering strategy
        if self.strategy == FilterStrategy.KEEP_FIRST_DISCARD_REST:
            selected_events = self._filter_keep_first(sorted_events, sequences)
        elif self.strategy == FilterStrategy.KEEP_BEST_IN_SEQUENCE:
            selected_events = self._filter_keep_best(sorted_events, sequences)
        elif self.strategy == FilterStrategy.ADAPTIVE_SAMPLING:
            selected_events = self._filter_adaptive(sorted_events, sequences)
        else:  # QUALITY_BASED
            selected_events = self._filter_quality_based(video_path, sorted_events, sequences, fps)
        
        # Convert to timepoints
        selected_timepoints = self._events_to_timepoints(selected_events)
        
        # Calculate statistics
        original_count = len(change_events)
        selected_count = len(selected_events)
        compression_ratio = selected_count / original_count if original_count > 0 else 1.0
        
        stats = {
            "original_events": original_count,
            "selected_events": selected_count,
            "discarded_events": original_count - selected_count,
            "sequences_identified": len(sequences),
            "compression_ratio": compression_ratio,
            "strategy_used": self.strategy.value
        }
        
        return FilteredResult(
            selected_timepoints=selected_timepoints,
            discarded_sequences=sequences,
            filtering_stats=stats,
            compression_ratio=compression_ratio
        )
    
    def _identify_sequences(self, 
                           video_path: Path,
                           events: List[ChangeEvent], 
                           fps: float) -> List[FrameSequence]:
        """Identify sequences of similar frames between significant changes."""
        sequences = []
        
        if len(events) < 2:
            return sequences
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return sequences
        
        try:
            current_sequence_start = 0
            
            for i in range(1, len(events)):
                prev_event = events[i-1]
                curr_event = events[i]
                
                # Check if events are far enough apart to analyze sequence
                time_gap = curr_event.timestamp - prev_event.timestamp
                
                if time_gap > 0.1:  # At least 100ms gap
                    # Analyze frames between these events
                    sequence = self._analyze_frame_sequence(
                        cap, prev_event, curr_event, fps
                    )
                    
                    if sequence and sequence.frame_count >= self.min_sequence_length:
                        sequences.append(sequence)
        
        finally:
            cap.release()
        
        return sequences
    
    def _analyze_frame_sequence(self,
                               cap: cv2.VideoCapture,
                               start_event: ChangeEvent,
                               end_event: ChangeEvent,
                               fps: float) -> Optional[FrameSequence]:
        """Analyze a sequence of frames between two events."""
        start_frame = start_event.frame_index
        end_frame = end_event.frame_index
        frame_count = end_frame - start_frame
        
        if frame_count < self.min_sequence_length:
            return None
        
        duration = (end_event.timestamp - start_event.timestamp)
        if duration > self.max_sequence_duration:
            return None
        
        # Sample frames from the sequence to assess similarity
        sample_frames = self._sample_frames_from_sequence(
            cap, start_frame, end_frame, max_samples=5
        )
        
        if len(sample_frames) < 2:
            return None
        
        # Calculate similarity between frames
        similarity_score = self._calculate_sequence_similarity(sample_frames)
        
        # Determine sequence type
        sequence_type = self._classify_sequence_type(sample_frames, start_event, end_event)
        
        # Calculate quality scores for frames
        quality_scores = [self.quality_analyzer.assess_frame_quality(frame) for frame in sample_frames]
        
        # Choose representative frame (first frame by default)
        representative_frame = start_frame
        
        return FrameSequence(
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=start_event.timestamp,
            end_time=end_event.timestamp,
            representative_frame=representative_frame,
            similarity_score=similarity_score,
            sequence_type=sequence_type,
            frame_count=frame_count,
            quality_scores=quality_scores
        )
    
    def _sample_frames_from_sequence(self,
                                   cap: cv2.VideoCapture,
                                   start_frame: int,
                                   end_frame: int,
                                   max_samples: int = 5) -> List[np.ndarray]:
        """Sample frames from a sequence for analysis."""
        frames = []
        frame_count = end_frame - start_frame
        
        if frame_count <= max_samples:
            # Sample all frames
            sample_indices = range(start_frame, end_frame)
        else:
            # Sample evenly distributed frames
            step = frame_count / max_samples
            sample_indices = [start_frame + int(i * step) for i in range(max_samples)]
        
        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
        
        return frames
    
    def _calculate_sequence_similarity(self, frames: List[np.ndarray]) -> float:
        """Calculate average similarity score for a sequence of frames."""
        if len(frames) < 2:
            return 1.0
        
        similarities = []
        
        for i in range(len(frames) - 1):
            similarity = self._calculate_frame_similarity(frames[i], frames[i + 1])
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 1.0
    
    def _calculate_frame_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate similarity between two frames using multiple metrics."""
        # Convert to same size if needed
        if frame1.shape != frame2.shape:
            h, w = min(frame1.shape[0], frame2.shape[0]), min(frame1.shape[1], frame2.shape[1])
            frame1 = cv2.resize(frame1, (w, h))
            frame2 = cv2.resize(frame2, (w, h))
        
        # Calculate histogram similarity
        hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Calculate structural similarity (simplified)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Mean squared error
        mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
        structural_similarity = 1.0 / (1.0 + mse / 1000.0)  # Normalize MSE
        
        # Combine similarities
        combined_similarity = (hist_similarity + structural_similarity) / 2.0
        return max(0.0, min(1.0, combined_similarity))
    
    def _classify_sequence_type(self,
                               frames: List[np.ndarray],
                               start_event: ChangeEvent,
                               end_event: ChangeEvent) -> str:
        """Classify the type of sequence based on frame analysis."""
        if len(frames) < 2:
            return "static"
        
        # Calculate motion between frames
        motion_scores = []
        for i in range(len(frames) - 1):
            motion = self._calculate_motion_between_frames(frames[i], frames[i + 1])
            motion_scores.append(motion)
        
        avg_motion = np.mean(motion_scores) if motion_scores else 0.0
        
        # Classify based on motion and change types
        if avg_motion < 5.0:
            return "static"
        elif avg_motion < 15.0:
            return "micro_movement"
        elif start_event.change_type == ChangeType.FACIAL_EXPRESSION or end_event.change_type == ChangeType.FACIAL_EXPRESSION:
            return "expression_change"
        else:
            return "gradual_change"
    
    def _calculate_motion_between_frames(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate motion magnitude between two frames."""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowPyrLK(
            gray1, gray2,
            cv2.goodFeaturesToTrack(gray1, 100, 0.01, 10),
            None
        )[0]
        
        if flow is not None and len(flow) > 0:
            motion_magnitude = np.mean(np.linalg.norm(flow, axis=2))
            return motion_magnitude
        
        return 0.0
    
    def _filter_keep_first(self,
                          events: List[ChangeEvent],
                          sequences: List[FrameSequence]) -> List[ChangeEvent]:
        """Filter strategy: Keep first frame, discard rest until next keyframe."""
        selected_events = []
        sequence_map = self._create_sequence_map(sequences)
        
        i = 0
        while i < len(events):
            current_event = events[i]
            selected_events.append(current_event)
            
            # Check if this event starts a similar sequence
            if current_event.frame_index in sequence_map:
                sequence = sequence_map[current_event.frame_index]
                # Skip all events in this sequence except the first
                while i < len(events) and events[i].frame_index <= sequence.end_frame:
                    i += 1
            else:
                i += 1
        
        return selected_events
    
    def _filter_keep_best(self,
                         events: List[ChangeEvent],
                         sequences: List[FrameSequence]) -> List[ChangeEvent]:
        """Filter strategy: Keep best quality frame from each sequence."""
        selected_events = []
        sequence_map = self._create_sequence_map(sequences)
        processed_sequences: Set[int] = set()
        
        for event in events:
            if event.frame_index in sequence_map:
                sequence = sequence_map[event.frame_index]
                
                if sequence.start_frame not in processed_sequences:
                    # Find best event in this sequence
                    sequence_events = [e for e in events 
                                     if sequence.start_frame <= e.frame_index <= sequence.end_frame]
                    
                    if sequence_events:
                        # Select event with highest combined score
                        best_event = max(sequence_events, 
                                       key=lambda e: e.change_score * e.confidence)
                        selected_events.append(best_event)
                    
                    processed_sequences.add(sequence.start_frame)
            else:
                # Event not in any sequence, keep it
                selected_events.append(event)
        
        return selected_events
    
    def _filter_adaptive(self,
                        events: List[ChangeEvent],
                        sequences: List[FrameSequence]) -> List[ChangeEvent]:
        """Filter strategy: Adaptive sampling based on sequence characteristics."""
        selected_events = []
        sequence_map = self._create_sequence_map(sequences)
        processed_sequences: Set[int] = set()
        
        for event in events:
            if event.frame_index in sequence_map:
                sequence = sequence_map[event.frame_index]
                
                if sequence.start_frame not in processed_sequences:
                    # Adaptive sampling based on sequence type
                    if sequence.sequence_type == "static":
                        # Keep only first frame for static sequences
                        selected_events.append(event)
                    elif sequence.sequence_type == "micro_movement":
                        # Keep first and last frame
                        sequence_events = [e for e in events 
                                         if sequence.start_frame <= e.frame_index <= sequence.end_frame]
                        if len(sequence_events) >= 2:
                            selected_events.extend([sequence_events[0], sequence_events[-1]])
                        else:
                            selected_events.extend(sequence_events)
                    else:
                        # Keep evenly distributed frames
                        sequence_events = [e for e in events 
                                         if sequence.start_frame <= e.frame_index <= sequence.end_frame]
                        sample_count = min(3, len(sequence_events))
                        step = len(sequence_events) // sample_count if sample_count > 0 else 1
                        selected_events.extend([sequence_events[i * step] 
                                              for i in range(sample_count)])
                    
                    processed_sequences.add(sequence.start_frame)
            else:
                selected_events.append(event)
        
        return selected_events
    
    def _filter_quality_based(self,
                             video_path: Path,
                             events: List[ChangeEvent],
                             sequences: List[FrameSequence],
                             fps: float) -> List[ChangeEvent]:
        """Filter strategy: Quality-based selection with temporal distribution."""
        # This would require loading frames and assessing quality
        # For now, fall back to keep_first strategy
        return self._filter_keep_first(events, sequences)
    
    def _create_sequence_map(self, sequences: List[FrameSequence]) -> Dict[int, FrameSequence]:
        """Create mapping from frame index to sequence."""
        sequence_map = {}
        
        for sequence in sequences:
            for frame_idx in range(sequence.start_frame, sequence.end_frame + 1):
                sequence_map[frame_idx] = sequence
        
        return sequence_map
    
    def _events_to_timepoints(self, events: List[ChangeEvent]) -> List[TimePoint]:
        """Convert change events to TimePoint objects."""
        timepoints = []
        
        for i, event in enumerate(events):
            timepoints.append(TimePoint(
                index=i + 1,
                seconds=event.timestamp,
                frame_index=event.frame_index
            ))
        
        return timepoints


class FrameQualityAnalyzer:
    """Analyzes frame quality for selection purposes."""
    
    def assess_frame_quality(self, frame: np.ndarray) -> float:
        """
        Assess the quality of a frame for keyframe selection.
        
        Args:
            frame: Input frame
            
        Returns:
            Quality score (0.0 to 10.0)
        """
        if frame is None or frame.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(10, laplacian_var / 100)
        
        # Contrast (standard deviation)
        contrast_score = min(10, gray.std() / 25.5)
        
        # Brightness balance
        mean_brightness = gray.mean()
        brightness_score = 10 - abs(mean_brightness - 127.5) / 12.75
        
        # Noise level (using bilateral filter difference)
        try:
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            noise_level = np.mean(np.abs(gray.astype(float) - filtered.astype(float)))
            noise_score = max(0, 10 - noise_level / 2.55)
        except:
            noise_score = 5.0  # Default if filtering fails
        
        # Combine scores
        total_score = (sharpness_score + contrast_score + brightness_score + noise_score) / 4
        return max(0.0, min(10.0, total_score))


def generate_filtering_report(result: FilteredResult, output_path: Path) -> None:
    """Generate a detailed filtering report."""
    import json
    
    report = {
        "filtering_summary": {
            "strategy_used": result.filtering_stats.get("strategy_used", "unknown"),
            "compression_ratio": result.compression_ratio,
            "original_events": result.filtering_stats.get("original_events", 0),
            "selected_events": result.filtering_stats.get("selected_events", 0),
            "discarded_events": result.filtering_stats.get("discarded_events", 0),
            "sequences_identified": result.filtering_stats.get("sequences_identified", 0)
        },
        "selected_timepoints": [
            {
                "index": tp.index,
                "timestamp": f"{tp.seconds:.3f}s",
                "frame_index": tp.frame_index
            }
            for tp in result.selected_timepoints
        ],
        "discarded_sequences": [
            {
                "start_frame": seq.start_frame,
                "end_frame": seq.end_frame,
                "duration": f"{seq.end_time - seq.start_time:.3f}s",
                "similarity_score": round(seq.similarity_score, 3),
                "sequence_type": seq.sequence_type,
                "frame_count": seq.frame_count,
                "representative_frame": seq.representative_frame
            }
            for seq in result.discarded_sequences
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)