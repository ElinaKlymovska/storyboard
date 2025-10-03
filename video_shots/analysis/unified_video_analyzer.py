"""Unified video analyzer with improved error handling and logging."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import cv2
from PIL import Image

from video_shots.config.analysis_config import VideoConfig, EventDetectionConfig, get_config
from video_shots.logging.logger import get_logger
from video_shots.core.models import TimePoint
from video_shots.utils.time_utils import collect_timepoints

try:
    from video_shots.core.keyframe_selector import KeyframeSelector
    KEYFRAME_SELECTOR_AVAILABLE = True
except ImportError:
    KEYFRAME_SELECTOR_AVAILABLE = False


@dataclass 
class KeyframeData:
    """Represents a keyframe with analysis data."""
    index: int
    timestamp: float
    frame_index: int
    timecode: str
    local_path: str
    scene_id: str
    shot_id: str
    analysis_note: str


@dataclass
class ShotData:
    """Represents a video shot with analysis data."""
    shot_id: str
    scene_id: str
    start_time: float
    end_time: float
    duration: float
    start_timecode: str
    end_timecode: str
    shot_type: str
    analysis_note: str


@dataclass
class VideoMetadata:
    """Video metadata container."""
    path: str
    duration: float
    fps: float
    frame_count: int
    resolution: Tuple[int, int]
    aspect_ratio: str
    format: str


@dataclass
class VideoAnalysisResult:
    """Complete video analysis result."""
    success: bool
    metadata: VideoMetadata
    keyframes: List[KeyframeData]
    shots: List[ShotData]
    output_dir: str
    error_message: Optional[str] = None
    
    @property
    def total_keyframes(self) -> int:
        """Get total number of keyframes."""
        return len(self.keyframes)
    
    @property
    def total_shots(self) -> int:
        """Get total number of shots."""
        return len(self.shots)
    
    @property
    def keyframes_per_minute(self) -> float:
        """Calculate keyframes per minute."""
        if self.metadata.duration > 0:
            return (len(self.keyframes) / self.metadata.duration) * 60
        return 0.0


class UnifiedVideoAnalyzer:
    """Unified video analyzer with improved error handling."""
    
    def __init__(
        self, 
        video_config: Optional[VideoConfig] = None,
        event_config: Optional[EventDetectionConfig] = None
    ):
        """Initialize analyzer with configuration."""
        config = get_config()
        self.video_config = video_config or config.video
        self.event_config = event_config or config.event_detection
        self.logger = get_logger()
        
        if not KEYFRAME_SELECTOR_AVAILABLE:
            self.logger.warning("KeyframeSelector not available - using interval-based extraction only")
    
    def analyze_video(
        self, 
        video_path: Union[str, Path], 
        output_dir: Optional[Union[str, Path]] = None
    ) -> VideoAnalysisResult:
        """Perform complete video analysis."""
        start_time = time.time()
        video_path = Path(video_path)
        
        self.logger.log_operation_start(
            "video analysis",
            video=video_path.name,
            mode=self.video_config.keyframe_selection_mode
        )
        
        if not video_path.exists():
            error_msg = f"Video file not found: {video_path}"
            self.logger.error(error_msg)
            return self._create_error_result(str(video_path), error_msg, output_dir)
        
        try:
            # Set up output directory
            if output_dir is None:
                config = get_config()
                output_dir = config.get_output_dir(video_path.stem)
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            keyframes_dir = output_dir / "keyframes"
            keyframes_dir.mkdir(exist_ok=True)
            
            # Extract video metadata
            metadata = self._extract_metadata(video_path)
            
            # Extract keyframes
            keyframes = self._extract_keyframes(video_path, metadata, keyframes_dir)
            
            # Analyze shots
            shots = self._analyze_shots(metadata)
            
            duration_seconds = time.time() - start_time
            self.logger.log_operation_complete(
                "video analysis",
                duration=duration_seconds,
                keyframes=len(keyframes),
                shots=len(shots)
            )
            
            return VideoAnalysisResult(
                success=True,
                metadata=metadata,
                keyframes=keyframes,
                shots=shots,
                output_dir=str(output_dir)
            )
            
        except Exception as e:
            self.logger.log_operation_error("video analysis", e)
            return self._create_error_result(str(video_path), str(e), output_dir)
    
    def _extract_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract video metadata."""
        self.logger.debug(f"Extracting metadata from {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fps <= 0 or frame_count <= 0:
                raise RuntimeError("Unable to determine FPS or frame count")
            
            # Determine aspect ratio
            aspect_ratio = self._calculate_aspect_ratio(width, height)
            
            metadata = VideoMetadata(
                path=str(video_path),
                duration=duration,
                fps=fps,
                frame_count=frame_count,
                resolution=(width, height),
                aspect_ratio=aspect_ratio,
                format=video_path.suffix.lower()
            )
            
            self.logger.debug(f"Video metadata: {duration:.1f}s, {fps:.1f}fps, {width}x{height}")
            return metadata
            
        finally:
            cap.release()
    
    def _calculate_aspect_ratio(self, width: int, height: int) -> str:
        """Calculate and format aspect ratio."""
        def gcd(a: int, b: int) -> int:
            while b:
                a, b = b, a % b
            return a
        
        common_divisor = gcd(width, height)
        ratio_w = width // common_divisor
        ratio_h = height // common_divisor
        
        # Check for common ratios
        if ratio_w == 16 and ratio_h == 9:
            return "16:9 (widescreen)"
        elif ratio_w == 9 and ratio_h == 16:
            return "9:16 (vertical/mobile)"
        elif ratio_w == 4 and ratio_h == 3:
            return "4:3 (standard)"
        elif ratio_w == 1 and ratio_h == 1:
            return "1:1 (square)"
        else:
            return f"{ratio_w}:{ratio_h}"
    
    def _extract_keyframes(
        self, 
        video_path: Path, 
        metadata: VideoMetadata, 
        output_dir: Path
    ) -> List[KeyframeData]:
        """Extract keyframes from video."""
        self.logger.debug("Extracting keyframes")
        
        # Calculate timepoints
        timepoints = self._calculate_timepoints(video_path, metadata)
        
        keyframes = []
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video for keyframe extraction: {video_path}")
        
        try:
            total_frames = len(timepoints)
            for i, tp in enumerate(timepoints):
                cap.set(cv2.CAP_PROP_POS_FRAMES, tp.frame_index)
                ok, frame = cap.read()
                
                if not ok or frame is None:
                    self.logger.warning(f"Failed to read frame at index {tp.frame_index}")
                    continue
                
                # Save frame
                frame_filename = f"keyframe_{i:03d}_{tp.seconds:.3f}s.png"
                frame_path = output_dir / frame_filename
                
                cv2.imwrite(str(frame_path), frame)
                
                # Create keyframe data
                keyframe = KeyframeData(
                    index=i,
                    timestamp=tp.seconds,
                    frame_index=tp.frame_index,
                    timecode=self._format_timecode(tp.seconds),
                    local_path=str(frame_path),
                    scene_id=f"scene_{(i // 5) + 1:02d}",
                    shot_id=f"shot_{(i // 3) + 1:02d}",
                    analysis_note="Keyframe extracted successfully"
                )
                keyframes.append(keyframe)
                
                # Log progress
                if total_frames > 5:
                    self.logger.log_progress(i + 1, total_frames, "keyframe extraction")
                    
        finally:
            cap.release()
        
        self.logger.debug(f"Extracted {len(keyframes)} keyframes")
        return keyframes
    
    def _calculate_timepoints(self, video_path: Path, metadata: VideoMetadata) -> List[TimePoint]:
        """Calculate timepoints for keyframe extraction."""
        if (self.video_config.keyframe_selection_mode == "event_driven" and 
            KEYFRAME_SELECTOR_AVAILABLE):
            return self._calculate_event_driven_timepoints(video_path, metadata)
        else:
            return self._calculate_interval_timepoints(metadata)
    
    def _calculate_event_driven_timepoints(
        self, 
        video_path: Path, 
        metadata: VideoMetadata
    ) -> List[TimePoint]:
        """Calculate timepoints using event-driven selection."""
        try:
            self.logger.debug("Using event-driven keyframe selection")
            
            keyframe_selector = KeyframeSelector(
                mode="event_driven",
                facial_sensitivity=self.event_config.change_sensitivity.get('facial', 0.15),
                pose_sensitivity=self.event_config.change_sensitivity.get('pose', 0.25),
                scene_sensitivity=self.event_config.change_sensitivity.get('scene', 0.4),
                similarity_threshold=self.event_config.similarity_threshold,
                filter_strategy=self.event_config.filter_strategy
            )
            
            # Calculate target count
            target_keyframes = min(
                max(10, int(metadata.duration * self.video_config.target_keyframes_per_minute / 60)),
                self.video_config.max_keyframes
            )
            
            timepoints = keyframe_selector.select_keyframes(
                video_path=video_path,
                target_count=target_keyframes
            )
            
            self.logger.debug(f"Event-driven selection: {len(timepoints)} timepoints")
            return timepoints
            
        except Exception as e:
            self.logger.warning(f"Event-driven selection failed: {e}, falling back to interval-based")
            return self._calculate_interval_timepoints(metadata)
    
    def _calculate_interval_timepoints(self, metadata: VideoMetadata) -> List[TimePoint]:
        """Calculate timepoints using interval-based selection."""
        self.logger.debug("Using interval-based keyframe selection")
        
        timepoints = collect_timepoints(
            start=0.0,
            end=metadata.duration,
            step=self.video_config.keyframe_interval,
            fps=metadata.fps,
            frame_count=metadata.frame_count
        )
        
        # Limit to max keyframes
        if len(timepoints) > self.video_config.max_keyframes:
            step = len(timepoints) // self.video_config.max_keyframes
            timepoints = timepoints[::step][:self.video_config.max_keyframes]
        
        self.logger.debug(f"Interval-based selection: {len(timepoints)} timepoints")
        return timepoints
    
    def _analyze_shots(self, metadata: VideoMetadata) -> List[ShotData]:
        """Analyze video shots."""
        self.logger.debug("Analyzing video shots")
        
        shots = []
        shot_duration = 5.0  # 5-second shots
        current_time = 0.0
        shot_index = 0
        
        while current_time < metadata.duration:
            end_time = min(current_time + shot_duration, metadata.duration)
            shot_index += 1
            
            shot = ShotData(
                shot_id=f"shot_{shot_index:02d}",
                scene_id=f"scene_{((shot_index - 1) // 3) + 1:02d}",
                start_time=current_time,
                end_time=end_time,
                duration=end_time - current_time,
                start_timecode=self._format_timecode(current_time),
                end_timecode=self._format_timecode(end_time),
                shot_type=self._estimate_shot_type(shot_index),
                analysis_note=f"Shot analysis for {shot_duration}s segment"
            )
            
            shots.append(shot)
            current_time += shot_duration
        
        self.logger.debug(f"Analyzed {len(shots)} shots")
        return shots
    
    def _estimate_shot_type(self, shot_index: int) -> str:
        """Estimate shot type based on position."""
        shot_types = ["WS", "MS", "CU", "MS", "WS"]  # Wide, Medium, Close-up pattern
        return shot_types[(shot_index - 1) % len(shot_types)]
    
    def _format_timecode(self, seconds: float) -> str:
        """Format seconds to timecode."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
    
    def _create_error_result(
        self, 
        video_path: str, 
        error_message: str, 
        output_dir: Optional[Union[str, Path]]
    ) -> VideoAnalysisResult:
        """Create error result."""
        return VideoAnalysisResult(
            success=False,
            metadata=VideoMetadata(
                path=video_path,
                duration=0.0,
                fps=0.0,
                frame_count=0,
                resolution=(0, 0),
                aspect_ratio="unknown",
                format="unknown"
            ),
            keyframes=[],
            shots=[],
            output_dir=str(output_dir) if output_dir else "",
            error_message=error_message
        )
    
    def get_keyframe_at_time(
        self, 
        result: VideoAnalysisResult, 
        target_time: float
    ) -> Optional[KeyframeData]:
        """Find keyframe closest to target time."""
        if not result.keyframes:
            return None
        
        closest_keyframe = min(
            result.keyframes,
            key=lambda kf: abs(kf.timestamp - target_time)
        )
        
        return closest_keyframe
    
    def get_shot_at_time(
        self, 
        result: VideoAnalysisResult, 
        target_time: float
    ) -> Optional[ShotData]:
        """Find shot containing target time."""
        for shot in result.shots:
            if shot.start_time <= target_time <= shot.end_time:
                return shot
        return None