"""Unified audio analyzer with improved error handling and logging."""

import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

try:
    import moviepy.editor as mp
    import numpy as np
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

from video_shots.config.analysis_config import AudioConfig, get_config
from video_shots.logging.logger import get_logger


@dataclass
class AudioSegment:
    """Represents an audio segment with analysis data."""
    start_time: float
    end_time: float
    duration: float
    rms_level: float
    db_level: float
    is_silence: bool
    
    @property
    def loudness_category(self) -> str:
        """Categorize loudness level."""
        if self.db_level > -15:
            return "loud"
        elif self.db_level > -25:
            return "normal" 
        elif self.db_level > -40:
            return "quiet"
        else:
            return "silence"


@dataclass
class TranscriptSegment:
    """Represents a transcript segment."""
    start_time: float
    end_time: float
    text: str
    confidence: float
    language: str


@dataclass
class AudioAnalysisResult:
    """Complete audio analysis result."""
    success: bool
    video_path: str
    audio_path: Optional[str]
    transcript_path: Optional[str]
    duration: float
    sample_rate: Optional[float]
    audio_segments: List[AudioSegment]
    transcript_segments: List[TranscriptSegment]
    error_message: Optional[str] = None
    
    @property
    def average_db_level(self) -> float:
        """Calculate average dB level."""
        if not self.audio_segments:
            return -60.0
        return sum(seg.db_level for seg in self.audio_segments) / len(self.audio_segments)
    
    @property
    def dynamic_range(self) -> float:
        """Calculate dynamic range."""
        if not self.audio_segments:
            return 0.0
        db_levels = [seg.db_level for seg in self.audio_segments]
        return max(db_levels) - min(db_levels)
    
    @property
    def silence_percentage(self) -> float:
        """Calculate percentage of silence."""
        if not self.audio_segments:
            return 100.0
        silence_count = sum(1 for seg in self.audio_segments if seg.is_silence)
        return (silence_count / len(self.audio_segments)) * 100
    
    @property
    def total_speech_segments(self) -> int:
        """Get total number of speech segments."""
        return len(self.transcript_segments)


class UnifiedAudioAnalyzer:
    """Unified audio analyzer with improved error handling."""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize analyzer with configuration."""
        self.config = config or get_config().audio
        self.logger = get_logger()
        
        if not MOVIEPY_AVAILABLE:
            self.logger.warning("MoviePy not available - audio extraction will be limited")
        
        if not SPEECH_RECOGNITION_AVAILABLE:
            self.logger.warning("SpeechRecognition not available - transcription will be limited")
    
    def analyze_video_audio(
        self, 
        video_path: Union[str, Path], 
        output_dir: Optional[Union[str, Path]] = None
    ) -> AudioAnalysisResult:
        """Perform complete audio analysis on video."""
        start_time = time.time()
        video_path = Path(video_path)
        
        self.logger.log_operation_start(
            "audio analysis", 
            video=video_path.name,
            config=f"segment_duration={self.config.segment_duration}s"
        )
        
        if not video_path.exists():
            error_msg = f"Video file not found: {video_path}"
            self.logger.error(error_msg)
            return AudioAnalysisResult(
                success=False,
                video_path=str(video_path),
                audio_path=None,
                transcript_path=None,
                duration=0.0,
                sample_rate=None,
                audio_segments=[],
                transcript_segments=[],
                error_message=error_msg
            )
        
        try:
            # Set up output directory
            if output_dir is None:
                output_dir = video_path.parent.parent / "audio"
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract audio
            audio_path, duration, sample_rate = self._extract_audio(video_path, output_dir)
            
            # Analyze audio levels
            audio_segments = []
            if audio_path and self.config.enable_level_analysis:
                audio_segments = self._analyze_audio_levels(audio_path)
            
            # Transcribe audio
            transcript_segments = []
            transcript_path = None
            if audio_path and self.config.enable_transcription:
                transcript_segments = self._transcribe_audio(audio_path)
                if transcript_segments:
                    transcript_path = self._save_transcript(transcript_segments, output_dir, video_path.stem)
            
            duration_seconds = time.time() - start_time
            self.logger.log_operation_complete(
                "audio analysis",
                duration=duration_seconds,
                audio_segments=len(audio_segments),
                transcript_segments=len(transcript_segments)
            )
            
            return AudioAnalysisResult(
                success=True,
                video_path=str(video_path),
                audio_path=str(audio_path) if audio_path else None,
                transcript_path=str(transcript_path) if transcript_path else None,
                duration=duration,
                sample_rate=sample_rate,
                audio_segments=audio_segments,
                transcript_segments=transcript_segments
            )
            
        except Exception as e:
            self.logger.log_operation_error("audio analysis", e)
            return AudioAnalysisResult(
                success=False,
                video_path=str(video_path),
                audio_path=None,
                transcript_path=None,
                duration=0.0,
                sample_rate=None,
                audio_segments=[],
                transcript_segments=[],
                error_message=str(e)
            )
    
    def _extract_audio(
        self, 
        video_path: Path, 
        output_dir: Path
    ) -> Tuple[Optional[Path], float, Optional[float]]:
        """Extract audio from video."""
        if not MOVIEPY_AVAILABLE:
            raise RuntimeError("MoviePy required for audio extraction")
        
        self.logger.debug(f"Extracting audio from {video_path}")
        
        try:
            video = mp.VideoFileClip(str(video_path))
            audio = video.audio
            
            if audio is None:
                self.logger.warning("No audio track found in video")
                return None, video.duration, None
            
            # Set output path
            audio_path = output_dir / f"{video_path.stem}_audio.wav"
            
            # Extract audio
            audio.write_audiofile(
                str(audio_path), 
                verbose=False, 
                logger=None
            )
            
            duration = audio.duration
            sample_rate = audio.fps
            
            # Clean up
            audio.close()
            video.close()
            
            self.logger.debug(f"Audio extracted to {audio_path}")
            return audio_path, duration, sample_rate
            
        except Exception as e:
            self.logger.error(f"Failed to extract audio: {e}")
            raise
    
    def _analyze_audio_levels(self, audio_path: Path) -> List[AudioSegment]:
        """Analyze audio levels in segments."""
        if not MOVIEPY_AVAILABLE:
            self.logger.warning("MoviePy not available for audio level analysis")
            return []
        
        self.logger.debug(f"Analyzing audio levels for {audio_path}")
        
        try:
            audio = mp.AudioFileClip(str(audio_path))
            duration = audio.duration
            segments = []
            
            current_time = 0.0
            segment_count = int(duration / self.config.segment_duration) + 1
            
            while current_time < duration:
                end_time = min(current_time + self.config.segment_duration, duration)
                
                # Extract segment
                segment_audio = audio.subclip(current_time, end_time)
                audio_array = segment_audio.to_soundarray()
                
                # Calculate RMS and dB
                if len(audio_array) > 0:
                    rms = np.sqrt(np.mean(audio_array ** 2))
                    db_level = 20 * np.log10(rms + 1e-10)
                else:
                    rms = 0.0
                    db_level = -60.0
                
                is_silence = db_level < self.config.silence_threshold_db
                
                segments.append(AudioSegment(
                    start_time=current_time,
                    end_time=end_time,
                    duration=end_time - current_time,
                    rms_level=float(rms),
                    db_level=float(db_level),
                    is_silence=is_silence
                ))
                
                current_time = end_time
                
                # Log progress for long audio
                if segment_count > 10:
                    self.logger.log_progress(
                        len(segments), 
                        segment_count, 
                        "audio level analysis"
                    )
            
            audio.close()
            self.logger.debug(f"Analyzed {len(segments)} audio segments")
            return segments
            
        except Exception as e:
            self.logger.error(f"Failed to analyze audio levels: {e}")
            return []
    
    def _transcribe_audio(self, audio_path: Path) -> List[TranscriptSegment]:
        """Transcribe audio to text."""
        if not SPEECH_RECOGNITION_AVAILABLE:
            self.logger.warning("SpeechRecognition not available")
            return []
        
        self.logger.debug(f"Transcribing audio {audio_path}")
        
        # Try multiple languages
        languages_to_try = [self.config.speech_recognition_language] + self.config.fallback_languages
        
        for language in languages_to_try:
            try:
                self.logger.debug(f"Trying transcription with language: {language}")
                segments = self._transcribe_with_language(audio_path, language)
                if segments:
                    self.logger.debug(f"Transcription successful with {language}")
                    return segments
            except Exception as e:
                self.logger.debug(f"Transcription failed with {language}: {e}")
                continue
        
        self.logger.warning("All transcription attempts failed")
        return []
    
    def _transcribe_with_language(self, audio_path: Path, language: str) -> List[TranscriptSegment]:
        """Transcribe audio with specific language."""
        recognizer = sr.Recognizer()
        
        # Convert to WAV if needed
        working_audio_path = audio_path
        if audio_path.suffix.lower() != '.wav':
            if MOVIEPY_AVAILABLE:
                audio_clip = mp.AudioFileClip(str(audio_path))
                working_audio_path = audio_path.with_suffix('.wav')
                audio_clip.write_audiofile(str(working_audio_path), verbose=False, logger=None)
                audio_clip.close()
            else:
                raise RuntimeError("Cannot convert audio format without MoviePy")
        
        with sr.AudioFile(str(working_audio_path)) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            
            # Try different recognition services
            text = None
            service_used = "google"
            
            try:
                # Google Web Speech API (free tier)
                text = recognizer.recognize_google(audio_data, language=language)
            except (sr.UnknownValueError, sr.RequestError):
                try:
                    # Fallback to offline recognition if available
                    text = recognizer.recognize_sphinx(audio_data)
                    service_used = "sphinx"
                except (sr.UnknownValueError, sr.RequestError):
                    raise sr.UnknownValueError("Could not understand audio")
            
            if text:
                # For now, return single segment covering whole audio
                # TODO: Implement timestamp detection for segments
                return [TranscriptSegment(
                    start_time=0.0,
                    end_time=self._get_audio_duration(working_audio_path),
                    text=text,
                    confidence=1.0,  # Default confidence
                    language=language
                )]
        
        return []
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration."""
        if MOVIEPY_AVAILABLE:
            try:
                audio = mp.AudioFileClip(str(audio_path))
                duration = audio.duration
                audio.close()
                return duration
            except:
                pass
        return 0.0
    
    def _save_transcript(
        self, 
        segments: List[TranscriptSegment], 
        output_dir: Path, 
        video_name: str
    ) -> Path:
        """Save transcript to file."""
        transcript_path = output_dir / f"{video_name}_transcript.txt"
        
        try:
            with open(transcript_path, 'w', encoding='utf-8') as f:
                for segment in segments:
                    start_tc = self._seconds_to_timecode(segment.start_time)
                    end_tc = self._seconds_to_timecode(segment.end_time)
                    f.write(f"[{start_tc} - {end_tc}]\n")
                    f.write(f" {segment.text}\n\n")
            
            self.logger.debug(f"Transcript saved to {transcript_path}")
            return transcript_path
            
        except Exception as e:
            self.logger.error(f"Failed to save transcript: {e}")
            raise
    
    def _seconds_to_timecode(self, seconds: float) -> str:
        """Convert seconds to timecode format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        frames = int((seconds % 1) * 25)  # Assuming 25 FPS
        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"
    
    def get_audio_summary_for_timerange(
        self, 
        result: AudioAnalysisResult, 
        start_time: float, 
        end_time: float
    ) -> Dict[str, Union[str, float, int]]:
        """Get audio summary for a specific time range."""
        # Find overlapping audio segments
        overlapping_segments = [
            seg for seg in result.audio_segments
            if not (seg.end_time <= start_time or seg.start_time >= end_time)
        ]
        
        # Find overlapping transcript segments
        overlapping_transcripts = [
            seg for seg in result.transcript_segments
            if not (seg.end_time <= start_time or seg.start_time >= end_time)
        ]
        
        # Calculate audio characteristics
        if overlapping_segments:
            avg_db = sum(seg.db_level for seg in overlapping_segments) / len(overlapping_segments)
            max_db = max(seg.db_level for seg in overlapping_segments)
            min_db = min(seg.db_level for seg in overlapping_segments)
            silence_count = sum(1 for seg in overlapping_segments if seg.is_silence)
            silence_pct = (silence_count / len(overlapping_segments)) * 100
        else:
            avg_db = max_db = min_db = -60.0
            silence_pct = 100.0
        
        # Collect transcript text
        transcript_text = " | ".join(seg.text for seg in overlapping_transcripts)
        
        return {
            "transcript_text": transcript_text,
            "avg_db_level": avg_db,
            "max_db_level": max_db,
            "min_db_level": min_db,
            "silence_percentage": silence_pct,
            "audio_segments_count": len(overlapping_segments),
            "transcript_segments_count": len(overlapping_transcripts),
            "loudness_category": self._categorize_loudness(avg_db)
        }
    
    def _categorize_loudness(self, avg_db: float) -> str:
        """Categorize loudness level."""
        if avg_db > -15:
            return "loud"
        elif avg_db > -25:
            return "normal"
        elif avg_db > -40:
            return "quiet"
        else:
            return "silence"