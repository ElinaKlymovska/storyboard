"""Audio analysis module for extracting and analyzing audio from videos."""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

try:
    import moviepy.editor as mp
    import numpy as np
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("⚠️ MoviePy not available. Audio extraction will be limited.", file=sys.stderr)

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("⚠️ SpeechRecognition not available. Speech-to-text will be limited.", file=sys.stderr)


class AudioAnalyzer:
    """Analyzer for extracting and processing audio from videos."""
    
    def __init__(self, video_path: Path):
        """Initialize audio analyzer with video path."""
        self.video_path = Path(video_path)
        self.audio_data = None
        self.sample_rate = None
        self.duration = None
        
    def extract_audio(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """Extract audio from video file."""
        if not MOVIEPY_AVAILABLE:
            print("❌ MoviePy required for audio extraction", file=sys.stderr)
            return None
            
        try:
            # Load video
            video = mp.VideoFileClip(str(self.video_path))
            
            # Extract audio
            audio = video.audio
            if audio is None:
                print("⚠️ No audio track found in video", file=sys.stderr)
                return None
                
            # Set output path
            if output_path is None:
                output_dir = self.video_path.parent.parent / "audio"
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"{self.video_path.stem}_audio.wav"
            
            # Write audio file
            audio.write_audiofile(str(output_path), verbose=False, logger=None)
            
            # Store audio properties
            self.sample_rate = audio.fps
            self.duration = audio.duration
            
            # Clean up
            audio.close()
            video.close()
            
            print(f"✅ Audio extracted to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error extracting audio: {e}", file=sys.stderr)
            return None
    
    def analyze_audio_levels(self, audio_path: Path, segment_duration: float = 1.0) -> List[Dict]:
        """Analyze audio levels in segments."""
        if not MOVIEPY_AVAILABLE:
            return []
            
        try:
            audio = mp.AudioFileClip(str(audio_path))
            duration = audio.duration
            
            segments = []
            current_time = 0.0
            
            while current_time < duration:
                end_time = min(current_time + segment_duration, duration)
                
                # Extract segment
                segment = audio.subclip(current_time, end_time)
                
                # Get audio array
                audio_array = segment.to_soundarray()
                
                # Calculate RMS (Root Mean Square) for volume level
                if len(audio_array) > 0:
                    rms = np.sqrt(np.mean(audio_array ** 2))
                    db_level = 20 * np.log10(rms + 1e-10)  # Add small value to avoid log(0)
                else:
                    db_level = -60  # Silence
                
                segments.append({
                    'start_time': current_time,
                    'end_time': end_time,
                    'duration': end_time - current_time,
                    'rms_level': float(rms) if len(audio_array) > 0 else 0.0,
                    'db_level': float(db_level),
                    'is_silence': db_level < -40  # Consider < -40dB as silence
                })
                
                current_time = end_time
            
            audio.close()
            return segments
            
        except Exception as e:
            print(f"❌ Error analyzing audio levels: {e}", file=sys.stderr)
            return []
    
    def transcribe_audio(self, audio_path: Path, language: str = "uk-UA") -> List[Dict]:
        """Transcribe audio to text with timestamps."""
        if not SPEECH_RECOGNITION_AVAILABLE:
            print("⚠️ SpeechRecognition library not available", file=sys.stderr)
            return []
        
        try:
            recognizer = sr.Recognizer()
            
            # Convert to WAV if needed
            if audio_path.suffix.lower() != '.wav':
                # Use MoviePy to convert
                if MOVIEPY_AVAILABLE:
                    audio_clip = mp.AudioFileClip(str(audio_path))
                    temp_wav = audio_path.with_suffix('.wav')
                    audio_clip.write_audiofile(str(temp_wav), verbose=False, logger=None)
                    audio_clip.close()
                    audio_path = temp_wav
                else:
                    print("⚠️ Cannot convert audio format without MoviePy", file=sys.stderr)
                    return []
            
            # Load audio file
            with sr.AudioFile(str(audio_path)) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Get full audio
                audio_data = recognizer.record(source)
                
                try:
                    # Try Google Web Speech API (free tier)
                    text = recognizer.recognize_google(audio_data, language=language)
                    
                    # For now, return the full transcription
                    # TODO: Implement timestamp detection
                    return [{
                        'start_time': 0.0,
                        'end_time': self.duration or 0.0,
                        'text': text,
                        'confidence': 1.0
                    }]
                    
                except sr.UnknownValueError:
                    print("⚠️ Could not understand audio", file=sys.stderr)
                    return []
                except sr.RequestError as e:
                    print(f"⚠️ Speech recognition error: {e}", file=sys.stderr)
                    return []
                    
        except Exception as e:
            print(f"❌ Error transcribing audio: {e}", file=sys.stderr)
            return []
    
    def create_transcript_file(self, transcript_data: List[Dict], output_path: Path) -> bool:
        """Create transcript file in the expected format."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for segment in transcript_data:
                    start_time = self._seconds_to_timecode(segment['start_time'])
                    end_time = self._seconds_to_timecode(segment['end_time'])
                    text = segment['text']
                    
                    f.write(f"[{start_time} - {end_time}]\n")
                    f.write(f" {text}\n\n")
            
            print(f"✅ Transcript saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating transcript file: {e}", file=sys.stderr)
            return False
    
    def _seconds_to_timecode(self, seconds: float) -> str:
        """Convert seconds to timecode format HH:MM:SS:FF."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        frames = int((seconds % 1) * 25)  # Assuming 25 FPS
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"
    
    def analyze_full_audio_track(self, output_dir: Optional[Path] = None) -> Dict:
        """Complete audio analysis pipeline."""
        if output_dir is None:
            output_dir = self.video_path.parent.parent / "audio"
        
        output_dir.mkdir(exist_ok=True)
        
        results = {
            'video_path': str(self.video_path),
            'audio_path': None,
            'transcript_path': None,
            'audio_levels': [],
            'transcript': [],
            'success': False
        }
        
        # Extract audio
        audio_path = self.extract_audio(output_dir / f"{self.video_path.stem}_audio.wav")
        if not audio_path:
            return results
        
        results['audio_path'] = str(audio_path)
        
        # Analyze audio levels
        print("🔊 Analyzing audio levels...")
        audio_levels = self.analyze_audio_levels(audio_path)
        results['audio_levels'] = audio_levels
        
        # Transcribe audio
        print("🎙️ Transcribing audio...")
        transcript = self.transcribe_audio(audio_path)
        results['transcript'] = transcript
        
        # Create transcript file
        if transcript:
            transcript_path = output_dir / f"{self.video_path.stem}_transcript.txt"
            if self.create_transcript_file(transcript, transcript_path):
                results['transcript_path'] = str(transcript_path)
        
        results['success'] = True
        return results


def analyze_video_audio(video_path: Path, output_dir: Optional[Path] = None) -> Dict:
    """Convenient function to analyze audio from video."""
    analyzer = AudioAnalyzer(video_path)
    return analyzer.analyze_full_audio_track(output_dir)


if __name__ == "__main__":
    # Test with the sample video
    video_path = Path("input/videos/nyane_30s.mp4")
    if video_path.exists():
        results = analyze_video_audio(video_path)
        print(f"Analysis results: {results}")
    else:
        print(f"Video file not found: {video_path}")