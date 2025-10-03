#!/usr/bin/env python3
"""Test complete video analysis with audio integration."""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from video_shots.core.audio_analyzer import AudioAnalyzer
from video_shots.utils.audio_utils import parse_audio_transcript
from video_shots.core.keyframe_selector import KeyframeSelector
from video_shots.core.models import TimePoint
from video_shots.utils.time_utils import collect_timepoints
import cv2


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    video_path: str
    duration: float
    fps: float
    resolution: tuple
    audio_analysis: Dict
    keyframes: List[Dict]
    shots: List[Dict]
    transcript: Dict
    timestamp: str


class LocalVideoAnalyzer:
    """Local video analyzer without external APIs."""
    
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.video_name = video_path.stem
        self.output_dir = Path("output") / f"{self.video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_complete_video(self) -> AnalysisResult:
        """Perform complete video analysis."""
        print(f"🎬 Starting complete analysis of: {self.video_path}")
        print("=" * 70)
        
        # Get video metadata
        metadata = self._get_video_metadata()
        print(f"📹 Video: {metadata['duration']:.1f}s, {metadata['fps']:.1f}fps, {metadata['resolution']}")
        
        # Audio analysis
        print("\n🎵 Analyzing audio...")
        audio_analyzer = AudioAnalyzer(self.video_path)
        audio_analysis = audio_analyzer.analyze_full_audio_track()
        
        # Load transcript
        transcript_path = Path("input/audio") / f"{self.video_name}.txt"
        transcript = parse_audio_transcript(transcript_path) if transcript_path.exists() else {}
        print(f"📝 Loaded transcript with {len(transcript)} segments")
        
        # Extract keyframes using event-driven selection
        print("\n🖼️ Selecting keyframes...")
        keyframes = self._extract_keyframes(metadata)
        print(f"✅ Selected {len(keyframes)} keyframes")
        
        # Analyze shots
        print("\n🎯 Analyzing shots...")
        shots = self._analyze_shots(metadata, audio_analysis, transcript)
        print(f"✅ Analyzed {len(shots)} shots")
        
        # Create result
        result = AnalysisResult(
            video_path=str(self.video_path),
            duration=metadata['duration'],
            fps=metadata['fps'],
            resolution=metadata['resolution'],
            audio_analysis=audio_analysis,
            keyframes=keyframes,
            shots=shots,
            transcript=transcript,
            timestamp=datetime.now().isoformat()
        )
        
        # Save results
        self._save_results(result)
        
        return result
    
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

            return {
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "resolution": (width, height),
            }
        finally:
            cap.release()
    
    def _extract_keyframes(self, metadata: Dict) -> List[Dict]:
        """Extract keyframes using event-driven selection."""
        try:
            # Use event-driven keyframe selection
            keyframe_selector = KeyframeSelector(
                mode="event_driven",
                facial_sensitivity=0.15,
                pose_sensitivity=0.25,
                scene_sensitivity=0.4,
                similarity_threshold=0.80,
                filter_strategy="keep_first_discard_rest"
            )
            
            # Select keyframes
            target_keyframes = max(10, int(metadata["duration"] / 2))
            timepoints = keyframe_selector.select_keyframes(
                video_path=self.video_path,
                target_count=target_keyframes
            )
            
            # Extract and analyze frames
            keyframes = []
            cap = cv2.VideoCapture(str(self.video_path))
            
            try:
                for i, tp in enumerate(timepoints):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, tp.frame_index)
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    
                    # Save frame
                    frame_filename = f"keyframe_{i:03d}_{tp.seconds:.3f}s.png"
                    frame_path = self.output_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    
                    # Create keyframe record (without external API analysis)
                    keyframe = {
                        "index": i,
                        "timestamp": tp.seconds,
                        "timecode": self._format_timecode(tp.seconds),
                        "frame_index": tp.frame_index,
                        "filename": frame_filename,
                        "local_path": str(frame_path),
                        "scene_id": f"scene_{(i // 5) + 1:02d}",
                        "shot_id": f"shot_{(i // 3) + 1:02d}",
                        "analysis": "Local analysis - frame extracted successfully"
                    }
                    keyframes.append(keyframe)
                    
            finally:
                cap.release()
            
            return keyframes
            
        except Exception as e:
            print(f"⚠️ Error in keyframe extraction: {e}")
            # Fallback to interval-based extraction
            return self._extract_keyframes_interval(metadata)
    
    def _extract_keyframes_interval(self, metadata: Dict) -> List[Dict]:
        """Fallback interval-based keyframe extraction."""
        timepoints = collect_timepoints(
            start=0.0,
            end=metadata["duration"],
            step=2.0,  # Every 2 seconds
            fps=metadata["fps"],
            frame_count=metadata["frame_count"],
        )
        
        keyframes = []
        cap = cv2.VideoCapture(str(self.video_path))
        
        try:
            for i, tp in enumerate(timepoints):
                cap.set(cv2.CAP_PROP_POS_FRAMES, tp.frame_index)
                ok, frame = cap.read()
                if not ok:
                    continue
                
                # Save frame
                frame_filename = f"keyframe_{i:03d}_{tp.seconds:.3f}s.png"
                frame_path = self.output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                
                keyframe = {
                    "index": i,
                    "timestamp": tp.seconds,
                    "timecode": self._format_timecode(tp.seconds),
                    "frame_index": tp.frame_index,
                    "filename": frame_filename,
                    "local_path": str(frame_path),
                    "scene_id": f"scene_{(i // 5) + 1:02d}",
                    "shot_id": f"shot_{(i // 3) + 1:02d}",
                    "analysis": "Interval-based extraction"
                }
                keyframes.append(keyframe)
                
        finally:
            cap.release()
        
        return keyframes
    
    def _analyze_shots(self, metadata: Dict, audio_analysis: Dict, transcript: Dict) -> List[Dict]:
        """Analyze video shots with audio correlation."""
        shots = []
        duration = metadata["duration"]
        shot_duration = 5.0  # 5 second shots
        
        start = 0.0
        shot_index = 0
        
        while start < duration:
            end = min(start + shot_duration, duration)
            shot_index += 1
            
            # Collect audio summary for this shot
            audio_summary = self._collect_audio_summary(start, end, audio_analysis, transcript)
            
            # Analyze audio characteristics
            audio_chars = self._analyze_shot_audio(start, end, audio_analysis)
            
            shot = {
                "shot_id": f"shot_{shot_index:02d}",
                "scene_id": f"scene_{((shot_index - 1) // 3) + 1:02d}",
                "start_timecode": self._format_timecode(start),
                "end_timecode": self._format_timecode(end),
                "duration": end - start,
                "start_seconds": start,
                "end_seconds": end,
                "audio_summary": audio_summary,
                "audio_characteristics": audio_chars,
                "shot_type": self._estimate_shot_type(shot_index),
                "status": "analyzed"
            }
            
            shots.append(shot)
            start += shot_duration
        
        return shots
    
    def _collect_audio_summary(self, start: float, end: float, audio_analysis: Dict, transcript: Dict) -> str:
        """Collect audio summary for shot."""
        summaries = []
        
        # From transcript
        for timestamp, data in transcript.items():
            if start <= timestamp <= end:
                summaries.append(data.get('text', ''))
        
        # From audio analysis transcript
        if audio_analysis.get('transcript'):
            for segment in audio_analysis['transcript']:
                seg_start = segment.get('start_time', 0)
                seg_end = segment.get('end_time', 0)
                if (seg_start <= end and seg_end >= start):
                    summaries.append(segment.get('text', ''))
        
        return ' | '.join(summaries) if summaries else ""
    
    def _analyze_shot_audio(self, start: float, end: float, audio_analysis: Dict) -> Dict:
        """Analyze audio characteristics for shot."""
        if not audio_analysis.get('audio_levels'):
            return {}
        
        relevant_segments = []
        for segment in audio_analysis['audio_levels']:
            seg_start = segment.get('start_time', 0)
            seg_end = segment.get('end_time', 0)
            if (seg_start <= end and seg_end >= start):
                relevant_segments.append(segment)
        
        if not relevant_segments:
            return {}
        
        # Calculate statistics
        db_levels = [seg['db_level'] for seg in relevant_segments]
        rms_levels = [seg['rms_level'] for seg in relevant_segments]
        silence_count = sum(1 for seg in relevant_segments if seg.get('is_silence', False))
        
        return {
            "avg_db_level": sum(db_levels) / len(db_levels),
            "max_db_level": max(db_levels),
            "min_db_level": min(db_levels),
            "avg_rms_level": sum(rms_levels) / len(rms_levels),
            "silence_percentage": silence_count / len(relevant_segments) * 100,
            "audio_segments_count": len(relevant_segments),
            "loudness_category": self._categorize_loudness(sum(db_levels) / len(db_levels))
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
    
    def _estimate_shot_type(self, shot_index: int) -> str:
        """Estimate shot type based on position."""
        types = ["WS", "MS", "CU", "MS", "WS"]
        return types[shot_index % len(types)]
    
    def _format_timecode(self, seconds: float) -> str:
        """Format seconds to timecode."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
    
    def _save_results(self, result: AnalysisResult) -> None:
        """Save analysis results."""
        # Save JSON
        json_path = self.output_dir / "complete_analysis.json"
        
        # Convert result to dict and handle non-serializable objects
        result_dict = {
            "video_path": result.video_path,
            "duration": result.duration,
            "fps": result.fps,
            "resolution": result.resolution,
            "audio_analysis": result.audio_analysis,
            "keyframes": result.keyframes,
            "shots": result.shots,
            "transcript": result.transcript,
            "timestamp": result.timestamp
        }
        
        # Clean audio analysis for JSON serialization
        if result_dict['audio_analysis'].get('audio_levels'):
            for segment in result_dict['audio_analysis']['audio_levels']:
                for key, value in segment.items():
                    if hasattr(value, 'item'):  # numpy types
                        segment[key] = value.item()
                    elif isinstance(value, bool):
                        segment[key] = bool(value)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {json_path}")
        
        # Create HTML summary
        self._create_html_summary(result)
    
    def _create_html_summary(self, result: AnalysisResult) -> None:
        """Create HTML summary report."""
        html_path = self.output_dir / "analysis_report.html"
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Analysis Report - {self.video_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #4CAF50; background-color: #f9f9f9; }}
        .keyframe {{ display: inline-block; margin: 10px; text-align: center; border: 1px solid #ddd; padding: 10px; background: white; }}
        .keyframe img {{ max-width: 200px; height: auto; }}
        .shot {{ padding: 10px; margin: 5px 0; background: #e8f5e8; border-radius: 4px; }}
        .audio-chart {{ background: #fff; padding: 15px; border-radius: 4px; margin: 10px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Video Analysis Report</h1>
            <h2>{self.video_name}</h2>
            <p>Duration: {result.duration:.1f}s | FPS: {result.fps:.1f} | Resolution: {result.resolution[0]}x{result.resolution[1]}</p>
            <p>Generated: {result.timestamp}</p>
        </div>
        
        <div class="section">
            <h3>🎵 Audio Analysis</h3>
            <p><strong>Audio segments analyzed:</strong> {len(result.audio_analysis.get('audio_levels', []))}</p>
            <p><strong>Speech segments:</strong> {len(result.audio_analysis.get('transcript', []))}</p>
            <p><strong>Audio file:</strong> {result.audio_analysis.get('audio_path', 'N/A')}</p>
        </div>
        
        <div class="section">
            <h3>🖼️ Keyframes ({len(result.keyframes)})</h3>
            <div class="keyframes-grid">
"""
        
        # Add keyframes
        for kf in result.keyframes[:12]:  # Limit to first 12 keyframes
            if Path(kf['local_path']).exists():
                html_content += f"""
                <div class="keyframe">
                    <img src="{kf['filename']}" alt="Keyframe {kf['index']}">
                    <br><strong>{kf['timecode']}</strong>
                    <br>{kf['scene_id']} - {kf['shot_id']}
                </div>
"""
        
        html_content += """
            </div>
        </div>
        
        <div class="section">
            <h3>🎯 Shot Analysis</h3>
            <table>
                <tr>
                    <th>Shot ID</th>
                    <th>Time Range</th>
                    <th>Duration</th>
                    <th>Audio Summary</th>
                    <th>Audio Level</th>
                </tr>
"""
        
        # Add shots
        for shot in result.shots:
            audio_char = shot.get('audio_characteristics', {})
            loudness = audio_char.get('loudness_category', 'unknown')
            avg_db = audio_char.get('avg_db_level', 0)
            
            html_content += f"""
                <tr>
                    <td>{shot['shot_id']}</td>
                    <td>{shot['start_timecode']} - {shot['end_timecode']}</td>
                    <td>{shot['duration']:.1f}s</td>
                    <td>{shot['audio_summary'][:100]}{'...' if len(shot['audio_summary']) > 100 else ''}</td>
                    <td>{loudness} ({avg_db:.1f}dB)</td>
                </tr>
"""
        
        html_content += """
            </table>
        </div>
        
        <div class="section">
            <h3>📝 Transcript</h3>
            <table>
                <tr>
                    <th>Timecode</th>
                    <th>Text</th>
                </tr>
"""
        
        # Add transcript
        for timestamp, data in result.transcript.items():
            html_content += f"""
                <tr>
                    <td>{data.get('start', 'N/A')}</td>
                    <td>{data.get('text', '')}</td>
                </tr>
"""
        
        html_content += """
            </table>
        </div>
    </div>
</body>
</html>
"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 HTML report saved to: {html_path}")


def main():
    """Run complete video analysis."""
    video_path = Path("input/videos/nyane_30s.mp4")
    
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return 1
    
    analyzer = LocalVideoAnalyzer(video_path)
    result = analyzer.analyze_complete_video()
    
    print(f"\n✅ Complete analysis finished!")
    print(f"📊 Summary:")
    print(f"   - Duration: {result.duration:.1f}s")
    print(f"   - Keyframes: {len(result.keyframes)}")
    print(f"   - Shots: {len(result.shots)}")
    print(f"   - Audio segments: {len(result.audio_analysis.get('audio_levels', []))}")
    print(f"   - Speech segments: {len(result.transcript)}")
    print(f"   - Output directory: {analyzer.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())