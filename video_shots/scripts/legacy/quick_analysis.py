#!/usr/bin/env python3
"""Quick video analysis test."""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from video_shots.core.audio_analyzer import AudioAnalyzer
from video_shots.utils.audio_utils import parse_audio_transcript
import cv2


def quick_analysis():
    """Quick analysis of video and audio."""
    video_path = Path("input/videos/nyane_30s.mp4")
    print(f"🎬 Quick analysis of: {video_path}")
    
    # Video metadata
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"📹 Video: {duration:.1f}s, {fps:.1f}fps, {width}x{height}")
    
    # Audio analysis (reuse existing if available)
    audio_path = Path("input/audio/nyane_30s_audio.wav")
    if audio_path.exists():
        print(f"🎵 Audio file exists: {audio_path}")
        
        analyzer = AudioAnalyzer(video_path)
        audio_levels = analyzer.analyze_audio_levels(audio_path, segment_duration=1.0)
        print(f"🔊 Audio levels: {len(audio_levels)} segments")
        
        # Audio statistics
        if audio_levels:
            avg_db = sum(s['db_level'] for s in audio_levels) / len(audio_levels)
            max_db = max(s['db_level'] for s in audio_levels)
            min_db = min(s['db_level'] for s in audio_levels)
            print(f"   Average: {avg_db:.1f}dB, Range: {min_db:.1f} to {max_db:.1f}dB")
    
    # Transcript
    transcript_path = Path("input/audio/nyane_30s.txt")
    if transcript_path.exists():
        transcript = parse_audio_transcript(transcript_path)
        print(f"📝 Transcript: {len(transcript)} segments")
        
        for timestamp, data in list(transcript.items())[:3]:
            print(f"   [{data['start']}] {data['text']}")
        if len(transcript) > 3:
            print(f"   ... and {len(transcript) - 3} more segments")
    
    # Extract a few sample frames
    print(f"\n🖼️ Extracting sample frames...")
    output_dir = Path("output/quick_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    sample_times = [0, duration/4, duration/2, 3*duration/4, duration-1]
    
    for i, time_sec in enumerate(sample_times):
        frame_index = int(time_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        
        if ok:
            filename = f"sample_{i:02d}_{time_sec:.1f}s.png"
            frame_path = output_dir / filename
            cv2.imwrite(str(frame_path), frame)
            print(f"   ✅ {filename}")
    
    cap.release()
    
    # Create quick summary
    summary = {
        "video_path": str(video_path),
        "duration": duration,
        "fps": fps,
        "resolution": [width, height],
        "audio_segments": len(audio_levels) if 'audio_levels' in locals() else 0,
        "transcript_segments": len(transcript) if 'transcript' in locals() else 0,
        "sample_frames": len(sample_times),
        "timestamp": datetime.now().isoformat()
    }
    
    summary_path = output_dir / "quick_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Quick summary saved to: {summary_path}")
    print(f"📁 Output directory: {output_dir}")
    
    return summary


if __name__ == "__main__":
    quick_analysis()