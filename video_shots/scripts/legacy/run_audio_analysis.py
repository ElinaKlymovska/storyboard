#!/usr/bin/env python3
"""Script to run audio analysis on the sample video."""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from video_shots.core.audio_analyzer import analyze_video_audio


def main():
    """Run audio analysis on the sample video."""
    video_path = Path("input/videos/nyane_30s.mp4")
    
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return 1
    
    print(f"🎬 Starting audio analysis for: {video_path}")
    print("=" * 50)
    
    try:
        results = analyze_video_audio(video_path)
        
        print("\n📊 Analysis Results:")
        print("=" * 50)
        print(f"Video: {results['video_path']}")
        print(f"Audio extracted: {results['audio_path']}")
        print(f"Transcript file: {results['transcript_path']}")
        print(f"Success: {results['success']}")
        
        if results['audio_levels']:
            print(f"\n🔊 Audio Levels Analysis:")
            print(f"Total segments: {len(results['audio_levels'])}")
            
            # Show first few segments
            for i, segment in enumerate(results['audio_levels'][:5]):
                print(f"  [{segment['start_time']:.1f}s-{segment['end_time']:.1f}s]: "
                      f"{segment['db_level']:.1f}dB {'(silence)' if segment['is_silence'] else ''}")
        
        if results['transcript']:
            print(f"\n🎙️ Transcript:")
            for segment in results['transcript']:
                print(f"  [{segment['start_time']:.1f}s-{segment['end_time']:.1f}s]: {segment['text']}")
        
        print("\n✅ Audio analysis completed!")
        return 0
        
    except Exception as e:
        print(f"❌ Error during audio analysis: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())