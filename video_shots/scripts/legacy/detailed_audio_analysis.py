#!/usr/bin/env python3
"""Detailed audio analysis script."""

import sys
from pathlib import Path
import json

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from video_shots.core.audio_analyzer import AudioAnalyzer


def detailed_audio_analysis():
    """Perform detailed audio analysis."""
    video_path = Path("input/videos/nyane_30s.mp4")
    
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return 1
    
    print(f"🎬 Starting detailed audio analysis for: {video_path}")
    print("=" * 60)
    
    analyzer = AudioAnalyzer(video_path)
    
    # Extract audio
    print("🎵 Extracting audio...")
    audio_path = analyzer.extract_audio()
    if not audio_path:
        print("❌ Failed to extract audio")
        return 1
    
    print(f"✅ Audio extracted to: {audio_path}")
    
    # Analyze audio levels in detail
    print("\n🔊 Analyzing audio levels in detail...")
    audio_levels = analyzer.analyze_audio_levels(audio_path, segment_duration=1.0)
    
    if audio_levels:
        print(f"📊 Found {len(audio_levels)} audio segments:")
        print("\nDetailed audio levels:")
        print("-" * 80)
        print(f"{'Time':<12} {'Duration':<10} {'RMS':<12} {'dB Level':<12} {'Status':<12}")
        print("-" * 80)
        
        for segment in audio_levels:
            start = segment['start_time']
            end = segment['end_time']
            duration = segment['duration']
            rms = segment['rms_level']
            db = segment['db_level']
            status = "SILENCE" if segment['is_silence'] else "AUDIO"
            
            print(f"{start:.1f}-{end:.1f}s   {duration:.1f}s      {rms:.4f}     {db:.1f}dB       {status}")
        
        # Statistics
        total_audio_time = sum(s['duration'] for s in audio_levels if not s['is_silence'])
        total_silence_time = sum(s['duration'] for s in audio_levels if s['is_silence'])
        avg_db = sum(s['db_level'] for s in audio_levels if not s['is_silence']) / max(1, len([s for s in audio_levels if not s['is_silence']]))
        
        print("-" * 80)
        print(f"📈 Audio Statistics:")
        print(f"   Total audio time: {total_audio_time:.1f}s")
        print(f"   Total silence time: {total_silence_time:.1f}s")
        print(f"   Average dB level: {avg_db:.1f}dB")
        print(f"   Audio percentage: {(total_audio_time / (total_audio_time + total_silence_time) * 100):.1f}%")
    
    # Try transcription with different settings
    print(f"\n🎙️ Attempting speech transcription...")
    
    # Try Ukrainian first
    print("Trying Ukrainian (uk-UA)...")
    transcript_uk = analyzer.transcribe_audio(audio_path, language="uk-UA")
    
    if transcript_uk:
        print("✅ Ukrainian transcription successful!")
        for segment in transcript_uk:
            print(f"   [{segment['start_time']:.1f}s-{segment['end_time']:.1f}s]: {segment['text']}")
    else:
        print("❌ Ukrainian transcription failed")
    
    # Try English
    print("Trying English (en-US)...")
    transcript_en = analyzer.transcribe_audio(audio_path, language="en-US")
    
    if transcript_en:
        print("✅ English transcription successful!")
        for segment in transcript_en:
            print(f"   [{segment['start_time']:.1f}s-{segment['end_time']:.1f}s]: {segment['text']}")
    else:
        print("❌ English transcription failed")
    
    # Try auto-detect
    print("Trying auto-detect...")
    transcript_auto = analyzer.transcribe_audio(audio_path, language="auto")
    
    if transcript_auto:
        print("✅ Auto-detect transcription successful!")
        for segment in transcript_auto:
            print(f"   [{segment['start_time']:.1f}s-{segment['end_time']:.1f}s]: {segment['text']}")
    else:
        print("❌ Auto-detect transcription failed")
    
    # Save detailed results
    results = {
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "audio_levels": audio_levels,
        "transcripts": {
            "ukrainian": transcript_uk,
            "english": transcript_en,
            "auto_detect": transcript_auto
        }
    }
    
    output_file = Path("detailed_audio_analysis.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(detailed_audio_analysis())