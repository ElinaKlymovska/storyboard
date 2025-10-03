#!/usr/bin/env python3
"""Analyze correlation between audio levels and transcript."""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from video_shots.core.audio_analyzer import AudioAnalyzer
from video_shots.utils.audio_utils import parse_audio_transcript


def analyze_audio_correlation():
    """Analyze correlation between audio levels and speech."""
    video_path = Path("input/videos/nyane_30s.mp4")
    transcript_path = Path("input/audio/nyane_30s.txt")
    
    print(f"🎬 Analyzing audio correlation for: {video_path.name}")
    print("=" * 70)
    
    # Load transcript
    transcript_data = parse_audio_transcript(transcript_path)
    print(f"📝 Loaded transcript with {len(transcript_data)} segments")
    
    # Analyze audio levels
    analyzer = AudioAnalyzer(video_path)
    audio_path = Path("input/audio/nyane_30s_audio.wav")
    
    if not audio_path.exists():
        print("🎵 Extracting audio...")
        audio_path = analyzer.extract_audio()
    
    audio_levels = analyzer.analyze_audio_levels(audio_path, segment_duration=0.5)
    print(f"🔊 Analyzed {len(audio_levels)} audio segments (0.5s each)")
    
    print("\n📊 Audio Analysis with Speech Correlation:")
    print("-" * 90)
    print(f"{'Time Range':<15} {'dB Level':<10} {'RMS':<10} {'Speech':<45}")
    print("-" * 90)
    
    for segment in audio_levels:
        start_time = segment['start_time']
        end_time = segment['end_time']
        db_level = segment['db_level']
        rms_level = segment['rms_level']
        
        # Find corresponding speech
        speech_text = ""
        for timestamp, data in transcript_data.items():
            if timestamp <= start_time <= timestamp + 2:  # within 2 seconds
                speech_text = data['text'][:40] + "..." if len(data['text']) > 40 else data['text']
                break
        
        time_range = f"{start_time:.1f}-{end_time:.1f}s"
        print(f"{time_range:<15} {db_level:>6.1f}dB   {rms_level:>6.4f}   {speech_text:<45}")
    
    # Analyze speech periods vs silence
    print(f"\n🎙️ Speech Analysis:")
    print("-" * 50)
    
    speech_segments = []
    for timestamp, data in transcript_data.items():
        # Find audio level during speech
        speech_db_levels = []
        for segment in audio_levels:
            if (segment['start_time'] <= timestamp <= segment['end_time'] + 1):
                speech_db_levels.append(segment['db_level'])
        
        if speech_db_levels:
            avg_db = sum(speech_db_levels) / len(speech_db_levels)
            speech_segments.append({
                'timestamp': timestamp,
                'text': data['text'],
                'avg_db': avg_db,
                'timecode': data['start']
            })
    
    for speech in speech_segments:
        print(f"[{speech['timecode']}] {speech['avg_db']:>6.1f}dB: {speech['text']}")
    
    # Calculate statistics
    if speech_segments:
        speech_db_levels = [s['avg_db'] for s in speech_segments]
        avg_speech_db = sum(speech_db_levels) / len(speech_db_levels)
        
        # Non-speech periods
        all_db_levels = [s['db_level'] for s in audio_levels]
        avg_all_db = sum(all_db_levels) / len(all_db_levels)
        
        print(f"\n📈 Statistics:")
        print(f"   Average dB during speech: {avg_speech_db:.1f}dB")
        print(f"   Average dB overall: {avg_all_db:.1f}dB")
        print(f"   Speech segments: {len(speech_segments)}")
        print(f"   Total audio segments: {len(audio_levels)}")
        
        # Correlation analysis
        speech_periods = len(speech_segments)
        total_periods = len(audio_levels)
        speech_ratio = speech_periods / total_periods * 100
        
        print(f"   Speech coverage: {speech_ratio:.1f}% of analyzed periods")
        
        # Loudest moments
        loudest_segments = sorted(audio_levels, key=lambda x: x['db_level'], reverse=True)[:5]
        print(f"\n🔊 Top 5 loudest moments:")
        for i, segment in enumerate(loudest_segments, 1):
            corresponding_speech = ""
            for speech in speech_segments:
                if abs(speech['timestamp'] - segment['start_time']) < 1:
                    corresponding_speech = f" - \"{speech['text'][:30]}...\""
                    break
            print(f"   {i}. {segment['start_time']:.1f}s: {segment['db_level']:.1f}dB{corresponding_speech}")


if __name__ == "__main__":
    analyze_audio_correlation()