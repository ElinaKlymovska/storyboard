"""Audio transcript utilities."""

import re
import sys
from pathlib import Path
from typing import Dict

from .time_utils import parse_timecode_to_seconds


def parse_audio_transcript(audio_file_path: Path) -> Dict:
    """Parse audio transcript and return dictionary with timecodes."""
    try:
        if not audio_file_path.exists():
            return {}
        
        with open(audio_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse timecodes and text
        pattern = r'\[(\d{2}:\d{2}:\d{2}:\d{2}) - (\d{2}:\d{2}:\d{2}:\d{2})\]\s*(.+)'
        matches = re.findall(pattern, content)
        
        transcript_data = {}
        for start_time, end_time, text in matches:
            # Convert timecode to seconds
            start_seconds = parse_timecode_to_seconds(start_time)
            end_seconds = parse_timecode_to_seconds(end_time)
            
            # Save for each frame
            transcript_data[start_seconds] = {
                'start': start_time,
                'end': end_time,
                'text': text.strip()
            }
        
        return transcript_data
        
    except Exception as exc:
        print(f"⚠️ Audio file parsing error: {exc}", file=sys.stderr)
        return {}


def get_vo_for_timestamp(transcript_data: Dict, timestamp: float) -> str:
    """Find VO/Sound for given timecode."""
    if not transcript_data:
        return ""
    
    # Find closest timecode
    closest_time = None
    min_diff = float('inf')
    
    for time_key in transcript_data.keys():
        diff = abs(time_key - timestamp)
        if diff < min_diff:
            min_diff = diff
            closest_time = time_key
    
    # If found close timecode (within 2 seconds)
    if closest_time is not None and min_diff <= 2.0:
        return transcript_data[closest_time]['text']
    
    return ""
