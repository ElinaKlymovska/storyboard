"""Image analysis using AI models."""

import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict

import replicate

from video_shots.config.config import DEFAULT_ANALYSIS_PROMPT, REPLICATE_MODEL, REPLICATE_MAX_TOKENS, REPLICATE_TEMPERATURE
from video_shots.core.exceptions import VideoShotsError


def create_structured_analysis_from_text(text: str) -> Dict:
    """Create structured analysis from text description."""
    
    # Determine frame type
    shot_type = "close-up"
    if "wide" in text.lower() or "full" in text.lower():
        shot_type = "wide"
    elif "medium" in text.lower():
        shot_type = "medium"
    
    # Determine emotions
    emotions = []
    if any(word in text.lower() for word in ["happy", "smiling", "joy", "excited"]):
        emotions.append("happy")
    if any(word in text.lower() for word in ["serious", "focused", "concentrated"]):
        emotions.append("serious")
    if any(word in text.lower() for word in ["relaxed", "calm", "peaceful"]):
        emotions.append("relaxed")
    if any(word in text.lower() for word in ["dramatic", "intense", "dramatic"]):
        emotions.append("dramatic")
    
    # Determine action level
    action_level = "static"
    if any(word in text.lower() for word in ["moving", "action", "motion", "dynamic"]):
        action_level = "medium"
    if any(word in text.lower() for word in ["fast", "quick", "rapid", "intense"]):
        action_level = "high"
    
    # Determine suitability score
    score = 5
    if "close-up" in text.lower() and "face" in text.lower():
        score = 7
    if "background" in text.lower() and "black" in text.lower():
        score += 1
    if emotions:
        score += 1
    
    # Determine characters
    characters = ["person"]
    if "woman" in text.lower():
        characters = ["woman"]
    elif "man" in text.lower():
        characters = ["man"]
    
    return {
        "scene_description": text[:200] + "..." if len(text) > 200 else text,
        "visual_elements": {
            "characters": characters,
            "objects": ["face", "hair"] if "hair" in text.lower() else ["face"],
            "background": "black background" if "black" in text.lower() else "background",
            "lighting": "cinematic lighting" if "lighting" in text.lower() else "natural lighting"
        },
        "composition": {
            "shot_type": shot_type,
            "camera_angle": "eye-level",
            "framing": "face-focused framing"
        },
        "storyboard_suitability": {
            "score": min(score, 10),
            "reasoning": f"Good for storyboard: {shot_type} shot with clear character focus",
            "key_moments": ["character expression", "visual focus"],
            "transition_potential": "good" if score >= 6 else "fair"
        },
        "production_notes": {
            "makeup_visible": "makeup" in text.lower(),
            "emotions": emotions if emotions else ["neutral"],
            "action_level": action_level,
            "technical_quality": "good"
        }
    }


def analyze_image_with_replicate(image_path: Path, prompt: str = None) -> Dict:
    """Analyze image using Replicate vision model and return structured result."""
    if prompt is None:
        prompt = DEFAULT_ANALYSIS_PROMPT
    
    if not image_path.exists():
        return _create_error_response(f"Image file not found: {image_path}")
    
    try:
        image_data = _encode_image_to_base64(image_path)
        response = _call_replicate_api(image_data, prompt)
        return _parse_response(response)
    except Exception as exc:
        return _create_error_response(f"Analysis failed: {exc}")


def _encode_image_to_base64(image_path: Path) -> str:
    """Encode image file to base64."""
    try:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{image_data}"
    except IOError as exc:
        raise VideoShotsError(f"Failed to read image file: {exc}")


def _call_replicate_api(image_url: str, prompt: str) -> str:
    """Call Replicate API for image analysis."""
    try:
        output = replicate.run(
            REPLICATE_MODEL,
            input={
                "image": image_url,
                "prompt": prompt,
                "max_tokens": REPLICATE_MAX_TOKENS,
                "temperature": REPLICATE_TEMPERATURE
            }
        )
        
        # Handle generator or string response
        if hasattr(output, '__iter__') and not isinstance(output, str):
            return ''.join(str(item) for item in output)
        return str(output)
    except Exception as exc:
        raise VideoShotsError(f"Replicate API call failed: {exc}")


def _parse_response(response: str) -> Dict:
    """Parse API response, trying JSON first, then structured text analysis."""
    try:
        # Try to find and parse JSON in response
        json_match = re.search(r'\{.*\}', response.strip(), re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # If JSON parsing fails, create structured analysis from text
    return create_structured_analysis_from_text(response.strip())


def _create_error_response(error_message: str) -> Dict:
    """Create standardized error response."""
    return {
        "error": error_message,
        "scene_description": "Analysis unavailable",
        "storyboard_suitability": {"score": 0, "reasoning": "Analysis failed"}
    }


def save_analysis_to_file(analysis: Dict, output_path: Path) -> None:
    """Save structured analysis to JSON file."""
    # Create directory if not exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)


def check_replicate_token() -> bool:
    """Check Replicate API token availability."""
    return bool(os.getenv("REPLICATE_API_TOKEN"))
