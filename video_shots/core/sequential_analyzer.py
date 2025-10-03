from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import replicate
from PIL import Image

from video_shots.config.config import REPLICATE_MODEL, REPLICATE_MAX_TOKENS, REPLICATE_TEMPERATURE
from video_shots.core.exceptions import VideoShotsError


class SequentialAnalyzer:
    """
    Analyzes a video sequentially, frame by frame, to detect visible changes.
    """

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        self.video_name = self.video_path.stem

    def run(self):
        """
        Runs the sequential analysis pipeline.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        
        last_emitted_frame = None
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index == 0:
                last_emitted_frame = frame
                frame_index += 1
                continue

            change_data = self._compare_frames(last_emitted_frame, frame)

            if change_data and change_data.get("change_type") != "no_change":
                timecode = frame_index / fps
                event = {
                    "timecode": self._format_timecode(timecode),
                    "frame": frame_index,
                    "change_type": change_data.get("change_type"),
                    "before": change_data.get("before"),
                    "after": change_data.get("after"),
                    "confidence": change_data.get("confidence"),
                }
                print(json.dumps(event))
                last_emitted_frame = frame

            frame_index += 1

        cap.release()

    def _compare_frames(self, frame1: Any, frame2: Any) -> Optional[Dict[str, Any]]:
        """
        Compares two frames for visible changes using a vision model.
        """
        # This is a placeholder implementation.
        # I will need to implement the actual comparison logic using Replicate.
        # This will involve saving frames to temp files, encoding them,
        # and calling the Replicate API with a specific prompt.
        
        try:
            stitched_image = self._stitch_images(frame1, frame2)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                stitched_image.save(tmp_file.name)
                tmp_path = Path(tmp_file.name)

            image_data = self._encode_image_to_base64(tmp_path)
            
            prompt = """
You are a frame-by-frame video analysis expert. The image provided contains two video frames side-by-side. 
The left frame is 'before' and the right frame is 'after'. Your task is to identify the single most significant visible change between them. 
Focus on these categories: eyes (closed↔open, gaze), face makeup (shade/coverage/finish), hair (style/position), 
objects/props (appear/disappear/move), gestures/hand in frame, mouth (closed↔speaking), lighting, camera move, 
scene transition (incl. cross-dissolve). Ignore micro-jitter. 
Output a single JSON object with the following keys: 'change_type' (from the categories), 
'before' (a brief description of the state in the left frame), 
'after' (a brief description of the state in the right frame), 
and 'confidence' (a float 0-1). 
If no significant change is detected, the value for 'change_type' should be 'no_change', and the other fields can be empty.
"""
            response_text = self._call_replicate_api(image_data, prompt)
            
            # Clean up temp file
            tmp_path.unlink()

            return self._parse_json_from_text(response_text)

        except Exception:
            # In case of any error during comparison, assume no change.
            return None


    def _stitch_images(self, frame1: Any, frame2: Any) -> Image.Image:
        """Stitches two OpenCV frames side-by-side."""
        img1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

        dst = Image.new('RGB', (img1.width + img2.width, img1.height))
        dst.paste(img1, (0, 0))
        dst.paste(img2, (img1.width, 0))
        return dst

    def _encode_image_to_base64(self, image_path: Path) -> str:
        """Encode image file to base64."""
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/png;base64,{image_data}"
        except IOError as exc:
            raise VideoShotsError(f"Failed to read image file: {exc}")

    def _call_replicate_api(self, image_url: str, prompt: str) -> str:
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
            
            if hasattr(output, '__iter__') and not isinstance(output, str):
                return ''.join(str(item) for item in output)
            return str(output)
        except Exception as exc:
            raise VideoShotsError(f"Replicate API call failed: {exc}")

    def _parse_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extracts a JSON object from a string."""
        try:
            # Find the first '{' and the last '}' to extract the JSON part.
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                json_str = text[start:end+1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            return None
        return None

    @staticmethod
    def _format_timecode(seconds: float) -> str:
        """Formats seconds into HH:MM:SS.ms string."""
        import math
        milliseconds = int(round((seconds - math.floor(seconds)) * 1000))
        total_seconds = int(math.floor(seconds))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
