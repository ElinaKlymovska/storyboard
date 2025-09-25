"""CLI утиліта для створення скріншотів з відео через заданий інтервал."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import base64
import io

import cv2
import numpy as np
from PIL import Image
import replicate
from notion_client import Client


CYRILLIC_S_LOWER = "с"
CYRILLIC_S_UPPER = "С"


class VideoShotsError(Exception):
    """Базове виключення для утиліти."""


@dataclass(frozen=True)
class TimePoint:
    index: int
    seconds: float
    frame_index: int


def parse_time_value(value: str, *, allow_zero: bool = False) -> float:
    """Повертає час у секундах з рядка користувача."""

    if value is None:
        raise VideoShotsError("Не вказано значення часу")

    normalized = (
        value.strip()
        .replace(" ", "")
        .replace(",", ".")
        .replace(CYRILLIC_S_LOWER, "s")
        .replace(CYRILLIC_S_UPPER, "s")
        .lower()
    )

    if not normalized:
        raise VideoShotsError("Порожнє значення часу")

    if ":" in normalized:
        parts = normalized.split(":")
        if len(parts) != 3:
            raise VideoShotsError(
                f"Некоректний формат часу '{value}'. Очікується HH:MM:SS.mmm"
            )
        hours_str, minutes_str, seconds_str = parts
        try:
            hours = int(hours_str)
            minutes = int(minutes_str)
            seconds = float(seconds_str)
        except ValueError as exc:
            raise VideoShotsError(f"Не вдалося розпарсити час '{value}'") from exc
        if minutes >= 60 or minutes < 0:
            raise VideoShotsError(f"Хвилини виходять за межі 0-59 у значенні '{value}'")
        if seconds < 0 or seconds >= 60:
            raise VideoShotsError(f"Секунди виходять за межі 0-59 у значенні '{value}'")
        total_seconds = hours * 3600 + minutes * 60 + seconds
    else:
        suffix = None
        if normalized.endswith("ms"):
            suffix = "ms"
            number_part = normalized[:-2]
        elif normalized.endswith("s"):
            suffix = "s"
            number_part = normalized[:-1]
        else:
            number_part = normalized

        if not number_part:
            raise VideoShotsError(f"Некоректний формат часу '{value}'")

        try:
            number = float(number_part)
        except ValueError as exc:
            raise VideoShotsError(f"Не вдалося розпарсити число у '{value}'") from exc

        if suffix == "ms":
            total_seconds = number / 1000.0
        else:
            total_seconds = number

    if total_seconds < 0 or (total_seconds == 0 and not allow_zero):
        raise VideoShotsError("Час має бути > 0")

    return total_seconds


def format_timestamp(seconds: float, *, precision: str) -> str:
    include_ms = precision == "ms"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if include_ms:
        sec_str = f"{secs:06.3f}" if hours else f"{secs:06.3f}"
    else:
        secs = int(round(secs))
        sec_str = f"{secs:02d}"

    if hours:
        if include_ms:
            return f"{hours:02d}:{minutes:02d}:{sec_str}"
        return f"{hours:02d}:{minutes:02d}:{sec_str}"
    return f"{minutes:02d}:{sec_str}"


def format_timestamp_for_name(seconds: float, *, precision: str) -> str:
    include_ms = precision == "ms"
    minutes_total = int(seconds // 60)
    secs_float = seconds % 60
    hours, minutes = divmod(minutes_total, 60)
    secs = int(secs_float)
    millis = int(round((secs_float - secs) * 1000))

    parts = []
    if hours:
        parts.append(f"{hours:02d}")
    parts.append(f"{minutes:02d}")
    parts.append(f"{secs:02d}")
    if include_ms:
        parts.append(f"{millis:03d}")
    return "-".join(parts)


def collect_timepoints(
    start: float,
    end: float,
    step: float,
    fps: float,
    frame_count: int,
) -> List[TimePoint]:
    if step <= 0:
        raise VideoShotsError("Інтервал має бути > 0")

    if end < start:
        raise VideoShotsError("Параметр --end має бути >= --start")

    duration = frame_count / fps if fps > 0 else 0
    hard_end = min(end, duration)
    if hard_end < start:
        raise VideoShotsError("Стартовий час перевищує тривалість відео")

    timepoints: List[TimePoint] = []
    index = 0
    max_iterations = 10_000_000
    epsilon = step / 10_000 if step else 1e-9

    while True:
        current_time = start + index * step
        if current_time > hard_end + epsilon:
            break

        frame_index = int(round(current_time * fps))
        frame_index = max(0, min(frame_index, frame_count - 1))

        timepoints.append(TimePoint(index=len(timepoints) + 1, seconds=current_time, frame_index=frame_index))

        index += 1
        if index > max_iterations:
            raise VideoShotsError(
                "Занадто велика кількість ітерацій. Перевірте інтервал або використайте --end/--start"
            )

    return timepoints


def apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 0:
        return frame
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise VideoShotsError(f"Непідтримуваний кут обертання: {rotation}")


def render_label_panel(
    frame: np.ndarray,
    label_lines: Sequence[str],
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.8,
    font_thickness: int = 2,
    padding: int = 12,
    panel_bg_color: Tuple[int, int, int] = (0, 0, 0),
    panel_alpha: float = 0.6,
    text_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    height, width = frame.shape[:2]

    # Розрахунок ширини панелі за довжиною тексту
    max_text_width = 0
    text_height_total = 0
    line_height = 0
    for line in label_lines:
        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, font_thickness)
        max_text_width = max(max_text_width, text_width)
        line_height = max(line_height, text_height + baseline)
        text_height_total += text_height + baseline

    panel_width = max_text_width + padding * 2
    if panel_width < int(width * 0.18):
        panel_width = int(width * 0.18)

    new_width = width + panel_width
    result = np.zeros((height, new_width, 3), dtype=np.uint8)
    result[:, :width] = frame

    overlay = result.copy()
    overlay[:, width:] = panel_bg_color
    cv2.addWeighted(overlay, panel_alpha, result, 1 - panel_alpha, 0, result)

    # Розміщуємо текст у верхній частині панелі
    y = padding + line_height
    x = width + padding
    for line in label_lines:
        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, font_thickness)
        cv2.putText(result, line, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        y += line_height + padding // 2

    return result


def save_frame_image(
    frame: np.ndarray,
    output_path: Path,
    image_format: str,
) -> None:
    params = []
    if image_format == "jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

    success = cv2.imwrite(str(output_path), frame, params)
    if not success:
        raise VideoShotsError(f"Не вдалося зберегти файл '{output_path}'")


def combine_to_pdf(image_paths: Sequence[Path], analyses: List[Path], timepoints: List[TimePoint], pdf_path: Path) -> None:
    """Об'єднує зображення з аналізом в PDF файл."""
    if not image_paths:
        return
    
    try:
        import json
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        
        # Створюємо PDF документ
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        story = []
        
        # Стилі
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.darkblue
        )
        
        frame_style = ParagraphStyle(
            'FrameTitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        analysis_style = ParagraphStyle(
            'Analysis',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            leftIndent=20,
            textColor=colors.darkgreen
        )
        
        # Заголовок
        story.append(Paragraph("🎬 AI-Generated Storyboard Analysis", title_style))
        story.append(Spacer(1, 12))
        
        # Додаємо кожен кадр з аналізом
        for i, (img_path, analysis_path, tp) in enumerate(zip(image_paths, analyses, timepoints), 1):
            time_label = format_timestamp(tp.seconds, precision='ms')
            
            # Заголовок кадру
            story.append(Paragraph(f"Frame {i} - {time_label}", frame_style))
            
            # Зображення
            try:
                img = RLImage(str(img_path), width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 12))
            except Exception as e:
                story.append(Paragraph(f"Error loading image: {e}", styles['Normal']))
            
            # Аналіз якщо є
            if analysis_path and analysis_path.exists():
                try:
                    with open(analysis_path, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                    
                    # Scene Description
                    scene_desc = analysis_data.get('scene_description', 'No description available')
                    story.append(Paragraph(f"<b>📝 Scene Description:</b>", analysis_style))
                    story.append(Paragraph(scene_desc, analysis_style))
                    story.append(Spacer(1, 6))
                    
                    # Visual Elements
                    if 'visual_elements' in analysis_data:
                        ve = analysis_data['visual_elements']
                        story.append(Paragraph(f"<b>👁️ Visual Elements:</b>", analysis_style))
                        if ve.get('characters'):
                            story.append(Paragraph(f"Characters: {', '.join(ve['characters'])}", analysis_style))
                        if ve.get('objects'):
                            story.append(Paragraph(f"Objects: {', '.join(ve['objects'])}", analysis_style))
                        if ve.get('background'):
                            story.append(Paragraph(f"Background: {ve['background']}", analysis_style))
                        if ve.get('lighting'):
                            story.append(Paragraph(f"Lighting: {ve['lighting']}", analysis_style))
                        story.append(Spacer(1, 6))
                    
                    # Composition
                    if 'composition' in analysis_data:
                        comp = analysis_data['composition']
                        story.append(Paragraph(f"<b>🎬 Composition:</b>", analysis_style))
                        if comp.get('shot_type'):
                            story.append(Paragraph(f"Shot Type: {comp['shot_type']}", analysis_style))
                        if comp.get('camera_angle'):
                            story.append(Paragraph(f"Camera Angle: {comp['camera_angle']}", analysis_style))
                        if comp.get('framing'):
                            story.append(Paragraph(f"Framing: {comp['framing']}", analysis_style))
                        story.append(Spacer(1, 6))
                    
                    # Storyboard Suitability
                    if 'storyboard_suitability' in analysis_data:
                        ss = analysis_data['storyboard_suitability']
                        score = ss.get('score', 0)
                        score_color = colors.green if score >= 7 else colors.orange if score >= 4 else colors.red
                        story.append(Paragraph(f"<b>⭐ Storyboard Suitability: <font color='{score_color}'>{score}/10</font></b>", analysis_style))
                        if ss.get('reasoning'):
                            story.append(Paragraph(f"Reasoning: {ss['reasoning']}", analysis_style))
                        if ss.get('key_moments'):
                            story.append(Paragraph(f"Key Moments: {', '.join(ss['key_moments'])}", analysis_style))
                        if ss.get('transition_potential'):
                            story.append(Paragraph(f"Transition Potential: {ss['transition_potential']}", analysis_style))
                        story.append(Spacer(1, 6))
                    
                    # Production Notes
                    if 'production_notes' in analysis_data:
                        pn = analysis_data['production_notes']
                        story.append(Paragraph(f"<b>🎭 Production Notes:</b>", analysis_style))
                        if pn.get('emotions'):
                            story.append(Paragraph(f"Emotions: {', '.join(pn['emotions'])}", analysis_style))
                        if pn.get('action_level'):
                            story.append(Paragraph(f"Action Level: {pn['action_level']}", analysis_style))
                        if pn.get('makeup_visible') is not None:
                            story.append(Paragraph(f"Makeup Visible: {pn['makeup_visible']}", analysis_style))
                        if pn.get('technical_quality'):
                            story.append(Paragraph(f"Technical Quality: {pn['technical_quality']}", analysis_style))
                    
                except Exception as e:
                    story.append(Paragraph(f"Error reading analysis: {e}", styles['Normal']))
            else:
                story.append(Paragraph("No analysis available for this frame.", analysis_style))
            
            # Розділювач між кадрами (крім останнього)
            if i < len(image_paths):
                story.append(PageBreak())
        
        # Генеруємо PDF
        doc.build(story)
        
    except ImportError:
        # Fallback до простого PDF якщо reportlab не встановлено
        images = []
        for idx, path in enumerate(image_paths):
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)

        head, *tail = images
        head.save(pdf_path, "PDF", resolution=100.0, save_all=True, append_images=tail)


def parse_audio_transcript(audio_file_path: Path) -> dict:
    """Парсить аудіо транскрипт та повертає словник з таймкодами."""
    try:
        if not audio_file_path.exists():
            return {}
        
        with open(audio_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Парсимо таймкоди та текст
        import re
        pattern = r'\[(\d{2}:\d{2}:\d{2}:\d{2}) - (\d{2}:\d{2}:\d{2}:\d{2})\]\s*(.+)'
        matches = re.findall(pattern, content)
        
        transcript_data = {}
        for start_time, end_time, text in matches:
            # Конвертуємо таймкод в секунди
            start_seconds = parse_timecode_to_seconds(start_time)
            end_seconds = parse_timecode_to_seconds(end_time)
            
            # Зберігаємо для кожного кадру
            transcript_data[start_seconds] = {
                'start': start_time,
                'end': end_time,
                'text': text.strip()
            }
        
        return transcript_data
        
    except Exception as exc:
        print(f"⚠️ Помилка парсингу аудіо файлу: {exc}", file=sys.stderr)
        return {}


def parse_timecode_to_seconds(timecode: str) -> float:
    """Конвертує таймкод HH:MM:SS:FF в секунди."""
    try:
        parts = timecode.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        frames = int(parts[3])
        
        # Припускаємо 30 FPS
        total_seconds = hours * 3600 + minutes * 60 + seconds + frames / 30.0
        return total_seconds
    except:
        return 0.0


def get_vo_for_timestamp(transcript_data: dict, timestamp: float) -> str:
    """Знаходить VO/Sound для заданого таймкоду."""
    if not transcript_data:
        return ""
    
    # Шукаємо найближчий таймкод
    closest_time = None
    min_diff = float('inf')
    
    for time_key in transcript_data.keys():
        diff = abs(time_key - timestamp)
        if diff < min_diff:
            min_diff = diff
            closest_time = time_key
    
    # Якщо знайшли близький таймкод (в межах 2 секунд)
    if closest_time is not None and min_diff <= 2.0:
        return transcript_data[closest_time]['text']
    
    return ""


def create_structured_analysis_from_text(text: str) -> dict:
    """Створює структурований аналіз з текстового опису."""
    import re
    
    # Визначаємо тип кадру
    shot_type = "close-up"
    if "wide" in text.lower() or "full" in text.lower():
        shot_type = "wide"
    elif "medium" in text.lower():
        shot_type = "medium"
    
    # Визначаємо емоції
    emotions = []
    if any(word in text.lower() for word in ["happy", "smiling", "joy", "excited"]):
        emotions.append("happy")
    if any(word in text.lower() for word in ["serious", "focused", "concentrated"]):
        emotions.append("serious")
    if any(word in text.lower() for word in ["relaxed", "calm", "peaceful"]):
        emotions.append("relaxed")
    if any(word in text.lower() for word in ["dramatic", "intense", "dramatic"]):
        emotions.append("dramatic")
    
    # Визначаємо рівень дії
    action_level = "static"
    if any(word in text.lower() for word in ["moving", "action", "motion", "dynamic"]):
        action_level = "medium"
    if any(word in text.lower() for word in ["fast", "quick", "rapid", "intense"]):
        action_level = "high"
    
    # Визначаємо оцінку придатності
    score = 5
    if "close-up" in text.lower() and "face" in text.lower():
        score = 7
    if "background" in text.lower() and "black" in text.lower():
        score += 1
    if emotions:
        score += 1
    
    # Визначаємо персонажів
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


def analyze_image_with_replicate(image_path: Path, prompt: str = None) -> dict:
    """Аналізує зображення за допомогою Replicate vision model та повертає структурований результат."""
    if prompt is None:
        prompt = """Analyze this video frame for storyboard creation. Provide ONLY a valid JSON response with this exact structure:

{
  "scene_description": "Brief description of what's happening",
  "visual_elements": {
    "characters": ["person", "character"],
    "objects": ["object1", "object2"],
    "background": "background description",
    "lighting": "lighting description"
  },
  "composition": {
    "shot_type": "close-up",
    "camera_angle": "eye-level",
    "framing": "framing description"
  },
  "storyboard_suitability": {
    "score": 7,
    "reasoning": "explanation",
    "key_moments": ["moment1", "moment2"],
    "transition_potential": "good"
  },
  "production_notes": {
    "makeup_visible": true,
    "emotions": ["emotion1", "emotion2"],
    "action_level": "static",
    "technical_quality": "good"
  }
}

Respond with ONLY the JSON, no other text."""
    
    try:
        # Читаємо зображення та конвертуємо в base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Створюємо data URL
        image_url = f"data:image/png;base64,{image_data}"
        
        # Використовуємо Replicate API з новою моделлю
        output = replicate.run(
            "yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb",
            input={
                "image": image_url,
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0.3
            }
        )
        
        # Replicate повертає generator, потрібно зібрати результат
        if hasattr(output, '__iter__') and not isinstance(output, str):
            result = ''.join(str(item) for item in output)
        else:
            result = str(output)
        
        # Спробуємо парсити JSON, якщо не вдається - створимо структурований аналіз
        try:
            import json
            import re
            
            # Спробуємо знайти JSON у відповіді
            json_match = re.search(r'\{.*\}', result.strip(), re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_result = json.loads(json_str)
                return parsed_result
            else:
                # Якщо JSON не знайдено, створимо структурований аналіз з тексту
                return create_structured_analysis_from_text(result.strip())
        except json.JSONDecodeError:
            return create_structured_analysis_from_text(result.strip())
            
    except Exception as exc:
        return {
            "error": f"Analysis failed: {exc}",
            "scene_description": "Analysis unavailable",
            "storyboard_suitability": {"score": 0, "reasoning": "Analysis failed"}
        }


def save_analysis_to_file(analysis: dict, output_path: Path) -> None:
    """Зберігає структурований аналіз у JSON файлі."""
    # Створюємо директорію якщо не існує
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)


def upload_image_to_notion(notion: Client, image_path: Path) -> str:
    """Завантажує зображення в Notion та повертає URL."""
    try:
        # Для теперішнього часу повертаємо placeholder
        # В майбутньому можна інтегрувати з зовнішнім хостингом зображень
        return f"📷 {image_path.name} (зображення збережено локально)"
    except Exception as exc:
        print(f"⚠️ Помилка завантаження зображення в Notion: {exc}", file=sys.stderr)
        return ""


def create_html_storyboard(
    video_name: str,
    screenshots: List[Path],
    analyses: List[Path],
    timepoints: List[TimePoint],
    output_path: Path
) -> None:
    """Створює HTML файл з storyboard."""
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Storyboard: {video_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f8f9fa;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .frame {{
            background: white;
            margin-bottom: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .frame-header {{
            background: #2563eb;
            color: white;
            padding: 15px 20px;
            font-weight: 600;
            font-size: 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .score-badge {{
            background: rgba(255,255,255,0.2);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
        }}
        .frame-content {{
            padding: 20px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .screenshot {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            margin-bottom: 15px;
        }}
        .analysis {{
            background: #f0f9ff;
            border-left: 4px solid #0ea5e9;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }}
        .analysis-section {{
            margin-bottom: 15px;
        }}
        .analysis-title {{
            font-weight: 600;
            color: #1e40af;
            margin-bottom: 5px;
        }}
        .metadata {{
            color: #6b7280;
            font-size: 14px;
            margin-top: 10px;
        }}
        .divider {{
            height: 1px;
            background: #e5e7eb;
            margin: 20px 0;
        }}
        .score-high {{ color: #10b981; }}
        .score-medium {{ color: #f59e0b; }}
        .score-low {{ color: #ef4444; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 Storyboard: {video_name}</h1>
        <p>AI-Generated Storyboard Analysis</p>
        <p>Frames: {len(screenshots)} | Created: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""

    for i, (screenshot, analysis, tp) in enumerate(zip(screenshots, analyses, timepoints), 1):
        time_label = format_timestamp(tp.seconds, precision='ms')
        
        # Читаємо структурований аналіз
        analysis_data = {}
        score = 5
        if analysis and analysis.exists():
            try:
                import json
                with open(analysis, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
                score = analysis_data.get('storyboard_suitability', {}).get('score', 5)
            except:
                analysis_data = {"scene_description": "Analysis unavailable"}
        
        # Копіюємо зображення в exports папку для HTML
        screenshot_name = f"frame_{i:03d}_{screenshot.name}"
        screenshot_dest = output_path.parent / screenshot_name
        import shutil
        shutil.copy2(screenshot, screenshot_dest)
        
        # Визначаємо клас для оцінки
        score_class = "score-high" if score >= 7 else "score-medium" if score >= 4 else "score-low"
        
        html_content += f"""
    <div class="frame">
        <div class="frame-header">
            Frame {i} - {time_label}
            <span class="score-badge {score_class}">Score: {score}/10</span>
        </div>
        <div class="frame-content">
            <div>
                <img src="{screenshot_name}" alt="Frame {i}" class="screenshot">
                <div class="metadata">
                    Time: {time_label} | Frame: #{tp.frame_index} | Index: {tp.index}
                </div>
            </div>
            <div class="analysis">
"""
        
        # Додаємо структурований аналіз
        if analysis_data:
            html_content += f"""
                <div class="analysis-section">
                    <div class="analysis-title">📝 Scene Description</div>
                    <div>{analysis_data.get('scene_description', 'N/A')}</div>
                </div>
"""
            
            if 'visual_elements' in analysis_data:
                ve = analysis_data['visual_elements']
                html_content += f"""
                <div class="analysis-section">
                    <div class="analysis-title">👁️ Visual Elements</div>
                    <div><strong>Characters:</strong> {', '.join(ve.get('characters', ['N/A']))}</div>
                    <div><strong>Objects:</strong> {', '.join(ve.get('objects', ['N/A']))}</div>
                    <div><strong>Background:</strong> {ve.get('background', 'N/A')}</div>
                    <div><strong>Lighting:</strong> {ve.get('lighting', 'N/A')}</div>
                </div>
"""
            
            if 'composition' in analysis_data:
                comp = analysis_data['composition']
                html_content += f"""
                <div class="analysis-section">
                    <div class="analysis-title">🎬 Composition</div>
                    <div><strong>Shot Type:</strong> {comp.get('shot_type', 'N/A')}</div>
                    <div><strong>Camera Angle:</strong> {comp.get('camera_angle', 'N/A')}</div>
                    <div><strong>Framing:</strong> {comp.get('framing', 'N/A')}</div>
                </div>
"""
            
            if 'storyboard_suitability' in analysis_data:
                ss = analysis_data['storyboard_suitability']
                html_content += f"""
                <div class="analysis-section">
                    <div class="analysis-title">⭐ Storyboard Suitability</div>
                    <div><strong>Score:</strong> {ss.get('score', 'N/A')}/10</div>
                    <div><strong>Reasoning:</strong> {ss.get('reasoning', 'N/A')}</div>
                    <div><strong>Key Moments:</strong> {', '.join(ss.get('key_moments', ['N/A']))}</div>
                    <div><strong>Transition Potential:</strong> {ss.get('transition_potential', 'N/A')}</div>
                </div>
"""
        
        html_content += """
            </div>
        </div>
    </div>
"""
        
        if i < len(screenshots):
            html_content += '<div class="divider"></div>'

    html_content += """
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def create_editing_table(
    video_name: str,
    screenshots: List[Path],
    analyses: List[Path],
    timepoints: List[TimePoint],
    output_path: Path,
    transcript_data: dict = None
) -> None:
    """Створює таблицю для монтажу на основі аналізу кадрів."""
    import json
    
    # Збираємо дані з аналізів
    editing_data = []
    
    for i, (screenshot, analysis, tp) in enumerate(zip(screenshots, analyses, timepoints), 1):
        time_label = format_timestamp(tp.seconds, precision='ms')
        
        # Читаємо аналіз
        analysis_data = {}
        if analysis and analysis.exists():
            try:
                with open(analysis, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
            except:
                analysis_data = {"scene_description": "Analysis unavailable"}
        
        # Визначаємо тип контенту на основі аналізу
        content_type = "Image"
        if analysis_data.get('production_notes', {}).get('action_level') in ['medium', 'high']:
            content_type = "Video"
        elif analysis_data.get('storyboard_suitability', {}).get('transition_potential', '').lower() in ['high', 'excellent']:
            content_type = "Montage"
        
        # Генеруємо опис для генеративних моделей
        scene_desc = analysis_data.get('scene_description', 'Scene analysis unavailable')
        visual_elements = analysis_data.get('visual_elements', {})
        characters = ', '.join(visual_elements.get('characters', ['character']))
        objects = ', '.join(visual_elements.get('objects', ['objects']))
        background = visual_elements.get('background', 'background')
        lighting = visual_elements.get('lighting', 'lighting')
        
        # Створюємо детальний опис для генерації
        generation_prompt = f"""Generate {content_type.lower()} content: {scene_desc}. 
Characters: {characters}. Objects: {objects}. 
Background: {background}. Lighting: {lighting}.
Maintain character identity and visual consistency."""
        
        # Визначаємо VO/Sound
        vo_sound = ""
        
        # Спочатку пробуємо знайти в транскрипті
        if transcript_data:
            vo_sound = get_vo_for_timestamp(transcript_data, tp.seconds)
        
        # Якщо не знайшли в транскрипті, використовуємо аналіз
        if not vo_sound:
            emotions = analysis_data.get('production_notes', {}).get('emotions', [])
            if emotions:
                if 'happy' in emotions or 'excited' in emotions:
                    vo_sound = "Upbeat dialogue or music"
                elif 'serious' in emotions or 'focused' in emotions:
                    vo_sound = "Professional narration"
                elif 'dramatic' in emotions:
                    vo_sound = "Dramatic music or sound effects"
        
        editing_data.append({
            'step': i,
            'timestamp': time_label,
            'type': content_type,
            'description': generation_prompt,
            'vo_sound': vo_sound,
            'score': analysis_data.get('storyboard_suitability', {}).get('score', 5),
            'key_moments': analysis_data.get('storyboard_suitability', {}).get('key_moments', [])
        })
    
    # Створюємо HTML таблицю
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Editing Table: {video_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background: #f8f9fa;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .table-container {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background: #2563eb;
            color: white;
            padding: 15px 10px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 15px 10px;
            border-bottom: 1px solid #e5e7eb;
            vertical-align: top;
        }}
        tr:nth-child(even) {{
            background: #f9fafb;
        }}
        .step {{
            font-weight: 600;
            color: #2563eb;
        }}
        .type-image {{ color: #10b981; }}
        .type-video {{ color: #f59e0b; }}
        .type-montage {{ color: #8b5cf6; }}
        .description {{
            max-width: 400px;
            word-wrap: break-word;
        }}
        .score-high {{ color: #10b981; font-weight: 600; }}
        .score-medium {{ color: #f59e0b; font-weight: 600; }}
        .score-low {{ color: #ef4444; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 Editing Table: {video_name}</h1>
        <p>AI-Generated Editing Guide for Generative Models</p>
        <p>Total Steps: {len(editing_data)} | Created: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Step</th>
                    <th>Timestamp</th>
                    <th>Type</th>
                    <th>Description</th>
                    <th>VO / Sound</th>
                    <th>Score</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for item in editing_data:
        score_class = "score-high" if item['score'] >= 7 else "score-medium" if item['score'] >= 4 else "score-low"
        type_class = f"type-{item['type'].lower()}"
        
        html_content += f"""
                <tr>
                    <td class="step">{item['step']}</td>
                    <td>{item['timestamp']}</td>
                    <td class="{type_class}">{item['type']}</td>
                    <td class="description">{item['description']}</td>
                    <td>{item['vo_sound']}</td>
                    <td class="{score_class}">{item['score']}/10</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def create_notion_storyboard(
    notion: Client,
    parent_page_id: str,
    video_name: str,
    screenshots: List[Path],
    analyses: List[Path],
    timepoints: List[TimePoint]
) -> str:
    """Створює Notion сторінку з storyboard."""
    try:
        # Створюємо головну сторінку storyboard
        page_title = f"Storyboard: {video_name}"
        
        page = notion.pages.create(
            parent={"page_id": parent_page_id},
            properties={
                "title": {
                    "title": [{"text": {"content": page_title}}]
                }
            }
        )
        
        page_id = page["id"]
        
        # Додаємо опис
        notion.blocks.children.append(
            block_id=page_id,
            children=[
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": f"Автоматично створений storyboard для відео {video_name}\n"
                                             f"Кількість кадрів: {len(screenshots)}\n"
                                             f"Створено: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                }
                            }
                        ]
                    }
                }
            ]
        )
        
        # Додаємо кожен кадр
        for i, (screenshot, analysis, tp) in enumerate(zip(screenshots, analyses, timepoints), 1):
            # Завантажуємо зображення
            image_url = upload_image_to_notion(notion, screenshot)
            
            if not image_url:
                continue
            
            # Читаємо аналіз
            analysis_text = ""
            if analysis.exists():
                with open(analysis, 'r', encoding='utf-8') as f:
                    analysis_text = f.read().strip()
            
            # Створюємо блок для кадру
            frame_blocks = [
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": f"Кадр {i} - {format_timestamp(tp.seconds, precision='ms')}"
                                }
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": image_url}
                            }
                        ]
                    }
                }
            ]
            
            # Додаємо аналіз якщо є
            if analysis_text:
                frame_blocks.append({
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": analysis_text}
                            }
                        ],
                        "icon": {"emoji": "🤖"}
                    }
                })
            
            # Додаємо метадані
            frame_blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": f"Час: {format_timestamp(tp.seconds, precision='ms')} | "
                                         f"Кадр: #{tp.frame_index} | "
                                         f"Індекс: {tp.index}"
                            }
                        }
                    ]
                }
            })
            
            # Додаємо розділювач
            frame_blocks.append({
                "object": "block",
                "type": "divider",
                "divider": {}
            })
            
            # Додаємо блоки до сторінки
            notion.blocks.children.append(
                block_id=page_id,
                children=frame_blocks
            )
        
        return page_id
        
    except Exception as exc:
        print(f"⚠️ Помилка створення Notion сторінки: {exc}", file=sys.stderr)
        return ""


def estimate_total_frames(duration: float, interval: float) -> int:
    if interval <= 0:
        return 0
    return int(math.floor(duration / interval + 1e-9)) + 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-shots",
        description="Створення скріншотів з відео по часовому інтервалу",
    )
    parser.add_argument("video_path", help="Шлях до файлу відео")
    parser.add_argument(
        "-n",
        "--interval",
        required=True,
        help="Часовий інтервал між скріншотами (приклади: 0.5s, 200ms, 00:00:03.5)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "jpg"],
        default="png",
        help="Формат зображень (дефолт: png)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Вихідна папка для скріншотів",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="shot",
        help="Префікс імен файлів (дефолт: shot)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="0",
        help="Початковий час (дефолт: 0)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Кінцевий час (дефолт: тривалість відео)",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        choices=[0, 90, 180, 270],
        default=0,
        help="Примусове обертання кадрів (0/90/180/270)",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Зберегти усі скріншоти у PDF-файл (вказати шлях)",
    )
    parser.add_argument(
        "--time-precision",
        choices=["auto", "ms", "sec"],
        default="auto",
        help="Формат часу у підписі (auto, ms, sec)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Максимальна кількість скріншотів (для захисту від занадто малого інтервалу)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Аналізувати кожен скріншот за допомогою AI vision model",
    )
    parser.add_argument(
        "--analysis-prompt",
        type=str,
        default="Опиши коротко що відбувається на цьому зображенні",
        help="Промпт для аналізу зображень (дефолт: 'Опиши коротко що відбувається на цьому зображенні')",
    )
    parser.add_argument(
        "--notion",
        type=str,
        default=None,
        help="Створити Notion сторінку з storyboard (вказати ID батьківської сторінки)",
    )
    parser.add_argument(
        "--html",
        type=str,
        default=None,
        help="Створити HTML файл з storyboard (вказати шлях до файлу)",
    )
    parser.add_argument(
        "--editing-table",
        type=str,
        default=None,
        help="Створити таблицю для монтажу (вказати шлях до файлу)",
    )
    parser.add_argument(
        "--audio-transcript",
        type=str,
        default=None,
        help="Шлях до аудіо транскрипту для VO/Sound",
    )
    return parser


def resolve_output_dir(base_dir: Path, video_path: Path, custom_outdir: str | None) -> Path:
    if custom_outdir:
        out_dir = Path(custom_outdir)
    else:
        base_name = video_path.stem
        # Створюємо папку з timestamp для кожної сесії
        timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = base_dir / f"{base_name}_{timestamp}"
    
    # Створюємо основну папку та підпапки
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "screenshots").mkdir(exist_ok=True)
    (out_dir / "analysis").mkdir(exist_ok=True)
    (out_dir / "exports").mkdir(exist_ok=True)
    
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    video_path = Path(args.video_path)
    if not video_path.exists():
        parser.error(f"Файл відео '{video_path}' не існує")

    try:
        interval_seconds = parse_time_value(args.interval)
    except VideoShotsError as exc:
        parser.error(str(exc))

    try:
        start_seconds = parse_time_value(args.start, allow_zero=True)
    except VideoShotsError as exc:
        parser.error(f"Помилка у --start: {exc}")

    end_seconds = None
    if args.end is not None:
        try:
            end_seconds = parse_time_value(args.end, allow_zero=True)
        except VideoShotsError as exc:
            parser.error(f"Помилка у --end: {exc}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        parser.error(f"Не вдалося відкрити відео '{video_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        parser.error("Неможливо отримати FPS відео")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        parser.error("Неможливо визначити кількість кадрів")

    duration = frame_count / fps
    if end_seconds is None:
        end_seconds = duration

    if start_seconds > duration:
        cap.release()
        parser.error("Початковий час перевищує тривалість відео")

    estimated_total = estimate_total_frames(end_seconds - start_seconds, interval_seconds)
    if args.max_frames is not None and estimated_total > args.max_frames:
        cap.release()
        parser.error(
            f"Очікувана кількість кадрів ({estimated_total}) перевищує --max-frames"
        )

    if estimated_total > 5000:
        print(
            f"⚠️ Попередження: буде створено {estimated_total} скрінів."
            " Розгляньте обмеження --start/--end або збільшення інтервалу.",
            file=sys.stderr,
        )

    theoretical_min_step = 1.0 / fps
    if interval_seconds < theoretical_min_step:
        print(
            f"ℹ️ Інтервал {interval_seconds:.6f}s менший за 1/FPS ({theoretical_min_step:.6f}s)."
            " Можливе повторення кадрів.",
            file=sys.stderr,
        )

    try:
        timepoints = collect_timepoints(start_seconds, end_seconds, interval_seconds, fps, frame_count)
    except VideoShotsError as exc:
        cap.release()
        parser.error(str(exc))

    output_dir = resolve_output_dir(Path.cwd() / "output", video_path, args.outdir)

    actual_precision = args.time_precision
    if actual_precision == "auto":
        actual_precision = "ms" if interval_seconds < 1.0 else "sec"

    image_paths: List[Path] = []
    analysis_paths: List[Path] = []
    duplicates = 0
    previous_frame_idx = None

    # Перевіряємо наявність Replicate API токена
    if args.analyze:
        if not os.getenv("REPLICATE_API_TOKEN"):
            print("⚠️ Для аналізу зображень потрібен REPLICATE_API_TOKEN в змінних середовища", file=sys.stderr)
            print("   Встановіть: export REPLICATE_API_TOKEN=your_token_here", file=sys.stderr)
            args.analyze = False

    # Перевіряємо наявність Notion API токена
    notion = None
    if args.notion:
        notion_token = os.getenv("NOTION_API_TOKEN")
        if not notion_token:
            print("⚠️ Для Notion інтеграції потрібен NOTION_API_TOKEN в змінних середовища", file=sys.stderr)
            print("   Встановіть: export NOTION_API_TOKEN=your_token_here", file=sys.stderr)
            args.notion = None
        else:
            try:
                notion = Client(auth=notion_token)
                # Тестуємо підключення
                notion.users.me()
                print("✅ Notion API підключено успішно")
            except Exception as exc:
                print(f"⚠️ Помилка підключення до Notion: {exc}", file=sys.stderr)
                args.notion = None

    for tp in timepoints:
        if previous_frame_idx is not None and tp.frame_index == previous_frame_idx:
            duplicates += 1
        previous_frame_idx = tp.frame_index

        cap.set(cv2.CAP_PROP_POS_FRAMES, tp.frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"⚠️ Пропущено кадр #{tp.frame_index}: не вдалося зчитати", file=sys.stderr)
            continue

        frame = apply_rotation(frame, args.rotate)

        time_label = format_timestamp(tp.seconds, precision=actual_precision)
        label_lines = [f"Time {time_label}"]
        framed = render_label_panel(frame, label_lines)

        time_tag = format_timestamp_for_name(tp.seconds, precision=actual_precision)
        filename = f"{args.prefix}_{tp.index:05d}_f{tp.frame_index:05d}_t{time_tag}.{args.format}"
        output_path = output_dir / "screenshots" / filename
        save_frame_image(framed, output_path, args.format)
        image_paths.append(output_path)

        # Аналіз зображення якщо увімкнено
        analysis_path = None
        if args.analyze:
            print(f"🔍 Аналізую кадр {tp.index}/{len(timepoints)}...", end=" ", flush=True)
            analysis_data = analyze_image_with_replicate(output_path, args.analysis_prompt)
            analysis_filename = f"{args.prefix}_{tp.index:05d}_f{tp.frame_index:05d}_t{time_tag}_analysis.json"
            analysis_path = output_dir / "analysis" / analysis_filename
            save_analysis_to_file(analysis_data, analysis_path)
            scene_desc = analysis_data.get('scene_description', 'Analysis unavailable')
            print(f"✅ {scene_desc[:50]}...")
        
        analysis_paths.append(analysis_path or Path())

    cap.release()

    if args.pdf:
        if args.pdf.startswith('/') or ':' in args.pdf:
            # Абсолютний шлях
            pdf_path = Path(args.pdf)
        else:
            # Відносний шлях - зберігаємо в exports папці
            pdf_path = output_dir / "exports" / args.pdf
        try:
            combine_to_pdf(image_paths, analysis_paths, timepoints, pdf_path)
            print(f"PDF збережено: {pdf_path}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"⚠️ Не вдалося створити PDF: {exc}", file=sys.stderr)

    # Створюємо HTML файл якщо увімкнено
    if args.html:
        print("🌐 Створюю HTML storyboard...", end=" ", flush=True)
        if args.html.startswith('/') or ':' in args.html:
            # Абсолютний шлях
            html_path = Path(args.html)
        else:
            # Відносний шлях - зберігаємо в exports папці
            html_path = output_dir / "exports" / args.html
        create_html_storyboard(
            video_path.stem,
            image_paths,
            analysis_paths,
            timepoints,
            html_path
        )
        print(f"✅ HTML storyboard збережено: {html_path}")

    # Парсимо аудіо транскрипт якщо вказано
    transcript_data = {}
    if args.audio_transcript:
        print("🎵 Парсую аудіо транскрипт...", end=" ", flush=True)
        transcript_data = parse_audio_transcript(Path(args.audio_transcript))
        print(f"✅ Знайдено {len(transcript_data)} сегментів")

    # Створюємо таблицю монтажу якщо увімкнено
    if args.editing_table:
        print("📊 Створюю таблицю монтажу...", end=" ", flush=True)
        if args.editing_table.startswith('/') or ':' in args.editing_table:
            # Абсолютний шлях
            table_path = Path(args.editing_table)
        else:
            # Відносний шлях - зберігаємо в exports папці
            table_path = output_dir / "exports" / args.editing_table
        create_editing_table(
            video_path.stem,
            image_paths,
            analysis_paths,
            timepoints,
            table_path,
            transcript_data
        )
        print(f"✅ Таблиця монтажу збережена: {table_path}")

    # Створюємо Notion сторінку якщо увімкнено
    if args.notion and notion:
        print("📝 Створюю Notion сторінку...", end=" ", flush=True)
        notion_page_id = create_notion_storyboard(
            notion, 
            args.notion, 
            video_path.stem, 
            image_paths, 
            analysis_paths, 
            timepoints
        )
        if notion_page_id:
            print(f"✅ Notion сторінка створена: https://notion.so/{notion_page_id.replace('-', '')}")
        else:
            print("❌ Помилка створення Notion сторінки")

    print(f"Готово. Збережено {len(image_paths)} файлів у '{output_dir}'.")
    if duplicates:
        print(f"ℹ️ {duplicates} таймштампів вказали на однаковий кадр (обмеження FPS).")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VideoShotsError as exc:
        print(f"Помилка: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

