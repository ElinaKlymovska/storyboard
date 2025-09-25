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


def combine_to_pdf(image_paths: Sequence[Path], pdf_path: Path) -> None:
    if not image_paths:
        return
    images = []
    for idx, path in enumerate(image_paths):
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)

    head, *tail = images
    head.save(pdf_path, "PDF", resolution=100.0, save_all=True, append_images=tail)


def analyze_image_with_replicate(image_path: Path, prompt: str = "Опиши коротко що відбувається на цьому зображенні") -> str:
    """Аналізує зображення за допомогою Replicate vision model."""
    try:
        # Читаємо зображення та конвертуємо в base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Створюємо data URL
        image_url = f"data:image/png;base64,{image_data}"
        
        # Використовуємо Replicate API
        output = replicate.run(
            "yorickvp/llava-13b:01359160a4cff57c6b7d4dc625d0019d390c7c46f553714069f114b392f4a726",
            input={
                "image": image_url,
                "prompt": prompt,
                "max_tokens": 150,
                "temperature": 0.7
            }
        )
        
        # Replicate повертає generator, потрібно зібрати результат
        if hasattr(output, '__iter__') and not isinstance(output, str):
            result = ''.join(str(item) for item in output)
        else:
            result = str(output)
        
        return result.strip()
    except Exception as exc:
        return f"Помилка аналізу: {exc}"


def save_analysis_to_file(analysis: str, output_path: Path) -> None:
    """Зберігає аналіз у текстовому файлі."""
    # Створюємо директорію якщо не існує
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(analysis)


def upload_image_to_notion(notion: Client, image_path: Path) -> str:
    """Завантажує зображення в Notion та повертає URL."""
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        # Конвертуємо в base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        # Створюємо data URL
        data_url = f"data:image/png;base64,{image_b64}"
        
        # Завантажуємо в Notion
        response = notion.files.upload(
            file=data_url,
            name=image_path.name
        )
        
        return response["url"]
    except Exception as exc:
        print(f"⚠️ Помилка завантаження зображення в Notion: {exc}", file=sys.stderr)
        return ""


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
                    "type": "image",
                    "image": {
                        "type": "external",
                        "external": {"url": image_url}
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
    return parser


def resolve_output_dir(base_dir: Path, video_path: Path, custom_outdir: str | None) -> Path:
    if custom_outdir:
        out_dir = Path(custom_outdir)
    else:
        base_name = video_path.stem
        out_dir = base_dir / f"{base_name}_shots"
    out_dir.mkdir(parents=True, exist_ok=True)
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
        if not os.getenv("NOTION_API_TOKEN"):
            print("⚠️ Для Notion інтеграції потрібен NOTION_API_TOKEN в змінних середовища", file=sys.stderr)
            print("   Встановіть: export NOTION_API_TOKEN=your_token_here", file=sys.stderr)
            args.notion = None
        else:
            notion = Client(auth=os.getenv("NOTION_API_TOKEN"))

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
        output_path = output_dir / filename
        save_frame_image(framed, output_path, args.format)
        image_paths.append(output_path)

        # Аналіз зображення якщо увімкнено
        analysis_path = None
        if args.analyze:
            print(f"🔍 Аналізую кадр {tp.index}/{len(timepoints)}...", end=" ", flush=True)
            analysis = analyze_image_with_replicate(output_path, args.analysis_prompt)
            analysis_filename = f"{args.prefix}_{tp.index:05d}_f{tp.frame_index:05d}_t{time_tag}_analysis.txt"
            analysis_path = output_dir / analysis_filename
            save_analysis_to_file(analysis, analysis_path)
            print(f"✅ {analysis[:50]}...")
        
        analysis_paths.append(analysis_path or Path())

    cap.release()

    if args.pdf:
        pdf_path = Path(args.pdf)
        try:
            combine_to_pdf(image_paths, pdf_path)
            print(f"PDF збережено: {pdf_path}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"⚠️ Не вдалося створити PDF: {exc}", file=sys.stderr)

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

