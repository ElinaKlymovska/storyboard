"""Експорт скріншотів та аналізу в Notion."""

import os
import sys
from pathlib import Path
from typing import List

try:
    from notion_client import Client
    NOTION_AVAILABLE = True
except ImportError:
    NOTION_AVAILABLE = False
    Client = None

from video_shots.core.models import TimePoint
from video_shots.utils.time_utils import format_timestamp
from video_shots.utils.github_utils import create_storyboard_gist, get_gist_instructions
from video_shots.utils.image_hosting import (
    upload_to_imgur, 
    upload_to_imgbb, 
    upload_to_google_drive,
    encode_image_to_base64,
    create_public_html_with_images
)


def upload_image_to_notion(notion, image_path: Path, hosting_method: str = "imgur") -> str:
    """Завантажує зображення в публічний хостинг та повертає URL для Notion."""
    try:
        # Спробуємо завантажити на публічний хостинг
        public_url = None
        
        if hosting_method == "imgur":
            public_url = upload_to_imgur(image_path)
        elif hosting_method == "imgbb":
            public_url = upload_to_imgbb(image_path)
        elif hosting_method == "googledrive":
            public_url = upload_to_google_drive(image_path)
        elif hosting_method == "base64":
            public_url = encode_image_to_base64(image_path)
        
        if public_url:
            if hosting_method == "base64":
                return f"📷 {image_path.name}\n🔗 Base64 зображення вбудовано в HTML\n💡 Переглядайте через HTML файл"
            return f"📷 {image_path.name}\n🔗 Публічний URL: {public_url}"
        else:
            # Fallback до локального шляху
            local_url = f"file://{image_path.absolute()}"
            return f"📷 {image_path.name}\n🔗 Локальний шлях: {local_url}\n💡 Для перегляду відкрийте HTML файл"
        
    except Exception as exc:
        print(f"⚠️ Помилка завантаження зображення в Notion: {exc}", file=sys.stderr)
        return f"📷 {image_path.name} (зображення збережено локально)"


def embed_html_in_notion(html_file: Path) -> str:
    """Вбудовує HTML контент в Notion як код."""
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Обмежуємо розмір контенту для Notion (максимум ~2000 символів)
        if len(html_content) > 2000:
            html_content = html_content[:2000] + "\n\n... (контент обрізано, повний HTML доступний в локальному файлі)"
        
        return html_content
        
    except Exception as exc:
        print(f"⚠️ Помилка читання HTML файлу: {exc}", file=sys.stderr)
        return ""


def create_public_html_url(html_file: Path, video_name: str = None) -> str:
    """Створює публічний URL для HTML файлу через GitHub Gist."""
    try:
        # Спробуємо створити GitHub Gist
        gist_url = create_storyboard_gist(html_file, video_name or "video")
        
        if gist_url:
            return get_gist_instructions(html_file, gist_url)
        else:
            # Якщо Gist не створено, повертаємо інструкції
            return get_gist_instructions(html_file, None)
        
    except Exception as exc:
        print(f"⚠️ Помилка створення публічного URL: {exc}", file=sys.stderr)
        return get_gist_instructions(html_file, None)


def create_notion_storyboard(
    notion,
    parent_page_id: str,
    video_name: str,
    screenshots: List[Path],
    analyses: List[Path],
    timepoints: List[TimePoint],
    html_file: Path = None,
    image_hosting_method: str = "imgur"
) -> str:
    """Створює Notion сторінку з storyboard."""
    try:
        # Створюємо головну сторінку storyboard
        page_title = f"Script & Timeline: {video_name}"
        
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
                                    "content": f"🎬 Автоматично створений script & timeline для відео {video_name}\n"
                                             f"📊 Кількість кадрів: {len(screenshots)}\n"
                                             f"📅 Створено: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                }
                            }
                        ]
                    }
                }
            ]
        )
        
        # Додаємо HTML розкадровку якщо є
        if html_file and html_file.exists():
            html_content = embed_html_in_notion(html_file)
            if html_content:
                notion.blocks.children.append(
                    block_id=page_id,
                    children=[
                        {
                            "object": "block",
                            "type": "heading_2",
                            "heading_2": {
                                "rich_text": [
                                    {
                                        "type": "text",
                                        "text": {"content": "📋 Розкадровка (Storyboard)"}
                                    }
                                ]
                            }
                        },
                        {
                            "object": "block",
                            "type": "code",
                            "code": {
                                "rich_text": [
                                    {
                                        "type": "text",
                                        "text": {"content": html_content}
                                    }
                                ],
                                "language": "html"
                            }
                        },
                        {
                            "object": "block",
                            "type": "callout",
                            "callout": {
                                "rich_text": [
                                    {
                                        "type": "text",
                                        "text": {
                                            "content": create_public_html_url(html_file, video_name)
                                        }
                                    }
                                ],
                                "icon": {"emoji": "🔗"}
                            }
                        }
                    ]
                )
        
        # Додаємо кожен кадр
        for i, (screenshot, analysis, tp) in enumerate(zip(screenshots, analyses, timepoints), 1):
            # Завантажуємо зображення
            image_url = upload_image_to_notion(notion, screenshot, image_hosting_method)
            
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


def check_notion_token() -> bool:
    """Перевіряє наявність Notion API токена."""
    return bool(os.getenv("NOTION_API_TOKEN"))


def create_notion_client():
    """Створює клієнт Notion з токеном з змінних середовища."""
    notion_token = os.getenv("NOTION_API_TOKEN")
    if not notion_token:
        raise ValueError("NOTION_API_TOKEN не встановлено в змінних середовища")
    
    return Client(auth=notion_token)
