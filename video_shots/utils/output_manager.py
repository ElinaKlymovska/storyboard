"""
Централізована система управління вихідними даними
=====================================================

Цей модуль забезпечує:
- Унікальні назви папок для кожного аналізу
- Стандартизовану структуру output
- Запобігання перезапису даних
- Метадані про кожен аналіз
- Простий API для управління результатами
"""

import uuid
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict

from video_shots.logging.logger import get_logger


@dataclass
class AnalysisMetadata:
    """Метадані про аналіз відео."""
    session_id: str
    video_name: str
    video_path: str
    start_time: str
    end_time: Optional[str] = None
    analysis_type: str = "complete"  # "complete", "audio_only", "video_only"
    config_used: Optional[Dict] = None
    success: bool = False
    error_message: Optional[str] = None
    total_duration: Optional[float] = None
    keyframes_count: int = 0
    audio_segments_count: int = 0
    output_files: List[str] = None

    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []


class OutputManager:
    """
    Централізований менеджер для управління вихідними даними.
    
    Створює унікальні папки для кожного аналізу та управляє структурою файлів.
    """
    
    def __init__(self, base_output_dir: Union[str, Path] = "output"):
        """
        Ініціалізація менеджера.
        
        Args:
            base_output_dir: Базова директорія для всіх результатів
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()
        
        # Створення архівної папки для старих результатів
        self.archive_dir = self.base_output_dir / "_archive"
        self.archive_dir.mkdir(exist_ok=True)
        
    def create_analysis_session(
        self,
        video_path: Union[str, Path],
        analysis_type: str = "complete",
        config: Optional[Dict] = None,
        custom_session_name: Optional[str] = None
    ) -> "AnalysisSession":
        """
        Створює нову сесію аналізу з унікальною папкою.
        
        Args:
            video_path: Шлях до відеофайлу
            analysis_type: Тип аналізу ("complete", "audio_only", "video_only")
            config: Конфігурація що використовується
            custom_session_name: Кастомна назва сесії (опціонально)
            
        Returns:
            AnalysisSession: Об'єкт сесії для управління результатами
        """
        video_path = Path(video_path)
        video_name = video_path.stem
        
        # Створення унікального ID сесії
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Формування назви папки
        if custom_session_name:
            session_name = f"{custom_session_name}_{session_id}"
        else:
            session_name = f"{video_name}_{timestamp}_{session_id}"
        
        session_dir = self.base_output_dir / session_name
        
        # Забезпечення унікальності (на всякий випадок)
        counter = 1
        original_session_dir = session_dir
        while session_dir.exists():
            session_dir = original_session_dir.parent / f"{original_session_dir.name}_{counter}"
            counter += 1
        
        # Створення сесії
        metadata = AnalysisMetadata(
            session_id=session_id,
            video_name=video_name,
            video_path=str(video_path),
            start_time=datetime.now().isoformat(),
            analysis_type=analysis_type,
            config_used=config
        )
        
        session = AnalysisSession(session_dir, metadata, self.logger)
        session.create_directory_structure()
        
        self.logger.info(f"📁 Створено нову сесію аналізу: {session_name}")
        self.logger.info(f"   📍 Директорія: {session_dir}")
        self.logger.info(f"   🎬 Відео: {video_name}")
        self.logger.info(f"   🔍 Тип аналізу: {analysis_type}")
        
        return session
        
    def get_recent_sessions(self, limit: int = 10) -> List[Path]:
        """
        Отримує список останніх сесій аналізу.
        
        Args:
            limit: Максимальна кількість сесій
            
        Returns:
            List[Path]: Список шляхів до папок сесій
        """
        sessions = []
        for item in self.base_output_dir.iterdir():
            if item.is_dir() and not item.name.startswith("_"):
                sessions.append(item)
        
        # Сортування за часом створення
        sessions.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        return sessions[:limit]
        
    def archive_old_sessions(self, days_old: int = 30) -> int:
        """
        Архівує старі сесії.
        
        Args:
            days_old: Вік сесій в днях для архівування
            
        Returns:
            int: Кількість архівованих сесій
        """
        current_time = datetime.now().timestamp()
        archived_count = 0
        
        for session_dir in self.base_output_dir.iterdir():
            if session_dir.is_dir() and not session_dir.name.startswith("_"):
                # Перевірка віку папки
                creation_time = session_dir.stat().st_ctime
                age_days = (current_time - creation_time) / (24 * 3600)
                
                if age_days > days_old:
                    archive_path = self.archive_dir / session_dir.name
                    shutil.move(str(session_dir), str(archive_path))
                    archived_count += 1
                    self.logger.info(f"📦 Архівовано сесію: {session_dir.name}")
        
        if archived_count > 0:
            self.logger.info(f"📦 Всього архівовано сесій: {archived_count}")
        
        return archived_count
        
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Отримує загальну інформацію про всі сесії.
        
        Returns:
            Dict: Інформація про сесії
        """
        total_sessions = len([d for d in self.base_output_dir.iterdir() 
                             if d.is_dir() and not d.name.startswith("_")])
        archived_sessions = len([d for d in self.archive_dir.iterdir() if d.is_dir()])
        
        total_size = sum(
            sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
            for d in self.base_output_dir.iterdir() if d.is_dir()
        )
        
        return {
            "total_sessions": total_sessions,
            "archived_sessions": archived_sessions,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "base_output_dir": str(self.base_output_dir)
        }


class AnalysisSession:
    """
    Представляє одну сесію аналізу відео з управлінням файлами та метаданими.
    """
    
    def __init__(self, session_dir: Path, metadata: AnalysisMetadata, logger):
        self.session_dir = session_dir
        self.metadata = metadata
        self.logger = logger
        
        # Стандартні підпапки
        self.keyframes_dir = session_dir / "keyframes"
        self.audio_dir = session_dir / "audio"
        self.reports_dir = session_dir / "reports"
        self.segments_dir = session_dir / "segments"
        self.temp_dir = session_dir / "temp"
        
    def create_directory_structure(self):
        """Створює стандартну структуру папок для сесії."""
        dirs_to_create = [
            self.session_dir,
            self.keyframes_dir,
            self.audio_dir,
            self.reports_dir,
            self.segments_dir,
            self.temp_dir
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Створення файлу метаданих
        self.save_metadata()
        
        # Створення README файлу
        self._create_readme()
        
    def _create_readme(self):
        """Створює README файл з інформацією про сесію."""
        readme_content = f"""# Аналіз відео: {self.metadata.video_name}

## Інформація про сесію
- **ID сесії**: {self.metadata.session_id}
- **Відеофайл**: {self.metadata.video_path}
- **Тип аналізу**: {self.metadata.analysis_type}
- **Час початку**: {self.metadata.start_time}

## Структура папок
- `keyframes/` - Ключові кадри з відео
- `audio/` - Аудіо файли та транскрипція
- `reports/` - Звіти аналізу (JSON, HTML, TXT)
- `segments/` - Сегменти відео (якщо створювались)
- `temp/` - Тимчасові файли

## Метадані
Детальні метадані зберігаються в файлі `session_metadata.json`

Згенеровано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        readme_path = self.session_dir / "README.md"
        readme_path.write_text(readme_content, encoding='utf-8')
        
    def save_metadata(self):
        """Зберігає метадані сесії."""
        metadata_path = self.session_dir / "session_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.metadata), f, indent=2, ensure_ascii=False)
            
    def update_metadata(self, **kwargs):
        """Оновлює метадані сесії."""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
        self.save_metadata()
        
    def mark_completed(self, success: bool = True, error_message: Optional[str] = None):
        """Позначає сесію як завершену."""
        self.metadata.end_time = datetime.now().isoformat()
        self.metadata.success = success
        if error_message:
            self.metadata.error_message = error_message
        self.save_metadata()
        
        status = "✅ успішно" if success else "❌ з помилкою"
        self.logger.info(f"📋 Сесія {self.metadata.session_id} завершена {status}")
        
    def add_output_file(self, file_path: Union[str, Path], description: Optional[str] = None):
        """Додає інформацію про створений файл."""
        file_path = Path(file_path)
        file_info = str(file_path.relative_to(self.session_dir))
        if description:
            file_info += f" ({description})"
        self.metadata.output_files.append(file_info)
        self.save_metadata()
        
    def get_keyframe_path(self, frame_filename: str) -> Path:
        """Отримує шлях для збереження ключового кадру."""
        return self.keyframes_dir / frame_filename
        
    def get_audio_path(self, audio_filename: str) -> Path:
        """Отримує шлях для збереження аудіо файлу."""
        return self.audio_dir / audio_filename
        
    def get_report_path(self, report_filename: str) -> Path:
        """Отримує шлях для збереження звіту."""
        return self.reports_dir / report_filename
        
    def get_segment_path(self, segment_filename: str) -> Path:
        """Отримує шлях для збереження сегменту відео."""
        return self.segments_dir / segment_filename
        
    def get_temp_path(self, temp_filename: str) -> Path:
        """Отримує шлях для тимчасових файлів."""
        return self.temp_dir / temp_filename
        
    def cleanup_temp(self):
        """Очищає тимчасові файли."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir()
            self.logger.info(f"🧹 Очищено тимчасові файли для сесії {self.metadata.session_id}")
            
    def get_summary(self) -> Dict[str, Any]:
        """Отримує інформацію про сесію."""
        total_files = len(list(self.session_dir.rglob('*'))) - len(list(self.temp_dir.rglob('*')))
        total_size = sum(
            f.stat().st_size for f in self.session_dir.rglob('*') 
            if f.is_file() and not f.is_relative_to(self.temp_dir)
        )
        
        return {
            "session_id": self.metadata.session_id,
            "video_name": self.metadata.video_name,
            "analysis_type": self.metadata.analysis_type,
            "success": self.metadata.success,
            "start_time": self.metadata.start_time,
            "end_time": self.metadata.end_time,
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "output_files": self.metadata.output_files,
            "session_dir": str(self.session_dir)
        }


# Глобальний екземпляр менеджера
_output_manager: Optional[OutputManager] = None


def get_output_manager(base_output_dir: Union[str, Path] = "output") -> OutputManager:
    """Отримує глобальний екземпляр OutputManager."""
    global _output_manager
    if _output_manager is None:
        _output_manager = OutputManager(base_output_dir)
    return _output_manager


def create_analysis_session(
    video_path: Union[str, Path],
    analysis_type: str = "complete",
    config: Optional[Dict] = None,
    custom_session_name: Optional[str] = None,
    base_output_dir: Union[str, Path] = "output"
) -> AnalysisSession:
    """
    Зручна функція для створення нової сесії аналізу.
    
    Args:
        video_path: Шлях до відеофайлу
        analysis_type: Тип аналізу
        config: Конфігурація
        custom_session_name: Кастомна назва сесії
        base_output_dir: Базова директорія output
        
    Returns:
        AnalysisSession: Новий об'єкт сесії
    """
    manager = get_output_manager(base_output_dir)
    return manager.create_analysis_session(
        video_path=video_path,
        analysis_type=analysis_type,
        config=config,
        custom_session_name=custom_session_name
    )