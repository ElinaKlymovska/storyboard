"""Unified configuration management for video analysis."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
import os
from dotenv import load_dotenv


@dataclass
class AudioConfig:
    """Audio analysis configuration."""
    segment_duration: float = 1.0
    speech_recognition_language: str = "en-US"
    silence_threshold_db: float = -40.0
    enable_transcription: bool = True
    enable_level_analysis: bool = True
    fallback_languages: List[str] = field(default_factory=lambda: ["uk-UA", "auto"])


@dataclass
class VideoConfig:
    """Video analysis configuration."""
    keyframe_interval: float = 2.0
    keyframe_selection_mode: str = "event_driven"  # "interval_based" or "event_driven"
    target_keyframes_per_minute: int = 30
    max_keyframes: int = 100
    frame_analysis_enabled: bool = True
    

@dataclass
class EventDetectionConfig:
    """Event-driven detection configuration."""
    change_sensitivity: Dict[str, float] = field(default_factory=lambda: {
        'facial': 0.15,
        'pose': 0.25,
        'scene': 0.4,
        'motion': 0.3,
        'visual_quality': 0.35
    })
    similarity_threshold: float = 0.80
    max_frames_per_second: int = 5
    filter_strategy: str = "keep_first_discard_rest"
    min_significance_threshold: float = 0.3
    min_confidence_threshold: float = 0.5


@dataclass
class OutputConfig:
    """Output configuration."""
    base_output_dir: str = "output"
    generate_html_report: bool = True
    generate_json_report: bool = True
    generate_text_summary: bool = True
    save_keyframes: bool = True
    save_audio_files: bool = True
    compress_images: bool = False
    image_quality: int = 95
    
    # New session management options
    use_session_system: bool = True
    auto_archive_days: int = 30  # Archive sessions older than X days
    create_readme: bool = True   # Create README.md in each session
    cleanup_temp_files: bool = True  # Clean up temp files after completion
    session_name_template: str = "{video_name}_{timestamp}_{session_id}"


@dataclass
class ExternalAPIConfig:
    """External API configuration."""
    replicate_model: str = "yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb"
    video_analysis_model: str = "lucataco/videollama3-7b:34a1f45f7068f7121a5b47c91f2d7e06c298850767f76f96660450a0a3bd5bbe"
    replicate_max_tokens: int = 500
    replicate_temperature: float = 0.3
    notion_keyframes_db_id: Optional[str] = None
    notion_storyboard_db_id: Optional[str] = None
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    use_external_apis: bool = False


@dataclass
class AnalysisConfig:
    """Main configuration container."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    event_detection: EventDetectionConfig = field(default_factory=EventDetectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    external_apis: ExternalAPIConfig = field(default_factory=ExternalAPIConfig)
    log_level: str = "INFO"
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'AnalysisConfig':
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AnalysisConfig':
        """Create configuration from dictionary."""
        return cls(
            audio=AudioConfig(**data.get('audio', {})),
            video=VideoConfig(**data.get('video', {})),
            event_detection=EventDetectionConfig(**data.get('event_detection', {})),
            output=OutputConfig(**data.get('output', {})),
            external_apis=ExternalAPIConfig(**data.get('external_apis', {})),
            log_level=data.get('log_level', 'INFO')
        )
    
    @classmethod
    def from_env(cls) -> 'AnalysisConfig':
        """Load configuration from environment variables."""
        load_dotenv()
        
        config = cls()
        
        # External API settings
        config.external_apis.notion_keyframes_db_id = os.getenv("KEYFRAMES_DB_ID")
        config.external_apis.notion_storyboard_db_id = os.getenv("STORYBOARD_DB_ID")
        config.external_apis.supabase_url = os.getenv("SUPABASE_URL")
        config.external_apis.supabase_key = os.getenv("SUPABASE_KEY")
        
        # Enable external APIs if credentials are provided
        config.external_apis.use_external_apis = bool(
            config.external_apis.notion_keyframes_db_id and 
            config.external_apis.supabase_url
        )
        
        # Log level
        config.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        return config
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'audio': {
                'segment_duration': self.audio.segment_duration,
                'speech_recognition_language': self.audio.speech_recognition_language,
                'silence_threshold_db': self.audio.silence_threshold_db,
                'enable_transcription': self.audio.enable_transcription,
                'enable_level_analysis': self.audio.enable_level_analysis,
                'fallback_languages': self.audio.fallback_languages
            },
            'video': {
                'keyframe_interval': self.video.keyframe_interval,
                'keyframe_selection_mode': self.video.keyframe_selection_mode,
                'target_keyframes_per_minute': self.video.target_keyframes_per_minute,
                'max_keyframes': self.video.max_keyframes,
                'frame_analysis_enabled': self.video.frame_analysis_enabled
            },
            'event_detection': {
                'change_sensitivity': self.event_detection.change_sensitivity,
                'similarity_threshold': self.event_detection.similarity_threshold,
                'max_frames_per_second': self.event_detection.max_frames_per_second,
                'filter_strategy': self.event_detection.filter_strategy,
                'min_significance_threshold': self.event_detection.min_significance_threshold,
                'min_confidence_threshold': self.event_detection.min_confidence_threshold
            },
            'output': {
                'base_output_dir': self.output.base_output_dir,
                'generate_html_report': self.output.generate_html_report,
                'generate_json_report': self.output.generate_json_report,
                'generate_text_summary': self.output.generate_text_summary,
                'save_keyframes': self.output.save_keyframes,
                'save_audio_files': self.output.save_audio_files,
                'compress_images': self.output.compress_images,
                'image_quality': self.output.image_quality,
                'use_session_system': self.output.use_session_system,
                'auto_archive_days': self.output.auto_archive_days,
                'create_readme': self.output.create_readme,
                'cleanup_temp_files': self.output.cleanup_temp_files,
                'session_name_template': self.output.session_name_template
            },
            'external_apis': {
                'replicate_model': self.external_apis.replicate_model,
                'video_analysis_model': self.external_apis.video_analysis_model,
                'replicate_max_tokens': self.external_apis.replicate_max_tokens,
                'replicate_temperature': self.external_apis.replicate_temperature,
                'use_external_apis': self.external_apis.use_external_apis
            },
            'log_level': self.log_level
        }
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def get_output_dir(self, video_name: str) -> Path:
        """Get output directory for a specific video."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(self.output.base_output_dir) / f"{video_name}_{timestamp}"


# Global configuration instance
_config_instance: Optional[AnalysisConfig] = None


def get_config() -> AnalysisConfig:
    """Get or create the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = AnalysisConfig.from_env()
    return _config_instance


def set_config(config: AnalysisConfig) -> None:
    """Set the global configuration instance."""
    global _config_instance
    _config_instance = config


def load_config(config_path: Optional[Union[str, Path]] = None) -> AnalysisConfig:
    """Load configuration from file or environment."""
    if config_path:
        config = AnalysisConfig.from_file(config_path)
    else:
        config = AnalysisConfig.from_env()
    
    set_config(config)
    return config


def create_default_config_file(config_path: Union[str, Path] = "config/default_config.json") -> None:
    """Create a default configuration file."""
    config = AnalysisConfig()
    config.save_to_file(config_path)
    print(f"Default configuration saved to: {config_path}")


if __name__ == "__main__":
    # Create default config file when run directly
    create_default_config_file()