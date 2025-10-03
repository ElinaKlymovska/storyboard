#!/usr/bin/env python3
"""
Unified Video Analysis Tool - Backward Compatibility Entry Point
================================================================

⚠️  DEPRECATED: Це застарілий entry point для зворотної сумісності.
    
Рекомендовано використовувати:
    python -m video_shots analyze input/videos/video.mp4
    python -m video_shots create-config -o config/my_config.json
    python -m video_shots info input/videos/video.mp4
    python -m video_shots sessions list

Цей файл залишено для зворотної сумісності, але краще використовувати
модульний запуск через python -m video_shots

Usage (deprecated):
    python video_shots/analyze.py analyze input/videos/video.mp4

For more options, use:
    python -m video_shots --help
"""

import sys
import warnings
from pathlib import Path

# Add the project root to Python path for namespace packages
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from video_shots.cli.main import main

def deprecated_entry_point():
    """Show deprecation warning and run main CLI."""
    warnings.warn(
        "Використання video_shots/analyze.py застаріло. "
        "Використовуйте 'python -m video_shots' замість цього.",
        DeprecationWarning,
        stacklevel=2
    )
    
    print("⚠️  УВАГА: Цей entry point застарілий!")
    print("💡 Рекомендовано: python -m video_shots [command]")
    print("📚 Довідка: python -m video_shots --help")
    print()
    
    return main()

if __name__ == "__main__":
    sys.exit(deprecated_entry_point())