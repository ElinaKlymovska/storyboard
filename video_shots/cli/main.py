"""Unified CLI interface for video analysis."""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video_shots.analysis.unified_analyzer import UnifiedAnalyzer
from video_shots.config.analysis_config import load_config, create_default_config_file
from video_shots.logging.logger import setup_logging


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified Video Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a video with default settings
  python -m video_shots.cli.main analyze input/videos/video.mp4
  
  # Analyze with custom output directory
  python -m video_shots.cli.main analyze input/videos/video.mp4 -o output/my_analysis
  
  # Analyze with custom configuration
  python -m video_shots.cli.main analyze input/videos/video.mp4 -c config/custom.json
  
  # Split video into 0.5 second segments with audio
  python -m video_shots.cli.main segment input/videos/video.mp4
  
  # Split video into 2 second segments with custom output
  python -m video_shots.cli.main segment input/videos/video.mp4 -d 2.0 -o output/segments
  
  # Create default configuration file
  python -m video_shots.cli.main create-config -o config/default.json
  
  # Set log level to debug
  python -m video_shots.cli.main analyze input/videos/video.mp4 --log-level DEBUG
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (JSON format)"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a video file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Perform complete video analysis including audio and visual components."
    )
    
    analyze_parser.add_argument(
        "video_path",
        type=str,
        help="Path to the video file to analyze"
    )
    
    analyze_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for results (default: auto-generated)"
    )
    
    analyze_parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Perform only audio analysis"
    )
    
    analyze_parser.add_argument(
        "--video-only", 
        action="store_true",
        help="Perform only video analysis"
    )
    
    analyze_parser.add_argument(
        "--no-transcription",
        action="store_true",
        help="Skip audio transcription"
    )
    
    analyze_parser.add_argument(
        "--keyframe-mode",
        choices=["interval_based", "event_driven"],
        help="Keyframe selection mode"
    )
    
    analyze_parser.add_argument(
        "--keyframe-interval",
        type=float,
        help="Interval between keyframes in seconds (for interval_based mode)"
    )
    
    analyze_parser.add_argument(
        "--max-keyframes",
        type=int,
        help="Maximum number of keyframes to extract"
    )
    
    # Create config command
    config_parser = subparsers.add_parser(
        "create-config",
        help="Create a default configuration file",
        description="Generate a default configuration file that can be customized."
    )
    
    config_parser.add_argument(
        "--output", "-o",
        type=str,
        default="config/default_config.json",
        help="Output path for configuration file (default: config/default_config.json)"
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about a video file",
        description="Display basic information about a video file without full analysis."
    )
    
    info_parser.add_argument(
        "video_path",
        type=str,
        help="Path to the video file"
    )
    
    # Sessions command
    sessions_parser = subparsers.add_parser(
        "sessions",
        help="Manage analysis sessions",
        description="View and manage analysis sessions and output data."
    )
    
    sessions_subparsers = sessions_parser.add_subparsers(dest="sessions_action", help="Session actions")
    
    # List sessions
    list_parser = sessions_subparsers.add_parser("list", help="List recent sessions")
    list_parser.add_argument("--limit", "-l", type=int, default=10, help="Number of sessions to show")
    
    # Session info
    info_sessions_parser = sessions_subparsers.add_parser("info", help="Show session information")
    info_sessions_parser.add_argument("session_id", help="Session ID or partial session ID")
    
    # Archive old sessions
    archive_parser = sessions_subparsers.add_parser("archive", help="Archive old sessions")
    archive_parser.add_argument("--days", "-d", type=int, default=30, help="Archive sessions older than X days")
    
    # Clean up sessions
    cleanup_parser = sessions_subparsers.add_parser("cleanup", help="Clean up temp files in all sessions")
    
    # Segment command
    segment_parser = subparsers.add_parser(
        "segment",
        help="Split video into segments with audio",
        description="Split a video file into smaller segments while preserving audio."
    )
    
    segment_parser.add_argument(
        "video_path",
        type=str,
        help="Path to the video file to segment"
    )
    
    segment_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for segments (default: auto-generated based on video name)"
    )
    
    segment_parser.add_argument(
        "--duration", "-d",
        type=float,
        default=0.5,
        help="Duration of each segment in seconds (default: 0.5)"
    )
    
    segment_parser.add_argument(
        "--prefix", "-p",
        type=str,
        default="segment",
        help="Prefix for segment filenames (default: segment)"
    )
    
    segment_parser.add_argument(
        "--max-segments", "-m",
        type=int,
        help="Maximum number of segments to create (default: unlimited)"
    )
    
    return parser


def handle_analyze_command(args, config) -> int:
    """Handle the analyze command."""
    from video_shots.logging.logger import get_logger
    
    logger = get_logger()
    video_path = Path(args.video_path)
    
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return 1
    
    # Update config based on command line arguments
    if args.no_transcription:
        config.audio.enable_transcription = False
    
    if args.keyframe_mode:
        config.video.keyframe_selection_mode = args.keyframe_mode
    
    if args.keyframe_interval:
        config.video.keyframe_interval = args.keyframe_interval
    
    if args.max_keyframes:
        config.video.max_keyframes = args.max_keyframes
    
    # Set output directory
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
    
    try:
        # Create analyzer
        analyzer = UnifiedAnalyzer(config)
        
        # Perform analysis
        if args.audio_only:
            logger.info("🎵 Performing audio-only analysis...")
            audio_result = analyzer.audio_analyzer.analyze_video_audio(video_path, output_dir)
            if audio_result.success:
                logger.info("✅ Audio analysis completed successfully!")
                logger.info(f"📊 Results: {len(audio_result.audio_segments)} audio segments, "
                           f"{len(audio_result.transcript_segments)} speech segments")
            else:
                logger.error(f"❌ Audio analysis failed: {audio_result.error_message}")
                return 1
                
        elif args.video_only:
            logger.info("🎬 Performing video-only analysis...")
            video_result = analyzer.video_analyzer.analyze_video(video_path, output_dir)
            if video_result.success:
                logger.info("✅ Video analysis completed successfully!")
                logger.info(f"📊 Results: {len(video_result.keyframes)} keyframes, "
                           f"{len(video_result.shots)} shots")
            else:
                logger.error(f"❌ Video analysis failed: {video_result.error_message}")
                return 1
        else:
            logger.info("🎬🎵 Performing complete analysis...")
            result = analyzer.analyze_complete_video(video_path, output_dir)
            
            if result.success:
                logger.info("✅ Complete analysis finished successfully!")
                logger.info(f"📊 Summary:")
                logger.info(f"   - Duration: {result.audio_result.duration:.1f}s")
                logger.info(f"   - Keyframes: {result.total_keyframes}")
                logger.info(f"   - Audio segments: {result.total_audio_segments}")
                logger.info(f"   - Speech segments: {result.total_speech_segments}")
                logger.info(f"   - Output directory: {result.output_dir}")
                logger.info(f"   - Content type: {result.content_analysis.get('content_type', 'Unknown')}")
                logger.info(f"   - Overall quality score: {result.technical_quality.get('overall_score', 0)}/100")
            else:
                logger.error(f"❌ Analysis failed: {result.error_message}")
                return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Unexpected error during analysis: {e}")
        return 1


def handle_create_config_command(args) -> int:
    """Handle the create-config command."""
    from video_shots.logging.logger import get_logger
    
    logger = get_logger()
    output_path = Path(args.output)
    
    try:
        create_default_config_file(output_path)
        logger.info(f"✅ Default configuration created: {output_path}")
        return 0
    except Exception as e:
        logger.error(f"❌ Failed to create configuration: {e}")
        return 1


def handle_info_command(args) -> int:
    """Handle the info command."""
    from video_shots.logging.logger import get_logger
    from video_shots.analysis.unified_video_analyzer import UnifiedVideoAnalyzer
    
    logger = get_logger()
    video_path = Path(args.video_path)
    
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return 1
    
    try:
        analyzer = UnifiedVideoAnalyzer()
        metadata = analyzer._extract_metadata(video_path)
        
        print(f"\n📹 Video Information: {video_path.name}")
        print("=" * 50)
        print(f"Duration: {metadata.duration:.1f} seconds")
        print(f"FPS: {metadata.fps:.1f}")
        print(f"Resolution: {metadata.resolution[0]}x{metadata.resolution[1]}")
        print(f"Aspect Ratio: {metadata.aspect_ratio}")
        print(f"Format: {metadata.format}")
        print(f"Total Frames: {metadata.frame_count}")
        print(f"File Size: {video_path.stat().st_size / (1024*1024):.1f} MB")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Failed to get video info: {e}")
        return 1


def handle_sessions_command(args) -> int:
    """Handle the sessions command."""
    from video_shots.logging.logger import get_logger
    from video_shots.utils.output_manager import get_output_manager
    
    logger = get_logger()
    manager = get_output_manager()
    
    if args.sessions_action == "list":
        # List recent sessions
        sessions = manager.get_recent_sessions(args.limit)
        
        if not sessions:
            print("📁 Немає збережених сесій аналізу")
            return 0
            
        print(f"📋 Останні {len(sessions)} сесій аналізу:")
        print("=" * 80)
        
        for i, session_dir in enumerate(sessions, 1):
            # Try to load session metadata
            metadata_file = session_dir / "session_metadata.json"
            if metadata_file.exists():
                try:
                    import json
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    session_id = metadata.get('session_id', 'unknown')
                    video_name = metadata.get('video_name', 'unknown')
                    start_time = metadata.get('start_time', 'unknown')
                    success = metadata.get('success', False)
                    analysis_type = metadata.get('analysis_type', 'unknown')
                    
                    status = "✅ Успішно" if success else "❌ Помилка"
                    
                    print(f"{i:2}. {session_dir.name}")
                    print(f"    📽️  Відео: {video_name}")
                    print(f"    🆔 ID: {session_id}")
                    print(f"    🔍 Тип: {analysis_type}")
                    print(f"    📅 Час: {start_time}")
                    print(f"    📊 Статус: {status}")
                    print()
                    
                except Exception as e:
                    print(f"{i:2}. {session_dir.name} (метадані недоступні)")
                    print()
            else:
                print(f"{i:2}. {session_dir.name} (без метаданих)")
                print()
                
    elif args.sessions_action == "info":
        # Show detailed session info
        sessions = manager.get_recent_sessions(100)  # Get more sessions to search
        
        # Find session by ID (partial match)
        target_session = None
        for session_dir in sessions:
            if args.session_id.lower() in session_dir.name.lower():
                target_session = session_dir
                break
                
        if not target_session:
            logger.error(f"❌ Сесію з ID '{args.session_id}' не знайдено")
            return 1
            
        # Load and display session info
        metadata_file = target_session / "session_metadata.json"
        if metadata_file.exists():
            try:
                import json
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                print(f"📋 Інформація про сесію: {target_session.name}")
                print("=" * 80)
                print(f"🆔 ID сесії: {metadata.get('session_id', 'unknown')}")
                print(f"📽️ Відео: {metadata.get('video_name', 'unknown')}")
                print(f"📁 Шлях до відео: {metadata.get('video_path', 'unknown')}")
                print(f"🔍 Тип аналізу: {metadata.get('analysis_type', 'unknown')}")
                print(f"📅 Початок: {metadata.get('start_time', 'unknown')}")
                print(f"⏰ Завершення: {metadata.get('end_time', 'в процесі')}")
                print(f"📊 Статус: {'✅ Успішно' if metadata.get('success', False) else '❌ Помилка'}")
                print(f"⏱️ Тривалість відео: {metadata.get('total_duration', 'невідомо')} сек")
                print(f"🖼️ Ключові кадри: {metadata.get('keyframes_count', 0)}")
                print(f"🎵 Аудіо сегменти: {metadata.get('audio_segments_count', 0)}")
                
                if metadata.get('error_message'):
                    print(f"❌ Помилка: {metadata['error_message']}")
                    
                print(f"📁 Директорія: {target_session}")
                
                # Show output files
                output_files = metadata.get('output_files', [])
                if output_files:
                    print("\n📄 Створені файли:")
                    for file_info in output_files:
                        print(f"   • {file_info}")
                        
            except Exception as e:
                logger.error(f"❌ Помилка читання метаданих: {e}")
                return 1
        else:
            logger.error(f"❌ Файл метаданих не знайдено для сесії {target_session.name}")
            return 1
            
    elif args.sessions_action == "archive":
        # Archive old sessions
        archived_count = manager.archive_old_sessions(args.days)
        if archived_count > 0:
            print(f"📦 Архівовано {archived_count} сесій старших за {args.days} днів")
        else:
            print(f"📦 Немає сесій для архівування (старших за {args.days} днів)")
            
    elif args.sessions_action == "cleanup":
        # Clean up temp files
        sessions = manager.get_recent_sessions(1000)  # Get all sessions
        cleaned_count = 0
        
        for session_dir in sessions:
            temp_dir = session_dir / "temp"
            if temp_dir.exists():
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                    temp_dir.mkdir()
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Помилка очищення {temp_dir}: {e}")
                    
        print(f"🧹 Очищено тимчасові файли в {cleaned_count} сесіях")
        
    else:
        print("❌ Невідома дія для команди sessions")
        return 1
        
    return 0


def handle_segment_command(args) -> int:
    """Handle the segment command."""
    from video_shots.logging.logger import get_logger
    from video_shots.core.video_processing import split_video_with_audio
    from video_shots.utils.output_manager import get_output_manager
    
    logger = get_logger()
    video_path = Path(args.video_path)
    
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return 1
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Create output directory based on video name
        video_stem = video_path.stem
        output_dir = Path("output") / f"{video_stem}_segments"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"🎬 Starting video segmentation...")
        logger.info(f"📽️ Input: {video_path}")
        logger.info(f"⏱️ Segment duration: {args.duration} seconds")
        logger.info(f"📁 Output directory: {output_dir}")
        
        # Split video with audio
        segment_paths = split_video_with_audio(
            video_path=video_path,
            output_dir=output_dir,
            segment_duration=args.duration,
            prefix=args.prefix,
            max_segments=args.max_segments
        )
        
        if segment_paths:
            logger.info(f"✅ Segmentation completed successfully!")
            logger.info(f"📊 Created {len(segment_paths)} segments")
            logger.info(f"📁 Segments saved to: {output_dir / 'segments'}")
            
            # Show first few segments as examples
            logger.info("📄 Generated segments:")
            for i, path in enumerate(segment_paths[:5]):
                logger.info(f"   {i+1}. {path.name}")
            
            if len(segment_paths) > 5:
                logger.info(f"   ... and {len(segment_paths) - 5} more segments")
                
        else:
            logger.warning("⚠️ No segments were created")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Segmentation failed: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Load configuration
    try:
        if args.config:
            config = load_config(args.config)
            logger.info(f"📄 Loaded configuration from: {args.config}")
        else:
            config = load_config()
            logger.debug("📄 Using default configuration")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Handle commands
    if args.command == "analyze":
        return handle_analyze_command(args, config)
    elif args.command == "create-config":
        return handle_create_config_command(args)
    elif args.command == "info":
        return handle_info_command(args)
    elif args.command == "sessions":
        return handle_sessions_command(args)
    elif args.command == "segment":
        return handle_segment_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())