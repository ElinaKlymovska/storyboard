# Video Analysis Pipeline

A professional video analysis tool that extracts frames, performs AI-powered analysis, and generates comprehensive reports for video production workflows.

## Features

- **AI-Powered Frame Analysis**: Uses Replicate API for intelligent image analysis
- **Flexible Time Intervals**: Extract frames at custom time intervals
- **Multiple Output Formats**: PNG screenshots, PDF reports, HTML exports
- **Audio Synchronization**: Parse and sync audio transcripts with video frames
- **Cloud Storage Integration**: Supabase storage for scalable file management
- **Notion Integration**: Automatic database synchronization for project management
- **Professional CLI**: Command-line interface with comprehensive error handling

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Screenshot Generation

```bash
python -m video_shots.cli input/videos/nyane_30s.mp4 --interval 0.5s
```

### Advanced Analysis Pipeline

```bash
python -m video_shots.pipeline input/videos/nyane_30s.mp4
```

### Command Line Options

- `--interval`: Time interval between frames (e.g., 0.5s, 200ms, 00:00:03.5)
- `--start`: Start time (default: 0)
- `--end`: End time (default: video duration)
- `--max-frames`: Limit number of frames

## Configuration

Set the following environment variables:

```bash
export REPLICATE_API_TOKEN="your_replicate_token"
export NOTION_API_TOKEN="your_notion_token"
export KEYFRAMES_DB_ID="your_keyframes_database_id"
export STORYBOARD_DB_ID="your_storyboard_database_id"
```

## Project Structure

```
video_shots/
├── cli.py              # Command-line interface
├── pipeline.py          # Main analysis pipeline
├── config.py            # Configuration settings
├── models.py            # Data models
├── exceptions.py        # Custom exceptions
├── core/
│   ├── image_analysis.py    # AI image analysis
│   └── video_processing.py  # Video processing utilities
├── exports/
│   ├── pdf_export.py        # PDF report generation
│   ├── html_export.py       # HTML export
│   └── notion_export.py     # Notion integration
└── utils/
    ├── time_utils.py        # Time and timecode utilities
    └── audio_utils.py       # Audio transcript parsing
```

## Requirements

- Python 3.8+
- OpenCV
- PIL/Pillow
- Replicate API
- Notion API
- Supabase (optional)

## License

This project is licensed under the MIT License.
