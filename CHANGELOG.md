# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Branch strategy documentation
- Development roadmap with short and long-term goals
- Comprehensive contributing guidelines
- Structured git workflow

### Changed
- Moved `supabase_storage.py` to `video_shots/exports/` module for better organization

## [0.1.0] - 2024-09-27

### Added
- Initial video analysis pipeline implementation
- AI-powered image analysis using Replicate API
- CLI interface for video screenshot generation
- PDF report generation
- Audio transcript parsing and synchronization
- Supabase storage integration
- Notion database synchronization
- Comprehensive error handling and validation
- Support for configurable time intervals and frame limits
- Professional documentation and type hints
- Support for multiple video formats (MP4, AVI, MOV, MKV, WMV, FLV)
- Frame rotation capabilities
- Label panel rendering for screenshots
- Time utility functions with multiple format support
- Audio transcript parsing with timecode matching

### Technical Details
- Python 3.8+ support
- OpenCV for video processing
- PIL/Pillow for image manipulation
- Replicate API for AI analysis
- Notion API for database sync
- Supabase for cloud storage
- ReportLab for PDF generation

### Configuration
- Environment variable support for API tokens
- Configurable analysis prompts
- Customizable output formats and quality settings
- Flexible time interval parsing

## [0.0.1] - Initial Development

### Added
- Basic project structure
- Core video processing functionality
- Initial AI integration
- Basic CLI interface
