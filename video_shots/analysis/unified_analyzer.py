"""Unified analyzer that combines audio and video analysis."""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

from video_shots.analysis.unified_audio_analyzer import UnifiedAudioAnalyzer, AudioAnalysisResult
from video_shots.analysis.unified_video_analyzer import UnifiedVideoAnalyzer, VideoAnalysisResult
from video_shots.config.analysis_config import AnalysisConfig, get_config
from video_shots.logging.logger import get_logger
from video_shots.utils.output_manager import create_analysis_session, AnalysisSession


@dataclass
class CompleteAnalysisResult:
    """Complete analysis result combining audio and video."""
    success: bool
    video_path: str
    output_dir: str
    timestamp: str
    duration: float
    session_id: str
    
    # Results
    audio_result: AudioAnalysisResult
    video_result: VideoAnalysisResult
    
    # Combined insights
    content_analysis: Dict[str, Any]
    technical_quality: Dict[str, Any]
    recommendations: list[str]
    
    # Analysis session for file management
    session: Optional[AnalysisSession] = None
    error_message: Optional[str] = None
    
    @property
    def total_keyframes(self) -> int:
        """Get total keyframes extracted."""
        return self.video_result.total_keyframes if self.video_result.success else 0
    
    @property
    def total_audio_segments(self) -> int:
        """Get total audio segments analyzed."""
        return len(self.audio_result.audio_segments) if self.audio_result.success else 0
    
    @property
    def total_speech_segments(self) -> int:
        """Get total speech segments detected."""
        return self.audio_result.total_speech_segments if self.audio_result.success else 0


class UnifiedAnalyzer:
    """Main analyzer that coordinates audio and video analysis."""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize unified analyzer."""
        self.config = config or get_config()
        self.logger = get_logger()
        
        # Initialize sub-analyzers
        self.audio_analyzer = UnifiedAudioAnalyzer(self.config.audio)
        self.video_analyzer = UnifiedVideoAnalyzer(
            self.config.video, 
            self.config.event_detection
        )
    
    def analyze_complete_video(
        self, 
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        custom_session_name: Optional[str] = None
    ) -> CompleteAnalysisResult:
        """Perform complete video analysis including audio and video components."""
        start_time = time.time()
        video_path = Path(video_path)
        
        self.logger.log_operation_start(
            "complete video analysis",
            video=video_path.name
        )
        
        if not video_path.exists():
            error_msg = f"Video file not found: {video_path}"
            self.logger.error(error_msg)
            return self._create_error_result(str(video_path), error_msg, output_dir)
        
        # Create analysis session
        session = None
        try:
            # Create new analysis session with unique directory
            session = create_analysis_session(
                video_path=video_path,
                analysis_type="complete",
                config=self.config.to_dict() if self.config else None,
                custom_session_name=custom_session_name,
                base_output_dir=output_dir or self.config.output.base_output_dir
            )
            
            # Update session metadata
            session.update_metadata(
                total_duration=None,  # Will be updated later
                keyframes_count=0,
                audio_segments_count=0
            )
            
            # Run audio analysis
            self.logger.info("🎵 Starting audio analysis...")
            audio_result = self.audio_analyzer.analyze_video_audio(
                video_path, 
                session.audio_dir
            )
            
            # Run video analysis  
            self.logger.info("🎬 Starting video analysis...")
            video_result = self.video_analyzer.analyze_video(
                video_path,
                session.session_dir  # Use session directory for keyframes
            )
            
            # Generate combined insights
            self.logger.info("🔍 Generating insights...")
            content_analysis = self._analyze_content(audio_result, video_result)
            technical_quality = self._assess_technical_quality(audio_result, video_result)
            recommendations = self._generate_recommendations(audio_result, video_result)
            
            duration_seconds = time.time() - start_time
            
            result = CompleteAnalysisResult(
                success=True,
                video_path=str(video_path),
                output_dir=str(session.session_dir),
                timestamp=datetime.now().isoformat(),
                duration=duration_seconds,
                session_id=session.metadata.session_id,
                audio_result=audio_result,
                video_result=video_result,
                content_analysis=content_analysis,
                technical_quality=technical_quality,
                recommendations=recommendations,
                session=session
            )
            
            # Update session metadata with final results
            session.update_metadata(
                total_duration=audio_result.duration if audio_result.success else None,
                keyframes_count=result.total_keyframes,
                audio_segments_count=result.total_audio_segments
            )
            
            # Save results to session
            self._save_results(result)
            
            # Mark session as completed successfully
            session.mark_completed(success=True)
            
            self.logger.log_operation_complete(
                "complete video analysis",
                duration=duration_seconds,
                keyframes=result.total_keyframes,
                audio_segments=result.total_audio_segments,
                speech_segments=result.total_speech_segments,
                session_id=session.metadata.session_id
            )
            
            return result
            
        except Exception as e:
            self.logger.log_operation_error("complete video analysis", e)
            
            # Mark session as failed if it was created
            if session:
                session.mark_completed(success=False, error_message=str(e))
            
            return self._create_error_result(str(video_path), str(e), output_dir)
    
    def _analyze_content(
        self, 
        audio_result: AudioAnalysisResult, 
        video_result: VideoAnalysisResult
    ) -> Dict[str, Any]:
        """Analyze content characteristics."""
        self.logger.debug("Analyzing content characteristics")
        
        analysis = {
            "content_type": "Unknown",
            "estimated_language": "Unknown",
            "estimated_speakers": 0,
            "pacing": "Unknown",
            "content_segments": [],
            "key_topics": []
        }
        
        if audio_result.success and audio_result.transcript_segments:
            # Analyze speech content
            all_text = " ".join(seg.text.lower() for seg in audio_result.transcript_segments)
            
            # Determine content type based on keywords
            if any(word in all_text for word in ['makeup', 'beauty', 'foundation', 'lipstick']):
                analysis["content_type"] = "Beauty/Makeup Tutorial"
            elif any(word in all_text for word in ['recipe', 'cook', 'ingredients', 'kitchen']):
                analysis["content_type"] = "Cooking/Recipe"
            elif any(word in all_text for word in ['tutorial', 'how to', 'learn', 'step']):
                analysis["content_type"] = "Educational Tutorial"
            elif any(word in all_text for word in ['music', 'song', 'dance', 'performance']):
                analysis["content_type"] = "Entertainment/Music"
            else:
                analysis["content_type"] = "General Content"
            
            # Language detection (basic)
            if audio_result.transcript_segments:
                analysis["estimated_language"] = audio_result.transcript_segments[0].language
            
            # Speaker estimation (simplified)
            analysis["estimated_speakers"] = 1  # Basic assumption
            
            # Extract key topics
            topics = self._extract_topics_from_text(all_text)
            analysis["key_topics"] = topics
            
            # Analyze content segments
            segments = []
            for i, segment in enumerate(audio_result.transcript_segments):
                segments.append({
                    "index": i,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "text": segment.text,
                    "estimated_category": self._categorize_segment_content(segment.text)
                })
            analysis["content_segments"] = segments
        
        if video_result.success:
            # Analyze pacing based on keyframes
            if video_result.metadata.duration > 0:
                kf_per_minute = video_result.keyframes_per_minute
                if kf_per_minute > 40:
                    analysis["pacing"] = "Fast-paced"
                elif kf_per_minute > 20:
                    analysis["pacing"] = "Medium-paced"
                else:
                    analysis["pacing"] = "Slow-paced"
        
        return analysis
    
    def _assess_technical_quality(
        self, 
        audio_result: AudioAnalysisResult, 
        video_result: VideoAnalysisResult
    ) -> Dict[str, Any]:
        """Assess technical quality of the video."""
        self.logger.debug("Assessing technical quality")
        
        quality = {
            "overall_score": 0,
            "video_quality": "Unknown",
            "audio_quality": "Unknown",
            "resolution_category": "Unknown",
            "format_suitability": {},
            "issues": []
        }
        
        score = 0
        max_score = 100
        
        if video_result.success:
            metadata = video_result.metadata
            
            # Resolution assessment
            width, height = metadata.resolution
            total_pixels = width * height
            
            if total_pixels >= 1920 * 1080:  # Full HD+
                quality["video_quality"] = "Excellent"
                quality["resolution_category"] = "High Definition"
                score += 30
            elif total_pixels >= 1280 * 720:  # HD
                quality["video_quality"] = "Good" 
                quality["resolution_category"] = "Standard Definition"
                score += 25
            elif total_pixels >= 640 * 480:  # SD
                quality["video_quality"] = "Fair"
                quality["resolution_category"] = "Low Definition"
                score += 15
            else:
                quality["video_quality"] = "Poor"
                quality["resolution_category"] = "Very Low Definition"
                score += 5
                quality["issues"].append("Very low resolution")
            
            # FPS assessment
            if metadata.fps >= 30:
                score += 20
            elif metadata.fps >= 24:
                score += 15
            else:
                score += 5
                quality["issues"].append("Low frame rate")
            
            # Format suitability
            quality["format_suitability"] = {
                "social_media": metadata.aspect_ratio in ["9:16 (vertical/mobile)", "1:1 (square)"],
                "web_streaming": metadata.aspect_ratio == "16:9 (widescreen)",
                "mobile_optimized": "vertical" in metadata.aspect_ratio or "mobile" in metadata.aspect_ratio
            }
        
        if audio_result.success and audio_result.audio_segments:
            # Audio quality assessment
            avg_db = audio_result.average_db_level
            
            if avg_db > -10:
                quality["audio_quality"] = "Excellent (very clear)"
                score += 25
            elif avg_db > -20:
                quality["audio_quality"] = "Good (clear and audible)"
                score += 20
            elif avg_db > -30:
                quality["audio_quality"] = "Fair (audible but could be improved)"
                score += 10
            else:
                quality["audio_quality"] = "Poor (too quiet)"
                score += 5
                quality["issues"].append("Audio level too low")
            
            # Check for excessive silence
            if audio_result.silence_percentage > 50:
                quality["issues"].append("High percentage of silence")
            
            # Dynamic range assessment
            if audio_result.dynamic_range < 5:
                quality["issues"].append("Limited audio dynamic range")
        
        quality["overall_score"] = min(score, max_score)
        return quality
    
    def _generate_recommendations(
        self, 
        audio_result: AudioAnalysisResult, 
        video_result: VideoAnalysisResult
    ) -> list[str]:
        """Generate recommendations based on analysis."""
        self.logger.debug("Generating recommendations")
        
        recommendations = []
        
        if video_result.success:
            metadata = video_result.metadata
            
            # Resolution recommendations
            width, height = metadata.resolution
            if width * height < 1280 * 720:
                recommendations.append("🎥 Consider recording in higher resolution (720p or higher)")
            
            # Aspect ratio recommendations
            if "mobile" in metadata.aspect_ratio:
                recommendations.append("📱 Vertical format is optimal for mobile and social media")
            elif metadata.aspect_ratio == "16:9 (widescreen)":
                recommendations.append("🖥️ Widescreen format is good for web and desktop viewing")
            
            # Duration recommendations
            if metadata.duration < 15:
                recommendations.append("⏱️ Very short content - consider adding more content")
            elif metadata.duration > 180:
                recommendations.append("⏱️ Long content - consider breaking into segments")
            else:
                recommendations.append("⏱️ Good duration for social media content")
        
        if audio_result.success:
            # Audio level recommendations
            if audio_result.average_db_level < -25:
                recommendations.append("🔊 Consider audio normalization to improve volume")
            
            # Transcript recommendations
            if audio_result.transcript_segments:
                recommendations.append("✅ Clear speech detected - good for accessibility/subtitles")
            else:
                recommendations.append("🎤 Consider adding voice narration for better engagement")
            
            # Silence recommendations
            if audio_result.silence_percentage > 30:
                recommendations.append("✂️ Consider editing to reduce silent sections")
        
        # Content recommendations
        if (audio_result.success and audio_result.transcript_segments and 
            video_result.success):
            recommendations.append("🎯 Content appears well-structured for tutorial format")
        
        # Format-specific recommendations
        if video_result.success:
            metadata = video_result.metadata
            if "mobile" in metadata.aspect_ratio:
                recommendations.append("📲 Perfect for TikTok, Instagram Stories, YouTube Shorts")
            elif metadata.aspect_ratio == "16:9 (widescreen)":
                recommendations.append("📺 Good for YouTube, Facebook, traditional video platforms")
        
        return recommendations
    
    def _extract_topics_from_text(self, text: str) -> list[Dict[str, Any]]:
        """Extract key topics from text."""
        # Simple keyword-based topic extraction
        topic_keywords = {
            'beauty': ['makeup', 'beauty', 'cosmetics', 'skincare'],
            'fashion': ['outfit', 'clothes', 'style', 'fashion'],
            'food': ['recipe', 'cook', 'food', 'ingredient'],
            'tutorial': ['tutorial', 'how to', 'learn', 'guide'],
            'review': ['review', 'opinion', 'rating', 'recommend'],
            'entertainment': ['fun', 'funny', 'entertainment', 'comedy']
        }
        
        topics = []
        for topic, keywords in topic_keywords.items():
            count = sum(text.count(keyword) for keyword in keywords)
            if count > 0:
                topics.append({"topic": topic, "mentions": count})
        
        return sorted(topics, key=lambda x: x["mentions"], reverse=True)
    
    def _categorize_segment_content(self, text: str) -> str:
        """Categorize content segment."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['hello', 'hi', 'welcome', 'intro']):
            return "Introduction"
        elif any(word in text_lower for word in ['thanks', 'bye', 'see you', 'outro']):
            return "Conclusion"
        elif any(word in text_lower for word in ['step', 'next', 'now', 'then']):
            return "Instruction"
        elif any(word in text_lower for word in ['product', 'brand', 'use', 'apply']):
            return "Product Usage"
        else:
            return "General Content"
    
    def _save_results(self, result: CompleteAnalysisResult) -> None:
        """Save analysis results to files using the session system."""
        session = result.session
        if not session:
            # Fallback to old system if no session
            output_dir = Path(result.output_dir)
            self._save_results_legacy(result, output_dir)
            return
        
        # Save JSON report
        if self.config.output.generate_json_report:
            json_path = session.get_report_path("complete_analysis.json")
            self._save_json_report(result, json_path)
            session.add_output_file(json_path, "Повний звіт аналізу в JSON форматі")
        
        # Save HTML report
        if self.config.output.generate_html_report:
            html_path = session.get_report_path("analysis_report.html")
            self._save_html_report(result, html_path)
            session.add_output_file(html_path, "Інтерактивний HTML звіт")
        
        # Save text summary
        if self.config.output.generate_text_summary:
            text_path = session.get_report_path("analysis_summary.txt")
            self._save_text_summary(result, text_path)
            session.add_output_file(text_path, "Текстовий підсумок аналізу")
            
        self.logger.info(f"📄 Звіти збережено в сесії {session.metadata.session_id}")
            
    def _save_results_legacy(self, result: CompleteAnalysisResult, output_dir: Path) -> None:
        """Legacy method for saving results without session system."""
        # Save JSON report
        if self.config.output.generate_json_report:
            json_path = output_dir / "complete_analysis.json"
            self._save_json_report(result, json_path)
        
        # Save HTML report
        if self.config.output.generate_html_report:
            html_path = output_dir / "analysis_report.html"
            self._save_html_report(result, html_path)
        
        # Save text summary
        if self.config.output.generate_text_summary:
            text_path = output_dir / "analysis_summary.txt"
            self._save_text_summary(result, text_path)
    
    def _save_json_report(self, result: CompleteAnalysisResult, json_path: Path) -> None:
        """Save JSON report."""
        try:
            # Convert to serializable dictionary
            data = self._result_to_dict(result)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.debug(f"JSON report saved to {json_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON report: {e}")
    
    def _save_html_report(self, result: CompleteAnalysisResult, html_path: Path) -> None:
        """Save HTML report."""
        try:
            from video_shots.exports.html_export import HTMLReportGenerator
            
            generator = HTMLReportGenerator()
            generator.generate_complete_report(result, html_path)
            
            self.logger.debug(f"HTML report saved to {html_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save HTML report: {e}")
    
    def _save_text_summary(self, result: CompleteAnalysisResult, text_path: Path) -> None:
        """Save text summary."""
        try:
            summary = self._generate_text_summary(result)
            
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            self.logger.debug(f"Text summary saved to {text_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save text summary: {e}")
    
    def _result_to_dict(self, result: CompleteAnalysisResult) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "success": result.success,
            "video_path": result.video_path,
            "output_dir": result.output_dir,
            "timestamp": result.timestamp,
            "duration": result.duration,
            "audio_result": self._audio_result_to_dict(result.audio_result),
            "video_result": self._video_result_to_dict(result.video_result),
            "content_analysis": result.content_analysis,
            "technical_quality": result.technical_quality,
            "recommendations": result.recommendations,
            "error_message": result.error_message
        }
    
    def _audio_result_to_dict(self, audio_result: AudioAnalysisResult) -> Dict[str, Any]:
        """Convert audio result to dictionary."""
        return {
            "success": audio_result.success,
            "audio_path": audio_result.audio_path,
            "transcript_path": audio_result.transcript_path,
            "duration": audio_result.duration,
            "sample_rate": audio_result.sample_rate,
            "audio_segments": [asdict(seg) for seg in audio_result.audio_segments],
            "transcript_segments": [asdict(seg) for seg in audio_result.transcript_segments],
            "statistics": {
                "average_db_level": audio_result.average_db_level,
                "dynamic_range": audio_result.dynamic_range,
                "silence_percentage": audio_result.silence_percentage,
                "total_speech_segments": audio_result.total_speech_segments
            },
            "error_message": audio_result.error_message
        }
    
    def _video_result_to_dict(self, video_result: VideoAnalysisResult) -> Dict[str, Any]:
        """Convert video result to dictionary."""
        return {
            "success": video_result.success,
            "metadata": asdict(video_result.metadata),
            "keyframes": [asdict(kf) for kf in video_result.keyframes],
            "shots": [asdict(shot) for shot in video_result.shots],
            "output_dir": video_result.output_dir,
            "statistics": {
                "total_keyframes": video_result.total_keyframes,
                "total_shots": video_result.total_shots,
                "keyframes_per_minute": video_result.keyframes_per_minute
            },
            "error_message": video_result.error_message
        }
    
    def _generate_text_summary(self, result: CompleteAnalysisResult) -> str:
        """Generate text summary."""
        summary = f"""
COMPLETE VIDEO ANALYSIS REPORT
Generated: {result.timestamp}
Duration: {result.duration:.2f} seconds

VIDEO FILE
==========
Path: {result.video_path}
Output Directory: {result.output_dir}

"""
        
        if result.video_result.success:
            metadata = result.video_result.metadata
            summary += f"""VIDEO ANALYSIS
==============
Duration: {metadata.duration:.1f} seconds
FPS: {metadata.fps:.1f}
Resolution: {metadata.resolution[0]}x{metadata.resolution[1]} ({metadata.aspect_ratio})
Format: {metadata.format}
Keyframes Extracted: {result.total_keyframes}
Shots Analyzed: {result.video_result.total_shots}

"""
        
        if result.audio_result.success:
            summary += f"""AUDIO ANALYSIS
==============
Audio Quality: {result.technical_quality.get('audio_quality', 'Unknown')}
Average dB Level: {result.audio_result.average_db_level:.1f}dB
Dynamic Range: {result.audio_result.dynamic_range:.1f}dB
Silence Percentage: {result.audio_result.silence_percentage:.1f}%
Speech Segments: {result.total_speech_segments}
Audio Segments Analyzed: {result.total_audio_segments}

"""
        
        summary += f"""CONTENT ANALYSIS
================
Content Type: {result.content_analysis.get('content_type', 'Unknown')}
Language: {result.content_analysis.get('estimated_language', 'Unknown')}
Pacing: {result.content_analysis.get('pacing', 'Unknown')}

Key Topics:
"""
        
        for topic in result.content_analysis.get('key_topics', []):
            summary += f"- {topic['topic'].title()}: {topic['mentions']} mentions\n"
        
        summary += f"""
TECHNICAL QUALITY
=================
Overall Score: {result.technical_quality.get('overall_score', 0)}/100
Video Quality: {result.technical_quality.get('video_quality', 'Unknown')}
Audio Quality: {result.technical_quality.get('audio_quality', 'Unknown')}

RECOMMENDATIONS
===============
"""
        
        for rec in result.recommendations:
            summary += f"• {rec}\n"
        
        return summary
    
    def _create_error_result(
        self, 
        video_path: str, 
        error_message: str, 
        output_dir: Optional[Union[str, Path]]
    ) -> CompleteAnalysisResult:
        """Create error result."""
        from video_shots.analysis.unified_audio_analyzer import AudioAnalysisResult
        from video_shots.analysis.unified_video_analyzer import VideoAnalysisResult, VideoMetadata
        
        return CompleteAnalysisResult(
            success=False,
            video_path=video_path,
            output_dir=str(output_dir) if output_dir else "",
            timestamp=datetime.now().isoformat(),
            duration=0.0,
            session_id="error",
            audio_result=AudioAnalysisResult(
                success=False,
                video_path=video_path,
                audio_path=None,
                transcript_path=None,
                duration=0.0,
                sample_rate=None,
                audio_segments=[],
                transcript_segments=[],
                error_message=error_message
            ),
            video_result=VideoAnalysisResult(
                success=False,
                metadata=VideoMetadata(
                    path=video_path,
                    duration=0.0,
                    fps=0.0,
                    frame_count=0,
                    resolution=(0, 0),
                    aspect_ratio="unknown",
                    format="unknown"
                ),
                keyframes=[],
                shots=[],
                output_dir="",
                error_message=error_message
            ),
            content_analysis={},
            technical_quality={},
            recommendations=[],
            error_message=error_message
        )