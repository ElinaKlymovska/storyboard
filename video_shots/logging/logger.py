"""Unified logging system for the video analysis application."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class VideoAnalysisLogger:
    """Centralized logger for video analysis operations."""
    
    def __init__(self, name: str = "video_analysis", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup console and file handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_dir / f"video_analysis_{timestamp}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    def log_operation_start(self, operation: str, **details) -> None:
        """Log the start of an operation."""
        details_str = ", ".join(f"{k}={v}" for k, v in details.items())
        self.info(f"🚀 Starting {operation}" + (f" ({details_str})" if details_str else ""))
    
    def log_operation_complete(self, operation: str, duration: Optional[float] = None, **results) -> None:
        """Log the completion of an operation."""
        duration_str = f" in {duration:.2f}s" if duration else ""
        results_str = ", ".join(f"{k}={v}" for k, v in results.items())
        self.info(f"✅ Completed {operation}{duration_str}" + (f" ({results_str})" if results_str else ""))
    
    def log_operation_error(self, operation: str, error: Exception) -> None:
        """Log an operation error."""
        self.error(f"❌ Failed {operation}: {str(error)}")
    
    def log_progress(self, current: int, total: int, operation: str = "") -> None:
        """Log progress for long operations."""
        percentage = (current / total) * 100 if total > 0 else 0
        op_str = f" {operation}" if operation else ""
        self.info(f"📊 Progress{op_str}: {current}/{total} ({percentage:.1f}%)")


# Global logger instance
_logger_instance: Optional[VideoAnalysisLogger] = None


def get_logger(name: str = "video_analysis", level: str = "INFO") -> VideoAnalysisLogger:
    """Get or create the global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = VideoAnalysisLogger(name, level)
    return _logger_instance


def setup_logging(level: str = "INFO") -> VideoAnalysisLogger:
    """Setup logging for the application."""
    return get_logger(level=level)


# Convenience functions
def log_info(message: str, **kwargs) -> None:
    """Log info message using global logger."""
    get_logger().info(message, **kwargs)


def log_debug(message: str, **kwargs) -> None:
    """Log debug message using global logger."""
    get_logger().debug(message, **kwargs)


def log_warning(message: str, **kwargs) -> None:
    """Log warning message using global logger."""
    get_logger().warning(message, **kwargs)


def log_error(message: str, **kwargs) -> None:
    """Log error message using global logger."""
    get_logger().error(message, **kwargs)


def log_operation_start(operation: str, **details) -> None:
    """Log operation start using global logger."""
    get_logger().log_operation_start(operation, **details)


def log_operation_complete(operation: str, duration: Optional[float] = None, **results) -> None:
    """Log operation completion using global logger."""
    get_logger().log_operation_complete(operation, duration, **results)


def log_operation_error(operation: str, error: Exception) -> None:
    """Log operation error using global logger."""
    get_logger().log_operation_error(operation, error)


def log_progress(current: int, total: int, operation: str = "") -> None:
    """Log progress using global logger."""
    get_logger().log_progress(current, total, operation)