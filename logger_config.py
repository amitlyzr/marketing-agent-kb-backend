"""
Professional logging configuration for Marketing Agent Application
Provides structured logging with different levels and formatters for API and background tasks
"""

import logging
import logging.handlers
import sys
import os
from datetime import datetime
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels in console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            record.levelname = colored_levelname
        
        # Format the message
        formatted = super().format(record)
        
        # Reset the levelname for other formatters
        record.levelname = levelname
        
        return formatted


class StructuredFormatter(logging.Formatter):
    """Structured JSON-like formatter for file logging"""
    
    def format(self, record):
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'email'):
            log_entry['email'] = record.email
        if hasattr(record, 'endpoint'):
            log_entry['endpoint'] = record.endpoint
        if hasattr(record, 'status_code'):
            log_entry['status_code'] = record.status_code
        if hasattr(record, 'execution_time'):
            log_entry['execution_time_ms'] = record.execution_time
        if hasattr(record, 'error_type'):
            log_entry['error_type'] = record.error_type
            
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        # Format as structured string
        parts = [f"{k}={v}" for k, v in log_entry.items()]
        return " | ".join(parts)


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Setup a professional logger with both console and file handlers
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Console Handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = ColoredFormatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File Handler for all logs (with rotation)
    file_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "marketing_agent.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = StructuredFormatter()
    file_handler.setFormatter(file_format)
    
    # Error File Handler (errors only)
    error_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "marketing_agent_errors.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_format)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
    return logger


def setup_api_logger() -> logging.Logger:
    """Setup logger specifically for FastAPI application"""
    return setup_logger("marketing_agent.api", os.getenv("LOG_LEVEL", "INFO"))


def setup_scheduler_logger() -> logging.Logger:
    """Setup logger specifically for scheduler/cron jobs"""
    return setup_logger("marketing_agent.scheduler", os.getenv("LOG_LEVEL", "INFO"))


def setup_email_logger() -> logging.Logger:
    """Setup logger specifically for email operations"""
    return setup_logger("marketing_agent.email", os.getenv("LOG_LEVEL", "INFO"))


def setup_interview_processing_logger() -> logging.Logger:
    """Setup logger specifically for interview processing operations"""
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("marketing_agent.interview_processing")
    logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Console Handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = ColoredFormatter(
        '%(asctime)s | %(levelname)-8s | INTERVIEW_PROCESSING | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # Dedicated File Handler for interview processing (with rotation)
    interview_file_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "interview_processing.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    interview_file_handler.setLevel(logging.DEBUG)
    interview_file_format = StructuredFormatter()
    interview_file_handler.setFormatter(interview_file_format)
    
    # Error File Handler for interview processing errors only
    interview_error_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "interview_processing_errors.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    interview_error_handler.setLevel(logging.ERROR)
    interview_error_handler.setFormatter(interview_file_format)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(interview_file_handler)
    logger.addHandler(interview_error_handler)
    
    return logger


def log_api_request(logger: logging.Logger, endpoint: str, method: str, user_id: str = None):
    """Helper function to log API requests - Only log errors"""
    pass  # Don't log every request


def log_api_response(logger: logging.Logger, endpoint: str, status_code: int, execution_time: float, user_id: str = None):
    """Helper function to log API responses - Only log errors"""
    if status_code >= 400:
        logger.error(f"API Error: {endpoint} - {status_code}")
    elif execution_time > 5000:  # Only log slow requests (>5s)
        logger.warning(f"Slow API: {endpoint} - {execution_time:.0f}ms")


def log_email_operation(logger: logging.Logger, operation: str, email: str, user_id: str, success: bool, details: str = None):
    """Helper function to log email operations - Only failures and important events"""
    if not success:
        logger.error(f"Email {operation} failed: {email} - {details}")
    elif operation == "send":
        logger.info(f"Email sent: {email}")


def log_database_operation(logger: logging.Logger, operation: str, collection: str, user_id: str = None, details: str = None):
    """Helper function to log database operations - Only log errors"""
    pass  # Don't log routine DB operations


def log_scheduler_event(logger: logging.Logger, event: str, user_id: str = None, details: str = None):
    """Helper function to log scheduler events - Only important events"""
    if event in ["cycle_start", "auto_started_interviews", "daily_trigger"]:
        logger.info(f"Scheduler: {event} - {details}" if details else f"Scheduler: {event}")


def log_interview_processing_step(logger: logging.Logger, step: str, user_id: str, email: str, details: str = None):
    """Helper function to log interview processing steps"""
    if details:
        logger.info(f"Interview Processing [{step}] for {user_id}/{email}: {details}")
    else:
        logger.info(f"Interview Processing [{step}] for {user_id}/{email}")


def log_interview_processing_error(logger: logging.Logger, step: str, user_id: str, email: str, error: str):
    """Helper function to log interview processing errors"""
    logger.error(f"Interview Processing [{step}] FAILED for {user_id}/{email}: {error}")


# Global loggers for easy import
api_logger = setup_api_logger()
scheduler_logger = setup_scheduler_logger()
email_logger = setup_email_logger()
interview_processing_logger = setup_interview_processing_logger()

# Log startup
api_logger.info("Marketing Agent Application - Logging System Initialized")
