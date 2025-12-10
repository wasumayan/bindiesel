"""
Logging module for Bin Diesel system
Provides structured logging with different log levels
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
import config

# Create logs directory if it doesn't exist
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Log file with timestamp
LOG_FILE = LOG_DIR / f"bindiesel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logger(name, level=logging.INFO):
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler (all logs)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (INFO and above, or DEBUG if config.DEBUG_MODE)
    console_level = logging.DEBUG if config.DEBUG_MODE else logging.INFO
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def log_error(logger, error, context=""):
    """
    Log an error with context and traceback
    
    Args:
        logger: Logger instance
        error: Exception object
        context: Additional context string
    """
    import traceback
    error_msg = f"{context}: {type(error).__name__}: {str(error)}"
    logger.error(error_msg)
    if config.DEBUG_MODE:
        logger.debug(f"Traceback:\n{''.join(traceback.format_exception(type(error), error, error.__traceback__))}")

def log_warning(logger, message, context=""):
    """Log a warning with optional context"""
    if context:
        logger.warning(f"{context}: {message}")
    else:
        logger.warning(message)

def log_info(logger, message, context=""):
    """Log info with optional context"""
    if context:
        logger.info(f"{context}: {message}")
    else:
        logger.info(message)

def log_debug(logger, message, context=""):
    """Log debug message with optional context"""
    if context:
        logger.debug(f"{context}: {message}")
    else:
        logger.debug(message)

