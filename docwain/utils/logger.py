import sys
from loguru import logger
import os,sys
path = os.getcwd()
sys.path.append(path)
from docwain.config import Config


def setup_logger():
    """Configure application logger"""
    logger.remove()  # Remove default handler

    # Add custom handlers
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG" if Config.DEBUG else "INFO",
        colorize=True
    )

    # Add file handler for errors
    logger.add(
        "logs/error.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="1 day",
        retention="7 days"
    )

    return logger