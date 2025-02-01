from typing import List, Dict, Any
from pathlib import Path
import aiofiles
import tempfile
import asyncio
import os,sys
path = os.getcwd()
sys.path.append(path)
from docwain.utils.logger import setup_logger

logger = setup_logger()

async def save_uploaded_file(file_data: bytes, filename: str) -> Path:
    """Save uploaded file data to temporary file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            tmp_path = Path(tmp.name)
            async with aiofiles.open(tmp_path, 'wb') as f:
                await f.write(file_data)
        return tmp_path
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise

async def cleanup_temp_files(file_paths: List[Path]):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            file_path.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Error cleaning up temp file {file_path}: {e}")