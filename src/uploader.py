import shutil
import os
from pathlib import Path
from typing import List, Dict, Union
import socket
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from ftplib import FTP
import boto3
import tempfile
import logging
from datetime import datetime

from streamlit.runtime.uploaded_file_manager import UploadedFile
from .config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileUploader:
    def __init__(self, remove_old_files: bool = True):
        self.remove_old_files = remove_old_files
        self._setup_directories()
        self.temp_dir = tempfile.mkdtemp()

    def _setup_directories(self):
        """Initialize or clean directories as needed"""
        if self.remove_old_files:
            shutil.rmtree(Config.Path.DATABASE_DIR, ignore_errors=True)
            shutil.rmtree(Config.Path.DOCUMENTS_DIR, ignore_errors=True)
        Config.Path.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    def _generate_unique_filename(self, original_name: str) -> str:
        """Generate a unique filename to avoid conflicts"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(original_name).stem
        extension = Path(original_name).suffix
        return f"{base_name}_{timestamp}{extension}"

    def _save_file(self, file_path: Union[str, Path], content: bytes) -> Path:
        """Save file content to the documents directory"""
        dest_path = Config.Path.DOCUMENTS_DIR / self._generate_unique_filename(str(file_path))
        with dest_path.open("wb") as f:
            f.write(content)
        logger.info(f"Saved file: {dest_path}")
        return dest_path

    def process_uploaded_files(self, files: List[UploadedFile]) -> List[Path]:
        """Process files uploaded through Streamlit"""
        file_paths = []
        for file in files:
            try:
                file_path = self._save_file(file.name, file.getvalue())
                file_paths.append(file_path)
            except Exception as e:
                logger.error(f"Error processing uploaded file {file.name}: {str(e)}")
        return file_paths

    def process_directory(self, directory_path: Union[str, Path]) -> List[Path]:
        """Process files from a local directory"""
        file_paths = []
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return file_paths

        for file_path in directory.glob("**/*.pdf"):
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                saved_path = self._save_file(file_path.name, content)
                file_paths.append(saved_path)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")

        return file_paths

    async def process_ftp(self, ftp_configs: List[Dict]) -> List[Path]:
        """Process files from FTP servers with connection pooling"""
        file_paths = []
        for config in ftp_configs:
            try:
                # Use a custom connection per server to avoid overwhelming sockets
                ftp = FTP()
                ftp.set_pasv(True)  # Use passive mode for better compatibility

                # Configure timeout and buffer size
                ftp.connect(
                    host=config['host'],
                    port=21,
                    timeout=30
                )
                ftp.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
                ftp.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8192)

                try:
                    ftp.login(user=config['user'], passwd=config['password'])
                    ftp.cwd(config['path'])

                    # Get file list in chunks
                    try:
                        files = []
                        ftp.retrlines('NLST', files.append)

                        # Process files in smaller batches
                        batch_size = 5
                        for i in range(0, len(files), batch_size):
                            batch = files[i:i + batch_size]
                            for filename in batch:
                                if filename.lower().endswith('.pdf'):
                                    try:
                                        temp_path = Path(self.temp_dir) / filename
                                        with open(temp_path, 'wb') as temp_file:
                                            ftp.retrbinary(f'RETR {filename}', temp_file.write, blocksize=8192)

                                        with open(temp_path, 'rb') as f:
                                            content = f.read()
                                        saved_path = self._save_file(filename, content)
                                        file_paths.append(saved_path)

                                        # Clean up temp file
                                        temp_path.unlink()

                                        # Add a small delay between files
                                        await asyncio.sleep(0.1)

                                    except Exception as e:
                                        logger.error(f"Error processing FTP file {filename}: {str(e)}")
                                        continue

                            # Add delay between batches
                            await asyncio.sleep(0.5)

                    except Exception as e:
                        logger.error(f"Error listing files in directory: {str(e)}")

                finally:
                    try:
                        ftp.quit()
                    except:
                        ftp.close()

            except Exception as e:
                logger.error(f"FTP connection error for {config['host']}: {str(e)}")
                continue

        return file_paths

    async def process_s3(self, s3_configs: List[Dict]) -> List[Path]:
        """Process files from S3 buckets with connection pooling"""
        file_paths = []
        try:
            session = boto3.Session()
            s3 = session.client('s3', config=boto3.Config(
                max_pool_connections=10,
                connect_timeout=30,
                read_timeout=30,
                retries={'max_attempts': 3}
            ))

            for config in s3_configs:
                try:
                    bucket_name = config['bucket']
                    prefix = config.get('prefix', '')

                    # Process objects in smaller batches
                    paginator = s3.get_paginator('list_objects_v2')
                    for page in paginator.paginate(
                            Bucket=bucket_name,
                            Prefix=prefix,
                            PaginationConfig={'PageSize': 10}
                    ):
                        if 'Contents' in page:
                            for obj in page['Contents']:
                                if obj['Key'].lower().endswith('.pdf'):
                                    try:
                                        temp_path = Path(self.temp_dir) / Path(obj['Key']).name

                                        # Download with transfer configuration
                                        transfer_config = boto3.s3.transfer.TransferConfig(
                                            max_concurrency=1,
                                            multipart_threshold=1024 * 1024 * 8,  # 8MB
                                            multipart_chunksize=1024 * 1024 * 8,  # 8MB
                                        )

                                        s3.download_file(
                                            bucket_name,
                                            obj['Key'],
                                            str(temp_path),
                                            Config=transfer_config
                                        )

                                        with open(temp_path, 'rb') as f:
                                            content = f.read()
                                        saved_path = self._save_file(Path(obj['Key']).name, content)
                                        file_paths.append(saved_path)

                                        # Clean up temp file
                                        temp_path.unlink()

                                        # Add a small delay between files
                                        await asyncio.sleep(0.1)

                                    except Exception as e:
                                        logger.error(f"Error processing S3 file {obj['Key']}: {str(e)}")
                                        continue

                except Exception as e:
                    logger.error(f"Error processing S3 bucket {config['bucket']}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"S3 client error: {str(e)}")

        return file_paths


async def upload_files(
        files: List[UploadedFile] = None,
        directory: Union[str, Path] = None,
        ftp_configs: List[Dict] = None,
        s3_configs: List[Dict] = None,
        remove_old_files: bool = True
) -> List[Path]:
    """
    Process files from multiple sources and save them to the documents directory.

    Args:
        files: List of files uploaded through Streamlit
        directory: Path to local directory containing PDF files
        ftp_configs: List of FTP server configurations
        s3_configs: List of S3 bucket configurations
        remove_old_files: Whether to clean up old files before processing

    Returns:
        List of paths to saved files
    """
    uploader = FileUploader(remove_old_files=remove_old_files)
    all_file_paths = []

    # Process uploaded files
    if files:
        file_paths = uploader.process_uploaded_files(files)
        all_file_paths.extend(file_paths)

    # Process local directory
    if directory:
        file_paths = uploader.process_directory(directory)
        all_file_paths.extend(file_paths)

    # Process FTP servers
    if ftp_configs:
        file_paths = await uploader.process_ftp(ftp_configs)
        all_file_paths.extend(file_paths)

    # Process S3 buckets
    if s3_configs:
        file_paths = await uploader.process_s3(s3_configs)
        all_file_paths.extend(file_paths)

    if not all_file_paths:
        logger.warning("No files were processed from any source")

    return all_file_paths