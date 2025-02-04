import boto3
import os
from ftplib import FTP

S3_BUCKET = "your-bucket-name"
LOCAL_STORAGE = "data/"

def fetch_from_s3(bucket, key):
    s3 = boto3.client("s3")
    file_path = os.path.join(LOCAL_STORAGE, os.path.basename(key))
    s3.download_file(bucket, key, file_path)
    return file_path

def fetch_from_ftp(server, file_path):
    ftp = FTP(server)
    ftp.login()
    local_path = os.path.join(LOCAL_STORAGE, os.path.basename(file_path))
    with open(local_path, "wb") as file:
        ftp.retrbinary(f"RETR {file_path}", file.write)
    ftp.quit()
    return local_path
