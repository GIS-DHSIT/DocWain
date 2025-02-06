import boto3
from azure.storage.blob import BlobServiceClient
import ftplib
import os


def fetch_files_from_source(source_type, s3_bucket=None, azure_container=None, ftp_path=None):
    files = []

    if source_type == "s3":
        s3 = boto3.client("s3")
        objects = s3.list_objects_v2(Bucket=s3_bucket)["Contents"]
        for obj in objects:
            file_data = s3.get_object(Bucket=s3_bucket, Key=obj["Key"])["Body"].read()
            files.append((obj["Key"], file_data))

    elif source_type == "azure":
        blob_service_client = BlobServiceClient.from_connection_string("AZURE_CONNECTION_STRING")
        container_client = blob_service_client.get_container_client(azure_container)
        blobs = container_client.list_blobs()
        for blob in blobs:
            blob_data = container_client.download_blob(blob.name).readall()
            files.append((blob.name, blob_data))

    elif source_type == "ftp":
        ftp = ftplib.FTP("FTP_SERVER")
        ftp.login("FTP_USERNAME", "FTP_PASSWORD")
        ftp.cwd(ftp_path)
        file_names = ftp.nlst()
        for file_name in file_names:
            file_data = []
            ftp.retrbinary(f"RETR {file_name}", file_data.append)
            files.append((file_name, b"".join(file_data)))

    return files
