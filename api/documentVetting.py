
from io import BytesIO
import pandas as pd
from api.config import Config
from azure.storage.blob import BlobServiceClient
import logging

def readVettingConf(file=Config.VettingAzureBlob.AZURE_BLOB_FILE_NAME):
    """
    Read the vetting configuration from a JSON file.

    Returns:
        dict: The vetting configuration.
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(Config.VettingAzureBlob.AZURE_BLOB_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(Config.VettingAzureBlob.AZURE_BLOB_CONTAINER_NAME)
        blob_client = container_client.get_blob_client(file)
        blob_data = blob_client.download_blob().readall()
        excel_data = pd.ExcelFile(BytesIO(blob_data))
        sheet_names = excel_data.sheet_names
        VettingConfig = {}
        for sheets in sheet_names:
            df = pd.read_excel(excel_data, sheet_name=sheets)
            VettingConfig[sheets] = df.to_dict(orient='records')
        return VettingConfig

    except Exception as e:
        print(f"Error reading Excel file from blob: {e}")
        return None

def vettingProcessor(docContent):
    """Process vetting configuration and return points."""
    try:
        VettingConfig = readVettingConf()
        if not VettingConfig:
            logging.warning("No vetting configuration found.")
            return 100

        initialPoints = 100

        # Flatten docContent if it's a dict
        if isinstance(docContent, dict):
            text_content = " ".join(
                str(v) for file_data in docContent.values() for v in file_data.values()
            )
        else:
            text_content = str(docContent)

        # Flatten vetting words
        words = []
        for sheet, rows in VettingConfig.items():
            for row in rows:
                words.extend(list(row.values()))

        cleaned_list = [str(s).replace("\xa0", " ").strip() for s in words]

        for word in cleaned_list:
            if word in text_content:
                initialPoints -= 5

        return max(initialPoints, 0)

    except Exception as e:
        logging.error(f"Error processing vetting configuration: {e}")
        return 100
