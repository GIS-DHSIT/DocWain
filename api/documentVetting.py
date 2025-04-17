from io import BytesIO
import pandas as pd
from api.config import Config
from azure.storage.blob import BlobServiceClient

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
    """
    Process the vetting configuration and return the processed data.

    Returns:
        dict: The processed vetting data.
    """
    try:
        VettingConfig = readVettingConf()
        if not VettingConfig:
            print("No vetting configuration found.")
            return None
        initialPoints = 100
        words = []
        for key, value in VettingConfig.items():
            for cond in value:
                words.append(list(cond.values()))
        vettingWords = [item for sublist in words for item in sublist]
        cleaned_list = [str(s).replace("\xa0", " ") for s in vettingWords]

        for word in cleaned_list:
            if word in docContent:
                initialPoints -= 5
                points = max(initialPoints, 0)
                processed_data = points
                return processed_data
            else:
                return 100

    except Exception as e:
        print(f"Error processing vetting configuration: {e}")
        return None

    print(vettingProcessor("testing the document `vetting` processor"))