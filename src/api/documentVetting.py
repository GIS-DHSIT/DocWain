
from io import BytesIO
import re
import json
import logging
import os
import pandas as pd
from src.api.config import Config
from azure.storage.blob import BlobServiceClient

# Patterns for basic PII detection/masking
PII_PATTERNS = [
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]"),
    (r"\b(?:\+?[\d]{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b", "[PHONE]"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    (r"\b(?:\d[ -]*?){13,16}\b", "[CARD]"),
    (r"\b[A-Z]{2}\d{6}\b", "[PASSPORT]"),
]

# Terms that indicate high confidentiality; presence will block training
HIGH_CONFIDENTIAL_TERMS = {
    "top secret",
    "strictly confidential",
    "classified",
    "do not distribute",
    "internal use only",
    "attorney-client privilege",
}

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
            text_parts = []
            for file_data in docContent.values():
                if isinstance(file_data, dict):
                    for key, val in file_data.items():
                        if key == "embeddings":
                            continue  # skip numeric arrays
                        text_parts.append(str(val))
                else:
                    text_parts.append(str(file_data))
            text_content = " ".join(text_parts)
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


def detect_pii_with_ai(text: str):
    """
    Best-effort AI-driven PII detection with regex fallback.
    Returns list of items: {"type": str, "value": str}
    """
    detected = []

    # Regex baseline
    for pattern, label in PII_PATTERNS:
        for match in re.findall(pattern, text):
            detected.append({"type": label.strip("[]"), "value": match})

    # Optional local Ollama refinement (llama3.2 by default)
    try:
        import ollama
        model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
        prompt = (
            "Identify PII (emails, phone numbers, SSNs, credit cards, passport numbers, addresses, bank accounts) "
            "in the provided text. Respond ONLY with JSON array of objects "
            '[{\"type\": \"<pii_type>\", \"value\": \"<exact_snippet>\"}]. '
            "Text:\n"
            f"{text[:4000]}"
        )
        response = ollama.generate(model=model_name, prompt=prompt)
        payload = response.get("response") if isinstance(response, dict) else None
        if payload:
            parsed = json.loads(payload)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "value" in item:
                        detected.append({
                            "type": item.get("type", "PII"),
                            "value": str(item["value"])
                        })
    except Exception as e:
        logging.warning(f"Ollama PII detection fallback used due to: {e}")

    # Deduplicate by value/type
    unique = {}
    for item in detected:
        key = (item.get("type"), item.get("value"))
        unique[key] = item
    return list(unique.values())


def _mask_text(text: str):
    """Mask PII in a string and return masked text, count, high_confidential flag, detected items."""
    count = 0
    high_confidential = False
    lower_text = text.lower()
    for term in HIGH_CONFIDENTIAL_TERMS:
        if term in lower_text:
            high_confidential = True
            break

    detected_items = detect_pii_with_ai(text)
    masked = text
    for pattern, replacement in PII_PATTERNS:
        def _sub(match):
            nonlocal count
            count += 1
            return replacement
        masked = re.sub(pattern, _sub, masked)
    count = count or len(detected_items)
    return masked, count, high_confidential, detected_items


def mask_document_content(doc_content):
    """
    Mask PII across document content.
    Returns (masked_content, total_pii_count, high_confidential_flag, pii_items).
    """
    total = 0
    high_conf = False
    found_items = []

    def _mask_value(val):
        nonlocal total, high_conf, found_items
        if isinstance(val, str):
            masked_val, c, h, items = _mask_text(val)
            total += c
            high_conf = high_conf or h
            if items:
                found_items.extend(items)
            return masked_val
        if isinstance(val, list):
            return [_mask_value(item) for item in val]
        if isinstance(val, dict):
            new_dict = {}
            for k, v in val.items():
                if k == "embeddings":
                    # Drop embeddings here to force recompute after masking
                    continue
                new_dict[k] = _mask_value(v)
            return new_dict
        return val

    masked = _mask_value(doc_content)
    # Deduplicate found items
    unique = {}
    for item in found_items:
        key = (item.get("type"), item.get("value"))
        unique[key] = item
    return masked, total, high_conf, list(unique.values())
