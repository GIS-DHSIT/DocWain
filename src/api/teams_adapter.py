import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests
from azure.storage.blob import BlobServiceClient

try:
    from botbuilder.core.teams import FileDownloadInfo  # type: ignore
except ImportError:  # pragma: no cover - fallback for alternate package layout
    try:
        from botbuilder.schema.teams import FileDownloadInfo  # type: ignore
    except ImportError as exc:  # pragma: no cover - explicit guidance when dependency missing
        raise ImportError(
            "botbuilder-core is required for Teams attachment handling"
        ) from exc

from src.api.config import Config
from src.api.dataHandler import fileProcessor, train_on_document
from src.api.dw_newron import answer_question

logger = logging.getLogger(__name__)


class TeamsAuthError(Exception):
    """Raised when the Teams request fails authentication."""


def _get_header(headers: Dict[str, str], key: str) -> str:
    return headers.get(key) or headers.get(key.lower()) or headers.get(key.upper()) or ""


def verify_shared_secret(headers: Dict[str, str]) -> bool:
    """Check shared secret (simple header-based auth for Teams outbound calls)."""
    expected = Config.Teams.SHARED_SECRET
    if not expected:
        # No secret configured; allow for local/dev use.
        return True

    candidate = _get_header(headers, "x-teams-shared-secret") or _get_header(headers, "x-teams-signature")
    if not candidate:
        candidate = _get_header(headers, "authorization")
        if candidate.lower().startswith("bearer "):
            candidate = candidate[7:].strip()

    if candidate != expected:
        logger.warning("Invalid Teams shared secret provided")
        return False
    return True


def _strip_mentions(text: str) -> str:
    """Remove <at> mentions so the question text is clean."""
    if not text:
        return ""
    return re.sub(r"<at[^>]*>.*?</at>", "", text).strip()


def extract_question(activity: Dict[str, Any]) -> str:
    """Pull the user question from the Teams activity payload."""
    return _strip_mentions((activity.get("text") or "").strip())


def extract_user_id(activity: Dict[str, Any]) -> str:
    """Derive a stable user id from Teams activity."""
    user = activity.get("from", {}) or {}
    return (
        user.get("aadObjectId")
        or user.get("id")
        or user.get("userPrincipalName")
        or "teams_user"
    )


def extract_session_id(activity: Dict[str, Any]) -> str:
    """Use conversation id as a lightweight session key."""
    convo = activity.get("conversation") or {}
    return convo.get("id") or "teams-session"


def format_sources(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return ""
    lines = []
    for src in sources[:5]:
        name = src.get("source_name") or src.get("source_id") or "Source"
        excerpt = (src.get("excerpt") or "")[:180]
        if excerpt:
            lines.append(f"- {name}: {excerpt}")
        else:
            lines.append(f"- {name}")
    return "\n\nSources:\n" + "\n".join(lines)


def build_teams_message(answer_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Construct a Bot Framework message activity from the DocWain answer."""
    text = answer_payload.get("response") or "I could not generate a response."
    sources_text = format_sources(answer_payload.get("sources") or [])
    if sources_text:
        text = f"{text}\n\n{sources_text}"

    return {
        "type": "message",
        "text": text,
    }


def _upload_to_blob(file_path: str, filename: str, subscription_id: str) -> None:
    """Upload a processed attachment to Azure Blob Storage if configured."""
    connection_string = getattr(Config.Teams, "BLOB_CONNECTION_STRING", "")
    container_name = getattr(Config.Teams, "BLOB_CONTAINER", "")
    prefix = getattr(Config.Teams, "BLOB_PATH_PREFIX", "") or getattr(Config.Teams, "UPLOAD_DIR", "")

    if not connection_string or not container_name:
        logger.info("Teams blob storage is not configured; skipping upload.")
        return

    blob_name_parts = [prefix.strip("/"), subscription_id.strip("/"), filename]
    blob_name = "/".join(part for part in blob_name_parts if part)

    try:
        service = BlobServiceClient.from_connection_string(connection_string)
        container = service.get_container_client(container_name)
        try:
            container.create_container()
        except Exception:
            # Container likely already exists; safe to ignore
            pass

        with open(file_path, "rb") as data:
            container.upload_blob(name=blob_name, data=data, overwrite=True)

        logger.info("Uploaded Teams attachment to blob storage at %s/%s", container_name, blob_name)
    except Exception as exc:
        logger.error("Failed to upload Teams attachment to blob storage: %s", exc)


async def handle_attachment_activity(activity: Dict[str, Any]) -> Dict[str, Any]:
    """Handle Teams activities that include file attachments."""
    attachments = activity.get("attachments") or []
    if not attachments:
        return {"type": "message", "text": "I did not find any attachments to process."}

    upload_dir = os.environ.get("TEAMS_UPLOAD_DIR", getattr(Config.Teams, "UPLOAD_DIR", "/tmp"))
    os.makedirs(upload_dir, exist_ok=True)

    subscription_id = (activity.get("conversation") or {}).get("id") or getattr(
        Config.Teams, "DEFAULT_SUBSCRIPTION", "default"
    )

    processed = False

    for attachment in attachments:
        content_type = (attachment.get("contentType") or "").lower()
        if "file.download.info" not in content_type:
            continue

        file_download = FileDownloadInfo.deserialize(attachment.get("content") or {})
        download_url = getattr(file_download, "download_url", None)
        if not download_url:
            logger.warning("Attachment missing download_url; skipping.")
            continue

        filename = (
            getattr(file_download, "name", None)
            or getattr(file_download, "unique_id", None)
            or getattr(file_download, "id", None)
            or "teams-upload"
        )
        temp_file_path = os.path.join(upload_dir, filename)

        try:
            response = requests.get(download_url)
            response.raise_for_status()

            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(response.content)

            with open(temp_file_path, "rb") as saved_file:
                file_bytes = saved_file.read()

            extracted_docs = fileProcessor(file_bytes, filename)
            if not extracted_docs:
                logger.warning("No extractable content found for attachment %s", filename)
                continue

            doc_tag = str(
                getattr(file_download, "unique_id", None)
                or getattr(file_download, "id", None)
                or filename
            )

            for doc_name, doc_content in extracted_docs.items():
                train_on_document(
                    doc_content,
                    subscription_id=subscription_id,
                    profile_tag=Config.Teams.DEFAULT_PROFILE,
                    doc_tag=doc_tag,
                    doc_name=doc_name,
                )

            _upload_to_blob(temp_file_path, filename, subscription_id)
            processed = True
        except Exception as exc:
            logger.error("Failed to process Teams attachment %s: %s", filename, exc)
        finally:
            try:
                os.remove(temp_file_path)
            except FileNotFoundError:
                pass
            except Exception as cleanup_exc:
                logger.warning("Could not remove temp file %s: %s", temp_file_path, cleanup_exc)

    if processed:
        return build_teams_message(
            {"response": "File processed successfully. You can now ask questions about it."}
        )

    return {"type": "message", "text": "Unable to process the attachment. Please try again."}


async def handle_teams_activity(activity: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Entrypoint used by FastAPI route to serve Teams messages."""
    if headers:
        if not verify_shared_secret(headers):
            raise TeamsAuthError("Unauthorized Teams request")
    elif getattr(Config.Teams, "SHARED_SECRET", ""):
        logger.warning("Teams shared secret configured but no headers provided; skipping verification.")

    if activity.get("attachments"):
        return await handle_attachment_activity(activity)

    question = extract_question(activity)
    if not question:
        return {"type": "message", "text": "I did not receive a question. Please type your message again."}

    user_id = extract_user_id(activity)
    session_id = extract_session_id(activity)

    try:
        answer = answer_question(
            query=question,
            user_id=user_id,
            profile_id=Config.Teams.DEFAULT_PROFILE,
            subscription_id=Config.Teams.DEFAULT_SUBSCRIPTION,
            model_name=Config.Teams.DEFAULT_MODEL,
            persona=Config.Teams.DEFAULT_PERSONA,
        )
    except Exception as exc:
        logger.error("Failed to answer Teams question: %s", exc)
        return {
            "type": "message",
            "text": "I hit a snag answering your question. Please try again in a moment.",
        }

    # Log session mapping to help debug multi-user chats.
    logger.info("Teams message handled | user=%s session=%s", user_id, session_id)

    return build_teams_message(answer)
