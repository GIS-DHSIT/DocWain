from __future__ import annotations

import io
from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import APIRouter, File, Header, UploadFile
from pydantic import BaseModel, Field
from PIL import Image, ImageEnhance, ImageOps

from src.api.dw_document_extractor import DocumentExtractor
from src.tools.base import ToolError, generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record
from src.tools.common.http_client import fetch_bytes
from src.tools.common.io_limits import MAX_BINARY_BYTES, decode_base64, enforce_limit, validate_upload

logger = get_logger(__name__)

router = APIRouter(prefix="/image", tags=["Tools-Image"])

ALLOWED_IMAGE_MIME = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

_OCR_RETRY_THRESHOLD = 70.0

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s\-()]{7,15})")
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b",
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(r"(?:USD|EUR|INR|GBP|\$|€|₹)\s?\d[\d,]*(?:\.\d{1,2})?")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - Pillow < 9
    _LANCZOS = Image.LANCZOS

class ImageAnalysisRequest(BaseModel):
    image_url: Optional[str] = Field(default=None, description="Public URL to the image")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded image bytes (data URI supported)")
    query: Optional[str] = Field(default=None, description="Optional question to focus insights")
    ocr_engine: str = Field(default="auto", pattern="^(auto|easyocr|pytesseract)$")
    high_accuracy: bool = Field(default=True, description="Enable multi-pass OCR with image variants")
    max_chars: int = Field(default=6000, ge=300, le=20000)
    min_confidence: float = Field(default=55.0, ge=0.0, le=100.0)

@dataclass
class OCRCandidate:
    engine: str
    variant: str
    text: str
    confidence: Optional[float]

def _strip_data_url_prefix(value: str) -> str:
    raw = (value or "").strip()
    if raw.startswith("data:") and "," in raw:
        return raw.split(",", 1)[1]
    return raw

def _dedupe(values: Sequence[str], *, limit: int) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        item = (value or "").strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out

def _normalize_text(text: str, *, max_chars: int) -> str:
    cleaned = (text or "").replace("\x00", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned

def _extract_key_values(text: str, *, limit: int = 20) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate or len(candidate) > 220:
            continue
        match = re.match(r"^([A-Za-z0-9][^:]{1,64}):\s*(.+)$", candidate)
        if not match:
            match = re.match(r"^([A-Za-z0-9][^-]{1,64})\s+-\s+(.+)$", candidate)
        if not match:
            continue
        key = match.group(1).strip()
        value = match.group(2).strip()
        if key and value:
            pairs.append({"key": key, "value": value})
        if len(pairs) >= limit:
            break
    return pairs

def _infer_document_type(text: str) -> str:
    lowered = (text or "").lower()
    if not lowered:
        return "unknown"
    if any(token in lowered for token in ("invoice", "bill to", "amount due", "total due", "purchase order")):
        return "invoice"
    if any(token in lowered for token in ("resume", "curriculum vitae", "skills", "experience", "education")):
        return "resume"
    if any(token in lowered for token in ("agreement", "contract", "clause", "liability", "terms and conditions")):
        return "legal"
    if any(token in lowered for token in ("patient", "diagnosis", "prescription", "medication", "clinical")):
        return "medical"
    return "generic"

def _query_hits(text: str, query: Optional[str], *, limit: int = 5) -> List[str]:
    if not query:
        return []
    keywords = {
        token.lower()
        for token in _TOKEN_RE.findall(query)
        if len(token) > 2
    }
    if not keywords:
        return []
    scored: List[Tuple[int, str]] = []
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        line_tokens = {token.lower() for token in _TOKEN_RE.findall(candidate)}
        overlap = len(keywords & line_tokens)
        if overlap > 0:
            scored.append((overlap, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    return _dedupe([item[1] for item in scored], limit=limit)

def _build_insights(text: str, query: Optional[str]) -> Dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    summary = " ".join(lines[:2])[:360] if lines else ""
    words = _TOKEN_RE.findall(text)
    return {
        "summary": summary,
        "document_type": _infer_document_type(text),
        "word_count": len(words),
        "line_count": len(lines),
        "entities": {
            "emails": _dedupe(_EMAIL_RE.findall(text), limit=15),
            "phones": _dedupe(_PHONE_RE.findall(text), limit=15),
            "urls": _dedupe(_URL_RE.findall(text), limit=15),
            "dates": _dedupe(_DATE_RE.findall(text), limit=15),
            "amounts": _dedupe(_AMOUNT_RE.findall(text), limit=15),
        },
        "key_values": _extract_key_values(text, limit=20),
        "query_hits": _query_hits(text, query, limit=5),
    }

@lru_cache(maxsize=2)
def _get_ocr_extractor(engine: str) -> DocumentExtractor:
    return DocumentExtractor(ocr_engine=engine)

def _ocr_with_engine(image: Image.Image, engine: str) -> Tuple[str, Optional[float]]:
    extractor = _get_ocr_extractor(engine if engine in {"easyocr", "pytesseract"} else "pytesseract")
    return extractor._ocr_image(image, engine=engine)

def _build_image_variants(image: Image.Image, *, high_accuracy: bool) -> List[Tuple[str, Image.Image]]:
    variants: List[Tuple[str, Image.Image]] = [("original", image.convert("RGB"))]
    if not high_accuracy:
        return variants

    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    variants.append(("gray_autocontrast", gray.convert("RGB")))

    sharpened = ImageEnhance.Sharpness(gray).enhance(2.0)
    variants.append(("sharpened", sharpened.convert("RGB")))

    width, height = image.size
    min_side = min(width, height)
    if min_side < 1400:
        scale = 2.0 if min_side < 900 else 1.5
        upscaled = image.resize((int(width * scale), int(height * scale)), _LANCZOS).convert("RGB")
        variants.append(("upscaled", upscaled))

    return variants

def _candidate_score(candidate: OCRCandidate) -> Tuple[int, float, int, int]:
    text = (candidate.text or "").strip()
    words = len(text.split())
    chars = len(text)
    conf = float(candidate.confidence) if candidate.confidence is not None else -1.0
    return (1 if text else 0, conf, words, chars)

def _extract_text_high_accuracy(
    image: Image.Image,
    *,
    ocr_engine: str,
    high_accuracy: bool,
) -> Tuple[OCRCandidate, List[OCRCandidate]]:
    preferred = "easyocr" if ocr_engine == "auto" else ocr_engine
    alternate = "pytesseract" if preferred == "easyocr" else "easyocr"

    candidates: List[OCRCandidate] = []
    variants = _build_image_variants(image, high_accuracy=high_accuracy)
    for variant_name, variant in variants:
        text, conf = _ocr_with_engine(variant, preferred)
        candidates.append(OCRCandidate(engine=preferred, variant=variant_name, text=text, confidence=conf))
        if high_accuracy and (not text or conf is None or conf < _OCR_RETRY_THRESHOLD):
            retry_text, retry_conf = _ocr_with_engine(variant, alternate)
            candidates.append(OCRCandidate(engine=alternate, variant=variant_name, text=retry_text, confidence=retry_conf))

    if not candidates:
        raise ToolError("OCR could not process the image", code="ocr_failed", status_code=500)

    best = max(candidates, key=_candidate_score)
    return best, candidates

def _load_image(raw: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ToolError("Invalid image payload", code="invalid_image") from exc

async def _resolve_image_bytes(*, upload: UploadFile | None, req: ImageAnalysisRequest) -> bytes:
    if upload:
        return validate_upload(upload, allowed_mime=ALLOWED_IMAGE_MIME, max_bytes=MAX_BINARY_BYTES, label="image_file")
    if req.image_base64:
        raw_b64 = _strip_data_url_prefix(req.image_base64)
        return decode_base64(raw_b64, max_bytes=MAX_BINARY_BYTES, label="image_base64")
    if req.image_url:
        raw = await fetch_bytes(req.image_url, max_bytes=MAX_BINARY_BYTES)
        enforce_limit(len(raw), MAX_BINARY_BYTES, "image_url")
        return raw
    raise ToolError("image_file, image_url, or image_base64 is required", code="missing_image")

def _coerce_request(payload: Dict[str, Any]) -> ImageAnalysisRequest:
    input_payload = payload.get("input") if isinstance(payload, dict) else payload
    if not isinstance(input_payload, dict):
        input_payload = {}
    context_payload = payload.get("context") if isinstance(payload, dict) else {}
    if not isinstance(context_payload, dict):
        context_payload = {}

    merged = dict(input_payload)
    for key in ("image_url", "image_base64", "query", "ocr_engine", "high_accuracy", "max_chars", "min_confidence"):
        if key not in merged and key in context_payload:
            merged[key] = context_payload[key]
    return ImageAnalysisRequest(**merged)

def _analyze_image_bytes(raw: bytes, req: ImageAnalysisRequest) -> Tuple[Dict[str, Any], List[str]]:
    image = _load_image(raw)
    best, candidates = _extract_text_high_accuracy(
        image,
        ocr_engine=req.ocr_engine,
        high_accuracy=req.high_accuracy,
    )
    extracted_text = _normalize_text(best.text or "", max_chars=req.max_chars)
    confidence = round(float(best.confidence), 2) if best.confidence is not None else None

    warnings: List[str] = []
    if not extracted_text:
        warnings.append("No OCR text detected in the image.")
    if confidence is not None and confidence < req.min_confidence:
        warnings.append(
            f"Low OCR confidence ({confidence}%). Consider a higher-resolution or cleaner image."
        )
    if best.engine != "easyocr" and req.ocr_engine in {"auto", "easyocr"}:
        warnings.append("EasyOCR output was unavailable or weaker than fallback OCR for this image.")

    ranked = sorted(candidates, key=_candidate_score, reverse=True)[:4]
    candidate_meta = [
        {
            "engine": item.engine,
            "variant": item.variant,
            "confidence": round(float(item.confidence), 2) if item.confidence is not None else None,
            "chars": len((item.text or "").strip()),
        }
        for item in ranked
    ]

    result = {
        "text": extracted_text,
        "ocr": {
            "engine": best.engine,
            "variant": best.variant,
            "confidence": confidence,
            "high_accuracy": req.high_accuracy,
            "candidates": candidate_meta,
        },
        "insights": _build_insights(extracted_text, req.query),
    }
    return result, warnings

def _build_source(req: ImageAnalysisRequest, correlation_id: Optional[str], byte_count: int) -> Dict[str, Any]:
    if req.image_url:
        return build_source_record(
            "url",
            req.image_url,
            title=req.image_url,
            metadata={"bytes": byte_count, "modality": "image"},
        )
    return build_source_record(
        "tool",
        correlation_id or "image_analysis",
        title="image_upload",
        metadata={"bytes": byte_count, "modality": "image"},
    )

@register_tool("image_analysis")
async def image_analysis_handler(payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    req = _coerce_request(payload)
    raw = await _resolve_image_bytes(upload=None, req=req)
    result, warnings = _analyze_image_bytes(raw, req)
    source = _build_source(req, correlation_id, len(raw))
    return {
        "result": result,
        "sources": [source],
        "warnings": warnings,
        "grounded": bool(result.get("text")),
        "context_found": bool(result.get("text")),
    }

@router.post("/analyze")
async def analyze(
    request: ImageAnalysisRequest,
    image_file: UploadFile | None = File(None),
    x_correlation_id: str | None = Header(None),
):
    cid = generate_correlation_id(x_correlation_id)
    try:
        raw = await _resolve_image_bytes(upload=image_file, req=request)
        result, warnings = _analyze_image_bytes(raw, request)
        source = _build_source(request, cid, len(raw))
        return standard_response(
            "image_analysis",
            grounded=bool(result.get("text")),
            context_found=bool(result.get("text")),
            result=result,
            sources=[source],
            warnings=warnings,
            correlation_id=cid,
        )
    except ToolError as exc:
        return standard_response(
            "image_analysis",
            status="error",
            grounded=False,
            context_found=False,
            result={},
            sources=[],
            warnings=[],
            error=exc.as_dict(),
            correlation_id=cid,
        )
