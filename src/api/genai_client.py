from src.utils.logging_utils import get_logger
from typing import Any, Optional, Tuple

try:
    from google import genai as _genai  # type: ignore
    _GENAI_IMPORT = "google.genai"
except ImportError:  # pragma: no cover - fallback for older SDK
    import google.generativeai as _genai  # type: ignore
    _GENAI_IMPORT = "google.generativeai"

logger = get_logger(__name__)

_client: Any | None = None
_client_api_key: str | None = None

def _coerce_generation_config(config: Optional[dict]) -> Optional[Any]:
    """Convert plain dicts into the SDK's config type when available."""
    if not config:
        return None

    try:
        types_module = getattr(_genai, "types", None)
        if types_module and isinstance(config, dict):
            for cls_name in ("GenerateContentConfig", "GenerationConfig"):
                generation_config_cls = getattr(types_module, cls_name, None)
                if generation_config_cls:
                    try:
                        return generation_config_cls(**config)
                    except Exception:
                        continue
    except Exception:
        # Gracefully fall back to the raw config
        return config
    return config

def get_genai_client(api_key: str) -> Any:
    """
    Return a configured GenAI client, compatible with both google.genai and
    the deprecated google.generativeai SDKs. The client is cached per API key.
    """
    global _client, _client_api_key
    if _client is not None and _client_api_key == api_key:
        return _client

    if hasattr(_genai, "Client"):
        _client = _genai.Client(api_key=api_key)
    else:
        _genai.configure(api_key=api_key)
        _client = _genai
    _client_api_key = api_key
    logger.info("Initialized %s client", _GENAI_IMPORT)
    return _client

def _extract_response_text(response: Any) -> str:
    """Best-effort extraction of text from GenAI responses across SDK versions."""
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()

    candidates = getattr(response, "candidates", None)
    if candidates:
        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        if content:
            parts = getattr(content, "parts", None)
            if parts:
                part_text = getattr(parts[0], "text", None)
                if part_text:
                    return str(part_text).strip()
            content_text = getattr(content, "text", None)
            if content_text:
                return str(content_text).strip()
    return ""

def generate_text(
    api_key: str,
    model: str,
    prompt: str,
    generation_config: Optional[dict] = None,
    system_instruction: Optional[str] = None,
) -> Tuple[str, Any]:
    """
    Generate text with Gemini, handling both the new google.genai and legacy
    google.generativeai packages. Returns (text, raw_response).
    """
    client = get_genai_client(api_key)

    # Inject system_instruction into generation config for the new SDK
    config_dict = dict(generation_config or {})
    if system_instruction:
        config_dict["system_instruction"] = system_instruction

    gen_config = _coerce_generation_config(config_dict) if config_dict else None

    if hasattr(client, "models"):  # google.genai
        if gen_config is None:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
        else:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=gen_config,
                )
            except TypeError as exc:
                if "unexpected keyword argument" in str(exc) and "config" in str(exc):
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        generation_config=gen_config,
                    )
                else:
                    raise
    else:  # google.generativeai
        kwargs = {}
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        model_client = client.GenerativeModel(model, **kwargs)
        response = model_client.generate_content(prompt, generation_config=gen_config)

    return _extract_response_text(response), response
