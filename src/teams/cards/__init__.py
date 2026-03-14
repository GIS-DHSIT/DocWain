import copy
import json
from src.utils.logging_utils import get_logger
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

logger = get_logger(__name__)

CARD_DIR = Path(__file__).parent

_FALLBACK_CARD = {
    "type": "AdaptiveCard",
    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
    "version": "1.5",
    "body": [{"type": "TextBlock", "text": "Something went wrong. Please try again.", "wrap": True}],
}

def load_card_template(name: str) -> Dict[str, Any]:
    path = CARD_DIR / f"{name}.json"
    try:
        content = path.read_text()
        return json.loads(content)
    except FileNotFoundError:
        logger.error("Card template not found: %s", name)
        return copy.deepcopy(_FALLBACK_CARD)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in card template %s: %s", name, exc)
        return copy.deepcopy(_FALLBACK_CARD)

def build_card(name: str, **kwargs) -> Dict[str, Any]:
    template = load_card_template(name)
    safe_kwargs = defaultdict(lambda: "", kwargs)

    def _apply(value):
        if isinstance(value, str):
            try:
                return value.format_map(safe_kwargs)
            except Exception:
                return value
        if isinstance(value, list):
            return [_apply(v) for v in value]
        if isinstance(value, dict):
            return {k: _apply(v) for k, v in value.items()}
        return value

    return _apply(copy.deepcopy(template))
