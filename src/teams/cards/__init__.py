import copy
import json
from pathlib import Path
from typing import Any, Dict

CARD_DIR = Path(__file__).parent


def load_card_template(name: str) -> Dict[str, Any]:
    path = CARD_DIR / f"{name}.json"
    content = path.read_text()
    return json.loads(content)


def build_card(name: str, **kwargs) -> Dict[str, Any]:
    template = load_card_template(name)

    def _apply(value):
        if isinstance(value, str):
            try:
                return value.format(**kwargs)
            except Exception:
                return value
        if isinstance(value, list):
            return [_apply(v) for v in value]
        if isinstance(value, dict):
            return {k: _apply(v) for k, v in value.items()}
        return value

    # Copy to avoid mutating cached template structures
    return _apply(copy.deepcopy(template))
