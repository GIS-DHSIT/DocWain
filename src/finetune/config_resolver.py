import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from src.api.config import Config


class ConfigCoercionError(ValueError):
    pass


@dataclass
class ResolvedField:
    name: str
    value: Any
    source: str
    expected_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "type": type(self.value).__name__,
            "source": self.source,
            "expected_type": self.expected_type,
        }


_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+(\.\d+)?$")


def as_int(value: Any, field: str, source: str) -> int:
    if value is None:
        raise ConfigCoercionError(f"{field} is required but missing (source={source})")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ConfigCoercionError(
            f"{field}={value!r} not an integer (expected int, source={source})"
        )
    if isinstance(value, str):
        stripped = value.strip()
        if _INT_RE.match(stripped):
            return int(stripped)
        if _FLOAT_RE.match(stripped):
            fval = float(stripped)
            if fval.is_integer():
                return int(fval)
        raise ConfigCoercionError(
            f"{field}={value!r} not coercible to int (source={source})"
        )
    try:
        return int(value)
    except Exception as exc:
        raise ConfigCoercionError(
            f"{field}={value!r} not coercible to int (source={source})"
        ) from exc


def as_float(value: Any, field: str, source: str) -> float:
    if value is None:
        raise ConfigCoercionError(f"{field} is required but missing (source={source})")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if _FLOAT_RE.match(stripped):
            return float(stripped)
        raise ConfigCoercionError(
            f"{field}={value!r} not coercible to float (source={source})"
        )
    try:
        return float(value)
    except Exception as exc:
        raise ConfigCoercionError(
            f"{field}={value!r} not coercible to float (source={source})"
        ) from exc


def _pick_value(
    field: str,
    request_value: Any,
    default_value: Any,
    env_keys: Tuple[str, ...],
) -> Tuple[Any, str]:
    if request_value is not None:
        return request_value, "request"
    for key in env_keys:
        if key in os.environ:
            return os.environ[key], f"env:{key}"
    return default_value, "default"


def resolve_finetune_numeric_config(request) -> Dict[str, ResolvedField]:
    defaults = Config.Finetune
    fields = {
        "batch_size": ("BATCH_SIZE", int, getattr(request, "batch_size", None), getattr(request, "batch_size", None)),
        "micro_batch_size": ("MICRO_BATCH_SIZE", int, getattr(request, "micro_batch_size", None), None),
        "gradient_accumulation_steps": (
            "GRADIENT_ACCUMULATION_STEPS",
            int,
            getattr(request, "gradient_accumulation", None),
            getattr(request, "gradient_accumulation", None),
        ),
        "max_steps": ("MAX_STEPS", int, getattr(request, "max_steps", None), getattr(request, "max_steps", None)),
        "num_train_epochs": ("NUM_TRAIN_EPOCHS", int, getattr(request, "num_epochs", None), getattr(request, "num_epochs", None)),
        "warmup_steps": ("WARMUP_STEPS", int, getattr(request, "warmup_steps", None), None),
        "lora_r": ("LORA_R", int, getattr(request, "lora_r", None), getattr(request, "lora_r", None)),
        "lora_alpha": ("LORA_ALPHA", int, getattr(request, "lora_alpha", None), getattr(request, "lora_alpha", None)),
        "max_seq_length": ("MAX_SEQ_LENGTH", int, getattr(request, "max_seq_length", None), None),
        "cutoff_len": ("CUTOFF_LEN", int, getattr(request, "cutoff_len", None), None),
        "eval_steps": ("EVAL_STEPS", int, getattr(request, "eval_steps", None), None),
        "save_steps": ("SAVE_STEPS", int, getattr(request, "save_steps", None), None),
        "merge_window": ("MERGE_WINDOW", int, None, getattr(defaults, "MERGE_WINDOW", None)),
        "min_pairs_per_profile": ("MIN_PAIRS_PER_PROFILE", int, None, getattr(defaults, "MIN_PAIRS_PER_PROFILE", None)),
        "max_pairs_per_profile": ("MAX_PAIRS_PER_PROFILE", int, None, getattr(defaults, "MAX_PAIRS_PER_PROFILE", None)),
        "min_chunk_chars": ("MIN_CHUNK_CHARS", int, None, getattr(defaults, "MIN_CHUNK_CHARS", None)),
        "min_merged_tokens": ("MIN_MERGED_TOKENS", int, None, getattr(defaults, "MIN_MERGED_TOKENS", None)),
        "temperature": ("GENERATION_TEMPERATURE", float, None, getattr(defaults, "GENERATION_TEMPERATURE", None)),
        "top_p": ("TOP_P", float, getattr(request, "top_p", None), None),
        "learning_rate": ("LEARNING_RATE", float, getattr(request, "learning_rate", None), getattr(request, "learning_rate", None)),
        "weight_decay": ("WEIGHT_DECAY", float, getattr(request, "weight_decay", None), None),
    }

    resolved: Dict[str, ResolvedField] = {}
    for name, (env_key, caster, req_val, default_val) in fields.items():
        env_keys = (f"FINETUNE_{env_key}", env_key)
        value, source = _pick_value(name, req_val, default_val, env_keys)
        if value is None:
            continue
        if caster is int:
            coerced = as_int(value, name, source)
            resolved[name] = ResolvedField(name=name, value=coerced, source=source, expected_type="int")
        else:
            coerced = as_float(value, name, source)
            resolved[name] = ResolvedField(name=name, value=coerced, source=source, expected_type="float")
    return resolved


def apply_numeric_config_to_request(request) -> Tuple[Any, Dict[str, ResolvedField]]:
    resolved = resolve_finetune_numeric_config(request)
    updates = {}
    mapping = {
        "batch_size": "batch_size",
        "gradient_accumulation_steps": "gradient_accumulation",
        "max_steps": "max_steps",
        "num_train_epochs": "num_epochs",
        "learning_rate": "learning_rate",
        "lora_r": "lora_r",
        "lora_alpha": "lora_alpha",
    }
    for resolved_key, req_key in mapping.items():
        if resolved_key in resolved:
            updates[req_key] = resolved[resolved_key].value
    updated = request.model_copy(update=updates)
    return updated, resolved
