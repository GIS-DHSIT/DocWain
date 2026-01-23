import pytest

from src.finetune.config_resolver import as_int, as_float, apply_numeric_config_to_request, ConfigCoercionError
from src.finetune.models import FinetuneRequest


def test_config_coercion_int_fields():
    req = FinetuneRequest(
        profile_id="p1",
        batch_size="4",
        gradient_accumulation="2",
        max_steps="100",
        num_epochs="3",
        lora_r="8",
        lora_alpha="16",
        learning_rate=0.0002,
    )
    updated, resolved = apply_numeric_config_to_request(req)
    assert updated.batch_size == 4
    assert updated.gradient_accumulation == 2
    assert updated.max_steps == 100
    assert updated.num_epochs == 3
    assert updated.lora_r == 8
    assert updated.lora_alpha == 16


def test_config_coercion_float_fields():
    req = FinetuneRequest(
        profile_id="p1",
        batch_size=4,
        gradient_accumulation=2,
        learning_rate="0.1",
        lora_r=8,
        lora_alpha=16,
    )
    updated, resolved = apply_numeric_config_to_request(req)
    assert abs(updated.learning_rate - 0.1) < 1e-6


def test_invalid_numeric_field_raises():
    req = FinetuneRequest.model_construct(
        profile_id="p1",
        batch_size="two",
        gradient_accumulation=2,
        learning_rate=0.1,
        lora_r=8,
        lora_alpha=16,
    )
    with pytest.raises(ConfigCoercionError) as exc:
        apply_numeric_config_to_request(req)
    assert "batch_size" in str(exc.value)


def test_regression_string_multiplier_handled():
    req = FinetuneRequest(
        profile_id="p1",
        batch_size="4",
        gradient_accumulation="2",
        learning_rate=0.1,
        lora_r=8,
        lora_alpha=16,
    )
    updated, _ = apply_numeric_config_to_request(req)
    # Should not throw TypeError; multiplication should be valid after coercion.
    value = int(updated.gradient_accumulation) * int(updated.batch_size)
    assert value == 8
