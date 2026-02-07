from src.finetune.models import (
    AutoFinetuneRunRequest,
    CollectionOnlyFinetuneRequest,
    FinetuneRequest,
    FinetuneStatus,
    ResolvedModel,
)


def get_finetune_manager():
    """Lazy import to avoid loading heavy deps until needed."""
    from src.finetune.unsloth_trainer import get_finetune_manager as _impl

    return _impl()


def resolve_model_for_profile(profile_id: str, requested_model: str | None) -> ResolvedModel:
    """Return the effective model/ backend to serve for a profile."""
    manager = get_finetune_manager()
    return manager.resolve_model(profile_id, requested_model)


def list_models():
    return get_finetune_manager().list_models()


__all__ = [
    "AutoFinetuneRunRequest",
    "CollectionOnlyFinetuneRequest",
    "FinetuneRequest",
    "FinetuneStatus",
    "ResolvedModel",
    "get_finetune_manager",
    "list_models",
    "resolve_model_for_profile",
]
