import pytest
from unittest.mock import patch, MagicMock


def test_model_pool_singleton():
    from src.visual_intelligence.model_pool import get_model_pool
    pool1 = get_model_pool()
    pool2 = get_model_pool()
    assert pool1 is pool2


def test_model_pool_tracks_disabled():
    from src.visual_intelligence.model_pool import ModelPool
    pool = ModelPool()
    assert pool.is_available("dit") is True
    pool.disabled_models.add("dit")
    assert pool.is_available("dit") is False


def test_model_pool_load_returns_none_when_unavailable():
    from src.visual_intelligence.model_pool import ModelPool
    pool = ModelPool()
    with patch.object(pool, "_try_load", return_value=None):
        with patch.object(pool, "_try_install", return_value=None):
            result = pool.load_model("dit")
            assert result is None
            assert pool.is_available("dit") is False


def test_model_pool_load_caches_result():
    from src.visual_intelligence.model_pool import ModelPool
    pool = ModelPool()
    mock_model = MagicMock()
    with patch.object(pool, "_try_load", return_value=mock_model):
        result1 = pool.load_model("dit")
        result2 = pool.load_model("dit")
        assert result1 is mock_model
        assert result2 is mock_model


def test_model_pool_device_selection():
    from src.visual_intelligence.model_pool import ModelPool
    pool = ModelPool()
    with patch("torch.cuda.is_available", return_value=False):
        device = pool.get_device("dit")
        assert str(device) == "cpu"
