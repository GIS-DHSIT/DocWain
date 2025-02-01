import pytest
from docwain_app.core.processor import create_llm
from docwain_app.config import Config


def test_create_local_llm():
    """Test creating local LLM"""
    Config.Model.USE_LOCAL = True
    llm = create_llm()
    assert llm is not None
    assert hasattr(llm, 'invoke')
    assert hasattr(llm, 'stream')


def test_create_remote_llm():
    """Test creating remote LLM"""
    Config.Model.USE_LOCAL = False
    llm = create_llm()
    assert llm is not None
    assert hasattr(llm, 'invoke')
    assert hasattr(llm, 'stream')


@pytest.mark.asyncio
async def test_llm_invoke(llm):
    """Test LLM invocation"""
    prompt = "What is 2+2?"
    response = await llm.ainvoke([{"role": "user", "content": prompt}])
    assert response is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_llm_streaming(llm):
    """Test LLM streaming"""
    prompt = "What is 2+2?"
    chunks = []
    async for chunk in llm.astream([{"role": "user", "content": prompt}]):
        assert chunk is not None
        assert isinstance(chunk.content, str)
        chunks.append(chunk.content)

    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert len(full_response) > 0


def test_llm_temperature_setting():
    """Test LLM temperature configuration"""
    original_temp = Config.Model.TEMPERATURE
    try:
        # Test with different temperatures
        temps = [0.0, 0.5, 1.0]
        for temp in temps:
            Config.Model.TEMPERATURE = temp
            llm = create_llm()
            assert llm is not None
            # Check if temperature was set correctly
            assert hasattr(llm, 'temperature')
            assert llm.temperature == temp
    finally:
        # Restore original temperature
        Config.Model.TEMPERATURE = original_temp


def test_llm_max_tokens_setting():
    """Test LLM max tokens configuration"""
    original_max_tokens = Config.Model.MAX_TOKENS
    try:
        # Test with different max token values
        token_values = [1000, 2000, 4000]
        for tokens in token_values:
            Config.Model.MAX_TOKENS = tokens
            llm = create_llm()
            assert llm is not None
            # Check if max tokens was set correctly
            assert hasattr(llm, 'max_tokens')
            assert llm.max_tokens == tokens
    finally:
        # Restore original max tokens
        Config.Model.MAX_TOKENS = original_max_tokens