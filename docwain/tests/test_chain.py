import pytest
from docwain_app.core.chain import create_chain


def test_chain_creation(llm, retriever):
    """Test creating a chain with valid inputs"""
    chain = create_chain(llm, retriever)
    assert chain is not None


def test_chain_invalid_llm(retriever):
    """Test chain creation with invalid LLM"""
    with pytest.raises(ValueError):
        create_chain(None, retriever)


def test_chain_invalid_retriever(llm):
    """Test chain creation with invalid retriever"""
    with pytest.raises(ValueError):
        create_chain(llm, None)


@pytest.mark.asyncio
async def test_chain_query(llm, retriever):
    """Test chain query functionality"""
    chain = create_chain(llm, retriever)
    question = "What is this document about?"
    response = await chain.ainvoke({"question": question})
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_chain_streaming(llm, retriever):
    """Test chain streaming functionality"""
    chain = create_chain(llm, retriever)
    question = "What is this document about?"

    chunks = []
    async for chunk in chain.astream({"question": question}):
        assert isinstance(chunk, str)
        chunks.append(chunk)

    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert len(full_response) > 0