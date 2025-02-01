import pytest
from pathlib import Path
from docwain_app.core.ingestor import Ingestor


@pytest.mark.asyncio
async def test_ingest_pdf(ingestor, sample_pdf):
    """Test PDF ingestion"""
    files = [sample_pdf]
    vector_store = await ingestor.ingest(files)
    assert vector_store is not None


@pytest.mark.asyncio
async def test_ingest_docx(ingestor, sample_docx):
    """Test DOCX ingestion"""
    files = [sample_docx]
    vector_store = await ingestor.ingest(files)
    assert vector_store is not None


@pytest.mark.asyncio
async def test_ingest_multiple_files(ingestor, sample_pdf, sample_docx):
    """Test ingesting multiple files"""
    files = [sample_pdf, sample_docx]
    vector_store = await ingestor.ingest(files)
    assert vector_store is not None


def test_load_document(ingestor, sample_pdf):
    """Test document loading"""
    docs = ingestor._load_document(sample_pdf)
    assert len(docs) > 0
    assert all(hasattr(doc, 'page_content') for doc in docs)


def test_load_unsupported_file(ingestor, test_dir):
    """Test loading unsupported file type"""
    unsupported_file = test_dir / "test.xyz"
    unsupported_file.touch()

    with pytest.raises(ValueError, match="Unsupported file type"):
        ingestor._load_document(unsupported_file)


@pytest.mark.asyncio
async def test_progress_callback(ingestor, sample_pdf):
    """Test progress callback functionality"""
    progress_updates = []

    def callback(current, total, message):
        progress_updates.append((current, total, message))

    test_ingestor = Ingestor(progress_callback=callback)
    files = [sample_pdf]
    await test_ingestor.ingest(files)

    assert len(progress_updates) > 0
    for update in progress_updates:
        assert len(update) == 3
        assert isinstance(update[0], int)
        assert isinstance(update[1], int)
        assert isinstance(update[2], str)