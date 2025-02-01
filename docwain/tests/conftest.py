import pytest
from pathlib import Path
import tempfile
import shutil
from docwain_app.config import Config
from docwain_app.core.processor import create_llm
from docwain_app.core.retriever import create_retriever
from docwain_app.core.ingestor import Ingestor

@pytest.fixture(scope="session")
def test_dir():
    """Create a temporary directory for test files"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def sample_pdf(test_dir):
    """Create a sample PDF file for testing"""
    pdf_path = test_dir / "test.pdf"
    # TODO: Create sample PDF
    yield pdf_path

@pytest.fixture(scope="session")
def sample_docx(test_dir):
    """Create a sample DOCX file for testing"""
    docx_path = test_dir / "test.docx"
    # TODO: Create sample DOCX
    yield docx_path

@pytest.fixture(scope="session")
def ingestor():
    """Create an Ingestor instance"""
    return Ingestor()

@pytest.fixture(scope="session")
def llm():
    """Create an LLM instance"""
    return create_llm()

@pytest.fixture(scope="session")
def vector_store(ingestor, sample_pdf, sample_docx):
    """Create a test vector store"""
    # TODO: Create vector store from sample documents
    return None

@pytest.fixture(scope="session")
def retriever(llm, vector_store):
    """Create a test retriever"""
    return create_retriever(llm, vector_store)