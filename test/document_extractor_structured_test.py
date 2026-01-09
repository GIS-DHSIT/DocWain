import pandas as pd

from src.api.dw_document_extractor import DocumentExtractor
from src.api.pipeline_models import ExtractedDocument


def test_extract_dataframe_builds_structured_document():
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Role": ["Engineer", "Analyst"],
            "Score": [9.5, 8.7],
        }
    )

    extractor = DocumentExtractor()
    doc = extractor.extract_dataframe(df, sheet_name="Team")

    assert isinstance(doc, ExtractedDocument)
    # Header + 2 rows
    assert len(doc.chunk_candidates) >= 3
    assert doc.chunk_candidates[0].chunk_type == "table_header"
    # Ensure row text captures values
    texts = " ".join(c.text for c in doc.chunk_candidates)
    assert "Alice" in texts and "Bob" in texts
    # Table CSV preserved
    assert doc.tables and "Name,Role,Score" in doc.tables[0].csv
