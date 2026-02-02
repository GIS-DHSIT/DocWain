from src.chunking.section_chunker import SectionChunker


def test_chunks_respect_section_headers():
    text = (
        "EDUCATION\n"
        "B.Sc. in Computer Science, Example University\n\n"
        "EXPERIENCE\n"
        "Software Engineer at ExampleCorp\n"
    )
    chunker = SectionChunker()
    chunks = chunker.chunk_document(text, doc_internal_id="doc1", source_filename="resume.pdf")
    sections = {chunk.section_title for chunk in chunks}
    assert "EDUCATION" in sections
    assert "EXPERIENCE" in sections


def test_no_incomplete_sentence_split():
    text = (
        "SUMMARY\n"
        "This line continues\n"
        "across lines but ends properly.\n"
    )
    chunker = SectionChunker()
    chunks = chunker.chunk_document(text, doc_internal_id="doc2", source_filename="summary.pdf")
    assert all(chunk.sentence_complete for chunk in chunks)


def test_dehyphenation_fixes_soft_hyphen():
    text = "SKILLS\nUnstruc\u00AD tured.io is referenced."
    chunker = SectionChunker()
    chunks = chunker.chunk_document(text, doc_internal_id="doc3", source_filename="skills.pdf")
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "Unstructured.io" in combined
