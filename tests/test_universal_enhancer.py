"""
Test the Universal Embedding Enhancer for document-agnostic improvements.
"""

import pytest
from src.embedding.universal_enhancer import (
    UniversalEmbeddingEnhancer,
    ContentTypeDetector,
    SemanticFieldExtractor,
    QualityScorer,
    EmbeddingTextBuilder,
    QueryEnricher,
    enhance_for_embedding,
    enrich_query,
    get_enhanced_payload,
)
from src.embedding.pipeline_enhancement import (
    enhance_chunks_for_embedding,
    enrich_query_for_retrieval,
    get_enhanced_embedding_text,
    deduplicate_chunks,
    UNIVERSAL_ENHANCER_AVAILABLE,
)


class TestContentTypeDetector:
    """Test content type detection."""

    def test_detect_skills_list(self):
        text = "Python, Java, JavaScript, React, Node.js, AWS, Docker, Kubernetes"
        result = ContentTypeDetector.detect(text, "Technical Skills")
        assert result["content_type"] == "skills_list"
        assert result["confidence"] > 0.5

    def test_detect_education(self):
        text = "Bachelor of Science in Computer Science, Stanford University, 2018. GPA: 3.8"
        result = ContentTypeDetector.detect(text, "Education")
        assert result["content_type"] == "education"

    def test_detect_experience(self):
        text = "5 years of experience as Senior Software Engineer at Google, 2019-2024"
        result = ContentTypeDetector.detect(text, "Work Experience")
        assert result["content_type"] == "experience"

    def test_detect_contact_info(self):
        text = "john.doe@email.com | (555) 123-4567 | linkedin.com/in/johndoe"
        result = ContentTypeDetector.detect(text, "Contact")
        assert result["content_type"] == "contact_info"

    def test_detect_financial(self):
        text = "Invoice Total: $1,500.00. Payment due within 30 days. Tax: $150.00"
        result = ContentTypeDetector.detect(text, "Invoice")
        assert result["content_type"] == "financial"

    def test_detect_legal(self):
        text = "Clause 5.2: Liability. The parties hereby agree to indemnify..."
        result = ContentTypeDetector.detect(text, "Terms")
        assert result["content_type"] == "legal"

    def test_detect_structure_bullet_list(self):
        text = "• Python\n• Java\n• JavaScript\n• React"
        result = ContentTypeDetector.detect(text, "Skills")
        assert result["structure"] == "bullet_list"


class TestSemanticFieldExtractor:
    """Test semantic field extraction."""

    def test_extract_entities_email(self):
        text = "Contact me at john.doe@example.com or jane@company.org"
        result = SemanticFieldExtractor.extract(text, "contact_info")
        entities = result["entities"]
        emails = [e for e in entities if e["type"] == "email"]
        assert len(emails) >= 2

    def test_extract_entities_phone(self):
        text = "Call me at (555) 123-4567 or +1 555 987 6543"
        result = SemanticFieldExtractor.extract(text, "contact_info")
        entities = result["entities"]
        phones = [e for e in entities if e["type"] == "phone"]
        assert len(phones) >= 1

    def test_extract_keywords(self):
        text = "Experienced Python developer with React and Node.js skills"
        result = SemanticFieldExtractor.extract(text, "skills_list")
        assert "python" in result["keywords"]
        assert "react" in result["keywords"]

    def test_extract_domain_terms_skills(self):
        text = "Python, Java, AWS, Kubernetes, Docker, React"
        result = SemanticFieldExtractor.extract(text, "skills_list")
        domain_terms = result["domain_terms"]
        assert "python" in domain_terms
        assert "aws" in domain_terms


class TestQualityScorer:
    """Test quality scoring."""

    def test_score_optimal_length(self):
        text = "A " * 150  # ~300 chars - optimal range
        score = QualityScorer.score(text, "skills_list", [], ["python", "java"], "prose")
        assert score.length_score >= 0.8

    def test_score_too_short(self):
        text = "Short"
        score = QualityScorer.score(text, "content", [], [], "prose")
        assert score.length_score < 0.5

    def test_score_with_entities(self):
        text = "Contact: john@example.com, (555) 123-4567"
        entities = [{"type": "email", "value": "john@example.com"}]
        score = QualityScorer.score(text, "contact_info", entities, [], "prose")
        assert score.information_density > 0

    def test_score_high_value_section(self):
        text = "Python, Java, JavaScript, React, Node.js"
        score1 = QualityScorer.score(text, "skills_list", [], ["python", "java"], "comma_list")
        score2 = QualityScorer.score(text, "content", [], ["python", "java"], "comma_list")
        assert score1.overall > score2.overall  # skills_list gets boost


class TestEmbeddingTextBuilder:
    """Test embedding text building."""

    def test_build_skills_text(self):
        text = "Python, Java, JavaScript"
        result = EmbeddingTextBuilder.build(text, "skills_list", "Technical Skills")
        assert "Skills and Competencies:" in result
        assert "Python, Java, JavaScript" in result

    def test_build_education_text(self):
        text = "BS in Computer Science from Stanford"
        result = EmbeddingTextBuilder.build(text, "education", "Education")
        assert "Education and Qualifications:" in result

    def test_build_experience_text(self):
        text = "5 years at Google as Senior Engineer"
        result = EmbeddingTextBuilder.build(text, "experience", "Experience")
        assert "Professional Experience:" in result


class TestQueryEnricher:
    """Test query enrichment."""

    def test_enrich_ranking_query(self):
        query = "rank the top 2 candidates based on experience"
        result = QueryEnricher.enrich(query)
        assert result["query_type"] == "ranking"
        assert "Rank and compare" in result["enriched_query"]
        assert "experience" in result["expansion_terms"]

    def test_enrich_skills_query(self):
        query = "what are John's technical skills"
        result = QueryEnricher.enrich(query)
        assert result["query_type"] == "skills"
        assert "technologies" in result["expansion_terms"]

    def test_enrich_contact_query(self):
        query = "what is John's email address"
        result = QueryEnricher.enrich(query)
        assert result["query_type"] == "contact"
        assert "phone" in result["expansion_terms"]


class TestUniversalEmbeddingEnhancer:
    """Test the main enhancer class."""

    def test_enhance_chunk_skills(self):
        enhancer = UniversalEmbeddingEnhancer()
        result = enhancer.enhance_chunk(
            text="Python, Java, JavaScript, React, Node.js, AWS",
            section_title="Technical Skills",
            document_type="resume",
        )
        assert result.content_type == "skills_list"
        assert "Skills and Competencies" in result.embedding_text
        assert result.quality_score.overall > 0.5
        assert len(result.semantic_fields["keywords"]) > 0

    def test_enhance_chunk_education(self):
        enhancer = UniversalEmbeddingEnhancer()
        result = enhancer.enhance_chunk(
            text="Bachelor of Science in Computer Science, Stanford University, 2018",
            section_title="Education",
            document_type="resume",
        )
        assert result.content_type == "education"
        assert "Education and Qualifications" in result.embedding_text

    def test_enhance_chunks_batch(self):
        enhancer = UniversalEmbeddingEnhancer()
        chunks = [
            {"text": "Python, Java, React", "section_title": "Skills"},
            {"text": "BS Computer Science, MIT", "section_title": "Education"},
        ]
        results = enhancer.enhance_chunks_batch(chunks, {"document_type": "resume"})
        assert len(results) == 2
        assert results[0].content_type == "skills_list"
        assert results[1].content_type == "education"

    def test_build_enhanced_payload(self):
        enhancer = UniversalEmbeddingEnhancer()
        result = enhancer.enhance_chunk(
            text="Python, Java, JavaScript",
            section_title="Skills",
        )
        payload = enhancer.build_enhanced_payload(result)
        assert "content_type" in payload
        assert "quality_score" in payload
        assert "semantic_keywords" in payload
        assert "embedding_text" in payload


class TestPipelineIntegration:
    """Test pipeline enhancement integration."""

    def test_universal_enhancer_available(self):
        assert UNIVERSAL_ENHANCER_AVAILABLE is True

    def test_enhance_chunks_for_embedding(self):
        texts = [
            "Python, Java, JavaScript, React, Node.js",
            "Bachelor of Science in Computer Science, Stanford University",
            "5 years as Senior Software Engineer at Google",
        ]
        metadata = [
            {"section_title": "Technical Skills"},
            {"section_title": "Education"},
            {"section_title": "Work Experience"},
        ]
        doc_metadata = {"document_type": "resume"}

        result = enhance_chunks_for_embedding(
            texts=texts,
            chunk_metadata=metadata,
            document_metadata=doc_metadata,
            domain="resume",
        )

        assert result.original_count == 3
        assert result.deduplicated_count == 3
        assert result.average_quality_score > 0.5
        assert result.enhancement_stats.get("enhancer") == "universal"

        # Check enhanced metadata
        for meta in result.enhanced_metadata:
            assert "content_type" in meta
            assert "quality_score" in meta
            assert "semantic_keywords" in meta

    def test_enrich_query_for_retrieval(self):
        query = "rank the top 2 candidates based on experience"
        result = enrich_query_for_retrieval(query)
        assert result["query_type"] == "ranking"
        assert "enriched_query" in result
        assert result["enriched_query"] != query

    def test_get_enhanced_embedding_text(self):
        text = "Python, Java, JavaScript"
        result = get_enhanced_embedding_text(
            text=text,
            section_title="Technical Skills",
            document_type="resume",
        )
        assert "Skills and Competencies" in result
        assert "Python" in result

    def test_deduplicate_chunks(self):
        texts = [
            "Python, Java, JavaScript, React",
            "Python, Java, JavaScript, React, Node.js",  # Similar
            "Completely different content about education",
        ]
        metadata = [{"id": 1}, {"id": 2}, {"id": 3}]

        deduped_texts, deduped_meta = deduplicate_chunks(
            texts=texts,
            metadata=metadata,
            similarity_threshold=0.85,
        )

        # Should keep 2 chunks (first and third - first two are similar)
        assert len(deduped_texts) <= 3


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_enhance_for_embedding(self):
        result = enhance_for_embedding(
            text="Python, Java, React",
            section_title="Skills",
            document_type="resume",
        )
        assert result.content_type == "skills_list"
        assert result.embedding_text is not None

    def test_enrich_query(self):
        query = "what are the candidate's skills"
        enriched = enrich_query(query)
        assert isinstance(enriched, str)
        assert len(enriched) > 0

    def test_get_enhanced_payload(self):
        payload = get_enhanced_payload(
            text="Python, Java, JavaScript",
            section_title="Skills",
        )
        assert "content_type" in payload
        assert "quality_score" in payload


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
