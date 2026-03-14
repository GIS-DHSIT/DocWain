"""Tests for intelligent answer generation pipeline."""
from __future__ import annotations
import pytest


class TestIntentClassification:
    """Tests for expanded 8-type intent classification."""

    def test_factual_extraction_detected(self):
        from src.rag_v3.llm_extract import classify_query_intent
        assert classify_query_intent("What is Prudhvi's email address?") == "factual"

    def test_comparison_detected(self):
        from src.rag_v3.llm_extract import classify_query_intent
        assert classify_query_intent("Compare Ajay and Prudhvi's cloud experience") == "comparison"

    def test_summary_detected(self):
        from src.rag_v3.llm_extract import classify_query_intent
        assert classify_query_intent("Summarize Gaurav's resume") == "summary"

    def test_ranking_detected(self):
        from src.rag_v3.llm_extract import classify_query_intent
        assert classify_query_intent("Rank all candidates by years of experience") == "ranking"

    def test_timeline_detected(self):
        from src.rag_v3.llm_extract import classify_query_intent
        assert classify_query_intent("Show Bharath's career progression over time") == "timeline"

    def test_multi_field_extraction_detected(self):
        from src.rag_v3.llm_extract import classify_query_intent
        assert classify_query_intent("Extract all invoice line items with amounts") == "multi_field"

    def test_reasoning_detected(self):
        from src.rag_v3.llm_extract import classify_query_intent
        assert classify_query_intent("Is Ajay qualified for a DevOps lead position?") == "reasoning"

    def test_cross_document_detected(self):
        from src.rag_v3.llm_extract import classify_query_intent
        result = classify_query_intent("What skills do all candidates share?")
        assert result == "cross_document"

    def test_generic_query_classifies_appropriately(self):
        from src.rag_v3.llm_extract import classify_query_intent
        # "tell me about" is an overview request — summary or factual are both valid
        result = classify_query_intent("tell me about this document")
        assert result in ("factual", "summary")

    def test_hint_overrides_regex(self):
        from src.rag_v3.llm_extract import classify_query_intent
        assert classify_query_intent("show me details", intent_hint="comparison") == "comparison"


class TestPromptTemplates:
    """Tests for intent-adaptive prompt construction."""

    def test_comparison_prompt_has_table_instruction(self):
        from src.rag_v3.llm_extract import build_generation_prompt
        prompt = build_generation_prompt(
            query="Compare Ajay and Prudhvi",
            evidence_text="Doc evidence here",
            intent="comparison",
            num_documents=2,
        )
        assert "table" in prompt.lower() or "side" in prompt.lower()
        assert "Compare Ajay and Prudhvi" in prompt

    def test_ranking_prompt_has_ranking_instruction(self):
        from src.rag_v3.llm_extract import build_generation_prompt
        prompt = build_generation_prompt(
            query="Rank candidates by skills",
            evidence_text="Doc evidence here",
            intent="ranking",
            num_documents=4,
        )
        assert "rank" in prompt.lower()
        assert "4" in prompt

    def test_reasoning_prompt_has_evidence_chain(self):
        from src.rag_v3.llm_extract import build_generation_prompt
        prompt = build_generation_prompt(
            query="Is Ajay qualified for DevOps?",
            evidence_text="Doc evidence here",
            intent="reasoning",
            num_documents=1,
        )
        assert "evidence" in prompt.lower()
        assert "gap" in prompt.lower() or "missing" in prompt.lower()

    def test_factual_prompt_is_direct(self):
        from src.rag_v3.llm_extract import build_generation_prompt
        prompt = build_generation_prompt(
            query="What is the email?",
            evidence_text="Doc evidence here",
            intent="factual",
            num_documents=1,
        )
        assert "direct" in prompt.lower() or "specific" in prompt.lower()

    def test_summary_prompt_has_structure(self):
        from src.rag_v3.llm_extract import build_generation_prompt
        prompt = build_generation_prompt(
            query="Summarize this resume",
            evidence_text="Doc evidence here",
            intent="summary",
            num_documents=1,
        )
        assert "key" in prompt.lower() or "highlights" in prompt.lower()

    def test_prompt_includes_evidence(self):
        from src.rag_v3.llm_extract import build_generation_prompt
        prompt = build_generation_prompt(
            query="test query",
            evidence_text="UNIQUE_EVIDENCE_MARKER",
            intent="factual",
            num_documents=1,
        )
        assert "UNIQUE_EVIDENCE_MARKER" in prompt

    def test_prompt_under_context_limit(self):
        from src.rag_v3.llm_extract import build_generation_prompt, LLM_MAX_CONTEXT_CHARS
        long_evidence = "x" * 20000
        prompt = build_generation_prompt(
            query="test",
            evidence_text=long_evidence,
            intent="factual",
            num_documents=1,
        )
        assert len(prompt) <= LLM_MAX_CONTEXT_CHARS + 7000  # GPT-parity prompts add ~6.5K overhead


class TestUnifiedGeneration:
    """Tests for unified generation path wiring."""

    def test_llm_extract_uses_expanded_intent(self):
        from unittest.mock import MagicMock, patch
        from src.rag_v3.llm_extract import llm_extract_and_respond
        from src.rag_v3.types import LLMBudget, Chunk, ChunkSource

        mock_client = MagicMock()
        _resp = "Ajay has 14 years of cloud experience supporting his qualification for DevOps lead role."
        mock_client.generate_with_metadata.return_value = (_resp, {})
        mock_client.chat_with_metadata.return_value = (_resp, {})
        budget = LLMBudget(llm_client=mock_client, max_calls=4)

        chunks = [Chunk(id="1", text="Ajay has 14 years cloud architecture", score=0.9,
                        source=ChunkSource(document_name="ajay.pdf"),
                        meta={"source_name": "ajay.pdf"})]

        with patch("src.rag_v3.llm_extract.classify_query_intent", return_value="reasoning") as mock_intent:
            result = llm_extract_and_respond(
                query="Is Ajay qualified for DevOps?",
                chunks=chunks, llm_client=mock_client, budget=budget,
            )

        mock_intent.assert_called_once()
        assert result is not None
        assert "Ajay" in result.text

    def test_llm_extract_passes_num_documents(self):
        from unittest.mock import MagicMock, patch
        from src.rag_v3.llm_extract import llm_extract_and_respond
        from src.rag_v3.types import LLMBudget, Chunk, ChunkSource

        mock_client = MagicMock()
        mock_client.generate_with_metadata.return_value = (
            "Candidate 1 ranks higher than Candidate 2 based on their skills.",
            {},
        )
        budget = LLMBudget(llm_client=mock_client, max_calls=4)

        chunks = [
            Chunk(id="1", text="Alice skills", score=0.9,
                  source=ChunkSource(document_name="alice.pdf"), meta={"source_name": "alice.pdf"}),
            Chunk(id="2", text="Bob skills", score=0.8,
                  source=ChunkSource(document_name="bob.pdf"), meta={"source_name": "bob.pdf"}),
        ]

        with patch("src.rag_v3.llm_extract.build_generation_prompt") as mock_prompt:
            mock_prompt.return_value = "test prompt"
            result = llm_extract_and_respond(
                query="Rank candidates", chunks=chunks,
                llm_client=mock_client, budget=budget, num_documents=2,
            )

        call_kwargs = mock_prompt.call_args[1]
        assert call_kwargs["num_documents"] == 2


class TestEvidenceChain:
    """Tests for chain-of-evidence reasoning."""

    def test_build_evidence_chain_identifies_supporting_facts(self):
        from src.rag_v3.evidence_chain import build_evidence_chain
        from src.rag_v3.types import Chunk, ChunkSource

        chunks = [
            Chunk(id="1", text="Ajay has 14 years of cloud architecture experience",
                  score=0.9, source=ChunkSource(document_name="ajay.pdf"), meta={}),
            Chunk(id="2", text="Led Azure migration reducing costs by 40%",
                  score=0.8, source=ChunkSource(document_name="ajay.pdf"), meta={}),
        ]
        chain = build_evidence_chain("Is Ajay qualified for DevOps lead?", chunks)
        assert len(chain.supporting_facts) >= 1
        assert any("cloud" in f.text.lower() or "azure" in f.text.lower()
                    for f in chain.supporting_facts)

    def test_build_evidence_chain_identifies_gaps(self):
        from src.rag_v3.evidence_chain import build_evidence_chain
        from src.rag_v3.types import Chunk, ChunkSource

        chunks = [
            Chunk(id="1", text="Ajay has cloud architecture experience with Azure",
                  score=0.9, source=ChunkSource(document_name="ajay.pdf"), meta={}),
        ]
        chain = build_evidence_chain("Does Ajay have CI/CD pipeline experience?", chunks)
        assert len(chain.gaps) >= 1

    def test_evidence_chain_renders_to_context(self):
        from src.rag_v3.evidence_chain import build_evidence_chain
        from src.rag_v3.types import Chunk, ChunkSource

        chunks = [
            Chunk(id="1", text="Ajay has 14 years cloud experience",
                  score=0.9, source=ChunkSource(document_name="ajay.pdf"), meta={}),
        ]
        chain = build_evidence_chain("Is Ajay qualified?", chunks)
        context = chain.render_for_prompt()
        assert isinstance(context, str)
        assert len(context) > 0

    def test_evidence_chain_cross_doc_groups_by_document(self):
        from src.rag_v3.evidence_chain import build_evidence_chain
        from src.rag_v3.types import Chunk, ChunkSource

        chunks = [
            Chunk(id="1", text="Alice has Python skills",
                  score=0.9, source=ChunkSource(document_name="alice.pdf"), meta={"source_name": "alice.pdf"}),
            Chunk(id="2", text="Bob has Python and Java skills",
                  score=0.8, source=ChunkSource(document_name="bob.pdf"), meta={"source_name": "bob.pdf"}),
        ]
        chain = build_evidence_chain("What skills do candidates share?", chunks)
        assert chain.num_documents >= 2


class TestEvidenceChainInjection:
    """Tests for evidence chain injection into prompts."""

    def test_reasoning_prompt_includes_evidence_chain(self):
        from unittest.mock import MagicMock
        from src.rag_v3.llm_extract import llm_extract_and_respond
        from src.rag_v3.types import LLMBudget, Chunk, ChunkSource

        mock_client = MagicMock()
        _resp = "Based on evidence, Ajay is qualified for DevOps lead role."
        mock_client.generate_with_metadata.return_value = (_resp, {})
        mock_client.chat_with_metadata.return_value = (_resp, {})
        budget = LLMBudget(llm_client=mock_client, max_calls=4)

        chunks = [Chunk(id="1", text="Ajay has cloud experience", score=0.9,
                        source=ChunkSource(document_name="ajay.pdf"), meta={"source_name": "ajay.pdf"})]

        result = llm_extract_and_respond(
            query="Is Ajay qualified for DevOps?",
            chunks=chunks, llm_client=mock_client, budget=budget,
        )

        # _generate uses chat_with_metadata (system/user messages) when available
        call_args = mock_client.chat_with_metadata.call_args
        if call_args is None:
            call_args = mock_client.generate_with_metadata.call_args
            prompt_sent = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        else:
            messages = call_args[0][0] if call_args[0] else call_args[1].get("messages", [])
            prompt_sent = " ".join(m.get("content", "") for m in messages)
        assert "EVIDENCE FOUND" in prompt_sent or "INFORMATION NOT FOUND" in prompt_sent

    def test_factual_prompt_does_not_include_evidence_chain(self):
        from unittest.mock import MagicMock
        from src.rag_v3.llm_extract import llm_extract_and_respond
        from src.rag_v3.types import LLMBudget, Chunk, ChunkSource

        mock_client = MagicMock()
        _resp = "The email is test@example.com which is the contact email."
        mock_client.generate_with_metadata.return_value = (_resp, {})
        mock_client.chat_with_metadata.return_value = (_resp, {})
        budget = LLMBudget(llm_client=mock_client, max_calls=4)

        chunks = [Chunk(id="1", text="Email: test@example.com", score=0.9,
                        source=ChunkSource(document_name="doc.pdf"), meta={})]

        result = llm_extract_and_respond(
            query="What is the email?",
            chunks=chunks, llm_client=mock_client, budget=budget,
        )

        # _generate uses chat_with_metadata (system/user messages) when available
        call_args = mock_client.chat_with_metadata.call_args
        if call_args is None:
            call_args = mock_client.generate_with_metadata.call_args
            prompt_sent = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        else:
            messages = call_args[0][0] if call_args[0] else call_args[1].get("messages", [])
            prompt_sent = " ".join(m.get("content", "") for m in messages)
        assert "EVIDENCE FOUND" not in prompt_sent


class TestGroundingGate:
    """Tests for post-generation grounding verification."""

    def test_grounded_answer_passes(self):
        from src.quality.fast_grounding import evaluate_grounding
        answer = "Ajay has 14 years of cloud architecture experience."
        chunks = ["Ajay has 14 years of cloud architecture experience with Azure."]
        result = evaluate_grounding(answer, chunks)
        assert result.supported_ratio >= 0.8

    def test_hallucinated_answer_detected(self):
        from src.quality.fast_grounding import evaluate_grounding
        answer = (
            "Ravi holds a PhD in quantum computing from Stanford. "
            "He published 15 papers on blockchain consensus protocols."
        )
        chunks = ["Ajay has 14 years of cloud architecture experience."]
        result = evaluate_grounding(answer, chunks)
        assert result.critical_supported_ratio < 0.8


class TestEntityAndNumericVerification:
    """Tests for entity consistency and numeric verification in judge."""

    def test_entity_mismatch_detected(self):
        from src.rag_v3.judge import _check_entity_consistency
        result = _check_entity_consistency(
            query="Tell me about Prudhvi",
            answer="Ajay has 14 years of cloud experience and excellent leadership skills.",
            intent="detail",
        )
        assert result is not None  # Should return a warning/issue

    def test_entity_match_passes(self):
        from src.rag_v3.judge import _check_entity_consistency
        result = _check_entity_consistency(
            query="Tell me about Ajay",
            answer="Ajay has 14 years of cloud experience.",
            intent="detail",
        )
        assert result is None  # No issue

    def test_comparison_allows_multiple_entities(self):
        from src.rag_v3.judge import _check_entity_consistency
        result = _check_entity_consistency(
            query="Compare Ajay and Prudhvi",
            answer="Ajay has cloud experience while Prudhvi specializes in data.",
            intent="comparison",
        )
        assert result is None  # Comparison allows multiple entities

    def test_hallucinated_number_detected(self):
        from src.rag_v3.judge import _check_numeric_fidelity
        answer = "Ajay has 25 years of experience."
        evidence_texts = ["Ajay has 14 years of cloud architecture experience."]
        result = _check_numeric_fidelity(answer, evidence_texts)
        assert len(result) >= 1  # 25 is hallucinated

    def test_correct_number_passes(self):
        from src.rag_v3.judge import _check_numeric_fidelity
        answer = "Ajay has 14 years of experience."
        evidence_texts = ["Ajay has 14 years of cloud architecture experience."]
        result = _check_numeric_fidelity(answer, evidence_texts)
        assert len(result) == 0  # 14 is grounded


class TestFullPipelineIntegration:
    """Integration tests for the complete answer generation pipeline."""

    def test_reasoning_query_full_pipeline(self):
        """Full pipeline for a reasoning query with mock LLM."""
        from unittest.mock import MagicMock
        from src.rag_v3.llm_extract import classify_query_intent, build_generation_prompt, llm_extract_and_respond
        from src.rag_v3.evidence_chain import build_evidence_chain
        from src.rag_v3.types import LLMBudget, Chunk, ChunkSource

        query = "Is Ajay qualified for a DevOps lead position?"
        chunks = [
            Chunk(id="1", text="Ajay has 14 years of cloud architecture experience with Azure and AWS.",
                  score=0.9, source=ChunkSource(document_name="ajay.pdf"), meta={"source_name": "ajay.pdf"}),
            Chunk(id="2", text="Led migration project at Bosch reducing costs by 40%.",
                  score=0.8, source=ChunkSource(document_name="ajay.pdf"), meta={"source_name": "ajay.pdf"}),
        ]

        # Step 1: Intent classification
        intent = classify_query_intent(query)
        assert intent == "reasoning"

        # Step 2: Evidence chain
        chain = build_evidence_chain(query, chunks)
        assert len(chain.supporting_facts) >= 1

        # Step 3: Mock LLM returns a well-reasoned response
        mock_llm = MagicMock()
        mock_llm.generate_with_metadata.return_value = (
            "Ajay appears well-qualified for a DevOps lead role based on several factors:\n\n"
            "Infrastructure expertise: He has 14 years of cloud architecture experience "
            "with hands-on Azure and AWS work.\n\n"
            "Leadership: Led migration project at Bosch reducing costs by 40%.\n\n"
            "Gap to consider: The documents don't mention specific CI/CD tooling experience.",
            {}
        )
        budget = LLMBudget(llm_client=mock_llm, max_calls=4)

        # Step 4: Generate response
        result = llm_extract_and_respond(
            query=query, chunks=chunks, llm_client=mock_llm, budget=budget,
        )

        assert result is not None
        assert "14 years" in result.text
        assert "gap" in result.text.lower() or "don't mention" in result.text.lower()

    def test_factual_query_full_pipeline(self):
        """Full pipeline for a simple factual query."""
        from unittest.mock import MagicMock
        from src.rag_v3.llm_extract import classify_query_intent, llm_extract_and_respond
        from src.rag_v3.types import LLMBudget, Chunk, ChunkSource

        query = "What is Bharath's email address?"
        chunks = [
            Chunk(id="1", text="Bharath Kumar\nofficalbharath7@gmail.com\n8686426242",
                  score=0.95, source=ChunkSource(document_name="bharath.pdf"), meta={}),
        ]

        intent = classify_query_intent(query)
        assert intent == "factual"

        mock_llm = MagicMock()
        mock_llm.generate_with_metadata.return_value = (
            "Bharath's email address is officialbharath7@gmail.com according to the resume.",
            {}
        )
        budget = LLMBudget(llm_client=mock_llm, max_calls=4)

        result = llm_extract_and_respond(
            query=query, chunks=chunks, llm_client=mock_llm, budget=budget,
        )

        assert result is not None
        assert "gmail.com" in result.text

    def test_comparison_query_produces_structured_output(self):
        """Comparison query should produce structured multi-entity response."""
        from unittest.mock import MagicMock
        from src.rag_v3.llm_extract import classify_query_intent, llm_extract_and_respond
        from src.rag_v3.types import LLMBudget, Chunk, ChunkSource

        query = "Compare Bharath and Gaurav's SAP experience"
        chunks = [
            Chunk(id="1", text="Bharath: SAP SD Consultant, 3+ years",
                  score=0.9, source=ChunkSource(document_name="bharath.pdf"),
                  meta={"source_name": "bharath.pdf"}),
            Chunk(id="2", text="Gaurav: SAP EWM Consultant, 5 years",
                  score=0.85, source=ChunkSource(document_name="gaurav.pdf"),
                  meta={"source_name": "gaurav.pdf"}),
        ]

        intent = classify_query_intent(query)
        assert intent == "comparison"

        mock_llm = MagicMock()
        mock_llm.generate_with_metadata.return_value = (
            "Comparison of SAP Experience:\n\n"
            "Bharath Kumar: SAP SD Consultant with 3+ years of experience.\n"
            "Gaurav Fegade: SAP EWM Consultant with 5 years of experience.\n\n"
            "Key difference: Gaurav has more years of experience.",
            {}
        )
        budget = LLMBudget(llm_client=mock_llm, max_calls=4)

        result = llm_extract_and_respond(
            query=query, chunks=chunks, llm_client=mock_llm, budget=budget,
            num_documents=2,
        )

        assert result is not None
        assert "Bharath" in result.text
        assert "Gaurav" in result.text
