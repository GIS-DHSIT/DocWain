from types import SimpleNamespace

from src.agentic.retriever_manager import RetrieverManager
from src.agentic.memory import AgentMemory
from src.agentic.model_arbitration import ModelArbitrationLayer, ModelCandidate
from src.agentic.post_processor import PostProcessor
from src.api.query_intelligence import QueryIntelligence
from src.api.reasoning_layer import AnswerVerifier, ConfidenceScorer


class DummyChunk:
    def __init__(self, chunk_id: str, text: str, score: float, method: str):
        self.id = chunk_id
        self.text = text
        self.score = score
        self.method = method
        self.metadata = {"chunk_id": chunk_id}


def test_parallel_merge_deduplicates_and_keeps_best_score():
    manager = RetrieverManager(qdrant_retriever=None, hybrid_retriever=None)
    memory = AgentMemory()
    chunk_a = DummyChunk("c1", "alpha", 0.2, "dense")
    chunk_b = DummyChunk("c1", "alpha", 0.8, "sparse")
    merged = manager._merge_chunks(
        [("dense", [chunk_a]), ("sparse", [chunk_b])],
        ["alpha query"],
        memory,
    )
    assert len(merged) == 1
    assert merged[0].score == 0.8
    assert "dense" in merged[0].metadata.get("methods", [])
    assert "sparse" in merged[0].metadata.get("methods", [])


def test_instruction_parsing_detects_table_and_empathy():
    analysis = QueryIntelligence().analyze("Compare plan A vs plan B in a table, be brief and empathetic.")
    assert analysis.instructions.get("use_table") is True
    assert analysis.instructions.get("brevity") == "brief"
    assert analysis.instructions.get("empathy") is True


def test_post_processor_prefers_grounded_candidate():
    verifier = AnswerVerifier()
    scorer = ConfidenceScorer()
    post = PostProcessor(verifier, scorer)
    sources = [{"excerpt": "Revenue increased to 10", "source_name": "Doc"}]
    chunks = [SimpleNamespace(score=0.6)]
    candidates = [
        ("model_a", "Revenue increased to 10 [SOURCE-1]."),
        ("model_b", "Revenue doubled last year."),
    ]
    result = post.select_best(
        candidates,
        sources=sources,
        retrieved_chunks=chunks,
        confidence_threshold=0.1,
    )
    assert "Revenue increased to 10" in result.answer
    assert result.verification.citations_valid is True


def test_model_arbitration_falls_back_on_failure():
    def factory(model_name: str, backend_override=None):
        class Client:
            def generate(self, prompt, **kwargs):
                if model_name == "bad":
                    raise RuntimeError("boom")
                return f"{model_name} response"

        return Client()

    layer = ModelArbitrationLayer(factory, max_workers=2)
    responses = layer.generate_candidates(
        "prompt",
        primary_model="good",
        candidate_models=[ModelCandidate(name="bad"), ModelCandidate(name="better")],
    )
    texts = [resp.text for resp in responses if resp.text]
    assert "good response" in texts
    assert "better response" in texts
