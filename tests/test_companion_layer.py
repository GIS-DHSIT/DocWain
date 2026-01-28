from src.agentic.post_processor import PostProcessor
from src.agentic.response_templates import ResponseTemplateSelector
from src.api.reasoning_layer import AnswerVerifier, ConfidenceScorer
from src.chat.companion_classifier import CompanionClassifier
from src.chat.opener_generator import contains_banned_opener, generate_opener


class DummyRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value


def test_opener_generator_avoids_banned_phrases():
    intents = [
        "factual",
        "summary",
        "comparison",
        "how-to",
        "troubleshooting",
        "extraction",
        "analysis",
    ]
    for intent in intents:
        opener = generate_opener(
            intent=intent,
            sentiment="neutral",
            follow_up=False,
            style_directives={},
            query="test query",
        )
        assert opener
        assert not contains_banned_opener(opener)


def test_companion_classifier_sentiment():
    classifier = CompanionClassifier(DummyRedis(), ttl_seconds=30)
    frustrated = classifier.classify(
        "bad answer", "", "factual", session_id="s1"
    )
    thankful = classifier.classify(
        "thank you", "", "factual", session_id="s2"
    )
    assert frustrated.sentiment == "frustrated"
    assert thankful.sentiment == "thankful"


def test_companion_classifier_cache_hit_miss():
    classifier = CompanionClassifier(DummyRedis(), ttl_seconds=30)
    _ = classifier.classify("hello", "", "factual", session_id="s3")
    _ = classifier.classify("hello", "", "factual", session_id="s3")
    metrics = classifier.cache.metrics["companion_classifier"].as_dict()
    assert metrics["hits"] == 1
    assert metrics["misses"] == 1


def test_template_selection_comparison_table():
    template = ResponseTemplateSelector.select("comparison", {})
    assert template.format_style == "comparison_table"


def test_template_selection_how_to_steps():
    template = ResponseTemplateSelector.select("how-to", {})
    assert template.format_style == "numbered_steps"


def test_postprocessor_rewrites_banned_opener():
    verifier = AnswerVerifier()
    scorer = ConfidenceScorer()
    post = PostProcessor(verifier, scorer)
    banned_text = "I reviewed the documents and pulled the most relevant passages. Fact one."
    rewritten = post._rewrite_banned_opener(
        banned_text, opener="Got it - here's what I found from the available context."
    )
    assert rewritten.startswith("Got it - here's what I found from the available context.")
    assert "I reviewed the documents" not in rewritten
