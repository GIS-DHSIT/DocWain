import json
from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.api.config import Config

logger = get_logger(__name__)

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "by",
    "is", "are", "was", "were", "be", "been", "it", "this", "that", "as", "at",
}

@dataclass
class VerificationReport:
    citations_valid: bool
    invalid_citations: List[int]
    missing_citations: int
    unsupported_sentences: List[str]
    support_score: float
    coverage_score: float
    numeric_support_rate: float
    overall_grounded: bool

class EvidencePlanner:
    """Internal reasoning layer to map claims to evidence without exposing chain-of-thought."""

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

    def plan(self, query: str, context: str) -> Dict[str, Any]:
        if not self._enabled() or not self.llm_client:
            return {}
        prompt = f"""You are an internal reasoning engine. Build a concise evidence plan to answer the question
using ONLY the provided context. Output strict JSON only.

Question: {query}

Context:
{context}

Return format:
{{
  "steps": [
    {{"claim": "...", "sources": [1,2]}}
  ],
  "missing_info": "..."
}}"""
        try:
            response = self.llm_client.generate(prompt, max_retries=2, backoff=0.4)
            return self._extract_json(response)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Evidence plan generation failed: %s", exc)
            return {}

    @staticmethod
    def _enabled() -> bool:
        return bool(getattr(Config.Retrieval, "REASONING_LAYER_ENABLED", True))

    @staticmethod
    def _extract_json(raw: str) -> Dict[str, Any]:
        if not raw:
            return {}
        raw = raw.strip()
        if raw.startswith("{") and raw.endswith("}"):
            return json.loads(raw)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        return json.loads(match.group(0))

class AnswerVerifier:
    """Verify citations and claim support against sources."""

    def __init__(self):
        self.min_support = float(getattr(Config.Retrieval, "MIN_SUPPORT_SCORE", 0.15))
        self.min_coverage = float(getattr(Config.Retrieval, "MIN_CITATION_COVERAGE", 0.75))

    def verify(self, answer: str, sources: List[Dict[str, Any]]) -> VerificationReport:
        sentences = self._split_sentences(answer)
        invalid_citations: List[int] = []
        unsupported_sentences: List[str] = []
        missing_citations = 0
        support_scores = []
        numeric_supported = 0
        numeric_total = 0

        for sentence in sentences:
            cited = self._extract_citations(sentence, sources)
            if not cited:
                if not self._is_meta_statement(sentence):
                    missing_citations += 1
                    unsupported_sentences.append(sentence)
                continue

            invalid = [c for c in cited if c < 1 or c > len(sources)]
            if invalid:
                invalid_citations.extend(invalid)
                unsupported_sentences.append(sentence)
                continue

            source_texts = [sources[c - 1].get("excerpt", "") for c in cited]
            support = max(self._overlap_score(sentence, text) for text in source_texts) if source_texts else 0.0
            support_scores.append(support)
            if support < self.min_support:
                unsupported_sentences.append(sentence)

            nums = re.findall(r"\d+(?:\.\d+)?", sentence)
            if nums:
                numeric_total += 1
                if any(num in " ".join(source_texts) for num in nums):
                    numeric_supported += 1

        coverage_score = 1.0 if not sentences else max(0.0, 1.0 - (missing_citations / len(sentences)))
        support_score = sum(support_scores) / len(support_scores) if support_scores else 0.0
        numeric_support_rate = (numeric_supported / numeric_total) if numeric_total else 1.0

        citations_valid = len(invalid_citations) == 0
        overall_grounded = (
            citations_valid
            and coverage_score >= self.min_coverage
            and support_score >= self.min_support
        )

        return VerificationReport(
            citations_valid=citations_valid,
            invalid_citations=sorted(set(invalid_citations)),
            missing_citations=missing_citations,
            unsupported_sentences=unsupported_sentences,
            support_score=round(support_score, 4),
            coverage_score=round(coverage_score, 4),
            numeric_support_rate=round(numeric_support_rate, 4),
            overall_grounded=overall_grounded,
        )

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        raw = re.split(r"(?<=[.!?])\s+", text or "")
        return [s.strip() for s in raw if s.strip()]

    @staticmethod
    def _extract_citations(text: str, sources: List[Dict[str, Any]]) -> List[int]:
        numeric = [int(c) for c in re.findall(r"\[SOURCE-(\d+)\]", text) if c.isdigit()]
        if numeric:
            return numeric

        # Match doc/section style citations
        labels = re.findall(r"\[(?:Doc|Document):([^\]]+)\]", text, flags=re.IGNORECASE)
        if not labels:
            return []

        indices: List[int] = []
        for label in labels:
            idx = AnswerVerifier._match_label_to_source(label, sources)
            if idx is not None:
                indices.append(idx + 1)
        return indices

    @staticmethod
    def _match_label_to_source(label: str, sources: List[Dict[str, Any]]) -> Optional[int]:
        cleaned = re.sub(r"[\|\:]", " ", label or "").lower()
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return None
        label_tokens = set(re.findall(r"[a-z0-9]{3,}", cleaned))
        if not label_tokens:
            return None
        best_idx = None
        best_score = 0.0
        for idx, src in enumerate(sources):
            name = str(src.get("source_name") or "")
            section = str(src.get("section") or "")
            page = str(src.get("page") or "")
            text = f"{name} {section} {page}".lower()
            tokens = set(re.findall(r"[a-z0-9]{3,}", text))
            if not tokens:
                continue
            overlap = len(label_tokens & tokens) / max(len(label_tokens), 1)
            if overlap > best_score:
                best_score = overlap
                best_idx = idx
        if best_score >= 0.15:
            return best_idx
        return None

    @staticmethod
    def _is_meta_statement(sentence: str) -> bool:
        lower = sentence.lower()
        return any(
            phrase in lower
            for phrase in [
                "the documents",
                "not contain",
                "cannot answer",
                "not enough detail",
                "unclear",
                "conflicting",
            ]
        )

    @staticmethod
    def _overlap_score(sentence: str, source: str) -> float:
        sentence_tokens = _tokenize(sentence)
        source_tokens = _tokenize(source)
        if not sentence_tokens or not source_tokens:
            return 0.0
        intersection = set(sentence_tokens) & set(source_tokens)
        return len(intersection) / max(len(set(sentence_tokens)), 1)

class ConfidenceScorer:
    """Compute a holistic confidence score from retrieval and verification signals."""

    def score(
        self,
        retrieved_chunks: List[Any],
        context_sources: List[Dict[str, Any]],
        verification: VerificationReport,
    ) -> Tuple[float, Dict[str, float]]:
        retrieval_strength = self._retrieval_strength(retrieved_chunks)
        diversity = self._source_diversity(context_sources)
        citation_validity = 1.0 if verification.citations_valid else 0.0

        confidence = (
            0.35 * verification.support_score
            + 0.2 * verification.coverage_score
            + 0.2 * retrieval_strength
            + 0.15 * citation_validity
            + 0.1 * diversity
        )
        confidence = max(0.0, min(1.0, confidence))

        breakdown = {
            "retrieval_strength": round(retrieval_strength, 4),
            "support_score": verification.support_score,
            "coverage_score": verification.coverage_score,
            "citation_validity": round(citation_validity, 4),
            "diversity_score": round(diversity, 4),
            "numeric_support_rate": verification.numeric_support_rate,
        }
        return confidence, breakdown

    @staticmethod
    def _retrieval_strength(chunks: List[Any]) -> float:
        if not chunks:
            return 0.0
        top_scores = [float(chunk.score) for chunk in chunks[:5] if chunk and chunk.score is not None]
        if not top_scores:
            return 0.0
        max_score = max(top_scores)
        if max_score <= 0:
            return 0.0
        if max_score < 0.2:
            return min(1.0, max_score / 0.2)
        if max_score <= 1.2:
            return min(1.0, max_score)
        return 1.0

    @staticmethod
    def _source_diversity(sources: List[Dict[str, Any]]) -> float:
        if not sources:
            return 0.0
        unique = len({str(src.get("source_name")) for src in sources if src.get("source_name")})
        return min(1.0, unique / max(1, len(sources)))

def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [tok for tok in tokens if tok not in STOPWORDS]
