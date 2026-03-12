"""
Enhanced semantic chunking with quality metrics and coherence scoring.

Provides advanced chunk quality evaluation, semantic coherence assessment,
and intelligent chunk merging/splitting for optimal retrieval.
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = get_logger(__name__)

@dataclass(frozen=True)
class ChunkQualityMetrics:
    """Quality metrics for a chunk."""

    chunk_id: str
    text: str
    length_score: float  # 0-1, optimal around 250-450 tokens
    completeness_score: float  # 0-1, based on sentence boundaries
    coherence_score: float  # 0-1, semantic coherence
    information_density_score: float  # 0-1, entity and concept density
    readability_score: float  # 0-1, based on sentence complexity
    overall_quality: float  # 0-1, weighted average
    quality_grade: str  # A, B, C, D, F
    issues: List[str]  # List of quality issues detected

@dataclass(frozen=True)
class SemanticCohesionScore:
    """Semantic cohesion between chunks."""

    chunk_a_id: str
    chunk_b_id: str
    semantic_similarity: float  # 0-1
    topic_continuity: float  # 0-1, whether topics continue
    should_merge: bool
    merge_score: float  # 0-1, confidence in merge recommendation

class SemanticChunkQualityEvaluator:
    """Evaluates and scores chunk quality with semantic coherence metrics."""

    def __init__(self):
        self.min_chunk_length = 100  # Minimum characters
        self.opt_chunk_length = 300  # Optimal characters
        self.max_chunk_length = 1500  # Maximum characters
        self.min_tokens = 20
        self.opt_tokens = 250
        self.max_tokens = 450

    def evaluate_chunk_quality(
        self,
        chunk_text: str,
        chunk_id: str,
        context: Optional[str] = None,
    ) -> ChunkQualityMetrics:
        """Comprehensive quality evaluation for a chunk."""
        issues = []

        # Score 1: Length appropriateness
        length_score, length_issues = self._score_length(chunk_text)
        issues.extend(length_issues)

        # Score 2: Completeness (sentence boundaries)
        completeness_score, completeness_issues = self._score_completeness(chunk_text)
        issues.extend(completeness_issues)

        # Score 3: Semantic coherence
        coherence_score, coherence_issues = self._score_coherence(
            chunk_text,
            context=context,
        )
        issues.extend(coherence_issues)

        # Score 4: Information density
        density_score, density_issues = self._score_information_density(chunk_text)
        issues.extend(density_issues)

        # Score 5: Readability
        readability_score, readability_issues = self._score_readability(chunk_text)
        issues.extend(readability_issues)

        # Compute overall quality (weighted average)
        overall = (
            0.25 * length_score
            + 0.20 * completeness_score
            + 0.25 * coherence_score
            + 0.15 * density_score
            + 0.15 * readability_score
        )

        # Determine grade
        if overall >= 0.85:
            grade = "A"
        elif overall >= 0.70:
            grade = "B"
        elif overall >= 0.55:
            grade = "C"
        elif overall >= 0.40:
            grade = "D"
        else:
            grade = "F"

        return ChunkQualityMetrics(
            chunk_id=chunk_id,
            text=chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text,
            length_score=length_score,
            completeness_score=completeness_score,
            coherence_score=coherence_score,
            information_density_score=density_score,
            readability_score=readability_score,
            overall_quality=overall,
            quality_grade=grade,
            issues=list(set(issues)),  # Deduplicate
        )

    def _score_length(self, text: str) -> Tuple[float, List[str]]:
        """Score chunk length appropriateness."""
        issues = []
        char_len = len(text)
        token_count = len(text.split())

        if char_len < self.min_chunk_length:
            issues.append(f"too_short({char_len} chars)")
            score = char_len / self.min_chunk_length
        elif char_len > self.max_chunk_length:
            issues.append(f"too_long({char_len} chars)")
            score = self.max_chunk_length / char_len
        else:
            # Optimal range scoring
            distance_from_opt = abs(char_len - self.opt_chunk_length)
            max_distance = max(
                self.opt_chunk_length - self.min_chunk_length,
                self.max_chunk_length - self.opt_chunk_length,
            )
            score = 1.0 - (distance_from_opt / max_distance)

        if token_count < self.min_tokens:
            issues.append(f"too_few_tokens({token_count})")
            score *= 0.8
        elif token_count > self.max_tokens:
            issues.append(f"too_many_tokens({token_count})")
            score *= 0.85

        return min(1.0, max(0.0, score)), issues

    def _score_completeness(self, text: str) -> Tuple[float, List[str]]:
        """Score chunk sentence boundary appropriateness."""
        issues = []

        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.1, ["no_sentences"]

        # Check if starts and ends with complete sentences
        score = 1.0

        # Penalty for incomplete start
        if text.strip() and not text.strip()[0].isupper():
            issues.append("incomplete_start")
            score -= 0.1

        # Penalty for incomplete end
        if text.strip() and text.strip()[-1] not in ".!?":
            issues.append("incomplete_end")
            score -= 0.15

        # Bonus for having 2+ complete sentences
        if len(sentences) >= 2:
            score += 0.1

        # Check for hanging fragments
        fragment_ratio = sum(
            1 for s in sentences if len(s.split()) < 3
        ) / len(sentences)
        if fragment_ratio > 0.4:
            issues.append(f"many_fragments({fragment_ratio:.1%})")
            score -= 0.15

        return min(1.0, max(0.0, score)), issues

    def _score_coherence(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> Tuple[float, List[str]]:
        """Score semantic coherence of chunk."""
        issues = []
        score = 1.0

        # Check for topic drift (abrupt topic changes)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) > 2:
            # Simple coherence: check if nouns and verbs are consistent
            nouns = []
            for sentence in sentences:
                # Extract capitalized words and potential entities
                words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", sentence)
                nouns.extend(words)

            if nouns:
                # High repetition of entities suggests coherence
                entity_repetition = (
                    len(nouns) - len(set(nouns))
                ) / len(nouns)
                score += min(0.2, entity_repetition * 0.3)

        # Check for common coherence issues
        abrupt_changes = self._detect_topic_shifts(sentences)
        if abrupt_changes > 0:
            issues.append(f"topic_shifts({abrupt_changes})")
            score -= 0.1 * abrupt_changes

        # Check context consistency if provided
        if context:
            context_match = self._assess_context_match(text, context)
            if context_match < 0.3:
                issues.append("weak_context_match")
                score -= 0.1
            else:
                score += min(0.15, context_match * 0.15)

        return min(1.0, max(0.0, score)), issues

    def _score_information_density(self, text: str) -> Tuple[float, List[str]]:
        """Score information density and entity presence."""
        issues = []

        # Extract entities and key terms
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        unique_entities = set(entities)
        keywords = re.findall(r"\b[a-z]{3,}\b", text.lower())
        unique_keywords = set(keywords)

        text_words = len(text.split())
        if text_words == 0:
            return 0.0, ["empty_text"]

        entity_density = len(unique_entities) / (text_words / 10)
        keyword_density = len(unique_keywords) / (text_words / 10)

        # Optimal density is 1-3 entities and 8-15 unique keywords per 100 words
        entity_score = min(1.0, entity_density / 3.0)
        keyword_score = min(1.0, keyword_density / 15.0)

        score = 0.4 * entity_score + 0.6 * keyword_score

        if len(unique_entities) == 0:
            issues.append("no_named_entities")
            score -= 0.1

        if len(unique_keywords) < 5:
            issues.append("low_keyword_diversity")
            score -= 0.1

        return min(1.0, max(0.0, score)), issues

    def _score_readability(self, text: str) -> Tuple[float, List[str]]:
        """Score readability and language quality."""
        issues = []

        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0, ["no_sentences"]

        # Average sentence length (optimal 15-20 words)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

        if avg_sentence_length < 5:
            issues.append(f"very_short_sentences({avg_sentence_length:.1f})")
            score = 0.6
        elif avg_sentence_length < 10:
            issues.append(f"short_sentences({avg_sentence_length:.1f})")
            score = 0.8
        elif avg_sentence_length <= 25:
            score = 1.0  # Optimal range
        elif avg_sentence_length <= 35:
            issues.append(f"long_sentences({avg_sentence_length:.1f})")
            score = 0.85
        else:
            issues.append(f"very_long_sentences({avg_sentence_length:.1f})")
            score = 0.7

        # Check for excessive punctuation or formatting issues
        punct_ratio = len(re.findall(r"[!*()-]", text)) / max(1, len(text))
        if punct_ratio > 0.1:
            issues.append(f"excessive_punctuation({punct_ratio:.1%})")
            score -= 0.1

        return min(1.0, max(0.0, score)), issues

    def _detect_topic_shifts(self, sentences: List[str]) -> int:
        """Detect abrupt topic shifts between sentences."""
        if len(sentences) < 2:
            return 0

        shifts = 0
        for i in range(1, len(sentences)):
            prev_keywords = set(
                re.findall(r"\b[a-z]{3,}\b", sentences[i - 1].lower()),
            )
            curr_keywords = set(re.findall(r"\b[a-z]{3,}\b", sentences[i].lower()))

            # If less than 30% keyword overlap, likely topic shift
            overlap = len(prev_keywords & curr_keywords)
            overlap_ratio = (
                overlap / max(len(prev_keywords), len(curr_keywords))
                if max(len(prev_keywords), len(curr_keywords)) > 0
                else 0
            )

            if overlap_ratio < 0.3:
                shifts += 1

        return shifts

    def _assess_context_match(self, text: str, context: str) -> float:
        """Assess how well chunk matches its context."""
        if not context:
            return 0.5

        # Simple keyword overlap
        text_keywords = set(re.findall(r"\b[a-z]{4,}\b", text.lower()))
        context_keywords = set(re.findall(r"\b[a-z]{4,}\b", context.lower()))

        if not text_keywords or not context_keywords:
            return 0.5

        overlap = len(text_keywords & context_keywords)
        match_score = overlap / max(len(text_keywords), len(context_keywords))

        return match_score

    def filter_low_quality_chunks(
        self,
        chunks: List[Tuple[str, str]],  # (chunk_id, text)
        min_quality_grade: str = "C",
    ) -> Tuple[List[str], List[str], Dict[str, ChunkQualityMetrics]]:
        """Filter chunks by quality grade."""
        grade_hierarchy = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
        min_score = grade_hierarchy.get(min_quality_grade, 3)

        kept_ids = []
        filtered_ids = []
        metrics_map = {}

        for chunk_id, text in chunks:
            metrics = self.evaluate_chunk_quality(chunk_id, text)
            metrics_map[chunk_id] = metrics

            chunk_score = grade_hierarchy.get(metrics.quality_grade, 0)
            if chunk_score >= min_score:
                kept_ids.append(chunk_id)
            else:
                filtered_ids.append(chunk_id)

        return kept_ids, filtered_ids, metrics_map

    def recommend_merges(
        self,
        chunks: List[Tuple[str, str]],  # (chunk_id, text)
        similarity_threshold: float = 0.7,
        min_merge_length: int = 200,
    ) -> List[Tuple[str, str, float]]:
        """Recommend chunks to merge based on semantic similarity and quality."""
        merge_recommendations = []

        for i in range(len(chunks) - 1):
            chunk_a_id, text_a = chunks[i]
            chunk_b_id, text_b = chunks[i + 1]

            # Only consider consecutive chunks
            combined_length = len(text_a) + len(text_b)
            if combined_length > self.max_chunk_length:
                continue  # Too long to merge

            # Compute semantic similarity
            similarity = self._compute_text_similarity(text_a, text_b)

            if similarity >= similarity_threshold:
                if combined_length >= min_merge_length:
                    merge_recommendations.append(
                        (chunk_a_id, chunk_b_id, similarity),
                    )

        return sorted(merge_recommendations, key=lambda x: x[2], reverse=True)

    def _compute_text_similarity(self, text_a: str, text_b: str) -> float:
        """Simple text similarity based on keyword overlap."""
        keywords_a = set(re.findall(r"\b[a-z]{3,}\b", text_a.lower()))
        keywords_b = set(re.findall(r"\b[a-z]{3,}\b", text_b.lower()))

        if not keywords_a or not keywords_b:
            return 0.0

        intersection = len(keywords_a & keywords_b)
        union = len(keywords_a | keywords_b)

        return intersection / union if union > 0 else 0.0

__all__ = [
    "SemanticChunkQualityEvaluator",
    "ChunkQualityMetrics",
    "SemanticCohesionScore",
]

