"""
Enhanced Context Building and Answer Generation
Replaces context building logic in dw_newron.py
"""

import re
import time
from src.utils.logging_utils import get_logger
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

from src.utils.payload_utils import get_content_text, get_source_name

from src.api.config import Config
from src.prompting.prompt_builder import inject_persona_prompt
from src.embedding.pipeline.chunk_integrity import is_valid_chunk_text

logger = get_logger(__name__)

class IntelligentContextBuilder:
    """
    Builds optimized context from retrieved chunks.
    Key improvements:
    1. Deduplicates similar content
    2. Preserves document structure
    3. Orders chunks logically
    4. Includes chunk relationship information
    """

    def __init__(self, max_context_chunks: int = 12):
        self.max_context_chunks = max_context_chunks

    @staticmethod
    def _is_extraction_like_query(query: str) -> bool:
        lowered = (query or "").lower()
        if not lowered:
            return False
        cues = (
            "extract",
            "fields",
            "list",
            "provide",
            "resume",
            "linkedin",
            "cv",
            "candidate",
            "skills",
            "education",
            "certification",
            "experience",
            "years of experience",
            "name",
            "email",
            "phone",
            "summary",
            "achievements",
            "awards",
        )
        return any(cue in lowered for cue in cues)

    def _deduplicate_by_chunk_id(self, chunks: List[Dict]) -> List[Dict]:
        """Keep highest scoring chunk per unique chunk_id or doc/index tuple."""
        best = {}
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            key = meta.get("chunk_id") or (
                meta.get("document_id"),
                meta.get("chunk_index"),
            )
            if key not in best or chunk.get("score", 0) > best[key].get("score", 0):
                best[key] = chunk
        deduped = list(best.values())
        if len(deduped) != len(chunks):
            logger.info("Deduplicated by chunk_id: %s -> %s", len(chunks), len(deduped))
        return deduped

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _deduplicate_chunks(self, chunks: List[Dict], threshold: float = 0.7) -> List[Dict]:
        """Remove near-duplicate chunks using shingled hashing (O(n) instead of O(n²)).

        Builds a set of 4-word shingles per chunk and uses their frozenset hash
        as a fingerprint. Chunks with identical fingerprints keep only the
        highest-scoring version.
        """
        if len(chunks) <= 1:
            return chunks

        def _fingerprint(text: str) -> int:
            words = text.lower().split()
            if len(words) < 4:
                return hash(tuple(words))
            shingles = set()
            for i in range(len(words) - 3):
                shingles.add(tuple(words[i:i + 4]))
            return hash(frozenset(shingles))

        best: Dict[int, Dict] = {}
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            chunk_text = get_content_text(meta) or chunk.get("text", "")
            if not chunk_text.strip():
                continue
            fp = _fingerprint(chunk_text)
            if fp not in best or float(chunk.get("score", 0)) > float(best[fp].get("score", 0)):
                best[fp] = chunk

        unique_chunks = list(best.values())
        if len(unique_chunks) != len(chunks):
            logger.info("Deduplicated %d -> %d chunks", len(chunks), len(unique_chunks))
        return unique_chunks

    def _group_by_document(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """Group chunks by source document"""
        doc_groups = defaultdict(list)

        for chunk in chunks:
            doc_id = chunk['metadata'].get('document_id', 'unknown')
            doc_groups[doc_id].append(chunk)

        return dict(doc_groups)

    @staticmethod
    def _recency_factor(meta: Dict) -> float:
        """Light recency boost based on known timestamp fields."""
        for key in ("ingested_at", "updated_at", "created_at", "timestamp"):
            if key in meta:
                try:
                    ts = float(meta[key])
                    age_days = max(0.0, (time.time() - ts) / 86400.0)
                    return 1.0 + max(0.0, (30.0 - age_days)) / 60.0
                except Exception:
                    continue
        return 1.0

    @staticmethod
    def _source_priority(meta: Dict) -> float:
        """Optional authority boost."""
        priority = meta.get("source_priority")
        try:
            return float(priority)
        except Exception:
            return 0.0

    @staticmethod
    def _filter_low_signal(chunks: List[Dict], min_len: int = 30) -> List[Dict]:
        filtered = []
        for chunk in chunks:
            text = (chunk.get("text") or "").strip()
            if len(text) < min_len:
                continue
            filtered.append(chunk)
        return filtered

    @staticmethod
    def _filter_low_confidence(chunks: List[Dict], min_confidence: Optional[float]) -> List[Dict]:
        if min_confidence is None:
            return chunks
        filtered = []
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            conf = meta.get("ocr_confidence")
            if conf is None:
                filtered.append(chunk)
                continue
            try:
                conf_val = float(conf)
            except Exception:
                filtered.append(chunk)
                continue
            conf_norm = conf_val / 100.0 if conf_val > 1.0 else conf_val
            if conf_norm >= min_confidence:
                filtered.append(chunk)
        return filtered

    @staticmethod
    def _score_chunk_quality(chunk: Dict) -> float:
        base = float(chunk.get("score", 0.0))
        meta = chunk.get("metadata") or {}
        boost = 1.0
        chunk_type = (meta.get("chunk_type") or "").lower()
        if chunk_type in {"section", "table", "table_row", "table_header", "slide"}:
            boost += 0.12
        section_title = (meta.get("section_title") or "").strip().lower()
        if section_title and section_title not in {"document", "introduction"}:
            boost += 0.06
        if meta.get("summary"):
            boost += 0.05
        conf = meta.get("ocr_confidence")
        if conf is not None:
            try:
                conf_val = float(conf)
                conf_norm = conf_val / 100.0 if conf_val > 1.0 else conf_val
                boost *= max(0.6, min(1.15, conf_norm + 0.4))
            except Exception:
                pass
        salience = meta.get("section_salience") or (meta.get("section") or {}).get("salience")
        if salience is not None:
            try:
                salience_val = float(salience)
                boost *= max(0.6, min(1.4, 0.6 + salience_val))
            except Exception:
                pass
        return base * boost

    def _diversify_chunks(
        self,
        chunks: List[Dict],
        max_chunks: int,
        similarity_threshold: float,
    ) -> List[Dict]:
        selected: List[Dict] = []
        ordered = sorted(
            chunks,
            key=lambda c: float(c.get("quality_score", c.get("score", 0.0))),
            reverse=True,
        )
        for chunk in ordered:
            if len(selected) >= max_chunks:
                break
            if not selected:
                selected.append(chunk)
                continue
            if all(
                self._calculate_text_similarity(chunk.get("text", ""), sel.get("text", ""))
                < similarity_threshold
                for sel in selected
            ):
                selected.append(chunk)
        if len(selected) < max_chunks:
            for chunk in ordered:
                if chunk in selected:
                    continue
                selected.append(chunk)
                if len(selected) >= max_chunks:
                    break
        return selected

    def _merge_adjacent_doc_chunks(self, doc_chunks: List[Dict]) -> List[Dict]:
        """Merge back-to-back chunks from the same document to keep context continuous."""
        merged: List[Dict] = []
        for chunk in doc_chunks:
            meta = chunk.get("metadata") or {}
            idx = meta.get("chunk_index")

            if merged:
                prev = merged[-1]
                prev_meta = prev.get("metadata") or {}
                prev_idx = prev_meta.get("chunk_index")

                if prev_idx is not None and idx is not None and idx - prev_idx == 1:
                    # Combine text with a newline boundary
                    prev_text = prev.get("text", "").rstrip()
                    next_text = chunk.get("text", "").lstrip()
                    prev["text"] = (prev_text + "\n" + next_text).strip()

                    # Preserve best score and union retrieval methods
                    prev["score"] = max(prev.get("score", 0), chunk.get("score", 0))
                    methods = set()
                    for candidate in (prev.get("methods"), prev.get("method"), chunk.get("methods"), chunk.get("method")):
                        if not candidate:
                            continue
                        if isinstance(candidate, (list, tuple, set)):
                            methods.update(candidate)
                        else:
                            methods.add(candidate)
                    if methods:
                        prev["methods"] = list(methods)

                    # Track merged spans for traceability
                    span = prev_meta.get("chunk_indices") or []
                    if prev_idx is not None:
                        span.append(prev_idx)
                    if idx is not None:
                        span.append(idx)
                    if span:
                        prev_meta["chunk_indices"] = sorted(set(span))

                    id_span = prev_meta.get("chunk_ids") or []
                    base_id = prev_meta.get("chunk_id")
                    if base_id and base_id not in id_span:
                        id_span.append(base_id)
                    next_id = meta.get("chunk_id")
                    if next_id:
                        id_span.append(next_id)
                    if id_span:
                        # Preserve insertion order while removing duplicates
                        prev_meta["chunk_ids"] = list(dict.fromkeys(id_span))

                    prev["metadata"] = prev_meta
                    continue

            merged.append(chunk)

        return merged

    def _order_chunks_logically(self, chunks: List[Dict]) -> List[Dict]:
        """
        Order chunks to present information logically.
        Prioritizes: relevance > document structure > section coherence
        """
        # Group by document
        doc_groups = self._group_by_document(chunks)

        ordered = []

        # Process each document group
        for doc_id, doc_chunks in doc_groups.items():
            # Sort chunks within document by:
            # 1. Score (primary)
            # 2. Chunk index (secondary, for same document flow)
            doc_chunks_sorted = sorted(
                doc_chunks,
                key=lambda x: (
                    -(
                        float(x.get("quality_score", x.get("score", 0.0)))
                        * self._recency_factor(x.get("metadata", {}))
                        + self._source_priority(x.get("metadata", {}))
                    ),
                    x['metadata'].get('chunk_index', 999)
                )
            )

            # Try to maintain sequential flow if chunks are adjacent
            final_doc_chunks = []
            remaining = doc_chunks_sorted.copy()

            # Start with highest scoring chunk
            if remaining:
                current = remaining.pop(0)
                final_doc_chunks.append(current)
                current_idx = current['metadata'].get('chunk_index', -1)

                # Try to add adjacent chunks
                while remaining:
                    # Look for next or previous chunk
                    found_adjacent = False

                    for i, chunk in enumerate(remaining):
                        chunk_idx = chunk['metadata'].get('chunk_index', -1)

                        if abs(chunk_idx - current_idx) == 1:
                            # Adjacent chunk found
                            if chunk_idx < current_idx:
                                final_doc_chunks.insert(0, chunk)
                            else:
                                final_doc_chunks.append(chunk)

                            remaining.pop(i)
                            current_idx = chunk_idx
                            found_adjacent = True
                            break

                    if not found_adjacent:
                        # No adjacent chunks, add highest scoring remaining
                        if remaining:
                            next_chunk = remaining.pop(0)
                            final_doc_chunks.append(next_chunk)
                            current_idx = next_chunk['metadata'].get('chunk_index', -1)
                        else:
                            break

            # Preserve natural document order for readability
            final_doc_chunks = sorted(
                final_doc_chunks,
                key=lambda x: (
                    x['metadata'].get('chunk_index') is None,
                    x['metadata'].get('chunk_index', 0)
                )
            )
            final_doc_chunks = self._merge_adjacent_doc_chunks(final_doc_chunks)
            ordered.extend(final_doc_chunks)

        return ordered

    def build_context(
            self,
            chunks: List[Dict],
            query: str,
            include_metadata: bool = True
    ) -> Tuple[str, List[Dict]]:
        """
        Build optimized context string from chunks.

        Args:
            chunks: Retrieved chunks with scores and metadata
            query: Original user query
            include_metadata: Whether to include structural metadata

        Returns:
            Tuple of (context_string, sources_list)
        """
        if not chunks:
            return "", []

        # Deduplicate exact chunk ids before text-level dedup
        min_conf = getattr(Config.Retrieval, "MIN_OCR_CONFIDENCE", None)
        min_conf = float(min_conf) / 100.0 if isinstance(min_conf, (int, float)) and min_conf > 1 else min_conf
        extraction_like = self._is_extraction_like_query(query)
        min_len = 1 if extraction_like else 30
        min_chars = int(getattr(Config.Retrieval, "MIN_CHARS", 50))
        min_tokens = int(getattr(Config.Retrieval, "MIN_TOKENS", 10))
        valid_chunks = [
            chunk
            for chunk in chunks
            if is_valid_chunk_text(chunk.get("text"), min_chars=min_chars, min_tokens=min_tokens)
        ]
        if not valid_chunks:
            logger.warning("Context builder: no valid chunks after integrity gate")
            return "", []

        filtered_signal = self._filter_low_signal(valid_chunks, min_len=min_len)
        if not filtered_signal:
            logger.debug("Context filter removed all chunks due to low signal; falling back to valid chunks")
            filtered_signal = list(valid_chunks)

        filtered = self._filter_low_confidence(filtered_signal, min_conf)
        if not filtered:
            logger.debug("Context filter removed all chunks due to low confidence; falling back to valid chunks")
            filtered = list(valid_chunks)
        elif extraction_like and len(filtered) < len(filtered_signal):
            # Avoid dropping short fields (e.g., names/skills) during extraction queries.
            logger.info("Relaxing OCR confidence filter for extraction-style query")
            filtered = filtered_signal
        dedup_by_id = self._deduplicate_by_chunk_id(filtered)

        # Deduplicate similar chunks
        unique_chunks = self._deduplicate_chunks(dedup_by_id)

        for chunk in unique_chunks:
            chunk["quality_score"] = self._score_chunk_quality(chunk)

        # Ensure multi-document coverage by round-robin selection across docs
        def _doc_key(chunk: Dict) -> str:
            meta = chunk.get('metadata', {})
            return meta.get('document_id') or get_source_name(meta) or "unknown"

        buckets: Dict[str, List[Dict]] = {}
        for c in unique_chunks:
            buckets.setdefault(_doc_key(c), []).append(c)

        for doc_chunks in buckets.values():
            doc_chunks.sort(key=lambda x: float(x.get('quality_score', x.get("score", 0.0))), reverse=True)

        selected_chunks: List[Dict] = []

        # Let the top doc contribute a deeper slice to preserve narrative, then
        # round-robin the remainder for coverage.
        if buckets:
            top_doc_id = max(
                buckets.keys(),
                key=lambda k: float(buckets[k][0].get('quality_score', buckets[k][0].get('score', 0.0)))
                if buckets[k] else 0.0
            )
            top_doc = buckets[top_doc_id]
            max_from_top = min(len(top_doc), max(3, self.max_context_chunks // 2))
            selected_chunks.extend(top_doc[:max_from_top])
            buckets[top_doc_id] = top_doc[max_from_top:]
            if not buckets[top_doc_id]:
                buckets.pop(top_doc_id, None)

        while len(selected_chunks) < self.max_context_chunks and buckets:
            # Order docs by best available score to favor stronger evidence
            doc_order = sorted(
                buckets.items(),
                key=lambda item: float(item[1][0].get('quality_score', item[1][0].get('score', 0.0)))
                if item[1] else 0.0,
                reverse=True
            )
            progress = False
            for doc_id, doc_chunks in doc_order:
                if not doc_chunks:
                    continue
                selected_chunks.append(doc_chunks.pop(0))
                progress = True
                if len(selected_chunks) >= self.max_context_chunks:
                    break
            buckets = {k: v for k, v in buckets.items() if v}
            if not progress:
                break

        diversity_threshold = float(getattr(Config.Retrieval, "DIVERSITY_THRESHOLD", 0.6))
        diversified = self._diversify_chunks(selected_chunks, self.max_context_chunks, diversity_threshold)

        # Order logically
        ordered_chunks = self._order_chunks_logically(diversified)

        # Build context string
        context_parts = []
        sources = []

        # Add query restatement for focus
        context_parts.append(f"[QUERY: {query}]\n")

        max_chars = int(getattr(Config.RagV3, "MAX_CONTEXT_CHARS", 6000))
        current_chars = 0
        truncated = False
        for i, chunk in enumerate(ordered_chunks, 1):
            metadata = chunk['metadata']

            # Extract source information
            source_name = get_source_name(metadata) or f"Document {i}"
            section = metadata.get('section_title', metadata.get('section', ''))
            page = metadata.get('page')
            citation = self._format_citation(metadata, fallback_name=source_name)

            # Build source header
            source_header_parts = [f"SOURCE-{i}: {source_name}"]
            if section:
                source_header_parts.append(f"Section: {section}")
            if page:
                source_header_parts.append(f"Page: {page}")

            source_header = " | ".join(source_header_parts)

            # Clean chunk text
            chunk_text = get_content_text(metadata) or chunk['text']

            # Remove context markers if present
            chunk_text = re.sub(r'\[SECTION:.*?\]', '', chunk_text)
            chunk_text = re.sub(r'\[CONTEXT:.*?\]', '', chunk_text)
            chunk_text = re.sub(r'\[CONTINUES:.*?\]', '', chunk_text)
            chunk_text = chunk_text.strip()

            # Add to context with hard character cap
            next_block = f"\n[{source_header}]\n{chunk_text}\n[/SOURCE]\n"
            if max_chars and (current_chars + len(next_block)) > max_chars:
                truncated = True
                break
            context_parts.append(f"\n[{source_header}]")
            context_parts.append(chunk_text)
            context_parts.append("[/SOURCE]\n")
            current_chars += len(next_block)

            # Add to sources list
            sources.append({
                'source_id': i,
                'source_name': source_name,
                'page': page,
                'citation': citation,
            })

        if truncated:
            logger.info("Context exceeded max_chars; running second-pass reduction")
            context_parts = [f"[QUERY: {query}]\n"]
            sources = []
            current_chars = 0
            per_chunk_budget = max(200, max_chars // max(1, len(ordered_chunks)))
            for i, chunk in enumerate(ordered_chunks, 1):
                metadata = chunk['metadata']
                source_name = get_source_name(metadata) or f"Document {i}"
                section = metadata.get('section_title', metadata.get('section', ''))
                page = metadata.get('page')
                citation = self._format_citation(metadata, fallback_name=source_name)

                source_header_parts = [f"SOURCE-{i}: {source_name}"]
                if section:
                    source_header_parts.append(f"Section: {section}")
                if page:
                    source_header_parts.append(f"Page: {page}")
                source_header = " | ".join(source_header_parts)

                chunk_text = get_content_text(metadata) or chunk['text']
                chunk_text = re.sub(r'\[SECTION:.*?\]', '', chunk_text)
                chunk_text = re.sub(r'\[CONTEXT:.*?\]', '', chunk_text)
                chunk_text = re.sub(r'\[CONTINUES:.*?\]', '', chunk_text)
                chunk_text = chunk_text.strip()
                if len(chunk_text) > per_chunk_budget:
                    chunk_text = chunk_text[:per_chunk_budget].rsplit(" ", 1)[0].strip()
                next_block = f"\n[{source_header}]\n{chunk_text}\n[/SOURCE]\n"
                if max_chars and (current_chars + len(next_block)) > max_chars:
                    break
                context_parts.append(f"\n[{source_header}]")
                context_parts.append(chunk_text)
                context_parts.append("[/SOURCE]\n")
                current_chars += len(next_block)
                sources.append({
                    'source_id': i,
                    'source_name': source_name,
                    'page': page,
                    'citation': citation,
                })

        context_string = "\n".join(context_parts)
        if max_chars and len(context_string) > max_chars:
            context_string = context_string[:max_chars]

        logger.info(
            f"Built context: {len(ordered_chunks)} chunks, "
            f"{len(context_string)} chars"
        )

        return context_string, sources

    @staticmethod
    def _format_citation(metadata: Dict[str, Any], *, fallback_name: str) -> str:
        name = get_source_name(metadata) or fallback_name
        name = str(name or "Document").strip()
        page = metadata.get("page") or metadata.get("page_start")
        if page is None:
            page = metadata.get("page_end")
        if page:
            return f"{name}, p. {page}"
        return f"{name}"

class AnswerGenerator:
    """
    Generates accurate answers with proper citations.
    Includes answer verification against sources.
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def build_prompt(
            self,
            query: str,
            context: str,
            persona: str,
            conversation_context: str = "",
            profile_id: Optional[str] = None,
            subscription_id: Optional[str] = None,
            redis_client: Optional[object] = None,
    ) -> str:
        """Build optimized prompt for answer generation"""

        prompt = f"""You are DocWain-Agent. Follow the system instructions exactly and stay grounded in the sources.

SOURCES:
{context}

RECENT CONVERSATION:
{conversation_context if conversation_context else "None"}

USER QUESTION: {query}

Answer requirements:
- Follow the mandatory output shape (Understanding & Scope, Answer, Evidence & Gaps, Optional next-step hint).
- Use domain-specific sections in the Answer.
- Every factual claim must cite [SOURCE-X]; multiple sources -> [SOURCE-1, SOURCE-2].
- If the exact answer is missing, say: Not found in the current profile documents, and list files searched.
- No filler or raw extraction dumps.

Answer:"""

        return inject_persona_prompt(
            prompt,
            persona,
            profile_id=profile_id,
            subscription_id=subscription_id,
            redis_client=redis_client,
        )

    @staticmethod
    def _render_citations(answer: str, sources: List[Dict[str, Any]]) -> str:
        citation_map = {idx + 1: src.get("citation") for idx, src in enumerate(sources)}

        def _replace(match: re.Match) -> str:
            content = match.group(1)
            ids = [int(val) for val in re.findall(r"SOURCE-(\d+)", content)]
            citations = []
            for ident in ids:
                citation = citation_map.get(ident)
                if citation:
                    citations.append(citation)
            if not citations:
                return ""
            unique = list(dict.fromkeys(citations))
            if len(unique) == 1:
                return f"({unique[0]})"
            return "(" + "; ".join(unique) + ")"

        rendered = re.sub(r"\[(SOURCE-[^\]]+)\]", _replace, answer)
        return rendered.strip()

    def _verify_citations(self, answer: str, num_sources: int) -> Tuple[bool, str]:
        """
        Verify that citations in answer are valid.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Extract all citations
        citations = re.findall(r'\[SOURCE-(\d+)\]', answer)

        if not citations:
            return False, "Answer contains no citations"

        # Check if all citations are valid
        invalid_citations = []
        for cite in citations:
            cite_num = int(cite)
            if cite_num < 1 or cite_num > num_sources:
                invalid_citations.append(cite_num)

        if invalid_citations:
            return False, f"Invalid citations: {invalid_citations}"

        return True, ""

    def _verify_factual_grounding(
            self,
            answer: str,
            sources: List[Dict]
    ) -> Tuple[bool, List[str]]:
        """
        Verify that answer statements are grounded in sources.

        Returns:
            Tuple of (is_grounded, list_of_issues)
        """
        issues = []

        # Split answer into sentences
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]

        for sentence in sentences:
            # Skip if sentence has no citations
            if not re.search(r'\[SOURCE-\d+\]', sentence):
                # Check if it's a meta statement
                meta_phrases = [
                    'the documents',
                    'not contain',
                    'information about',
                    'cannot answer',
                    'unclear',
                    'conflicting',
                    'understanding & scope',
                    'evidence & gaps',
                    'answer:',
                ]

                if not any(phrase in sentence.lower() for phrase in meta_phrases):
                    issues.append(f"Sentence without citation: {sentence[:100]}")

        is_grounded = len(issues) == 0
        return is_grounded, issues

    def generate_answer(
            self,
            query: str,
            context: str,
            sources: List[Dict],
            persona: str,
            conversation_context: str = "",
            max_retries: int = 2,
            profile_id: Optional[str] = None,
            subscription_id: Optional[str] = None,
            redis_client: Optional[object] = None,
    ) -> Dict[str, Any]:
        """
        Generate and verify answer.

        Returns:
            Dict with answer, sources, and verification metadata
        """
        prompt = self.build_prompt(
            query,
            context,
            persona,
            conversation_context,
            profile_id=profile_id,
            subscription_id=subscription_id,
            redis_client=redis_client,
        )

        # Generate answer
        try:
            answer = self.llm_client.generate(prompt, max_retries=max_retries)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                'answer': "I apologize, but I encountered an error generating a response.",
                'sources': sources,
                'verified': False,
                'error': str(e)
            }

        # Verify citations against SOURCE tags before rendering to external format
        citations_valid, citation_error = self._verify_citations(answer, len(sources))

        # Verify factual grounding
        is_grounded, grounding_issues = self._verify_factual_grounding(answer, sources)

        verification_status = {
            'citations_valid': citations_valid,
            'citation_error': citation_error if not citations_valid else None,
            'is_grounded': is_grounded,
            'grounding_issues': grounding_issues if not is_grounded else None,
            'overall_verified': citations_valid and is_grounded
        }

        if not verification_status['overall_verified']:
            logger.warning(f"Answer verification failed: {verification_status}")

        answer = self._render_citations(answer, sources)

        return {
            'answer': answer,
            'sources': sources,
            'verification': verification_status,
            'verified': verification_status['overall_verified']
        }

# Integration function to replace existing answer_question logic
def generate_accurate_answer(
        query: str,
        retrieved_chunks: List[Dict],
        llm_client,
        persona: str,
        conversation_context: str = ""
) -> Dict[str, Any]:
    """
    Main function to generate accurate, verified answers.

    Args:
        query: User question
        retrieved_chunks: List of chunks from adaptive retrieval
        llm_client: LLM client (Ollama or Gemini)
        persona: Assistant persona
        conversation_context: Recent conversation history

    Returns:
        Dict with answer, sources, and verification info
    """
    # Build context
    context_builder = IntelligentContextBuilder(max_context_chunks=16)
    context, sources = context_builder.build_context(
        chunks=retrieved_chunks,
        query=query,
        include_metadata=True
    )

    if not context:
        return {
            'answer': "I couldn't find relevant evidence in the provided documents. Exact context not found in the documents.",
            'sources': [],
            'verified': True  # Absence acknowledgment is valid
        }

    # Generate answer
    answer_generator = AnswerGenerator(llm_client)
    result = answer_generator.generate_answer(
        query=query,
        context=context,
        sources=sources,
        persona=persona,
        conversation_context=conversation_context
    )

    return result
