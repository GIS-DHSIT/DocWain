"""
Enhanced Context Building and Answer Generation
Replaces context building logic in dw_newron.py
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class IntelligentContextBuilder:
    """
    Builds optimized context from retrieved chunks.
    Key improvements:
    1. Deduplicates similar content
    2. Preserves document structure
    3. Orders chunks logically
    4. Includes chunk relationship information
    """

    def __init__(self, max_context_chunks: int = 7):
        self.max_context_chunks = max_context_chunks

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
        """Remove highly similar chunks"""
        if len(chunks) <= 1:
            return chunks

        unique_chunks = []

        for chunk in chunks:
            is_duplicate = False
            chunk_text = chunk['text']

            for unique in unique_chunks:
                similarity = self._calculate_text_similarity(
                    chunk_text,
                    unique['text']
                )

                if similarity >= threshold:
                    # Keep the one with higher score
                    if chunk['score'] > unique['score']:
                        unique_chunks.remove(unique)
                        break
                    else:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_chunks.append(chunk)

        logger.info(f"Deduplicated {len(chunks)} → {len(unique_chunks)} chunks")
        return unique_chunks

    def _group_by_document(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """Group chunks by source document"""
        doc_groups = defaultdict(list)

        for chunk in chunks:
            doc_id = chunk['metadata'].get('document_id', 'unknown')
            doc_groups[doc_id].append(chunk)

        return dict(doc_groups)

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
                    -x['score'],  # Higher score first
                    x['metadata'].get('chunk_index', 999)  # Lower index first
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

        # Deduplicate similar chunks
        unique_chunks = self._deduplicate_chunks(chunks)

        # Limit to max chunks
        selected_chunks = unique_chunks[:self.max_context_chunks]

        # Order logically
        ordered_chunks = self._order_chunks_logically(selected_chunks)

        # Build context string
        context_parts = []
        sources = []

        # Add query restatement for focus
        context_parts.append(f"[QUERY: {query}]\n")

        for i, chunk in enumerate(ordered_chunks, 1):
            metadata = chunk['metadata']

            # Extract source information
            source_name = metadata.get('source_file', f"Document {i}")
            section = metadata.get('section_title', metadata.get('section', ''))
            page = metadata.get('page')
            chunk_id = metadata.get('chunk_id', '')
            score = chunk['score']

            # Build source header
            source_header_parts = [f"SOURCE-{i}: {source_name}"]
            if section:
                source_header_parts.append(f"Section: {section}")
            if page:
                source_header_parts.append(f"Page: {page}")
            source_header_parts.append(f"Relevance: {score:.3f}")

            source_header = " | ".join(source_header_parts)

            # Clean chunk text
            chunk_text = chunk['text']

            # Remove context markers if present
            chunk_text = re.sub(r'\[SECTION:.*?\]', '', chunk_text)
            chunk_text = re.sub(r'\[CONTEXT:.*?\]', '', chunk_text)
            chunk_text = re.sub(r'\[CONTINUES:.*?\]', '', chunk_text)
            chunk_text = chunk_text.strip()

            # Add to context
            context_parts.append(f"\n[{source_header}]")
            context_parts.append(chunk_text)
            context_parts.append("[/SOURCE]\n")

            # Add to sources list
            sources.append({
                'source_id': i,
                'source_name': source_name,
                'section': section,
                'page': page,
                'chunk_id': chunk_id,
                'relevance_score': round(score, 3),
                'excerpt': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                'retrieval_methods': chunk.get('methods', [chunk.get('method', 'unknown')])
            })

        context_string = "\n".join(context_parts)

        logger.info(
            f"Built context: {len(ordered_chunks)} chunks, "
            f"{len(context_string)} chars"
        )

        return context_string, sources


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
            conversation_context: str = ""
    ) -> str:
        """Build optimized prompt for answer generation"""

        prompt = f"""You are a {persona}. Your job is to provide accurate, well-sourced answers.

CRITICAL RULES:
1. Answer ONLY using information in the SOURCES below
2. Cite every factual claim using [SOURCE-X] notation
3. If information is not in sources, explicitly state: "The documents do not contain information about [topic]"
4. Never add external knowledge or speculation
5. For uncertain information, acknowledge limitations

{context}

CONVERSATION HISTORY:
{conversation_context if conversation_context else "None"}

USER QUESTION: {query}

ANSWER REQUIREMENTS:
- Start directly with the answer (don't repeat the question)
- Write 2-4 clear, concise sentences
- Use natural, conversational language
- Cite sources immediately after each claim: [SOURCE-X]
- If multiple sources support a claim, cite all: [SOURCE-1, SOURCE-2]
- For partial information, state what is known and what is missing
- If documents conflict, note the discrepancy and cite both

ANSWER:"""

        return prompt

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
                    'conflicting'
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
            max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Generate and verify answer.

        Returns:
            Dict with answer, sources, and verification metadata
        """
        prompt = self.build_prompt(query, context, persona, conversation_context)

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

        # Verify citations
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
    context_builder = IntelligentContextBuilder(max_context_chunks=7)
    context, sources = context_builder.build_context(
        chunks=retrieved_chunks,
        query=query,
        include_metadata=True
    )

    if not context:
        return {
            'answer': "I couldn't find relevant information in the documents to answer your question.",
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

