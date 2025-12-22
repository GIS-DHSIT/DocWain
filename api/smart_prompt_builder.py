"""
Smart Prompt Builder - FIXED FOR MULTI-DOCUMENT ACCURACY
Key Fixes:
1. Stronger document boundary enforcement
2. Explicit "answer only if found" instructions
3. Document mismatch detection
4. Answer verification prompt
"""

import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SmartPromptBuilder:
    """Builds prompts that enforce strict document grounding"""

    @staticmethod
    def extract_document_names_from_query(query: str, available_docs: List[str]) -> List[str]:
        """Extract which documents the user is asking about"""
        query_lower = query.lower()
        matched_docs = []

        for doc in available_docs:
            doc_lower = doc.lower()
            doc_parts = doc_lower.replace('.pdf', '').replace('.docx', '').replace('_', ' ').split()

            for part in doc_parts:
                if len(part) > 3 and part in query_lower:
                    matched_docs.append(doc)
                    break

        return matched_docs

    @staticmethod
    def build_source_map(chunks: List[Dict], max_sources: int = 7) -> str:
        """
        FIXED: Show document distribution clearly
        """
        if not chunks:
            return ""

        # Group by document
        doc_groups = {}
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            source = metadata.get('source_file', 'Unknown')
            doc_id = metadata.get('document_id', 'unknown')

            if source not in doc_groups:
                doc_groups[source] = {
                    'chunks': [],
                    'doc_id': doc_id,
                    'max_score': 0
                }

            doc_groups[source]['chunks'].append(chunk)
            doc_groups[source]['max_score'] = max(
                doc_groups[source]['max_score'],
                chunk.get('score', 0)
            )

        # Sort by max score
        sorted_docs = sorted(
            doc_groups.items(),
            key=lambda x: x[1]['max_score'],
            reverse=True
        )

        lines = ["=" * 80]
        lines.append("AVAILABLE SOURCES (ranked by relevance):")
        lines.append("=" * 80)

        for i, (source, info) in enumerate(sorted_docs[:max_sources], 1):
            chunk_count = len(info['chunks'])
            max_score = info['max_score']
            doc_id = info['doc_id']

            lines.append(
                f"[SOURCE-{i}] {source}\n"
                f"  Document ID: {doc_id[:12]}...\n"
                f"  Chunks available: {chunk_count}\n"
                f"  Best relevance: {max_score:.3f}\n"
            )

        lines.append("=" * 80)
        return "\n".join(lines)

    @staticmethod
    def build_context_with_document_boundaries(chunks: List[Dict], max_chunks: int = 7) -> str:
        """
        FIXED: Enforce strict document boundaries in context
        """
        if not chunks:
            return ""

        context_parts = []
        current_doc_id = None

        for i, chunk in enumerate(chunks[:max_chunks], 1):
            metadata = chunk.get('metadata', {})
            source = metadata.get('source_file', f'Document-{i}')
            doc_id = metadata.get('document_id', 'unknown')
            section = metadata.get('section_title', '')
            text = chunk.get('text', '')

            # FIXED: Clear document boundaries
            if doc_id != current_doc_id:
                if current_doc_id is not None:
                    context_parts.append(f"\n{'=' * 80}")
                    context_parts.append("END OF DOCUMENT")
                    context_parts.append(f"{'=' * 80}\n\n")

                context_parts.append(f"\n{'=' * 80}")
                context_parts.append(f"START OF NEW DOCUMENT: {source}")
                context_parts.append(f"Document ID: {doc_id}")
                context_parts.append(f"{'=' * 80}\n")
                current_doc_id = doc_id

            # Chunk header
            header = f"[SOURCE-{i}]"
            if section:
                header += f" | Section: {section}"
            context_parts.append(f"\n{header}")
            context_parts.append("-" * 80)
            context_parts.append(text)
            context_parts.append("-" * 80)

        # End final document
        if current_doc_id is not None:
            context_parts.append(f"\n{'=' * 80}")
            context_parts.append("END OF DOCUMENT")
            context_parts.append(f"{'=' * 80}\n")

        return "\n".join(context_parts)

    @staticmethod
    def build_strict_qa_prompt(
            query: str,
            chunks: List[Dict],
            persona: str = "document analysis assistant",
            conversation_context: str = ""
    ) -> str:
        """
        FIXED: Ultra-strict prompt to prevent hallucinations
        """

        # Build components
        source_map = SmartPromptBuilder.build_source_map(chunks)
        context = SmartPromptBuilder.build_context_with_document_boundaries(chunks)

        # Get unique documents
        unique_docs = list(set([
            c.get('metadata', {}).get('source_file', 'Unknown')
            for c in chunks
        ]))

        # FIXED: More aggressive prompt
        prompt = f"""You are a {persona}. Your ONLY job is to answer questions using information that EXISTS in the documents below.

{source_map}

 CRITICAL GROUNDING RULES - VIOLATION = FAILURE 

1. **READ THE QUESTION CAREFULLY**
   - Identify exactly what entity/document/person is being asked about
   - Check if that entity EXISTS in the available sources

2. **VERIFY DOCUMENT MATCH**
   - If question asks about "Person X" but sources contain "Person Y"
   - YOU MUST IMMEDIATELY STATE: "The provided documents are about [Person Y], not [Person X]. I cannot answer questions about [Person X] using these sources."
   - DO NOT try to answer anyway

3. **ANSWER ONLY IF INFORMATION EXISTS**
   - Every sentence MUST be directly supported by the sources
   - If information is not in sources, state: "This information is not present in the provided documents."
   - NEVER fill gaps with general knowledge or assumptions

4. **MANDATORY CITATIONS**
   - Every factual claim MUST have [SOURCE-X] citation
   - Format: "According to [SOURCE-1], the city is Mumbai."
   - Multiple sources: "[SOURCE-1, SOURCE-3]"

5. **HANDLE CONFLICTS**
   - If sources contradict, cite both: "According to [SOURCE-1]... however [SOURCE-2] states..."

6. **MULTI-DOCUMENT SCENARIOS**
   - If multiple documents are present, specify which document you're referring to
   - Example: "In the resume for John Doe [SOURCE-1], the location is listed as..."

Available documents: {', '.join(unique_docs)}

DOCUMENT CONTENT:
{context}

{"CONVERSATION HISTORY:\n" + conversation_context + "\n" if conversation_context else ""}

USER QUESTION: {query}

BEFORE ANSWERING, ASK YOURSELF:
- Is the entity/person mentioned in the question actually in these documents?
- Can I answer this question using ONLY the text above?
- Have I cited every factual claim?

ANSWER REQUIREMENTS:
- First, verify you're answering about the correct entity/document
- Write 2-4 clear, factual sentences
- Cite sources [SOURCE-X] after EVERY claim
- If wrong documents provided, explicitly state this
- Use natural but precise language
- NEVER speculate or add external knowledge

ANSWER:"""

        return prompt

    @staticmethod
    def build_verification_prompt(answer: str, chunks: List[Dict], query: str) -> str:
        """
        FIXED: Verification prompt to catch hallucinations
        """

        # Extract unique documents from chunks
        unique_docs = list(set([
            c.get('metadata', {}).get('source_file', 'Unknown')
            for c in chunks
        ]))

        # Build mini context for verification
        context_snippets = []
        for i, chunk in enumerate(chunks[:5], 1):
            source = chunk.get('metadata', {}).get('source_file', f'Doc-{i}')
            text_snippet = chunk.get('text', '')[:300]
            context_snippets.append(f"[SOURCE-{i}] {source}:\n{text_snippet}...")

        context_text = "\n\n".join(context_snippets)

        prompt = f"""You are a fact-checker. Verify if this answer is ACCURATELY grounded in the provided documents.

QUERY: {query}

AVAILABLE DOCUMENTS: {', '.join(unique_docs)}

DOCUMENT EXCERPTS:
{context_text}

ANSWER TO VERIFY:
{answer}

CHECK THE FOLLOWING:
1. Does the answer address the correct entity/document mentioned in the query?
2. Is every factual claim supported by the document excerpts?
3. Are citations present and correct?
4. Does the answer acknowledge if information is missing?
5. Are there any claims that could not have come from these documents?

Respond in this format:
VERIFIED: [Yes/No]
CONFIDENCE: [High/Medium/Low]
ISSUES: [List any problems, or "None"]
ENTITY_MATCH: [Yes/No - Does answer match the entity asked about?]
HALLUCINATIONS: [List any facts not in documents, or "None"]

Your verification:"""

        return prompt


class DocumentMatcher:
    """Helps match queries to relevant documents"""

    @staticmethod
    def analyze_query_intent(query: str) -> Dict[str, Any]:
        """Extract what the user is asking about"""
        query_lower = query.lower()

        # Extract person names (capitalized words)
        potential_names = re.findall(
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z]\.?)?',
            query
        )

        # Extract company names (common patterns)
        potential_companies = re.findall(
            r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\s+(?:Inc|Corp|LLC|Ltd|Limited|Company)\b',
            query
        )

        # Extract specific query types
        location_keywords = ['city', 'location', 'address', 'lives', 'belongs', 'from', 'based in', 'where']
        has_location_query = any(kw in query_lower for kw in location_keywords)

        summary_keywords = ['summary', 'profile', 'about', 'who is', 'background', 'experience', 'describe']
        has_summary_query = any(kw in query_lower for kw in summary_keywords)

        comparison_keywords = ['compare', 'difference', 'versus', 'vs', 'which', 'better']
        has_comparison_query = any(kw in query_lower for kw in comparison_keywords)

        return {
            'potential_names': potential_names,
            'potential_companies': potential_companies,
            'has_location_query': has_location_query,
            'has_summary_query': has_summary_query,
            'has_comparison_query': has_comparison_query,
            'query_type': (
                'comparison' if has_comparison_query else
                'location' if has_location_query else
                'summary' if has_summary_query else
                'general'
            )
        }

    @staticmethod
    def suggest_document_filter(query: str, available_sources: List[str]) -> Optional[List[str]]:
        """
        FIXED: More intelligent document matching
        """
        intent = DocumentMatcher.analyze_query_intent(query)

        # If no specific names/companies, return all docs (multi-doc query)
        if not intent['potential_names'] and not intent['potential_companies']:
            if intent['has_comparison_query']:
                # Comparison queries need multiple docs
                return None  # Return all
            return None

        # Try to match names/companies to documents
        matched_docs = []
        search_terms = intent['potential_names'] + intent['potential_companies']

        for term in search_terms:
            term_parts = term.lower().split()
            for doc in available_sources:
                doc_lower = doc.lower()
                # Match if any significant part appears
                if any(part in doc_lower for part in term_parts if len(part) > 2):
                    if doc not in matched_docs:
                        matched_docs.append(doc)

        # Log matching
        if matched_docs:
            logger.info(f"Matched query terms {search_terms} to documents: {matched_docs}")
        else:
            logger.info(f"No specific document match for terms: {search_terms}")

        return matched_docs if matched_docs else None


def build_enhanced_answer_with_verification(
        query: str,
        chunks: List[Dict],
        llm_client,
        persona: str = "document analysis assistant",
        conversation_context: str = ""
) -> Dict[str, Any]:
    """
    FIXED: Answer generation with verification
    """

    # Analyze query
    intent = DocumentMatcher.analyze_query_intent(query)
    logger.info(f"Query intent: {intent}")

    # Get retrieved documents
    retrieved_sources = list(set([
        c.get('metadata', {}).get('source_file', 'Unknown')
        for c in chunks
    ]))
    logger.info(f"Retrieved sources: {retrieved_sources}")

    # Build strict prompt
    prompt_builder = SmartPromptBuilder()
    prompt = prompt_builder.build_strict_qa_prompt(
        query, chunks, persona, conversation_context
    )

    # Generate answer
    try:
        answer = llm_client.generate(prompt, max_retries=2)
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return {
            'answer': "I apologize, but I encountered an error generating a response.",
            'sources': [],
            'verified': False,
            'error': str(e)
        }

    # ADDED: Verification step
    verification_result = None
    if answer and '[SOURCE-' in answer:
        try:
            verification_prompt = prompt_builder.build_verification_prompt(
                answer, chunks, query
            )
            verification_result = llm_client.generate(verification_prompt, max_retries=1)
            logger.info(f"Verification result: {verification_result[:200]}")
        except Exception as e:
            logger.warning(f"Verification failed: {e}")

    # Extract sources
    sources = []
    seen_sources = set()

    for i, chunk in enumerate(chunks[:7], 1):
        metadata = chunk.get('metadata', {})
        source_name = metadata.get('source_file', f'Document {i}')

        if source_name not in seen_sources:
            sources.append({
                'source_id': i,
                'source_name': source_name,
                'section': metadata.get('section_title', ''),
                'page': metadata.get('page'),
                'document_id': metadata.get('document_id', ''),
                'relevance_score': round(chunk.get('score', 0), 3),
                'excerpt': chunk.get('text', '')[:200] + "...",
            })
            seen_sources.add(source_name)

    # Check for warnings
    has_citations = '[SOURCE-' in answer
    mentions_wrong_doc = any(
        phrase in answer.lower()
        for phrase in [
            'not in the documents',
            'documents are about',
            'cannot answer about',
            'not present in',
            'information is not available'
        ]
    )

    # Parse verification
    verified = has_citations
    if verification_result:
        verified = 'VERIFIED: Yes' in verification_result or 'VERIFIED: High' in verification_result

    return {
        'answer': answer,
        'sources': sources,
        'verified': verified,
        'query_intent': intent,
        'retrieved_sources': retrieved_sources,
        'document_match_warning': mentions_wrong_doc,
        'verification_result': verification_result
    }