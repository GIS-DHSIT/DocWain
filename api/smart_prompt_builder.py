"""
Smart Prompt Builder with Document-Aware Context
Ensures answers are grounded in the CORRECT documents
"""

import logging
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

            # Check if any significant part of doc name is in query
            for part in doc_parts:
                if len(part) > 3 and part in query_lower:
                    matched_docs.append(doc)
                    break

        return matched_docs

    @staticmethod
    def build_source_map(chunks: List[Dict], max_sources: int = 5) -> str:
        """Create clear source mapping"""
        if not chunks:
            return ""

        lines = ["AVAILABLE SOURCES (ranked by relevance):"]
        seen_sources = set()
        source_num = 1

        for chunk in chunks[:max_sources]:
            metadata = chunk.get('metadata', {})
            source = metadata.get('source_file', 'Unknown')

            if source not in seen_sources:
                score = chunk.get('score', 0)
                section = metadata.get('section_title', '')
                doc_id = metadata.get('document_id', '')

                parts = [f"[SOURCE-{source_num}] {source}"]
                parts.append(f"relevance={score:.3f}")
                if section:
                    parts.append(f"section=\"{section}\"")
                if doc_id:
                    parts.append(f"doc_id={doc_id[:12]}")

                lines.append(" | ".join(parts))
                seen_sources.add(source)
                source_num += 1

        return "\n".join(lines)

    @staticmethod
    def build_context_with_clear_boundaries(chunks: List[Dict], max_chunks: int = 5) -> str:
        """Build context with very clear document boundaries"""
        if not chunks:
            return ""

        context_parts = []

        for i, chunk in enumerate(chunks[:max_chunks], 1):
            metadata = chunk.get('metadata', {})
            source = metadata.get('source_file', f'Document-{i}')
            section = metadata.get('section_title', '')
            text = chunk.get('text', '')

            # Clear boundary markers
            header = f"\n{'=' * 80}\nSOURCE-{i}: {source}"
            if section:
                header += f" | Section: {section}"
            header += f"\n{'=' * 80}"

            context_parts.append(header)
            context_parts.append(text)
            context_parts.append(f"{'=' * 80}\n")

        return "\n".join(context_parts)

    @staticmethod
    def build_strict_qa_prompt(
            query: str,
            chunks: List[Dict],
            persona: str = "document analysis assistant",
            conversation_context: str = ""
    ) -> str:
        """Build prompt that enforces strict grounding"""

        # Build components
        source_map = SmartPromptBuilder.build_source_map(chunks)
        context = SmartPromptBuilder.build_context_with_clear_boundaries(chunks)

        # Get unique source files
        source_files = list(set([
            c.get('metadata', {}).get('source_file', 'Unknown')
            for c in chunks[:10]
        ]))

        prompt = f"""You are a {persona}. Your ONLY job is to answer questions using EXCLUSIVELY the information in the sources below.

{source_map}

CRITICAL GROUNDING RULES:
1. READ THE QUESTION CAREFULLY - identify what document/person/entity is being asked about
2. CHECK SOURCES - confirm the relevant document is actually present in the sources
3. If the question asks about "Person X" but the sources contain "Person Y", YOU MUST state: "The documents provided are about [Person Y], not [Person X]. I cannot answer about [Person X] from these sources."
4. EVERY factual claim MUST have [SOURCE-X] citation immediately after it
5. NEVER add information not explicitly in the sources
6. If information is missing, state: "The documents do not contain information about [specific topic]"
7. If documents conflict, cite both: "According to [SOURCE-1]... but [SOURCE-2] states..."

DOCUMENT CONTENT:
{context}

{"PREVIOUS CONVERSATION:\n" + conversation_context + "\n" if conversation_context else ""}

USER QUESTION: {query}

ANSWER REQUIREMENTS:
- First, confirm you're answering about the RIGHT documents/entities mentioned in the question
- Write 2-4 clear sentences in natural language
- Cite sources using [SOURCE-X] after each claim
- If the wrong documents were retrieved, explicitly state this
- Be conversational but precise

ANSWER:"""

        return prompt

    @staticmethod
    def build_verification_prompt(answer: str, context: str, query: str) -> str:
        """Build prompt to verify answer accuracy"""

        prompt = f"""You are a fact-checker. Verify if this answer is grounded in the provided documents.

DOCUMENTS:
{context}

QUERY: {query}

ANSWER TO VERIFY:
{answer}

Check:
1. Are all factual claims supported by the documents?
2. Are citations correct?
3. Does the answer address the right entity/document?

Respond with ONLY:
VERIFIED: Yes/No
ISSUES: [list any problems]
"""
        return prompt


class DocumentMatcher:
    """Helps match queries to relevant documents"""

    @staticmethod
    def analyze_query_intent(query: str) -> Dict[str, Any]:
        """Extract what the user is asking about"""
        query_lower = query.lower()

        # Extract person names (capitalized words)
        import re
        potential_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z]\.?[A-Z]?\.?[A-Z]?\.?)?', query)

        # Extract location indicators
        location_keywords = ['city', 'location', 'address', 'lives', 'belongs', 'from', 'based in']
        has_location_query = any(kw in query_lower for kw in location_keywords)

        # Extract summary/profile indicators
        summary_keywords = ['summary', 'profile', 'about', 'who is', 'background', 'experience']
        has_summary_query = any(kw in query_lower for kw in summary_keywords)

        # Extract document indicators
        doc_indicators = re.findall(r'\b(?:resume|cv|profile|document|file|paper)\b', query_lower)

        return {
            'potential_names': potential_names,
            'has_location_query': has_location_query,
            'has_summary_query': has_summary_query,
            'document_indicators': doc_indicators,
            'query_type': 'location' if has_location_query else 'summary' if has_summary_query else 'general'
        }

    @staticmethod
    def suggest_document_filter(query: str, available_sources: List[str]) -> Optional[List[str]]:
        """Suggest which documents to filter retrieval by"""
        intent = DocumentMatcher.analyze_query_intent(query)

        if not intent['potential_names'] or not available_sources:
            return None

        # Try to match names to document names
        matched_docs = []
        for name in intent['potential_names']:
            name_parts = name.lower().split()
            for doc in available_sources:
                doc_lower = doc.lower()
                if any(part in doc_lower for part in name_parts if len(part) > 2):
                    if doc not in matched_docs:
                        matched_docs.append(doc)

        return matched_docs if matched_docs else None


def build_enhanced_answer_with_verification(
        query: str,
        chunks: List[Dict],
        llm_client,
        persona: str = "document analysis assistant",
        conversation_context: str = ""
) -> Dict[str, Any]:
    """
    Build answer with document matching and verification
    """

    # Analyze query intent
    intent = DocumentMatcher.analyze_query_intent(query)
    logger.info(f"Query intent: {intent}")

    # Check if retrieved documents match query intent
    retrieved_sources = list(set([
        c.get('metadata', {}).get('source_file', 'Unknown')
        for c in chunks[:5]
    ]))
    logger.info(f"Retrieved sources: {retrieved_sources}")

    # Build prompt
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

    # Extract sources
    sources = []
    for i, chunk in enumerate(chunks[:7], 1):
        metadata = chunk.get('metadata', {})
        sources.append({
            'source_id': i,
            'source_name': metadata.get('source_file', f'Document {i}'),
            'section': metadata.get('section_title', ''),
            'page': metadata.get('page'),
            'relevance_score': round(chunk.get('score', 0), 3),
            'excerpt': chunk.get('text', '')[:200] + "...",
            'document_id': metadata.get('document_id', ''),
        })

    # Basic verification
    has_citations = '[SOURCE-' in answer
    mentions_wrong_doc = any(
        phrase in answer.lower()
        for phrase in ['not in the documents', 'documents are about', 'cannot answer about']
    )

    return {
        'answer': answer,
        'sources': sources,
        'verified': has_citations,
        'query_intent': intent,
        'retrieved_sources': retrieved_sources,
        'document_match_warning': mentions_wrong_doc
    }



# ==================================================
# for testing enhanced prompt

# """
# Smart Prompt Builder - FIXED FOR MULTI-DOCUMENT ACCURACY
# Key Fixes:
# 1. Stronger document boundary enforcement
# 2. Explicit "answer only if found" instructions
# 3. Document mismatch detection
# 4. Answer verification prompt
# """
#
# import logging
# import re
# from typing import List, Dict, Any, Optional
#
# logger = logging.getLogger(__name__)
#
#
# class SmartPromptBuilder:
#     """Builds prompts that enforce strict document grounding"""
#
#     @staticmethod
#     def extract_document_names_from_query(query: str, available_docs: List[str]) -> List[str]:
#         """Extract which documents the user is asking about"""
#         query_lower = query.lower()
#         matched_docs = []
#
#         for doc in available_docs:
#             doc_lower = doc.lower()
#             doc_parts = doc_lower.replace('.pdf', '').replace('.docx', '').replace('_', ' ').split()
#
#             for part in doc_parts:
#                 if len(part) > 3 and part in query_lower:
#                     matched_docs.append(doc)
#                     break
#
#         return matched_docs
#
#     @staticmethod
#     def build_source_map(chunks: List[Dict], max_sources: int = 7) -> str:
#         """
#         FIXED: Show document distribution clearly
#         """
#         if not chunks:
#             return ""
#
#         # Group by document
#         doc_groups = {}
#         for chunk in chunks:
#             metadata = chunk.get('metadata', {})
#             source = metadata.get('source_file', 'Unknown')
#             doc_id = metadata.get('document_id', 'unknown')
#
#             if source not in doc_groups:
#                 doc_groups[source] = {
#                     'chunks': [],
#                     'doc_id': doc_id,
#                     'max_score': 0
#                 }
#
#             doc_groups[source]['chunks'].append(chunk)
#             doc_groups[source]['max_score'] = max(
#                 doc_groups[source]['max_score'],
#                 chunk.get('score', 0)
#             )
#
#         # Sort by max score
#         sorted_docs = sorted(
#             doc_groups.items(),
#             key=lambda x: x[1]['max_score'],
#             reverse=True
#         )
#
#         lines = ["=" * 80]
#         lines.append("AVAILABLE SOURCES (ranked by relevance):")
#         lines.append("=" * 80)
#
#         for i, (source, info) in enumerate(sorted_docs[:max_sources], 1):
#             chunk_count = len(info['chunks'])
#             max_score = info['max_score']
#             doc_id = info['doc_id']
#
#             lines.append(
#                 f"[SOURCE-{i}] {source}\n"
#                 f"  Document ID: {doc_id[:12]}...\n"
#                 f"  Chunks available: {chunk_count}\n"
#                 f"  Best relevance: {max_score:.3f}\n"
#             )
#
#         lines.append("=" * 80)
#         return "\n".join(lines)
#
#     @staticmethod
#     def build_context_with_document_boundaries(chunks: List[Dict], max_chunks: int = 7) -> str:
#         """
#         FIXED: Enforce strict document boundaries in context
#         """
#         if not chunks:
#             return ""
#
#         context_parts = []
#         current_doc_id = None
#
#         for i, chunk in enumerate(chunks[:max_chunks], 1):
#             metadata = chunk.get('metadata', {})
#             source = metadata.get('source_file', f'Document-{i}')
#             doc_id = metadata.get('document_id', 'unknown')
#             section = metadata.get('section_title', '')
#             text = chunk.get('text', '')
#
#             # FIXED: Clear document boundaries
#             if doc_id != current_doc_id:
#                 if current_doc_id is not None:
#                     context_parts.append(f"\n{'=' * 80}")
#                     context_parts.append("END OF DOCUMENT")
#                     context_parts.append(f"{'=' * 80}\n\n")
#
#                 context_parts.append(f"\n{'=' * 80}")
#                 context_parts.append(f"START OF NEW DOCUMENT: {source}")
#                 context_parts.append(f"Document ID: {doc_id}")
#                 context_parts.append(f"{'=' * 80}\n")
#                 current_doc_id = doc_id
#
#             # Chunk header
#             header = f"[SOURCE-{i}]"
#             if section:
#                 header += f" | Section: {section}"
#             context_parts.append(f"\n{header}")
#             context_parts.append("-" * 80)
#             context_parts.append(text)
#             context_parts.append("-" * 80)
#
#         # End final document
#         if current_doc_id is not None:
#             context_parts.append(f"\n{'=' * 80}")
#             context_parts.append("END OF DOCUMENT")
#             context_parts.append(f"{'=' * 80}\n")
#
#         return "\n".join(context_parts)
#
#     @staticmethod
#     def build_strict_qa_prompt(
#             query: str,
#             chunks: List[Dict],
#             persona: str = "document analysis assistant",
#             conversation_context: str = ""
#     ) -> str:
#         """
#         FIXED: Ultra-strict prompt to prevent hallucinations
#         """
#
#         # Build components
#         source_map = SmartPromptBuilder.build_source_map(chunks)
#         context = SmartPromptBuilder.build_context_with_document_boundaries(chunks)
#
#         # Get unique documents
#         unique_docs = list(set([
#             c.get('metadata', {}).get('source_file', 'Unknown')
#             for c in chunks
#         ]))
#
#         # FIXED: More aggressive prompt
#         prompt = f"""You are a {persona}. Your ONLY job is to answer questions using information that EXISTS in the documents below.
#
# {source_map}
#
# CRITICAL GROUNDING RULES - VIOLATION = FAILURE
#
# 1. **READ THE QUESTION CAREFULLY**
#     -Identify exactly what entity/



