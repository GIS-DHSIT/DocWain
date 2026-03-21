DOCWAIN_PERSONA = (
    "You are DocWain — Document Wise AI Node. You are an expert document intelligence assistant "
    "that helps users understand complex documents. You adapt your tone to match the user's style. "
    "You cite evidence from documents, reason across sections, handle tables and layouts expertly, "
    "and clearly communicate uncertainty when information is incomplete."
)

CATEGORY_CONTEXT = {
    "table_extraction": "The user is asking about data in a table. Demonstrate expert table reasoning.",
    "layout_parsing": "The user is asking about document structure. Show understanding of headings, sections, lists.",
    "cross_reference": "The user is asking about connections between different parts. Link related information across sections.",
    "section_hierarchy": "The user is asking about document organization. Show parent-child section relationships.",
    "multi_page_reasoning": "The user needs information synthesized across multiple pages. Follow arguments and connect evidence.",
    "uncertainty_handling": "The query may touch on incomplete information. Demonstrate honest uncertainty while being helpful.",
    "adaptive_tone": "Match the tone and detail level of your response to the user's query style.",
    "feedback": "A user previously found this type of response unhelpful. Generate an improved version.",
}


def build_sft_prompt(query: str, category: str, subcategory: str) -> str:
    context = CATEGORY_CONTEXT.get(subcategory, "Respond helpfully and accurately.")
    return f"""You are generating training data for DocWain, a document intelligence AI assistant.

DocWain's persona: {DOCWAIN_PERSONA}

Scenario category: {category} / {subcategory}
Context: {context}

User query: "{query}"

Generate the IDEAL DocWain response for this query. Requirements:
1. Respond as DocWain would
2. Focus on PATTERNS of document understanding, NOT specific document content
3. Show HOW to reason about this type of query
4. Use markdown formatting where appropriate
5. If the query is short/casual, keep the response proportionally concise
6. If the query is detailed, provide thorough analysis

Write ONLY the ideal response, nothing else."""
