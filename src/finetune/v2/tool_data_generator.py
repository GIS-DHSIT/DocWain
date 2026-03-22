"""Synthetic training-data generator for DocWain V2 tool-calling fine-tuning.

Generates four categories of examples:
  1. Single tool calls (100+)  -- one tool per query, covering all 9 tools
  2. Parallel tool calls (50+) -- 2+ tools called simultaneously
  3. No-tool-needed (50+)      -- greetings, meta questions, simple answers
  4. Auto-invocation (50+)     -- pre-filled tool_response in context

Each example is produced in the chat-message format consumed by the V2 SFT
trainer, using helpers from ``dataset_preprocess`` and ``tool_schemas``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from src.finetune.v2.dataset_preprocess import (
    format_no_tool_sft,
    format_tool_call_sft,
)
from src.finetune.v2.tool_schemas import (
    format_tools_for_prompt,
    get_core_tool_schemas,
)

# ---------------------------------------------------------------------------
# Randomisation seed (reproducible but overridable)
# ---------------------------------------------------------------------------

_RNG = random.Random(42)

# ---------------------------------------------------------------------------
# Template pools for random fill
# ---------------------------------------------------------------------------

_TOPICS = [
    "revenue", "compliance", "safety procedures", "quarterly earnings",
    "employee benefits", "supply chain", "risk management", "patent claims",
    "customer feedback", "product roadmap", "budget allocation",
    "market analysis", "regulatory filings", "audit findings",
    "operational metrics", "merger terms", "insurance coverage",
    "environmental impact", "staffing plan", "technology migration",
]

_SECTIONS = [
    "Executive Summary", "Introduction", "Methodology", "Results",
    "Discussion", "Conclusion", "Appendix A", "Financial Overview",
    "Risk Assessment", "Recommendations", "Legal Terms",
    "Technical Specifications", "Project Timeline", "Budget Summary",
]

_ENTITIES = ["PERSON", "ORG", "DATE", "MONEY", "PERCENT", "GPE", "LOC"]

_CHART_TYPES = ["bar", "line", "pie", "scatter", "heatmap", "auto"]

_GREETINGS = [
    "Hello!", "Hi there", "Hey", "Good morning", "Good afternoon",
    "What can you do?", "Who are you?", "Help me understand DocWain",
    "Thanks!", "Thank you for your help", "What is DocWain?",
    "How does this work?", "Can you explain your capabilities?",
]

_META_QUESTIONS = [
    "What types of documents can you process?",
    "How accurate is the OCR extraction?",
    "What languages do you support?",
    "Can you handle scanned PDFs?",
    "How do you ensure data privacy?",
    "What file formats are supported?",
    "How long does document processing take?",
    "Can you process multiple documents at once?",
    "What is the maximum file size?",
    "How do you handle tables in documents?",
]

_META_ANSWERS = [
    "I can process PDFs, Word documents, scanned images, and many other formats using OCR and layout analysis.",
    "DocWain uses state-of-the-art vision models for OCR, achieving high accuracy even on complex layouts.",
    "I support English as the primary language, with growing support for other languages.",
    "Yes, scanned PDFs are handled through our vision-based OCR pipeline.",
    "All documents are processed within your tenant scope. Data is isolated by subscription and profile.",
    "Supported formats include PDF, DOCX, PPTX, images (PNG/JPG), and plain text files.",
    "Processing time depends on document size, but most documents are ingested within seconds.",
    "Yes, you can upload and process multiple documents. Each is chunked and embedded independently.",
    "The default maximum file size is 50MB, but this is configurable.",
    "Tables are detected via layout analysis and extracted into structured row/column data.",
]

_GREETING_ANSWERS = [
    "Hello! I'm DocWain, your document intelligence assistant. How can I help you today?",
    "Hi! I'm here to help you analyse and extract information from your documents.",
    "Hey! I can help you search, summarise, and extract data from your uploaded documents.",
    "Good morning! Ready to help with your document questions.",
    "Good afternoon! What would you like to know about your documents?",
    "I can analyse documents, extract tables and entities, answer questions, create summaries, and generate visualisations from your data.",
    "I'm DocWain, an enterprise document intelligence assistant built to help you understand and extract insights from your documents.",
    "DocWain is a document intelligence platform that uses AI to help you search, analyse, and extract information from documents.",
    "You're welcome! Let me know if you need anything else.",
    "Happy to help! Feel free to ask more questions about your documents.",
    "I'm DocWain, an AI-powered document intelligence assistant. I can help with OCR, table extraction, search, summarisation, and more.",
    "Upload a document and ask me questions about it. I use semantic search, OCR, and layout analysis to find answers.",
    "I can extract text via OCR, detect document layout, pull tables, find entities, search across documents, summarise sections, cross-reference claims, and generate charts.",
]


# ---------------------------------------------------------------------------
# Helper: random picks
# ---------------------------------------------------------------------------

def _pick(pool: list, n: int = 1) -> Any:
    if n == 1:
        return _RNG.choice(pool)
    return _RNG.sample(pool, min(n, len(pool)))


def _rand_page() -> int:
    return _RNG.randint(1, 50)


def _rand_pages(n: int = 2) -> List[int]:
    return sorted(_RNG.sample(range(1, 51), n))


def _tools_json() -> str:
    return format_tools_for_prompt()


# ---------------------------------------------------------------------------
# Single-tool call generators (per tool)
# ---------------------------------------------------------------------------

def _gen_ocr_extract() -> Dict[str, Any]:
    page = _rand_page()
    query = _pick([
        f"Extract the text from page {page}",
        f"What does page {page} say?",
        f"Read the content on page {page}",
        f"OCR page {page} of the document",
        f"Can you extract text from page {page}?",
    ])
    call = {"name": "ocr_extract", "arguments": {"page": page}}
    result = {"text": f"[Extracted text from page {page}]", "confidence": 0.95}
    answer = f"Here is the extracted text from page {page}:\n\n[Extracted text from page {page}]"
    return format_tool_call_sft(query, [call], [result], answer, _tools_json())


def _gen_layout_extract() -> Dict[str, Any]:
    page = _rand_page()
    detail = _pick(["basic", "detailed"])
    query = _pick([
        f"Analyse the layout of page {page}",
        f"What is the structure of page {page}?",
        f"Show me the layout blocks on page {page}",
        f"Detect headings and paragraphs on page {page}",
        f"What structural elements are on page {page}?",
    ])
    call = {"name": "layout_extract", "arguments": {"page": page, "detail_level": detail}}
    result = {"blocks": [{"type": "heading", "text": "Section Title"}, {"type": "paragraph", "text": "Body text..."}]}
    answer = f"Page {page} has the following layout structure: a heading followed by body paragraphs."
    return format_tool_call_sft(query, [call], [result], answer, _tools_json())


def _gen_extract_table() -> Dict[str, Any]:
    page = _rand_page()
    topic = _pick(_TOPICS)
    query = _pick([
        f"Extract the table on page {page}",
        f"Pull the {topic} table from page {page}",
        f"Get the table data from page {page}",
        f"What does the table on page {page} contain?",
        f"Show me the {topic} table on page {page}",
    ])
    call = {"name": "extract_table", "arguments": {"page": page}}
    if "topic" in query.lower() or topic in query:
        call["arguments"]["table_hint"] = topic
    result = {"rows": [["Header A", "Header B"], ["Value 1", "Value 2"]], "row_count": 2}
    answer = f"The table on page {page} contains 2 rows with columns Header A and Header B."
    return format_tool_call_sft(query, [call], [result], answer, _tools_json())


def _gen_extract_entities() -> Dict[str, Any]:
    page = _rand_page()
    entity_types = _pick(_ENTITIES, _RNG.randint(1, 3))
    if isinstance(entity_types, str):
        entity_types = [entity_types]
    scope = _pick([f"page:{page}", "full", f"section:{_pick(_SECTIONS)}"])
    query = _pick([
        f"Find all {', '.join(entity_types)} entities on page {page}",
        f"Extract named entities from the {_pick(_SECTIONS)} section",
        f"What people and organisations are mentioned in the document?",
        f"List all dates mentioned on page {page}",
        f"Identify entities in {scope}",
    ])
    call = {"name": "extract_entities", "arguments": {"scope": scope, "entity_types": entity_types}}
    result = {"entities": [{"text": "Acme Corp", "label": "ORG", "confidence": 0.92}]}
    answer = f"I found the following entities: Acme Corp (ORG, confidence 0.92)."
    return format_tool_call_sft(query, [call], [result], answer, _tools_json())


def _gen_context_understand() -> Dict[str, Any]:
    topic = _pick(_TOPICS)
    query = _pick([
        f"What does the document say about {topic}?",
        f"Find information related to {topic}",
        f"Explain the {topic} section of the document",
        f"What evidence supports {topic} in this document?",
        f"How does the document address {topic}?",
    ])
    call = {"name": "context_understand", "arguments": {"query": query}}
    result = {"evidence": [{"text": f"The document discusses {topic} in detail...", "confidence": 0.88}]}
    answer = f"Based on the document, {topic} is discussed in detail. The key finding is that [relevant information about {topic}]."
    return format_tool_call_sft(query, [call], [result], answer, _tools_json())


def _gen_cross_reference() -> Dict[str, Any]:
    topic = _pick(_TOPICS)
    claim = _pick([
        f"The {topic} figures are consistent across sections",
        f"Revenue increased by 15% according to the report",
        f"The {topic} section contradicts the executive summary",
        f"All departments met their {topic} targets",
        f"The {topic} data matches the appendix tables",
    ])
    scope = _pick(["document", "collection"])
    call = {"name": "cross_reference", "arguments": {"claim": claim, "scope": scope}}
    result = {"pairs": [{"source": "Section 2", "target": "Appendix A", "label": "agreement"}]}
    answer = f"Cross-referencing the claim: Section 2 and Appendix A are in agreement on this point."
    return format_tool_call_sft(query=f"Verify: {claim}", tool_calls=[call], tool_results=[result], final_answer=answer, tools_json=_tools_json())


def _gen_search_documents() -> Dict[str, Any]:
    topic = _pick(_TOPICS)
    top_k = _RNG.randint(3, 10)
    query = _pick([
        f"Search for information about {topic}",
        f"Find documents mentioning {topic}",
        f"What do our documents say about {topic}?",
        f"Look up {topic} across all uploaded documents",
        f"Search the collection for {topic} references",
    ])
    call = {"name": "search_documents", "arguments": {"query": topic, "top_k": top_k}}
    result = {"results": [{"chunk": f"... {topic} is discussed ...", "score": 0.91, "source": "report.pdf"}]}
    answer = f"I found relevant information about {topic} in report.pdf with a relevance score of 0.91."
    return format_tool_call_sft(query, [call], [result], answer, _tools_json())


def _gen_summarize_section() -> Dict[str, Any]:
    section = _pick(_SECTIONS)
    detail = _pick(["brief", "standard", "comprehensive"])
    query = _pick([
        f"Summarise the {section} section",
        f"Give me a {detail} summary of {section}",
        f"What is the {section} about?",
        f"Summarise pages 3-7",
        f"Provide a {detail} overview of the {section}",
    ])
    section_arg = section if "pages" not in query else "pages:3-7"
    call = {"name": "summarize_section", "arguments": {"section": section_arg, "detail": detail}}
    result = {"summary": f"The {section} covers the main findings and recommendations..."}
    answer = f"Summary of {section}: The section covers the main findings and recommendations related to the document's core topics."
    return format_tool_call_sft(query, [call], [result], answer, _tools_json())


def _gen_visualize_data() -> Dict[str, Any]:
    page = _rand_page()
    chart_type = _pick(_CHART_TYPES)
    topic = _pick(_TOPICS)
    query = _pick([
        f"Create a {chart_type} chart from the table on page {page}",
        f"Visualise the {topic} data",
        f"Generate a chart showing {topic} trends",
        f"Plot the data from page {page}",
        f"Make a {chart_type} chart of the {topic} figures",
    ])
    call = {
        "name": "visualize_data",
        "arguments": {
            "data": f"table:page{page}:0",
            "chart_type": chart_type,
            "title": f"{topic.title()} Overview",
        },
    }
    result = {"spec": {"mark": chart_type, "data": []}, "rendered": True}
    answer = f"I have generated a {chart_type} chart titled '{topic.title()} Overview' from the data on page {page}."
    return format_tool_call_sft(query, [call], [result], answer, _tools_json())


# Map tool name -> generator function
_SINGLE_GENERATORS = [
    _gen_ocr_extract,
    _gen_layout_extract,
    _gen_extract_table,
    _gen_extract_entities,
    _gen_context_understand,
    _gen_cross_reference,
    _gen_search_documents,
    _gen_summarize_section,
    _gen_visualize_data,
]

# ---------------------------------------------------------------------------
# Parallel tool-call combos
# ---------------------------------------------------------------------------

_PARALLEL_COMBOS = [
    # (query template, list of (gen_call_fn, result_fn) tuples)
    ("ocr_extract", "layout_extract"),
    ("ocr_extract", "extract_table"),
    ("search_documents", "context_understand"),
    ("extract_entities", "summarize_section"),
    ("search_documents", "summarize_section"),
    ("extract_table", "visualize_data"),
    ("ocr_extract", "extract_entities"),
    ("context_understand", "cross_reference"),
    ("layout_extract", "extract_table"),
    ("search_documents", "extract_entities"),
    ("ocr_extract", "layout_extract", "extract_table"),
    ("search_documents", "context_understand", "summarize_section"),
]


def _make_call_for_tool(tool_name: str) -> tuple:
    """Return (call_dict, result_dict) for a given tool name."""
    page = _rand_page()
    topic = _pick(_TOPICS)
    section = _pick(_SECTIONS)

    dispatch = {
        "ocr_extract": (
            {"name": "ocr_extract", "arguments": {"page": page}},
            {"text": f"[Text from page {page}]", "confidence": 0.94},
        ),
        "layout_extract": (
            {"name": "layout_extract", "arguments": {"page": page, "detail_level": "detailed"}},
            {"blocks": [{"type": "heading"}, {"type": "paragraph"}]},
        ),
        "extract_table": (
            {"name": "extract_table", "arguments": {"page": page}},
            {"rows": [["A", "B"], ["1", "2"]], "row_count": 2},
        ),
        "extract_entities": (
            {"name": "extract_entities", "arguments": {"scope": f"page:{page}"}},
            {"entities": [{"text": "Example Corp", "label": "ORG"}]},
        ),
        "context_understand": (
            {"name": "context_understand", "arguments": {"query": f"information about {topic}"}},
            {"evidence": [{"text": f"Document discusses {topic}...", "confidence": 0.87}]},
        ),
        "cross_reference": (
            {"name": "cross_reference", "arguments": {"claim": f"{topic} is addressed consistently"}},
            {"pairs": [{"source": "Section 1", "target": "Section 3", "label": "agreement"}]},
        ),
        "search_documents": (
            {"name": "search_documents", "arguments": {"query": topic, "top_k": 5}},
            {"results": [{"chunk": f"...{topic}...", "score": 0.89}]},
        ),
        "summarize_section": (
            {"name": "summarize_section", "arguments": {"section": section, "detail": "standard"}},
            {"summary": f"{section} covers key findings..."},
        ),
        "visualize_data": (
            {"name": "visualize_data", "arguments": {"data": f"table:page{page}:0", "chart_type": "auto"}},
            {"spec": {}, "rendered": True},
        ),
    }
    return dispatch[tool_name]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_single_tool_examples(count: int = 108) -> List[Dict[str, Any]]:
    """Generate single-tool-call training examples (at least 100).

    Distributes evenly across all 9 tools (12 per tool), with the remainder
    filled by random tools.
    """
    examples: List[Dict[str, Any]] = []
    per_tool = count // len(_SINGLE_GENERATORS)
    for gen_fn in _SINGLE_GENERATORS:
        for _ in range(per_tool):
            examples.append(gen_fn())
    # Fill remainder
    while len(examples) < count:
        examples.append(_pick(_SINGLE_GENERATORS)())
    _RNG.shuffle(examples)
    return examples


def generate_parallel_tool_examples(count: int = 55) -> List[Dict[str, Any]]:
    """Generate parallel-tool-call training examples (at least 50).

    Each example invokes 2 or 3 tools simultaneously.
    """
    examples: List[Dict[str, Any]] = []
    tools_json = _tools_json()
    topic = _pick(_TOPICS)

    for i in range(count):
        combo = _pick(_PARALLEL_COMBOS)
        calls = []
        results = []
        for tool_name in combo:
            call, result = _make_call_for_tool(tool_name)
            calls.append(call)
            results.append(result)

        topic = _pick(_TOPICS)
        tool_names = " and ".join(combo)
        query = _pick([
            f"Analyse page {_rand_page()} completely - extract text, layout, and any tables",
            f"Search for {topic} and give me a summary of what you find",
            f"Extract the table from page {_rand_page()} and create a chart from it",
            f"Find information about {topic} and check for consistency across sections",
            f"Get the entities from page {_rand_page()} and summarise the section",
            f"Extract text from page {_rand_page()} and identify the entities",
            f"Search for {topic}, understand the context, and summarise the findings",
        ])
        answer = f"I used {tool_names} to fulfil your request. Here are the combined results from the analysis."
        examples.append(
            format_tool_call_sft(query, calls, results, answer, tools_json)
        )

    return examples


def generate_no_tool_examples(count: int = 55) -> List[Dict[str, Any]]:
    """Generate no-tool-needed training examples (at least 50).

    Covers greetings, meta questions, and simple conversational exchanges
    where no document tool should be invoked.
    """
    examples: List[Dict[str, Any]] = []
    tools_json = _tools_json()

    # Greetings
    for i, greeting in enumerate(_GREETINGS):
        answer = _GREETING_ANSWERS[i % len(_GREETING_ANSWERS)]
        examples.append(format_no_tool_sft(greeting, answer, tools_json))

    # Meta questions
    for i, question in enumerate(_META_QUESTIONS):
        answer = _META_ANSWERS[i % len(_META_ANSWERS)]
        examples.append(format_no_tool_sft(question, answer, tools_json))

    # Conversational / simple-answer queries
    _simple_pairs = [
        ("What format should I upload documents in?", "You can upload PDF, DOCX, PPTX, images, and plain text files."),
        ("How many documents have I uploaded?", "I don't have access to your document count in this context. Please check the documents page in the UI."),
        ("Can you delete a document for me?", "I can help you with document analysis, but deletion must be done through the documents management interface."),
        ("Is my data secure?", "Yes, all data is isolated by subscription and profile. Documents are stored securely in Azure Blob Storage."),
        ("What's the weather like?", "I'm a document intelligence assistant and don't have access to weather data. I can help you with your documents though!"),
        ("Tell me a joke", "I'm focused on document intelligence, but here's a quick one: Why did the PDF go to therapy? It had too many unresolved layers! How can I help with your documents?"),
        ("Goodbye", "Goodbye! Feel free to come back whenever you need help with your documents."),
        ("Can you write code for me?", "My speciality is document analysis and extraction. I can help you understand, search, and extract information from your uploaded documents."),
        ("What time is it?", "I don't have access to real-time clock data. I'm here to help you with document analysis and extraction."),
        ("Who made you?", "I'm DocWain, built as an enterprise document intelligence assistant. How can I help with your documents?"),
        ("How do I use you?", "Upload your documents, then ask me questions about them. I can extract text, find entities, search across documents, summarise sections, and create visualisations."),
        ("Are you AI?", "Yes, I'm an AI-powered document intelligence assistant. I use advanced models for OCR, layout analysis, semantic search, and natural language understanding."),
        ("Can you remember our conversation?", "I maintain context within the current session. For long-term memory, the platform stores your document collection and profile settings."),
        ("What's new?", "I'm continuously improving my document analysis capabilities. Upload a document and try asking questions to see the latest features in action."),
        ("How accurate are your answers?", "My answers are grounded in the document content I retrieve. I provide confidence scores and source references so you can verify the information."),
        ("Can you process handwritten documents?", "I have OCR capabilities that work on printed text. Handwritten document support depends on legibility, but modern vision models handle many handwriting styles."),
        ("What is RAG?", "RAG stands for Retrieval-Augmented Generation. It's the technique I use to find relevant document passages and generate grounded answers based on them."),
        ("Explain embeddings", "Embeddings are numerical representations of text that capture semantic meaning. I use them to find relevant document chunks when you ask questions."),
        ("How do you handle large documents?", "Large documents are split into overlapping chunks (250-450 tokens each) and each chunk is embedded separately for efficient retrieval."),
        ("What models do you use?", "I use BAAI/bge-large-en-v1.5 for embeddings and various LLMs for generation. The exact model depends on your deployment configuration."),
        ("Can you compare two documents?", "Yes, you can upload multiple documents and I can cross-reference information between them using semantic search and entity extraction."),
        ("Do you support Excel files?", "Currently I support PDF, DOCX, PPTX, images, and plain text. Spreadsheet support may vary by deployment configuration."),
    ]

    for q, a in _simple_pairs:
        examples.append(format_no_tool_sft(q, a, tools_json))

    # Pad to count if needed
    while len(examples) < count:
        q, a = _pick(_simple_pairs)
        examples.append(format_no_tool_sft(q, a, tools_json))

    _RNG.shuffle(examples)
    return examples


def generate_auto_invocation_examples(count: int = 55) -> List[Dict[str, Any]]:
    """Generate auto-invocation examples (at least 50).

    These simulate scenarios where a tool has already been invoked by the
    pipeline and the model sees the tool_response in context. The model
    must produce a final answer incorporating the tool results.
    """
    examples: List[Dict[str, Any]] = []
    tools_json = _tools_json()

    auto_tools = [
        "ocr_extract", "layout_extract", "context_understand",
        "extract_table", "extract_entities", "cross_reference",
        "search_documents",
    ]

    queries_by_tool = {
        "ocr_extract": [
            "What text is on page {page}?",
            "Read page {page} for me",
            "Extract the content from page {page}",
        ],
        "layout_extract": [
            "What is the structure of page {page}?",
            "Show me the layout of page {page}",
            "Detect the document structure on page {page}",
        ],
        "context_understand": [
            "What does the document say about {topic}?",
            "Find information on {topic}",
            "Explain {topic} based on the document",
        ],
        "extract_table": [
            "Show me the table on page {page}",
            "Extract the data table from page {page}",
            "What are the table contents on page {page}?",
        ],
        "extract_entities": [
            "What entities are mentioned on page {page}?",
            "Find all named entities in the document",
            "List the organisations mentioned on page {page}",
        ],
        "cross_reference": [
            "Is the {topic} data consistent across the document?",
            "Verify {topic} claims across sections",
            "Cross-check the {topic} figures",
        ],
        "search_documents": [
            "Search for {topic} in my documents",
            "Find all mentions of {topic}",
            "What documents discuss {topic}?",
        ],
    }

    answers_by_tool = {
        "ocr_extract": "Based on the extracted text from page {page}, the content discusses {topic}. The OCR extraction was performed with high confidence.",
        "layout_extract": "Page {page} has a structured layout with headings, paragraphs, and a table. The main sections are clearly delineated.",
        "context_understand": "The document provides detailed information about {topic}. Key evidence includes relevant passages with confidence score 0.87.",
        "extract_table": "The table on page {page} contains structured data with 2 columns and multiple rows showing {topic} information.",
        "extract_entities": "I identified several entities on page {page}: Example Corp (ORG), John Smith (PERSON), and 2025-01-15 (DATE).",
        "cross_reference": "Cross-referencing {topic}: the data is consistent between Section 1 and the Appendix, with both sources in agreement.",
        "search_documents": "I found relevant results about {topic} across your documents. The most relevant match (score 0.91) comes from report.pdf.",
    }

    for i in range(count):
        tool_name = _pick(auto_tools)
        page = _rand_page()
        topic = _pick(_TOPICS)

        query_template = _pick(queries_by_tool[tool_name])
        query = query_template.format(page=page, topic=topic)

        call, result = _make_call_for_tool(tool_name)
        answer_template = answers_by_tool[tool_name]
        answer = answer_template.format(page=page, topic=topic)

        examples.append(
            format_tool_call_sft(query, [call], [result], answer, tools_json)
        )

    return examples


def build_tool_calling_dataset(output_path: str | Path) -> Path:
    """Build the full tool-calling dataset and write to a JSONL file.

    Combines all four categories:
      - Single tool calls (100+)
      - Parallel tool calls (50+)
      - No-tool-needed (50+)
      - Auto-invocation (50+)

    Returns the Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_examples: List[Dict[str, Any]] = []
    all_examples.extend(generate_single_tool_examples())
    all_examples.extend(generate_parallel_tool_examples())
    all_examples.extend(generate_no_tool_examples())
    all_examples.extend(generate_auto_invocation_examples())

    _RNG.shuffle(all_examples)

    with open(output_path, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    return output_path
