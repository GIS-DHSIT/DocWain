"""Function-calling schemas for DocWain's 9 core document intelligence tools.

Each schema follows the OpenAI function-calling format so it can be embedded
in system prompts and used during tool-calling fine-tune data generation.

Tools are split into two categories:
  - Auto-invoked: triggered automatically by the pipeline during ingestion/query
  - Model-decided: the LLM chooses when to call them based on user intent
"""

from __future__ import annotations

import json
from typing import List, Set

# ---------------------------------------------------------------------------
# Individual tool schemas
# ---------------------------------------------------------------------------

_OCR_EXTRACT = {
    "type": "function",
    "function": {
        "name": "ocr_extract",
        "description": (
            "Vision-based text extraction from a document page or region. "
            "Returns raw recognised text with bounding-box metadata."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "page": {
                    "type": "integer",
                    "description": "1-based page number to extract text from.",
                },
                "region": {
                    "type": "string",
                    "description": (
                        "Optional bounding-box region in 'x0,y0,x1,y1' format "
                        "(normalised 0-1 coordinates). Omit to extract full page."
                    ),
                },
            },
            "required": ["page"],
        },
    },
}

_LAYOUT_EXTRACT = {
    "type": "function",
    "function": {
        "name": "layout_extract",
        "description": (
            "Detect and classify the structural layout of a document page — "
            "headings, paragraphs, tables, figures, headers/footers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "page": {
                    "type": "integer",
                    "description": "1-based page number to analyse.",
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["basic", "detailed"],
                    "description": (
                        "'basic' returns block types only; 'detailed' includes "
                        "bounding boxes and reading-order indices."
                    ),
                },
            },
            "required": ["page"],
        },
    },
}

_EXTRACT_TABLE = {
    "type": "function",
    "function": {
        "name": "extract_table",
        "description": (
            "Extract a table from a document page and return it as structured "
            "row/column data (list of lists)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "page": {
                    "type": "integer",
                    "description": "1-based page number containing the table.",
                },
                "table_hint": {
                    "type": "string",
                    "description": (
                        "Optional natural-language hint to disambiguate when a "
                        "page contains multiple tables, e.g. 'revenue breakdown'."
                    ),
                },
            },
            "required": ["page"],
        },
    },
}

_EXTRACT_ENTITIES = {
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": (
            "Named-entity recognition over document text. Returns entities with "
            "labels, spans, and confidence scores."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "description": (
                        "Scope of extraction: 'page:<n>' for a single page, "
                        "'section:<title>' for a named section, or 'full' for "
                        "the entire document."
                    ),
                },
                "entity_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional filter list of entity types to extract, "
                        "e.g. ['PERSON', 'ORG', 'DATE']. Omit to extract all."
                    ),
                },
            },
            "required": ["scope"],
        },
    },
}

_CONTEXT_UNDERSTAND = {
    "type": "function",
    "function": {
        "name": "context_understand",
        "description": (
            "Deep comprehension of document context relative to a query. "
            "Identifies relevant passages, resolves co-references, and returns "
            "grounded evidence snippets with confidence scores."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user question or information need.",
                },
                "scope": {
                    "type": "string",
                    "description": (
                        "Scope to search: 'full', 'page:<n>', or "
                        "'section:<title>'. Defaults to 'full'."
                    ),
                },
                "min_confidence": {
                    "type": "number",
                    "description": (
                        "Minimum confidence threshold (0.0-1.0) for returned "
                        "evidence. Defaults to 0.5."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

_CROSS_REFERENCE = {
    "type": "function",
    "function": {
        "name": "cross_reference",
        "description": (
            "Find supporting or contradicting passages across multiple "
            "sections or documents for a given claim. Returns linked evidence "
            "pairs with agreement/conflict labels."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "The statement or claim to cross-reference.",
                },
                "scope": {
                    "type": "string",
                    "description": (
                        "Scope: 'document' (within current doc) or 'collection' "
                        "(across all profile documents). Defaults to 'document'."
                    ),
                },
            },
            "required": ["claim"],
        },
    },
}

_SEARCH_DOCUMENTS = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Semantic vector search across the document collection. Returns "
            "ranked chunks with relevance scores and source metadata."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "Maximum number of results to return. Defaults to 5."
                    ),
                },
                "date_filter": {
                    "type": "string",
                    "description": (
                        "Optional ISO-8601 date range filter in "
                        "'YYYY-MM-DD..YYYY-MM-DD' format."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

_SUMMARIZE_SECTION = {
    "type": "function",
    "function": {
        "name": "summarize_section",
        "description": (
            "Generate a targeted summary of a specific document section. "
            "Supports different detail levels from one-liner to comprehensive."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": (
                        "Section identifier: a section title, page range "
                        "'pages:3-7', or 'full' for the whole document."
                    ),
                },
                "detail": {
                    "type": "string",
                    "enum": ["brief", "standard", "comprehensive"],
                    "description": (
                        "Level of detail: 'brief' (~1-2 sentences), "
                        "'standard' (~1 paragraph), 'comprehensive' (multi-paragraph)."
                    ),
                },
            },
            "required": ["section"],
        },
    },
}

_VISUALIZE_DATA = {
    "type": "function",
    "function": {
        "name": "visualize_data",
        "description": (
            "Generate a chart or visualisation from extracted document data. "
            "Returns a rendering specification (Vega-Lite compatible)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": (
                        "JSON-encoded data array or a reference to a previously "
                        "extracted table, e.g. 'table:page3:0'."
                    ),
                },
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "pie", "scatter", "heatmap", "auto"],
                    "description": (
                        "Desired chart type. Use 'auto' to let the system "
                        "choose the best visualisation."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Optional chart title.",
                },
            },
            "required": ["data", "chart_type"],
        },
    },
}

# ---------------------------------------------------------------------------
# Ordered registry
# ---------------------------------------------------------------------------

_CORE_TOOL_SCHEMAS: List[dict] = [
    _OCR_EXTRACT,
    _LAYOUT_EXTRACT,
    _EXTRACT_TABLE,
    _EXTRACT_ENTITIES,
    _CONTEXT_UNDERSTAND,
    _CROSS_REFERENCE,
    _SEARCH_DOCUMENTS,
    _SUMMARIZE_SECTION,
    _VISUALIZE_DATA,
]

_AUTO_INVOKED_TOOLS: Set[str] = {
    "ocr_extract",
    "layout_extract",
    "context_understand",
    "extract_table",
    "extract_entities",
    "cross_reference",
    "search_documents",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_core_tool_schemas() -> List[dict]:
    """Return the list of all 9 core tool schemas."""
    return list(_CORE_TOOL_SCHEMAS)


def get_auto_invoked_tools() -> Set[str]:
    """Return the set of tool names that are auto-invoked by the pipeline."""
    return set(_AUTO_INVOKED_TOOLS)


def format_tools_for_prompt() -> str:
    """Serialise all tool schemas as a JSON string for embedding in a system prompt."""
    return json.dumps(_CORE_TOOL_SCHEMAS, indent=2)
