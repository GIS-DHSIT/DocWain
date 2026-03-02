"""Role-specific prompt templates for multi-agent LLM pipeline.

Each agent role has optimized system prompts and templates designed for
its specific Ollama model (classifier=llama3.2, extractor=mistral,
generator=gpt-oss, verifier=deepseek-r1).
"""
from __future__ import annotations

# ── Classifier (llama3.2 — fast, JSON-only output) ───────────────────

CLASSIFIER_SYSTEM = (
    "You are a query classifier. Respond ONLY with valid JSON. "
    "No commentary, no explanation, no markdown fences."
)

CLASSIFIER_INTENT_TEMPLATE = """\
Classify this query. Return JSON with these exact keys:
- "intent": one of factual|comparison|summary|ranking|timeline|reasoning|multi_field|cross_document|contact|extract
- "domain": one of hr|invoice|legal|medical|policy|generic
- "entity": person or document name mentioned, or null
- "scope": one of all_profile|specific_document|targeted
- "confidence": float 0.0 to 1.0

Query: {query}
"""

# ── Extractor (mistral — precise structured extraction) ──────────────

EXTRACTOR_SYSTEM = (
    "You are a precise document extraction assistant. Extract ONLY information "
    "explicitly stated in the evidence. Never infer, guess, or add information "
    "not present in the source text. If information is not found, omit it."
)

EXTRACTOR_TEMPLATE = """\
Extract structured information from the following evidence to answer the query.

Query: {query}

Evidence:
{evidence}

Return a JSON object with the extracted fields. Only include fields that have
clear evidence in the text above.
"""

# ── Generator (gpt-oss — powerful synthesis) ─────────────────────────

GENERATOR_SYSTEM = (
    "You are a document intelligence assistant. Synthesize clear, accurate "
    "responses grounded in the provided evidence. Be thorough but concise."
)

# ── Verifier (deepseek-r1 — chain-of-thought reasoning) ─────────────

VERIFIER_SYSTEM = (
    "You are a grounding verification specialist. Your task is to check whether "
    "a generated answer is fully supported by the provided evidence. Think "
    "step-by-step and identify any claims not grounded in the evidence."
)

VERIFIER_TEMPLATE = """\
Verify whether the answer is fully supported by the evidence.

Query: {query}

Answer to verify:
{answer}

Evidence:
{evidence}

Think step-by-step:
1. List each factual claim in the answer.
2. For each claim, check if it is explicitly supported by the evidence.
3. Flag any claims that are not supported or are contradicted.

Return JSON with these exact keys:
- "supported": boolean (true if all major claims are grounded)
- "confidence": float 0.0 to 1.0
- "issues": list of strings (unsupported claims, empty if none)
- "reasoning": string (your step-by-step analysis)
"""
