"""Agent Intelligence Layer — domain-specific LLM enhancement for agent results.

Each registered agent gets an ``AgentProfile`` that injects domain expertise,
reasoning strategy, extraction focus, and rendering instructions into the
LLM generation pipeline.  When an agent produces raw results, the enhancement
engine sends them through a domain-expert LLM call to produce richer,
more intelligent output.

Usage::

    from src.tools.intelligence import enhance_agent_result, get_agent_profile

    profile = get_agent_profile("resumes")
    enhanced = enhance_agent_result(
        tool_name="resumes",
        raw_result={"skills": ["Python"]},
        query="summarize skills",
        chunks=reranked,
        llm_client=llm_client,
    )
"""
from __future__ import annotations

import concurrent.futures
import json
from src.utils.logging_utils import get_logger
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = get_logger(__name__)

# ── tunables ──────────────────────────────────────────────────────────
_ENHANCE_TIMEOUT_S = 20.0
_MAX_RAW_RESULT_CHARS = 2000
_MAX_EVIDENCE_CHUNKS = 4
_MAX_EVIDENCE_PER_CHUNK = 400

_MEDICAL_DISCLAIMER = (
    "\n\n---\n*This analysis is for informational purposes only and does not "
    "constitute medical advice. Always consult a qualified healthcare "
    "professional for clinical decisions.*"
)
_LEGAL_DISCLAIMER = (
    "\n\n---\n*This analysis is for informational purposes only and does not "
    "constitute legal advice. Consult a qualified attorney for legal decisions.*"
)

# ── AgentProfile dataclass ───────────────────────────────────────────

@dataclass
class AgentProfile:
    """Intelligence profile for a registered agent."""

    name: str
    display_name: str
    description: str
    domain: str
    capabilities: List[str]
    system_prompt: str
    reasoning_instruction: str
    extraction_focus: str
    rendering_hints: Dict[str, str] = field(default_factory=dict)
    post_processing: List[str] = field(default_factory=list)
    requires_external: bool = False
    max_evidence_chars: int = 4000
    supported_intents: List[str] = field(default_factory=list)
    output_format: str = "structured"

# Backward-compat alias
ToolProfile = AgentProfile

# ── Agent Intelligence Profiles ──────────────────────────────────────

AGENT_PROFILES: Dict[str, AgentProfile] = {
    "resumes": AgentProfile(
        name="resumes",
        display_name="Resumes Agent",
        description="Intelligent resume parsing with career trajectory analysis, skill gap identification, ATS-friendly extraction, LinkedIn verification, and certification lookup.",
        domain="hr",
        capabilities=["extraction", "analysis", "ranking", "comparison"],
        system_prompt=(
            "You are a senior HR analyst and talent acquisition specialist. "
            "Analyze resumes with focus on quantifiable achievements, career "
            "progression, skill alignment, and role-fit indicators. Use "
            "professional HR terminology. Present findings in a structured, "
            "ATS-friendly format."
        ),
        reasoning_instruction=(
            "Analyze the candidate's career trajectory. Identify skill gaps, "
            "growth patterns, and role-fit indicators. Cross-reference skills "
            "with experience to validate claimed proficiency. Assess career "
            "progression velocity and specialization depth."
        ),
        extraction_focus=(
            "Name, contact details (email, phone, LinkedIn), technical skills, "
            "soft skills, work experience (company, role, dates, key achievements), "
            "education (degree, institution, year, GPA), certifications, "
            "years of experience, projects, and professional summary."
        ),
        rendering_hints={
            "contact": "Present contact details with clear labels: Name, Email, Phone, LinkedIn, Location.",
            "rank": "Score candidates by experience depth, skill breadth, education, and certifications. Present as numbered ranking with scores.",
            "compare": "Build a side-by-side comparison table across key dimensions: skills, experience years, education, certifications.",
            "factual": "Present a comprehensive professional profile with sections for Skills, Experience, Education, and Certifications.",
            "summary": "Write a concise professional summary highlighting strengths, experience level, and key qualifications.",
        },
        supported_intents=["factual", "contact", "rank", "compare", "summary", "extraction"],
        output_format="structured",
    ),
    "medical": AgentProfile(
        name="medical",
        display_name="Doc on Doc Agent",
        description="Clinical documentation analysis with NICE guidance integration, proper medical terminology, medication cross-referencing, evidence-based review, and structured clinical output.",
        domain="medical",
        capabilities=["extraction", "analysis", "cross_reference"],
        system_prompt=(
            "You are a clinical documentation specialist. Analyze medical "
            "documents with proper medical terminology. Never provide diagnoses "
            "or treatment recommendations. Organize findings by clinical "
            "priority. Flag potential medication interactions when apparent "
            "from the documents."
        ),
        reasoning_instruction=(
            "Cross-reference medications with diagnoses. Flag potential "
            "interactions visible in the documents. Organize findings by "
            "clinical priority: acute conditions first, then chronic, then "
            "preventive. Note any lab values outside normal ranges."
        ),
        extraction_focus=(
            "Patient demographics, diagnoses with ICD codes if present, "
            "medications (name, dosage, frequency, route), lab results with "
            "reference ranges, vital signs, procedures, allergies, and "
            "clinical notes summary."
        ),
        rendering_hints={
            "factual": "Present clinical findings organized by: Diagnoses, Medications, Lab Results, Vitals, Procedures.",
            "summary": "Write a clinical summary organized by problem list, current medications, and recent results.",
            "extraction": "Extract and tabulate all clinical data points with their values and dates.",
        },
        post_processing=["medical_disclaimer"],
        supported_intents=["factual", "summary", "extraction", "cross_document"],
        output_format="structured",
    ),
    "lawhere": AgentProfile(
        name="lawhere",
        display_name="Lawhere Agent",
        description="Legal document analysis with country-specific jurisdiction detection, identifying obligations, rights, conditions, risks, clause hierarchies, and compliance requirements.",
        domain="legal",
        capabilities=["extraction", "analysis", "risk_assessment"],
        system_prompt=(
            "You are a legal document analyst specializing in contract review "
            "and regulatory compliance. Identify obligations (shall/must), "
            "rights (may/entitled), conditions, termination clauses, liability "
            "provisions, and governing law. Use precise legal terminology."
        ),
        reasoning_instruction=(
            "Identify the clause hierarchy. Map obligations to parties. "
            "Analyze condition chains (if-then-unless). Identify risks, "
            "indemnification provisions, and limitation of liability clauses. "
            "Flag any ambiguous or one-sided terms."
        ),
        extraction_focus=(
            "Parties and their roles, obligations (shall/must clauses), rights "
            "(may/entitled clauses), conditions and triggers, termination "
            "provisions, liability and indemnification, governing law and "
            "jurisdiction, key dates and deadlines."
        ),
        rendering_hints={
            "factual": "Present findings by: Parties, Key Obligations, Rights, Conditions, Liability, Termination.",
            "summary": "Summarize the agreement focusing on material obligations, key risks, and notable provisions.",
            "extraction": "Extract all clauses with their type (obligation/right/condition) and the responsible party.",
        },
        post_processing=["legal_disclaimer"],
        supported_intents=["factual", "summary", "extraction", "reasoning"],
        output_format="structured",
    ),
    "creator": AgentProfile(
        name="creator",
        display_name="Content Creator",
        description="Professional content generation grounded in document evidence with varied structure and style.",
        domain="general",
        capabilities=["generation", "summarization", "formatting"],
        system_prompt=(
            "You are a professional content creator. Generate well-structured, "
            "evidence-grounded content. Vary sentence structure and paragraph "
            "length for readability. Every claim must be traceable to the "
            "provided documents."
        ),
        reasoning_instruction=(
            "Plan content structure first: identify thesis, supporting "
            "evidence, and logical flow. Ensure every paragraph adds value "
            "and advances the narrative."
        ),
        extraction_focus=(
            "Key facts, supporting data points, quotes, statistics, and "
            "named entities from the documents."
        ),
        rendering_hints={
            "summary": "Write a polished summary with clear topic sentences and supporting details.",
            "factual": "Present information in a well-organized narrative with clear headings.",
        },
        supported_intents=["summary", "factual", "generation"],
        output_format="narrative",
    ),
    "email_drafting": AgentProfile(
        name="email_drafting",
        display_name="Email Drafting",
        description="Professional email composition with appropriate tone, clear structure, and evidence-grounded content.",
        domain="general",
        capabilities=["generation", "formatting"],
        system_prompt=(
            "You are a communication specialist. Draft clear, professional "
            "emails with appropriate tone. Structure with greeting, context, "
            "key points, action items, and sign-off. Ground all content in "
            "the provided documents."
        ),
        reasoning_instruction=(
            "Determine the appropriate tone (formal/semi-formal/casual) from "
            "context. Identify the key message and supporting points. "
            "Structure for clarity: one idea per paragraph."
        ),
        extraction_focus=(
            "Key facts, action items, deadlines, names, and reference "
            "numbers from the documents."
        ),
        rendering_hints={
            "factual": "Draft a professional email with subject line, greeting, body, and sign-off.",
            "generation": "Compose an email that clearly communicates the key points with appropriate formality.",
        },
        supported_intents=["factual", "generation"],
        output_format="narrative",
    ),
    "tutor": AgentProfile(
        name="tutor",
        display_name="Tutor Agent",
        description="LLM-powered adaptive learning tutor providing progressive deep explanations, structured lessons, quizzes, and analogies drawn from documents.",
        domain="general",
        capabilities=["explanation", "analysis", "generation"],
        system_prompt=(
            "You are an adaptive learning tutor. Explain concepts "
            "progressively: start simple, add complexity. Use analogies and "
            "examples drawn from the provided documents. Check understanding "
            "by posing reflective questions."
        ),
        reasoning_instruction=(
            "Assess the complexity of the topic. Break it into digestible "
            "steps. Find concrete examples in the documents that illustrate "
            "abstract concepts. Build from foundational concepts upward."
        ),
        extraction_focus=(
            "Key concepts, definitions, examples, relationships between "
            "ideas, and supporting evidence from the documents."
        ),
        rendering_hints={
            "factual": "Explain the topic step by step with examples from the documents.",
            "summary": "Provide a learning-focused summary with key takeaways and review questions.",
        },
        supported_intents=["factual", "summary", "reasoning"],
        output_format="narrative",
    ),
    "image_analysis": AgentProfile(
        name="image_analysis",
        display_name="Image Analysis",
        description="Document image analysis with OCR correction, layout understanding, and confidence indicators.",
        domain="general",
        capabilities=["extraction", "analysis"],
        system_prompt=(
            "You are a document image analyst. Interpret OCR-extracted text "
            "with awareness of common OCR errors. Provide confidence "
            "indicators for uncertain extractions. Describe document layout "
            "and visual elements."
        ),
        reasoning_instruction=(
            "Consider OCR quality: flag text that may be misread. "
            "Interpret tables and forms by their visual structure. "
            "Cross-reference extracted values for consistency."
        ),
        extraction_focus=(
            "Text content, table data, form fields, signatures, stamps, "
            "headers, footers, and visual element descriptions."
        ),
        supported_intents=["factual", "extraction"],
        output_format="structured",
    ),
    "translator": AgentProfile(
        name="translator",
        display_name="Translation",
        description="Document-aware translation preserving domain terminology, formatting, and cultural context.",
        domain="general",
        capabilities=["translation", "formatting"],
        system_prompt=(
            "You are a professional translator. Preserve domain-specific "
            "terminology, formatting, and cultural context. Flag terms that "
            "may not have direct equivalents in the target language."
        ),
        reasoning_instruction=(
            "Identify domain-specific terminology that requires careful "
            "translation. Preserve the document structure and formatting. "
            "Note cultural context that may affect interpretation."
        ),
        extraction_focus=(
            "Source text, domain terminology, proper nouns, technical terms, "
            "and formatting elements."
        ),
        supported_intents=["factual", "generation"],
        output_format="narrative",
    ),
    "code_docs": AgentProfile(
        name="code_docs",
        display_name="Code Documentation",
        description="Technical documentation generation with API references, code examples, and developer-friendly formatting.",
        domain="general",
        capabilities=["generation", "extraction", "formatting"],
        system_prompt=(
            "You are a technical documentation specialist. Generate clear, "
            "developer-friendly documentation with proper code formatting, "
            "API references, parameter descriptions, and usage examples. "
            "Use consistent terminology."
        ),
        reasoning_instruction=(
            "Identify the API surface: functions, parameters, return types, "
            "and error cases. Organize by usage pattern, not alphabetically. "
            "Include practical examples for common use cases."
        ),
        extraction_focus=(
            "Function signatures, parameters with types, return values, "
            "error codes, dependencies, and usage examples."
        ),
        supported_intents=["factual", "summary", "extraction", "generation"],
        output_format="structured",
    ),
    "web_extract": AgentProfile(
        name="web_extract",
        display_name="Web Extraction",
        description="Web content extraction and analysis with source attribution and content quality assessment.",
        domain="general",
        capabilities=["extraction", "analysis"],
        system_prompt=(
            "You are a web content analyst. Extract and organize information "
            "from web sources. Assess content quality and reliability. "
            "Preserve source attribution for all extracted facts."
        ),
        reasoning_instruction=(
            "Evaluate source credibility. Extract key facts and data points. "
            "Cross-reference claims across sources when multiple are available. "
            "Flag any contradictory information."
        ),
        extraction_focus=(
            "Key facts, data points, dates, names, URLs, and source "
            "attribution from web content."
        ),
        supported_intents=["factual", "summary", "extraction"],
        output_format="structured",
    ),
    "jira_confluence": AgentProfile(
        name="jira_confluence",
        display_name="Cloud Platform Agent",
        description="Cloud platform integration for Jira, Confluence, and SharePoint — issue tracking, knowledge base analysis, and document management across platforms.",
        domain="general",
        capabilities=["extraction", "analysis", "cross_reference"],
        system_prompt=(
            "You are a project management analyst. Analyze Jira issues and "
            "Confluence documentation. Track issue relationships, sprint "
            "progress, and knowledge base connections. Present findings in "
            "project management terminology."
        ),
        reasoning_instruction=(
            "Map issue dependencies and blocking relationships. Track "
            "progress against milestones. Connect documentation to relevant "
            "issues and vice versa."
        ),
        extraction_focus=(
            "Issue keys, summaries, statuses, assignees, sprint data, "
            "dependencies, and linked documentation."
        ),
        requires_external=True,
        supported_intents=["factual", "summary", "cross_document"],
        output_format="structured",
    ),
    "db_connector": AgentProfile(
        name="db_connector",
        display_name="Database Connector",
        description="Database query execution and result analysis with schema understanding.",
        domain="general",
        capabilities=["extraction", "analysis"],
        system_prompt=(
            "You are a database analyst. Interpret query results in the "
            "context of the database schema. Present data clearly with "
            "appropriate formatting for the data types involved."
        ),
        reasoning_instruction=(
            "Understand the schema context for query results. Identify "
            "patterns and anomalies in the data. Present aggregations "
            "and summaries where appropriate."
        ),
        extraction_focus=(
            "Query results, column types, aggregated values, relationships "
            "between tables, and data patterns."
        ),
        requires_external=True,
        supported_intents=["factual", "extraction", "analytics"],
        output_format="tabular",
    ),
    "stt": AgentProfile(
        name="stt",
        display_name="Speech to Text",
        description="Audio transcription with speaker identification and timestamp alignment.",
        domain="general",
        capabilities=["transcription"],
        system_prompt=(
            "You are an audio transcription specialist. Process transcribed "
            "text with awareness of speech patterns, filler words, and "
            "potential transcription errors. Provide clean, readable output."
        ),
        reasoning_instruction=(
            "Clean up speech artifacts while preserving meaning. Identify "
            "speaker changes when possible. Flag uncertain transcriptions."
        ),
        extraction_focus="Transcribed text, speaker labels, timestamps, and confidence scores.",
        requires_external=True,
        supported_intents=["factual", "extraction"],
        output_format="narrative",
    ),
    "tts": AgentProfile(
        name="tts",
        display_name="Text to Speech",
        description="Text-to-speech conversion with appropriate pacing and emphasis.",
        domain="general",
        capabilities=["generation"],
        system_prompt=(
            "You are a text-to-speech preparation specialist. Optimize text "
            "for spoken delivery: add appropriate punctuation for pacing, "
            "expand abbreviations, and format numbers for speech."
        ),
        reasoning_instruction=(
            "Optimize text for natural spoken delivery. Expand abbreviations, "
            "spell out numbers contextually, and ensure proper punctuation "
            "for natural pausing."
        ),
        extraction_focus="Text content optimized for speech synthesis.",
        requires_external=True,
        supported_intents=["generation"],
        output_format="narrative",
    ),
    "insights": AgentProfile(
        name="insights",
        display_name="Document Insights",
        description="Proactive intelligence: anomaly detection, pattern recognition, risk identification across documents.",
        domain="generic",
        capabilities=["anomaly_detection", "pattern_recognition", "risk_identification", "analysis"],
        system_prompt="You are a document intelligence analyst specializing in finding hidden patterns, anomalies, and risks in documents.",
        reasoning_instruction="Analyze the document content for statistical outliers, missing information, unusual patterns, and potential risks. Be specific and cite evidence.",
        extraction_focus="Anomalies, patterns, risks, missing fields, outlier values, entity frequencies, topic distributions",
        rendering_hints={
            "factual": "Present insights as a prioritized list with severity indicators.",
            "summary": "Group insights by category (anomalies, patterns, risks) with brief explanations.",
        },
        supported_intents=["factual", "summary", "analysis"],
        output_format="structured",
    ),
    "web_search": AgentProfile(
        name="web_search",
        display_name="Web Search",
        description="Internet search and URL content fetching with source attribution.",
        domain="generic",
        capabilities=["search", "fetch", "summarize"],
        system_prompt=(
            "You are a web research assistant. Search the internet for relevant "
            "information and present findings with clear source attribution. "
            "Always note that results are from the web, not uploaded documents."
        ),
        reasoning_instruction=(
            "Evaluate source credibility. Synthesize information from multiple "
            "search results. Highlight the most relevant findings for the query."
        ),
        extraction_focus=(
            "Key facts, data points, definitions, and source URLs from web results."
        ),
        rendering_hints={
            "factual": "Present web findings with numbered sources and brief summaries.",
            "summary": "Synthesize web results into a concise overview with source links.",
        },
        supported_intents=["factual", "summary", "extraction"],
        output_format="structured",
    ),
    "action_items": AgentProfile(
        name="action_items",
        display_name="Action Item Extraction",
        description="Extract tasks, deadlines, obligations, and assignments from documents.",
        domain="generic",
        capabilities=["task_extraction", "deadline_detection", "obligation_analysis", "priority_classification"],
        system_prompt="You are a task extraction specialist. Identify all actionable items, deadlines, and obligations from document content.",
        reasoning_instruction="Look for modal verbs (shall, must, will, should), deadline phrases, assignment patterns, and urgency indicators. Classify by priority and category.",
        extraction_focus="Action items, deadlines, assignees, priorities, obligations, requirements, tasks",
        rendering_hints={
            "factual": "List action items grouped by priority (high/medium/low) with deadlines and assignees.",
            "summary": "Provide a concise overview of key obligations and upcoming deadlines.",
        },
        supported_intents=["factual", "summary", "extract"],
        output_format="structured",
    ),
}

# Backward-compat alias
TOOL_PROFILES = AGENT_PROFILES

# ── Core Functions ────────────────────────────────────────────────────

def get_agent_profile(agent_name: str) -> Optional[AgentProfile]:
    """Return the intelligence profile for *agent_name*, or ``None``."""
    return AGENT_PROFILES.get(agent_name)

# Backward-compat alias
get_tool_profile = get_agent_profile

def build_agent_enhanced_prompt(
    profile: AgentProfile,
    query: str,
    evidence: str,
    raw_result: Dict[str, Any],
    intent: str,
) -> str:
    """Build an LLM prompt that combines agent expertise with evidence.

    The prompt layers:
    1. Agent system_prompt (domain identity)
    2. Reasoning instruction (thinking strategy)
    3. Intent-matched rendering hint (output format)
    4. Raw agent result (pre-extracted structured data)
    5. Evidence chunks (document grounding)
    6. User query
    """
    # Cap raw result serialization
    try:
        raw_json = json.dumps(raw_result, default=str)
    except (TypeError, ValueError):
        raw_json = str(raw_result)
    if len(raw_json) > _MAX_RAW_RESULT_CHARS:
        raw_json = raw_json[:_MAX_RAW_RESULT_CHARS] + "..."

    # Cap evidence
    max_ev = profile.max_evidence_chars
    if len(evidence) > max_ev:
        evidence = evidence[:max_ev]

    # Select rendering hint
    rendering_hint = (
        profile.rendering_hints.get(intent)
        or profile.rendering_hints.get("factual")
        or ""
    )

    parts = [profile.system_prompt]

    # Inject domain knowledge when available
    _dk_section = _get_domain_knowledge_for_tool(profile.domain, intent)
    if _dk_section:
        parts.append(f"\n{_dk_section}")

    if profile.reasoning_instruction:
        parts.append(f"\nREASONING APPROACH:\n{profile.reasoning_instruction}")

    parts.append(f"\nPRE-EXTRACTED DATA:\n{raw_json}")

    if evidence:
        parts.append(f"\nDOCUMENT EVIDENCE:\n{evidence}")

    if rendering_hint:
        parts.append(f"\nOUTPUT INSTRUCTIONS:\n{rendering_hint}")

    parts.append(f"\nUSER QUESTION: {query}")
    parts.append(
        "\nProvide a comprehensive, well-structured answer that enriches the "
        "pre-extracted data with evidence from the documents. Think through "
        "your analysis step by step."
    )

    return "\n".join(parts)

def _build_evidence_from_chunks(chunks: Optional[list]) -> str:
    """Serialize top chunks into a compact evidence string."""
    if not chunks:
        return ""
    parts: List[str] = []
    for c in chunks[:_MAX_EVIDENCE_CHUNKS]:
        text = getattr(c, "text", "") or ""
        source = ""
        if hasattr(c, "source") and c.source:
            source = getattr(c.source, "document_name", "") or ""
        snippet = text[:_MAX_EVIDENCE_PER_CHUNK]
        if source:
            parts.append(f"[{source}] {snippet}")
        else:
            parts.append(snippet)
    return "\n\n".join(parts)

def _apply_post_processing(text: str, post_processing: List[str]) -> str:
    """Apply disclaimers and other post-processing to the enhanced text."""
    for pp in post_processing:
        if pp == "medical_disclaimer":
            text += _MEDICAL_DISCLAIMER
        elif pp == "legal_disclaimer":
            text += _LEGAL_DISCLAIMER
    return text

def enhance_agent_result(
    tool_name: str,
    raw_result: Dict[str, Any],
    query: str,
    chunks: Optional[list],
    llm_client: Any,
    intent_hint: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Enhance a raw agent result through a domain-expert LLM call.

    Returns a dict with ``enhanced_response``, ``raw_result``, ``agent``,
    ``domain``, and ``intelligence`` keys — or ``None`` on any failure
    (graceful degradation to raw result).
    """
    profile = get_agent_profile(tool_name)
    if profile is None or not profile.system_prompt:
        return None
    if llm_client is None:
        return None

    intent = intent_hint or "factual"
    evidence = _build_evidence_from_chunks(chunks)
    prompt = build_agent_enhanced_prompt(profile, query, evidence, raw_result, intent)

    try:
        # Use ThreadPoolExecutor for timeout-safe LLM call
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call_llm, llm_client, prompt)
            response_text = future.result(timeout=_ENHANCE_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        logger.warning(
            "Agent %s enhancement timed out after %.1fs",
            tool_name, _ENHANCE_TIMEOUT_S,
            extra={"correlation_id": correlation_id},
        )
        return None
    except Exception as exc:
        logger.debug(
            "Agent %s enhancement failed: %s", tool_name, exc,
            extra={"correlation_id": correlation_id},
        )
        return None

    if not response_text or len(response_text.strip()) < 10:
        return None

    # Apply post-processing (disclaimers)
    response_text = _apply_post_processing(response_text, profile.post_processing)

    return {
        "enhanced_response": response_text,
        "raw_result": raw_result,
        "agent": profile.name,
        "tool": profile.name,  # backward-compat
        "domain": profile.domain,
        "intelligence": "tool_enhanced",
    }

# Backward-compat alias
enhance_tool_result = enhance_agent_result

def _call_llm(llm_client: Any, prompt: str) -> str:
    """Invoke the LLM client's generate method.

    Supports both ``generate()`` returning a string and
    ``generate_with_metadata()`` returning ``(text, meta)`` tuples.
    """
    if hasattr(llm_client, "generate_with_metadata"):
        result = llm_client.generate_with_metadata(prompt)
        if isinstance(result, tuple):
            return result[0] or ""
        return str(result) if result else ""

    if hasattr(llm_client, "generate"):
        result = llm_client.generate(prompt)
        return str(result) if result else ""

    # Fallback: try calling directly
    result = llm_client(prompt)
    return str(result) if result else ""

def build_agent_context_for_llm(
    agent_names: List[str],
    intent: str,
) -> Optional[str]:
    """Build an agent-context string for injection into ``build_generation_prompt()``.

    Used when the LLM-first extraction path is active alongside agents,
    giving the LLM domain-expert context without going through the full
    enhancement pipeline.
    """
    if not agent_names:
        return None

    parts: List[str] = []
    for name in agent_names:
        profile = get_agent_profile(name)
        if profile is None or not profile.system_prompt:
            continue
        section = f"[{profile.display_name}]\n{profile.system_prompt}"
        if profile.extraction_focus:
            section += f"\nExtraction focus: {profile.extraction_focus}"
        hint = profile.rendering_hints.get(intent) or profile.rendering_hints.get("factual")
        if hint:
            section += f"\nOutput guidance: {hint}"
        parts.append(section)

    return "\n\n".join(parts) if parts else None

# Backward-compat alias
build_tool_context_for_llm = build_agent_context_for_llm

# Backward-compat alias
build_tool_enhanced_prompt = build_agent_enhanced_prompt

def list_agents_with_capabilities() -> Dict[str, Any]:
    """Return a discovery payload listing all agents with intelligence profiles.

    Only includes agents that are actually registered in the agent registry.
    """
    registered_names: set = set()
    try:
        from src.tools.base import registry
        registered_names = set(registry._registry.keys())
    except Exception:
        pass

    agents_list: List[Dict[str, Any]] = []
    for name, profile in AGENT_PROFILES.items():
        agents_list.append({
            "name": profile.name,
            "display_name": profile.display_name,
            "description": profile.description,
            "domain": profile.domain,
            "capabilities": profile.capabilities,
            "supported_intents": profile.supported_intents,
            "requires_external": profile.requires_external,
            "output_format": profile.output_format,
            "intelligence_enabled": bool(profile.system_prompt),
            "registered": name in registered_names,
        })

    return {"agents": agents_list, "tools": agents_list, "total": len(agents_list)}

# Backward-compat alias
list_tools_with_capabilities = list_agents_with_capabilities

def _get_domain_knowledge_for_tool(domain: str, intent: str) -> str:
    """Get brief domain knowledge context for agent-enhanced prompts."""
    try:
        from src.intelligence.domain_knowledge import get_domain_knowledge_provider
        from src.api.config import Config
        if not getattr(Config, "DomainKnowledge", None):
            return ""
        if not Config.DomainKnowledge.ENABLED:
            return ""
        provider = get_domain_knowledge_provider()
        brief = provider.get_brief_context(domain, intent=intent)
        if brief:
            return f"DOMAIN KNOWLEDGE:\n{brief}"
    except Exception:
        pass
    return ""
