"""Content prompt builder — domain-specific LLM prompts for content generation.

Builds structured prompts that inject extracted facts, source evidence,
and content type templates to guide LLM generation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .registry import ContentType


# ---------------------------------------------------------------------------
# System prompt fragments per domain
# ---------------------------------------------------------------------------

_DOMAIN_SYSTEM_PROMPTS: Dict[str, str] = {
    "hr": (
        "You are a professional HR content writer. Generate content that is "
        "accurate, evidence-based, and uses professional language appropriate "
        "for human resources and recruitment contexts."
    ),
    "invoice": (
        "You are a financial document specialist. Generate content that is "
        "precise with numbers, dates, and amounts. Always cite specific "
        "invoice data when referencing figures."
    ),
    "legal": (
        "You are a legal document analyst. Generate content using clear, "
        "precise language. Flag uncertainties. Do not provide legal advice — "
        "present facts and analysis from the source documents only."
    ),
    "medical": (
        "You are a medical documentation specialist. Generate content that is "
        "factual, uses appropriate medical terminology, and clearly attributes "
        "all claims to source documents. Do not make diagnoses."
    ),
    "report": (
        "You are a professional report writer. Generate clear, structured "
        "content with evidence-based findings and actionable insights."
    ),
    "general": (
        "You are a professional content writer. Generate clear, well-structured "
        "content that accurately reflects the source documents."
    ),
    "cross_document": (
        "You are an analytical content writer specializing in cross-document "
        "synthesis. Generate comparative and consolidated content that "
        "accurately attributes information to specific source documents."
    ),
}


# ---------------------------------------------------------------------------
# Content type-specific instructions
# ---------------------------------------------------------------------------

_TYPE_INSTRUCTIONS: Dict[str, str] = {
    # HR
    "cover_letter": (
        "Write a professional cover letter for the candidate. Include:\n"
        "- Opening paragraph referencing relevant experience\n"
        "- Body paragraph(s) highlighting key skills and achievements\n"
        "- Closing paragraph expressing interest\n"
        "Use only facts evidenced in the source documents."
    ),
    "professional_summary": (
        "Write a concise professional summary (3-5 sentences) that highlights:\n"
        "- Years of experience and primary expertise areas\n"
        "- Key technical and soft skills\n"
        "- Notable achievements or certifications\n"
        "Use third person. Only include facts from the documents."
    ),
    "skills_matrix": (
        "Create a structured skills breakdown organized by category:\n"
        "- Technical Skills\n"
        "- Soft Skills\n"
        "- Certifications\n"
        "- Domain Expertise\n"
        "List only skills evidenced in the source documents."
    ),
    "candidate_comparison": (
        "Create a side-by-side comparison of the candidates covering:\n"
        "- Experience and qualifications\n"
        "- Key skills (overlapping and unique)\n"
        "- Strengths per candidate\n"
        "Base all comparisons on documented evidence only."
    ),
    "interview_prep": (
        "Generate interview preparation materials including:\n"
        "- Background summary of the candidate\n"
        "- Suggested questions based on their experience\n"
        "- Key areas to explore based on resume gaps or highlights\n"
        "Ground all suggestions in the source documents."
    ),
    # Invoice
    "invoice_summary": (
        "Summarize the invoice(s) including:\n"
        "- Invoice number, date, and parties\n"
        "- Total amount and currency\n"
        "- Key line items\n"
        "- Payment terms if mentioned\n"
        "Use exact figures from the documents."
    ),
    "expense_report": (
        "Create an expense report consolidating all invoices:\n"
        "- Total expenses by category\n"
        "- Individual invoice breakdowns\n"
        "- Date range covered\n"
        "Use exact amounts from the source documents."
    ),
    "payment_reminder": (
        "Draft a professional payment reminder including:\n"
        "- Reference to specific invoice(s)\n"
        "- Amount due and original due date\n"
        "- Professional but firm tone\n"
        "Use only facts from the invoice documents."
    ),
    # Legal
    "contract_summary": (
        "Summarize the contract in plain language covering:\n"
        "- Parties involved\n"
        "- Key terms and conditions\n"
        "- Important dates and deadlines\n"
        "- Financial terms\n"
        "Use only information from the source document."
    ),
    "compliance_report": (
        "Generate a compliance assessment covering:\n"
        "- Identified compliance areas\n"
        "- Status of each requirement\n"
        "- Gaps or concerns\n"
        "Base all findings on document evidence."
    ),
    "risk_assessment": (
        "Analyze risks identified in the documents:\n"
        "- Risk categories and severity\n"
        "- Potential impact\n"
        "- Mitigation suggestions based on document content\n"
        "Only cite risks supported by the source material."
    ),
    # Medical
    "patient_summary": (
        "Create a structured patient summary including:\n"
        "- Patient demographics\n"
        "- Diagnoses and conditions\n"
        "- Current medications\n"
        "- Recent procedures or tests\n"
        "Use only information from the medical records."
    ),
    "medical_report": (
        "Format a medical report including:\n"
        "- Clinical findings\n"
        "- Diagnostic results\n"
        "- Treatment history\n"
        "- Follow-up recommendations from the documents\n"
        "Use only facts from the source documents."
    ),
    # Report
    "executive_summary": (
        "Write a concise executive summary (150-300 words) covering:\n"
        "- Key purpose of the document(s)\n"
        "- Main findings or conclusions\n"
        "- Critical numbers or metrics\n"
        "- Recommended actions if applicable"
    ),
    "key_findings": (
        "Extract and present key findings:\n"
        "- List each finding as a clear statement\n"
        "- Include supporting evidence\n"
        "- Note significance or implications\n"
        "Only include findings supported by the documents."
    ),
    "recommendations": (
        "Generate actionable recommendations based on document evidence:\n"
        "- Each recommendation should be specific and actionable\n"
        "- Include rationale from the source documents\n"
        "- Prioritize by impact if possible"
    ),
    # General
    "document_summary": (
        "Write a comprehensive summary of the document(s):\n"
        "- Main topics and themes\n"
        "- Key information and data points\n"
        "- Important conclusions\n"
        "Capture the essence while staying grounded in source text."
    ),
    "key_points": (
        "Extract the most important points as bullet items:\n"
        "- Each point should be self-contained\n"
        "- Order by importance\n"
        "- Include only points supported by the documents"
    ),
    "faq_generation": (
        "Generate FAQ entries (question and answer pairs) based on the documents:\n"
        "- Questions should be natural and useful\n"
        "- Answers must be grounded in document evidence\n"
        "- Cover the most important topics"
    ),
    "action_items": (
        "Extract actionable items from the documents:\n"
        "- Each item should be specific and clear\n"
        "- Include context (who, what, when if available)\n"
        "- Only include items supported by document content"
    ),
    "talking_points": (
        "Generate key talking points:\n"
        "- Each point should be concise (1-2 sentences)\n"
        "- Include supporting evidence references\n"
        "- Order by relevance"
    ),
    "meeting_notes": (
        "Format meeting notes from the documents:\n"
        "- Date and participants (if available)\n"
        "- Key discussion points\n"
        "- Decisions made\n"
        "- Action items with owners\n"
        "Use only information from the source documents."
    ),
    # Cross-document
    "comparison_report": (
        "Create a detailed comparison across the documents:\n"
        "- Identify common themes and differences\n"
        "- Compare key metrics or data points\n"
        "- Attribute all information to specific source documents"
    ),
    "consolidated_summary": (
        "Create a unified summary combining information from all documents:\n"
        "- Synthesize related information\n"
        "- Note where documents agree or differ\n"
        "- Attribute key facts to source documents"
    ),
    "trend_analysis": (
        "Analyze trends and patterns across the documents:\n"
        "- Identify changes over time\n"
        "- Highlight significant patterns\n"
        "- Support each trend with specific document evidence"
    ),
}


# ---------------------------------------------------------------------------
# Grounding instructions (appended to all prompts)
# ---------------------------------------------------------------------------

_GROUNDING_SUFFIX = (
    "\n\n## CRITICAL RULES\n"
    "1. ONLY use information found in the provided source evidence.\n"
    "2. Do NOT invent, assume, or hallucinate any facts.\n"
    "3. If information is insufficient, say so clearly.\n"
    "4. When citing specific data (numbers, dates, names), ensure it appears in the evidence.\n"
)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


class ContentPromptBuilder:
    """Builds domain-specific LLM prompts for content generation."""

    def build_system_prompt(self, content_type: ContentType) -> str:
        """Build the system prompt for a content type."""
        domain_prompt = _DOMAIN_SYSTEM_PROMPTS.get(
            content_type.domain,
            _DOMAIN_SYSTEM_PROMPTS["general"],
        )
        return domain_prompt

    def build_generation_prompt(
        self,
        content_type: ContentType,
        facts: Dict[str, Any],
        evidence_text: str,
        query: str = "",
        extra_instructions: str = "",
    ) -> str:
        """Build the full generation prompt with facts, evidence, and instructions.

        Args:
            content_type: The content type to generate.
            facts: Extracted facts dict (keys match required/optional fields).
            evidence_text: Raw evidence text from source chunks.
            query: The original user query.
            extra_instructions: Additional user-provided instructions.

        Returns:
            Complete prompt string for LLM generation.
        """
        parts: List[str] = []

        # Task description
        parts.append(f"## Task: Generate {content_type.name}")
        if query:
            parts.append(f"User request: {query}")

        # Type-specific instructions
        instructions = _TYPE_INSTRUCTIONS.get(content_type.id, "")
        if instructions:
            parts.append(f"\n## Instructions\n{instructions}")

        if extra_instructions:
            parts.append(f"\n## Additional Instructions\n{extra_instructions}")

        # Extracted facts
        if facts:
            parts.append("\n## Extracted Facts")
            for key, value in facts.items():
                if isinstance(value, list):
                    if value:
                        parts.append(f"- {key}: {', '.join(str(v) for v in value)}")
                elif value:
                    parts.append(f"- {key}: {value}")

        # Evidence
        if evidence_text:
            # Truncate evidence to prevent context overflow
            max_evidence = 6144
            if len(evidence_text) > max_evidence:
                evidence_text = evidence_text[:max_evidence] + "\n[Evidence truncated]"
            parts.append(f"\n## Source Evidence\n{evidence_text}")

        # Grounding rules
        parts.append(_GROUNDING_SUFFIX)

        return "\n".join(parts)

    def build_fact_extraction_prompt(
        self,
        content_type: ContentType,
        evidence_text: str,
    ) -> str:
        """Build a prompt for extracting facts from evidence before generation.

        Used in the EXTRACT step to pull structured facts from raw chunks.
        """
        all_fields = list(content_type.required_fields) + list(content_type.optional_fields)
        if not all_fields:
            all_fields = ["key_information", "entities", "dates", "numbers"]

        fields_str = ", ".join(all_fields)
        return (
            f"Extract the following information from the provided text:\n"
            f"Fields: {fields_str}\n\n"
            f"Return a JSON object with these fields as keys. "
            f"Use null for fields not found in the text. "
            f"Use arrays for fields that may have multiple values.\n\n"
            f"Text:\n{evidence_text[:4096]}"
        )
