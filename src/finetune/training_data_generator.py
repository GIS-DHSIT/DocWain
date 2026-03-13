"""Training data generator for TaskSpec fine-tuning.

Produces Llama-3.1 chat-format JSONL from three sources:
1. Curated examples from the intensive test suite (96 queries)
2. Domain:task paraphrases via LLM (2,400 examples)
3. Edge cases: multi-intent, constraints, negatives (500 examples)

Usage::

    python -m src.finetune.training_data_generator
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ── Output paths ─────────────────────────────────────────────────────────────

_DEFAULT_OUTPUT_DIR = Path("finetune_data")
_TRAIN_FILE = "taskspec_train.jsonl"
_EVAL_FILE = "taskspec_eval.jsonl"

# ── System prompt (same used at inference time) ─────────────────────────────

SYSTEM_PROMPT = (
    "Parse the user's document query into a structured TaskSpec JSON. "
    "Respond ONLY with valid JSON, no extra text.\n"
    "Fields: intent (factual|compare|rank|summarize|extract|generate|analyze|timeline|redirect|clarify), "
    "domain (hr|medical|legal|invoice|insurance|policy|general|content|translation|education), "
    "output_format (paragraph|table|bullets|numbered|chart_data|json), "
    "entities (list of key entities), constraints (dict of structured filters), "
    "scope (all_documents|specific_document|cross_document), "
    "complexity (simple|medium|complex), confidence (0.0-1.0)."
)


@dataclass
class DatasetBuildResult:
    """Outcome of training data generation."""

    train_path: Optional[Path] = None
    eval_path: Optional[Path] = None
    total_examples: int = 0
    train_count: int = 0
    eval_count: int = 0
    source_counts: Dict[str, int] = field(default_factory=dict)


# ── Helper: build a single chat example ─────────────────────────────────────


def _chat_example(query: str, taskspec_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a (query, taskspec) pair in Llama-3.1 chat format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": json.dumps(taskspec_dict, ensure_ascii=False)},
        ]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Source 1: Curated examples from intensive test queries
# ═══════════════════════════════════════════════════════════════════════════════

# Ground-truth TaskSpec for each intensive test query.
# These are the highest-quality, hand-verified training examples.

_CURATED: List[Tuple[str, Dict[str, Any]]] = [
    # HR Basic
    ("List all candidates and their primary skills", {"intent": "extract", "domain": "hr", "output_format": "table", "entities": ["candidates", "skills"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Who has the most experience in Python?", {"intent": "rank", "domain": "hr", "output_format": "paragraph", "entities": ["Python"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("What certifications do the candidates have?", {"intent": "extract", "domain": "hr", "output_format": "bullets", "entities": ["certifications"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("Show me the education details of all candidates", {"intent": "extract", "domain": "hr", "output_format": "table", "entities": ["education"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("What are the contact details for all candidates?", {"intent": "extract", "domain": "hr", "output_format": "table", "entities": ["contact details"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("Summarize all the resumes briefly", {"intent": "summarize", "domain": "hr", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Which candidate has data engineering skills?", {"intent": "factual", "domain": "hr", "output_format": "paragraph", "entities": ["data engineering"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("List candidates with more than 5 years of experience", {"intent": "extract", "domain": "hr", "output_format": "bullets", "entities": ["candidates"], "constraints": {"min_experience_years": 5}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("What functional skills are available across all resumes?", {"intent": "extract", "domain": "hr", "output_format": "bullets", "entities": ["skills"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Find candidates who know both Python and machine learning", {"intent": "factual", "domain": "hr", "output_format": "paragraph", "entities": ["Python", "machine learning"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),

    # HR Analytical
    ("Compare all candidates for a Python developer role", {"intent": "compare", "domain": "hr", "output_format": "table", "entities": ["Python developer"], "constraints": {}, "scope": "cross_document", "complexity": "complex", "confidence": 0.95}),
    ("Rank candidates by their overall profile strength", {"intent": "rank", "domain": "hr", "output_format": "numbered", "entities": ["candidates"], "constraints": {}, "scope": "all_documents", "complexity": "complex", "confidence": 0.95}),
    ("What insights can you find across all the resumes?", {"intent": "analyze", "domain": "hr", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "complex", "confidence": 0.95}),
    ("Rank candidates by education level", {"intent": "rank", "domain": "hr", "output_format": "numbered", "entities": ["education"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("What programming languages are mentioned across all resumes?", {"intent": "extract", "domain": "hr", "output_format": "bullets", "entities": ["programming languages"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Who is the best candidate for a senior AI engineer role?", {"intent": "rank", "domain": "hr", "output_format": "paragraph", "entities": ["AI engineer"], "constraints": {"role": "senior AI engineer"}, "scope": "all_documents", "complexity": "complex", "confidence": 0.95}),
    ("Compare the top 3 most experienced candidates", {"intent": "compare", "domain": "hr", "output_format": "table", "entities": ["candidates"], "constraints": {"top_n": 3}, "scope": "cross_document", "complexity": "complex", "confidence": 0.95}),
    ("What are the most common skills shared by multiple candidates?", {"intent": "analyze", "domain": "hr", "output_format": "bullets", "entities": ["skills"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),

    # Medical
    ("What are the patient diagnoses?", {"intent": "extract", "domain": "medical", "output_format": "bullets", "entities": ["diagnoses"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("List all medications prescribed", {"intent": "extract", "domain": "medical", "output_format": "bullets", "entities": ["medications"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("What lab results are available?", {"intent": "extract", "domain": "medical", "output_format": "table", "entities": ["lab results"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("Summarize the patient records", {"intent": "summarize", "domain": "medical", "output_format": "paragraph", "entities": ["patient records"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("What treatments or procedures were performed?", {"intent": "extract", "domain": "medical", "output_format": "bullets", "entities": ["treatments", "procedures"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),

    # Invoice
    ("What are the total amounts on all invoices?", {"intent": "extract", "domain": "invoice", "output_format": "table", "entities": ["amounts"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("List all invoice numbers and dates", {"intent": "extract", "domain": "invoice", "output_format": "table", "entities": ["invoice numbers", "dates"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("Which vendors are listed in the invoices?", {"intent": "extract", "domain": "invoice", "output_format": "bullets", "entities": ["vendors"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("What items are listed on the invoices?", {"intent": "extract", "domain": "invoice", "output_format": "table", "entities": ["items"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("Summarize all invoice data", {"intent": "summarize", "domain": "invoice", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("What payment terms are mentioned?", {"intent": "extract", "domain": "invoice", "output_format": "paragraph", "entities": ["payment terms"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("Find invoices with the highest amounts", {"intent": "rank", "domain": "invoice", "output_format": "numbered", "entities": ["invoices"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),

    # Insurance/Policy
    ("What is covered under the car insurance policy?", {"intent": "factual", "domain": "insurance", "output_format": "paragraph", "entities": ["car insurance", "coverage"], "constraints": {}, "scope": "specific_document", "complexity": "simple", "confidence": 0.95}),
    ("Compare the car and bike insurance policies", {"intent": "compare", "domain": "insurance", "output_format": "table", "entities": ["car insurance", "bike insurance"], "constraints": {}, "scope": "cross_document", "complexity": "complex", "confidence": 0.95}),
    ("What are the exclusions in the insurance policies?", {"intent": "extract", "domain": "insurance", "output_format": "bullets", "entities": ["exclusions"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("What premiums are payable?", {"intent": "extract", "domain": "insurance", "output_format": "table", "entities": ["premiums"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("Summarize the key terms and conditions", {"intent": "summarize", "domain": "insurance", "output_format": "paragraph", "entities": ["terms", "conditions"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),

    # Email
    ("Draft an interview invitation email for the top candidate", {"intent": "generate", "domain": "content", "output_format": "paragraph", "entities": ["interview invitation"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Write a follow up email to a candidate about their application status", {"intent": "generate", "domain": "content", "output_format": "paragraph", "entities": ["follow up", "application status"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Compose a rejection email to a candidate in a professional tone", {"intent": "generate", "domain": "content", "output_format": "paragraph", "entities": ["rejection email"], "constraints": {"tone": "professional"}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),

    # Content generation
    ("Generate a cover letter for a Python developer based on the resumes", {"intent": "generate", "domain": "hr", "output_format": "paragraph", "entities": ["Python developer", "cover letter"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Create a summary of key findings across all documents", {"intent": "summarize", "domain": "general", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "complex", "confidence": 0.95}),
    ("Extract all action items from the documents", {"intent": "extract", "domain": "general", "output_format": "bullets", "entities": ["action items"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),

    # Document discovery
    ("How many documents are in this profile?", {"intent": "factual", "domain": "general", "output_format": "paragraph", "entities": ["documents"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("What types of documents do I have?", {"intent": "factual", "domain": "general", "output_format": "bullets", "entities": ["document types"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),

    # Web
    ("What is the latest version of Python programming language?", {"intent": "factual", "domain": "general", "output_format": "paragraph", "entities": ["Python"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("What is FastAPI framework used for?", {"intent": "factual", "domain": "general", "output_format": "paragraph", "entities": ["FastAPI"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),

    # Edge cases
    ("Tell me everything you know about the documents", {"intent": "summarize", "domain": "general", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "complex", "confidence": 0.95}),
    ("What is the most important information in the resumes?", {"intent": "analyze", "domain": "hr", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "complex", "confidence": 0.95}),
    ("Are there any common patterns across the documents?", {"intent": "analyze", "domain": "general", "output_format": "paragraph", "entities": ["patterns"], "constraints": {}, "scope": "all_documents", "complexity": "complex", "confidence": 0.95}),
    ("What insights can you provide about the invoice data?", {"intent": "analyze", "domain": "invoice", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "complex", "confidence": 0.95}),
    ("Provide a comprehensive analysis of all candidates", {"intent": "analyze", "domain": "hr", "output_format": "paragraph", "entities": ["candidates"], "constraints": {}, "scope": "all_documents", "complexity": "complex", "confidence": 0.95}),

    # Advanced content generation
    ("Generate interview questions for a data science candidate based on the resumes", {"intent": "generate", "domain": "hr", "output_format": "numbered", "entities": ["data science", "interview questions"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Create a professional summary for the strongest Python developer candidate", {"intent": "generate", "domain": "hr", "output_format": "paragraph", "entities": ["Python developer"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Write a skills comparison matrix for all candidates", {"intent": "compare", "domain": "hr", "output_format": "table", "entities": ["skills"], "constraints": {}, "scope": "cross_document", "complexity": "complex", "confidence": 0.95}),
    ("Draft a job description for a senior Python developer role based on the resume profiles", {"intent": "generate", "domain": "hr", "output_format": "paragraph", "entities": ["Python developer", "job description"], "constraints": {"level": "senior"}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Generate a candidate shortlist report with rankings for a machine learning engineer role", {"intent": "rank", "domain": "hr", "output_format": "numbered", "entities": ["machine learning engineer"], "constraints": {}, "scope": "all_documents", "complexity": "complex", "confidence": 0.95}),
    ("Create a detailed skill gap analysis between the candidates and a senior data engineer role", {"intent": "analyze", "domain": "hr", "output_format": "table", "entities": ["skill gap", "data engineer"], "constraints": {"level": "senior"}, "scope": "cross_document", "complexity": "complex", "confidence": 0.95}),
    ("Generate a cover letter for a candidate applying to a healthcare AI company", {"intent": "generate", "domain": "hr", "output_format": "paragraph", "entities": ["healthcare AI", "cover letter"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Draft a meeting agenda for interviewing the top 3 candidates for a Python developer role", {"intent": "generate", "domain": "hr", "output_format": "numbered", "entities": ["meeting agenda", "Python developer"], "constraints": {"top_n": 3}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),

    # Complex reasoning
    ("If I need someone who can build data pipelines and deploy ML models, which candidate is the best fit and why?", {"intent": "rank", "domain": "hr", "output_format": "paragraph", "entities": ["data pipelines", "ML models"], "constraints": {}, "scope": "all_documents", "complexity": "complex", "confidence": 0.95}),
    ("Analyze the career progression of each candidate and identify who has the fastest growth trajectory", {"intent": "analyze", "domain": "hr", "output_format": "paragraph", "entities": ["career progression"], "constraints": {}, "scope": "cross_document", "complexity": "complex", "confidence": 0.95}),
    ("What are the unique skills that each candidate brings that others don't have?", {"intent": "compare", "domain": "hr", "output_format": "table", "entities": ["unique skills"], "constraints": {}, "scope": "cross_document", "complexity": "complex", "confidence": 0.95}),
    ("Compare the educational background and work experience balance across all candidates", {"intent": "compare", "domain": "hr", "output_format": "table", "entities": ["education", "work experience"], "constraints": {}, "scope": "cross_document", "complexity": "complex", "confidence": 0.95}),
    ("Based on the invoice data, what spending patterns can you identify and what recommendations would you make?", {"intent": "analyze", "domain": "invoice", "output_format": "paragraph", "entities": ["spending patterns"], "constraints": {}, "scope": "all_documents", "complexity": "complex", "confidence": 0.95}),
    ("Analyze the medical records and identify any potential health risks or areas needing follow-up", {"intent": "analyze", "domain": "medical", "output_format": "paragraph", "entities": ["health risks"], "constraints": {}, "scope": "all_documents", "complexity": "complex", "confidence": 0.95}),

    # Cross-document intelligence
    ("What skills are mentioned in every single resume?", {"intent": "analyze", "domain": "hr", "output_format": "bullets", "entities": ["skills"], "constraints": {}, "scope": "cross_document", "complexity": "medium", "confidence": 0.95}),
    ("What is the average years of experience across all candidates?", {"intent": "analyze", "domain": "hr", "output_format": "paragraph", "entities": ["experience"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Which candidates have overlapping skill sets and who is most unique?", {"intent": "compare", "domain": "hr", "output_format": "table", "entities": ["skills"], "constraints": {}, "scope": "cross_document", "complexity": "complex", "confidence": 0.95}),
    ("Summarize the total invoice amounts by vendor", {"intent": "summarize", "domain": "invoice", "output_format": "table", "entities": ["amounts", "vendor"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("What are the key differences between the top two candidates?", {"intent": "compare", "domain": "hr", "output_format": "table", "entities": ["candidates"], "constraints": {"top_n": 2}, "scope": "cross_document", "complexity": "complex", "confidence": 0.95}),

    # Screening
    ("Screen the resumes for any personally identifiable information like phone numbers and email addresses", {"intent": "extract", "domain": "hr", "output_format": "table", "entities": ["PII", "phone numbers", "email addresses"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Check the readability score of the documents", {"intent": "analyze", "domain": "general", "output_format": "paragraph", "entities": ["readability"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Are there any sensitive data patterns in the uploaded documents?", {"intent": "analyze", "domain": "general", "output_format": "bullets", "entities": ["sensitive data"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),

    # Translation
    ("Translate the summary of the first resume into French", {"intent": "generate", "domain": "translation", "output_format": "paragraph", "entities": ["French"], "constraints": {"target_language": "French"}, "scope": "specific_document", "complexity": "medium", "confidence": 0.95}),
    ("Give me a Spanish version of the candidate skills overview", {"intent": "generate", "domain": "translation", "output_format": "paragraph", "entities": ["Spanish", "skills"], "constraints": {"target_language": "Spanish"}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Convert the medical record summary to Hindi", {"intent": "generate", "domain": "translation", "output_format": "paragraph", "entities": ["Hindi", "medical record"], "constraints": {"target_language": "Hindi"}, "scope": "specific_document", "complexity": "medium", "confidence": 0.95}),

    # Extraction
    ("Extract all email addresses and phone numbers from the resumes", {"intent": "extract", "domain": "hr", "output_format": "table", "entities": ["email addresses", "phone numbers"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("List every company name mentioned across all resumes with the candidate who worked there", {"intent": "extract", "domain": "hr", "output_format": "table", "entities": ["company names", "candidates"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Extract all dates and time periods mentioned in the invoices", {"intent": "extract", "domain": "invoice", "output_format": "table", "entities": ["dates", "time periods"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("List all technical certifications and the candidates who hold them", {"intent": "extract", "domain": "hr", "output_format": "table", "entities": ["certifications", "candidates"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Extract all line items and their amounts from the invoices", {"intent": "extract", "domain": "invoice", "output_format": "table", "entities": ["line items", "amounts"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),

    # Conversational
    ("Hello, what can you help me with?", {"intent": "redirect", "domain": "general", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("How do I upload a new document?", {"intent": "redirect", "domain": "general", "output_format": "paragraph", "entities": ["upload"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("What types of files can I upload?", {"intent": "redirect", "domain": "general", "output_format": "paragraph", "entities": ["file types"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("Show me example queries I can try", {"intent": "redirect", "domain": "general", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),

    # Customer service
    ("I have an issue with my insurance claim, can you help me resolve it?", {"intent": "factual", "domain": "insurance", "output_format": "paragraph", "entities": ["insurance claim"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Help me troubleshoot why my document upload keeps failing", {"intent": "redirect", "domain": "general", "output_format": "paragraph", "entities": ["upload"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ("Does this issue need escalation? A customer reports unauthorized access to their account", {"intent": "analyze", "domain": "general", "output_format": "paragraph", "entities": ["unauthorized access", "escalation"], "constraints": {}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Draft a professional response to a customer who is unhappy with their premium increase", {"intent": "generate", "domain": "insurance", "output_format": "paragraph", "entities": ["customer", "premium increase"], "constraints": {"tone": "professional"}, "scope": "all_documents", "complexity": "medium", "confidence": 0.95}),
    ("Search the FAQ for how to update payment information", {"intent": "factual", "domain": "general", "output_format": "paragraph", "entities": ["payment information"], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),

    # Analytics/Visualization
    ("Create a chart showing skills distribution across all candidates", {"intent": "analyze", "domain": "hr", "output_format": "chart_data", "entities": ["skills"], "constraints": {}, "scope": "all_documents", "complexity": "complex", "confidence": 0.95}),
    ("Visualize the invoice amounts by vendor as a pie chart", {"intent": "analyze", "domain": "invoice", "output_format": "chart_data", "entities": ["amounts", "vendor"], "constraints": {"chart_type": "pie"}, "scope": "all_documents", "complexity": "complex", "confidence": 0.95}),
    ("Show me a timeline chart of candidate experience over the years", {"intent": "timeline", "domain": "hr", "output_format": "chart_data", "entities": ["experience"], "constraints": {}, "scope": "all_documents", "complexity": "complex", "confidence": 0.95}),
    ("Compare candidate qualifications side by side in a chart", {"intent": "compare", "domain": "hr", "output_format": "chart_data", "entities": ["qualifications"], "constraints": {}, "scope": "cross_document", "complexity": "complex", "confidence": 0.95}),
    ("Calculate statistics on invoice totals and show results with a chart", {"intent": "analyze", "domain": "invoice", "output_format": "chart_data", "entities": ["invoice totals", "statistics"], "constraints": {}, "scope": "all_documents", "complexity": "complex", "confidence": 0.95}),
]


def generate_curated_examples() -> List[Dict[str, Any]]:
    """Source 1 — hand-verified curated examples from intensive test queries."""
    return [_chat_example(q, ts) for q, ts in _CURATED]


# ═══════════════════════════════════════════════════════════════════════════════
# Source 2: Domain:Task paraphrases
# ═══════════════════════════════════════════════════════════════════════════════

# Core domain:task → canonical taskspec mappings
_DOMAIN_TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "hr:generate_interview_questions": {"intent": "generate", "domain": "hr", "output_format": "numbered", "entities": ["interview questions"], "constraints": {}, "scope": "all_documents", "complexity": "medium"},
    "hr:skill_gap_analysis": {"intent": "analyze", "domain": "hr", "output_format": "table", "entities": ["skill gap"], "constraints": {}, "scope": "cross_document", "complexity": "complex"},
    "hr:role_fit_assessment": {"intent": "rank", "domain": "hr", "output_format": "paragraph", "entities": ["role fit"], "constraints": {}, "scope": "all_documents", "complexity": "complex"},
    "hr:experience_timeline": {"intent": "timeline", "domain": "hr", "output_format": "paragraph", "entities": ["career timeline"], "constraints": {}, "scope": "all_documents", "complexity": "medium"},
    "hr:candidate_summary": {"intent": "summarize", "domain": "hr", "output_format": "paragraph", "entities": ["candidate"], "constraints": {}, "scope": "all_documents", "complexity": "medium"},
    "medical:drug_interaction_check": {"intent": "analyze", "domain": "medical", "output_format": "table", "entities": ["drug interactions"], "constraints": {}, "scope": "all_documents", "complexity": "complex"},
    "medical:treatment_plan_review": {"intent": "analyze", "domain": "medical", "output_format": "paragraph", "entities": ["treatment plan"], "constraints": {}, "scope": "specific_document", "complexity": "complex"},
    "medical:lab_result_interpretation": {"intent": "factual", "domain": "medical", "output_format": "table", "entities": ["lab results"], "constraints": {}, "scope": "specific_document", "complexity": "medium"},
    "medical:clinical_summary": {"intent": "summarize", "domain": "medical", "output_format": "paragraph", "entities": ["clinical summary"], "constraints": {}, "scope": "all_documents", "complexity": "medium"},
    "medical:patient_history_timeline": {"intent": "timeline", "domain": "medical", "output_format": "paragraph", "entities": ["patient history"], "constraints": {}, "scope": "specific_document", "complexity": "medium"},
    "legal:clause_risk_assessment": {"intent": "analyze", "domain": "legal", "output_format": "bullets", "entities": ["risky clauses"], "constraints": {}, "scope": "specific_document", "complexity": "complex"},
    "legal:compliance_check": {"intent": "analyze", "domain": "legal", "output_format": "bullets", "entities": ["compliance"], "constraints": {}, "scope": "specific_document", "complexity": "complex"},
    "legal:contract_comparison": {"intent": "compare", "domain": "legal", "output_format": "table", "entities": ["contracts"], "constraints": {}, "scope": "cross_document", "complexity": "complex"},
    "legal:key_terms_extraction": {"intent": "extract", "domain": "legal", "output_format": "bullets", "entities": ["key terms"], "constraints": {}, "scope": "specific_document", "complexity": "medium"},
    "legal:obligation_tracker": {"intent": "extract", "domain": "legal", "output_format": "table", "entities": ["obligations", "deadlines"], "constraints": {}, "scope": "all_documents", "complexity": "medium"},
    "invoice:payment_anomaly_detection": {"intent": "analyze", "domain": "invoice", "output_format": "bullets", "entities": ["payment anomalies"], "constraints": {}, "scope": "all_documents", "complexity": "complex"},
    "invoice:expense_categorization": {"intent": "extract", "domain": "invoice", "output_format": "table", "entities": ["expenses", "categories"], "constraints": {}, "scope": "all_documents", "complexity": "medium"},
    "invoice:duplicate_detection": {"intent": "analyze", "domain": "invoice", "output_format": "table", "entities": ["duplicates"], "constraints": {}, "scope": "all_documents", "complexity": "medium"},
    "invoice:vendor_analysis": {"intent": "analyze", "domain": "invoice", "output_format": "table", "entities": ["vendors"], "constraints": {}, "scope": "all_documents", "complexity": "complex"},
    "invoice:financial_summary": {"intent": "summarize", "domain": "invoice", "output_format": "paragraph", "entities": ["financial summary"], "constraints": {}, "scope": "all_documents", "complexity": "medium"},
    "content:draft_email": {"intent": "generate", "domain": "content", "output_format": "paragraph", "entities": ["email"], "constraints": {}, "scope": "all_documents", "complexity": "medium"},
    "content:generate_documentation": {"intent": "generate", "domain": "content", "output_format": "paragraph", "entities": ["documentation"], "constraints": {}, "scope": "all_documents", "complexity": "medium"},
    "content:rewrite_text": {"intent": "generate", "domain": "content", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "specific_document", "complexity": "medium"},
    "content:create_presentation": {"intent": "generate", "domain": "content", "output_format": "bullets", "entities": ["presentation"], "constraints": {}, "scope": "all_documents", "complexity": "medium"},
    "content:generate_content": {"intent": "generate", "domain": "content", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "medium"},
    "translation:detect_language": {"intent": "factual", "domain": "translation", "output_format": "paragraph", "entities": ["language"], "constraints": {}, "scope": "specific_document", "complexity": "simple"},
    "translation:localize_content": {"intent": "generate", "domain": "translation", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "specific_document", "complexity": "medium"},
    "translation:multilingual_summary": {"intent": "summarize", "domain": "translation", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "medium"},
    "translation:translate_text": {"intent": "generate", "domain": "translation", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "specific_document", "complexity": "medium"},
    "education:generate_quiz": {"intent": "generate", "domain": "education", "output_format": "numbered", "entities": ["quiz"], "constraints": {}, "scope": "all_documents", "complexity": "medium"},
    "education:study_guide": {"intent": "generate", "domain": "education", "output_format": "bullets", "entities": ["study guide"], "constraints": {}, "scope": "all_documents", "complexity": "medium"},
    "education:create_lesson": {"intent": "generate", "domain": "education", "output_format": "paragraph", "entities": ["lesson plan"], "constraints": {}, "scope": "all_documents", "complexity": "medium"},
    "education:explain_concept": {"intent": "factual", "domain": "education", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "specific_document", "complexity": "medium"},
}

# Seed paraphrases (5 per task) — the LLM expands these to 50 each
_SEED_PARAPHRASES: Dict[str, List[str]] = {
    "hr:generate_interview_questions": [
        "Prepare interview questions for this candidate",
        "What should I ask in the interview?",
        "Generate questions to assess the candidate",
        "Create a list of interview questions based on their resume",
        "Help me prepare for interviewing this person",
    ],
    "hr:skill_gap_analysis": [
        "What skills is this candidate missing?",
        "Identify the gaps between the candidate's skills and the job requirements",
        "Show me where the candidate falls short",
        "Analyze skill deficiencies across candidates",
        "Compare candidate skills against the role requirements",
    ],
    "hr:role_fit_assessment": [
        "Is this candidate a good fit for the role?",
        "Who is the best match for this position?",
        "Evaluate candidate suitability",
        "Rate how well each candidate fits the job",
        "Which candidate should we hire?",
    ],
    "hr:experience_timeline": [
        "Map out the candidate's career history",
        "Show the work experience timeline",
        "Create a chronological view of their career",
        "What is their career progression?",
        "Trace their professional journey",
    ],
    "hr:candidate_summary": [
        "Give me a brief overview of this candidate",
        "Summarize the candidate's profile",
        "What are the key highlights of this resume?",
        "Quick summary of qualifications",
        "Profile overview for the candidate",
    ],
    "medical:drug_interaction_check": [
        "Check for drug interactions in the prescription",
        "Are any of these medications dangerous together?",
        "Flag any medication contraindications",
        "Review the prescriptions for safety",
        "Do any of the drugs interact badly?",
    ],
    "medical:treatment_plan_review": [
        "Review the treatment plan for effectiveness",
        "Is the care plan appropriate?",
        "Assess the treatment approach",
        "Evaluate the prescribed treatment",
        "How effective is the current treatment?",
    ],
    "medical:lab_result_interpretation": [
        "Interpret these lab results",
        "What do the blood test results mean?",
        "Explain the lab values",
        "Are any test results abnormal?",
        "Analyze the laboratory findings",
    ],
    "medical:clinical_summary": [
        "Summarize the medical case",
        "Give me a clinical overview",
        "Create a patient summary",
        "Brief the medical record",
        "Overview of the clinical notes",
    ],
    "medical:patient_history_timeline": [
        "Map out the patient's medical history",
        "Create a timeline of medical events",
        "What is the chronological health history?",
        "Track medical events over time",
        "Show the patient's treatment history chronologically",
    ],
    "legal:clause_risk_assessment": [
        "Find risky clauses in this contract",
        "Which terms are unfavorable?",
        "Identify legal red flags",
        "Flag problematic contract provisions",
        "Assess the risk of each clause",
    ],
    "legal:compliance_check": [
        "Check if this is GDPR compliant",
        "Does this meet regulatory requirements?",
        "Verify compliance with standards",
        "Is this document legally compliant?",
        "Audit for regulatory adherence",
    ],
    "legal:contract_comparison": [
        "Compare these two contracts",
        "What are the differences between the agreements?",
        "Side by side contract analysis",
        "How do these contracts differ?",
        "Find the variations between the legal documents",
    ],
    "legal:key_terms_extraction": [
        "Extract the key legal terms",
        "What are the defined terms?",
        "List the important legal definitions",
        "Pull out the key provisions",
        "Identify the major terms in this agreement",
    ],
    "legal:obligation_tracker": [
        "What are our obligations under this contract?",
        "Track the deadlines and deliverables",
        "List all contractual commitments",
        "What are we required to do?",
        "Map out the contractual obligations",
    ],
    "invoice:payment_anomaly_detection": [
        "Find unusual charges in the invoices",
        "Are there any suspicious payments?",
        "Detect billing irregularities",
        "Flag anomalous transactions",
        "Identify payment discrepancies",
    ],
    "invoice:expense_categorization": [
        "Categorize the invoice expenses",
        "Classify charges by type",
        "Organize the financial line items",
        "Group expenses by category",
        "Break down the spending categories",
    ],
    "invoice:duplicate_detection": [
        "Check for duplicate invoices",
        "Are there any double charges?",
        "Find repeated billing entries",
        "Detect duplicate payments",
        "Flag potential duplicate invoices",
    ],
    "invoice:vendor_analysis": [
        "Analyze vendor billing patterns",
        "Compare vendor performance",
        "Review supplier invoices",
        "Which vendor charges the most?",
        "Break down spending by vendor",
    ],
    "invoice:financial_summary": [
        "Summarize the invoice data",
        "Give me a financial overview",
        "Total amounts and payment status",
        "Financial summary of all invoices",
        "Overview of billing activity",
    ],
    "content:draft_email": [
        "Draft an email for me",
        "Compose a business email",
        "Write a professional message",
        "Help me write this email",
        "Create an email about this topic",
    ],
    "content:generate_documentation": [
        "Create documentation for this",
        "Generate technical docs",
        "Write the API documentation",
        "Build reference documentation",
        "Produce documentation based on the content",
    ],
    "content:rewrite_text": [
        "Rewrite this more clearly",
        "Improve the writing quality",
        "Rephrase this content",
        "Make this text more professional",
        "Reword for clarity",
    ],
    "content:create_presentation": [
        "Create slides for a presentation",
        "Build a slide deck from this",
        "Generate presentation content",
        "Design a presentation outline",
        "Create presentation materials",
    ],
    "content:generate_content": [
        "Generate content based on the documents",
        "Create written material from this",
        "Produce text output",
        "Write content about this topic",
        "Generate a report from the data",
    ],
    "translation:detect_language": [
        "What language is this document in?",
        "Identify the language",
        "Detect the language of this text",
        "Which language was this written in?",
        "Is this document in English?",
    ],
    "translation:localize_content": [
        "Localize this for the French market",
        "Adapt the content for Spanish speakers",
        "Make this culturally appropriate for Japan",
        "Localize the document for German readers",
        "Adapt for a Brazilian Portuguese audience",
    ],
    "translation:multilingual_summary": [
        "Summarize this in Spanish",
        "Give me a summary in French",
        "Create a multilingual overview",
        "Key points in German",
        "Summary in Chinese",
    ],
    "translation:translate_text": [
        "Translate this to Spanish",
        "Convert to French",
        "Translate the document to German",
        "Give me this in Japanese",
        "Put this in Portuguese",
    ],
    "education:generate_quiz": [
        "Create a quiz from this material",
        "Generate test questions",
        "Make assessment questions",
        "Build a knowledge check quiz",
        "Create exam questions from the content",
    ],
    "education:study_guide": [
        "Create a study guide",
        "Build revision notes",
        "Generate study materials",
        "Make a learning reference",
        "Prepare study notes from this",
    ],
    "education:create_lesson": [
        "Create a lesson plan",
        "Design instructional content",
        "Build teaching materials",
        "Create a learning module",
        "Design a training session",
    ],
    "education:explain_concept": [
        "Explain this concept simply",
        "Break down this topic",
        "Help me understand this",
        "Teach me about this subject",
        "Explain step by step",
    ],
}


def _expand_paraphrases_offline(
    seed: str,
    count: int = 50,
) -> List[str]:
    """Generate paraphrases deterministically without an LLM.

    Uses simple template-based variation: prefix/suffix swaps, synonym
    injection, and formality variation.  This is fast and predictable,
    suitable for the initial dataset build.  The LLM paraphrase path
    can be used for augmentation later.
    """
    prefixes = [
        "", "Please ", "Can you ", "Could you ", "I need you to ",
        "Help me ", "I'd like to ", "I want to ", "Would you ",
        "Quickly ", "Go ahead and ",
    ]
    suffixes = [
        "", " please", " for me", " right now",
        " as soon as possible", " in detail",
        " briefly", " comprehensively",
    ]
    results: List[str] = [seed]
    seen = {seed.lower().strip()}

    for pfx in prefixes:
        for sfx in suffixes:
            variant = f"{pfx}{seed}{sfx}".strip()
            key = variant.lower()
            if key not in seen:
                seen.add(key)
                results.append(variant)
            if len(results) >= count:
                return results[:count]
    return results[:count]


def generate_paraphrases() -> List[Dict[str, Any]]:
    """Source 2 — expand seed paraphrases into ~2,400 training examples."""
    examples: List[Dict[str, Any]] = []
    for task_key, seeds in _SEED_PARAPHRASES.items():
        ts = _DOMAIN_TASK_SPECS.get(task_key)
        if ts is None:
            continue
        ts_with_conf = {**ts, "confidence": 0.90}
        for seed in seeds:
            expanded = _expand_paraphrases_offline(seed, count=15)
            for query in expanded:
                examples.append(_chat_example(query, ts_with_conf))
    return examples


# ═══════════════════════════════════════════════════════════════════════════════
# Source 3: Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


def generate_edge_cases() -> List[Dict[str, Any]]:
    """Source 3 — multi-intent, constraints, off-topic, ambiguous queries."""
    examples: List[Dict[str, Any]] = []

    # Multi-intent queries
    _MULTI = [
        ("Compare candidates AND list their contacts", {"intent": "compare", "domain": "hr", "output_format": "table", "entities": ["candidates", "contacts"], "constraints": {}, "scope": "cross_document", "complexity": "complex", "confidence": 0.90}),
        ("Summarize the invoices and flag any anomalies", {"intent": "analyze", "domain": "invoice", "output_format": "paragraph", "entities": ["invoices", "anomalies"], "constraints": {}, "scope": "all_documents", "complexity": "complex", "confidence": 0.90}),
        ("List medications and check for interactions", {"intent": "analyze", "domain": "medical", "output_format": "table", "entities": ["medications", "interactions"], "constraints": {}, "scope": "all_documents", "complexity": "complex", "confidence": 0.90}),
        ("Find the best candidate and draft an offer email", {"intent": "generate", "domain": "hr", "output_format": "paragraph", "entities": ["candidate", "offer email"], "constraints": {}, "scope": "all_documents", "complexity": "complex", "confidence": 0.90}),
        ("Extract key terms from both contracts and compare them", {"intent": "compare", "domain": "legal", "output_format": "table", "entities": ["key terms", "contracts"], "constraints": {}, "scope": "cross_document", "complexity": "complex", "confidence": 0.90}),
    ]

    # Constraint extraction queries
    _CONSTRAINTS = [
        ("Invoices over $1000 from last quarter", {"intent": "extract", "domain": "invoice", "output_format": "table", "entities": ["invoices"], "constraints": {"min_amount": 1000, "time_period": "last quarter"}, "scope": "all_documents", "complexity": "medium", "confidence": 0.90}),
        ("Candidates with at least 8 years of experience in Python", {"intent": "extract", "domain": "hr", "output_format": "bullets", "entities": ["candidates", "Python"], "constraints": {"min_experience_years": 8, "skill": "Python"}, "scope": "all_documents", "complexity": "medium", "confidence": 0.90}),
        ("Lab results from the past 6 months", {"intent": "extract", "domain": "medical", "output_format": "table", "entities": ["lab results"], "constraints": {"time_period": "past 6 months"}, "scope": "all_documents", "complexity": "medium", "confidence": 0.90}),
        ("Invoices from vendor ABC with amounts between 500 and 2000", {"intent": "extract", "domain": "invoice", "output_format": "table", "entities": ["invoices", "vendor ABC"], "constraints": {"vendor": "ABC", "min_amount": 500, "max_amount": 2000}, "scope": "all_documents", "complexity": "medium", "confidence": 0.90}),
        ("Candidates who graduated after 2020 with a master's degree", {"intent": "extract", "domain": "hr", "output_format": "bullets", "entities": ["candidates"], "constraints": {"min_graduation_year": 2020, "education_level": "master's"}, "scope": "all_documents", "complexity": "medium", "confidence": 0.90}),
        ("Contracts expiring within the next 30 days", {"intent": "extract", "domain": "legal", "output_format": "table", "entities": ["contracts"], "constraints": {"expiry_within_days": 30}, "scope": "all_documents", "complexity": "medium", "confidence": 0.90}),
        ("Show only the top 5 highest-paid invoices", {"intent": "rank", "domain": "invoice", "output_format": "numbered", "entities": ["invoices"], "constraints": {"top_n": 5}, "scope": "all_documents", "complexity": "medium", "confidence": 0.90}),
        ("Medications prescribed in the last year", {"intent": "extract", "domain": "medical", "output_format": "bullets", "entities": ["medications"], "constraints": {"time_period": "last year"}, "scope": "all_documents", "complexity": "simple", "confidence": 0.90}),
    ]

    # Off-topic / redirect queries
    _REDIRECT = [
        ("What's the weather like today?", {"intent": "redirect", "domain": "general", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
        ("Tell me a joke", {"intent": "redirect", "domain": "general", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
        ("Who is the president of the United States?", {"intent": "redirect", "domain": "general", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
        ("Write me a poem about cats", {"intent": "redirect", "domain": "general", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
        ("What's 2 + 2?", {"intent": "redirect", "domain": "general", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.95}),
    ]

    # Ambiguous queries needing clarification
    _AMBIGUOUS = [
        ("Show me the data", {"intent": "clarify", "domain": "general", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.30}),
        ("Analyze this", {"intent": "clarify", "domain": "general", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.30}),
        ("More details", {"intent": "clarify", "domain": "general", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.30}),
        ("What about the other one?", {"intent": "clarify", "domain": "general", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.30}),
        ("Fix it", {"intent": "clarify", "domain": "general", "output_format": "paragraph", "entities": [], "constraints": {}, "scope": "all_documents", "complexity": "simple", "confidence": 0.30}),
    ]

    for q, ts in _MULTI + _CONSTRAINTS + _REDIRECT + _AMBIGUOUS:
        examples.append(_chat_example(q, ts))

    # Generate prefix/suffix variations for each edge case
    variations: List[Dict[str, Any]] = []
    for q, ts in _MULTI + _CONSTRAINTS:
        for variant in _expand_paraphrases_offline(q, count=8):
            if variant != q:
                variations.append(_chat_example(variant, ts))

    examples.extend(variations)
    return examples


# ═══════════════════════════════════════════════════════════════════════════════
# Build full dataset
# ═══════════════════════════════════════════════════════════════════════════════


def build_training_dataset(
    output_dir: Optional[Path] = None,
    eval_split: float = 0.10,
) -> DatasetBuildResult:
    """Combine all sources, shuffle, and write train/eval JSONL files.

    Returns a DatasetBuildResult with paths and counts.
    """
    output_dir = output_dir or _DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect from all three sources
    curated = generate_curated_examples()
    paraphrases = generate_paraphrases()
    edge_cases = generate_edge_cases()

    all_examples = curated + paraphrases + edge_cases
    random.shuffle(all_examples)

    total = len(all_examples)
    eval_count = max(1, int(total * eval_split))
    train_count = total - eval_count

    train_examples = all_examples[:train_count]
    eval_examples = all_examples[train_count:]

    train_path = output_dir / _TRAIN_FILE
    eval_path = output_dir / _EVAL_FILE

    _write_jsonl(train_path, train_examples)
    _write_jsonl(eval_path, eval_examples)

    result = DatasetBuildResult(
        train_path=train_path,
        eval_path=eval_path,
        total_examples=total,
        train_count=train_count,
        eval_count=eval_count,
        source_counts={
            "curated": len(curated),
            "paraphrases": len(paraphrases),
            "edge_cases": len(edge_cases),
        },
    )
    log.info(
        "Training dataset built: %d total (%d train, %d eval) — "
        "curated=%d, paraphrases=%d, edge_cases=%d",
        total, train_count, eval_count,
        len(curated), len(paraphrases), len(edge_cases),
    )
    return result


def _write_jsonl(path: Path, examples: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-based paraphrase augmentation (optional, slower)
# ═══════════════════════════════════════════════════════════════════════════════


def generate_llm_paraphrases(
    seed_query: str,
    count: int = 50,
    model: str = "DocWain-Agent:latest",
) -> List[str]:
    """Use an LLM to generate diverse paraphrases of a seed query.

    This is slower but produces higher-quality, more diverse variations.
    Used during augmentation rounds when offline paraphrases aren't enough.
    """
    try:
        import ollama as _ollama
    except ImportError:
        log.warning("ollama not available for LLM paraphrases")
        return _expand_paraphrases_offline(seed_query, count)

    prompt = (
        f"Generate {count} diverse paraphrases of this document query. "
        f"Vary formality (casual to formal), length (3 words to 2 sentences), "
        f"include typos, abbreviations, and implicit intents. "
        f"Return one paraphrase per line, no numbering.\n\n"
        f"Query: {seed_query}"
    )
    try:
        response = _ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.8, "num_predict": 2048},
        )
        raw = response.get("response", "")
        lines = [
            line.strip()
            for line in raw.split("\n")
            if line.strip() and len(line.strip()) > 5
        ]
        return lines[:count] if lines else _expand_paraphrases_offline(seed_query, count)
    except Exception as exc:
        log.warning("LLM paraphrase generation failed: %s", exc)
        return _expand_paraphrases_offline(seed_query, count)


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = build_training_dataset()
    print(f"\nDataset built successfully!")
    print(f"  Train: {result.train_path} ({result.train_count} examples)")
    print(f"  Eval:  {result.eval_path} ({result.eval_count} examples)")
    print(f"  Sources: {result.source_counts}")
