from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChunkSource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_name: str
    page: Optional[int] = None


class Chunk(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    text: str
    score: float
    source: ChunkSource
    meta: Dict[str, Any] = Field(default_factory=dict)


MISSING_REASON = "Not explicitly mentioned in documents."


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    snippet: str


class FieldValue(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: Optional[str] = None
    value: str
    document_name: Optional[str] = None
    section: Optional[str] = None
    evidence_spans: List[EvidenceSpan]


class InvoiceItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str
    quantity: Optional[str] = None
    unit_price: Optional[str] = None
    amount: Optional[str] = None
    evidence_spans: List[EvidenceSpan]


class InvoiceItemsField(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: Optional[List[InvoiceItem]] = None
    missing_reason: Optional[str] = None


class FieldValuesField(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: Optional[List[FieldValue]] = None
    missing_reason: Optional[str] = None


class InvoiceSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: InvoiceItemsField = Field(default_factory=InvoiceItemsField)
    totals: FieldValuesField = Field(default_factory=FieldValuesField)
    parties: FieldValuesField = Field(default_factory=FieldValuesField)
    terms: FieldValuesField = Field(default_factory=FieldValuesField)
    invoice_metadata: FieldValuesField = Field(default_factory=FieldValuesField)


class Candidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: Optional[str] = None
    role: Optional[str] = None
    details: Optional[str] = None
    total_years_experience: Optional[str] = None
    experience_summary: Optional[str] = None
    technical_skills: Optional[List[str]] = None
    functional_skills: Optional[List[str]] = None
    certifications: Optional[List[str]] = None
    education: Optional[List[str]] = None
    achievements: Optional[List[str]] = None
    emails: Optional[List[str]] = None
    phones: Optional[List[str]] = None
    linkedins: Optional[List[str]] = None
    source_type: Optional[str] = None
    missing_reason: Optional[Dict[str, str]] = None
    evidence_spans: List[EvidenceSpan]


class CandidateField(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: Optional[List[Candidate]] = None
    missing_reason: Optional[str] = None


class HRSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidates: CandidateField = Field(default_factory=CandidateField)


class Clause(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: Optional[str] = None
    text: str
    evidence_spans: List[EvidenceSpan]


class ClauseField(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: Optional[List[Clause]] = None
    missing_reason: Optional[str] = None


class LegalSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    clauses: ClauseField = Field(default_factory=ClauseField)
    parties: FieldValuesField = Field(default_factory=FieldValuesField)
    obligations: FieldValuesField = Field(default_factory=FieldValuesField)


class MedicalSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    patient_info: FieldValuesField = Field(default_factory=FieldValuesField)
    diagnoses: FieldValuesField = Field(default_factory=FieldValuesField)
    medications: FieldValuesField = Field(default_factory=FieldValuesField)
    procedures: FieldValuesField = Field(default_factory=FieldValuesField)
    lab_results: FieldValuesField = Field(default_factory=FieldValuesField)
    vitals: FieldValuesField = Field(default_factory=FieldValuesField)


class PolicySchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    policy_info: FieldValuesField = Field(default_factory=FieldValuesField)
    coverage: FieldValuesField = Field(default_factory=FieldValuesField)
    premiums: FieldValuesField = Field(default_factory=FieldValuesField)
    exclusions: FieldValuesField = Field(default_factory=FieldValuesField)
    terms: FieldValuesField = Field(default_factory=FieldValuesField)


class GenericSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    facts: FieldValuesField = Field(default_factory=FieldValuesField)


class EntitySummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: Optional[str] = None
    document_name: Optional[str] = None
    document_id: Optional[str] = None
    evidence_spans: List[EvidenceSpan] = Field(default_factory=list)


class MultiEntitySchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entities: Optional[List[EntitySummary]] = None
    missing_reason: Optional[str] = None


class LLMResponseSchema(BaseModel):
    """Schema returned by LLM-first extraction. Contains the final answer text
    directly, bypassing the render step."""
    model_config = ConfigDict(extra="ignore")

    text: str
    evidence_chunks: List[str] = Field(default_factory=list)
    thinking_used: bool = False
    grounding_confidence: float = 0.0  # 0.0-1.0, set by lightweight grounding check


@dataclass
class LLMBudget:
    llm_client: Optional[Any]
    max_calls: int = 2
    used: int = 0

    def allow(self) -> bool:
        return bool(self.llm_client) and self.used < self.max_calls

    def consume(self) -> bool:
        if not self.allow():
            return False
        self.used += 1
        return True
