"""
Auto Q&A Generator for DocWain.

Generates question-answer pairs from extracted document content
for improved retrieval quality and caching.
"""

from __future__ import annotations

import hashlib
import json
from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.intelligence.document_intelligence import (
    DocumentDomain,
    StructuredDocument,
    ExtractedEntities,
    DocumentSection,
)

logger = get_logger(__name__)

@dataclass
class GeneratedQA:
    """A generated question-answer pair."""
    question: str
    answer: str
    question_type: str  # factual, definitional, comparative, procedural
    confidence: float
    source_section_id: Optional[str] = None
    source_entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def qa_id(self) -> str:
        """Generate unique ID for this Q&A pair."""
        content = f"{self.question}|{self.answer}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "qa_id": self.qa_id,
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
            "confidence": self.confidence,
            "source_section_id": self.source_section_id,
            "source_entities": self.source_entities,
            "metadata": self.metadata,
        }

@dataclass
class QAGenerationResult:
    """Result of Q&A generation for a document."""
    document_id: str
    qa_pairs: List[GeneratedQA]
    generation_stats: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "qa_pairs": [qa.to_dict() for qa in self.qa_pairs],
            "qa_count": len(self.qa_pairs),
            "generation_stats": self.generation_stats,
        }
    
    def to_redis_format(self) -> List[Tuple[str, str]]:
        """Convert to key-value pairs for Redis storage."""
        pairs = []
        for qa in self.qa_pairs:
            key = f"qa:{self.document_id}:{qa.qa_id}"
            value = json.dumps(qa.to_dict())
            pairs.append((key, value))
        return pairs

class QAGenerator:
    """
    Generates question-answer pairs from structured documents.
    
    Creates various types of Q&A:
    - Factual: Who, What, When, Where questions
    - Definitional: What is X?
    - Comparative: How does X compare to Y?
    - Procedural: How to do X?
    - Domain-specific: Based on document type
    """
    
    # Question templates by type
    FACTUAL_TEMPLATES = [
        ("What is the {entity_type} mentioned in this document?", "{entity_value}"),
        ("Who is {person_name}?", "Based on the document, {person_name} is mentioned as {context}."),
        ("When was {event}?", "The date mentioned is {date_value}."),
        ("What is the total amount?", "The total amount is {money_value}."),
    ]
    
    RESUME_QUESTIONS = [
        ("What skills does {name} have?", "The candidate has skills in: {skills}."),
        ("What is {name}'s work experience?", "{name} has experience in: {experience}."),
        ("What is the candidate's education?", "Education: {education}."),
        ("What certifications does the candidate have?", "Certifications: {certifications}."),
        ("How many years of experience does {name} have?", "{name} has {years} years of experience."),
        ("What is the candidate's contact information?", "Contact: Email - {email}, Phone - {phone}."),
    ]
    
    INVOICE_QUESTIONS = [
        ("What is the total amount on this invoice?", "The total amount is {total}."),
        ("What are the line items on this invoice?", "Line items: {items}."),
        ("When is the payment due?", "Payment is due on {due_date}."),
        ("Who is the invoice from?", "The invoice is from {vendor}."),
        ("What are the payment terms?", "Payment terms: {terms}."),
    ]
    
    LEGAL_QUESTIONS = [
        ("What are the key terms in this agreement?", "Key terms include: {terms}."),
        ("Who are the parties involved?", "The parties are: {parties}."),
        ("What is the effective date?", "The effective date is {date}."),
        ("What are the main clauses?", "Main clauses: {clauses}."),
    ]
    
    def __init__(
        self,
        max_qa_per_document: int = 20,
        min_confidence: float = 0.5,
        enable_domain_specific: bool = True,
    ):
        """
        Initialize the Q&A generator.
        
        Args:
            max_qa_per_document: Maximum Q&A pairs to generate per document.
            min_confidence: Minimum confidence threshold for Q&A pairs.
            enable_domain_specific: Enable domain-specific question generation.
        """
        self.max_qa_per_document = max_qa_per_document
        self.min_confidence = min_confidence
        self.enable_domain_specific = enable_domain_specific
    
    def generate(self, document: StructuredDocument) -> QAGenerationResult:
        """
        Generate Q&A pairs from a structured document.
        
        Args:
            document: The structured document to process.
        
        Returns:
            QAGenerationResult with generated Q&A pairs.
        """
        qa_pairs: List[GeneratedQA] = []
        stats = {"total_generated": 0, "filtered": 0}
        
        # 1. Generate entity-based questions
        entity_qas = self._generate_entity_questions(document)
        qa_pairs.extend(entity_qas)
        stats["entity_based"] = len(entity_qas)
        
        # 2. Generate section-based questions
        section_qas = self._generate_section_questions(document)
        qa_pairs.extend(section_qas)
        stats["section_based"] = len(section_qas)
        
        # 3. Generate domain-specific questions
        if self.enable_domain_specific:
            domain_qas = self._generate_domain_questions(document)
            qa_pairs.extend(domain_qas)
            stats["domain_specific"] = len(domain_qas)
        
        # 4. Generate factual questions from content
        factual_qas = self._generate_factual_questions(document)
        qa_pairs.extend(factual_qas)
        stats["factual"] = len(factual_qas)
        
        # Filter and deduplicate
        filtered_pairs = self._filter_and_dedupe(qa_pairs)
        stats["filtered"] = len(qa_pairs) - len(filtered_pairs)
        stats["total_generated"] = len(filtered_pairs)
        
        # Limit to max
        final_pairs = filtered_pairs[:self.max_qa_per_document]
        
        return QAGenerationResult(
            document_id=document.document_id,
            qa_pairs=final_pairs,
            generation_stats=stats,
        )
    
    def _generate_entity_questions(self, document: StructuredDocument) -> List[GeneratedQA]:
        """Generate questions based on extracted entities."""
        qa_pairs = []
        entities = document.entities
        
        # Person questions
        for person in entities.persons[:3]:
            qa_pairs.append(GeneratedQA(
                question=f"Who is {person.value}?",
                answer=f"{person.value} is mentioned in this document.",
                question_type="factual",
                confidence=person.confidence,
                source_entities=[person.value],
            ))
        
        # Organization questions
        for org in entities.organizations[:3]:
            qa_pairs.append(GeneratedQA(
                question=f"What is {org.value}?",
                answer=f"{org.value} is an organization mentioned in this document.",
                question_type="definitional",
                confidence=org.confidence,
                source_entities=[org.value],
            ))
        
        # Date questions
        for date_entity in entities.dates[:3]:
            qa_pairs.append(GeneratedQA(
                question="What dates are mentioned in this document?",
                answer=f"The date {date_entity.value} is mentioned.",
                question_type="factual",
                confidence=date_entity.confidence,
            ))
        
        # Money questions
        for money in entities.monetary_values[:3]:
            qa_pairs.append(GeneratedQA(
                question="What amounts are mentioned in this document?",
                answer=f"The amount {money.value} is mentioned.",
                question_type="factual",
                confidence=money.confidence,
            ))
        
        # Skills questions
        if entities.skills:
            skill_list = ", ".join(e.value for e in entities.skills[:10])
            qa_pairs.append(GeneratedQA(
                question="What skills are mentioned in this document?",
                answer=f"The following skills are mentioned: {skill_list}.",
                question_type="factual",
                confidence=0.8,
                source_entities=[e.value for e in entities.skills[:10]],
            ))
        
        # Contact info
        if entities.emails:
            qa_pairs.append(GeneratedQA(
                question="What email addresses are in this document?",
                answer=f"Email: {', '.join(entities.emails[:3])}",
                question_type="factual",
                confidence=0.95,
            ))
        
        if entities.phones:
            qa_pairs.append(GeneratedQA(
                question="What phone numbers are in this document?",
                answer=f"Phone: {', '.join(entities.phones[:3])}",
                question_type="factual",
                confidence=0.95,
            ))
        
        return qa_pairs
    
    def _generate_section_questions(self, document: StructuredDocument) -> List[GeneratedQA]:
        """Generate questions based on document sections."""
        qa_pairs = []
        
        for section in document.sections:
            if not section.heading or not section.content:
                continue
            
            # What does section X contain?
            if len(section.content) > 50:
                summary = section.content[:200].strip()
                if len(section.content) > 200:
                    summary += "..."
                
                qa_pairs.append(GeneratedQA(
                    question=f"What information is in the {section.heading} section?",
                    answer=summary,
                    question_type="definitional",
                    confidence=0.7,
                    source_section_id=section.section_id,
                ))
        
        # Table of contents question
        if document.table_of_contents:
            headings = [t["heading"] for t in document.table_of_contents[:10]]
            qa_pairs.append(GeneratedQA(
                question="What are the main sections in this document?",
                answer=f"The document contains: {', '.join(headings)}.",
                question_type="factual",
                confidence=0.85,
            ))
        
        return qa_pairs
    
    def _generate_domain_questions(self, document: StructuredDocument) -> List[GeneratedQA]:
        """Generate domain-specific questions."""
        qa_pairs = []
        
        if document.domain == DocumentDomain.RESUME:
            qa_pairs.extend(self._generate_resume_questions(document))
        elif document.domain == DocumentDomain.INVOICE:
            qa_pairs.extend(self._generate_invoice_questions(document))
        elif document.domain == DocumentDomain.LEGAL:
            qa_pairs.extend(self._generate_legal_questions(document))
        
        return qa_pairs
    
    def _generate_resume_questions(self, document: StructuredDocument) -> List[GeneratedQA]:
        """Generate resume-specific Q&A pairs."""
        qa_pairs = []
        entities = document.entities
        
        # Try to find candidate name
        candidate_name = "the candidate"
        if entities.persons:
            candidate_name = entities.persons[0].value
        
        # Skills summary
        if entities.skills:
            skills_text = ", ".join(e.value for e in entities.skills[:8])
            qa_pairs.append(GeneratedQA(
                question=f"What are {candidate_name}'s technical skills?",
                answer=f"{candidate_name} has skills in: {skills_text}.",
                question_type="factual",
                confidence=0.85,
                source_entities=[e.value for e in entities.skills[:8]],
            ))
        
        # Contact info summary
        contact_parts = []
        if entities.emails:
            contact_parts.append(f"Email: {entities.emails[0]}")
        if entities.phones:
            contact_parts.append(f"Phone: {entities.phones[0]}")
        if entities.urls:
            linkedin = next((u for u in entities.urls if "linkedin" in u.lower()), None)
            if linkedin:
                contact_parts.append(f"LinkedIn: {linkedin}")
        
        if contact_parts:
            qa_pairs.append(GeneratedQA(
                question=f"How can I contact {candidate_name}?",
                answer=f"Contact information: {'; '.join(contact_parts)}.",
                question_type="factual",
                confidence=0.9,
            ))
        
        # Summary from sections
        for section in document.sections:
            if section.heading and any(h in section.heading.lower() for h in ["summary", "objective", "profile"]):
                qa_pairs.append(GeneratedQA(
                    question=f"What is {candidate_name}'s professional summary?",
                    answer=section.content[:300].strip(),
                    question_type="definitional",
                    confidence=0.8,
                    source_section_id=section.section_id,
                ))
                break
        
        return qa_pairs
    
    def _generate_invoice_questions(self, document: StructuredDocument) -> List[GeneratedQA]:
        """Generate invoice-specific Q&A pairs."""
        qa_pairs = []
        entities = document.entities
        
        # Total amount
        if entities.monetary_values:
            amounts = [e.value for e in entities.monetary_values]
            # Assume largest is total
            qa_pairs.append(GeneratedQA(
                question="What is the total amount on this invoice?",
                answer=f"Amounts found: {', '.join(amounts[:5])}.",
                question_type="factual",
                confidence=0.85,
            ))
        
        # Dates
        if entities.dates:
            dates = [e.value for e in entities.dates]
            qa_pairs.append(GeneratedQA(
                question="What are the important dates on this invoice?",
                answer=f"Dates: {', '.join(dates[:3])}.",
                question_type="factual",
                confidence=0.8,
            ))
        
        return qa_pairs
    
    def _generate_legal_questions(self, document: StructuredDocument) -> List[GeneratedQA]:
        """Generate legal document Q&A pairs."""
        qa_pairs = []
        entities = document.entities
        
        # Parties involved
        if entities.organizations:
            orgs = [e.value for e in entities.organizations[:5]]
            qa_pairs.append(GeneratedQA(
                question="Who are the parties in this agreement?",
                answer=f"Parties mentioned: {', '.join(orgs)}.",
                question_type="factual",
                confidence=0.75,
                source_entities=orgs,
            ))
        
        # Dates
        if entities.dates:
            qa_pairs.append(GeneratedQA(
                question="What is the effective date of this agreement?",
                answer=f"Dates found: {', '.join(e.value for e in entities.dates[:3])}.",
                question_type="factual",
                confidence=0.7,
            ))
        
        return qa_pairs
    
    def _generate_factual_questions(self, document: StructuredDocument) -> List[GeneratedQA]:
        """Generate general factual questions from document content."""
        qa_pairs = []
        
        # Document summary question
        full_text = document.get_full_text()
        if len(full_text) > 100:
            snippet = full_text[:500].strip()
            if len(full_text) > 500:
                snippet += "..."
            
            qa_pairs.append(GeneratedQA(
                question="What is this document about?",
                answer=f"This document contains: {snippet}",
                question_type="definitional",
                confidence=0.75,
            ))
        
        # Key facts as Q&A
        for fact in document.key_facts:
            if ":" in fact:
                label, value = fact.split(":", 1)
                qa_pairs.append(GeneratedQA(
                    question=f"What {label.strip().lower()} are in this document?",
                    answer=value.strip(),
                    question_type="factual",
                    confidence=0.7,
                ))
        
        return qa_pairs
    
    def _filter_and_dedupe(self, qa_pairs: List[GeneratedQA]) -> List[GeneratedQA]:
        """Filter low-quality and deduplicate Q&A pairs."""
        # Filter by confidence
        filtered = [qa for qa in qa_pairs if qa.confidence >= self.min_confidence]
        
        # Deduplicate by question similarity
        seen_questions = set()
        unique_pairs = []
        
        for qa in filtered:
            # Normalize question for comparison
            normalized = re.sub(r'\s+', ' ', qa.question.lower().strip())
            if normalized not in seen_questions:
                seen_questions.add(normalized)
                unique_pairs.append(qa)
        
        # Sort by confidence (descending)
        unique_pairs.sort(key=lambda x: x.confidence, reverse=True)
        
        return unique_pairs

class QACacheManager:
    """
    Manages Q&A pair caching in Redis for fast retrieval.
    """
    
    def __init__(self, redis_client=None, key_prefix: str = "docwain:qa"):
        """
        Initialize the cache manager.
        
        Args:
            redis_client: Redis client instance.
            key_prefix: Prefix for Redis keys.
        """
        self.redis_client = redis_client
        self.key_prefix = key_prefix
    
    def cache_qa_pairs(
        self,
        document_id: str,
        qa_pairs: List[GeneratedQA],
        ttl_seconds: int = 86400 * 30,  # 30 days
    ) -> int:
        """
        Cache Q&A pairs for a document.
        
        Args:
            document_id: Document identifier.
            qa_pairs: List of Q&A pairs to cache.
            ttl_seconds: Time-to-live in seconds.
        
        Returns:
            Number of pairs cached.
        """
        if not self.redis_client:
            logger.debug("Redis client not available for Q&A caching")
            return 0
        
        cached = 0
        try:
            pipe = self.redis_client.pipeline()
            
            # Store individual Q&A pairs
            for qa in qa_pairs:
                key = f"{self.key_prefix}:{document_id}:{qa.qa_id}"
                value = json.dumps(qa.to_dict())
                pipe.setex(key, ttl_seconds, value)
                cached += 1
            
            # Store index of questions for this document
            index_key = f"{self.key_prefix}:index:{document_id}"
            question_index = {qa.qa_id: qa.question for qa in qa_pairs}
            pipe.setex(index_key, ttl_seconds, json.dumps(question_index))
            
            pipe.execute()
            logger.info("Cached %d Q&A pairs for document %s", cached, document_id)
            
        except Exception as e:
            logger.error("Failed to cache Q&A pairs: %s", e)
            return 0
        
        return cached
    
    def get_qa_pairs(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve cached Q&A pairs for a document.
        
        Args:
            document_id: Document identifier.
        
        Returns:
            List of Q&A pair dictionaries.
        """
        if not self.redis_client:
            return []
        
        try:
            # Get index first
            index_key = f"{self.key_prefix}:index:{document_id}"
            index_raw = self.redis_client.get(index_key)
            if not index_raw:
                return []
            
            question_index = json.loads(index_raw)
            
            # Fetch individual Q&A pairs
            qa_pairs = []
            for qa_id in question_index.keys():
                key = f"{self.key_prefix}:{document_id}:{qa_id}"
                raw = self.redis_client.get(key)
                if raw:
                    qa_pairs.append(json.loads(raw))
            
            return qa_pairs
            
        except Exception as e:
            logger.error("Failed to retrieve Q&A pairs: %s", e)
            return []
    
    def find_matching_qa(
        self,
        document_id: str,
        query: str,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find Q&A pairs matching a query.
        
        Uses simple keyword matching for fast retrieval.
        
        Args:
            document_id: Document to search in.
            query: User query to match.
            max_results: Maximum results to return.
        
        Returns:
            List of matching Q&A pairs with scores.
        """
        qa_pairs = self.get_qa_pairs(document_id)
        if not qa_pairs:
            return []
        
        query_words = set(query.lower().split())
        scored_pairs = []
        
        for qa in qa_pairs:
            question_words = set(qa["question"].lower().split())
            overlap = len(query_words & question_words)
            if overlap > 0:
                score = overlap / max(len(query_words), len(question_words))
                scored_pairs.append({**qa, "match_score": score})
        
        # Sort by score and return top results
        scored_pairs.sort(key=lambda x: x["match_score"], reverse=True)
        return scored_pairs[:max_results]

__all__ = [
    "QAGenerator",
    "GeneratedQA",
    "QAGenerationResult",
    "QACacheManager",
]
