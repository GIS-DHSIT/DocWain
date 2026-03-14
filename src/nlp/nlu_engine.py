"""Centralized NLU Engine for DocWain.

Provides generic, description-based classification that any component can use.
Instead of hardcoded regex/keyword patterns, components register categories with
natural language descriptions. Classification uses:

1. Embedding cosine similarity (primary — most accurate)
2. spaCy structural NLP overlap (secondary — action verbs, target nouns)
3. Returns None when no confident match (no false positives)

Usage:
    from src.nlp.nlu_engine import get_registry, classify

    # Register once at module load / startup
    reg = get_registry("intent")
    reg.register("comparison", "Comparing two or more items side by side, finding differences")
    reg.register("ranking", "Ordering or scoring items from best to worst, finding top candidates")

    # Classify at request time
    result = classify("compare these candidates", "intent")
    # result = ClassificationResult(name="comparison", score=0.82, method="embedding")
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = get_logger(__name__)

# ── Data structures ───────────────────────────────────────────────────────

@dataclass
class CategoryEntry:
    """A registered category with its natural language description."""
    name: str
    description: str
    action_verbs: List[str] = field(default_factory=list)
    target_nouns: List[str] = field(default_factory=list)
    _embedding: Optional[Any] = field(default=None, repr=False)

@dataclass
class ClassificationResult:
    """Result of classifying a query against a registry."""
    name: str
    score: float
    method: str  # "embedding", "nlu_structural", or "combined"
    gap: float = 0.0  # Score gap above second-best

@dataclass
class QuerySemantics:
    """Parsed semantic structure of a user query."""
    action_verbs: List[str]
    target_nouns: List[str]
    context_words: List[str]
    raw_text: str
    _embedding: Optional[Any] = field(default=None, repr=False)

# ── Shared NLP singletons ─────────────────────────────────────────────────

_nlp_model = None
_nlp_lock = threading.Lock()

def _get_nlp():
    """Get spaCy NLP model (singleton, thread-safe)."""
    global _nlp_model
    if _nlp_model is not None:
        return _nlp_model
    with _nlp_lock:
        if _nlp_model is not None:
            return _nlp_model
        try:
            import spacy
            _nlp_model = spacy.load("en_core_web_sm", disable=["ner"])
            return _nlp_model
        except Exception as exc:
            logger.debug("Failed to load spaCy NLP model", exc_info=True)
            return None

_embedder_instance = None
_embedder_lock = threading.Lock()

def get_embedder() -> Any:
    """Get the sentence-transformer embedder (singleton).

    Tries the main model first (dw_newron), then dataHandler fallback,
    then loads directly from sentence_transformers as final fallback.
    """
    global _embedder_instance
    if _embedder_instance is not None:
        return _embedder_instance
    try:
        from src.api.dw_newron import _get_model
        model = _get_model()
        if model is not None and hasattr(model, "encode"):
            return model
    except Exception as exc:
        logger.debug("Failed to load embedder from dw_newron", exc_info=True)
    try:
        from src.api.dataHandler import get_model
        model = get_model()
        if model is not None and hasattr(model, "encode"):
            return model
    except Exception as exc:
        logger.debug("Failed to load embedder from dataHandler", exc_info=True)
    # Direct load fallback — ensures NLU engine works even in tests
    with _embedder_lock:
        if _embedder_instance is not None:
            return _embedder_instance
        try:
            import warnings
            warnings.filterwarnings("ignore", message=r".*_target_device.*has been deprecated", category=FutureWarning)
            from sentence_transformers import SentenceTransformer
            _embedder_instance = SentenceTransformer(
                "BAAI/bge-large-en-v1.5", device="cpu",
            )
            return _embedder_instance
        except Exception as exc:
            logger.debug("Failed to load SentenceTransformer directly as embedder fallback", exc_info=True)
    return None

# ── Query parsing ─────────────────────────────────────────────────────────

def parse_query(query: str) -> QuerySemantics:
    """Extract semantic structure from a query using spaCy NLP.

    Identifies:
    - Action verbs: what the user wants to do (compare, rank, translate, etc.)
    - Target nouns: what they want to act on (candidates, invoice, document, etc.)
    - Context words: additional context (skills, Python, financial, etc.)
    """
    nlp = _get_nlp()
    actions: List[str] = []
    targets: List[str] = []
    context: List[str] = []

    if nlp is not None:
        doc = nlp(query.lower())
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space:
                continue
            lemma = token.lemma_
            if len(lemma) <= 2:
                continue

            if token.pos_ == "VERB":
                actions.append(lemma)
            elif token.pos_ in ("NOUN", "PROPN"):
                if token.dep_ in ("dobj", "pobj", "attr", "nsubj"):
                    targets.append(lemma)
                else:
                    context.append(lemma)
            elif token.pos_ == "ADJ" and token.dep_ in ("amod", "acomp"):
                context.append(lemma)
    else:
        # Fallback: simple word extraction
        _STOP = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "and", "or", "not", "but", "if", "this", "that", "it", "its",
            "i", "me", "my", "we", "you", "your", "he", "she", "they",
        }
        for w in query.lower().split():
            w = w.strip(".,!?;:\"'()[]{}")
            if len(w) > 2 and w not in _STOP:
                targets.append(w)

    return QuerySemantics(
        action_verbs=actions,
        target_nouns=targets,
        context_words=context,
        raw_text=query,
    )

# ── Classification Registry ──────────────────────────────────────────────

class ClassificationRegistry:
    """A named registry of categories for NLU-based classification.

    Each category has a natural language description. Classification compares
    a query against all registered descriptions using embedding similarity
    and/or structural NLP overlap.
    """

    def __init__(
        self,
        name: str,
        *,
        threshold: float = 0.35,
        gap: float = 0.05,
        embedding_weight: float = 0.70,
        nlu_weight: float = 0.30,
    ):
        self.name = name
        self.threshold = threshold
        self.gap = gap
        self.embedding_weight = embedding_weight
        self.nlu_weight = nlu_weight
        self._entries: Dict[str, CategoryEntry] = {}
        self._lock = threading.Lock()

    def register(self, name: str, description: str) -> None:
        """Register a category with its natural language description."""
        with self._lock:
            entry = CategoryEntry(name=name, description=description)
            _extract_features(entry)
            self._entries[name] = entry
            logger.debug("Registry '%s': registered '%s'", self.name, name)

    def register_many(self, entries: Dict[str, str]) -> None:
        """Register multiple categories at once. entries = {name: description}."""
        for name, desc in entries.items():
            self.register(name, desc)

    def precompute_embeddings(self, embedder: Any) -> None:
        """Batch-encode all entry descriptions once. Call after registration."""
        if embedder is None:
            return
        entries_needing_embed = [
            (name, entry) for name, entry in self._entries.items()
            if entry._embedding is None
        ]
        if not entries_needing_embed:
            return
        descs = [entry.description for _, entry in entries_needing_embed]
        try:
            vecs = embedder.encode(descs, normalize_embeddings=True, show_progress_bar=False)
            for i, (name, entry) in enumerate(entries_needing_embed):
                entry._embedding = vecs[i]
            logger.debug(
                "Registry '%s': pre-computed %d embeddings",
                self.name, len(entries_needing_embed),
            )
        except Exception as exc:
            logger.warning("Registry '%s': batch embed failed: %s", self.name, exc)

    @property
    def entries(self) -> Dict[str, CategoryEntry]:
        return dict(self._entries)

    def classify(
        self,
        query: str,
        *,
        embedder: Any = None,
        max_results: int = 1,
    ) -> Optional[ClassificationResult]:
        """Classify a query against all registered categories.

        Returns the best match if it exceeds the threshold with sufficient gap,
        or None if no confident match.
        """
        if not query or not query.strip() or not self._entries:
            return None

        query_sem = parse_query(query)
        scores: Dict[str, float] = {}

        # Encode query ONCE, reuse across all entries
        query_vec = None
        if embedder is not None:
            try:
                query_vec = embedder.encode([query], normalize_embeddings=True)[0]
            except Exception as exc:
                logger.debug("Failed to encode query for classification", exc_info=True)

        for name, entry in self._entries.items():
            nlu_score = _compute_nlu_score(query_sem, entry)

            emb_score = 0.0
            if query_vec is not None:
                emb_score = _embedding_score_vec(query_vec, entry, embedder)

            if query_vec is not None:
                combined = self.embedding_weight * emb_score + self.nlu_weight * nlu_score
            else:
                combined = nlu_score

            scores[name] = combined

        if not scores:
            return None

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_name, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        score_gap = best_score - second_score

        if best_score < self.threshold:
            logger.debug(
                "Registry '%s': best=%s (%.3f) below threshold %.3f",
                self.name, best_name, best_score, self.threshold,
            )
            return None

        if score_gap < self.gap:
            logger.debug(
                "Registry '%s': best=%s (%.3f) gap=%.3f too small",
                self.name, best_name, best_score, score_gap,
            )
            return None

        method = "combined" if embedder else "nlu_structural"
        logger.debug(
            "Registry '%s': classified as '%s' (score=%.3f, gap=%.3f, method=%s)",
            self.name, best_name, best_score, score_gap, method,
        )
        return ClassificationResult(
            name=best_name,
            score=best_score,
            method=method,
            gap=score_gap,
        )

# ── NLU scoring functions ────────────────────────────────────────────────

def _extract_features(entry: CategoryEntry) -> None:
    """Extract action verbs and target nouns from description using spaCy."""
    try:
        nlp = _get_nlp()
        if nlp is None:
            return
        doc = nlp(entry.description.lower())
        entry.action_verbs = list({
            token.lemma_ for token in doc
            if token.pos_ == "VERB" and not token.is_stop and len(token.lemma_) > 2
        })
        entry.target_nouns = list({
            token.lemma_ for token in doc
            if token.pos_ in ("NOUN", "PROPN") and not token.is_stop and len(token.lemma_) > 2
        })
    except Exception as exc:
        logger.debug("Failed to extract NLU features from category description", exc_info=True)

def _compute_nlu_score(query_sem: QuerySemantics, entry: CategoryEntry) -> float:
    """Compute NLU structural overlap between parsed query and category entry.

    Weighted combination:
    - 40% action verb overlap (what the user wants to DO)
    - 40% target noun overlap (what the user acts ON)
    - 20% context overlap (supporting words)
    """
    if not entry.action_verbs and not entry.target_nouns:
        return 0.0

    q_actions = set(query_sem.action_verbs)
    q_targets = set(query_sem.target_nouns)
    q_context = set(query_sem.context_words)
    q_all = q_actions | q_targets | q_context

    if not q_all:
        return 0.0

    e_actions = set(entry.action_verbs)
    e_targets = set(entry.target_nouns)
    e_all = e_actions | e_targets

    action_overlap = len(q_actions & e_actions) / max(len(q_actions), 1)
    target_overlap = len(q_targets & e_targets) / max(len(q_targets), 1)
    context_overlap = len(q_all & e_all) / max(len(q_all), 1)

    return 0.4 * action_overlap + 0.4 * target_overlap + 0.2 * context_overlap

def _embedding_score(query: str, entry: CategoryEntry, embedder: Any) -> float:
    """Compute embedding cosine similarity between query and entry description."""
    try:
        if entry._embedding is None:
            vecs = embedder.encode([entry.description], normalize_embeddings=True)
            entry._embedding = vecs[0]

        query_vec = embedder.encode([query], normalize_embeddings=True)[0]
        return float(np.dot(query_vec, entry._embedding))
    except Exception as exc:
        logger.debug("Failed to compute embedding score", exc_info=True)
        return 0.0

def _embedding_score_vec(query_vec: Any, entry: CategoryEntry, embedder: Any) -> float:
    """Compute cosine similarity using pre-computed query vector."""
    try:
        if entry._embedding is None:
            vecs = embedder.encode([entry.description], normalize_embeddings=True)
            entry._embedding = vecs[0]
        return float(np.dot(query_vec, entry._embedding))
    except Exception as exc:
        logger.debug("Failed to compute embedding vector score", exc_info=True)
        return 0.0

# ── Global registry management ───────────────────────────────────────────

_registries: Dict[str, ClassificationRegistry] = {}
_registries_lock = threading.Lock()

def get_registry(
    name: str,
    *,
    threshold: float = 0.35,
    gap: float = 0.05,
    embedding_weight: float = 0.70,
    nlu_weight: float = 0.30,
    create: bool = True,
) -> Optional[ClassificationRegistry]:
    """Get or create a named classification registry.

    Registries are singletons keyed by name. If create=True (default),
    creates the registry if it doesn't exist.
    """
    with _registries_lock:
        if name in _registries:
            return _registries[name]
        if not create:
            return None
        reg = ClassificationRegistry(
            name,
            threshold=threshold,
            gap=gap,
            embedding_weight=embedding_weight,
            nlu_weight=nlu_weight,
        )
        _registries[name] = reg
        return reg

def classify(
    query: str,
    registry_name: str,
    *,
    embedder: Any = None,
) -> Optional[ClassificationResult]:
    """Classify a query using a named registry.

    Convenience wrapper: gets the registry and classifies in one call.
    Auto-resolves embedder if not provided.
    """
    reg = get_registry(registry_name, create=False)
    if reg is None:
        logger.debug("NLU registry '%s' not found", registry_name)
        return None

    if embedder is None:
        embedder = get_embedder()

    return reg.classify(query, embedder=embedder)

# ── Pre-built registries ─────────────────────────────────────────────────
# These are initialized lazily on first access. Each registry contains
# natural language descriptions — NO hardcoded patterns or keywords.

_registries_initialized: Dict[str, bool] = {}
_init_lock = threading.Lock()

def _ensure_registry(name: str) -> ClassificationRegistry:
    """Ensure a pre-built registry is initialized."""
    if name in _registries_initialized:
        reg = get_registry(name, create=False)
        if reg is not None:
            return reg

    with _init_lock:
        if name in _registries_initialized:
            reg = get_registry(name, create=False)
            if reg is not None:
                return reg

        initializer = _REGISTRY_INITIALIZERS.get(name)
        if initializer is None:
            raise ValueError(f"Unknown pre-built registry: {name}")

        initializer()
        _registries_initialized[name] = True
        reg = get_registry(name)
        # Pre-compute embeddings for all entries (batch encode once)
        try:
            reg.precompute_embeddings(get_embedder())
        except Exception as exc:
            logger.debug("Failed to precompute embeddings (embedder not available yet, will encode lazily)", exc_info=True)
        return reg

def _init_intent_registry() -> None:
    """Initialize the query intent classification registry."""
    reg = get_registry("intent", threshold=0.40, gap=0.03)
    reg.register_many({
        "comparison": (
            "Comparing two or more items side by side, finding differences "
            "and similarities between documents, candidates, or data, "
            "what are the differences, how do they compare, versus, contrast"
        ),
        "ranking": (
            "Ordering, rating, or scoring items from best to worst, "
            "finding top candidates, who is the best, rank by criteria, "
            "which one is strongest, most experienced, highest rated, "
            "shortlist the top performers"
        ),
        "summary": (
            "Providing a brief overview, highlights, executive summary, "
            "or condensed outline of document content"
        ),
        "timeline": (
            "Tracking chronological progression, career history, "
            "sequence of events over time, career path"
        ),
        "reasoning": (
            "Evaluating fitness, suitability, qualifications, "
            "making recommendations, assessing whether someone is capable, "
            "if I need someone who can, what would happen if, "
            "hypothetical scenario analysis, which candidate would be best for"
        ),
        "multi_field": (
            "Extracting multiple data fields, filling out forms, "
            "listing all line items, extracting every field or entry"
        ),
        "analytics": (
            "Counting items within document content, computing totals "
            "from invoices or records, averages of data points, sums of "
            "financial amounts, statistical distribution of values in "
            "documents, numerical aggregation of document data, "
            "aggregate financial amounts across records"
        ),
        "cross_document": (
            "Analyzing information across all documents, finding shared "
            "or common elements, cross-document patterns and comparisons"
        ),
        "generate": (
            "Creating, generating, writing, or drafting new content such as "
            "cover letters, professional summaries, interview questions, "
            "job descriptions, meeting agendas, gap analysis reports, "
            "shortlist reports, screening reports, comparison matrices, "
            "compose a document, build a report, prepare content, produce output"
        ),
    })

def _init_conversational_registry() -> None:
    """Initialize the conversational intent classification registry."""
    reg = get_registry("conversational", threshold=0.40, gap=0.03)
    reg.register_many({
        "GREETING": (
            "A friendly hello, hi, or greeting to start a conversation, "
            "saying good morning, namaste, or salutations"
        ),
        "FAREWELL": (
            "Saying goodbye, bye, see you later, ending the conversation, "
            "signing off, take care, peace out"
        ),
        "THANKS": (
            "Expressing gratitude, thank you, thanks for your help, "
            "appreciation for assistance received"
        ),
        "PRAISE": (
            "Positive feedback about response quality, great job, "
            "awesome answer, well done, excellent work"
        ),
        "NEGATIVE_MILD": (
            "Mild dissatisfaction, not quite right, could be better, "
            "close but not exact, try again please"
        ),
        "NEGATIVE_STRONG": (
            "Strong dissatisfaction, this is wrong, completely wrong, "
            "you got it wrong, that is incorrect, terrible answer, "
            "useless response, not helpful, not accurate, the worst"
        ),
        "IDENTITY": (
            "What is DocWain, who are you, describe this system, "
            "tell me about DocWain, introduce yourself, "
            "are you a bot or an AI, what kind of assistant are you, "
            "what system is DocWain, who is DocWain"
        ),
        "CAPABILITY": (
            "Asking what DocWain can do, what features or capabilities, "
            "what can you help with, what else can you do, list your abilities, "
            "show what you do, what are your features"
        ),
        "HOW_IT_WORKS": (
            "How do you work, how does DocWain work, "
            "explaining how the system processes and retrieves documents, "
            "what technology or AI model powers this, "
            "how accurate is the system, how does the search work"
        ),
        "PRIVACY": (
            "Questions about data security, privacy, who can access data, "
            "is my data safe, GDPR compliance, data retention"
        ),
        "LIMITATIONS": (
            "Asking what DocWain cannot do, limitations, "
            "can you browse the internet, what don't you support"
        ),
        "USAGE_HELP": (
            "Requesting help on how to use the tool, how do I get started, "
            "how can I upload documents, how should I use this, "
            "what file formats are supported, tutorial, example queries, "
            "help me begin, what types of files can I upload, "
            "how do I compare candidates, how can I rank resumes, "
            "how do I summarize documents, how to use a feature"
        ),
        "SMALL_TALK": (
            "Casual conversation, how are you, what's up, "
            "how's it going, how have you been"
        ),
        "CLARIFICATION": (
            "Asking to repeat, rephrase, or clarify a previous response, "
            "I didn't understand, can you elaborate, be more specific"
        ),
        "DOCUMENT_DISCOVERY": (
            "Asking what documents are available, uploaded, or stored in my profile, "
            "show my files, document inventory, list my documents, "
            "how many documents do I have, what is in my collection, "
            "what can I ask about, what topics are covered in my documents, "
            "what types of documents do I have, what kind of documents, "
            "what categories of files, document types in my profile"
        ),
    })

def _init_scope_registry() -> None:
    """Initialize the query scope classification registry."""
    reg = get_registry("scope", threshold=0.38, gap=0.06)
    reg.register_many({
        "all_profile": (
            "Query spanning all documents, all candidates, every item, "
            "comprehensive cross-document analysis, comparing or ranking "
            "multiple items, finding the best across all, common patterns, "
            "shared skills, who has the most experience"
        ),
        "targeted": (
            "Query about one specific named person, document, or item, "
            "asking about a particular individual by name, "
            "single document details"
        ),
    })

def _init_domain_task_registry() -> None:
    """Initialize the domain task classification registry."""
    reg = get_registry("domain_task", threshold=0.36, gap=0.025)
    reg.register_many({
        # HR tasks
        "hr:generate_interview_questions": (
            "Generate interview questions for a resume, preparing interview "
            "questions for job candidates based on their resume profile, "
            "creating HR interview preparation materials"
        ),
        "hr:skill_gap_analysis": (
            "Identifying missing skills and skill gaps between "
            "what a candidate has versus what a job requires, "
            "what skills are lacking, missing qualifications"
        ),
        "hr:role_fit_assessment": (
            "Evaluating overall candidate-job fit, is this person right "
            "for the position, who is the best fit for the role, "
            "suitability assessment, match score for a position"
        ),
        "hr:experience_timeline": (
            "Building a career timeline, mapping experience history "
            "and career progression for a candidate"
        ),
        "hr:candidate_summary": (
            "Creating a candidate profile summary, summarizing "
            "qualifications, skills, and experience"
        ),
        # Medical tasks
        "medical:drug_interaction_check": (
            "Checking for dangerous drug interactions, medication "
            "contraindications, prescription safety review"
        ),
        "medical:treatment_plan_review": (
            "Reviewing a treatment plan, assessing care plan effectiveness, "
            "evaluating therapeutic approach"
        ),
        "medical:lab_result_interpretation": (
            "Interpreting laboratory results, blood work analysis, "
            "understanding test result values and ranges"
        ),
        "medical:clinical_summary": (
            "Creating a clinical summary, medical case overview, "
            "patient diagnosis summary"
        ),
        "medical:patient_history_timeline": (
            "Building a patient history timeline, mapping medical events "
            "and treatments chronologically"
        ),
        # Legal tasks
        "legal:clause_risk_assessment": (
            "Identifying risky or problematic clauses in contracts, "
            "flagging legal red flags and unfavorable terms, "
            "find risky clauses in a contract, dangerous clause detection, "
            "problematic terms, clause risk review"
        ),
        "legal:compliance_check": (
            "Checking compliance with regulations, verifying regulatory "
            "adherence, GDPR or industry standard conformity"
        ),
        "legal:contract_comparison": (
            "Comparing two or more contracts, finding differences "
            "between legal agreements, side-by-side contract analysis"
        ),
        "legal:key_terms_extraction": (
            "Extracting key legal terms, defined terms, important "
            "legal definitions from contracts"
        ),
        "legal:obligation_tracker": (
            "Tracking legal obligations, deadlines, deliverables, "
            "and contractual commitments"
        ),
        # Invoice tasks
        "invoice:payment_anomaly_detection": (
            "Finding payment anomalies in invoices, detecting unusual charges, "
            "suspicious transactions, or billing irregularities, "
            "detect payment anomalies, unusual payment patterns, "
            "anomalous billing amounts, payment discrepancy detection, "
            "identify anomalies in invoice payments"
        ),
        "invoice:expense_categorization": (
            "Categorizing expenses, classifying charges by type, "
            "organizing financial line items"
        ),
        "invoice:duplicate_detection": (
            "Finding duplicate invoices, identifying double charges "
            "or repeated billing entries, same invoice submitted twice"
        ),
        "invoice:vendor_analysis": (
            "Analyzing vendor or supplier performance, reviewing "
            "vendor billing patterns and history"
        ),
        "invoice:financial_summary": (
            "Creating a financial summary, invoice overview, "
            "total amounts and payment status"
        ),
        # Content tasks
        "content:draft_email": (
            "Drafting an email, composing a professional message, "
            "writing business correspondence"
        ),
        "content:generate_documentation": (
            "Creating documentation, generating API docs, "
            "writing technical or software documentation"
        ),
        "content:rewrite_text": (
            "Rewriting or rephrasing text, improving writing quality, "
            "rewording content for clarity"
        ),
        "content:create_presentation": (
            "Creating a presentation, building slides, "
            "designing a slide deck"
        ),
        "content:generate_content": (
            "Generating general content, creating written material, "
            "producing text-based output"
        ),
        # Translation tasks
        "translation:detect_language": (
            "Detecting or identifying what language a document or text is "
            "written in, language identification, what language is this text, "
            "identify the language, determine the language of this content"
        ),
        "translation:localize_content": (
            "Localizing content for a specific region, adapting text "
            "for cultural or regional differences"
        ),
        "translation:multilingual_summary": (
            "Summarizing content in a different language, creating "
            "multilingual summaries or key points"
        ),
        "translation:translate_text": (
            "Translating text from one language to another, "
            "converting documents between languages such as Spanish, "
            "French, German, or any target language"
        ),
        # Education tasks
        "education:generate_quiz": (
            "Creating a quiz or test, generating assessment questions "
            "to test knowledge"
        ),
        "education:study_guide": (
            "Creating a study guide, revision material, "
            "learning reference document"
        ),
        "education:create_lesson": (
            "Creating a lesson plan, designing instructional content, "
            "building teaching materials"
        ),
        "education:explain_concept": (
            "Explaining a concept step by step, breaking down complex "
            "topics for understanding, teaching"
        ),
        # Image tasks
        "image:extract_text_from_image": (
            "Extracting text from an image using OCR, reading "
            "text in a scanned document or screenshot"
        ),
        "image:describe_image": (
            "Describing what an image shows, visual content analysis, "
            "image caption generation"
        ),
        "image:extract_data_from_image": (
            "Extracting structured data from an image, reading tables "
            "or forms from photographs"
        ),
        "image:analyze_image": (
            "General image analysis, understanding image content, "
            "visual document processing"
        ),
        # Web tasks
        "web:fetch_url": (
            "Fetching content from a URL, opening a web page, "
            "retrieving data from a web address"
        ),
        "web:fact_check": (
            "Fact-checking a claim, verifying whether something is true, "
            "validating information accuracy"
        ),
        "web:research_topic": (
            "Researching a topic in depth, investigating a subject, "
            "deep dive into a question"
        ),
        "web:search_web": (
            "Searching the web for information, finding current data online, "
            "looking up the latest information"
        ),
        # Analytics tasks
        "analytics:detect_anomalies": (
            "Detecting anomalies or outliers in data, finding unusual "
            "patterns or deviations"
        ),
        "analytics:find_patterns": (
            "Finding patterns and trends across data, identifying "
            "recurring themes or regularities"
        ),
        "analytics:extract_action_items": (
            "Extracting action items and tasks from documents, "
            "identifying to-do items and deadlines"
        ),
        "analytics:risk_assessment": (
            "Performing a risk assessment, assessing risks, identifying "
            "potential problems, evaluating risk factors and red flags"
        ),
        "analytics:generate_report": (
            "Generating an analytical report, creating a data-driven "
            "summary with findings and insights"
        ),
        # Screening tasks
        "screening:screen_pii": (
            "Screening or checking for personally identifiable information, "
            "PII detection, finding PII data, and privacy scanning"
        ),
        "screening:detect_ai_content": (
            "Detecting AI-generated content, checking whether text "
            "was written by AI"
        ),
        "screening:screen_resume": (
            "Screening resumes for quality and formatting issues, "
            "grading resume writing, evaluating resume layout, "
            "checking resume best practices and completeness"
        ),
        "screening:assess_readability": (
            "Assessing text readability, checking reading level, "
            "evaluating writing quality and clarity"
        ),
        "screening:compliance_scan": (
            "Scanning for compliance issues, checking adherence to "
            "standards and regulations"
        ),
        # Cloud platform tasks
        "cloud:jira_analysis": (
            "Working with Jira tickets, sprint management, "
            "issue tracking and project boards"
        ),
        "cloud:confluence_analysis": (
            "Working with Confluence wiki pages, documentation hubs, "
            "knowledge base management"
        ),
        "cloud:sharepoint_analysis": (
            "Working with SharePoint document libraries, "
            "SharePoint site content management"
        ),
        "cloud:cross_platform_summary": (
            "Cross-platform analysis, unified view across multiple "
            "cloud services and tools"
        ),
        # Customer service tasks
        "customer_service:resolve_issue": (
            "Resolve a customer issue, answer a customer question, "
            "help with a customer problem, handle customer complaint, "
            "support request resolution, customer query resolution"
        ),
        "customer_service:troubleshoot": (
            "Troubleshoot a problem step by step, diagnose an issue, "
            "provide troubleshooting steps, debug a problem, "
            "step-by-step fix, help me fix this issue"
        ),
        "customer_service:escalation_assessment": (
            "Assess whether an issue needs escalation, determine severity, "
            "evaluate if this needs a manager, escalation recommendation, "
            "severity assessment, triage customer issue"
        ),
        "customer_service:generate_response": (
            "Draft a customer response, write a reply to a customer, "
            "compose a customer-facing message, generate support reply, "
            "create a professional response to customer inquiry"
        ),
        "customer_service:faq_search": (
            "Search frequently asked questions, find FAQ entries, "
            "look up knowledge base, find answers in FAQ, "
            "search help articles, find relevant support documentation"
        ),
        # Analytics visualization tasks
        "analytics_viz:generate_chart": (
            "Create a chart, generate a graph, make a visualization "
            "from data, plot data visually, show data as a chart, "
            "bar chart, pie chart, visual representation of data, "
            "create a bar chart graph visualization from the data, "
            "generate a pie chart showing data proportions visually, "
            "render a chart image, draw a graph of the numbers, "
            "visualize this data as a chart or graph"
        ),
        "analytics_viz:generate_distribution": (
            "Show distribution of data, create a histogram, "
            "frequency breakdown, show how values are spread, "
            "data distribution visualization, "
            "plot a histogram of value frequencies, "
            "visualize the spread and density of data points"
        ),
        "analytics_viz:generate_comparison_chart": (
            "Create a comparison chart, compare items visually, "
            "side-by-side comparison visualization, compare values "
            "between categories, grouped comparison chart, "
            "create a comparison chart to visualize differences side by side, "
            "visualize differences between items in a chart"
        ),
        "analytics_viz:generate_timeline_chart": (
            "Visualize data over time, create a timeline chart, "
            "show chronological trend, temporal progression chart, "
            "time series visualization, monthly or yearly trend, "
            "visualize chronological temporal progression over time as a timeline chart, "
            "plot a timeline showing changes over time periods"
        ),
        "analytics_viz:generate_summary_dashboard": (
            "Create a dashboard with multiple charts, summary dashboard, "
            "overview visualization with key metrics, multi-chart summary, "
            "comprehensive visual overview, "
            "create a dashboard with multiple charts summarizing key metrics overview, "
            "build an overview dashboard with several charts and metrics"
        ),
        "analytics_viz:compute_statistics": (
            "Calculate statistics and show results visually, "
            "statistical analysis with chart, compute averages and totals "
            "with visualization, data statistics summary, "
            "calculate statistics averages and totals with visualization chart showing results, "
            "compute mean median and standard deviation with a chart"
        ),
    })

def _init_content_type_registry() -> None:
    """Initialize the content type detection registry."""
    reg = get_registry("content_type", threshold=0.36, gap=0.05)
    reg.register_many({
        # HR domain
        "cover_letter": (
            "Writing a cover letter or application letter for a job position, "
            "professional correspondence to accompany a resume"
        ),
        "professional_summary": (
            "Drafting a professional career summary, executive profile, "
            "career overview or professional bio"
        ),
        "skills_matrix": (
            "Creating a skills matrix, competency breakdown, "
            "skills comparison table or proficiency inventory"
        ),
        "candidate_comparison": (
            "Comparing candidates side by side, candidate versus analysis, "
            "comparative evaluation of applicants"
        ),
        "interview_prep": (
            "Preparing interview questions and guides, "
            "creating interview preparation materials for candidates"
        ),
        # Invoice / Finance domain
        "invoice_summary": (
            "Summarizing invoice details, invoice overview including totals, "
            "line items and payment information"
        ),
        "expense_report": (
            "Creating an expense report, spending report, "
            "expenditure summary from invoices or receipts"
        ),
        "payment_reminder": (
            "Writing a payment reminder letter, overdue notice, "
            "payment due notification for outstanding invoices"
        ),
        # Legal / Contract domain
        "contract_summary": (
            "Summarizing a contract or legal agreement, "
            "plain-language overview of contract terms and obligations"
        ),
        "compliance_report": (
            "Creating a compliance report, regulatory assessment, "
            "evaluating adherence to rules and regulations"
        ),
        "risk_assessment": (
            "Performing a risk assessment, risk analysis, "
            "identifying and evaluating contractual or legal risks"
        ),
        # Medical / Healthcare domain
        "patient_summary": (
            "Create a patient summary, patient report, clinical summary, "
            "overview of patient diagnosis medications and treatment plan, "
            "medical case summary from healthcare records"
        ),
        "medical_report": (
            "Creating a medical report, clinical report, "
            "formatted healthcare document from clinical evidence"
        ),
        # Report / Analysis domain
        "executive_summary": (
            "Writing an executive summary, high-level overview, "
            "condensed briefing of key findings for leadership"
        ),
        "key_findings": (
            "Extracting key findings, main findings, "
            "structured report of important discoveries from analysis"
        ),
        "recommendations": (
            "Generating recommendations, actionable suggestions, "
            "advice based on document evidence and analysis"
        ),
        # General domain
        "document_summary": (
            "Summarize a general document, create a concise overview "
            "of non-specialized document contents, main points and "
            "highlights from a generic text or report"
        ),
        "key_points": (
            "Extracting key points, main points, bullet points, "
            "listing the most important information from a document"
        ),
        "faq_generation": (
            "Generating frequently asked questions and answers, "
            "creating FAQ content from document material"
        ),
        "action_items": (
            "Extracting action items, to-do list, next steps, "
            "identifying actionable tasks from documents"
        ),
        "talking_points": (
            "Creating talking points, discussion points, "
            "key conversation topics extracted from documents"
        ),
        "meeting_notes": (
            "Writing meeting notes, meeting minutes, meeting summary, "
            "structured record of discussions and decisions"
        ),
        # Cross-document domain
        "comparison_report": (
            "Creating a comparison report, comparative analysis, "
            "detailed side-by-side analysis across multiple documents"
        ),
        "consolidated_summary": (
            "Writing a consolidated summary, combined summary, "
            "unified overview merging information from multiple documents"
        ),
        "trend_analysis": (
            "Performing trend analysis, identifying trends and patterns "
            "across multiple documents over time"
        ),
    })

def _init_document_query_registry() -> None:
    """Initialize document-query prototypes for contrastive NLU routing.

    These describe the kinds of queries that should be handled by the RAG
    pipeline (document retrieval + extraction), NOT the conversational handler.
    """
    reg = get_registry("document_query", threshold=0.30, gap=0.02)
    reg.register_many({
        "extract_data": (
            "Extract specific data or values from documents including totals, "
            "amounts, email addresses, phone numbers, dates, monetary figures, "
            "names, addresses, or structured fields from resumes, invoices, "
            "medical records, or contracts, what is the total amount"
        ),
        "compare_evaluate": (
            "Compare two or more items side by side, evaluate candidates, "
            "find differences between documents, rank profiles by qualification, "
            "determine who is better suited, compare invoices, rank by skills"
        ),
        "summarize_overview": (
            "Summarize document content, create an executive summary, "
            "provide a brief overview of resumes, contracts, or medical records, "
            "give the highlights or key points of uploaded files"
        ),
        "generate_content": (
            "Write or generate new content based on documents, create a cover letter, "
            "draft interview questions, compose a professional summary, "
            "build a skills matrix, prepare a screening report from document data"
        ),
        "analyze_patterns": (
            "Analyze patterns and trends across documents, identify common skills, "
            "find trends in data, assess risks in contracts, "
            "evaluate certifications, review career progression, experience levels"
        ),
        "domain_entities": (
            "Questions about specific domain content like candidates, patients, "
            "medications, invoices, vendors, policies, treatments, lab results, "
            "certifications, contracts, education history, or work experience"
        ),
        "cross_document": (
            "Find common or shared elements across multiple documents, "
            "calculate averages across records, identify unique elements, "
            "discover overlapping skills or experience across all candidates"
        ),
        "screen_review": (
            "Screen documents for issues, detect problems or red flags, "
            "review compliance, check for security risks, assess quality, "
            "evaluate suitability based on document content"
        ),
        "entity_lookup": (
            "Who is a specific person like John or Smith, looking up a named "
            "individual in the documents, tell me about someone by name, "
            "details about a particular candidate, patient, vendor, or person, "
            "information about someone mentioned in uploaded files"
        ),
    })

# ── Sub-intent classification (replaces keyword sets in extract.py) ──────

# Registry for query sub-intents: contact, product/item, totals, fit/rank
_subintent_registry_initialized = False

def _init_subintent_registry() -> None:
    """Register sub-intent prototypes for fine-grained extraction intent detection."""
    global _subintent_registry_initialized
    if _subintent_registry_initialized:
        return
    reg = get_registry("subintent", threshold=0.35, gap=0.02)
    reg.register_many({
        "contact": (
            "Extract contact information such as email addresses, phone numbers, "
            "LinkedIn profiles, how to reach someone, contact details, ways to "
            "get in touch with a person or company"
        ),
        "product_item": (
            "List products, line items, services, goods, inventory items, "
            "things being sold or purchased, items on an invoice or order"
        ),
        "totals": (
            "What is the total amount, how much is owed, balance due, subtotals, "
            "grand total, total cost, aggregate financial sum, what is the total"
        ),
        "fit_rank": (
            "Evaluate fitness, suitability, find the best match, rank candidates, "
            "determine who is most qualified, top performer, best fit for a role, "
            "most suitable candidate"
        ),
    })
    _subintent_registry_initialized = True

# Registry initializer map
_REGISTRY_INITIALIZERS = {
    "intent": _init_intent_registry,
    "conversational": _init_conversational_registry,
    "scope": _init_scope_registry,
    "domain_task": _init_domain_task_registry,
    "content_type": _init_content_type_registry,
    "document_query": _init_document_query_registry,
    "subintent": _init_subintent_registry,
}

# ── Convenience API ──────────────────────────────────────────────────────

def classify_intent(query: str, *, intent_hint: str | None = None) -> str:
    """Classify query into an intent type. Returns intent name or 'factual'."""
    # Fast path: use hint mapping if provided
    if intent_hint:
        _HINT_MAP = {
            "rank": "ranking", "compare": "comparison", "comparison": "comparison",
            "contact": "factual", "email": "factual", "phone": "factual",
            "summary": "summary", "summarize": "summary",
            "timeline": "timeline", "reasoning": "reasoning",
            "extraction": "multi_field", "cross_document": "cross_document",
            "analytics": "analytics", "aggregate": "analytics", "count": "analytics",
        }
        mapped = _HINT_MAP.get(intent_hint.lower())
        if mapped:
            return mapped

    # Structural fast-paths for intents that embeddings sometimes misclassify.
    ql = query.lower()

    # Comparison fast-path: "compare X and Y", "X vs Y", "differences between"
    import re as _re
    if _re.search(r'\b(compare|comparison|versus|vs\.?)\b', ql):
        # "compare" / "comparison" alone is a strong enough signal when
        # followed by at least one noun-like word (e.g. "Compare candidates").
        if (" and " in ql or _re.search(r'\bvs\.?\b', ql)
                or _re.search(r'\bcompar\w+\b.+\band\b', ql)
                or _re.search(r'\bcompar\w+\s+\w{3,}', ql)):
            return "comparison"
    if _re.search(r'\bdifference(?:s)?\s+between\b', ql):
        return "comparison"

    # Ranking fast-path: "rank", "top N", "best/most qualified candidate"
    if _re.search(r'\b(rank|ranking)\b', ql):
        return "ranking"
    if _re.search(r'\btop\s+\d+\b', ql) and any(w in ql for w in ("candidate", "resume", "document", "invoice")):
        return "ranking"
    # "most qualified/experienced/skilled", "best candidate/fit"
    if _re.search(r'\b(most|best|strongest|least)\s+(qualified|experienced|skilled|suitable|fit)\b', ql):
        return "ranking"
    if _re.search(r'\bbest\s+(candidate|applicant|resume|fit)\b', ql):
        return "ranking"

    # Structural detection for analytics queries where embeddings may
    # confuse "how many" with other intents (e.g., timeline).
    # Only trigger for genuinely numeric/counting queries — NOT for
    # "total experience" or "total overview" where "total" is an adjective.
    _analytics_match = False
    if any(p in ql for p in ("how many", "how much", "count the",
                              "count of", "count?")):
        _analytics_match = True
    elif any(p in ql for p in ("sum of", "sum the", "average of",
                                "average the")):
        _analytics_match = True
    elif "total" in ql:
        # "total" is analytics only when followed by amount/number words
        # NOT "total experience", "total overview", "totals"
        import re
        _total_analytics = re.search(
            r'\btotal\s+(amount|cost|sum|dollar|price|value|invoice|bill|charge|number\b)',
            ql,
        )
        _total_standalone = re.search(
            r'\b(what\s+is\s+the\s+total|calculate\s+the\s+total|find\s+the\s+total)\b',
            ql,
        )
        if _total_analytics or _total_standalone:
            _analytics_match = True
    if _analytics_match:
        return "analytics"

    # Summary fast-path: "summarize X", "summary of X", "give me a summary"
    if _re.search(r'\b(summarize|summarise)\b', ql):
        return "summary"
    if _re.search(r'\b(give|provide|create|write|show)\b.*\bsummary\b', ql):
        return "summary"
    if _re.search(r'\bsummary\s+(of|for)\b', ql):
        return "summary"

    # Reasoning fast-path: "is X qualified", "should we", "would X be"
    if _re.search(r'\b(is|are)\s+\w+\s+(qualified|suitable|fit|eligible|appropriate|ready)\b', ql):
        return "reasoning"
    if _re.search(r'\b(should|would|could)\s+(we|i|they|the\s+\w+)\b', ql):
        return "reasoning"

    # Cross-document fast-path: "across all", "what do all X share", "common across"
    if _re.search(r'\b(across\s+all|common\s+across|shared?\s+(?:across|between|among))\b', ql):
        return "cross_document"
    if _re.search(r'\bwhat\b.*\ball\s+\w+\s+(share|have\s+in\s+common)\b', ql):
        return "cross_document"
    # "Which/what candidates/patients/invoices have/show/include X"
    if _re.search(r'\b(which|what)\s+\w+s\s+(have|has|show|shows?|include|includes?|contain|mention)\b', ql):
        return "cross_document"
    # "What technologies/skills are common" or "technologies common across"
    if _re.search(r'\bcommon\b', ql) and _re.search(r'\b\w+s\b', ql):
        return "cross_document"

    # Timeline fast-path: "career progression", "over time", "chronolog"
    if _re.search(r'\b(chronolog\w*|over\s+time|progression|timeline)\b', ql):
        return "timeline"

    reg = _ensure_registry("intent")
    result = reg.classify(query, embedder=get_embedder())
    return result.name if result else "factual"

def classify_conversational(
    text: str,
) -> Optional[Tuple[str, float]]:
    """Classify text as a conversational intent using embedding-based NLU.

    Returns (intent_name, confidence) or None if not conversational.
    Uses embedding cosine similarity + spaCy structural overlap against
    natural language intent descriptions.
    """
    reg = _ensure_registry("conversational")
    result = reg.classify(text, embedder=get_embedder())
    if result is None:
        return None
    return (result.name, result.score)

# ── Contrastive NLU routing ──────────────────────────────────────────────

def _score_entry_with_vec(
    query_sem: QuerySemantics,
    entry: CategoryEntry,
    embedder: Any,
    query_vec: Any,
    emb_weight: float,
    nlu_weight: float,
) -> float:
    """Score a query against a single entry using a pre-computed query vector.

    Avoids redundant embedding of the query text when scoring against
    multiple registries in a single pass.
    """
    nlu_score = _compute_nlu_score(query_sem, entry)

    emb_score = 0.0
    if embedder is not None and query_vec is not None:
        if entry._embedding is None:
            try:
                vecs = embedder.encode(
                    [entry.description], normalize_embeddings=True,
                )
                entry._embedding = vecs[0]
            except Exception as exc:
                logger.debug("Failed to encode category entry embedding", exc_info=True)
        if entry._embedding is not None:
            emb_score = float(np.dot(query_vec, entry._embedding))

    if embedder is not None and query_vec is not None:
        return emb_weight * emb_score + nlu_weight * nlu_score
    return nlu_score

def classify_query_routing(text: str) -> Tuple[str, str, float]:
    """Holistic NLU routing: classify a query as document or conversational.

    Encodes the query once and scores against all prototypes (document-query
    types + conversational intent types) in a single pass.  The category with
    the highest score determines the routing.

    Returns ``(routing, intent, score)`` where *routing* is ``"document"`` or
    ``"conversational"``, *intent* is the matched prototype name, and *score*
    is the classification confidence.
    """
    text = (text or "").strip()
    if not text:
        return ("document", "", 0.0)

    # Fast-path: document inventory/count questions → conversational routing
    # Only matches inventory queries ("how many documents", "list my files"),
    # NOT content queries ("what does the invoice say").
    import re as _re
    _lower = text.lower().rstrip("?!. ")

    # Fast-path: short gratitude expressions → THANKS (not PRAISE)
    # "thanks!", "thank you", "thanks a lot" are gratitude, not praise
    if _re.match(r'^(thanks?|thank\s+you|thx|ty)\b', _lower):
        return ("conversational", "THANKS", 0.85)

    _DOC_INVENTORY_RE = _re.compile(
        r"(?:how many|list\s+(?:my\s+|all\s+)?|show\s+(?:my\s+|all\s+)?|count\s+(?:my\s+)?|do i have any|are there any)"
        r"(?:document|file|upload|collection|pdf|resume|invoice)s?",
        _re.IGNORECASE,
    )
    _DOC_WHAT_RE = _re.compile(
        r"what\s+(?:document|file)s?\s+(?:do i have|are (?:there|available|uploaded|stored))",
        _re.IGNORECASE,
    )
    if _DOC_INVENTORY_RE.search(_lower) or _DOC_WHAT_RE.search(_lower) or _lower in (
        "what do i have", "show my documents", "list my files",
        "how many documents", "what documents do i have",
        "what files do i have", "do i have any documents",
    ):
        return ("conversational", "DOCUMENT_DISCOVERY", 0.85)

    # Fast-path: usage help / how-to questions → conversational routing.
    # "how do I compare candidates?" and "how can I rank resumes?" are asking
    # about system capabilities, NOT about document content.
    _USAGE_HELP_RE = _re.compile(
        r"^how\s+(do|can|should|would)\s+(i|we)\s+(compare|rank|search|filter|sort|upload|delete|use|find|extract|analyze|view|export|download)",
        _re.IGNORECASE,
    )
    if _USAGE_HELP_RE.search(_lower):
        return ("conversational", "USAGE_HELP", 0.75)

    # Fast-path: strong domain nouns → document query.
    # Queries mentioning medical/HR/legal/invoice-specific terms are almost
    # certainly about document content, even if phrased conversationally
    # (e.g. "I need to understand the patient's condition").
    _STRONG_DOMAIN_NOUNS = {
        # Medical
        "patient", "patients", "diagnosis", "diagnoses", "medication", "medications",
        "prescription", "prescriptions", "treatment", "treatments", "symptom", "symptoms",
        "lab", "clinical", "condition", "conditions", "vitals", "prognosis",
        # HR / Resume
        "candidate", "candidates", "resume", "resumes", "applicant", "applicants",
        "qualification", "qualifications", "hire", "role", "roles",
        "skills", "skill", "experience", "education", "certifications", "certification",
        # Invoice / Financial
        "invoice", "invoices", "vendor", "vendors", "payment", "payments",
        "billing", "expense", "expenses", "amount", "subtotal", "total",
        # Legal / Contract
        "contract", "contracts", "clause", "clauses", "liability", "compliance",
        "obligation", "obligations",
        # Policy / Insurance
        "policy", "policies", "coverage", "premium", "premiums",
        "exclusion", "exclusions", "deductible", "deductibles",
        # Contact / Personal info (prevent PRIVACY misroute)
        "email", "phone", "address", "linkedin", "contact",
        # Quotation
        "quotation", "quotations", "quote", "quotes",
    }
    # Also match if query contains a proper noun (capitalized word that is
    # not a sentence-start common word) — likely a person/entity name
    _PROPER_NOUN_INDICATOR = _re.compile(
        r"(?:^|\s)([A-Z][a-z]{2,})(?:\s+[A-Z][a-z]{2,})*(?:\s|$|'s)"
    )
    _COMMON_STARTS = frozenset({
        "what", "who", "how", "when", "where", "why", "which", "can", "could",
        "does", "did", "is", "are", "was", "were", "will", "would", "should",
        "do", "has", "have", "had", "give", "tell", "show", "list", "get",
        "find", "compare", "rank", "summarize", "write", "create", "the",
        "this", "that", "these", "those", "here", "there", "please", "help",
        "let", "may", "i", "we", "my", "our", "it", "they", "he", "she",
        # Greetings and common words that look like proper nouns
        "hello", "hey", "good", "morning", "evening", "afternoon", "night",
        "thanks", "thank", "sure", "okay", "yes", "great", "nice", "fine",
        "welcome", "bye", "goodbye", "sorry", "wow", "cool", "awesome",
    })
    _words = set(_lower.split())
    if _words & _STRONG_DOMAIN_NOUNS:
        # Don't override inventory queries (already handled above)
        return ("document", "domain_noun_match", 0.80)

    # Proper noun detection: if the query mentions a capitalized name
    # (not a common sentence-start word), it's almost certainly about
    # document content (a specific person, entity, or product).
    _pn_matches = _PROPER_NOUN_INDICATOR.findall(text)
    if _pn_matches:
        for _pn in _pn_matches:
            if _pn.lower() not in _COMMON_STARTS:
                return ("document", "proper_noun_match", 0.80)

    embedder = get_embedder()
    query_sem = parse_query(text)

    # Encode query once, reuse across both registries
    query_vec = None
    if embedder is not None:
        try:
            query_vec = embedder.encode(
                [text], normalize_embeddings=True,
            )[0]
        except Exception as exc:
            logger.debug("Failed to encode text for scope classification", exc_info=True)

    doc_reg = _ensure_registry("document_query")
    conv_reg = _ensure_registry("conversational")

    # Best score against document-query prototypes
    doc_best = 0.0
    doc_best_name = ""
    for name, entry in doc_reg._entries.items():
        score = _score_entry_with_vec(
            query_sem, entry, embedder, query_vec,
            doc_reg.embedding_weight, doc_reg.nlu_weight,
        )
        if score > doc_best:
            doc_best = score
            doc_best_name = name

    # Best score against conversational prototypes
    conv_best = 0.0
    conv_best_name = ""
    for name, entry in conv_reg._entries.items():
        score = _score_entry_with_vec(
            query_sem, entry, embedder, query_vec,
            conv_reg.embedding_weight, conv_reg.nlu_weight,
        )
        if score > conv_best:
            conv_best = score
            conv_best_name = name

    logger.debug(
        "Query routing: doc=%.3f(%s) conv=%.3f(%s) -> %s | '%s'",
        doc_best, doc_best_name, conv_best, conv_best_name,
        "DOC" if doc_best >= conv_best else "CONV",
        text[:80],
    )

    # Detect meta-question patterns: "how do I...", "how can I...", "help", "can you..."
    # These are questions about usage/capabilities, not document operations.
    _is_meta_question = False
    nlp = _get_nlp()
    if nlp is not None:
        try:
            doc = nlp(text.lower())
            tokens = [t.text for t in doc if not t.is_space]
            # "how do I X", "how can I X", "how to X", "how should I X"
            if len(tokens) >= 3 and tokens[0] == "how":
                if tokens[1] in ("do", "can", "to", "should", "would", "could"):
                    _is_meta_question = True
            # "can you...", "could you..." — capability questions
            if len(tokens) >= 2 and tokens[0] in ("can", "could") and tokens[1] == "you":
                _is_meta_question = True
            # Bare help phrases
            lower_stripped = text.strip().lower().rstrip("?!.")
            if lower_stripped in ("help", "help me", "i need help",
                                   "show me example queries",
                                   "what can you do", "what do you do"):
                _is_meta_question = True
        except Exception as exc:
            logger.debug("Failed NLU meta-question detection", exc_info=True)

    # Conversational wins only if it scores higher AND has reasonable confidence
    _MIN_CONV_CONFIDENCE = 0.25
    if _is_meta_question and conv_best >= _MIN_CONV_CONFIDENCE:
        # Meta-questions strongly favor conversational routing
        return ("conversational", conv_best_name, conv_best)
    if conv_best > doc_best and conv_best >= _MIN_CONV_CONFIDENCE:
        return ("conversational", conv_best_name, conv_best)

    return ("document", doc_best_name, doc_best)

def classify_document_query(text: str) -> bool:
    """Convenience wrapper: returns True if the text is a document query.

    Uses contrastive NLU classification against document-query vs
    conversational prototypes.
    """
    routing, _, _ = classify_query_routing(text)
    return routing == "document"

def classify_scope(query: str) -> Optional[str]:
    """Classify query scope as 'all_profile' or 'targeted' or None."""
    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(query.lower())
        _has_quantifier = False
        _has_entity = False
        _ENTITY_PREPS = {"of", "for", "about", "from", "by"}
        # Known domain plural nouns that signal all-profile scope
        _PLURAL_DOMAIN_NOUNS = {
            "candidates", "resumes", "applicants", "profiles",
            "invoices", "documents", "records", "patients",
            "contracts", "policies", "files",
        }
        for token in doc:
            # Universal quantifiers + plural nouns → all_profile
            if token.lemma_ in ("all", "every", "each") and token.pos_ == "DET":
                for child in token.head.children:
                    if child.pos_ in ("NOUN", "PROPN") and child.tag_ in ("NNS", "NNPS"):
                        _has_quantifier = True
            # Superlative + question → all_profile
            if token.tag_ == "JJS" and any(
                t.lemma_ in ("who", "which") for t in doc
            ):
                _has_quantifier = True
            # "the" + known domain plural noun → all_profile (weak signal)
            # e.g., "summarize the resumes", "compare the candidates"
            if (token.pos_ == "DET" and token.text == "the"
                    and token.head.text.lower() in _PLURAL_DOMAIN_NOUNS):
                _has_quantifier = True
            # "multiple", "several" + plural → all_profile
            if token.lemma_ in ("multiple", "several") and token.pos_ == "ADJ":
                _has_quantifier = True
            # Entity detection: OOV token in pobj-of-preposition or PROPN position
            if (token.dep_ == "pobj" and token.head.text.lower() in _ENTITY_PREPS
                    and token.is_oov and not token.is_stop and len(token.text) > 2):
                _has_entity = True
            if token.pos_ == "PROPN" and token.is_oov and len(token.text) > 2:
                _has_entity = True
            if token.dep_ == "poss" and token.is_oov and len(token.text) > 2:
                _has_entity = True

        if _has_quantifier and not _has_entity:
            return "all_profile"
        if _has_entity and not _has_quantifier:
            return "targeted"

    reg = _ensure_registry("scope")
    result = reg.classify(query, embedder=get_embedder())
    return result.name if result else None

_DOMAIN_FILTERED_MIN_THRESHOLD = 0.40

def _domain_filtered_classify(
    query: str,
    domain: str,
    reg: ClassificationRegistry,
    embedder: Any,
) -> Optional[Dict[str, str]]:
    """Score only entries matching *domain* and return the best above threshold.

    Uses a higher minimum threshold than the registry default to prevent
    vague queries (e.g. "tell me about this") from matching when a domain
    is pre-set — the reduced candidate set makes spurious matches more likely.
    """
    filtered_scores: Dict[str, float] = {}
    query_sem = parse_query(query)
    for name, entry in reg._entries.items():
        if not name.startswith(f"{domain}:"):
            continue
        nlu_score = _compute_nlu_score(query_sem, entry)
        emb_score = _embedding_score(query, entry, embedder) if embedder else 0.0
        if embedder:
            combined = reg.embedding_weight * emb_score + reg.nlu_weight * nlu_score
        else:
            combined = nlu_score
        filtered_scores[name] = combined

    if not filtered_scores:
        return None

    best = max(filtered_scores.items(), key=lambda kv: kv[1])
    effective_threshold = max(reg.threshold, _DOMAIN_FILTERED_MIN_THRESHOLD)
    if best[1] >= effective_threshold:
        parts = best[0].split(":", 1)
        return {"domain": parts[0], "task_type": parts[1]}
    return None

def classify_query_subintent(query: str) -> Optional[str]:
    """Classify a query into a sub-intent category using NLU.

    Returns one of: "contact", "product_item", "totals", "fit_rank", or None.
    Replaces the keyword-set approach (_CONTACT_INTENTS, etc.) in extract.py.
    """
    if not query or not query.strip():
        return None
    _init_subintent_registry()
    reg = _ensure_registry("subintent")
    embedder = get_embedder()
    result = reg.classify(query, embedder=embedder)
    if result is not None:
        return result.name
    return None

def is_contact_query(query: str) -> bool:
    """Determine if a query is asking for contact information using NLU.

    Uses spaCy semantic analysis: checks for contact-related target nouns
    (email, phone, contact, linkedin) and extraction action verbs.
    Falls back to embedding-based classification.
    """
    if not query or not query.strip():
        return False

    # Primary: spaCy structural analysis
    sem = parse_query(query)
    _CONTACT_NOUNS = {"contact", "email", "phone", "linkedin", "address",
                      "telephone", "mobile", "number", "reach"}
    if any(n in _CONTACT_NOUNS for n in sem.target_nouns):
        return True
    if any(n in _CONTACT_NOUNS for n in sem.context_words):
        return True

    # Secondary: embedding-based sub-intent classification
    subintent = classify_query_subintent(query)
    return subintent == "contact"

def classify_domain_task(
    query: str,
    domain: str = "",
) -> Optional[Dict[str, str]]:
    """Classify query into a domain:task_type combination.

    Returns {"domain": ..., "task_type": ...} or None.
    """
    reg = _ensure_registry("domain_task")
    embedder = get_embedder()

    # When domain is pre-set, try domain-filtered scoring first so that
    # ambiguous queries (e.g. ties across domains) still resolve when
    # the caller already knows the domain.
    if domain:
        filtered = _domain_filtered_classify(query, domain, reg, embedder)
        if filtered:
            return filtered

    # Full (unfiltered) classification
    result = reg.classify(query, embedder=embedder)
    if result is None:
        return None

    # Parse "domain:task_type" from the result name
    if ":" in result.name:
        parts = result.name.split(":", 1)
        detected_domain = parts[0]
        task_type = parts[1]

        # If domain was pre-set but unfiltered picked a different domain,
        # fall back to domain-filtered scoring (already tried above, so
        # reaching here means it returned None — return None).
        if domain and detected_domain != domain:
            return None

        return {"domain": detected_domain, "task_type": task_type}

    return None
