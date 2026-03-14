"""Trained multi-head MLP for intent + domain classification.

A lightweight NumPy-only MLP trained self-supervised on synthetic query
templates.  Two classification heads share a hidden layer:

    Input:  query embedding (1024-dim, BAAI/bge-large-en-v1.5)
    Shared: Linear(1024 → 128) → ReLU
    Intent: Linear(128 → 8) → Softmax
    Domain: Linear(128 → 6) → Softmax

Training uses cross-entropy loss on both heads simultaneously.
Word-dropout augmentation expands ~128 templates to ~384 effective samples.

Usage::

    from src.intent.intent_classifier import ensure_intent_classifier, get_intent_classifier

    clf = ensure_intent_classifier(embedder)
    result = clf.predict(query_embedding)
    # result = {"intent": "qa", "intent_confidence": 0.91,
    #           "domain": "resume", "domain_confidence": 0.85}
"""

from __future__ import annotations

import pickle
import random
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------
INTENT_NAMES: List[str] = [
    "qa", "summarize", "compare", "rank", "list", "extract", "contact", "generate",
]
DOMAIN_NAMES: List[str] = [
    "resume", "invoice", "legal", "policy", "report", "generic",
]

_INTENT_INDEX = {name: i for i, name in enumerate(INTENT_NAMES)}
_DOMAIN_INDEX = {name: i for i, name in enumerate(DOMAIN_NAMES)}

# ---------------------------------------------------------------------------
# Self-supervised training templates: (query, intent, domain)
# ~10 per intent × 8 intents + ~8 per domain × 6 domains = ~128 templates
# Each template has ONE intent label and ONE domain label.
# ---------------------------------------------------------------------------
TRAINING_TEMPLATES: List[Tuple[str, str, str]] = [
    # ── qa ──────────────────────────────────────────────────
    ("What is the candidate's highest qualification?", "qa", "resume"),
    ("What does this section say about payment terms?", "qa", "invoice"),
    ("Tell me about the confidentiality clause.", "qa", "legal"),
    ("What is the coverage limit for flood damage?", "qa", "policy"),
    ("What were the quarterly revenue figures?", "qa", "report"),
    ("What is the value of this field?", "qa", "generic"),
    ("Which department does this person belong to?", "qa", "resume"),
    ("When is the invoice due date?", "qa", "invoice"),
    ("What is the governing law for this contract?", "qa", "legal"),
    ("What is the annual premium amount?", "qa", "policy"),
    ("How many defects were reported last month?", "qa", "report"),
    ("What does this document contain?", "qa", "generic"),
    # ── summarize ──────────────────────────────────────────
    ("Summarize this candidate's profile.", "summarize", "resume"),
    ("Give me an overview of the invoice.", "summarize", "invoice"),
    ("Provide a brief summary of the contract.", "summarize", "legal"),
    ("Recap the key coverage details.", "summarize", "policy"),
    ("What are the main findings of this report?", "summarize", "report"),
    ("Summarize the document contents.", "summarize", "generic"),
    ("Give me the highlights of this resume.", "summarize", "resume"),
    ("Overview of all charges and fees.", "summarize", "invoice"),
    ("Brief recap of the legal provisions.", "summarize", "legal"),
    ("Outline the policy benefits.", "summarize", "policy"),
    # ── compare ────────────────────────────────────────────
    ("Compare the two candidates.", "compare", "resume"),
    ("What are the differences between these invoices?", "compare", "invoice"),
    ("How do these contracts differ?", "compare", "legal"),
    ("Compare the coverage of both policies.", "compare", "policy"),
    ("Side by side comparison of the profiles.", "compare", "resume"),
    ("Compare skills and experience of all candidates.", "compare", "resume"),
    ("What are the differences in payment terms?", "compare", "invoice"),
    ("How do these two reports differ in conclusions?", "compare", "report"),
    ("Compare the indemnification clauses.", "compare", "legal"),
    ("Which policy offers better flood coverage?", "compare", "policy"),
    # ── rank ───────────────────────────────────────────────
    ("Rank the candidates by their qualifications.", "rank", "resume"),
    ("Who is the best fit for this Python developer role?", "rank", "resume"),
    ("Order them by relevance to the position.", "rank", "resume"),
    ("Which candidate has the most experience?", "rank", "resume"),
    ("Rank invoices by total amount.", "rank", "invoice"),
    ("Which report scored highest on quality?", "rank", "report"),
    ("Top candidates for senior engineer.", "rank", "resume"),
    ("Best qualified candidate for data scientist.", "rank", "resume"),
    ("Rank by years of relevant experience.", "rank", "resume"),
    ("Who has the strongest technical background?", "rank", "resume"),
    # ── list ───────────────────────────────────────────────
    ("List all the documents.", "list", "generic"),
    ("Show me all available resumes.", "list", "resume"),
    ("What invoices are in the system?", "list", "invoice"),
    ("List all legal agreements.", "list", "legal"),
    ("Enumerate the uploaded policies.", "list", "policy"),
    ("How many documents do I have?", "list", "generic"),
    ("What documents are available?", "list", "generic"),
    ("Show all candidates.", "list", "resume"),
    ("List reports by date.", "list", "report"),
    ("Show all contracts in the profile.", "list", "legal"),
    # ── extract ────────────────────────────────────────────
    ("Extract the key details from this resume.", "extract", "resume"),
    ("Pull out the line items from the invoice.", "extract", "invoice"),
    ("Identify the important legal clauses.", "extract", "legal"),
    ("Get the structured data from the document.", "extract", "generic"),
    ("Extract all skills mentioned.", "extract", "resume"),
    ("Find the payment due date and amount.", "extract", "invoice"),
    ("Extract coverage limits and exclusions.", "extract", "policy"),
    ("Pull out the methodology section.", "extract", "report"),
    ("Identify the parties in this contract.", "extract", "legal"),
    ("Get all dates and deadlines.", "extract", "generic"),
    # ── contact ────────────────────────────────────────────
    ("What is the email address?", "contact", "resume"),
    ("Get me the phone number.", "contact", "resume"),
    ("How can I reach this person?", "contact", "resume"),
    ("What is the LinkedIn profile URL?", "contact", "resume"),
    ("Contact details for this candidate.", "contact", "resume"),
    ("What is the vendor's phone number?", "contact", "invoice"),
    ("Email address on this invoice.", "contact", "invoice"),
    ("How do I contact the vendor?", "contact", "invoice"),
    ("Supplier contact information.", "contact", "invoice"),
    ("Contact info for the signing party.", "contact", "legal"),
    # ── generate ───────────────────────────────────────────
    ("Write a cover letter for this candidate.", "generate", "resume"),
    ("Draft a summary report.", "generate", "report"),
    ("Generate a recommendation based on the profiles.", "generate", "resume"),
    ("Create new content from the source material.", "generate", "generic"),
    ("Compose a professional bio.", "generate", "resume"),
    ("Draft a response to this legal notice.", "generate", "legal"),
    ("Write a comparison summary.", "generate", "resume"),
    ("Generate interview questions for this candidate.", "generate", "resume"),
    ("Create a policy renewal notice.", "generate", "policy"),
    ("Draft a payment reminder.", "generate", "invoice"),
    # ── domain-focused extras (reinforce domain signal) ────
    ("Work experience, education, skills, and career history.", "qa", "resume"),
    ("Professional certifications and employment background.", "extract", "resume"),
    ("Invoice with line items, amounts, and payment terms.", "extract", "invoice"),
    ("Billing document with unit prices and quantities.", "qa", "invoice"),
    ("Contract with clauses, terms, and governing law.", "qa", "legal"),
    ("Agreement with indemnification and termination.", "extract", "legal"),
    ("Insurance policy with coverage and premiums.", "qa", "policy"),
    ("Claim procedures and deductibles.", "extract", "policy"),
    ("Business report with analysis and recommendations.", "summarize", "report"),
    ("Technical report with data and conclusions.", "qa", "report"),
    ("General document with miscellaneous content.", "qa", "generic"),
    ("Unclassified information from various sources.", "extract", "generic"),
    # ── more domain reinforcement ──────────────────────────
    ("What certifications does this person hold?", "qa", "resume"),
    ("Describe the candidate's project experience.", "qa", "resume"),
    ("What is the total tax on this bill?", "qa", "invoice"),
    ("Breakdown of all charges.", "extract", "invoice"),
    ("What are the warranty terms?", "qa", "legal"),
    ("Dispute resolution mechanism in the agreement.", "qa", "legal"),
    ("What natural disasters are covered?", "qa", "policy"),
    ("Premium payment schedule.", "extract", "policy"),
    ("Summary of audit findings.", "summarize", "report"),
    ("Data analysis methodology used.", "qa", "report"),
    # ── scope-reinforcing: all_profile routing ────────────
    ("Summarize all the resumes.", "summarize", "resume"),
    ("Overview of every candidate.", "summarize", "resume"),
    ("Give me details about all the documents.", "summarize", "generic"),
    ("Who among the candidates has the best skills?", "rank", "resume"),
    ("Show the differences between the contracts.", "compare", "legal"),
    ("How many invoices are pending?", "list", "invoice"),
    ("List every uploaded report.", "list", "report"),
    ("What are the top skills across all resumes?", "rank", "resume"),
    ("Analyze all uploaded files.", "summarize", "generic"),
    ("Overview of all reports.", "summarize", "report"),
    ("Can you compare all candidates' experience?", "compare", "resume"),
    ("Rank all policies by coverage.", "rank", "policy"),
]

# Stop words that should not be dropped during augmentation
_AUGMENT_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "and", "or", "but", "not", "this", "that", "these", "those",
    "what", "which", "who", "how", "do", "does", "did", "i", "me", "my",
})


def _augment_with_word_dropout(
    templates: List[Tuple[str, str, str]],
    n_variants: int = 2,
    seed: int = 42,
) -> List[Tuple[str, str, str]]:
    """Create augmented variants by dropping 1-2 non-stopwords per query."""
    rng = random.Random(seed)
    augmented: List[Tuple[str, str, str]] = []
    for query, intent, domain in templates:
        words = query.split()
        # Find droppable word indices (non-stopwords, not first/last)
        droppable = [
            i for i, w in enumerate(words)
            if w.lower().rstrip("?.,!") not in _AUGMENT_STOPWORDS
            and len(w) > 2
            and 0 < i < len(words) - 1
        ]
        if len(droppable) < 2:
            # Not enough words to drop — just duplicate original
            for _ in range(n_variants):
                augmented.append((query, intent, domain))
            continue
        for _ in range(n_variants):
            n_drop = rng.randint(1, min(2, len(droppable)))
            drop_indices = set(rng.sample(droppable, n_drop))
            new_words = [w for i, w in enumerate(words) if i not in drop_indices]
            augmented.append((" ".join(new_words), intent, domain))
    return augmented


# ---------------------------------------------------------------------------
# IntentDomainClassifier — NumPy-only multi-head MLP
# ---------------------------------------------------------------------------

class IntentDomainClassifier:
    """Multi-head MLP: Shared(1024→128)→ReLU, Intent(128→8)→Softmax, Domain(128→6)→Softmax.

    Trained with cross-entropy loss on both heads via SGD.
    No PyTorch dependency — pure NumPy.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 128,
        n_intents: int = 8,
        n_domains: int = 6,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_intents = n_intents
        self.n_domains = n_domains
        self.intent_names = INTENT_NAMES[:n_intents]
        self.domain_names = DOMAIN_NAMES[:n_domains]

        # He initialization
        self.W_shared = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b_shared = np.zeros(hidden_dim, dtype=np.float32)
        self.W_intent = np.random.randn(hidden_dim, n_intents).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b_intent = np.zeros(n_intents, dtype=np.float32)
        self.W_domain = np.random.randn(hidden_dim, n_domains).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b_domain = np.zeros(n_domains, dtype=np.float32)

        self._trained = False

    # --- Forward pass ---

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass returning (hidden, intent_probs, domain_probs)."""
        h = np.maximum(0, X @ self.W_shared + self.b_shared)  # ReLU
        intent_logits = h @ self.W_intent + self.b_intent
        domain_logits = h @ self.W_domain + self.b_domain
        intent_probs = self._softmax(intent_logits)
        domain_probs = self._softmax(domain_logits)
        return h, intent_probs, domain_probs

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    # --- Training ---

    def train(
        self,
        embedder: Any,
        epochs: int = 500,
        lr: float = 0.5,
        templates: Optional[List[Tuple[str, str, str]]] = None,
    ) -> List[float]:
        """Train on synthetic templates. Returns per-epoch losses."""
        templates = templates or TRAINING_TEMPLATES
        if not templates:
            log.warning("No training templates provided")
            return []

        # Augment with word-dropout
        augmented = _augment_with_word_dropout(templates, n_variants=2)
        all_templates = list(templates) + augmented
        log.info("Intent classifier: %d base + %d augmented = %d training samples",
                 len(templates), len(augmented), len(all_templates))

        # Encode all template queries
        queries = [t[0] for t in all_templates]
        try:
            X = embedder.encode(queries, normalize_embeddings=True)
            if not isinstance(X, np.ndarray):
                X = np.array(X, dtype=np.float32)
            else:
                X = X.astype(np.float32)
        except Exception:
            log.error("Failed to encode training templates", exc_info=True)
            return []

        # Update input_dim if embedder produces different size
        if X.shape[1] != self.input_dim:
            self.input_dim = X.shape[1]
            self.W_shared = np.random.randn(self.input_dim, self.hidden_dim).astype(np.float32) * np.sqrt(2.0 / self.input_dim)

        n = len(X)

        # Build one-hot label matrices
        Y_intent = np.zeros((n, self.n_intents), dtype=np.float32)
        Y_domain = np.zeros((n, self.n_domains), dtype=np.float32)
        for i, (_, intent_label, domain_label) in enumerate(all_templates):
            if intent_label in _INTENT_INDEX:
                Y_intent[i, _INTENT_INDEX[intent_label]] = 1.0
            if domain_label in _DOMAIN_INDEX:
                Y_domain[i, _DOMAIN_INDEX[domain_label]] = 1.0

        losses: List[float] = []
        eps = 1e-7

        for epoch in range(epochs):
            h, intent_probs, domain_probs = self._forward(X)

            # Cross-entropy loss (both heads)
            loss_intent = -np.mean(np.sum(Y_intent * np.log(intent_probs + eps), axis=1))
            loss_domain = -np.mean(np.sum(Y_domain * np.log(domain_probs + eps), axis=1))
            loss = loss_intent + loss_domain
            losses.append(float(loss))

            # Backward: softmax + cross-entropy gradient = (probs - labels)
            # The /n is already accounted for in the mean loss; using raw
            # (probs - labels) gives the per-sample gradient which, when
            # multiplied by X.T (matrix multiply sums over samples), produces
            # the full batch gradient.  lr handles scaling.
            d_intent = (intent_probs - Y_intent)
            d_domain = (domain_probs - Y_domain)

            # Intent head gradients
            dW_intent = h.T @ d_intent / n
            db_intent = np.mean(d_intent, axis=0)

            # Domain head gradients
            dW_domain = h.T @ d_domain / n
            db_domain = np.mean(d_domain, axis=0)

            # Shared layer gradient (sum from both heads)
            dh = d_intent @ self.W_intent.T + d_domain @ self.W_domain.T
            dh[h <= 0] = 0  # ReLU gradient

            dW_shared = X.T @ dh / n
            db_shared = np.mean(dh, axis=0)

            # SGD update
            self.W_shared -= lr * dW_shared
            self.b_shared -= lr * db_shared
            self.W_intent -= lr * dW_intent
            self.b_intent -= lr * db_intent
            self.W_domain -= lr * dW_domain
            self.b_domain -= lr * db_domain

        self._trained = True
        log.info(
            "Intent classifier trained: %d epochs, final loss=%.4f (intent=%.4f, domain=%.4f)",
            epochs, losses[-1] if losses else float("nan"),
            loss_intent, loss_domain,
        )
        return losses

    # --- Prediction ---

    def predict(self, query_embedding: np.ndarray) -> Dict[str, Any]:
        """Predict intent and domain for a query embedding.

        Returns::

            {"intent": "qa", "intent_confidence": 0.91,
             "domain": "resume", "domain_confidence": 0.85}

        Returns empty dict on error or dimension mismatch.
        """
        if query_embedding is None:
            return {}

        x = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != self.input_dim:
            return {}

        _, intent_probs, domain_probs = self._forward(x)
        intent_probs = intent_probs[0]
        domain_probs = domain_probs[0]

        intent_idx = int(np.argmax(intent_probs))
        domain_idx = int(np.argmax(domain_probs))

        return {
            "intent": self.intent_names[intent_idx],
            "intent_confidence": float(intent_probs[intent_idx]),
            "domain": self.domain_names[domain_idx],
            "domain_confidence": float(domain_probs[domain_idx]),
        }

    # --- Persistence ---

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "W_shared": self.W_shared,
            "b_shared": self.b_shared,
            "W_intent": self.W_intent,
            "b_intent": self.b_intent,
            "W_domain": self.W_domain,
            "b_domain": self.b_domain,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "n_intents": self.n_intents,
            "n_domains": self.n_domains,
            "intent_names": self.intent_names,
            "domain_names": self.domain_names,
            "_trained": self._trained,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        log.info("Intent classifier saved to %s", path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.W_shared = data["W_shared"]
        self.b_shared = data["b_shared"]
        self.W_intent = data["W_intent"]
        self.b_intent = data["b_intent"]
        self.W_domain = data["W_domain"]
        self.b_domain = data["b_domain"]
        self.input_dim = data.get("input_dim", 1024)
        self.hidden_dim = data.get("hidden_dim", 128)
        self.n_intents = data.get("n_intents", 8)
        self.n_domains = data.get("n_domains", 6)
        self.intent_names = data.get("intent_names", INTENT_NAMES[:self.n_intents])
        self.domain_names = data.get("domain_names", DOMAIN_NAMES[:self.n_domains])
        self._trained = data.get("_trained", True)
        log.info("Intent classifier loaded from %s", path)


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------
_classifier_instance: Optional[IntentDomainClassifier] = None
_classifier_lock = threading.Lock()

_DEFAULT_MODEL_PATH = Path("models/intent_classifier.pkl")


def get_intent_classifier() -> Optional[IntentDomainClassifier]:
    """Return the singleton classifier, or None if not yet initialized."""
    return _classifier_instance


def set_intent_classifier(clf: Optional[IntentDomainClassifier]) -> None:
    """Set the singleton classifier (used in tests)."""
    global _classifier_instance
    _classifier_instance = clf


def ensure_intent_classifier(
    embedder: Any,
    model_path: Optional[Path] = None,
) -> IntentDomainClassifier:
    """Load or train the intent classifier singleton.

    Thread-safe: only one thread will train/load at a time.
    """
    global _classifier_instance
    if _classifier_instance is not None:
        return _classifier_instance

    with _classifier_lock:
        if _classifier_instance is not None:
            return _classifier_instance

        path = model_path or _DEFAULT_MODEL_PATH
        clf = IntentDomainClassifier()

        if path.exists():
            try:
                clf.load(path)
                _classifier_instance = clf
                return clf
            except Exception:
                log.warning("Failed to load intent classifier from %s, retraining", path, exc_info=True)

        # Train from scratch
        clf.train(embedder)
        try:
            clf.save(path)
        except Exception:
            log.warning("Failed to save intent classifier to %s", path, exc_info=True)

        _classifier_instance = clf
        return clf
