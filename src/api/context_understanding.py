from src.utils.logging_utils import get_logger
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.api.pipeline_models import ExtractedDocument, Section

logger = get_logger(__name__)

try:
    import ollama
except Exception:  # noqa: BLE001
    ollama = None

def list_local_models() -> List[str]:
    """Return available local Ollama models (empty list when unavailable)."""
    if not ollama:
        return []
    try:
        result = ollama.list()
        # ollama SDK returns a Pydantic ListResponse with .models attribute
        models = getattr(result, "models", None) or result.get("models", []) if isinstance(result, dict) else getattr(result, "models", [])
        names = []
        for m in models:
            name = getattr(m, "model", None) or (m.get("name") if isinstance(m, dict) else None)
            if name:
                names.append(name)
        return names
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to list ollama models: %s", exc)
        return []

class ContextUnderstanding:
    """Generates document and section summaries using local models or extractive fallback."""

    def __init__(self, preferred_model: Optional[str] = None):
        self.available_models = list_local_models()
        self.model_name = preferred_model or self._select_model()
        if self.model_name:
            logger.info("Using local model for context understanding: %s", self.model_name)
        else:
            logger.info("No local summarization model available; using extractive summaries")

    def _select_model(self) -> Optional[str]:
        if not self.available_models:
            return None
        # Prefer DocWain fine-tuned model, then general-purpose models
        for candidate in ("DHS/DocWain", "DocWain", "qwen3", "llama3.2", "llama3.1", "llama3", "mistral"):
            for model in self.available_models:
                if candidate in model:
                    return model
        return self.available_models[0]

    @staticmethod
    def _clean_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text or "")
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    def _extractive_summary(self, text: str, max_sentences: int = 4) -> str:
        sentences = self._clean_sentences(text)
        if not sentences:
            return ""
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        try:
            tfidf = vectorizer.fit_transform(sentences)
            scores = np.asarray(tfidf.sum(axis=1)).ravel()
            top_indices = np.argsort(scores)[::-1][:max_sentences]
            ordered = [sentences[i] for i in sorted(top_indices)]
            return " ".join(ordered)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Extractive summary failed: %s", exc)
            return " ".join(sentences[:max_sentences])

    def _llm_summary(self, text: str, instruction: str) -> str:
        if not self.model_name:
            return ""
        prompt = (
            f"{instruction}\n\nTEXT:\n{text}\n\nReturn 3-5 bullet points or a short paragraph."
        )
        try:
            from src.llm.gateway import get_llm_gateway
            content = get_llm_gateway().generate(prompt)
            return (content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Local model summary failed: %s", exc)
            return ""

    def summarize_section(self, section: Section) -> str:
        summary = self._llm_summary(section.text, f"Summarize the section titled '{section.title}'.")
        if not summary:
            summary = self._extractive_summary(section.text, max_sentences=3)
        return summary.strip()

    def summarize_document(self, extracted: ExtractedDocument) -> Dict[str, Any]:
        full_text = extracted.full_text or ""
        section_summaries: Dict[str, str] = {}
        for section in extracted.sections:
            section_summaries[section.section_id] = self.summarize_section(section)

        abstract = self._llm_summary(full_text, "Provide a concise abstract of the document.")
        if not abstract:
            abstract = self._extractive_summary(full_text, max_sentences=5)

        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", full_text.lower())
        counts = Counter(tokens)
        topics = [w for w, _ in counts.most_common(10)]

        glossary = []
        for term, _ in counts.most_common(30):
            if term.isdigit() or len(term) < 4:
                continue
            glossary.append(term)
            if len(glossary) >= 12:
                break

        title_guess = extracted.sections[0].title if extracted.sections else "Document"

        return {
            "title": title_guess,
            "abstract": abstract.strip(),
            "topics": topics,
            "glossary": glossary,
            "section_summaries": section_summaries,
        }

    @staticmethod
    def attach_summaries_to_chunks(chunk_metadata: List[dict], section_summaries: Dict[str, str]) -> List[str]:
        summaries: List[str] = []
        for meta in chunk_metadata:
            section_id = meta.get("section_id")
            summaries.append(section_summaries.get(section_id, ""))
        return summaries
