"""
Training data bootstrapper for DPIE.

Generates training data for all DPIE models from DocWain's existing
Qdrant data and metadata -- no manual labelling needed.

Sources:
    1. Qdrant payloads: chunk metadata with section_title, doc_type, source_file.
    2. Document text: reconstructed from ordered chunks.
    3. Knowledge-graph entities: from ``kg/entity_extractor`` existing extractions.
    4. Section kind lexicon: weak labels from ``section_intelligence_builder``.

No regex anywhere in this module.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Section kind lexicon (weak labelling from keyword presence -- no regex).
# Keys are section kinds, values are keywords to look for with ``in``.
_SECTION_KIND_LEXICON: Dict[str, Tuple[str, ...]] = {
    "identity_contact": ("contact", "address", "phone", "email", "profile", "candidate"),
    "summary_objective": ("summary", "objective", "about", "overview", "introduction"),
    "experience": ("experience", "employment", "work history", "professional experience"),
    "projects": ("project", "portfolio"),
    "education": ("education", "academic", "qualification", "degree", "university"),
    "certifications": ("certification", "certificate", "license", "accreditation"),
    "skills_technical": ("technical skill", "programming", "technologies", "tech stack"),
    "skills_functional": ("skill", "competenc", "abilit"),
    "tools_technologies": ("tool", "software", "platform", "framework"),
    "achievements_awards": ("achievement", "award", "honor", "recognition"),
    "publications_patents": ("publication", "patent", "paper", "journal"),
    "leadership_management": ("leadership", "management", "team lead", "supervisor"),
    "compliance_regulatory": ("compliance", "regulation", "governance", "audit"),
    "invoice_metadata": ("invoice", "invoice number", "invoice date", "due date", "bill date"),
    "financial_summary": ("subtotal", "total", "amount due", "balance", "summary"),
    "transactions": ("transaction", "payment", "transfer", "deposit"),
    "line_items": ("item", "description", "quantity", "unit price", "line item"),
    "parties_addresses": ("bill to", "ship to", "vendor", "supplier", "buyer", "seller"),
    "terms_conditions": ("term", "condition", "payment term", "warranty", "liability"),
    "totals": ("total", "grand total", "net total", "tax total"),
    "misc": (),
}


class TrainingBootstrap:
    """Bootstraps training data from DocWain's existing infrastructure.

    Uses Qdrant payloads, reconstructed document text, and the existing
    regex-based entity extractor to create labelled datasets for all
    DPIE model components.
    """

    def __init__(
        self,
        qdrant_client: Any,
        embedding_model: Any,
        collection_name: str,
        subscription_id: str = "",
    ) -> None:
        self._qdrant = qdrant_client
        self._embedding_model = embedding_model
        self._collection = collection_name
        self._subscription_id = subscription_id

    # -- document type training data ---------------------------------------

    def generate_doc_type_data(
        self,
        profile_id: str,
        max_docs: int = 200,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Generate document-type classification training data.

        Scrolls Qdrant for all documents in a profile, reconstructs their
        text from chunks, encodes lines, and uses the stored ``doc_type``
        as the label.

        Args:
            profile_id: Profile to extract data from.
            max_docs: Maximum number of documents to process.

        Returns:
            ``(features_list, labels_list)`` where each entry in features
            is a ``(num_lines, 852)`` array and labels are doc-type strings.
        """
        from .line_encoder import LineFeatureEncoder

        docs = self._get_documents_for_profile(profile_id)
        encoder = LineFeatureEncoder(self._embedding_model)

        features_list: List[np.ndarray] = []
        labels_list: List[str] = []

        for doc_id, chunks in list(docs.items())[:max_docs]:
            text = self._reconstruct_text(chunks)
            if not text.strip():
                continue

            doc_type = self._get_doc_type(chunks)
            if not doc_type:
                continue

            line_features, _ = encoder.encode_document(text)
            if line_features.shape[0] == 0:
                continue

            features_list.append(line_features)
            labels_list.append(doc_type)

        logger.info("Generated doc-type data: %d documents", len(features_list))
        return features_list, labels_list

    # -- section boundary training data ------------------------------------

    def generate_section_boundary_data(
        self,
        profile_id: str,
        max_docs: int = 200,
    ) -> Tuple[List[np.ndarray], List[List[bool]]]:
        """Generate section-boundary detection training data.

        Uses ``section.title`` changes between consecutive chunks as
        boundary signals.

        Args:
            profile_id: Profile to extract data from.
            max_docs: Maximum number of documents.

        Returns:
            ``(features_list, boundary_labels_list)``
        """
        from .line_encoder import LineFeatureEncoder

        docs = self._get_documents_for_profile(profile_id)
        encoder = LineFeatureEncoder(self._embedding_model)

        features_list: List[np.ndarray] = []
        boundary_labels_list: List[List[bool]] = []

        for doc_id, chunks in list(docs.items())[:max_docs]:
            text = self._reconstruct_text(chunks)
            if not text.strip():
                continue

            line_features, lines = encoder.encode_document(text)
            if line_features.shape[0] == 0:
                continue

            # Derive boundary labels from section title changes
            boundaries = self._derive_boundaries(chunks, lines)
            if len(boundaries) != len(lines):
                # Fallback: first line is boundary, rest are not
                boundaries = [i == 0 for i in range(len(lines))]

            features_list.append(line_features)
            boundary_labels_list.append(boundaries)

        logger.info("Generated boundary data: %d documents", len(features_list))
        return features_list, boundary_labels_list

    # -- section kind training data ----------------------------------------

    def generate_section_kind_data(self, profile_id: str) -> List[Dict[str, str]]:
        """Generate section-kind classification examples.

        Uses stored ``section.kind`` from payload when available, otherwise
        falls back to weak labelling via keyword matching (using ``in``,
        not regex).

        Args:
            profile_id: Profile to extract data from.

        Returns:
            List of dicts with ``title``, ``content``, ``kind`` keys.
        """
        points = self._scroll_profile(profile_id, max_points=500)
        seen_sections: Set[str] = set()
        examples: List[Dict[str, str]] = []

        for point in points:
            payload = getattr(point, "payload", None) or {}
            # Flat fields (post-rebuild) with nested fallback (pre-rebuild)
            section_title = (
                payload.get("section_title", "")
                or (payload.get("section", {}) or {}).get("title", "")
                or ""
            )
            section_id = (
                payload.get("section_id", "")
                or (payload.get("section", {}) or {}).get("id", "")
                or ""
            )

            dedup_key = f"{section_id}|{section_title}"
            if dedup_key in seen_sections:
                continue
            seen_sections.add(dedup_key)

            # Get section content from canonical_text
            content = (
                payload.get("canonical_text", "")
                or payload.get("embedding_text", "")
                or ""
            )[:500]

            # Determine section kind
            section_kind = (
                payload.get("section_kind", "")
                or (payload.get("section", {}) or {}).get("kind", "")
                or ""
            )
            if not section_kind:
                section_kind = self._weak_label_section_kind(section_title)

            if not section_kind or not section_title:
                continue

            examples.append({
                "title": section_title,
                "content": content,
                "kind": section_kind,
            })

        logger.info("Generated section-kind data: %d examples", len(examples))
        return examples

    # -- entity training data ----------------------------------------------

    def generate_entity_data(
        self,
        profile_id: str,
        max_chunks: int = 500,
    ) -> List[Dict[str, Any]]:
        """Generate entity recognition training data.

        Runs the *existing* regex-based entity extractor on chunks to
        bootstrap labels for the ML replacement.

        Args:
            profile_id: Profile to extract data from.
            max_chunks: Maximum chunks to process.

        Returns:
            List of dicts with ``text`` and ``entities`` keys.
        """
        try:
            from src.kg.entity_extractor import EntityExtractor
            extractor = EntityExtractor()
        except Exception as exc:
            logger.warning("Cannot import EntityExtractor: %s", exc)
            return []

        points = self._scroll_profile(profile_id, max_points=max_chunks)
        examples: List[Dict[str, Any]] = []

        for point in points:
            payload = getattr(point, "payload", None) or {}
            text = (
                payload.get("canonical_text", "")
                or payload.get("embedding_text", "")
                or ""
            )

            if not text or len(text) < 20:
                continue

            try:
                extracted = extractor.extract_with_metadata(text)
            except Exception:
                try:
                    extracted = extractor.extract(text)
                except Exception:
                    continue

            entities: List[Dict[str, Any]] = []
            for ent in extracted:
                name = getattr(ent, "name", "") or ""
                ent_type = getattr(ent, "type", "") or ""
                if not name or not ent_type:
                    continue

                # Locate entity surface form in text (using str.find, not regex)
                start = text.find(name)
                if start == -1:
                    # Try case-insensitive
                    start = text.lower().find(name.lower())
                if start == -1:
                    continue

                end = start + len(name)
                entities.append({
                    "span": name,
                    "type": ent_type.upper(),
                    "start": start,
                    "end": end,
                })

            if entities:
                examples.append({"text": text, "entities": entities})

        logger.info("Generated entity data: %d examples", len(examples))
        return examples

    # -- generate all ------------------------------------------------------

    def generate_all(self, profile_id: str) -> Dict[str, Any]:
        """Run all generators and return packaged training data.

        Args:
            profile_id: Profile to bootstrap from.

        Returns:
            Dict with keys ``doc_type``, ``section_boundary``,
            ``section_kind``, ``entity``.
        """
        doc_features, doc_labels = self.generate_doc_type_data(profile_id)
        boundary_features, boundary_labels = self.generate_section_boundary_data(profile_id)
        section_kind_examples = self.generate_section_kind_data(profile_id)
        entity_examples = self.generate_entity_data(profile_id)

        return {
            "doc_type": (doc_features, doc_labels),
            "section_boundary": (boundary_features, boundary_labels),
            "section_kind": section_kind_examples,
            "entity": entity_examples,
        }

    # -- internal helpers --------------------------------------------------

    def _scroll_profile(self, profile_id: str, max_points: int = 500) -> List[Any]:
        """Scroll all points for a profile from Qdrant."""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            must_conditions = [
                FieldCondition(key="profile_id", match=MatchValue(value=profile_id)),
            ]
            # Defense-in-depth: also filter by subscription_id when available
            if self._subscription_id:
                must_conditions.append(
                    FieldCondition(key="subscription_id", match=MatchValue(value=self._subscription_id)),
                )
            filt = Filter(must=must_conditions)

            all_points: List[Any] = []
            offset = None
            while len(all_points) < max_points:
                result = self._qdrant.scroll(
                    collection_name=self._collection,
                    scroll_filter=filt,
                    limit=min(100, max_points - len(all_points)),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points, next_offset = result
                all_points.extend(points)
                if not next_offset or not points:
                    break
                offset = next_offset

            return all_points
        except Exception as exc:
            logger.warning("Qdrant scroll failed: %s", exc)
            return []

    def _get_documents_for_profile(
        self,
        profile_id: str,
    ) -> Dict[str, List[Any]]:
        """Group points by document_id, sorted by chunk index."""
        points = self._scroll_profile(profile_id, max_points=1000)
        docs: Dict[str, List[Any]] = defaultdict(list)

        for point in points:
            payload = getattr(point, "payload", None) or {}
            doc_id = payload.get("document_id", "")
            if doc_id:
                docs[doc_id].append(point)

        # Sort chunks within each document by chunk index
        for doc_id in docs:
            docs[doc_id].sort(key=lambda p: self._get_chunk_index(p))

        return dict(docs)

    @staticmethod
    def _get_chunk_index(point: Any) -> int:
        """Extract chunk index from a Qdrant point."""
        payload = getattr(point, "payload", None) or {}
        # Flat field (post-rebuild) with nested fallback (pre-rebuild)
        idx = payload.get("chunk_index") or (payload.get("chunk", {}) or {}).get("index", 0)
        try:
            return int(idx)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _reconstruct_text(chunks: List[Any]) -> str:
        """Reconstruct document text from ordered chunks."""
        parts: List[str] = []
        for point in chunks:
            payload = getattr(point, "payload", None) or {}
            text = (
                payload.get("canonical_text", "")
                or payload.get("embedding_text", "")
                or ""
            )
            if text.strip():
                parts.append(text.strip())
        return "\n\n".join(parts)

    @staticmethod
    def _get_doc_type(chunks: List[Any]) -> str:
        """Extract document type from chunk payloads."""
        for point in chunks:
            payload = getattr(point, "payload", None) or {}
            doc_type = (
                payload.get("doc_domain", "")
                or payload.get("document_type", "")
                or ""
            )
            if doc_type:
                return doc_type
        return ""

    def _derive_boundaries(
        self,
        chunks: List[Any],
        lines: List[str],
    ) -> List[bool]:
        """Derive boundary labels from section title changes in chunks.

        Maps chunk section titles to reconstructed lines to create
        boundary labels.
        """
        # Build a list of (chunk_text, section_title) pairs
        chunk_sections: List[Tuple[str, str]] = []
        for point in chunks:
            payload = getattr(point, "payload", None) or {}
            text = (
                payload.get("canonical_text", "")
                or payload.get("embedding_text", "")
                or ""
            ).strip()
            title = (
                payload.get("section_title", "")
                or (payload.get("section", {}) or {}).get("title", "")
                or ""
            )
            if text:
                chunk_sections.append((text, title))

        if not chunk_sections:
            return [i == 0 for i in range(len(lines))]

        # Map each line to the chunk it likely belongs to
        boundaries: List[bool] = []
        current_chunk_idx = 0
        prev_title = ""

        for i, line in enumerate(lines):
            # Advance chunk pointer if the line belongs to a later chunk
            while (
                current_chunk_idx + 1 < len(chunk_sections)
                and line.strip()
                and line.strip() in chunk_sections[current_chunk_idx + 1][0]
                and line.strip() not in chunk_sections[current_chunk_idx][0]
            ):
                current_chunk_idx += 1

            if current_chunk_idx < len(chunk_sections):
                curr_title = chunk_sections[current_chunk_idx][1]
            else:
                curr_title = ""

            is_boundary = (i == 0) or (curr_title != prev_title and curr_title != "")
            boundaries.append(is_boundary)
            prev_title = curr_title

        return boundaries

    @staticmethod
    def _weak_label_section_kind(title: str) -> str:
        """Assign a section kind using keyword matching (no regex).

        Args:
            title: Section heading text.

        Returns:
            Section kind string, or ``"misc"`` if no match.
        """
        title_lower = title.lower()

        for kind, keywords in _SECTION_KIND_LEXICON.items():
            for kw in keywords:
                if kw in title_lower:
                    return kind

        return "misc"
