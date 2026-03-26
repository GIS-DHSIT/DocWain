"""
3-stage autonomous document pipeline for Teams: Identify -> Screen -> Embed.

Each stage produces an Adaptive Card that is sent to the user in real time.
The pipeline supports pause/resume for consent flows (MEDIUM/HIGH/CRITICAL risk).
"""

import asyncio
import pickle
import uuid
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_CONTENT_CACHE_TTL = 7 * 24 * 3600  # 7 days


def _build_teams_collection_name(subscription_id: str, profile_id: str) -> str:
    """Build teams_-prefixed Qdrant collection name for isolation."""
    safe_sub = str(subscription_id).strip().replace(" ", "_").replace(":", "-")[:32]
    safe_prof = str(profile_id).strip().replace(" ", "_").replace(":", "-")[:32]
    return f"teams_{safe_sub}_{safe_prof}"


async def _send_card(turn_context, card: Dict[str, Any], text: Optional[str], log) -> Optional[str]:
    """Send an Adaptive Card via the Bot Framework turn context.

    Returns the activity ID so the card can be updated in-place later.
    """
    try:
        from src.teams.tools import _card_activity
        activity = _card_activity(card, text=text)
        resp = await turn_context.send_activity(activity)
        return getattr(resp, "id", None)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to send card: %s", exc, exc_info=True)
        return None


async def _update_card(turn_context, activity_id: Optional[str], card: Dict[str, Any], text: Optional[str], log) -> Optional[str]:
    """Update an existing card in-place, falling back to send if update fails.

    Returns the activity ID (same as input on success, new on fallback).
    """
    from src.teams.tools import _card_activity
    activity = _card_activity(card, text=text)

    if not activity_id:
        return await _send_card(turn_context, card, text, log)

    # Try in-place update first
    try:
        if isinstance(activity, dict):
            activity["id"] = activity_id
        elif hasattr(activity, "id"):
            activity.id = activity_id
        await turn_context.update_activity(activity)
        return activity_id
    except Exception:  # noqa: BLE001
        pass

    # Fallback: delete old + send new
    try:
        await turn_context.delete_activity(activity_id)
    except Exception:  # noqa: BLE001
        pass

    return await _send_card(turn_context, card, text, log)


class TeamsDocumentPipeline:
    """Orchestrates the 3-stage document pipeline: Identify -> Screen -> Embed."""

    def __init__(self, storage=None, state_store=None):
        if storage is None:
            from src.teams.teams_storage import TeamsDocumentStorage
            storage = TeamsDocumentStorage()
        if state_store is None:
            from src.teams.state import TeamsStateStore
            state_store = TeamsStateStore()
        self.storage = storage
        self.state_store = state_store

    # ── Stage 1: Identify ────────────────────────────────────────────

    async def stage_identify(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        context,
        correlation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract text, run DI pipeline, return identification card."""
        log = get_logger(__name__, correlation_id)
        document_id = str(uuid.uuid4())

        self.storage.create_document(
            document_id=document_id,
            filename=filename,
            subscription_id=context.subscription_id,
            profile_id=context.profile_id,
            content_type=content_type,
            content_size=len(file_bytes),
        )

        # Lazy import to avoid circular deps
        from src.api.dataHandler import fileProcessor

        extracted_docs = await asyncio.to_thread(fileProcessor, file_bytes, filename)
        if not extracted_docs:
            log.warning("No extractable content from %s", filename)
            return None

        # Collect all extracted text
        all_text_parts: List[str] = []
        first_extracted = None
        for doc_name, doc_content in extracted_docs.items():
            if first_extracted is None:
                first_extracted = doc_content
            if isinstance(doc_content, str):
                all_text_parts.append(doc_content)
            elif hasattr(doc_content, "full_text") and doc_content.full_text:
                all_text_parts.append(str(doc_content.full_text))
            elif hasattr(doc_content, "text") and doc_content.text:
                all_text_parts.append(str(doc_content.text))

        extracted_text = "\n\n".join(all_text_parts)

        # Run DI pipeline: identify -> content_map -> understand
        # Teams always uses cloud models for document processing
        from src.doc_understanding.identify import identify_document
        from src.doc_understanding.content_map import build_content_map
        from src.doc_understanding.understand import understand_document
        from src.llm.gateway import get_llm_gateway
        cloud_llm = get_llm_gateway()

        id_result = await asyncio.to_thread(
            identify_document, extracted=first_extracted, filename=filename, llm_client=cloud_llm,
        )
        doc_type = "other"
        confidence = 0.0
        if isinstance(id_result, dict):
            doc_type = id_result.get("document_type", "other")
            confidence = float(id_result.get("confidence", 0.0))
        else:
            doc_type = getattr(id_result, "document_type", "other") or "other"
            confidence = float(getattr(id_result, "confidence", 0.0))

        await asyncio.to_thread(build_content_map, first_extracted)

        # Understand ALL extracted documents concurrently
        async def _understand_one(doc_content):
            return await asyncio.to_thread(
                understand_document, extracted=doc_content, doc_type=doc_type, llm_client=cloud_llm,
            )

        understanding_tasks = [_understand_one(doc_content) for doc_content in extracted_docs.values()]
        all_understandings = await asyncio.gather(*understanding_tasks, return_exceptions=True)

        # Aggregate understanding from all documents
        summary_parts: List[str] = []
        key_entities: List[str] = []
        key_facts: List[str] = []
        intent_tags: List[str] = []
        seen_entities = set()
        seen_facts = set()

        for understanding in all_understandings:
            if isinstance(understanding, Exception):
                log.warning("Understanding failed for one sub-document: %s", understanding)
                continue
            if not isinstance(understanding, dict):
                continue
            s = understanding.get("document_summary", "")
            if s:
                summary_parts.append(s)
            for ent in understanding.get("key_entities", []):
                ent_key = str(ent.get("text", "")).lower() if isinstance(ent, dict) else str(ent).lower()
                if ent_key and ent_key not in seen_entities:
                    seen_entities.add(ent_key)
                    key_entities.append(ent if isinstance(ent, str) else ent.get("text", str(ent)))
            for fact in understanding.get("key_facts", []):
                fact_key = str(fact.get("fact", "")).lower() if isinstance(fact, dict) else str(fact).lower()
                if fact_key and fact_key not in seen_facts:
                    seen_facts.add(fact_key)
                    key_facts.append(fact if isinstance(fact, str) else fact.get("fact", str(fact)))
            for tag in understanding.get("intent_tags", []):
                if tag not in intent_tags:
                    intent_tags.append(tag)

        summary = " | ".join(summary_parts) if summary_parts else ""

        # Store extraction result in MongoDB
        self.storage.store_extraction_result(
            document_id=document_id,
            doc_type=doc_type,
            summary=summary,
            key_entities=key_entities,
            key_facts=key_facts,
            intent_tags=intent_tags,
            confidence=confidence,
        )

        # Generate proactive insights for card actions
        try:
            from src.teams.insights import generate_proactive_insights, get_domain_actions
            insights = await asyncio.to_thread(
                generate_proactive_insights,
                doc_type, summary, key_entities, key_facts, filename,
            )
            suggested_questions = insights.suggested_questions[:3] if insights.suggested_questions else []
            domain_actions = get_domain_actions(doc_type)
        except Exception:  # noqa: BLE001
            suggested_questions = []
            domain_actions = []

        # Build identification card
        from src.teams.cards import build_card

        entities_text = "\n".join(f"- {e}" for e in key_entities[:5]) if key_entities else "None detected"
        intent_text = ", ".join(intent_tags[:5]) if intent_tags else "general"
        confidence_pct = f"{confidence * 100:.0f}%" if confidence <= 1.0 else f"{confidence:.0f}%"

        # Pick action titles/queries from domain actions or suggested questions
        action1_title = domain_actions[0]["title"] if domain_actions else "Ask about this document"
        action1_query = domain_actions[0].get("query", "Summarize this document") if domain_actions else "Summarize this document"
        action2_title = domain_actions[1]["title"] if len(domain_actions) > 1 else "Key details"
        action2_query = domain_actions[1].get("query", "Extract key details") if len(domain_actions) > 1 else "Extract key details"

        card = build_card(
            "identification_card",
            doc_type=doc_type.title(),
            confidence=confidence_pct,
            summary=summary or "No summary available.",
            entities_text=entities_text,
            intent_text=intent_text,
            action1_title=action1_title,
            action1_query=action1_query,
            action2_title=action2_title,
            action2_query=action2_query,
        )

        log.info(
            "Stage 1 (identify) complete for %s: type=%s confidence=%s",
            filename, doc_type, confidence_pct,
        )

        return {
            "document_id": document_id,
            "doc_type": doc_type,
            "confidence": confidence,
            "summary": summary,
            "key_entities": key_entities,
            "key_facts": key_facts,
            "intent_tags": intent_tags,
            "extracted_text": extracted_text,
            "extracted_docs": extracted_docs,
            "first_extracted": first_extracted,
            "suggested_questions": suggested_questions,
            "card": card,
        }

    # ── Stage 2: Screen ──────────────────────────────────────────────

    async def stage_screen(
        self,
        document_id: str,
        extracted_text: str,
        filename: str,
        correlation_id: str,
    ) -> Dict[str, Any]:
        """Run security screening and return appropriate card."""
        log = get_logger(__name__, correlation_id)

        from src.teams.attachments import _run_security_screening
        from src.teams.teams_storage import CONSENT_REQUIRED_RISKS

        screening = await asyncio.to_thread(
            _run_security_screening,
            extracted_text[:50000],
            filename,
            document_id,
            log,
        )

        if screening is None:
            # Screening failed — treat as LOW risk, auto-proceed
            risk_level = "LOW"
            risk_score = 0.0
            findings: List[str] = ["Screening unavailable — defaulting to LOW risk"]
            log.warning("Screening failed for %s, defaulting to LOW risk", filename)
        else:
            risk_level = screening.risk_level
            risk_score = screening.overall_score
            findings = screening.top_findings

        needs_consent = risk_level.upper() in CONSENT_REQUIRED_RISKS

        # Store screening result
        self.storage.store_screening_result(
            document_id=document_id,
            risk_level=risk_level,
            risk_score=risk_score,
            findings=findings,
        )

        # Build appropriate card
        from src.teams.cards import build_card

        risk_score_text = f"{risk_score:.0f}/100"
        findings_text = "\n".join(f"- {f}" for f in findings) if findings else "No findings"

        if needs_consent:
            card = build_card(
                "screening_consent_card",
                risk_level=risk_level,
                risk_score=risk_score_text,
                findings_text=findings_text,
                document_id=document_id,
            )
        else:
            card = build_card(
                "screening_passed_card",
                risk_level=risk_level,
                risk_score=risk_score_text,
            )

        log.info(
            "Stage 2 (screen) complete for %s: risk=%s consent_needed=%s",
            filename, risk_level, needs_consent,
        )

        return {
            "needs_consent": needs_consent,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "findings": findings,
            "card": card,
        }

    # ── Stage 3: Embed ───────────────────────────────────────────────

    async def stage_embed(
        self,
        document_id: str,
        extracted_content: Dict[str, Any],
        filename: str,
        doc_type: str,
        context,
        correlation_id: str,
        suggested_questions: Optional[List[str]] = None,
        doc_intelligence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Chunk, embed into Teams-isolated Qdrant collection, return completion card."""
        log = get_logger(__name__, correlation_id)

        self.storage.mark_embedding_started(document_id)

        try:
            # Build teams-scoped collection name
            collection_name = _build_teams_collection_name(
                context.subscription_id, context.profile_id,
            )

            # Ensure Qdrant collection exists
            from qdrant_client import QdrantClient
            from src.api.vector_store import QdrantVectorStore
            from src.api.config import Config

            client = QdrantClient(
                url=Config.Qdrant.URL,
                api_key=Config.Qdrant.API,
                timeout=120,
            )
            vector_store = QdrantVectorStore(client)
            vector_size = int(getattr(Config.Model, "EMBEDDING_DIM", 1024))
            vector_store.ensure_collection(collection_name, vector_size)

            # Enrich extracted content with DI context for better embedding quality
            if doc_intelligence:
                di_context = []
                if doc_intelligence.get("summary"):
                    di_context.append(f"Document Summary: {doc_intelligence['summary']}")
                if doc_intelligence.get("key_entities"):
                    entities_str = ", ".join(str(e) for e in doc_intelligence["key_entities"][:10])
                    di_context.append(f"Key Entities: {entities_str}")
                if doc_intelligence.get("doc_type"):
                    di_context.append(f"Document Type: {doc_intelligence['doc_type']}")

                if di_context:
                    di_prefix = "\n".join(di_context) + "\n\n---\n\n"
                    enriched_content = {}
                    for doc_name, doc_content in extracted_content.items():
                        if isinstance(doc_content, str):
                            enriched_content[doc_name] = di_prefix + doc_content
                        elif hasattr(doc_content, "full_text") and isinstance(getattr(doc_content, "full_text", None), str):
                            enriched_content[doc_name] = di_prefix + doc_content.full_text
                        elif hasattr(doc_content, "text") and isinstance(getattr(doc_content, "text", None), str):
                            enriched_content[doc_name] = di_prefix + doc_content.text
                        else:
                            enriched_content[doc_name] = doc_content
                    extracted_content = enriched_content

            # Embed each extracted doc
            from src.api.dataHandler import train_on_document

            chunks_count = 0
            for doc_name, doc_content in extracted_content.items():
                result = await asyncio.to_thread(
                    train_on_document,
                    doc_content,
                    subscription_id=collection_name,  # Use teams collection as subscription_id
                    profile_id=context.profile_id,
                    doc_tag=document_id,
                    doc_name=doc_name,
                )
                if isinstance(result, int):
                    chunks_count += result
                else:
                    chunks_count += 1

            quality_grade = "A" if chunks_count > 10 else "B" if chunks_count > 3 else "C"

            # Mark embedding complete
            self.storage.mark_embedding_completed(
                document_id,
                chunks_count=chunks_count,
                quality_grade=quality_grade,
            )

            # Record upload in state store
            self.state_store.record_upload(
                context.subscription_id,
                context.profile_id,
                filename,
                document_id,
                chunks_count,
                document_type=doc_type,
            )

            # Build completion card
            from src.teams.cards import build_card
            from src.teams.insights import _fallback_questions, _normalize_domain

            questions = suggested_questions or []
            if len(questions) < 3:
                domain = _normalize_domain(doc_type)
                fallbacks = _fallback_questions(domain)
                questions.extend(fallbacks[: 3 - len(questions)])

            quality_labels = {"A": "A (Excellent)", "B": "B (Good)", "C": "C (Fair)"}
            quality_text = quality_labels.get(quality_grade, quality_grade)

            # Get domain actions for the card
            from src.teams.insights import get_domain_actions
            domain_actions = get_domain_actions(doc_type)
            action1_title = domain_actions[0]["title"] if domain_actions else "Ask a question"
            action1_query = domain_actions[0].get("query", "What is this document about?") if domain_actions else "What is this document about?"

            card = build_card(
                "embedding_complete_card",
                filename=filename,
                chunks_count=str(chunks_count),
                quality_text=quality_text,
                question1=questions[0] if len(questions) > 0 else "Summarize this document",
                question2=questions[1] if len(questions) > 1 else "What are the key details?",
                question3=questions[2] if len(questions) > 2 else "Extract important dates",
                action1_title=action1_title,
                action1_query=action1_query,
            )

            log.info(
                "Stage 3 (embed) complete for %s: chunks=%d grade=%s collection=%s",
                filename, chunks_count, quality_grade, collection_name,
            )

            return {
                "chunks_count": chunks_count,
                "quality_grade": quality_grade,
                "card": card,
            }

        except Exception as exc:  # noqa: BLE001
            log.error("Embedding failed for %s: %s", filename, exc, exc_info=True)
            self.storage.mark_embedding_failed(document_id, error=str(exc))

            from src.teams.cards import build_card
            card = build_card(
                "error_card",
                message=f"Embedding failed for {filename}. Please try uploading again.",
            )
            return {
                "chunks_count": 0,
                "quality_grade": "",
                "error": str(exc),
                "card": card,
            }

    # ── Full pipeline ────────────────────────────────────────────────

    async def run_full_pipeline(
        self,
        attachment: Dict[str, Any],
        context,
        turn_context,
        correlation_id: str,
        auth_token: str = "",
    ) -> None:
        """Run the complete Identify -> Screen -> Embed pipeline with parallel stages."""
        log = get_logger(__name__, correlation_id)

        from src.teams.attachments import (
            _resolve_filename,
            _resolve_download_url,
            _download_bytes,
            _build_download_headers,
            _upload_to_blob,
        )
        from src.api.config import Config
        from src.teams.cards import build_card

        filename = _resolve_filename(attachment)
        download_url = _resolve_download_url(attachment)
        content_type = (attachment.get("contentType") or "").lower()

        if not download_url:
            log.error("No download URL for attachment %s", filename)
            return

        # Download file
        timeout = float(getattr(Config.Teams, "HTTP_TIMEOUT_SEC", 20))
        retries = int(getattr(Config.Teams, "HTTP_RETRIES", 2))
        max_bytes = int(getattr(Config.Teams, "MAX_ATTACHMENT_MB", 50)) * 1024 * 1024
        headers = _build_download_headers(auth_token)

        try:
            file_bytes = await _download_bytes(
                download_url, headers=headers,
                timeout=timeout, retries=retries, max_bytes=max_bytes,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Download failed for %s: %s", filename, exc)
            return

        log.info("Downloaded %s: %d bytes", filename, len(file_bytes))

        # Kick off blob upload (non-blocking)
        asyncio.create_task(
            asyncio.to_thread(_upload_to_blob, file_bytes, filename, context.subscription_id, log)
        )

        # ── Single progress card updated in-place across all stages ──
        progress_card = build_card(
            "stage_progress_card",
            step_indicator="1/3",
            stage_title=f"Analyzing: {filename}",
            stage_detail="Extracting content and identifying document type...",
            progress_bar="[██░░░░░░░░░░░░] 33%",
        )
        progress_id = await _send_card(turn_context, progress_card, "Step 1/3: Analyzing document...", log)

        # Stage 1: Identify
        result = await self.stage_identify(
            file_bytes, filename, content_type, context, correlation_id,
        )
        if result is None:
            # Replace progress card with error
            error_card = build_card("error_card", message=f"No extractable content found in {filename}.")
            await _update_card(turn_context, progress_id, error_card, "No content found.", log)
            return

        # Update progress → Stage 2
        progress_card = build_card(
            "stage_progress_card",
            step_indicator="2/3",
            stage_title=f"Screening: {filename}",
            stage_detail=f"Document identified as {result['doc_type'].title()} ({result['confidence']*100:.0f}% confidence). Running security checks...",
            progress_bar="[██████░░░░░░░░] 50%",
        )
        progress_id = await _update_card(turn_context, progress_id, progress_card, "Step 2/3: Screening...", log)

        # Cache extracted content for consent flow
        self.cache_content(result["document_id"], result["extracted_docs"])

        # Stage 2: Screen
        screen_result = await self.stage_screen(
            result["document_id"],
            result["extracted_text"],
            filename,
            correlation_id,
        )

        if screen_result["needs_consent"]:
            # Replace progress card with consent card (user must interact)
            await _update_card(turn_context, progress_id, screen_result["card"], "Consent required.", log)
            log.info("Pipeline paused for consent: doc=%s", result["document_id"])
            return

        # Update progress → Stage 3
        progress_card = build_card(
            "stage_progress_card",
            step_indicator="3/3",
            stage_title=f"Embedding: {filename}",
            stage_detail=f"Security: {screen_result['risk_level']} risk. Chunking and indexing for intelligent retrieval...",
            progress_bar="[██████████░░░░] 80%",
        )
        progress_id = await _update_card(turn_context, progress_id, progress_card, "Step 3/3: Embedding...", log)

        # Stage 3: Embed
        embed_result = await self.stage_embed(
            document_id=result["document_id"],
            extracted_content=result["extracted_docs"],
            filename=filename,
            doc_type=result["doc_type"],
            context=context,
            correlation_id=correlation_id,
            suggested_questions=result.get("suggested_questions"),
            doc_intelligence={
                "summary": result.get("summary", ""),
                "key_entities": result.get("key_entities", []),
                "doc_type": result.get("doc_type", ""),
                "key_facts": result.get("key_facts", []),
            },
        )

        # Replace progress card with final completion card
        await _update_card(turn_context, progress_id, embed_result["card"], "Document ready.", log)

    # ── Resume after consent ─────────────────────────────────────────

    async def resume_after_consent(
        self,
        document_id: str,
        context,
        turn_context,
        correlation_id: str,
    ) -> None:
        """Resume pipeline after user consents to embed a flagged document."""
        log = get_logger(__name__, correlation_id)

        self.storage.consent_proceed(document_id)

        doc = self.storage.get_document(document_id)
        if not doc:
            log.error("Document %s not found for consent resume", document_id)
            return

        filename = doc.get("filename", "unknown")
        doc_type = doc.get("extraction", {}).get("doc_type", "other")

        # Retrieve cached content
        extracted_content = self._get_cached_content(document_id)
        if not extracted_content:
            log.error("No cached content for document %s", document_id)
            from src.teams.cards import build_card
            card = build_card(
                "error_card",
                message="Cached content expired. Please re-upload the document.",
            )
            await _send_card(turn_context, card, "Content expired.", log)
            return

        embed_result = await self.stage_embed(
            document_id=document_id,
            extracted_content=extracted_content,
            filename=filename,
            doc_type=doc_type,
            context=context,
            correlation_id=correlation_id,
            doc_intelligence={
                "doc_type": doc_type,
                "summary": doc.get("extraction", {}).get("summary", ""),
                "key_entities": doc.get("extraction", {}).get("key_entities", []),
            },
        )
        await _send_card(turn_context, embed_result["card"], "Document ready.", log)

    # ── Content caching ──────────────────────────────────────────────

    def cache_content(self, document_id: str, content: Any) -> None:
        """Store extracted content in Redis with pickle, TTL 7 days."""
        key = f"teams:pipeline:content:{document_id}"
        redis_client = getattr(self.state_store, "client", None)
        if redis_client:
            try:
                redis_client.setex(key, _CONTENT_CACHE_TTL, pickle.dumps(content))
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to cache content in Redis: %s", exc)
        # Fallback: store in memory on the state store
        if not hasattr(self.state_store, "_pipeline_cache"):
            self.state_store._pipeline_cache = {}
        self.state_store._pipeline_cache[key] = content

    def _get_cached_content(self, document_id: str) -> Optional[Any]:
        """Retrieve pickled content from Redis or memory fallback."""
        key = f"teams:pipeline:content:{document_id}"
        redis_client = getattr(self.state_store, "client", None)
        if redis_client:
            try:
                raw = redis_client.get(key)
                if raw:
                    return pickle.loads(raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to retrieve cached content from Redis: %s", exc)
        # Fallback: check in-memory cache
        cache = getattr(self.state_store, "_pipeline_cache", {})
        return cache.get(key)
