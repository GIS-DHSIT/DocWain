import asyncio
from src.utils.logging_utils import get_logger
from typing import Any, Dict, Optional

from src.api.config import Config
from src.teams.cards import build_card
from src.teams.logic import TeamsChatContext, TeamsChatService
from src.teams.state import TeamsStateStore

logger = get_logger(__name__)

ADAPTIVE_CARD_TYPE = "application/vnd.microsoft.card.adaptive"
DEFAULT_WEB_URL = getattr(Config.Teams, "WEB_APP_URL", "https://www.docwain.ai")

_MAX_INPUT_LEN = 2000
_VALID_PRESETS = frozenset({"invoice", "contract", "hr", "medical", "legal", "policy"})

def _card_activity(card: Dict[str, Any], text: Optional[str] = None) -> Dict[str, Any]:
    activity: Dict[str, Any] = {
        "type": "message",
        "attachments": [
            {
                "contentType": ADAPTIVE_CARD_TYPE,
                "content": card,
            }
        ],
    }
    if text:
        activity["text"] = text
    return activity

def format_text_answer(
    response_text: str,
    sources: list,
    domain: str = "",
    grounded: bool = False,
) -> Dict[str, Any]:
    """Build a plain-text Teams activity for Q&A answers.

    Cards are reserved for status/progress indicators only.
    """
    parts: list[str] = []

    # Domain badge
    if domain and domain not in {"generic", "general", "other", ""}:
        parts.append(f"[{domain.title()}]")

    # Confidence
    source_count = len(sources) if sources else 0
    if grounded and source_count >= 3:
        parts.append(f"High confidence ({source_count} sources)")
    elif grounded and source_count >= 1:
        parts.append(f"Partial confidence ({source_count} source{'s' if source_count > 1 else ''})")
    elif source_count > 0:
        parts.append(f"Low confidence ({source_count} source{'s' if source_count > 1 else ''})")

    # Header line
    header = " | ".join(parts) if parts else ""

    # Build message
    lines: list[str] = []
    if header:
        lines.append(header)
        lines.append("")
    lines.append(response_text)

    # Inline sources
    if sources:
        lines.append("")
        lines.append("---")
        lines.append(f"Sources ({source_count}):")
        for src in sources[:5]:
            name = src.get("source_name") or src.get("source_id") or src.get("document_id") or "Source"
            excerpt = (src.get("excerpt") or "")[:150]
            if excerpt:
                lines.append(f"- **{name}**: {excerpt}")
            else:
                lines.append(f"- **{name}**")

    return {
        "type": "message",
        "text": "\n".join(lines),
    }

class TeamsToolRouter:
    """Routes Teams Adaptive Card Submit actions to DocWain tools."""

    def __init__(self, chat_service: TeamsChatService, state_store: TeamsStateStore):
        self.chat_service = chat_service
        self.state_store = state_store

    async def _ask_docwain(self, question: str, context: TeamsChatContext) -> Dict[str, Any]:
        try:
            from src.api import dw_newron

            answer = await asyncio.to_thread(
                dw_newron.answer_question,
                query=question,
                user_id=context.user_id,
                profile_id=context.profile_id,
                subscription_id=context.subscription_id,
                model_name=context.model_name,
                persona=context.persona,
                session_id=context.session_id,
            )
            return answer
        except Exception as exc:  # noqa: BLE001
            logger.error("DocWain question failed: %s", exc, exc_info=True)
            raise

    def _apply_preferences(self, context: TeamsChatContext) -> TeamsChatContext:
        prefs = self.state_store.get_preferences(context.subscription_id, context.profile_id)
        model_override = prefs.get("model_name")
        persona_override = prefs.get("persona")
        if model_override or persona_override:
            return TeamsChatService.build_context(
                user_id=context.user_id,
                session_id=context.session_id,
                model_name=model_override or context.model_name,
                persona=persona_override or context.persona,
            )
        return context

    async def handle_action(
        self,
        action_value: Dict[str, Any],
        context: TeamsChatContext,
    ) -> Dict[str, Any]:
        context = self._apply_preferences(context)
        action = str(action_value.get("action") or action_value.get("type") or "").lower()

        # Sanitize user inputs
        for key in ("query", "text", "preset", "model", "persona", "content_type"):
            val = action_value.get(key)
            if isinstance(val, str) and len(val) > _MAX_INPUT_LEN:
                action_value[key] = val[:_MAX_INPUT_LEN]

        if action in {"summarize", "summarize_recent"}:
            return await self._summarize_recent(context)
        if action == "extract_fields":
            return await self._extract_fields(context, preset=action_value.get("preset"))
        if action == "list_docs":
            return self._list_docs(context)
        if action == "set_model":
            return self._set_model(context, action_value.get("model"))
        if action == "set_persona":
            return self._set_persona(context, action_value.get("persona"))
        if action == "generate_content":
            return await self._generate_content(
                context,
                content_type=action_value.get("content_type"),
                query=action_value.get("query") or action_value.get("text", ""),
            )
        if action == "show_preferences":
            return self._show_preferences(context)
        if action == "set_preferences":
            return self._set_preferences(context, action_value)
        if action in {"open_ui", "open_web"}:
            return self._open_ui()
        if action == "delete_documents":
            return await self._delete_documents(context)
        if action == "confirm_delete":
            return await self._confirm_delete(context)
        if action == "domain_query":
            return await self._domain_query(context, query=action_value.get("query", ""))
        if action == "pipeline_consent_proceed":
            return await self._pipeline_consent_proceed(context, document_id=action_value.get("document_id", ""))
        if action == "pipeline_consent_reject":
            return await self._pipeline_consent_reject(context, document_id=action_value.get("document_id", ""))
        if action in {"help", "show_tools"}:
            return self._tools_menu(context)

        return _card_activity(build_card("help_card"))

    async def _summarize_recent(self, context: TeamsChatContext) -> Dict[str, Any]:
        uploads = self.state_store.list_uploads(context.subscription_id, context.profile_id, limit=3)
        if not uploads:
            return _card_activity(
                build_card("error_card", message="No uploads found yet. Upload a document and try again.")
            )
        doc_tags = [u.get("doc_tag") for u in uploads if u.get("doc_tag")]
        filenames = [u.get("filename") for u in uploads if u.get("filename")]
        question = (
            "Summarize the latest uploaded documents. Highlight key facts and dates. "
            f"Document tags: {', '.join(doc_tags)}. Filenames: {', '.join(filenames)}."
        )
        answer = await self._ask_docwain(question, context)
        body_text = answer.get("response") or "I could not generate a summary."
        sources = answer.get("sources") or []
        return format_text_answer(
            response_text=body_text,
            sources=sources,
            domain="summary",
            grounded=bool(sources),
        )

    async def _extract_fields(self, context: TeamsChatContext, preset: Optional[str] = None) -> Dict[str, Any]:
        uploads = self.state_store.list_uploads(context.subscription_id, context.profile_id, limit=3)
        if not uploads:
            return _card_activity(
                build_card("error_card", message="No uploads found yet. Upload a document and try again.")
            )
        doc_tags = [u.get("doc_tag") for u in uploads if u.get("doc_tag")]
        filenames = [u.get("filename") for u in uploads if u.get("filename")]
        safe_preset = preset if preset in _VALID_PRESETS else None
        preset_text = ""
        if safe_preset == "invoice":
            preset_text = "Extract invoice number, vendor, total, currency, issue date, and due date."
        elif safe_preset == "contract":
            preset_text = "Extract parties, effective date, termination date, renewal terms, and obligations."
        elif safe_preset == "hr":
            preset_text = "Extract candidate name, experience, education, skills, and certifications."
        elif safe_preset == "medical":
            preset_text = "Extract patient name, diagnosis, medications, dosage, and treatment plan."
        elif safe_preset == "legal":
            preset_text = "Extract parties, jurisdiction, clauses, obligations, and effective dates."
        elif safe_preset == "policy":
            preset_text = "Extract policy number, coverage, premium, deductible, and effective period."
        else:
            preset_text = "Extract key fields (dates, totals, parties, identifiers) from the uploaded documents."

        question = (
            f"{preset_text} Use the documents tagged {', '.join(doc_tags)} "
            f"(files: {', '.join(filenames)}). Present fields as bullet points."
        )
        answer = await self._ask_docwain(question, context)
        body_text = answer.get("response") or "No fields were extracted."
        sources = answer.get("sources") or []
        return format_text_answer(
            response_text=body_text,
            sources=sources,
            domain=safe_preset or "extraction",
            grounded=bool(sources),
        )

    async def _generate_content(
        self,
        context: TeamsChatContext,
        content_type: Optional[str] = None,
        query: str = "",
    ) -> Dict[str, Any]:
        """Generate document-grounded content using the RAG pipeline."""
        uploads = self.state_store.list_uploads(context.subscription_id, context.profile_id, limit=5)
        if not uploads:
            return _card_activity(
                build_card("error_card", message="No uploads found. Upload a document first to generate content from it.")
            )

        filenames = [u.get("filename") for u in uploads if u.get("filename")]
        if not query:
            query = "Generate a professional summary based on the uploaded documents."
        generation_query = f"{query} Use documents: {', '.join(filenames)}."
        if content_type:
            generation_query = f"[{content_type}] {generation_query}"

        try:
            answer = await self._ask_docwain(generation_query, context)
            response_text = answer.get("response") or "Could not generate content. Try rephrasing your request."
        except Exception as exc:  # noqa: BLE001
            logger.error("Content generation via Teams failed: %s", exc, exc_info=True)
            return _card_activity(
                build_card("error_card", message="Content generation failed. Please try again.")
            )

        sources = answer.get("sources") or []
        return format_text_answer(
            response_text=response_text,
            sources=sources,
            domain=content_type or "content",
            grounded=bool(sources),
        )

    def _list_docs(self, context: TeamsChatContext) -> Dict[str, Any]:
        uploads = self.state_store.list_uploads(context.subscription_id, context.profile_id, limit=10)
        if not uploads:
            return _card_activity(build_card("error_card", message="No uploads have been ingested yet."))

        import time as _time
        now = _time.time()

        body = [
            {"type": "TextBlock", "text": f"Your Documents ({len(uploads)} file{'s' if len(uploads) != 1 else ''})", "weight": "Bolder", "size": "Medium"},
        ]

        for upload in uploads:
            name = upload.get("filename") or "file"
            doc_type = upload.get("document_type") or "general"
            chunks = upload.get("documents_created") or 0
            ts = upload.get("timestamp", 0)

            # Friendly time ago
            elapsed = now - ts if ts else 0
            if elapsed < 60:
                time_ago = "just now"
            elif elapsed < 3600:
                time_ago = f"{int(elapsed / 60)} min ago"
            elif elapsed < 86400:
                time_ago = f"{int(elapsed / 3600)} hr ago"
            else:
                time_ago = f"{int(elapsed / 86400)} day(s) ago"

            body.append({
                "type": "ColumnSet",
                "separator": True,
                "spacing": "Small",
                "columns": [
                    {
                        "type": "Column",
                        "width": "stretch",
                        "items": [
                            {"type": "TextBlock", "text": name, "weight": "Bolder", "size": "Small"},
                            {"type": "TextBlock", "text": f"{chunks} chunk(s) | {time_ago}", "size": "Small", "isSubtle": True, "spacing": "None"},
                        ],
                    },
                    {
                        "type": "Column",
                        "width": "auto",
                        "items": [
                            {"type": "TextBlock", "text": f"[{doc_type.title()}]", "size": "Small", "color": "Accent"},
                        ],
                        "verticalContentAlignment": "Center",
                    },
                ],
            })

        card = {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.5",
            "body": body,
            "actions": [
                {"type": "Action.Submit", "title": "Summarize all", "data": {"action": "summarize_recent"}},
                {"type": "Action.Submit", "title": "Delete all docs", "data": {"action": "delete_documents"}},
                {"type": "Action.Submit", "title": "Tools", "data": {"action": "show_tools"}},
            ],
        }
        return _card_activity(card, text=f"{len(uploads)} document(s) uploaded")

    def _set_model(self, context: TeamsChatContext, model: Optional[str]) -> Dict[str, Any]:
        if not model:
            model = Config.Teams.DEFAULT_MODEL
        prefs = self.state_store.set_preferences(context.subscription_id, context.profile_id, model_name=model)
        return _card_activity(
            build_card("answer_card", title="Model updated", text=f"Using model '{prefs.get('model_name')}' for this chat.", sources_text="")
        )

    def _set_persona(self, context: TeamsChatContext, persona: Optional[str]) -> Dict[str, Any]:
        if not persona:
            persona = Config.Teams.DEFAULT_PERSONA
        prefs = self.state_store.set_preferences(context.subscription_id, context.profile_id, persona=persona)
        return _card_activity(
            build_card(
                "answer_card",
                title="Persona updated",
                text=f"Using persona '{prefs.get('persona')}' for this chat.",
                sources_text="",
            )
        )

    def _show_preferences(self, context: TeamsChatContext) -> Dict[str, Any]:
        prefs = self.state_store.get_preferences(context.subscription_id, context.profile_id)
        current_model = prefs.get("model_name") or context.model_name or Config.Teams.DEFAULT_MODEL
        current_persona = prefs.get("persona") or context.persona or Config.Teams.DEFAULT_PERSONA
        return _card_activity(
            build_card("preferences_card", current_model=current_model, current_persona=current_persona)
        )

    def _set_preferences(self, context: TeamsChatContext, data: Dict[str, Any]) -> Dict[str, Any]:
        model = data.get("model")
        persona = data.get("persona")
        kwargs: Dict[str, str] = {}
        if model:
            kwargs["model_name"] = model
        if persona:
            kwargs["persona"] = persona
        if kwargs:
            self.state_store.set_preferences(context.subscription_id, context.profile_id, **kwargs)
        parts = []
        if model:
            parts.append(f"Model: **{model}**")
        if persona:
            parts.append(f"Persona: **{persona}**")
        summary = ", ".join(parts) if parts else "No changes made."
        return _card_activity(
            build_card("answer_card", title="Preferences saved", text=summary, sources_text="")
        )

    def _open_ui(self) -> Dict[str, Any]:
        url = DEFAULT_WEB_URL
        return _card_activity(
            build_card("answer_card", title="DocWain Web", text=f"Open DocWain in your browser: {url}", sources_text=""),
            text=url,
        )

    async def _delete_documents(self, context: TeamsChatContext) -> Dict[str, Any]:
        """Show a confirmation card before deleting all documents."""
        uploads = self.state_store.list_uploads(context.subscription_id, context.profile_id, limit=10)
        if not uploads:
            return _card_activity(
                build_card("error_card", message="No uploaded documents found to delete.")
            )
        filenames = [u.get("filename", "unknown") for u in uploads]
        return _card_activity(
            build_card(
                "delete_confirm_card",
                file_count=str(len(uploads)),
                file_list=", ".join(filenames[:5]) + ("..." if len(filenames) > 5 else ""),
            )
        )

    async def _confirm_delete(self, context: TeamsChatContext) -> Dict[str, Any]:
        """Actually delete all documents for this Teams user session."""
        try:
            deleted_count = await asyncio.to_thread(
                self.chat_service.delete_all_documents,
                context.subscription_id,
                context.profile_id,
            )
            cleared = self.state_store.clear_uploads(context.subscription_id, context.profile_id)
            logger.info(
                "Teams documents deleted | subscription=%s profile=%s embeddings=%d upload_records=%d",
                context.subscription_id, context.profile_id, deleted_count, cleared,
            )
            return _card_activity(
                build_card(
                    "answer_card",
                    title="Documents deleted",
                    text=f"Deleted {deleted_count} embedding(s) and {cleared} upload record(s). You can upload new documents to start fresh.",
                    sources_text="",
                ),
                text="Documents deleted successfully.",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Teams document deletion failed: %s", exc, exc_info=True)
            return _card_activity(
                build_card("error_card", message="Failed to delete documents. Please try again.")
            )

    async def _domain_query(self, context: TeamsChatContext, query: str) -> Dict[str, Any]:
        """Handle domain-specific follow-up queries from DI report card buttons."""
        if not query:
            return _card_activity(build_card("error_card", message="No query provided."))
        try:
            answer = await self._ask_docwain(query, context)
            response_text = answer.get("response") or "I could not find an answer."
            sources = answer.get("sources") or []
            domain = answer.get("domain") or "general"
            grounded = answer.get("grounded", False)

            return format_text_answer(
                response_text=response_text,
                sources=sources,
                domain=domain,
                grounded=grounded,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Domain query failed: %s", exc, exc_info=True)
            return _card_activity(
                build_card("error_card", message="Failed to answer the query. Please try again.")
            )

    async def _pipeline_consent_proceed(self, context: TeamsChatContext, document_id: str) -> Dict[str, Any]:
        """User clicked 'Proceed Anyway' — resume pipeline to embedding stage."""
        if not document_id:
            return _card_activity(build_card("error_card", message="Missing document reference. Please re-upload."))
        try:
            from src.teams.pipeline import TeamsDocumentPipeline
            from src.teams.teams_storage import TeamsDocumentStorage
            pipeline = TeamsDocumentPipeline(state_store=self.state_store)
            storage = TeamsDocumentStorage()
            storage.consent_proceed(document_id)

            doc = storage.get_document(document_id)
            if doc:
                cached_content = pipeline._get_cached_content(document_id)
                if cached_content:
                    filename = doc.get("filename", "document")
                    doc_type = doc.get("document_type", "general")
                    # Try to get doc_type from extraction subdoc if top-level is missing
                    if doc_type == "general" and isinstance(doc.get("extraction"), dict):
                        doc_type = doc["extraction"].get("doc_type", "general")

                    stage3 = await pipeline.stage_embed(
                        document_id=document_id,
                        extracted_content=cached_content,
                        filename=filename,
                        doc_type=doc_type,
                        context=context,
                        correlation_id="consent-resume",
                    )
                    return _card_activity(stage3["card"], text="Document ready")

            return _card_activity(
                build_card("answer_card", title="Consent granted", text="Embedding in progress. You'll be able to query the document shortly.", sources_text=""),
                text="Consent granted",
            )
        except Exception as exc:
            logger.error("Pipeline consent proceed failed: %s", exc, exc_info=True)
            return _card_activity(build_card("error_card", message="Failed to process consent. Please try again."))

    async def _pipeline_consent_reject(self, context: TeamsChatContext, document_id: str) -> Dict[str, Any]:
        """User clicked 'Cancel Embedding' on screening consent card."""
        if not document_id:
            return _card_activity(build_card("error_card", message="Missing document reference."))
        try:
            from src.teams.teams_storage import TeamsDocumentStorage
            storage = TeamsDocumentStorage()
            storage.consent_reject(document_id)
            return _card_activity(
                build_card("answer_card", title="Embedding cancelled", text="The document will not be embedded. It remains in the system but is not queryable. You can re-upload to try again.", sources_text=""),
                text="Embedding cancelled",
            )
        except Exception as exc:
            logger.error("Pipeline consent reject failed: %s", exc, exc_info=True)
            return _card_activity(build_card("error_card", message="Failed to process cancellation. Please try again."))

    def _tools_menu(self, context: TeamsChatContext) -> Dict[str, Any]:
        prefs = self.state_store.get_preferences(context.subscription_id, context.profile_id)
        model = prefs.get("model_name") or context.model_name or Config.Teams.DEFAULT_MODEL
        persona = prefs.get("persona") or context.persona or Config.Teams.DEFAULT_PERSONA

        # Build domain-aware tool actions based on uploaded documents
        uploads = self.state_store.list_uploads(context.subscription_id, context.profile_id, limit=10)
        doc_types = set()
        for u in uploads:
            dt = u.get("document_type", "general")
            if dt and dt != "general":
                doc_types.add(dt)

        actions = [
            {"type": "Action.Submit", "title": "Summarize recent", "data": {"action": "summarize_recent"}},
        ]

        # Add domain-specific extraction buttons
        if doc_types:
            try:
                from src.teams.insights import get_domain_actions
                for dt in sorted(doc_types):
                    domain_acts = get_domain_actions(dt)
                    if domain_acts:
                        first = domain_acts[0]
                        actions.append({
                            "type": "Action.Submit",
                            "title": f"{dt.title()}: {first['title']}",
                            "data": {"action": "domain_query", "query": first["query"]},
                        })
            except Exception:  # noqa: BLE001
                pass

        # Add extract fields for any preset with matching uploads
        _preset_map = {"invoice": "invoice", "hr": "hr", "resume": "hr", "contract": "contract",
                       "legal": "legal", "medical": "medical", "policy": "policy"}
        added_presets = set()
        for dt in doc_types:
            preset = _preset_map.get(dt)
            if preset and preset not in added_presets:
                added_presets.add(preset)
                actions.append({
                    "type": "Action.Submit",
                    "title": f"Extract {preset} fields",
                    "data": {"action": "extract_fields", "preset": preset},
                })

        # Fallback: show generic extract if no specific types detected
        if not added_presets:
            actions.append({"type": "Action.Submit", "title": "Extract fields", "data": {"action": "extract_fields"}})

        actions.extend([
            {"type": "Action.Submit", "title": "Generate content", "data": {"action": "generate_content"}},
            {"type": "Action.Submit", "title": "List docs", "data": {"action": "list_docs"}},
            {"type": "Action.Submit", "title": "Delete all docs", "data": {"action": "delete_documents"}},
            {"type": "Action.Submit", "title": "Preferences", "data": {"action": "show_preferences"}},
        ])

        # Build card programmatically with dynamic actions
        card = {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.5",
            "body": [
                {"type": "TextBlock", "text": "DocWain tools", "weight": "Bolder", "size": "Medium"},
                {"type": "TextBlock", "text": "Run actions against your uploaded documents.", "wrap": True},
            ],
            "actions": actions[:6],  # Teams limits to 6 actions
        }
        return _card_activity(card)
