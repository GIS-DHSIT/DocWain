import asyncio
import logging
from typing import Any, Dict, Optional

from src.api.config import Config
from src.teams.cards import build_card
from src.teams.logic import TeamsChatContext, TeamsChatService
from src.teams.state import TeamsStateStore

logger = logging.getLogger(__name__)

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


def _format_sources(sources: Any) -> str:
    items = sources or []
    if not isinstance(items, list):
        return ""
    snippets = []
    for src in items[:5]:
        name = src.get("source_name") or src.get("source_id") or src.get("document_id") or "Source"
        excerpt = (src.get("excerpt") or "")[:200]
        if excerpt:
            snippets.append(f"- {name}: {excerpt}")
        else:
            snippets.append(f"- {name}")
    return "\n".join(snippets)


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
        sources_text = _format_sources(answer.get("sources"))
        body_text = answer.get("response") or "I could not generate a summary."
        return _card_activity(
            build_card("answer_card", title="Summary", text=body_text, sources_text=sources_text or "No sources available."),
            text="Summary ready.",
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
        sources_text = _format_sources(answer.get("sources"))
        return _card_activity(
            build_card("answer_card", title="Extracted fields", text=body_text, sources_text=sources_text or "No sources available."),
            text="Field extraction ready.",
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

        return _card_activity(
            build_card("answer_card", title="Generated content", text=response_text, sources_text=""),
            text="Content generation complete.",
        )

    def _list_docs(self, context: TeamsChatContext) -> Dict[str, Any]:
        uploads = self.state_store.list_uploads(context.subscription_id, context.profile_id, limit=10)
        if not uploads:
            return _card_activity(build_card("error_card", message="No uploads have been ingested yet."))
        lines = []
        for upload in uploads:
            name = upload.get("filename") or "file"
            tag = upload.get("doc_tag") or "n/a"
            docs = upload.get("documents_created") or 0
            lines.append(f"- {name} (tag: {tag}, docs: {docs})")
        return _card_activity(
            build_card("answer_card", title="Recent uploads", text="\n".join(lines), sources_text=""),
            text="Recent uploads",
        )

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

    def _tools_menu(self, context: TeamsChatContext) -> Dict[str, Any]:
        prefs = self.state_store.get_preferences(context.subscription_id, context.profile_id)
        model = prefs.get("model_name") or context.model_name or Config.Teams.DEFAULT_MODEL
        persona = prefs.get("persona") or context.persona or Config.Teams.DEFAULT_PERSONA
        return _card_activity(
            build_card(
                "tools_menu_card",
                default_model=model,
                default_persona=persona,
                open_url=DEFAULT_WEB_URL,
            )
        )
