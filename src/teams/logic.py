import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from src.api.config import Config
from src.api.vector_store import QdrantVectorStore, build_collection_name

try:
    from src.api import dw_newron
except Exception:  # noqa: BLE001
    dw_newron = None

logger = logging.getLogger(__name__)


class TeamsChatError(Exception):
    """Raised when Teams chat handling cannot complete."""


@dataclass
class TeamsChatContext:
    """Normalized identifiers for a Teams chat exchange."""

    user_id: str
    session_id: str
    subscription_id: str
    profile_id: str
    model_name: str
    persona: str


@dataclass
class TeamsAnswerResult:
    """Answer payload paired with the scope that produced it."""

    answer: Dict[str, Any]
    subscription_id: str
    profile_id: str
    fallback_used: bool = False
    internet_mode: bool = False


class TeamsChatService:
    """
    Teams-specific chat orchestrator. It keeps Teams traffic isolated from the
    base chat logic by resolving the correct subscription/profile to target and
    providing an internet-enabled fallback when no document context is present.
    """

    def __init__(self, qdrant_client: Optional[QdrantClient] = None, vector_store: Optional[QdrantVectorStore] = None):
        self.client = qdrant_client or QdrantClient(
            url=Config.Qdrant.URL,
            api_key=Config.Qdrant.API,
            timeout=120,
        )
        self.vector_store = vector_store or QdrantVectorStore(self.client)
        self.vector_size = int(getattr(Config.Model, "EMBEDDING_DIM", 1024))

    @staticmethod
    def build_context(
        user_id: str,
        session_id: str,
        model_name: Optional[str] = None,
        persona: Optional[str] = None,
    ) -> TeamsChatContext:
        """Compose a Teams chat context with per-session collection and per-user profile."""
        effective_model = model_name or Config.Teams.DEFAULT_MODEL
        effective_persona = persona or Config.Teams.DEFAULT_PERSONA

        if Config.Teams.SESSION_AS_SUBSCRIPTION:
            subscription_id = session_id or "teams-session"
        else:
            subscription_id = Config.Teams.DEFAULT_SUBSCRIPTION

        if Config.Teams.PROFILE_PER_USER:
            profile_id = user_id or "teams_user"
        else:
            profile_id = Config.Teams.DEFAULT_PROFILE

        return TeamsChatContext(
            user_id=user_id or "teams_user",
            session_id=session_id or "teams-session",
            subscription_id=subscription_id,
            profile_id=profile_id,
            model_name=effective_model,
            persona=effective_persona,
        )

    def ensure_collection(self, subscription_id: str) -> None:
        """Create or validate the Teams collection for the given subscription/session."""
        collection_name = build_collection_name(subscription_id)
        try:
            self.vector_store.ensure_collection(collection_name, self.vector_size)
        except UnexpectedResponse as exc:
            logger.error(
                "Failed to ensure Teams collection | subscription=%s collection=%s error=%s",
                subscription_id,
                collection_name,
                exc,
            )
            raise TeamsChatError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Unexpected error ensuring Teams collection | subscription=%s collection=%s error=%s",
                subscription_id,
                collection_name,
                exc,
            )
            raise TeamsChatError(str(exc)) from exc

    def _answer_with_internet_mode(self, question: str, context: TeamsChatContext) -> Dict[str, Any]:
        """
        Internet-enabled fallback: answer using the LLM without relying on document context.
        """
        prompt = (
            f"You are an internet-enabled assistant helping a Teams user. "
            f"Answer with your general knowledge. If you do not know, say so plainly.\n\n"
            f"Persona: {context.persona}\n"
            f"Question: {question}"
        )
        try:
            from src.api.dw_newron import create_llm_client

            llm_client = create_llm_client(context.model_name)
            response_text = llm_client.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.error("Internet fallback failed for Teams chat: %s", exc)
            response_text = (
                "I couldn't find relevant document context, and the internet fallback was unavailable."
            )
        return {
            "response": response_text,
            "sources": [],
            "context_found": False,
            "grounded": False,
            "mode": "internet",
        }

    def answer_question(self, question: str, context: TeamsChatContext) -> TeamsAnswerResult:
        """
        Answer a Teams question with Teams-aware scope. Ensures the per-session
        collection exists, and if no document context is found, answers using an
        internet-enabled fallback.
        """
        self.ensure_collection(context.subscription_id)

        try:
            if dw_newron is None:
                from src.api import dw_newron as _dw_newron
            else:
                _dw_newron = dw_newron
            answer = _dw_newron.answer_question(
                query=question,
                user_id=context.user_id,
                profile_id=context.profile_id,
                subscription_id=context.subscription_id,
                model_name=context.model_name,
                persona=context.persona,
                session_id=context.session_id,
            )
        except Exception as exc:  # noqa: BLE001
            raise TeamsChatError(str(exc)) from exc

        context_found = bool(answer.get("context_found"))
        if context_found:
            return TeamsAnswerResult(
                answer=answer,
                subscription_id=context.subscription_id,
                profile_id=context.profile_id,
                fallback_used=False,
                internet_mode=False,
            )

        internet_answer = self._answer_with_internet_mode(question, context)
        return TeamsAnswerResult(
            answer=internet_answer,
            subscription_id=context.subscription_id,
            profile_id=context.profile_id,
            fallback_used=True,
            internet_mode=True,
        )
