
import re
import json
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import faiss
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
from api.config import Config
import nltk
from nltk.corpus import wordnet
from spellchecker import SpellChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")

# Initialize models and clients (lazy loading to avoid startup errors)
_MODEL = None
_CROSS_ENCODER = None
_SPELL_CHECKER = None
_QDRANT_CLIENT = None


def get_model():
    """Lazy load sentence transformer model."""
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = SentenceTransformer(Config.Model.SENTENCE_TRANSFORMERS)
            logger.info(f"Loaded model: {Config.Model.SENTENCE_TRANSFORMERS}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            raise
    return _MODEL


def get_cross_encoder():
    """Lazy load cross encoder model."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            _CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Loaded cross-encoder model")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            raise
    return _CROSS_ENCODER


def get_spell_checker():
    """Lazy load spell checker."""
    global _SPELL_CHECKER
    if _SPELL_CHECKER is None:
        try:
            _SPELL_CHECKER = SpellChecker()
            logger.info("Initialized spell checker")
        except Exception as e:
            logger.error(f"Failed to initialize spell checker: {e}")
            raise
    return _SPELL_CHECKER


def get_qdrant_client():
    """Lazy load Qdrant client."""
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        try:
            _QDRANT_CLIENT = QdrantClient(
                url=Config.Qdrant.URL,
                api_key=Config.Qdrant.API,
                timeout=120
            )
            logger.info("Initialized Qdrant client")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    return _QDRANT_CLIENT


def configure_gemini():
    """Configure Gemini API with proper error handling."""
    try:
        api_key = Config.Model.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in configuration")
        genai.configure(api_key=api_key)
        logger.info("Gemini API configured successfully")
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
        raise


@dataclass
class RetrievedChunk:
    """Data class for retrieved document chunks with metadata."""
    id: str
    text: str
    score: float
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConversationTurn:
    """Data class for conversation history."""
    user_message: str
    assistant_response: str
    timestamp: float


class ConversationHistory:
    """Manages conversation history with a sliding window."""

    def __init__(self, max_turns: int = 3):
        self.max_turns = max_turns
        self.histories: Dict[str, deque] = {}

    def add_turn(self, user_id: str, user_message: str, assistant_response: str):
        """Add a conversation turn to history."""
        if user_id not in self.histories:
            self.histories[user_id] = deque(maxlen=self.max_turns)

        turn = ConversationTurn(
            user_message=user_message,
            assistant_response=assistant_response,
            timestamp=time.time()
        )
        self.histories[user_id].append(turn)

    def get_context(self, user_id: str, max_turns: int = 2) -> str:
        """Get recent conversation context as formatted string."""
        if user_id not in self.histories or not self.histories[user_id]:
            return ""

        context_parts = []
        recent_turns = list(self.histories[user_id])[-max_turns:]

        for turn in recent_turns:
            context_parts.append(f"User: {turn.user_message}")
            context_parts.append(f"Assistant: {turn.assistant_response}")

        return "\n".join(context_parts)

    def clear_history(self, user_id: str):
        """Clear conversation history for a user."""
        if user_id in self.histories:
            self.histories[user_id].clear()


class SpellCorrector:
    """Handles spelling correction with domain-aware corrections."""

    def __init__(self):
        self.spell_checker = get_spell_checker()
        self.domain_terms = set()

    def add_domain_terms(self, terms: List[str]):
        """Add domain-specific terms to whitelist."""
        self.domain_terms.update(term.lower() for term in terms)
        self.spell_checker.word_frequency.load_words(terms)

    def correct_text(self, text: str) -> Tuple[str, List[str]]:
        """
        Correct spelling in text while preserving domain terms.

        Returns:
            Tuple of (corrected_text, list_of_corrections)
        """
        words = text.split()
        corrected_words = []
        corrections = []

        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if not clean_word or clean_word in self.domain_terms or len(clean_word) <= 2:
                corrected_words.append(word)
                continue

            if clean_word not in self.spell_checker:
                corrected = self.spell_checker.correction(clean_word)
                if corrected and corrected != clean_word:
                    corrections.append(f"{clean_word} � {corrected}")
                    corrected_words.append(word.replace(clean_word, corrected))
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        return ' '.join(corrected_words), corrections


class QueryExpander:
    """Expands queries with synonyms and related terms."""

    @staticmethod
    def get_synonyms(word: str, max_synonyms: int = 2) -> List[str]:
        """Get synonyms using WordNet."""
        synonyms = set()
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower():
                        synonyms.add(synonym.lower())
                    if len(synonyms) >= max_synonyms:
                        break
                if len(synonyms) >= max_synonyms:
                    break
        except Exception as e:
            logger.debug(f"Error getting synonyms for {word}: {e}")

        return list(synonyms)

    @staticmethod
    def expand_query(query: str, max_synonyms_per_word: int = 2) -> str:
        """Expand query with synonyms for key terms."""
        try:
            words = nltk.word_tokenize(query.lower())
            tagged = nltk.pos_tag(words)

            try:
                stopwords = set(nltk.corpus.stopwords.words('english'))
            except:
                stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}

            expanded_terms = [query]

            for word, pos in tagged:
                if word in stopwords or len(word) <= 3:
                    continue

                if pos.startswith('NN') or pos.startswith('VB'):
                    synonyms = QueryExpander.get_synonyms(word, max_synonyms_per_word)
                    expanded_terms.extend(synonyms)

            return ' '.join(expanded_terms)

        except Exception as e:
            logger.debug(f"Error expanding query: {e}")
            return query


class TextPreprocessor:
    """Handles text preprocessing for consistent tokenization."""

    def __init__(self):
        try:
            self.stopwords = set(nltk.corpus.stopwords.words('english'))
        except:
            self.stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text by lowercasing and removing extra whitespace."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text with lemmatization and stopword removal for BM25."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()

        tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]

        lemmatized = []
        for token in tokens:
            if token.endswith('ing'):
                lemmatized.append(token[:-3])
            elif token.endswith('ed'):
                lemmatized.append(token[:-2])
            elif token.endswith('s') and len(token) > 3:
                lemmatized.append(token[:-1])
            else:
                lemmatized.append(token)

        return lemmatized


class GreetingHandler:
    """Handles greeting and farewell detection."""

    GREETINGS = {
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'greetings', 'howdy', 'what\'s up', 'how are you', 'how do you do',
        'nice to meet you', 'good day', 'hiya', 'sup', 'yo', 'salutations',
        'welcome', 'hola', 'bonjour', 'namaste', 'aloha','hii','Hii'   }

    FAREWELLS = {
        'bye', 'goodbye', 'good bye', 'see you', 'see ya', 'farewell',
        'adieu', 'cheerio', 'tata', 'ta ta', 'catch you later', 'later',
        'take care', 'until next time', 'signing off', 'gotta go', 'gtg',
        'peace out', 'ciao', 'au revoir', 'sayonara', 'hasta la vista',
        'talk to you later', 'ttyl', 'see you soon', 'see you around',
        'good night', 'goodnight', 'have a good day', 'have a great day',
        'thanks for your help', 'that\'s all', 'i\'m done', 'end chat',
        'quit', 'exit', 'close', 'finish', 'terminate', 'stop'
    }

    POSITIVE_FEEDBACK = {
        'thanks', 'thank you', 'thanks a lot', 'thank you so much',
        'appreciate it', 'much appreciated'
    }

    @classmethod
    def is_greeting(cls, message: str) -> bool:
        """Check if message is a greeting."""
        message = message.lower().strip()

        if len(message) <= 60:
            pattern = r'\b(' + '|'.join(re.escape(g) for g in cls.GREETINGS) + r')\b'
            if re.search(pattern, message):
                words = message.split()
                if len(words) <= 5 or any(g in ' '.join(words[:5]) for g in cls.GREETINGS):
                    return True

        return False

    @classmethod
    def is_farewell(cls, message: str) -> bool:
        """Check if message is a farewell."""
        message = message.lower().strip()

        if len(message) <= 50:
            pattern = r'\b(' + '|'.join(re.escape(f) for f in cls.FAREWELLS) + r')\b'
            return bool(re.search(pattern, message))

        return False

    @classmethod
    def is_positive_feedback(cls, message: str) -> bool:
        """Check if message is positive feedback."""
        message = message.lower().strip()

        if len(message) <= 40:
            pattern = r'\b(' + '|'.join(re.escape(f) for f in cls.POSITIVE_FEEDBACK) + r')\b'
            return bool(re.search(pattern, message))

        return False


class GeminiClient:
    """Handles Gemini API calls with structured output and retries."""

    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        try:
            configure_gemini()
            self.model = genai.GenerativeModel(model_name)
            self.generation_config = genai.GenerationConfig(
                temperature=0.0,
                top_p=0.95,
                top_k=40,
                max_output_tokens=2048,
            )
            logger.info(f"Initialized GeminiClient with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize GeminiClient: {e}")
            raise

    def generate(
            self,
            prompt: str,
            max_retries: int = 3,
            backoff: float = 1.0
    ) -> str:
        """Generate response with retry logic and robust parsing."""
        for attempt in range(1, max_retries + 1):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )

                text = None

                if hasattr(response, 'text') and response.text:
                    text = response.text.strip()
                elif hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content'):
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            text = candidate.content.parts[0].text.strip()
                        elif hasattr(candidate.content, 'text'):
                            text = candidate.content.text.strip()

                if text:
                    return text
                else:
                    logger.warning(f"No text in response: {response}")
                    return "I apologize, but I couldn't generate a proper response."

            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
                else:
                    logger.error(f"All retry attempts failed: {e}")
                    raise

        return "I apologize, but I encountered an error generating a response."


class QueryReformulator:
    """Reformulates conversational queries into clear, concise search queries."""

    def __init__(self, llm_client: GeminiClient):
        self.llm_client = llm_client

    def reformulate(self, query: str, conversation_context: str = "") -> str:
        """Reformulate query using LLM to make it more search-friendly."""
        if len(query.split()) <= 5 and not conversation_context:
            return query

        prompt = f"""You are a query reformulation assistant. Convert the user's conversational question into a clear, concise search query optimized for semantic search.

RULES:
1. Extract the core information need
2. Remove filler words and conversational elements
3. Keep domain-specific terms and technical vocabulary
4. Make it 3-10 words maximum
5. If conversation context is provided, resolve pronouns and references
6. Output ONLY the reformulated query, nothing else

{f"CONVERSATION CONTEXT:\\n{conversation_context}\\n" if conversation_context else ""}
USER QUERY: {query}

REFORMULATED QUERY:"""

        try:
            reformulated = self.llm_client.generate(
                prompt,
                max_retries=2,
                backoff=0.5
            )

            reformulated = reformulated.strip().strip('"\'')

            if 2 <= len(reformulated.split()) <= 15 and reformulated.lower() != query.lower():
                logger.info(f"Reformulated: '{query}' � '{reformulated}'")
                return reformulated
            else:
                return query

        except Exception as e:
            logger.warning(f"Query reformulation failed: {e}")
            return query


class QdrantRetriever:
    """Handles retrieval from Qdrant using native search functionality."""

    def __init__(self, client: QdrantClient, model: SentenceTransformer):
        self.client = client
        self.model = model
        self.preprocessor = TextPreprocessor()

    def retrieve(
            self,
            collection_name: str,
            query: str,
            top_k: int = 50,
            score_threshold: float = 0.2
    ) -> List[RetrievedChunk]:
        """Retrieve relevant chunks using Qdrant's native search."""
        try:
            query_vector = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32)

            logger.info(f"Searching collection '{collection_name}' for query: {query[:100]}")

            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold
            )

            if not search_results:
                logger.warning(f"No results found in collection '{collection_name}'")
                return []

            chunks = []
            for result in search_results:
                chunk = RetrievedChunk(
                    id=str(result.id),
                    text=result.payload.get('text', ''),
                    score=float(result.score),
                    source=result.payload.get('source_file', 'unknown'),
                    metadata=result.payload
                )
                chunks.append(chunk)

            logger.info(f"Retrieved {len(chunks)} chunks from Qdrant")
            return chunks

        except Exception as e:
            logger.error(f"Error retrieving from Qdrant: {e}", exc_info=True)
            return []


class HybridReranker:
    """Reranks retrieved chunks using BM25 + vector scores with dynamic weighting."""

    def __init__(self, alpha: float = 0.7, cross_encoder: Optional[CrossEncoder] = None):
        """Initialize reranker."""
        self.alpha = alpha
        self.preprocessor = TextPreprocessor()
        self.cross_encoder = cross_encoder

    def adjust_alpha(self, query: str) -> float:
        """Dynamically adjust alpha based on query characteristics."""
        words = query.lower().split()

        question_words = {'what', 'why', 'how', 'explain', 'describe', 'understand'}
        if any(qw in words for qw in question_words):
            return 0.75

        if any(re.search(r'\d+', word) for word in words) or len(words) <= 3:
            return 0.6

        return self.alpha

    def rerank(
            self,
            chunks: List[RetrievedChunk],
            query: str,
            top_k: int = 10,
            use_cross_encoder: bool = True
    ) -> List[RetrievedChunk]:
        """Rerank chunks using hybrid BM25 + vector scoring."""
        if not chunks:
            return []

        try:
            alpha = self.adjust_alpha(query)
            logger.info(f"Using alpha={alpha:.2f} for reranking")

            texts = [chunk.text for chunk in chunks]
            tokenized_corpus = [self.preprocessor.tokenize(text) for text in texts]
            tokenized_query = self.preprocessor.tokenize(query)

            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = np.array(bm25.get_scores(tokenized_query), dtype=np.float64)

            vector_scores = np.array([float(chunk.score) for chunk in chunks], dtype=np.float64)

            if len(vector_scores) > 1:
                v_min, v_max = vector_scores.min(), vector_scores.max()
                b_min, b_max = bm25_scores.min(), bm25_scores.max()

                v_range = v_max - v_min if v_max > v_min else 1.0
                b_range = b_max - b_min if b_max > b_min else 1.0

                vector_scores_norm = (vector_scores - v_min) / v_range
                bm25_scores_norm = (bm25_scores - b_min) / b_range
            else:
                vector_scores_norm = np.ones_like(vector_scores)
                bm25_scores_norm = np.ones_like(bm25_scores)

            hybrid_scores = (
                    alpha * vector_scores_norm +
                    (1 - alpha) * bm25_scores_norm
            )

            top_n_for_ce = min(top_k * 2, len(chunks))
            sorted_indices = np.argsort(hybrid_scores)[::-1][:top_n_for_ce]

            candidate_chunks = [chunks[idx] for idx in sorted_indices]

            if use_cross_encoder and self.cross_encoder and len(candidate_chunks) > 1:
                try:
                    pairs = [[query, chunk.text] for chunk in candidate_chunks]
                    ce_scores = self.cross_encoder.predict(pairs)

                    for i, score in enumerate(ce_scores):
                        candidate_chunks[i].score = float(score)

                    candidate_chunks.sort(key=lambda c: c.score, reverse=True)
                    logger.info(f"Applied CrossEncoder reranking to {len(candidate_chunks)} chunks")

                except Exception as e:
                    logger.warning(f"CrossEncoder reranking failed: {e}")
                    for i, chunk in enumerate(candidate_chunks):
                        chunk.score = float(hybrid_scores[sorted_indices[i]])
            else:
                for i, chunk in enumerate(candidate_chunks):
                    chunk.score = float(hybrid_scores[sorted_indices[i]])

            reranked_chunks = candidate_chunks[:top_k]

            logger.info(f"Reranked to {len(reranked_chunks)} chunks (alpha={alpha:.2f})")
            return reranked_chunks

        except Exception as e:
            logger.error(f"Error in reranking: {e}", exc_info=True)
            return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]


class ContextBuilder:
    """Builds formatted context for LLM with source citations."""

    @staticmethod
    def build_context(chunks: List[RetrievedChunk], max_chunks: int = 3) -> str:
        """Build context string with source citations."""
        if not chunks:
            return ""

        seen_texts = set()
        unique_chunks = []
        for chunk in chunks:
            normalized = ' '.join(chunk.text.split())
            if normalized not in seen_texts and normalized.strip():
                seen_texts.add(normalized)
                unique_chunks.append(chunk)

        selected_chunks = unique_chunks[:max_chunks]

        context_parts = []
        for i, chunk in enumerate(selected_chunks, 1):
            # Use source_file directly from metadata as fallback
            source_name = chunk.source or chunk.metadata.get('source_file', f"doc_{chunk.id[:8]}")
            context_parts.append(
                f"[SOURCE: {source_name}]\n{chunk.text}\n[/SOURCE]"
            )

        return "\n".join(context_parts)

    @staticmethod
    def extract_sources(chunks: List[RetrievedChunk], max_sources: int = 3) -> List[Dict[str, Any]]:
        """Extract source information for response metadata."""
        sources = []
        for i, chunk in enumerate(chunks[:max_sources], 1):
            sources.append({
                'source_id': i,
                'source_name': chunk.source or f"Document {chunk.id[:8]}",
                'excerpt': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                'relevance_score': round(float(chunk.score), 3)
            })
        return sources


class AnswerabilityDetector:
    """Detects if a question can be answered from provided context."""

    def __init__(self, llm_client: GeminiClient):
        self.llm_client = llm_client

    def check_answerability(self, query: str, context: str) -> Tuple[bool, str]:
        """Check if the query can be answered from the context."""
        prompt = f"""You are an answerability classifier. Determine if the USER QUESTION can be answered using ONLY the information in the DOCUMENT CONTEXT.

DOCUMENT CONTEXT:
{context}

USER QUESTION: {query}

Respond with ONLY one of these formats:
- If answerable: "ANSWERABLE: <brief reason>"
- If not answerable: "NOT_ANSWERABLE: <what information is missing>"

Your response:"""

        try:
            response = self.llm_client.generate(prompt, max_retries=2)
            response = response.strip()

            if response.startswith("ANSWERABLE"):
                return True, response.replace("ANSWERABLE:", "").strip()
            elif response.startswith("NOT_ANSWERABLE"):
                return False, response.replace("NOT_ANSWERABLE:", "").strip()
            else:
                return True, "Classification unclear"

        except Exception as e:
            logger.warning(f"Answerability check failed: {e}")
            return True, "Check failed"


class PromptBuilder:
    """Builds structured prompts with strict grounding and citation requirements."""

    @staticmethod
    def build_qa_prompt(query: str, context: str, persona: str) -> str:
        """Build a structured QA prompt with grounding and citation requirements."""
        prompt = f"""You are a {persona} specialized in providing accurate, document-based answers with STRICT GROUNDING.

CRITICAL RULES FOR GROUNDING:
1. Answer ONLY using information explicitly stated in the DOCUMENT CONTEXT below
2. You MUST cite sources using [SOURCE-X] notation for EVERY factual claim
3. If information is not in the documents, you MUST explicitly state: "The provided documents do not contain information about [specific topic]"
4. NEVER add facts, opinions, estimates, or information not present in the documents
5. When quoting, use exact phrases from the documents
6. If the documents provide partial information, state what is available and what is missing
7. If documents conflict, cite both sources and note the discrepancy

DOCUMENT CONTEXT:
{context}

USER QUESTION: {query}

RESPONSE REQUIREMENTS:
- Provide a clear, direct answer grounded ONLY in the document context
- Cite sources using [SOURCE-X] notation after each claim
- If information is incomplete, explicitly state what is missing
- Use professional language appropriate for the domain
- Be concise but complete
- NEVER speculate or add external knowledge

Provide your grounded answer now:"""

        return prompt


class EnterpriseRAGSystem:
    """
    Enhanced RAG system with query reformulation, spelling correction,
    conversational context, cross-encoder reranking, and grounding.
    """

    def __init__(self):
        """Initialize the RAG system with lazy-loaded components."""
        try:
            # Initialize Gemini client first
            self.llm_client = GeminiClient(Config.Model.GEMINI_MODEL_NAME)

            # Initialize other components
            qdrant_client = get_qdrant_client()
            model = get_model()
            cross_encoder = get_cross_encoder()

            self.retriever = QdrantRetriever(qdrant_client, model)
            self.reranker = HybridReranker(alpha=0.7, cross_encoder=cross_encoder)
            self.context_builder = ContextBuilder()
            self.prompt_builder = PromptBuilder()
            self.greeting_handler = GreetingHandler()
            self.spell_corrector = SpellCorrector()
            self.query_expander = QueryExpander()
            self.query_reformulator = QueryReformulator(self.llm_client)
            self.answerability_detector = AnswerabilityDetector(self.llm_client)
            self.conversation_history = ConversationHistory(max_turns=3)

            logger.info("EnterpriseRAGSystem initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EnterpriseRAGSystem: {e}")
            raise

    def preprocess_query(
            self,
            query: str,
            user_id: str,
            use_reformulation: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """Preprocess query with spelling correction, reformulation, and expansion."""
        metadata = {
            'original_query': query,
            'corrections': [],
            'reformulated': False,
            'expanded': False
        }

        corrected_query, corrections = self.spell_corrector.correct_text(query)
        if corrections:
            metadata['corrections'] = corrections
            logger.info(f"Spelling corrections: {corrections}")

        processed_query = corrected_query
        if use_reformulation and len(query.split()) > 5:
            conv_context = self.conversation_history.get_context(user_id, max_turns=2)
            reformulated = self.query_reformulator.reformulate(corrected_query, conv_context)
            if reformulated != corrected_query:
                processed_query = reformulated
                metadata['reformulated'] = True

        expanded_query = self.query_expander.expand_query(processed_query)
        if expanded_query != processed_query:
            metadata['expanded'] = True
            metadata['expanded_query'] = expanded_query

        return processed_query, metadata

    def answer_question(
            self,
            query: str,
            profile_id: str,
            user_id: str,
            persona: str = "professional document analysis assistant",
            top_k_retrieval: int = 50,
            top_k_rerank: int = 10,
            final_k: int = 3
    ) -> Dict[str, Any]:
        """Main method to answer questions using enhanced RAG pipeline."""
        start_time = time.time()

        try:
            if self.greeting_handler.is_positive_feedback(query):
                return {
                    "response": "You're welcome! I'm glad I could help. Feel free to ask any other questions about your documents.",
                    "sources": [],
                    "user_id": user_id,
                    "collection": profile_id,
                    "context_found": True,
                    "query_type": "positive_feedback",
                    "grounded": True
                }

            if self.greeting_handler.is_greeting(query):
                greeting_response = f"Hello! I'm your doxa AI assistant. I can help you find specific information from your documents. What would you like to know?"
                self.conversation_history.add_turn(user_id, query, greeting_response)

                return {
                    "response": greeting_response,
                    "sources": [],
                    "user_id": user_id,
                    "collection": profile_id,
                    "context_found": True,
                    "query_type": "greeting",
                    "grounded": True
                }

            if self.greeting_handler.is_farewell(query):
                farewell_response = f"Goodbye! Feel free to return whenever you need information from your documents. Have a great day!"
                self.conversation_history.clear_history(user_id)

                return {
                    "response": farewell_response,
                    "sources": [],
                    "user_id": user_id,
                    "collection": profile_id,
                    "context_found": True,
                    "query_type": "farewell",
                    "grounded": True
                }

            logger.info(f"Processing query for collection '{profile_id}': {query[:100]}")

            processed_query, preprocessing_metadata = self.preprocess_query(query, user_id)
            logger.info(f"Preprocessed query: {processed_query}")

            retrieved_chunks = self.retriever.retrieve(
                collection_name=profile_id,
                query=processed_query,
                top_k=top_k_retrieval,
                score_threshold=0.15
            )

            if not retrieved_chunks:
                no_results_response = f"I searched through all your department documents but couldn't find relevant information for your question: '{query}'. Please try rephrasing or asking about a different topic covered in your documents."
                self.conversation_history.add_turn(user_id, query, no_results_response)

                return {
                    "response": no_results_response,
                    "sources": [],
                    "user_id": user_id,
                    "collection": profile_id,
                    "context_found": False,
                    "query_type": "no_results",
                    "preprocessing": preprocessing_metadata,
                    "grounded": True,
                    "processing_time": time.time() - start_time
                }

            reranked_chunks = self.reranker.rerank(
                chunks=retrieved_chunks,
                query=processed_query,
                top_k=top_k_rerank,
                use_cross_encoder=True
            )

            final_chunks = reranked_chunks[:final_k]

            context = self.context_builder.build_context(
                chunks=final_chunks,
                max_chunks=final_k
            )

            logger.info(f"Built context with {len(final_chunks)} chunks, {len(context)} chars")

            is_answerable, answerability_reason = self.answerability_detector.check_answerability(
                query, context
            )

            if not is_answerable:
                not_answerable_response = f"Based on the available  documents, I cannot fully answer your question. {answerability_reason}"
                self.conversation_history.add_turn(user_id, query, not_answerable_response)

                return {
                    "response": not_answerable_response,
                    "sources": self.context_builder.extract_sources(final_chunks),
                    "user_id": user_id,
                    "collection": profile_id,
                    "context_found": True,
                    "query_type": "not_answerable",
                    "answerability_reason": answerability_reason,
                    "preprocessing": preprocessing_metadata,
                    "grounded": True,
                    "processing_time": time.time() - start_time
                }

            prompt = self.prompt_builder.build_qa_prompt(
                query=query,
                context=context,
                persona=persona
            )

            answer = self.llm_client.generate(prompt)

            sources = self.context_builder.extract_sources(final_chunks)

            self.conversation_history.add_turn(user_id, query, answer)

            has_citations = bool(re.search(r'\[SOURCE-\d+\]', answer))

            processing_time = time.time() - start_time

            return {
                "response": answer,
                "sources": sources,
                "user_id": user_id,
                "collection": profile_id,
                "context_found": True,
                "query_type": "document_qa",
                "num_sources": len(sources),
                "preprocessing": preprocessing_metadata,
                "answerability": {
                    "is_answerable": is_answerable,
                    "reason": answerability_reason
                },
                "grounded": True,
                "has_citations": has_citations,
                "processing_time": round(processing_time, 2),
                "retrieval_stats": {
                    "initial_retrieved": len(retrieved_chunks),
                    "after_rerank": len(reranked_chunks),
                    "final_context": len(final_chunks)
                }
            }

        except Exception as e:
            logger.error(f"Error in answer_question: {e}", exc_info=True)

            error_response = "I apologize, but I encountered an error while processing your question. Please try again or contact support if the issue persists."

            return {
                "response": error_response,
                "sources": [],
                "user_id": user_id,
                "collection": profile_id,
                "context_found": False,
                "query_type": "error",
                "error": str(e),
                "grounded": False,
                "processing_time": time.time() - start_time
            }


# Global RAG system instance (lazy initialization)
_RAG_SYSTEM = None


def get_rag_system() -> EnterpriseRAGSystem:
    """Get or create the RAG system instance (singleton with lazy loading)."""
    global _RAG_SYSTEM
    if _RAG_SYSTEM is None:
        try:
            _RAG_SYSTEM = EnterpriseRAGSystem()
            logger.info("RAG system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    return _RAG_SYSTEM


def answer_question(
        query: str,
        user_id: str,
        profile_id: str,
        model_name: str = "gemini-2.0-flash-exp",
        persona: str = "professional document analysis assistant"
) -> Dict[str, Any]:
    """
    Main entry point for answering questions with enhanced NLU.

    Args:
        query: User's question
        user_id: User identifier
        profile_id: Department/collection name ('finance', 'banking', 'technical')
        model_name: LLM model name (for compatibility)
        persona: Assistant persona

    Returns:
        Response dictionary with enhanced metadata
    """
    rag_system = get_rag_system()
    return rag_system.answer_question(
        query=query,
        profile_id=profile_id,
        user_id=user_id,
        persona=persona
    )


def debug_collection(profile_id: str) -> Dict[str, Any]:
    """
    Debug utility to check collection status with defensive error handling.

    Args:
        profile_id: Collection name

    Returns:
        Collection statistics
    """
    try:
        qdrant_client = get_qdrant_client()
        collection_info = qdrant_client.get_collection(profile_id)

        scroll_result = qdrant_client.scroll(
            collection_name=profile_id,
            limit=3,
            with_payload=True,
            with_vectors=False
        )

        sample_points = []
        if isinstance(scroll_result, tuple) and len(scroll_result) > 0:
            points_list = scroll_result[0]
            if points_list:
                sample_points = [
                    {
                        "id": str(p.id),
                        "text_preview": p.payload.get('text', '')[:200] if p.payload else 'No text',
                        "source": p.payload.get('source', 'unknown') if p.payload else 'unknown'
                    }
                    for p in points_list
                ]

        return {
            "collection_name": profile_id,
            "points_count": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "distance": str(collection_info.config.params.vectors.distance),
            "sample_points": sample_points,
            "status": "healthy"
        }
    except Exception as e:
        logger.error(f"Error debugging collection '{profile_id}': {e}", exc_info=True)
        return {
            "collection_name": profile_id,
            "error": str(e),
            "status": "error"
        }


def add_domain_terms(terms: List[str]):
    """
    Add domain-specific terms to spell checker whitelist.

    Args:
        terms: List of domain-specific terms to whitelist
    """
    try:
        rag_system = get_rag_system()
        rag_system.spell_corrector.add_domain_terms(terms)
        logger.info(f"Added {len(terms)} domain terms to spell checker")
    except Exception as e:
        logger.error(f"Error adding domain terms: {e}")


def clear_conversation_history(user_id: str):
    """
    Clear conversation history for a specific user.

    Args:
        user_id: User identifier
    """
    try:
        rag_system = get_rag_system()
        rag_system.conversation_history.clear_history(user_id)
        logger.info(f"Cleared conversation history for user {user_id}")
    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}")


class RAGEvaluator:
    """Evaluation utilities for monitoring RAG performance."""

    @staticmethod
    def evaluate_retrieval(
            queries: List[str],
            profile_id: str,
            ground_truth_docs: List[List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality using MRR and Precision@K.

        Args:
            queries: List of test queries
            profile_id: Collection name
            ground_truth_docs: List of lists of relevant document IDs for each query

        Returns:
            Dictionary of evaluation metrics
        """
        if len(queries) != len(ground_truth_docs):
            raise ValueError("Queries and ground truth must have same length")

        rag_system = get_rag_system()
        mrr_scores = []
        p_at_1 = []
        p_at_3 = []
        p_at_5 = []

        for query, relevant_docs in zip(queries, ground_truth_docs):
            try:
                chunks = rag_system.retriever.retrieve(
                    collection_name=profile_id,
                    query=query,
                    top_k=10
                )

                retrieved_ids = [chunk.id for chunk in chunks]

                for rank, doc_id in enumerate(retrieved_ids, 1):
                    if doc_id in relevant_docs:
                        mrr_scores.append(1.0 / rank)
                        break
                else:
                    mrr_scores.append(0.0)

                p_at_1.append(1.0 if retrieved_ids[:1] and retrieved_ids[0] in relevant_docs else 0.0)

                if len(retrieved_ids) >= 3:
                    p_at_3.append(
                        len(set(retrieved_ids[:3]) & set(relevant_docs)) / 3.0
                    )

                if len(retrieved_ids) >= 5:
                    p_at_5.append(
                        len(set(retrieved_ids[:5]) & set(relevant_docs)) / 5.0
                    )

            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {e}")

        return {
            "mrr": np.mean(mrr_scores) if mrr_scores else 0.0,
            "precision_at_1": np.mean(p_at_1) if p_at_1 else 0.0,
            "precision_at_3": np.mean(p_at_3) if p_at_3 else 0.0,
            "precision_at_5": np.mean(p_at_5) if p_at_5 else 0.0,
            "num_queries": len(queries)
        }


__all__ = [
    'answer_question',
    'debug_collection',
    'add_domain_terms',
    'clear_conversation_history',
    'RAGEvaluator',
    'get_rag_system'
]



