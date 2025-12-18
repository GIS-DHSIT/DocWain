# import re
# from qdrant_client.http.exceptions import UnexpectedResponse
# import json
# import time
# import logging
# import hashlib
# import numpy as np
# from typing import List, Dict, Any, Optional, Tuple
# from dataclasses import dataclass
# from collections import deque, Counter
# import faiss
# from rank_bm25 import BM25Okapi
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, PointStruct
# from sklearn.preprocessing import MinMaxScaler
# from sentence_transformers import SentenceTransformer, CrossEncoder
# import ollama
# import os
# import redis
# import google.generativeai as genai
# from api.config import Config
# import nltk
# from nltk.corpus import wordnet
# from spellchecker import SpellChecker
#
# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)
#
#
# # Download required NLTK data only when missing to avoid repeated network calls
# def ensure_nltk_data():
#     required = {
#         "wordnet": "corpora/wordnet",
#         "omw-1.4": "corpora/omw-1.4",
#         "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
#         "stopwords": "corpora/stopwords",
#         "punkt": "tokenizers/punkt",
#     }
#     for package, resource in required.items():
#         try:
#             nltk.data.find(resource)
#         except LookupError:
#             try:
#                 nltk.download(package, quiet=True)
#             except Exception as e:  # pragma: no cover - defensive logging
#                 logger.warning(f"Failed to download NLTK package {package}: {e}")
#
#
# ensure_nltk_data()
#
# # Initialize models and clients (lazy loading to avoid startup errors)
# _MODEL = None
# _CROSS_ENCODER = None
# _SPELL_CHECKER = None
# _QDRANT_CLIENT = None
# _REDIS_CLIENT = None
# _MODEL_CACHE: Dict[int, SentenceTransformer] = {}
#
#
# def _load_model_candidates(required_dim: Optional[int] = None) -> SentenceTransformer:
#     candidates = [getattr(Config.Model, "SENTENCE_TRANSFORMERS", "sentence-transformers/all-mpnet-base-v2")]
#     last_error = None
#     for name in candidates:
#         try:
#             logger.info(f"Loading sentence transformer model: {name}")
#             model = SentenceTransformer(name)
#             dim = model.get_sentence_embedding_dimension()
#             logger.info(f"Loaded model '{name}' with dim={dim}")
#             if required_dim is None or dim == required_dim:
#                 return model
#             # cache for later but continue if dim mismatch
#             _MODEL_CACHE[dim] = model
#         except Exception as e:
#             last_error = e
#             logger.warning(f"Failed to load model '{name}': {e}")
#     raise RuntimeError(f"Could not load any sentence transformer model from {candidates}: {last_error}")
#
#
# def get_model(required_dim: Optional[int] = None):
#     """Lazy load sentence transformer model."""
#     global _MODEL
#     if _MODEL is None:
#         _MODEL = _load_model_candidates()
#
#     if required_dim is None:
#         return _MODEL
#
#     dim = _MODEL.get_sentence_embedding_dimension()
#     if dim != required_dim:
#         raise ValueError(f"Loaded model dim {dim} does not match required {required_dim}; using single model only.")
#     return _MODEL
#
#
# def get_cross_encoder():
#     """Lazy load cross encoder model."""
#     global _CROSS_ENCODER
#     if _CROSS_ENCODER is None:
#         try:
#             _CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
#             logger.info("Loaded cross-encoder model")
#         except Exception as e:
#             logger.error(f"Failed to load cross-encoder: {e}")
#             raise
#     return _CROSS_ENCODER
#
#
# def get_spell_checker():
#     """Lazy load spell checker."""
#     global _SPELL_CHECKER
#     if _SPELL_CHECKER is None:
#         try:
#             _SPELL_CHECKER = SpellChecker()
#             logger.info("Initialized spell checker")
#         except Exception as e:
#             logger.error(f"Failed to initialize spell checker: {e}")
#             raise
#     return _SPELL_CHECKER
#
#
# def get_qdrant_client():
#     """Lazy load Qdrant client."""
#     global _QDRANT_CLIENT
#     if _QDRANT_CLIENT is None:
#         try:
#             _QDRANT_CLIENT = QdrantClient(
#                 url=Config.Qdrant.URL,
#                 api_key=Config.Qdrant.API,
#                 timeout=120
#             )
#             logger.info("Initialized Qdrant client")
#         except Exception as e:
#             logger.error(f"Failed to initialize Qdrant client: {e}")
#             raise
#     return _QDRANT_CLIENT
#
#
# def get_redis_client():
#     """Lazy init for Redis client."""
#     global _REDIS_CLIENT
#     if _REDIS_CLIENT is None:
#         try:
#             _REDIS_CLIENT = redis.Redis(
#                 host=Config.Redis.HOST,
#                 port=Config.Redis.PORT,
#                 username=Config.Redis.USERNAME or None,
#                 password=Config.Redis.PASSWORD or None,
#                 db=Config.Redis.DB,
#                 decode_responses=True,
#                 socket_timeout=5,
#                 ssl=getattr(Config.Redis, "SSL", False),
#             )
#             # simple ping
#             _REDIS_CLIENT.ping()
#             logger.info(
#                 "Initialized Redis client at %s:%s (ssl=%s)",
#                 Config.Redis.HOST,
#                 Config.Redis.PORT,
#                 getattr(Config.Redis, "SSL", False)
#             )
#         except Exception as e:
#             logger.warning(f"Failed to initialize Redis client, caching disabled: {e}")
#             _REDIS_CLIENT = None
#     return _REDIS_CLIENT
#
#
# def configure_gemini():
#     """Configure Gemini API with proper error handling."""
#     try:
#         api_key = getattr(Config.Model, "GEMINI_API_KEY", None) or getattr(Config.Gemini, "GEMINI_API_KEY", None)
#         if not api_key:
#             raise ValueError("GEMINI_API_KEY is not set in configuration")
#         genai.configure(api_key=api_key)
#         logger.info("Gemini API configured successfully")
#     except Exception as e:
#         logger.error(f"Failed to configure Gemini API: {e}")
#         raise
#
#
# @dataclass
# class RetrievedChunk:
#     """Data class for retrieved document chunks with metadata."""
#     id: str
#     text: str
#     score: float
#     source: Optional[str] = None
#     metadata: Optional[Dict[str, Any]] = None
#
#
# @dataclass
# class ConversationTurn:
#     """Data class for conversation history."""
#     user_message: str
#     assistant_response: str
#     timestamp: float
#
#
# @dataclass
# class ProfileContextSnapshot:
#     """Lightweight snapshot of profile-specific vocabulary and hints."""
#     top_keywords: List[str]
#     document_hints: List[str]
#     total_chunks: int
#     last_updated: float
#
#
# class ChatFeedbackMemory:
#     """Stores compact Q/A feedback to steer future responses."""
#
#     def __init__(self, max_items: int = 12):
#         self.max_items = max_items
#         self.memories: Dict[str, deque] = {}
#
#     def add_feedback(self, user_id: str, query: str, answer: str, sources: List[Dict[str, Any]]):
#         if user_id not in self.memories:
#             self.memories[user_id] = deque(maxlen=self.max_items)
#         src_names = [s.get("source_name") for s in (sources or []) if s.get("source_name")]
#         self.memories[user_id].append({
#             "q": query.strip(),
#             "a": answer.strip(),
#             "sources": src_names,
#             "ts": time.time()
#         })
#
#     def build_feedback_context(self, user_id: str, limit: int = 5) -> str:
#         if user_id not in self.memories or not self.memories[user_id]:
#             return ""
#         recent = list(self.memories[user_id])[-limit:]
#         lines = []
#         for idx, item in enumerate(recent, 1):
#             source_hint = f" | sources: {', '.join(item['sources'][:3])}" if item.get("sources") else ""
#             lines.append(f"{idx}) Q: {item['q']} | A: {item['a'][:180]}{source_hint}")
#         return "RECENT CHAT FEEDBACK (reuse tone/precision):\n" + "\n".join(lines) + "\n"
#
#
# class ConversationHistory:
#     """Manages conversation history with a sliding window."""
#
#     def __init__(self, max_turns: int = 3):
#         self.max_turns = max_turns
#         self.histories: Dict[str, deque] = {}
#         self.recent_docs: Dict[str, deque] = {}
#
#     def add_turn(self, user_id: str, user_message: str, assistant_response: str):
#         """Add a conversation turn to history."""
#         if user_id not in self.histories:
#             self.histories[user_id] = deque(maxlen=self.max_turns)
#
#         turn = ConversationTurn(
#             user_message=user_message,
#             assistant_response=assistant_response,
#             timestamp=time.time()
#         )
#         self.histories[user_id].append(turn)
#
#     def add_sources(self, user_id: str, doc_ids: List[str]):
#         """Track recently used document IDs for recency-based boosting."""
#         if user_id not in self.recent_docs:
#             self.recent_docs[user_id] = deque(maxlen=10)
#         for doc_id in doc_ids:
#             if doc_id:
#                 self.recent_docs[user_id].append(doc_id)
#
#     def get_context(self, user_id: str, max_turns: int = 2) -> str:
#         """Get recent conversation context as formatted string."""
#         if user_id not in self.histories or not self.histories[user_id]:
#             return ""
#
#         context_parts = []
#         recent_turns = list(self.histories[user_id])[-max_turns:]
#
#         for turn in recent_turns:
#             context_parts.append(f"User: {turn.user_message}")
#             context_parts.append(f"Assistant: {turn.assistant_response}")
#
#         return "\n".join(context_parts)
#
#     def get_recent_doc_ids(self, user_id: str) -> List[str]:
#         """Return a list of recently cited document IDs for this user."""
#         if user_id not in self.recent_docs:
#             return []
#         return list(self.recent_docs[user_id])
#
#     def clear_history(self, user_id: str):
#         """Clear conversation history for a user."""
#         if user_id in self.histories:
#             self.histories[user_id].clear()
#         if user_id in self.recent_docs:
#             self.recent_docs[user_id].clear()
#
#
# class SpellCorrector:
#     """Handles spelling correction with domain-aware corrections."""
#
#     def __init__(self):
#         self.spell_checker = get_spell_checker()
#         self.domain_terms = set()
#
#     def add_domain_terms(self, terms: List[str]):
#         """Add domain-specific terms to whitelist."""
#         self.domain_terms.update(term.lower() for term in terms)
#         self.spell_checker.word_frequency.load_words(terms)
#
#     def correct_text(self, text: str) -> Tuple[str, List[str]]:
#         """
#         Correct spelling in text while preserving domain terms.
#
#         Returns:
#             Tuple of (corrected_text, list_of_corrections)
#         """
#         words = text.split()
#         corrected_words = []
#         corrections = []
#
#         for word in words:
#             clean_word = re.sub(r'[^\w]', '', word.lower())
#             if not clean_word or clean_word in self.domain_terms or len(clean_word) <= 2:
#                 corrected_words.append(word)
#                 continue
#
#             if clean_word not in self.spell_checker:
#                 corrected = self.spell_checker.correction(clean_word)
#                 if corrected and corrected != clean_word:
#                     corrections.append(f"{clean_word} � {corrected}")
#                     corrected_words.append(word.replace(clean_word, corrected))
#                 else:
#                     corrected_words.append(word)
#             else:
#                 corrected_words.append(word)
#
#         return ' '.join(corrected_words), corrections
#
#
# class QueryExpander:
#     """Expands queries with synonyms and related terms."""
#
#     @staticmethod
#     def get_synonyms(word: str, max_synonyms: int = 2) -> List[str]:
#         """Get synonyms using WordNet."""
#         synonyms = set()
#         try:
#             for syn in wordnet.synsets(word):
#                 for lemma in syn.lemmas():
#                     synonym = lemma.name().replace('_', ' ')
#                     if synonym.lower() != word.lower():
#                         synonyms.add(synonym.lower())
#                     if len(synonyms) >= max_synonyms:
#                         break
#                 if len(synonyms) >= max_synonyms:
#                     break
#         except Exception as e:
#             logger.debug(f"Error getting synonyms for {word}: {e}")
#
#         return list(synonyms)
#
#     @staticmethod
#     def expand_query(query: str, max_synonyms_per_word: int = 2) -> str:
#         """Expand query with synonyms for key terms."""
#         try:
#             words = nltk.word_tokenize(query.lower())
#             tagged = nltk.pos_tag(words)
#
#             try:
#                 stopwords = set(nltk.corpus.stopwords.words('english'))
#             except:
#                 stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
#
#             expanded_terms = [query]
#
#             for word, pos in tagged:
#                 if word in stopwords or len(word) <= 3:
#                     continue
#
#                 if pos.startswith('NN') or pos.startswith('VB'):
#                     synonyms = QueryExpander.get_synonyms(word, max_synonyms_per_word)
#                     expanded_terms.extend(synonyms)
#
#             return ' '.join(expanded_terms)
#
#         except Exception as e:
#             logger.debug(f"Error expanding query: {e}")
#             return query
#
#
# class TextPreprocessor:
#     """Handles text preprocessing for consistent tokenization."""
#
#     def __init__(self):
#         try:
#             self.stopwords = set(nltk.corpus.stopwords.words('english'))
#         except:
#             self.stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
#
#     @staticmethod
#     def normalize_text(text: str) -> str:
#         """Normalize text by lowercasing and removing extra whitespace."""
#         text = text.lower().strip()
#         text = re.sub(r'\s+', ' ', text)
#         return text
#
#     def tokenize(self, text: str) -> List[str]:
#         """Tokenize text with lemmatization and stopword removal for BM25."""
#         text = text.lower()
#         text = re.sub(r'[^\w\s]', ' ', text)
#         tokens = text.split()
#
#         tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]
#
#         lemmatized = []
#         for token in tokens:
#             if token.endswith('ing'):
#                 lemmatized.append(token[:-3])
#             elif token.endswith('ed'):
#                 lemmatized.append(token[:-2])
#             elif token.endswith('s') and len(token) > 3:
#                 lemmatized.append(token[:-1])
#             else:
#                 lemmatized.append(token)
#
#         return lemmatized
#
#
# class GreetingHandler:
#     """Handles greeting and farewell detection."""
#
#     GREETINGS = {
#         'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
#         'greetings', 'howdy', 'what\'s up', 'how are you', 'how do you do',
#         'nice to meet you', 'good day', 'hiya', 'sup', 'yo', 'salutations',
#         'welcome', 'hola', 'bonjour', 'namaste', 'aloha', 'hii', 'Hii'}
#
#     FAREWELLS = {
#         'bye', 'goodbye', 'good bye', 'see you', 'see ya', 'farewell',
#         'adieu', 'cheerio', 'tata', 'ta ta', 'catch you later', 'later',
#         'take care', 'until next time', 'signing off', 'gotta go', 'gtg',
#         'peace out', 'ciao', 'au revoir', 'sayonara', 'hasta la vista',
#         'talk to you later', 'ttyl', 'see you soon', 'see you around',
#         'good night', 'goodnight', 'have a good day', 'have a great day',
#         'thanks for your help', 'that\'s all', 'i\'m done', 'end chat',
#         'quit', 'exit', 'close', 'finish', 'terminate', 'stop'
#     }
#
#     POSITIVE_FEEDBACK = {
#         'thanks', 'thank you', 'thanks a lot', 'thank you so much',
#         'appreciate it', 'much appreciated'
#     }
#
#     @classmethod
#     def is_greeting(cls, message: str) -> bool:
#         """Check if message is a greeting."""
#         message = message.lower().strip()
#
#         if len(message) <= 60:
#             pattern = r'\b(' + '|'.join(re.escape(g) for g in cls.GREETINGS) + r')\b'
#             if re.search(pattern, message):
#                 words = message.split()
#                 if len(words) <= 5 or any(g in ' '.join(words[:5]) for g in cls.GREETINGS):
#                     return True
#
#         return False
#
#     @classmethod
#     def is_farewell(cls, message: str) -> bool:
#         """Check if message is a farewell."""
#         message = message.lower().strip()
#
#         if len(message) <= 50:
#             pattern = r'\b(' + '|'.join(re.escape(f) for f in cls.FAREWELLS) + r')\b'
#             return bool(re.search(pattern, message))
#
#         return False
#
#     @classmethod
#     def is_positive_feedback(cls, message: str) -> bool:
#         """Check if message is positive feedback."""
#         message = message.lower().strip()
#
#         if len(message) <= 40:
#             pattern = r'\b(' + '|'.join(re.escape(f) for f in cls.POSITIVE_FEEDBACK) + r')\b'
#             return bool(re.search(pattern, message))
#
#         return False
#
#
# class OllamaClient:
#     """Handles local Ollama model calls with structured output and retries."""
#
#     def __init__(self, model_name: Optional[str] = None):
#         self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.2")
#         if not self.model_name:
#             raise ValueError("OLLAMA_MODEL environment variable is not set")
#         logger.info(f"Initialized OllamaClient with model: {self.model_name}")
#
#     def warm_up(self):
#         """Eagerly load the model so first user queries do not pay startup cost."""
#         try:
#             ollama.generate(model=self.model_name, prompt="ping", options={"num_predict": 1})
#             logger.info("Ollama warm-up successful")
#         except Exception as e:
#             logger.warning(f"Ollama warm-up failed (continuing without warm cache): {e}")
#
#     def generate(
#             self,
#             prompt: str,
#             max_retries: int = 3,
#             backoff: float = 1.0
#     ) -> str:
#         """Generate response with retry logic and robust parsing."""
#         for attempt in range(1, max_retries + 1):
#             try:
#                 response = ollama.generate(model=self.model_name, prompt=prompt)
#                 text = (
#                         response.get("response")
#                         or (response.get("message", {}) or {}).get("content")
#                         or ""
#                 ).strip()
#                 if text:
#                     return text
#                 logger.warning(f"No text in Ollama response: {response}")
#                 return "I apologize, but I couldn't generate a proper response."
#             except Exception as e:
#                 logger.warning(f"Ollama attempt {attempt}/{max_retries} failed: {e}")
#                 if attempt < max_retries:
#                     time.sleep(backoff * attempt)
#                 else:
#                     logger.error(f"All retry attempts failed: {e}")
#                     raise
#
#         return "I apologize, but I encountered an error generating a response."
#
#
# class GeminiClient:
#     """Handles Gemini API calls with structured output and retries."""
#
#     def __init__(self, model_name: Optional[str] = None):
#         configure_gemini()
#         self.model_name = model_name or Config.Model.GEMINI_MODEL_NAME
#         if not self.model_name:
#             raise ValueError("Gemini model name is not configured")
#         self.model = genai.GenerativeModel(self.model_name)
#         self.generation_config = genai.GenerationConfig(
#             temperature=0.0,
#             top_p=0.95,
#             top_k=40,
#             max_output_tokens=2048,
#         )
#         logger.info(f"Initialized GeminiClient with model: {self.model_name}")
#
#     def generate(
#             self,
#             prompt: str,
#             max_retries: int = 3,
#             backoff: float = 1.0
#     ) -> str:
#         """Generate response with retry logic and robust parsing."""
#         for attempt in range(1, max_retries + 1):
#             try:
#                 response = self.model.generate_content(
#                     prompt,
#                     generation_config=self.generation_config
#                 )
#
#                 text = None
#
#                 if hasattr(response, 'text') and response.text:
#                     text = response.text.strip()
#                 elif hasattr(response, 'candidates') and response.candidates:
#                     candidate = response.candidates[0]
#                     if hasattr(candidate, 'content'):
#                         if hasattr(candidate.content, 'parts') and candidate.content.parts:
#                             text = candidate.content.parts[0].text.strip()
#                         elif hasattr(candidate.content, 'text'):
#                             text = candidate.content.text.strip()
#
#                 if text:
#                     return text
#                 else:
#                     logger.warning(f"No text in response: {response}")
#                     return "I apologize, but I couldn't generate a proper response."
#
#             except Exception as e:
#                 logger.warning(f"Gemini API attempt {attempt}/{max_retries} failed: {e}")
#                 if attempt < max_retries:
#                     time.sleep(backoff * attempt)
#                 else:
#                     logger.error(f"All retry attempts failed: {e}")
#                     raise
#
#         return "I apologize, but I encountered an error generating a response."
#
#
# class QueryReformulator:
#     """Reformulates conversational queries into clear, concise search queries."""
#
#     def __init__(self, llm_client):
#         self.llm_client = llm_client
#
#     def reformulate(self, query: str, conversation_context: str = "") -> str:
#         """Reformulate query using LLM to make it more search-friendly."""
#         if len(query.split()) <= 5 and not conversation_context:
#             return query
#
#         prompt = f"""You are a query reformulation assistant. Convert the user's conversational question into a clear, concise search query optimized for semantic search.
#
# RULES:
# 1. Extract the core information need
# 2. Remove filler words and conversational elements
# 3. Keep domain-specific terms and technical vocabulary
# 4. Make it 3-10 words maximum
# 5. If conversation context is provided, resolve pronouns and references
# 6. Output ONLY the reformulated query, nothing else
#
# {f"CONVERSATION CONTEXT:\\n{conversation_context}\\n" if conversation_context else ""}
# USER QUERY: {query}
#
# REFORMULATED QUERY:"""
#
#         try:
#             reformulated = self.llm_client.generate(
#                 prompt,
#                 max_retries=2,
#                 backoff=0.5
#             )
#
#             reformulated = reformulated.strip().strip('"\'')
#
#             if 2 <= len(reformulated.split()) <= 15 and reformulated.lower() != query.lower():
#                 logger.info(f"Reformulated: '{query}' � '{reformulated}'")
#                 return reformulated
#             else:
#                 return query
#
#         except Exception as e:
#             logger.warning(f"Query reformulation failed: {e}")
#             return query
#
#
# class QdrantRetriever:
#     """Handles retrieval from Qdrant using native search functionality."""
#
#     def __init__(self, client: QdrantClient, model: SentenceTransformer):
#         self.client = client
#         self.model = model
#         self.preprocessor = TextPreprocessor()
#         self.profile_context_cache: Dict[Tuple[str, str], ProfileContextSnapshot] = {}
#         self.collection_dims: Dict[str, int] = {}
#
#     def run_search(
#             self,
#             collection_name: str,
#             query_vector: List[float],
#             query_filter: Optional[dict] = None,
#             limit: int = 50,
#             vector_name: str = "content_vector",
#             score_threshold: Optional[float] = None
#     ):
#         """Execute a vector search against Qdrant using query_points."""
#         try:
#             kwargs = dict(
#                 collection_name=collection_name,
#                 query=query_vector,
#                 using=vector_name,
#                 limit=limit,
#                 query_filter=query_filter,
#                 with_payload=True,
#                 with_vectors=False,
#             )
#             if score_threshold is not None:
#                 kwargs["score_threshold"] = score_threshold
#
#             results = self.client.query_points(**kwargs)
#             return results
#         except Exception as e:
#             logger.error("Qdrant query_points error: %s", e, exc_info=True)
#             return None
#
#     def get_collection_vector_dim(self, collection_name: str) -> Optional[int]:
#         """Fetch and cache the expected vector dimension for a collection."""
#         if collection_name in self.collection_dims:
#             return self.collection_dims[collection_name]
#         try:
#             info = self.client.get_collection(collection_name)
#             cfg = getattr(info, "config", None) or {}
#             params = getattr(cfg, "params", None) or {}
#             vectors = getattr(params, "vectors", None) or getattr(params, "vector_size", None) or {}
#             dim = None
#             if hasattr(vectors, "size"):
#                 dim = vectors.size
#             elif isinstance(vectors, dict):
#                 if "size" in vectors:
#                     dim = vectors.get("size")
#                 elif "content_vector" in vectors and isinstance(vectors["content_vector"], dict):
#                     dim = vectors["content_vector"].get("size")
#             if dim is None:
#                 dim = 768  # default to mpnet dimension
#             self.collection_dims[collection_name] = dim
#             logger.info(f"Collection '{collection_name}' expects dim={dim}")
#             return dim
#         except Exception as e:
#             logger.warning(f"Could not fetch vector dim for collection '{collection_name}': {e}")
#             return None
#
#     def retrieve(
#             self,
#             collection_name: str,
#             query: str,
#             filter_profile: str = None,
#             top_k: int = 50,
#             score_threshold: float = None
#     ) -> List[RetrievedChunk]:
#         """Retrieve relevant chunks using Qdrant's native search."""
#         try:
#             target_dim = self.get_collection_vector_dim(collection_name)
#             model = get_model(required_dim=target_dim)
#             q_dim = getattr(model, "get_sentence_embedding_dimension", lambda: None)()
#             if target_dim and q_dim and target_dim != q_dim:
#                 logger.warning(
#                     f"Embedding dim {q_dim} does not match collection dim {target_dim}; using model regardless")
#             query_vector = model.encode(
#                 query,
#                 convert_to_numpy=True,
#                 normalize_embeddings=True
#             ).astype(np.float32).tolist()
#         except Exception as err:
#             logger.error(f"Failed to embed query for retrieval: {err}", exc_info=True)
#             return []
#
#         logger.info(f"Searching collection '{collection_name}' for query: {query[:100]}")
#
#         profile_filter_val = str(filter_profile).strip() if filter_profile is not None else None
#         filter_clauses = []
#         if profile_filter_val:
#             filter_clauses.append({"key": "profile_id", "match": {"value": profile_filter_val}})
#
#         query_filter = {"must": filter_clauses} if filter_clauses else None
#
#         results = self.run_search(
#             collection_name=collection_name,
#             query_vector=query_vector,
#             query_filter=query_filter,
#             limit=top_k,
#             vector_name="content_vector",
#             score_threshold=score_threshold
#         )
#
#         if not results or not getattr(results, "points", []):
#             logger.warning(f"No results found in collection '{collection_name}'")
#             return []
#
#         points = results.points or []
#         logger.info("Qdrant returned %d hits", len(points))
#         logger.info("Top scores: %s", [p.score for p in points[:3]])
#
#         chunks: List[RetrievedChunk] = []
#         for pt in points:
#             payload = pt.payload or {}
#             text = payload.get("text", "")
#             snippet = text[:120].replace("\n", " ")
#             logger.debug("Hit score=%.4f snippet=%s", pt.score, snippet)
#             chunk = RetrievedChunk(
#                 id=str(pt.id),
#                 text=text,
#                 score=float(pt.score),
#                 source=payload.get("source_file", "unknown"),
#                 metadata=payload
#             )
#             chunks.append(chunk)
#
#         return chunks
#
#     def get_profile_context(
#             self,
#             collection_name: str,
#             profile_id: str,
#             max_points: int = 400,
#             refresh_seconds: int = 300
#     ) -> ProfileContextSnapshot:
#         """Build lightweight context from existing embeddings to guide vague queries."""
#         cache_key = (collection_name, str(profile_id))
#         now = time.time()
#         cached = self.profile_context_cache.get(cache_key)
#         if cached and (now - cached.last_updated) < refresh_seconds:
#             return cached
#
#         filter_ = {"must": [{"key": "profile_id", "match": {"value": str(profile_id)}}]}
#         collected_points = []
#         next_offset = None
#         batch_size = min(120, max_points)
#
#         try:
#             while len(collected_points) < max_points:
#                 scroll_result = self.client.scroll(
#                     collection_name=collection_name,
#                     scroll_filter=filter_,
#                     offset=next_offset,
#                     limit=min(batch_size, max_points - len(collected_points)),
#                     with_payload=True,
#                     with_vectors=False
#                 )
#
#                 if hasattr(scroll_result, "points"):
#                     batch = scroll_result.points or []
#                     next_offset = getattr(scroll_result, "next_page_offset", None)
#                 elif isinstance(scroll_result, tuple):
#                     batch = scroll_result[0] if len(scroll_result) > 0 else []
#                     next_offset = scroll_result[1] if len(scroll_result) > 1 else None
#                 else:
#                     batch = []
#                     next_offset = None
#
#                 if not batch:
#                     break
#
#                 collected_points.extend(batch)
#
#                 if not next_offset:
#                     break
#         except Exception as e:
#             logger.warning(f"Failed to build profile context for {profile_id}: {e}")
#             snapshot = ProfileContextSnapshot([], [], 0, now)
#             self.profile_context_cache[cache_key] = snapshot
#             return snapshot
#
#         token_counts: Counter = Counter()
#         doc_hints: List[str] = []
#         seen_hints = set()
#
#         for pt in collected_points:
#             payload = pt.payload or {}
#             text = payload.get("text") or ""
#             if text:
#                 token_counts.update(self.preprocessor.tokenize(text))
#
#             for hint_key in ("source_file", "document_id", "section"):
#                 hint_val = payload.get(hint_key)
#                 if hint_val:
#                     hint_val = str(hint_val)
#                     if hint_val not in seen_hints:
#                         doc_hints.append(hint_val)
#                         seen_hints.add(hint_val)
#
#         top_keywords = [w for w, _ in token_counts.most_common(40)]
#         snapshot = ProfileContextSnapshot(
#             top_keywords=top_keywords,
#             document_hints=doc_hints[:12],
#             total_chunks=len(collected_points),
#             last_updated=now
#         )
#         self.profile_context_cache[cache_key] = snapshot
#         return snapshot
#
#
# class HybridReranker:
#     """Reranks retrieved chunks using BM25 + vector scores with dynamic weighting."""
#
#     def __init__(self, alpha: float = 0.7, cross_encoder: Optional[CrossEncoder] = None):
#         """Initialize reranker."""
#         self.alpha = alpha
#         self.preprocessor = TextPreprocessor()
#         self.cross_encoder = cross_encoder
#
#     def adjust_alpha(self, query: str) -> float:
#         """Dynamically adjust alpha based on query characteristics."""
#         words = query.lower().split()
#
#         question_words = {'what', 'why', 'how', 'explain', 'describe', 'understand'}
#         if any(qw in words for qw in question_words):
#             return 0.75
#
#         if any(re.search(r'\d+', word) for word in words) or len(words) <= 3:
#             return 0.6
#
#         return self.alpha
#
#     def rerank(
#             self,
#             chunks: List[RetrievedChunk],
#             query: str,
#             top_k: int = 10,
#             use_cross_encoder: bool = True
#     ) -> List[RetrievedChunk]:
#         """Rerank chunks using hybrid BM25 + vector scoring."""
#         if not chunks:
#             return []
#
#         try:
#             alpha = self.adjust_alpha(query)
#             logger.info(f"Using alpha={alpha:.2f} for reranking")
#
#             texts = [chunk.text for chunk in chunks]
#             tokenized_corpus = [self.preprocessor.tokenize(text) for text in texts]
#             tokenized_query = self.preprocessor.tokenize(query)
#
#             bm25 = BM25Okapi(tokenized_corpus)
#             bm25_scores = np.array(bm25.get_scores(tokenized_query), dtype=np.float64)
#
#             vector_scores = np.array([float(chunk.score) for chunk in chunks], dtype=np.float64)
#
#             if len(vector_scores) > 1:
#                 v_min, v_max = vector_scores.min(), vector_scores.max()
#                 b_min, b_max = bm25_scores.min(), bm25_scores.max()
#
#                 v_range = v_max - v_min if v_max > v_min else 1.0
#                 b_range = b_max - b_min if b_max > b_min else 1.0
#
#                 vector_scores_norm = (vector_scores - v_min) / v_range
#                 bm25_scores_norm = (bm25_scores - b_min) / b_range
#             else:
#                 vector_scores_norm = np.ones_like(vector_scores)
#                 bm25_scores_norm = np.ones_like(bm25_scores)
#
#             hybrid_scores = (
#                     alpha * vector_scores_norm +
#                     (1 - alpha) * bm25_scores_norm
#             )
#
#             top_n_for_ce = min(top_k * 2, len(chunks))
#             sorted_indices = np.argsort(hybrid_scores)[::-1][:top_n_for_ce]
#
#             candidate_chunks = [chunks[idx] for idx in sorted_indices]
#
#             if use_cross_encoder and self.cross_encoder and len(candidate_chunks) > 1:
#                 try:
#                     pairs = [[query, chunk.text] for chunk in candidate_chunks]
#                     ce_scores = self.cross_encoder.predict(pairs)
#
#                     for i, score in enumerate(ce_scores):
#                         candidate_chunks[i].score = float(score)
#
#                     candidate_chunks.sort(key=lambda c: c.score, reverse=True)
#                     logger.info(f"Applied CrossEncoder reranking to {len(candidate_chunks)} chunks")
#
#                 except Exception as e:
#                     logger.warning(f"CrossEncoder reranking failed: {e}")
#                     for i, chunk in enumerate(candidate_chunks):
#                         chunk.score = float(hybrid_scores[sorted_indices[i]])
#             else:
#                 for i, chunk in enumerate(candidate_chunks):
#                     chunk.score = float(hybrid_scores[sorted_indices[i]])
#
#             reranked_chunks = candidate_chunks[:top_k]
#
#             logger.info(f"Reranked to {len(reranked_chunks)} chunks (alpha={alpha:.2f})")
#             return reranked_chunks
#
#         except Exception as e:
#             logger.error(f"Error in reranking: {e}", exc_info=True)
#             return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]
#
#
# class ContextBuilder:
#     """Builds formatted context for LLM with source citations."""
#
#     @staticmethod
#     def build_source_hints(chunks: List[RetrievedChunk]) -> str:
#         """Create a compact source map to bias model attention to top evidence."""
#         if not chunks:
#             return ""
#         lines = []
#         for i, chunk in enumerate(chunks[:5], 1):
#             meta = chunk.metadata or {}
#             source_name = chunk.source or meta.get('source_file', f"doc_{chunk.id[:8]}")
#             page = meta.get('page')
#             section = meta.get('section')
#             score = round(float(chunk.score), 3)
#             parts = [f"{i}) {source_name}", f"score={score}"]
#             if page is not None:
#                 parts.append(f"page={page}")
#             if section:
#                 parts.append(f"section={section}")
#             lines.append(" | ".join(parts))
#         return "SOURCE MAP:\n" + "\n".join(lines) + "\n"
#
#     @staticmethod
#     def build_context(chunks: List[RetrievedChunk], max_chunks: int = 3) -> str:
#         """Build context string with source citations."""
#         if not chunks:
#             return ""
#
#         seen_texts = set()
#         unique_chunks = []
#         for chunk in chunks:
#             normalized = ' '.join(chunk.text.split())
#             if normalized not in seen_texts and normalized.strip():
#                 seen_texts.add(normalized)
#                 unique_chunks.append(chunk)
#
#         selected_chunks = unique_chunks[:max_chunks]
#
#         context_parts = []
#         source_map = ContextBuilder.build_source_hints(selected_chunks)
#         if source_map:
#             context_parts.append(source_map)
#         for i, chunk in enumerate(selected_chunks, 1):
#             # Use source_file directly from metadata as fallback
#             source_name = chunk.source or chunk.metadata.get('source_file', f"doc_{chunk.id[:8]}")
#             context_parts.append(
#                 f"[SOURCE: {source_name}]\n{chunk.text}\n[/SOURCE]"
#             )
#
#         return "\n".join(context_parts)
#
#     @staticmethod
#     def extract_sources(chunks: List[RetrievedChunk], max_sources: int = 3) -> List[Dict[str, Any]]:
#         """Extract source information for response metadata."""
#         sources = []
#         for i, chunk in enumerate(chunks[:max_sources], 1):
#             sources.append({
#                 'source_id': i,
#                 'source_name': chunk.source or f"Document {chunk.id[:8]}",
#                 'excerpt': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
#                 'relevance_score': round(float(chunk.score), 3)
#             })
#         return sources
#
#
# class DomainPromptAdapter:
#     """Creates lightweight, in-context adapters to steer local LLM responses."""
#
#     @staticmethod
#     def build_adapter(profile_context: Dict[str, Any], query: str) -> str:
#         if not profile_context:
#             return ""
#
#         keywords = profile_context.get("keywords") or []
#         hints = profile_context.get("hints") or []
#         sampled = profile_context.get("sampled_chunks", 0)
#
#         adapter_lines = []
#         if keywords:
#             adapter_lines.append(
#                 f"Domain keywords to preserve and prefer in wording: {', '.join(keywords[:12])}."
#             )
#         if hints:
#             adapter_lines.append(f"Relevant documents/sections to anchor on: {', '.join(hints[:6])}.")
#         if sampled:
#             adapter_lines.append(f"Context built from {sampled} profile chunks; avoid generic responses.")
#
#         adapter_lines.append(
#             "Favor the terminology above when answering; align synonyms in the question to these domain terms."
#         )
#         adapter_lines.append(
#             f"If the user query is vague ('{query}'), proactively ground the answer in the domain cues above."
#         )
#         return "\n".join(adapter_lines)
#
#
# class AnswerabilityDetector:
#     """Detects if a question can be answered from provided context."""
#
#     def __init__(self, llm_client):
#         self.llm_client = llm_client
#
#     def check_answerability(self, query: str, context: str, has_chunks: bool = False) -> Tuple[bool, str]:
#         """
#         Check if the query can be answered from the context.
#
#         If we already have retrieved chunks, bias toward answering to avoid premature
#         "cannot answer" responses when relevant evidence exists.
#         """
#         if has_chunks and context.strip():
#             return True, "Context present from retrieved chunks"
#
#         prompt = f"""You are an answerability classifier. Determine if the USER QUESTION can be answered using ONLY the information in the DOCUMENT CONTEXT.
#
# DOCUMENT CONTEXT:
# {context}
#
# USER QUESTION: {query}
#
# Respond with ONLY one of these formats:
# - If answerable: "ANSWERABLE: <brief reason>"
# - If not answerable: "NOT_ANSWERABLE: <what information is missing>"
#
# Your response:"""
#
#         try:
#             response = self.llm_client.generate(prompt, max_retries=2)
#             response = response.strip()
#
#             if response.startswith("ANSWERABLE"):
#                 return True, response.replace("ANSWERABLE:", "").strip()
#             elif response.startswith("NOT_ANSWERABLE"):
#                 return False, response.replace("NOT_ANSWERABLE:", "").strip()
#             else:
#                 return True, "Classification unclear"
#
#         except Exception as e:
#             logger.warning(f"Answerability check failed: {e}")
#             return True, "Check failed"
#
#
# class PromptBuilder:
#     """Builds structured prompts with strict grounding and citation requirements."""
#
#     @staticmethod
#     def build_qa_prompt(
#             query: str,
#             context: str,
#             persona: str,
#             conversation_summary: str = "",
#             domain_guidance: str = "",
#             feedback_memory: str = ""
#     ) -> str:
#         """Build a structured QA prompt with grounding and citation requirements."""
#         convo_block = f"\nPRIOR CONVERSATION SUMMARY:\n{conversation_summary}\n" if conversation_summary else ""
#         domain_block = f"\nDOMAIN ADAPTER:\n{domain_guidance}\n" if domain_guidance else ""
#         feedback_block = f"\nRECENT FEEDBACK:\n{feedback_memory}\n" if feedback_memory else ""
#         prompt = f"""You are a {persona} specialized in providing accurate, document-based answers with STRICT GROUNDING.
#
# CRITICAL RULES FOR GROUNDING:
# 1. Answer ONLY using information explicitly stated in the DOCUMENT CONTEXT below
# 2. You MUST cite sources using [SOURCE-X] notation for EVERY factual claim
# 3. If information is not in the documents, you MUST explicitly state: "The provided documents do not contain information about [specific topic]"
# 4. NEVER add facts, opinions, estimates, or information not present in the documents
# 5. When quoting, use exact phrases from the documents
# 6. If the documents provide partial information, state what is available and what is missing
# 7. If documents conflict, cite both sources and note the discrepancy
#
# DOCUMENT CONTEXT:
# {context}
# {convo_block}
# {domain_block}
# {feedback_block}
#
# USER QUESTION: {query}
#
# RESPONSE REQUIREMENTS:
# - Write in a natural, human tone with 2-5 concise sentences.
# - Keep it conversational and approachable, as if explaining to a teammate.
# - Do NOT repeat or paraphrase the user question; go straight to the answer.
# - Lead with the direct answer; include brief reasoning or supporting evidence.
# - Cite sources using [SOURCE-X] notation after each claim.
# - If information is partial, say what is known and what is missing (without generic disclaimers).
# - Avoid filler like "based on the available documents"; just state the supported facts.
# - NEVER speculate or add external knowledge.
#
# Provide your grounded answer now:"""
#
#         return prompt
#
#
# class ConversationSummarizer:
#     """Summarizes the last few turns to keep context tight."""
#
#     def __init__(self, llm_client):
#         self.llm_client = llm_client
#
#     def summarize(self, conversation_text: str) -> str:
#         if not conversation_text:
#             return ""
#         prompt = f"""Summarize the following conversation turns into 3-5 concise bullets capturing user intent and assistant answers. Do NOT invent details.
#
# CONVERSATION:
# {conversation_text}
#
# SUMMARY:"""
#         try:
#             summary = self.llm_client.generate(prompt, max_retries=2, backoff=0.5)
#             return summary.strip()
#         except Exception as e:
#             logger.warning(f"Conversation summarization failed: {e}")
#             return ""
#
#
# class EnterpriseRAGSystem:
#     """
#     Enhanced RAG system with query reformulation, spelling correction,
#     conversational context, cross-encoder reranking, and grounding.
#     """
#
#     def __init__(self, model_name: Optional[str] = None):
#         """Initialize the RAG system with lazy-loaded components."""
#         try:
#             # Initialize LLM client (Ollama by default, Gemini when requested)
#             self.llm_client = create_llm_client(model_name)
#
#             # Initialize other components
#             qdrant_client = get_qdrant_client()
#             model = get_model()
#             cross_encoder = get_cross_encoder()
#
#             self.client = qdrant_client
#             self.retriever = QdrantRetriever(qdrant_client, model)
#             self.reranker = HybridReranker(alpha=0.7, cross_encoder=cross_encoder)
#             self.context_builder = ContextBuilder()
#             self.prompt_builder = PromptBuilder()
#             self.greeting_handler = GreetingHandler()
#             self.spell_corrector = SpellCorrector()
#             self.query_expander = QueryExpander()
#             self.query_reformulator = QueryReformulator(self.llm_client)
#             self.answerability_detector = AnswerabilityDetector(self.llm_client)
#             self.conversation_history = ConversationHistory(max_turns=3)
#             self.conversation_summarizer = ConversationSummarizer(self.llm_client)
#             self.feedback_memory = ChatFeedbackMemory(max_items=12)
#             self._warm_up_llm()
#
#             logger.info("EnterpriseRAGSystem initialized successfully")
#         except Exception as e:
#             logger.error(f"Failed to initialize EnterpriseRAGSystem: {e}")
#             raise
#
#     def preprocess_query(
#             self,
#             query: str,
#             user_id: str,
#             use_reformulation: bool = True
#     ) -> Tuple[str, Dict[str, Any]]:
#         """Preprocess query with spelling correction, reformulation, and light expansion."""
#         metadata = {
#             'original_query': query,
#             'corrections': [],
#             'reformulated': False,
#             'expanded': False
#         }
#
#         processed_query = query
#
#         # Spelling corrections (preserve names if no corrections)
#         corrected_query, corrections = self.spell_corrector.correct_text(query)
#         if corrections:
#             metadata['corrections'] = corrections
#             processed_query = corrected_query
#
#         # Reformulate for vaguer prompts to tighten intent
#         if use_reformulation and len(processed_query.split()) >= 4:
#             conv_context = self.conversation_history.get_context(user_id, max_turns=2)
#             reformulated = self.query_reformulator.reformulate(processed_query, conv_context)
#             if reformulated and reformulated != processed_query:
#                 processed_query = reformulated
#                 metadata['reformulated'] = True
#
#         # Light expansion to add semantically related terms
#         expanded_query = self.query_expander.expand_query(processed_query)
#         if expanded_query and expanded_query != processed_query:
#             processed_query = expanded_query
#             metadata['expanded'] = True
#             metadata['expanded_query'] = expanded_query
#
#         return processed_query, metadata
#
#     @staticmethod
#     def _is_retrieval_sufficient(chunks: List[RetrievedChunk], min_hits: int = 3, min_score: float = 0.18) -> bool:
#         """Decide if current retrieval is good enough to stop trying fallbacks."""
#         if not chunks:
#             return False
#         if len(chunks) >= min_hits:
#             return True
#         top_score = float(chunks[0].score)
#         return top_score >= min_score
#
#     @staticmethod
#     def _is_query_vague(query: str) -> bool:
#         """Heuristic to detect short or underspecified questions."""
#         tokens = query.split()
#         if len(tokens) <= 3:
#             return True
#         meaningful_tokens = [t for t in tokens if len(t) > 3]
#         return len(meaningful_tokens) <= 2
#
#     @staticmethod
#     def _contextualize_query(query: str, profile_context: ProfileContextSnapshot) -> Tuple[str, Dict[str, Any]]:
#         """Blend the user query with profile-specific hints to guide retrieval."""
#         if not profile_context or not (profile_context.top_keywords or profile_context.document_hints):
#             return query, {}
#
#         keywords = profile_context.top_keywords[:8]
#         hints = profile_context.document_hints[:3]
#         extras = []
#         if hints:
#             extras.append("related to " + ", ".join(hints))
#         if keywords:
#             extras.append("keywords: " + ", ".join(keywords))
#
#         contextual_query = " ; ".join([query] + extras)
#         return contextual_query, {"profile_keywords_used": keywords, "profile_hints_used": hints}
#
#     def retrieve_with_priorities(
#             self,
#             query: str,
#             user_id: str,
#             profile_id: str,
#             collection_name: str,
#             top_k_retrieval: int = 50
#     ) -> Dict[str, Any]:
#         """
#         Try Qdrant retrieval first, fall back to relaxed thresholds, then chat-history
#         reformulation as a last resort.
#         """
#         profile_context = self.retriever.get_profile_context(collection_name, profile_id)
#         profile_context_data = {
#             "keywords": profile_context.top_keywords[:12],
#             "hints": profile_context.document_hints[:6],
#             "sampled_chunks": profile_context.total_chunks
#         }
#
#         if profile_context.top_keywords:
#             # Protect domain terms from being "corrected" away
#             self.spell_corrector.add_domain_terms(profile_context.top_keywords[:50])
#
#         is_vague = self._is_query_vague(query)
#         primary_min_hits = 2 if is_vague else 3
#         primary_min_score = 0.12 if is_vague else 0.18
#         primary_threshold = 0.18 if is_vague else 0.25
#
#         attempt_records = []
#         retrieval_runs = []
#
#         def run_attempt(label: str, query_text: str, threshold: Optional[float], use_history: bool,
#                         metadata: Dict[str, Any]):
#             chunks = self.retriever.retrieve(
#                 collection_name=collection_name,
#                 filter_profile=profile_id,
#                 query=query_text,
#                 top_k=top_k_retrieval,
#                 score_threshold=threshold
#             )
#             top_score = float(chunks[0].score) if chunks else 0.0
#             record = {
#                 "label": label,
#                 "query": query_text,
#                 "score_threshold": threshold,
#                 "hits": len(chunks),
#                 "top_score": round(top_score, 4),
#                 "used_history": use_history
#             }
#             attempt_records.append(record)
#             retrieval_runs.append((chunks, record, metadata, query_text))
#             return chunks
#
#         primary_query, primary_metadata = self.preprocess_query(query, user_id, use_reformulation=False)
#         primary_metadata["vague_query"] = is_vague
#         primary_metadata["profile_context"] = profile_context_data
#
#         primary_chunks = run_attempt("direct_qdrant", primary_query, primary_threshold, False, primary_metadata)
#         if self._is_retrieval_sufficient(primary_chunks, min_hits=primary_min_hits, min_score=primary_min_score):
#             return {
#                 "chunks": primary_chunks,
#                 "query": primary_query,
#                 "metadata": primary_metadata,
#                 "attempts": attempt_records,
#                 "selected_strategy": "direct_qdrant",
#                 "profile_context": profile_context_data
#             }
#
#         contextual_query, contextual_meta = self._contextualize_query(primary_query, profile_context)
#         if contextual_query != primary_query:
#             contextual_metadata = {**primary_metadata, **contextual_meta, "contextualized": True}
#             contextual_threshold = 0.15 if is_vague else 0.2
#             contextual_chunks = run_attempt("contextual_qdrant", contextual_query, contextual_threshold, False,
#                                             contextual_metadata)
#             if self._is_retrieval_sufficient(contextual_chunks, min_hits=primary_min_hits,
#                                              min_score=primary_min_score):
#                 return {
#                     "chunks": contextual_chunks,
#                     "query": contextual_query,
#                     "metadata": contextual_metadata,
#                     "attempts": attempt_records,
#                     "selected_strategy": "contextual_qdrant",
#                     "profile_context": profile_context_data
#                 }
#
#         relaxed_chunks = run_attempt("relaxed_qdrant", primary_query, None, False, primary_metadata)
#         if self._is_retrieval_sufficient(relaxed_chunks, min_hits=primary_min_hits, min_score=primary_min_score):
#             return {
#                 "chunks": relaxed_chunks,
#                 "query": primary_query,
#                 "metadata": primary_metadata,
#                 "attempts": attempt_records,
#                 "selected_strategy": "relaxed_qdrant",
#                 "profile_context": profile_context_data
#             }
#
#         history_query, history_metadata = self.preprocess_query(query, user_id, use_reformulation=True)
#         history_metadata["profile_context"] = profile_context_data
#         if history_query != primary_query or history_metadata.get("reformulated") or history_metadata.get("expanded"):
#             history_chunks = run_attempt("history_guided", history_query, None, True, history_metadata)
#             if self._is_retrieval_sufficient(history_chunks, min_hits=primary_min_hits, min_score=primary_min_score):
#                 return {
#                     "chunks": history_chunks,
#                     "query": history_query,
#                     "metadata": history_metadata,
#                     "attempts": attempt_records,
#                     "selected_strategy": "history_guided",
#                     "profile_context": profile_context_data
#                 }
#
#         best_run = max(retrieval_runs, key=lambda r: r[1]["top_score"], default=None)
#         selected_chunks, selected_record, selected_meta, selected_query = best_run if best_run else ([], {},
#                                                                                                      primary_metadata,
#                                                                                                      primary_query)
#         return {
#             "chunks": selected_chunks,
#             "query": selected_query,
#             "metadata": selected_meta,
#             "attempts": attempt_records,
#             "selected_strategy": selected_record.get("label", "none"),
#             "profile_context": profile_context_data
#         }
#
#     def _warm_up_llm(self):
#         """Warm LLM backend so first user calls do not fail cold."""
#         warm_fn = getattr(self.llm_client, "warm_up", None)
#         if callable(warm_fn):
#             try:
#                 warm_fn()
#             except Exception as e:
#                 logger.warning(f"LLM warm-up skipped due to error: {e}")
#
#     def answer_question(
#             self,
#             query: str,
#             profile_id: str,
#             subscription_id: str,
#             user_id: str,
#             persona: str = "professional document analysis assistant",
#             top_k_retrieval: int = 50,
#             top_k_rerank: int = 15,
#             final_k: int = 7
#     ) -> Dict[str, Any]:
#         """Enhanced answer generation with document-aware retrieval"""
#         start_time = time.time()
#
#         try:
#             collection_name = f"{subscription_id}".replace(" ", "_")
#
#             # Collection diagnostics
#             try:
#                 stats = self.client.count(collection_name=collection_name, exact=False)
#                 total_points = getattr(stats, "count", 0)
#                 logger.info(f"Collection '{collection_name}' point count: {total_points}")
#             except Exception as diag_exc:
#                 logger.warning(f"Could not count collection '{collection_name}': {diag_exc}")
#
#             # Handle special cases
#             if self.greeting_handler.is_positive_feedback(query):
#                 return self._build_response("You're welcome! I'm glad I could help.", "positive_feedback")
#
#             if self.greeting_handler.is_greeting(query):
#                 greeting = f"Hello! I'm your doxa AI assistant. I can help you find specific information from your documents. What would you like to know?"
#                 self.conversation_history.add_turn(user_id, query, greeting)
#                 return self._build_response(greeting, "greeting")
#
#             if self.greeting_handler.is_farewell(query):
#                 farewell = f"Goodbye! Feel free to return whenever you need information from your documents. Have a great day!"
#                 self.conversation_history.clear_history(user_id)
#                 return self._build_response(farewell, "farewell")
#
#             logger.info(f"Processing query for collection '{collection_name}': {query[:100]}")
#
#             # NEW: Analyze query to identify target documents
#             from smart_prompt_builder import DocumentMatcher
#
#             # First, do a quick scan to see what sources are available
#             sample_scroll = self.client.scroll(
#                 collection_name=collection_name,
#                 limit=20,
#                 with_payload=True,
#                 with_vectors=False
#             )
#
#             if hasattr(sample_scroll, 'points'):
#                 sample_points = sample_scroll.points or []
#             elif isinstance(sample_scroll, tuple):
#                 sample_points = sample_scroll[0] if len(sample_scroll) > 0 else []
#             else:
#                 sample_points = []
#
#             available_sources = list(set([
#                 pt.payload.get('source_file')
#                 for pt in sample_points
#                 if pt.payload and pt.payload.get('source_file')
#             ]))
#
#             logger.info(f"Available sources in collection: {available_sources}")
#
#             # Suggest document filter based on query
#             suggested_docs = DocumentMatcher.suggest_document_filter(query, available_sources)
#             logger.info(f"Suggested document filter: {suggested_docs}")
#
#             # NEW: Use adaptive retrieval with document filtering
#             from enhanced_retrieval import AdaptiveRetriever
#
#             adaptive_retriever = AdaptiveRetriever(self.client, get_model())
#
#             retrieved_chunks = adaptive_retriever.retrieve_adaptive(
#                 collection_name=collection_name,
#                 query=query,
#                 profile_id=profile_id,
#                 top_k=top_k_retrieval,
#                 source_files=suggested_docs,  # Filter by relevant documents
#                 use_expansion=True,
#                 use_keyword_boost=True
#             )
#
#             if not retrieved_chunks:
#                 no_results = f"I searched through your documents but couldn't find relevant information for: '{query}'. Try rephrasing or asking about a different topic."
#                 self.conversation_history.add_turn(user_id, query, no_results)
#                 return self._build_response(no_results, "no_results")
#
#             # Rerank
#             reranked_chunks = self.reranker.rerank(
#                 chunks=[RetrievedChunk(
#                     id=c['id'],
#                     text=c['text'],
#                     score=c['score'],
#                     source=c['metadata'].get('source_file'),
#                     metadata=c['metadata']
#                 ) for c in retrieved_chunks],
#                 query=query,
#                 top_k=top_k_rerank,
#                 use_cross_encoder=True
#             )
#
#             # Convert back
#             final_chunks = [
#                 {
#                     'id': c.id,
#                     'text': c.text,
#                     'score': c.score,
#                     'metadata': c.metadata,
#                     'methods': next((rc.get('methods', ['dense']) for rc in retrieved_chunks if rc['id'] == c.id),
#                                     ['dense'])
#                 }
#                 for c in reranked_chunks[:final_k]
#             ]
#
#             # NEW: Use smart prompt builder
#             from smart_prompt_builder import build_enhanced_answer_with_verification
#
#             conversation_context = self.conversation_history.get_context(user_id, 2)
#
#             result = build_enhanced_answer_with_verification(
#                 query=query,
#                 chunks=final_chunks,
#                 llm_client=self.llm_client,
#                 persona=persona,
#                 conversation_context=conversation_context
#             )
#
#             # Update history
#             self.conversation_history.add_turn(user_id, query, result['answer'])
#
#             return {
#                 "response": result['answer'],
#                 "sources": result['sources'],
#                 "user_id": user_id,
#                 "collection": collection_name,
#                 "context_found": True,
#                 "query_type": "document_qa",
#                 "grounded": result['verified'],
#                 "verified": result['verified'],
#                 "query_intent": result.get('query_intent', {}),
#                 "retrieved_sources": result.get('retrieved_sources', []),
#                 "document_match_warning": result.get('document_match_warning', False),
#                 "retrieval_methods": list(set(sum([c.get('methods', []) for c in final_chunks], []))),
#                 "processing_time": round(time.time() - start_time, 2),
#                 "retrieval_stats": {
#                     "initial_retrieved": len(retrieved_chunks),
#                     "after_rerank": len(reranked_chunks),
#                     "final_context": len(final_chunks)
#                 }
#             }
#
#         except Exception as e:
#             logger.error(f"Error in answer_question: {e}", exc_info=True)
#             return self._build_error_response(str(e), user_id,
#                                               collection_name if 'collection_name' in locals() else subscription_id,
#                                               start_time)
#
#     def _build_response(self, text: str, query_type: str) -> Dict[str, Any]:
#         """Helper to build standard response"""
#         return {
#             "response": text,
#             "sources": [],
#             "context_found": True,
#             "query_type": query_type,
#             "grounded": True,
#             "verified": True
#         }
#
#     def _build_error_response(self, error: str, user_id: str, collection: str, start_time: float) -> Dict[str, Any]:
#         """Helper to build error response"""
#         return {
#             "response": "I apologize, but I encountered an error. Please try again or contact support.",
#             "sources": [],
#             "user_id": user_id,
#             "collection": collection,
#             "context_found": False,
#             "query_type": "error",
#             "error": error,
#             "grounded": False,
#             "verified": False,
#             "processing_time": time.time() - start_time
#         }
#
#
# # Global RAG system instance (lazy initialization)
# _RAG_SYSTEM = None
# _RAG_MODEL = None
#
#
# def create_llm_client(model_name: Optional[str] = None):
#     """Factory to select LLM backend based on requested model name."""
#     name = (model_name or "").lower()
#     if name.startswith("gemini"):
#         return GeminiClient(model_name)
#     # default to Ollama
#     return OllamaClient(model_name)
#
#
# def get_rag_system(model_name: Optional[str] = None) -> EnterpriseRAGSystem:
#     """Get or create the RAG system instance (singleton with lazy loading)."""
#     global _RAG_SYSTEM, _RAG_MODEL
#     if _RAG_SYSTEM is None or (model_name and model_name != _RAG_MODEL):
#         try:
#             _RAG_SYSTEM = EnterpriseRAGSystem(model_name=model_name)
#             _RAG_MODEL = model_name
#             logger.info("RAG system initialized")
#         except Exception as e:
#             logger.error(f"Failed to initialize RAG system: {e}")
#             raise
#     return _RAG_SYSTEM
#
#
# def answer_question(
#         query: str,
#         user_id: str,
#         profile_id: str,
#         subscription_id: str = "default",
#         model_name: str = "llama3.2",
#         persona: str = "professional document analysis assistant"
# ) -> Dict[str, Any]:
#     """
#     Main entry point for answering questions with enhanced NLU.
#
#     Args:
#         query: User's question
#         user_id: User identifier
#         profile_id: Department/collection name ('finance', 'banking', 'technical')
#         model_name: LLM model name (for compatibility)
#         persona: Assistant persona
#
#     Returns:
#         Response dictionary with enhanced metadata
#     """
#     rag_system = get_rag_system(model_name)
#     return rag_system.answer_question(
#         query=query,
#         profile_id=profile_id,
#         subscription_id=subscription_id,
#         user_id=user_id,
#         persona=persona
#     )
#
#
# def debug_collection(profile_id: str, subscription_id: str = "default") -> Dict[str, Any]:
#     """
#     Debug utility to check collection status with defensive error handling.
#
#     Args:
#         profile_id: Profile/department identifier
#         subscription_id: Tenant/subscription identifier
#
#     Returns:
#         Collection statistics
#     """
#     try:
#         collection_name = f"{subscription_id}".replace(" ", "_")
#         qdrant_client = get_qdrant_client()
#         collection_info = qdrant_client.get_collection(collection_name)
#
#         scroll_result = qdrant_client.scroll(
#             collection_name=collection_name,
#             limit=3,
#             with_payload=True,
#             with_vectors=False
#         )
#
#         sample_points = []
#         if isinstance(scroll_result, tuple) and len(scroll_result) > 0:
#             points_list = scroll_result[0]
#             if points_list:
#                 sample_points = [
#                     {
#                         "id": str(p.id),
#                         "text_preview": p.payload.get('text', '')[:200] if p.payload else 'No text',
#                         "source": p.payload.get('source', 'unknown') if p.payload else 'unknown'
#                     }
#                     for p in points_list
#                 ]
#
#         return {
#             "collection_name": collection_name,
#             "points_count": collection_info.points_count,
#             "vector_size": collection_info.config.params.vectors.size,
#             "distance": str(collection_info.config.params.vectors.distance),
#             "sample_points": sample_points,
#             "status": "healthy"
#         }
#     except Exception as e:
#         logger.error(
#             f"Error debugging collection '{collection_name if 'collection_name' in locals() else profile_id}': {e}",
#             exc_info=True)
#         return {
#             "collection_name": collection_name if 'collection_name' in locals() else profile_id,
#             "error": str(e),
#             "status": "error"
#         }
#
#
# def add_domain_terms(terms: List[str]):
#     """
#     Add domain-specific terms to spell checker whitelist.
#
#     Args:
#         terms: List of domain-specific terms to whitelist
#     """
#     try:
#         rag_system = get_rag_system()
#         rag_system.spell_corrector.add_domain_terms(terms)
#         logger.info(f"Added {len(terms)} domain terms to spell checker")
#     except Exception as e:
#         logger.error(f"Error adding domain terms: {e}")
#
#
# def clear_conversation_history(user_id: str):
#     """
#     Clear conversation history for a specific user.
#
#     Args:
#         user_id: User identifier
#     """
#     try:
#         rag_system = get_rag_system()
#         rag_system.conversation_history.clear_history(user_id)
#         logger.info(f"Cleared conversation history for user {user_id}")
#     except Exception as e:
#         logger.error(f"Error clearing conversation history: {e}")
#
#
# class RAGEvaluator:
#     """Evaluation utilities for monitoring RAG performance."""
#
#     @staticmethod
#     def evaluate_retrieval(
#             queries: List[str],
#             profile_id: str,
#             ground_truth_docs: List[List[str]]
#     ) -> Dict[str, float]:
#         """
#         Evaluate retrieval quality using MRR and Precision@K.
#
#         Args:
#             queries: List of test queries
#             profile_id: Collection name
#             ground_truth_docs: List of lists of relevant document IDs for each query
#
#         Returns:
#             Dictionary of evaluation metrics
#         """
#         if len(queries) != len(ground_truth_docs):
#             raise ValueError("Queries and ground truth must have same length")
#
#         rag_system = get_rag_system()
#         mrr_scores = []
#         p_at_1 = []
#         p_at_3 = []
#         p_at_5 = []
#
#         for query, relevant_docs in zip(queries, ground_truth_docs):
#             try:
#                 chunks = rag_system.retriever.retrieve(
#                     collection_name=profile_id,
#                     query=query,
#                     top_k=10
#                 )
#
#                 retrieved_ids = [chunk.id for chunk in chunks]
#
#                 for rank, doc_id in enumerate(retrieved_ids, 1):
#                     if doc_id in relevant_docs:
#                         mrr_scores.append(1.0 / rank)
#                         break
#                 else:
#                     mrr_scores.append(0.0)
#
#                 p_at_1.append(1.0 if retrieved_ids[:1] and retrieved_ids[0] in relevant_docs else 0.0)
#
#                 if len(retrieved_ids) >= 3:
#                     p_at_3.append(
#                         len(set(retrieved_ids[:3]) & set(relevant_docs)) / 3.0
#                     )
#
#                 if len(retrieved_ids) >= 5:
#                     p_at_5.append(
#                         len(set(retrieved_ids[:5]) & set(relevant_docs)) / 5.0
#                     )
#
#             except Exception as e:
#                 logger.error(f"Error evaluating query '{query}': {e}")
#
#         return {
#             "mrr": np.mean(mrr_scores) if mrr_scores else 0.0,
#             "precision_at_1": np.mean(p_at_1) if p_at_1 else 0.0,
#             "precision_at_3": np.mean(p_at_3) if p_at_3 else 0.0,
#             "precision_at_5": np.mean(p_at_5) if p_at_5 else 0.0,
#             "num_queries": len(queries)
#         }
#
#
# __all__ = [
#     'answer_question',
#     'debug_collection',
#     'add_domain_terms',
#     'clear_conversation_history',
#     'RAGEvaluator',
#     'get_rag_system'
# ]




# =======================================================
import re
from qdrant_client.http.exceptions import UnexpectedResponse
import json
import time
import logging
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque, Counter
import faiss
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama
import os
import redis
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


# Download required NLTK data only when missing to avoid repeated network calls
def ensure_nltk_data():
    required = {
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "stopwords": "corpora/stopwords",
        "punkt": "tokenizers/punkt",
    }
    for package, resource in required.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception as e:  # pragma: no cover - defensive logging
                logger.warning(f"Failed to download NLTK package {package}: {e}")


ensure_nltk_data()

# Initialize models and clients (lazy loading to avoid startup errors)
_MODEL = None
_CROSS_ENCODER = None
_SPELL_CHECKER = None
_QDRANT_CLIENT = None
_REDIS_CLIENT = None
_MODEL_CACHE: Dict[int, SentenceTransformer] = {}


def _load_model_candidates(required_dim: Optional[int] = None) -> SentenceTransformer:
    candidates = [getattr(Config.Model, "SENTENCE_TRANSFORMERS", "sentence-transformers/all-mpnet-base-v2")]
    last_error = None
    for name in candidates:
        try:
            logger.info(f"Loading sentence transformer model: {name}")
            model = SentenceTransformer(name)
            dim = model.get_sentence_embedding_dimension()
            logger.info(f"Loaded model '{name}' with dim={dim}")
            if required_dim is None or dim == required_dim:
                return model
            # cache for later but continue if dim mismatch
            _MODEL_CACHE[dim] = model
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to load model '{name}': {e}")
    raise RuntimeError(f"Could not load any sentence transformer model from {candidates}: {last_error}")


def get_model(required_dim: Optional[int] = None):
    """Lazy load sentence transformer model."""
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model_candidates()

    if required_dim is None:
        return _MODEL

    dim = _MODEL.get_sentence_embedding_dimension()
    if dim != required_dim:
        raise ValueError(f"Loaded model dim {dim} does not match required {required_dim}; using single model only.")
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


def get_redis_client():
    """Lazy init for Redis client."""
    global _REDIS_CLIENT
    if _REDIS_CLIENT is None:
        try:
            _REDIS_CLIENT = redis.Redis(
                host=Config.Redis.HOST,
                port=Config.Redis.PORT,
                username=Config.Redis.USERNAME or None,
                password=Config.Redis.PASSWORD or None,
                db=Config.Redis.DB,
                decode_responses=True,
                socket_timeout=5,
                ssl=getattr(Config.Redis, "SSL", False),
            )
            # simple ping
            _REDIS_CLIENT.ping()
            logger.info(
                "Initialized Redis client at %s:%s (ssl=%s)",
                Config.Redis.HOST,
                Config.Redis.PORT,
                getattr(Config.Redis, "SSL", False)
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Redis client, caching disabled: {e}")
            _REDIS_CLIENT = None
    return _REDIS_CLIENT


def configure_gemini():
    """Configure Gemini API with proper error handling."""
    try:
        api_key = getattr(Config.Model, "GEMINI_API_KEY", None) or getattr(Config.Gemini, "GEMINI_API_KEY", None)
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


@dataclass
class ProfileContextSnapshot:
    """Lightweight snapshot of profile-specific vocabulary and hints."""
    top_keywords: List[str]
    document_hints: List[str]
    total_chunks: int
    last_updated: float


class ChatFeedbackMemory:
    """Stores compact Q/A feedback to steer future responses."""

    def __init__(self, max_items: int = 12):
        self.max_items = max_items
        self.memories: Dict[str, deque] = {}

    def add_feedback(self, user_id: str, query: str, answer: str, sources: List[Dict[str, Any]]):
        if user_id not in self.memories:
            self.memories[user_id] = deque(maxlen=self.max_items)
        src_names = [s.get("source_name") for s in (sources or []) if s.get("source_name")]
        self.memories[user_id].append({
            "q": query.strip(),
            "a": answer.strip(),
            "sources": src_names,
            "ts": time.time()
        })

    def build_feedback_context(self, user_id: str, limit: int = 5) -> str:
        if user_id not in self.memories or not self.memories[user_id]:
            return ""
        recent = list(self.memories[user_id])[-limit:]
        lines = []
        for idx, item in enumerate(recent, 1):
            source_hint = f" | sources: {', '.join(item['sources'][:3])}" if item.get("sources") else ""
            lines.append(f"{idx}) Q: {item['q']} | A: {item['a'][:180]}{source_hint}")
        return "RECENT CHAT FEEDBACK (reuse tone/precision):\n" + "\n".join(lines) + "\n"


class ConversationHistory:
    """Manages conversation history with a sliding window."""

    def __init__(self, max_turns: int = 3):
        self.max_turns = max_turns
        self.histories: Dict[str, deque] = {}
        self.recent_docs: Dict[str, deque] = {}

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

    def add_sources(self, user_id: str, doc_ids: List[str]):
        """Track recently used document IDs for recency-based boosting."""
        if user_id not in self.recent_docs:
            self.recent_docs[user_id] = deque(maxlen=10)
        for doc_id in doc_ids:
            if doc_id:
                self.recent_docs[user_id].append(doc_id)

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

    def get_recent_doc_ids(self, user_id: str) -> List[str]:
        """Return a list of recently cited document IDs for this user."""
        if user_id not in self.recent_docs:
            return []
        return list(self.recent_docs[user_id])

    def clear_history(self, user_id: str):
        """Clear conversation history for a user."""
        if user_id in self.histories:
            self.histories[user_id].clear()
        if user_id in self.recent_docs:
            self.recent_docs[user_id].clear()


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
        'welcome', 'hola', 'bonjour', 'namaste', 'aloha', 'hii', 'Hii'}

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


class OllamaClient:
    """Handles local Ollama model calls with structured output and retries."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.2")
        if not self.model_name:
            raise ValueError("OLLAMA_MODEL environment variable is not set")
        logger.info(f"Initialized OllamaClient with model: {self.model_name}")

    def warm_up(self):
        """Eagerly load the model so first user queries do not pay startup cost."""
        try:
            ollama.generate(model=self.model_name, prompt="ping", options={"num_predict": 1})
            logger.info("Ollama warm-up successful")
        except Exception as e:
            logger.warning(f"Ollama warm-up failed (continuing without warm cache): {e}")

    def generate(
            self,
            prompt: str,
            max_retries: int = 3,
            backoff: float = 1.0
    ) -> str:
        """Generate response with retry logic and robust parsing."""
        for attempt in range(1, max_retries + 1):
            try:
                response = ollama.generate(model=self.model_name, prompt=prompt)
                text = (
                        response.get("response")
                        or (response.get("message", {}) or {}).get("content")
                        or ""
                ).strip()
                if text:
                    return text
                logger.warning(f"No text in Ollama response: {response}")
                return "I apologize, but I couldn't generate a proper response."
            except Exception as e:
                logger.warning(f"Ollama attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
                else:
                    logger.error(f"All retry attempts failed: {e}")
                    raise

        return "I apologize, but I encountered an error generating a response."


class GeminiClient:
    """Handles Gemini API calls with structured output and retries."""

    def __init__(self, model_name: Optional[str] = None):
        configure_gemini()
        self.model_name = model_name or Config.Model.GEMINI_MODEL_NAME
        if not self.model_name:
            raise ValueError("Gemini model name is not configured")
        self.model = genai.GenerativeModel(self.model_name)
        self.generation_config = genai.GenerationConfig(
            temperature=0.0,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
        )
        logger.info(f"Initialized GeminiClient with model: {self.model_name}")

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

    def __init__(self, llm_client):
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
        self.profile_context_cache: Dict[Tuple[str, str], ProfileContextSnapshot] = {}
        self.collection_dims: Dict[str, int] = {}

    def run_search(
            self,
            collection_name: str,
            query_vector: List[float],
            query_filter: Optional[dict] = None,
            limit: int = 50,
            vector_name: str = "content_vector",
            score_threshold: Optional[float] = None
    ):
        """Execute a vector search against Qdrant using query_points."""
        try:
            kwargs = dict(
                collection_name=collection_name,
                query=query_vector,
                using=vector_name,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )
            if score_threshold is not None:
                kwargs["score_threshold"] = score_threshold

            results = self.client.query_points(**kwargs)
            return results
        except Exception as e:
            logger.error("Qdrant query_points error: %s", e, exc_info=True)
            return None

    def get_collection_vector_dim(self, collection_name: str) -> Optional[int]:
        """Fetch and cache the expected vector dimension for a collection."""
        if collection_name in self.collection_dims:
            return self.collection_dims[collection_name]
        try:
            info = self.client.get_collection(collection_name)
            cfg = getattr(info, "config", None) or {}
            params = getattr(cfg, "params", None) or {}
            vectors = getattr(params, "vectors", None) or getattr(params, "vector_size", None) or {}
            dim = None
            if hasattr(vectors, "size"):
                dim = vectors.size
            elif isinstance(vectors, dict):
                if "size" in vectors:
                    dim = vectors.get("size")
                elif "content_vector" in vectors and isinstance(vectors["content_vector"], dict):
                    dim = vectors["content_vector"].get("size")
            if dim is None:
                dim = 768  # default to mpnet dimension
            self.collection_dims[collection_name] = dim
            logger.info(f"Collection '{collection_name}' expects dim={dim}")
            return dim
        except Exception as e:
            logger.warning(f"Could not fetch vector dim for collection '{collection_name}': {e}")
            return None

    def retrieve(
            self,
            collection_name: str,
            query: str,
            filter_profile: str = None,
            top_k: int = 50,
            score_threshold: float = None
    ) -> List[RetrievedChunk]:
        """Retrieve relevant chunks using Qdrant's native search."""
        try:
            target_dim = self.get_collection_vector_dim(collection_name)
            model = get_model(required_dim=target_dim)
            q_dim = getattr(model, "get_sentence_embedding_dimension", lambda: None)()
            if target_dim and q_dim and target_dim != q_dim:
                logger.warning(
                    f"Embedding dim {q_dim} does not match collection dim {target_dim}; using model regardless")
            query_vector = model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32).tolist()
        except Exception as err:
            logger.error(f"Failed to embed query for retrieval: {err}", exc_info=True)
            return []

        logger.info(f"Searching collection '{collection_name}' for query: {query[:100]}")

        profile_filter_val = str(filter_profile).strip() if filter_profile is not None else None
        filter_clauses = []
        if profile_filter_val:
            filter_clauses.append({"key": "profile_id", "match": {"value": profile_filter_val}})

        query_filter = {"must": filter_clauses} if filter_clauses else None

        results = self.run_search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            vector_name="content_vector",
            score_threshold=score_threshold
        )

        if not results or not getattr(results, "points", []):
            logger.warning(f"No results found in collection '{collection_name}'")
            return []

        points = results.points or []
        logger.info("Qdrant returned %d hits", len(points))
        logger.info("Top scores: %s", [p.score for p in points[:3]])

        chunks: List[RetrievedChunk] = []
        for pt in points:
            payload = pt.payload or {}
            text = payload.get("text", "")
            snippet = text[:120].replace("\n", " ")
            logger.debug("Hit score=%.4f snippet=%s", pt.score, snippet)
            chunk = RetrievedChunk(
                id=str(pt.id),
                text=text,
                score=float(pt.score),
                source=payload.get("source_file", "unknown"),
                metadata=payload
            )
            chunks.append(chunk)

        return chunks

    def get_profile_context(
            self,
            collection_name: str,
            profile_id: str,
            max_points: int = 400,
            refresh_seconds: int = 300
    ) -> ProfileContextSnapshot:
        """Build lightweight context from existing embeddings to guide vague queries."""
        cache_key = (collection_name, str(profile_id))
        now = time.time()
        cached = self.profile_context_cache.get(cache_key)
        if cached and (now - cached.last_updated) < refresh_seconds:
            return cached

        filter_ = {"must": [{"key": "profile_id", "match": {"value": str(profile_id)}}]}
        collected_points = []
        next_offset = None
        batch_size = min(120, max_points)

        try:
            while len(collected_points) < max_points:
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_,
                    offset=next_offset,
                    limit=min(batch_size, max_points - len(collected_points)),
                    with_payload=True,
                    with_vectors=False
                )

                if hasattr(scroll_result, "points"):
                    batch = scroll_result.points or []
                    next_offset = getattr(scroll_result, "next_page_offset", None)
                elif isinstance(scroll_result, tuple):
                    batch = scroll_result[0] if len(scroll_result) > 0 else []
                    next_offset = scroll_result[1] if len(scroll_result) > 1 else None
                else:
                    batch = []
                    next_offset = None

                if not batch:
                    break

                collected_points.extend(batch)

                if not next_offset:
                    break
        except Exception as e:
            logger.warning(f"Failed to build profile context for {profile_id}: {e}")
            snapshot = ProfileContextSnapshot([], [], 0, now)
            self.profile_context_cache[cache_key] = snapshot
            return snapshot

        token_counts: Counter = Counter()
        doc_hints: List[str] = []
        seen_hints = set()

        for pt in collected_points:
            payload = pt.payload or {}
            text = payload.get("text") or ""
            if text:
                token_counts.update(self.preprocessor.tokenize(text))

            for hint_key in ("source_file", "document_id", "section"):
                hint_val = payload.get(hint_key)
                if hint_val:
                    hint_val = str(hint_val)
                    if hint_val not in seen_hints:
                        doc_hints.append(hint_val)
                        seen_hints.add(hint_val)

        top_keywords = [w for w, _ in token_counts.most_common(40)]
        snapshot = ProfileContextSnapshot(
            top_keywords=top_keywords,
            document_hints=doc_hints[:12],
            total_chunks=len(collected_points),
            last_updated=now
        )
        self.profile_context_cache[cache_key] = snapshot
        return snapshot


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
    def build_source_hints(chunks: List[RetrievedChunk]) -> str:
        """Create a compact source map to bias model attention to top evidence."""
        if not chunks:
            return ""
        lines = []
        for i, chunk in enumerate(chunks[:5], 1):
            meta = chunk.metadata or {}
            source_name = chunk.source or meta.get('source_file', f"doc_{chunk.id[:8]}")
            page = meta.get('page')
            section = meta.get('section')
            score = round(float(chunk.score), 3)
            parts = [f"{i}) {source_name}", f"score={score}"]
            if page is not None:
                parts.append(f"page={page}")
            if section:
                parts.append(f"section={section}")
            lines.append(" | ".join(parts))
        return "SOURCE MAP:\n" + "\n".join(lines) + "\n"

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
        source_map = ContextBuilder.build_source_hints(selected_chunks)
        if source_map:
            context_parts.append(source_map)
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


class DomainPromptAdapter:
    """Creates lightweight, in-context adapters to steer local LLM responses."""

    @staticmethod
    def build_adapter(profile_context: Dict[str, Any], query: str) -> str:
        if not profile_context:
            return ""

        keywords = profile_context.get("keywords") or []
        hints = profile_context.get("hints") or []
        sampled = profile_context.get("sampled_chunks", 0)

        adapter_lines = []
        if keywords:
            adapter_lines.append(
                f"Domain keywords to preserve and prefer in wording: {', '.join(keywords[:12])}."
            )
        if hints:
            adapter_lines.append(f"Relevant documents/sections to anchor on: {', '.join(hints[:6])}.")
        if sampled:
            adapter_lines.append(f"Context built from {sampled} profile chunks; avoid generic responses.")

        adapter_lines.append(
            "Favor the terminology above when answering; align synonyms in the question to these domain terms."
        )
        adapter_lines.append(
            f"If the user query is vague ('{query}'), proactively ground the answer in the domain cues above."
        )
        return "\n".join(adapter_lines)


class AnswerabilityDetector:
    """Detects if a question can be answered from provided context."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def check_answerability(self, query: str, context: str, has_chunks: bool = False) -> Tuple[bool, str]:
        """
        Check if the query can be answered from the context.

        If we already have retrieved chunks, bias toward answering to avoid premature
        "cannot answer" responses when relevant evidence exists.
        """
        if has_chunks and context.strip():
            return True, "Context present from retrieved chunks"

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
    def build_qa_prompt(
            query: str,
            context: str,
            persona: str,
            conversation_summary: str = "",
            domain_guidance: str = "",
            feedback_memory: str = ""
    ) -> str:
        """Build a structured QA prompt with grounding and citation requirements."""
        convo_block = f"\nPRIOR CONVERSATION SUMMARY:\n{conversation_summary}\n" if conversation_summary else ""
        domain_block = f"\nDOMAIN ADAPTER:\n{domain_guidance}\n" if domain_guidance else ""
        feedback_block = f"\nRECENT FEEDBACK:\n{feedback_memory}\n" if feedback_memory else ""
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
{convo_block}
{domain_block}
{feedback_block}

USER QUESTION: {query}

RESPONSE REQUIREMENTS:
- Write in a natural, human tone with 2-5 concise sentences.
- Keep it conversational and approachable, as if explaining to a teammate.
- Do NOT repeat or paraphrase the user question; go straight to the answer.
- Lead with the direct answer; include brief reasoning or supporting evidence.
- Cite sources using [SOURCE-X] notation after each claim.
- If information is partial, say what is known and what is missing (without generic disclaimers).
- Avoid filler like "based on the available documents"; just state the supported facts.
- NEVER speculate or add external knowledge.

Provide your grounded answer now:"""

        return prompt


class ConversationSummarizer:
    """Summarizes the last few turns to keep context tight."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def summarize(self, conversation_text: str) -> str:
        if not conversation_text:
            return ""
        prompt = f"""Summarize the following conversation turns into 3-5 concise bullets capturing user intent and assistant answers. Do NOT invent details.

CONVERSATION:
{conversation_text}

SUMMARY:"""
        try:
            summary = self.llm_client.generate(prompt, max_retries=2, backoff=0.5)
            return summary.strip()
        except Exception as e:
            logger.warning(f"Conversation summarization failed: {e}")
            return ""


class EnterpriseRAGSystem:
    """
    Enhanced RAG system with query reformulation, spelling correction,
    conversational context, cross-encoder reranking, and grounding.
    """

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the RAG system with lazy-loaded components."""
        try:
            # Initialize LLM client (Ollama by default, Gemini when requested)
            self.llm_client = create_llm_client(model_name)

            # Initialize other components
            qdrant_client = get_qdrant_client()
            model = get_model()
            cross_encoder = get_cross_encoder()

            self.client = qdrant_client
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
            self.conversation_summarizer = ConversationSummarizer(self.llm_client)
            self.feedback_memory = ChatFeedbackMemory(max_items=12)
            self._warm_up_llm()

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
        """Preprocess query with spelling correction, reformulation, and light expansion."""
        metadata = {
            'original_query': query,
            'corrections': [],
            'reformulated': False,
            'expanded': False
        }

        processed_query = query

        # Spelling corrections (preserve names if no corrections)
        corrected_query, corrections = self.spell_corrector.correct_text(query)
        if corrections:
            metadata['corrections'] = corrections
            processed_query = corrected_query

        # Reformulate for vaguer prompts to tighten intent
        if use_reformulation and len(processed_query.split()) >= 4:
            conv_context = self.conversation_history.get_context(user_id, max_turns=2)
            reformulated = self.query_reformulator.reformulate(processed_query, conv_context)
            if reformulated and reformulated != processed_query:
                processed_query = reformulated
                metadata['reformulated'] = True

        # Light expansion to add semantically related terms
        expanded_query = self.query_expander.expand_query(processed_query)
        if expanded_query and expanded_query != processed_query:
            processed_query = expanded_query
            metadata['expanded'] = True
            metadata['expanded_query'] = expanded_query

        return processed_query, metadata

    @staticmethod
    def _is_retrieval_sufficient(chunks: List[RetrievedChunk], min_hits: int = 3, min_score: float = 0.18) -> bool:
        """Decide if current retrieval is good enough to stop trying fallbacks."""
        if not chunks:
            return False
        if len(chunks) >= min_hits:
            return True
        top_score = float(chunks[0].score)
        return top_score >= min_score

    @staticmethod
    def _is_query_vague(query: str) -> bool:
        """Heuristic to detect short or underspecified questions."""
        tokens = query.split()
        if len(tokens) <= 3:
            return True
        meaningful_tokens = [t for t in tokens if len(t) > 3]
        return len(meaningful_tokens) <= 2

    @staticmethod
    def _contextualize_query(query: str, profile_context: ProfileContextSnapshot) -> Tuple[str, Dict[str, Any]]:
        """Blend the user query with profile-specific hints to guide retrieval."""
        if not profile_context or not (profile_context.top_keywords or profile_context.document_hints):
            return query, {}

        keywords = profile_context.top_keywords[:8]
        hints = profile_context.document_hints[:3]
        extras = []
        if hints:
            extras.append("related to " + ", ".join(hints))
        if keywords:
            extras.append("keywords: " + ", ".join(keywords))

        contextual_query = " ; ".join([query] + extras)
        return contextual_query, {"profile_keywords_used": keywords, "profile_hints_used": hints}

    def retrieve_with_priorities(
            self,
            query: str,
            user_id: str,
            profile_id: str,
            collection_name: str,
            top_k_retrieval: int = 50
    ) -> Dict[str, Any]:
        """
        Try Qdrant retrieval first, fall back to relaxed thresholds, then chat-history
        reformulation as a last resort.
        """
        profile_context = self.retriever.get_profile_context(collection_name, profile_id)
        profile_context_data = {
            "keywords": profile_context.top_keywords[:12],
            "hints": profile_context.document_hints[:6],
            "sampled_chunks": profile_context.total_chunks
        }

        if profile_context.top_keywords:
            # Protect domain terms from being "corrected" away
            self.spell_corrector.add_domain_terms(profile_context.top_keywords[:50])

        is_vague = self._is_query_vague(query)
        primary_min_hits = 2 if is_vague else 3
        primary_min_score = 0.12 if is_vague else 0.18
        primary_threshold = 0.18 if is_vague else 0.25

        attempt_records = []
        retrieval_runs = []

        def run_attempt(label: str, query_text: str, threshold: Optional[float], use_history: bool,
                        metadata: Dict[str, Any]):
            chunks = self.retriever.retrieve(
                collection_name=collection_name,
                filter_profile=profile_id,
                query=query_text,
                top_k=top_k_retrieval,
                score_threshold=threshold
            )
            top_score = float(chunks[0].score) if chunks else 0.0
            record = {
                "label": label,
                "query": query_text,
                "score_threshold": threshold,
                "hits": len(chunks),
                "top_score": round(top_score, 4),
                "used_history": use_history
            }
            attempt_records.append(record)
            retrieval_runs.append((chunks, record, metadata, query_text))
            return chunks

        primary_query, primary_metadata = self.preprocess_query(query, user_id, use_reformulation=False)
        primary_metadata["vague_query"] = is_vague
        primary_metadata["profile_context"] = profile_context_data

        primary_chunks = run_attempt("direct_qdrant", primary_query, primary_threshold, False, primary_metadata)
        if self._is_retrieval_sufficient(primary_chunks, min_hits=primary_min_hits, min_score=primary_min_score):
            return {
                "chunks": primary_chunks,
                "query": primary_query,
                "metadata": primary_metadata,
                "attempts": attempt_records,
                "selected_strategy": "direct_qdrant",
                "profile_context": profile_context_data
            }

        contextual_query, contextual_meta = self._contextualize_query(primary_query, profile_context)
        if contextual_query != primary_query:
            contextual_metadata = {**primary_metadata, **contextual_meta, "contextualized": True}
            contextual_threshold = 0.15 if is_vague else 0.2
            contextual_chunks = run_attempt("contextual_qdrant", contextual_query, contextual_threshold, False,
                                            contextual_metadata)
            if self._is_retrieval_sufficient(contextual_chunks, min_hits=primary_min_hits,
                                             min_score=primary_min_score):
                return {
                    "chunks": contextual_chunks,
                    "query": contextual_query,
                    "metadata": contextual_metadata,
                    "attempts": attempt_records,
                    "selected_strategy": "contextual_qdrant",
                    "profile_context": profile_context_data
                }

        relaxed_chunks = run_attempt("relaxed_qdrant", primary_query, None, False, primary_metadata)
        if self._is_retrieval_sufficient(relaxed_chunks, min_hits=primary_min_hits, min_score=primary_min_score):
            return {
                "chunks": relaxed_chunks,
                "query": primary_query,
                "metadata": primary_metadata,
                "attempts": attempt_records,
                "selected_strategy": "relaxed_qdrant",
                "profile_context": profile_context_data
            }

        history_query, history_metadata = self.preprocess_query(query, user_id, use_reformulation=True)
        history_metadata["profile_context"] = profile_context_data
        if history_query != primary_query or history_metadata.get("reformulated") or history_metadata.get("expanded"):
            history_chunks = run_attempt("history_guided", history_query, None, True, history_metadata)
            if self._is_retrieval_sufficient(history_chunks, min_hits=primary_min_hits, min_score=primary_min_score):
                return {
                    "chunks": history_chunks,
                    "query": history_query,
                    "metadata": history_metadata,
                    "attempts": attempt_records,
                    "selected_strategy": "history_guided",
                    "profile_context": profile_context_data
                }

        best_run = max(retrieval_runs, key=lambda r: r[1]["top_score"], default=None)
        selected_chunks, selected_record, selected_meta, selected_query = best_run if best_run else ([], {},
                                                                                                     primary_metadata,
                                                                                                     primary_query)
        return {
            "chunks": selected_chunks,
            "query": selected_query,
            "metadata": selected_meta,
            "attempts": attempt_records,
            "selected_strategy": selected_record.get("label", "none"),
            "profile_context": profile_context_data
        }

    def _warm_up_llm(self):
        """Warm LLM backend so first user calls do not fail cold."""
        warm_fn = getattr(self.llm_client, "warm_up", None)
        if callable(warm_fn):
            try:
                warm_fn()
            except Exception as e:
                logger.warning(f"LLM warm-up skipped due to error: {e}")

    def answer_question(
            self,
            query: str,
            profile_id: str,
            subscription_id: str,
            user_id: str,
            persona: str = "professional document analysis assistant",
            top_k_retrieval: int = 50,
            top_k_rerank: int = 15,
            final_k: int = 7
    ) -> Dict[str, Any]:
        """Enhanced answer generation with document-aware retrieval"""
        start_time = time.time()

        try:
            collection_name = f"{subscription_id}".replace(" ", "_")

            # Collection diagnostics
            try:
                stats = self.client.count(collection_name=collection_name, exact=False)
                total_points = getattr(stats, "count", 0)
                logger.info(f"Collection '{collection_name}' point count: {total_points}")
            except Exception as diag_exc:
                logger.warning(f"Could not count collection '{collection_name}': {diag_exc}")

            # Handle special cases
            if self.greeting_handler.is_positive_feedback(query):
                return self._build_response("You're welcome! I'm glad I could help.", "positive_feedback")

            if self.greeting_handler.is_greeting(query):
                greeting = f"Hello! I'm your doxa AI assistant. I can help you find specific information from your documents. What would you like to know?"
                self.conversation_history.add_turn(user_id, query, greeting)
                return self._build_response(greeting, "greeting")

            if self.greeting_handler.is_farewell(query):
                farewell = f"Goodbye! Feel free to return whenever you need information from your documents. Have a great day!"
                self.conversation_history.clear_history(user_id)
                return self._build_response(farewell, "farewell")

            logger.info(f"Processing query for collection '{collection_name}': {query[:100]}")

            # NEW: Analyze query to identify target documents
            from smart_prompt_builder import DocumentMatcher

            # First, do a quick scan to see what sources are available
            sample_scroll = self.client.scroll(
                collection_name=collection_name,
                limit=20,
                with_payload=True,
                with_vectors=False
            )

            if hasattr(sample_scroll, 'points'):
                sample_points = sample_scroll.points or []
            elif isinstance(sample_scroll, tuple):
                sample_points = sample_scroll[0] if len(sample_scroll) > 0 else []
            else:
                sample_points = []

            available_sources = list(set([
                pt.payload.get('source_file')
                for pt in sample_points
                if pt.payload and pt.payload.get('source_file')
            ]))

            logger.info(f"Available sources in collection: {available_sources}")

            # Suggest document filter based on query
            suggested_docs = DocumentMatcher.suggest_document_filter(query, available_sources)
            logger.info(f"Suggested document filter: {suggested_docs}")

            # NEW: Use adaptive retrieval with document filtering
            from enhanced_retrieval import AdaptiveRetriever

            adaptive_retriever = AdaptiveRetriever(self.client, get_model())

            retrieved_chunks = adaptive_retriever.retrieve_adaptive(
                collection_name=collection_name,
                query=query,
                profile_id=profile_id,
                top_k=top_k_retrieval,
                source_files=suggested_docs,  # Filter by relevant documents
                use_expansion=True,
                use_keyword_boost=True
            )

            if not retrieved_chunks:
                no_results = f"I searched through your documents but couldn't find relevant information for: '{query}'. Try rephrasing or asking about a different topic."
                self.conversation_history.add_turn(user_id, query, no_results)
                return self._build_response(no_results, "no_results")

            # Rerank
            reranked_chunks = self.reranker.rerank(
                chunks=[RetrievedChunk(
                    id=c['id'],
                    text=c['text'],
                    score=c['score'],
                    source=c['metadata'].get('source_file'),
                    metadata=c['metadata']
                ) for c in retrieved_chunks],
                query=query,
                top_k=top_k_rerank,
                use_cross_encoder=True
            )

            # Convert back
            final_chunks = [
                {
                    'id': c.id,
                    'text': c.text,
                    'score': c.score,
                    'metadata': c.metadata,
                    'methods': next((rc.get('methods', ['dense']) for rc in retrieved_chunks if rc['id'] == c.id),
                                    ['dense'])
                }
                for c in reranked_chunks[:final_k]
            ]

            # NEW: Use smart prompt builder
            from smart_prompt_builder import build_enhanced_answer_with_verification

            conversation_context = self.conversation_history.get_context(user_id, 2)

            result = build_enhanced_answer_with_verification(
                query=query,
                chunks=final_chunks,
                llm_client=self.llm_client,
                persona=persona,
                conversation_context=conversation_context
            )

            # Update history
            self.conversation_history.add_turn(user_id, query, result['answer'])

            return {
                "response": result['answer'],
                "sources": result['sources'],
                "user_id": user_id,
                "collection": collection_name,
                "context_found": True,
                "query_type": "document_qa",
                "grounded": result['verified'],
                "verified": result['verified'],
                "query_intent": result.get('query_intent', {}),
                "retrieved_sources": result.get('retrieved_sources', []),
                "document_match_warning": result.get('document_match_warning', False),
                "retrieval_methods": list(set(sum([c.get('methods', []) for c in final_chunks], []))),
                "processing_time": round(time.time() - start_time, 2),
                "retrieval_stats": {
                    "initial_retrieved": len(retrieved_chunks),
                    "after_rerank": len(reranked_chunks),
                    "final_context": len(final_chunks)
                }
            }

        except Exception as e:
            logger.error(f"Error in answer_question: {e}", exc_info=True)
            return self._build_error_response(str(e), user_id,
                                              collection_name if 'collection_name' in locals() else subscription_id,
                                              start_time)

    def _build_response(self, text: str, query_type: str) -> Dict[str, Any]:
        """Helper to build standard response"""
        return {
            "response": text,
            "sources": [],
            "context_found": True,
            "query_type": query_type,
            "grounded": True,
            "verified": True
        }

    def _build_error_response(self, error: str, user_id: str, collection: str, start_time: float) -> Dict[str, Any]:
        """Helper to build error response"""
        return {
            "response": "I apologize, but I encountered an error. Please try again or contact support.",
            "sources": [],
            "user_id": user_id,
            "collection": collection,
            "context_found": False,
            "query_type": "error",
            "error": error,
            "grounded": False,
            "verified": False,
            "processing_time": time.time() - start_time
        }


# Global RAG system instance (lazy initialization)
_RAG_SYSTEM = None
_RAG_MODEL = None


def create_llm_client(model_name: Optional[str] = None):
    """Factory to select LLM backend based on requested model name."""
    name = (model_name or "").lower()
    if name.startswith("gemini"):
        return GeminiClient(model_name)
    # default to Ollama
    return OllamaClient(model_name)


def get_rag_system(model_name: Optional[str] = None) -> EnterpriseRAGSystem:
    """Get or create the RAG system instance (singleton with lazy loading)."""
    global _RAG_SYSTEM, _RAG_MODEL
    if _RAG_SYSTEM is None or (model_name and model_name != _RAG_MODEL):
        try:
            _RAG_SYSTEM = EnterpriseRAGSystem(model_name=model_name)
            _RAG_MODEL = model_name
            logger.info("RAG system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    return _RAG_SYSTEM


def answer_question(
        query: str,
        user_id: str,
        profile_id: str,
        subscription_id: str = "default",
        model_name: str = "llama3.2",
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
    rag_system = get_rag_system(model_name)
    return rag_system.answer_question(
        query=query,
        profile_id=profile_id,
        subscription_id=subscription_id,
        user_id=user_id,
        persona=persona
    )


def debug_collection(profile_id: str, subscription_id: str = "default") -> Dict[str, Any]:
    """
    Debug utility to check collection status with defensive error handling.

    Args:
        profile_id: Profile/department identifier
        subscription_id: Tenant/subscription identifier

    Returns:
        Collection statistics
    """
    try:
        collection_name = f"{subscription_id}".replace(" ", "_")
        qdrant_client = get_qdrant_client()
        collection_info = qdrant_client.get_collection(collection_name)

        scroll_result = qdrant_client.scroll(
            collection_name=collection_name,
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
            "collection_name": collection_name,
            "points_count": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "distance": str(collection_info.config.params.vectors.distance),
            "sample_points": sample_points,
            "status": "healthy"
        }
    except Exception as e:
        logger.error(
            f"Error debugging collection '{collection_name if 'collection_name' in locals() else profile_id}': {e}",
            exc_info=True)
        return {
            "collection_name": collection_name if 'collection_name' in locals() else profile_id,
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