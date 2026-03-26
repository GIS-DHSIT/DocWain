"""LayoutLMv3 wrapper for key information extraction (KIE) and form understanding.

Uses Microsoft LayoutLMv3 to classify document tokens as question, answer,
header, or other — then pairs adjacent question/answer spans into KVPairs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from PIL import Image

from src.visual_intelligence.datatypes import KVPair
from src.visual_intelligence.model_pool import get_model_pool

logger = logging.getLogger(__name__)

# Maps model prediction indices to semantic labels (FUNSD-style).
LABEL_MAP: Dict[int, str] = {
    0: "other",
    1: "question",
    2: "question",
    3: "answer",
    4: "answer",
    5: "header",
    6: "header",
}


class LayoutLMv3Extractor:
    """Extract key-value pairs from document images using LayoutLMv3."""

    _MODEL_KEY = "layoutlmv3"

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        image: Image.Image,
        words: List[str],
        boxes: List[Tuple[int, int, int, int]],
        page_number: int,
    ) -> List[KVPair]:
        """Run KIE on *image* and return extracted key-value pairs.

        Returns an empty list when the model is unavailable, no words are
        provided, or inference fails.
        """
        if not self._ensure_loaded():
            return []
        if not words or not boxes:
            return []

        try:
            result = self._run_inference(image, words, boxes)
            return self._pair_question_answer(result["token_labels"], page=page_number)
        except Exception:
            logger.exception("LayoutLMv3 extraction failed")
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Ensure model and processor are loaded from the pool."""
        pool = get_model_pool()
        if not pool.is_available(self._MODEL_KEY):
            return False

        if self._model is None:
            self._model = pool.load_model(self._MODEL_KEY)
        if self._processor is None:
            self._processor = pool.load_processor(self._MODEL_KEY)

        return self._model is not None and self._processor is not None

    def _run_inference(
        self,
        image: Image.Image,
        words: List[str],
        boxes: List[Tuple[int, int, int, int]],
    ) -> Dict:
        """Run LayoutLMv3 token classification and return word-level labels.

        Returns a dict with key ``"token_labels"`` containing a list of
        ``{"text", "label", "bbox"}`` dicts, one per input word.
        """
        # Normalize bounding boxes to 0-1000 range (LayoutLMv3 convention).
        img_w, img_h = image.size
        normalized_boxes = []
        for x0, y0, x1, y1 in boxes:
            normalized_boxes.append([
                int(x0 / img_w * 1000),
                int(y0 / img_h * 1000),
                int(x1 / img_w * 1000),
                int(y1 / img_h * 1000),
            ])

        # Tokenize with the processor.
        encoding = self._processor(
            image,
            words,
            boxes=normalized_boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        pool = get_model_pool()
        device = pool.get_device(self._MODEL_KEY)
        encoding = {k: v.to(device) for k, v in encoding.items()}

        # Forward pass.
        with torch.no_grad():
            outputs = self._model(**encoding)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

        # Ensure predictions is a list even for single-token outputs.
        if isinstance(predictions, int):
            predictions = [predictions]

        # Map sub-word predictions back to word-level labels using word_ids().
        word_ids = encoding.word_ids() if hasattr(encoding, "word_ids") else None
        if word_ids is None:
            # Fallback: try the BatchEncoding method.
            try:
                word_ids = encoding.word_ids(batch_index=0)
            except Exception:
                word_ids = list(range(len(words)))

        token_labels: List[Dict] = []
        seen_word_ids: set = set()

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id in seen_word_ids:
                continue
            if word_id >= len(words):
                continue
            seen_word_ids.add(word_id)

            pred_idx = predictions[idx] if idx < len(predictions) else 0
            label = LABEL_MAP.get(pred_idx, "other")

            token_labels.append({
                "text": words[word_id],
                "label": label,
                "bbox": tuple(boxes[word_id]),
            })

        return {"token_labels": token_labels}

    def _pair_question_answer(
        self,
        tokens: List[Dict],
        page: int,
    ) -> List[KVPair]:
        """Walk through token labels and pair question spans with answer spans.

        Accumulates consecutive question tokens, then consecutive answer tokens.
        Flushes a KVPair when a new question starts after an answer, or when an
        "other" token appears after an answer.
        """
        pairs: List[KVPair] = []
        question_parts: List[str] = []
        answer_parts: List[str] = []

        for token in tokens:
            label = token["label"]
            text = token["text"]

            if label == "question":
                if answer_parts:
                    # A new question after an answer — flush the pair.
                    pairs.append(KVPair(
                        key=" ".join(question_parts),
                        value=" ".join(answer_parts),
                        confidence=0.8,
                        page=page,
                        source="layoutlmv3",
                    ))
                    question_parts = []
                    answer_parts = []
                question_parts.append(text)

            elif label == "answer":
                answer_parts.append(text)

            else:
                # "other" or "header" — flush if we have a complete pair.
                if question_parts and answer_parts:
                    pairs.append(KVPair(
                        key=" ".join(question_parts),
                        value=" ".join(answer_parts),
                        confidence=0.8,
                        page=page,
                        source="layoutlmv3",
                    ))
                    question_parts = []
                    answer_parts = []
                elif not answer_parts:
                    # Discard orphan question tokens on "other".
                    question_parts = []

        # Flush any trailing pair.
        if question_parts and answer_parts:
            pairs.append(KVPair(
                key=" ".join(question_parts),
                value=" ".join(answer_parts),
                confidence=0.8,
                page=page,
                source="layoutlmv3",
            ))

        return pairs
