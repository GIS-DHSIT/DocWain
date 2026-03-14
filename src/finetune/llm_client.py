from src.utils.logging_utils import get_logger
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = get_logger(__name__)

class UnslothLLMClient:
    """Lightweight inference client for Unsloth fine-tuned models."""

    def __init__(self, model_path: str, max_new_tokens: int = 512, temperature: float = 0.2):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipeline = None
        self.model_name = Path(model_path).name

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return
        logger.info("Loading Unsloth model from %s", self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
        )

    def generate(self, prompt: str, max_retries: int = 1, **kwargs) -> str:
        self._ensure_pipeline()
        max_tokens = kwargs.get("max_tokens") or self.max_new_tokens
        outputs = self._pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=self.temperature,
            num_return_sequences=1,
        )
        text = outputs[0]["generated_text"]
        if text.startswith(prompt):
            text = text[len(prompt) :].strip()
        return text
