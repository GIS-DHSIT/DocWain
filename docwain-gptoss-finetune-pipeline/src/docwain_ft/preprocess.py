import argparse
import logging
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from docwain_ft.config import CONFIG
from docwain_ft.utils import read_jsonl, setup_logging, write_jsonl


REQUIRED_MESSAGE_KEYS = {"role", "content"}


def validate_sample(sample: dict) -> None:
    if "messages" not in sample:
        raise ValueError("Sample missing 'messages'.")
    if not isinstance(sample["messages"], list):
        raise ValueError("'messages' must be a list.")
    for msg in sample["messages"]:
        if not REQUIRED_MESSAGE_KEYS.issubset(msg):
            raise ValueError("Each message must include role and content.")


def format_messages(tokenizer: AutoTokenizer, messages: list[dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    joined = []
    for msg in messages:
        joined.append(f"{msg['role'].upper()}: {msg['content']}")
    return "\n".join(joined)


def preprocess_dataset(data_path: str, out_path: str, max_len: int) -> None:
    setup_logging()
    logger = logging.getLogger("preprocess")
    raw = read_jsonl(data_path)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.hf_base_model, use_fast=True)
    processed: list[dict[str, Any]] = []

    for sample in raw:
        validate_sample(sample)
        text = format_messages(tokenizer, sample["messages"])
        tokens = tokenizer(text, truncation=True, max_length=max_len)
        processed.append({"text": text, "input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]})

    write_jsonl(out_path, processed)
    logger.info("Preprocessed %s samples.", len(processed))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess DocWain dataset for training.")
    parser.add_argument("--data", default="data/docwain_sft.jsonl")
    parser.add_argument("--out", default="data/processed/docwain_sft_processed.jsonl")
    parser.add_argument("--max-len", type=int, default=CONFIG.max_seq_len)
    args = parser.parse_args()

    preprocess_dataset(args.data, args.out, args.max_len)


if __name__ == "__main__":
    main()
