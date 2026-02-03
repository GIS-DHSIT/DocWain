import argparse
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from docwain_ft.config import CONFIG
from docwain_ft.utils import ensure_dir, read_jsonl, setup_logging


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_dataset(path: str, tokenizer: AutoTokenizer) -> Dataset:
    rows = read_jsonl(path)
    processed = []
    for row in rows:
        messages = row.get("messages")
        if not messages:
            raise ValueError("Each row must include messages.")
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            text = "\n".join([f\"{m['role'].upper()}: {m['content']}\" for m in messages])
        processed.append({"text": text})
    return Dataset.from_list(processed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DocWain LoRA/QLoRA.")
    parser.add_argument("--data", default="data/docwain_sft.jsonl")
    parser.add_argument("--out", default="out/")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA into base model after training.")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("train_lora")

    output_root = Path(args.out)
    lora_dir = ensure_dir(output_root / "lora")
    merged_dir = ensure_dir(output_root / "merged")

    use_qlora = CONFIG.use_qlora
    bnb_config = None
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    logger.info("Loading base model %s", CONFIG.hf_base_model)
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG.hf_base_model,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    tokenizer = AutoTokenizer.from_pretrained(CONFIG.hf_base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    dataset = load_dataset(args.data, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_root / "training"),
        per_device_train_batch_size=CONFIG.batch_size,
        gradient_accumulation_steps=CONFIG.grad_accum,
        num_train_epochs=CONFIG.num_epochs,
        learning_rate=CONFIG.lr,
        logging_steps=10,
        save_steps=200,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=CONFIG.max_seq_len,
        packing=False,
    )

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving LoRA adapter to %s", lora_dir)
    trainer.model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    if args.merge:
        logger.info("Merging adapter into base model")
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
