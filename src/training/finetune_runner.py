from src.utils.logging_utils import get_logger
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

logger = get_logger(__name__)

def _messages_to_text(messages):
    parts = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        parts.append(f"{role}: {msg.get('content','')}")
    parts.append("Assistant:")
    return "\n".join(parts)

def _build_training_args(output_dir: Path, steps: int, batch_size: int, learning_rate: float, eval_steps: int = 0):
    from transformers import TrainingArguments

    kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "learning_rate": learning_rate,
        "num_train_epochs": 1,
        "max_steps": steps,
        "logging_steps": 10,
        "save_strategy": "no",
    }
    sig = TrainingArguments.__init__.__code__.co_varnames
    if "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = "no" if eval_steps == 0 else "steps"
        if eval_steps > 0 and "eval_steps" in sig:
            kwargs["eval_steps"] = eval_steps
    elif "eval_strategy" in sig:
        kwargs["eval_strategy"] = "no" if eval_steps == 0 else "steps"
        if eval_steps > 0 and "eval_steps" in sig:
            kwargs["eval_steps"] = eval_steps
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in sig})

@dataclass
class FinetuneConfig:
    base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    learning_rate: float = 2e-4
    max_steps: int = 200
    batch_size: int = 1
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    min_pairs: int = 1

class FinetuneRunner:
    def __init__(self, config: FinetuneConfig):
        self.config = config

    def run(self, train_path: Path, run_dir: Path) -> Dict[str, Any]:
        if not train_path.exists():
            return {"status": "skipped", "reason": "dataset_missing"}
        pair_count = sum(1 for _ in train_path.open("r", encoding="utf-8"))
        if pair_count < self.config.min_pairs:
            return {"status": "skipped", "reason": f"min_pairs_not_met ({pair_count})"}

        dataset = load_dataset("json", data_files=str(train_path), split="train")
        dataset = dataset.map(
            lambda ex: {"text": _messages_to_text(ex["messages"])},
            remove_columns=[c for c in dataset.column_names if c != "messages"],
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

        load_kwargs = {"device_map": "auto"}
        if torch.cuda.is_available():
            try:
                import bitsandbytes  # noqa: F401
                load_kwargs.update({"load_in_4bit": True})
            except Exception:
                pass
        model = AutoModelForCausalLM.from_pretrained(self.config.base_model, **load_kwargs)

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        args = _build_training_args(
            output_dir=run_dir / "checkpoints",
            steps=self.config.max_steps,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            eval_steps=0,
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        from transformers import Trainer

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        logger.info("Starting finetune run with %d examples", len(tokenized))
        trainer.train()
        adapter_dir = run_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        manifest = {
            "status": "success",
            "pairs": pair_count,
            "adapter_dir": str(adapter_dir),
            "base_model": self.config.base_model,
            "max_steps": self.config.max_steps,
        }
        (run_dir / "finetune_manifest.json").write_text(json.dumps(manifest, indent=2))
        return {"status": "success", "adapter_dir": str(adapter_dir), "pairs": pair_count}

__all__ = ["FinetuneRunner", "FinetuneConfig"]
