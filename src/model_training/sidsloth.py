
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

@dataclass
class SlothConfig:
    """Configuration for fine-tuning"""
    base_model: str = "mistralai/Mistral-7B-v0.1"
    dataset_path: str = "sidsloth_dataset.jsonl"
    output_dir: str = "sidsloth_adapter"
    use_4bit: bool = True
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    save_steps: int = 100
    logging_steps: int = 10
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])

def train_sidsloth(config: SlothConfig):
    print(" Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=config.dataset_path)


    def tokenize(example):
        prompt = f"### Instruction:\n{example['instruction']}\n### Input:\n{example['input']}\n### Response:\n{example['output']}"
        return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized_dataset = dataset["train"].map(tokenize, remove_columns=dataset["train"].column_names)

    print(" Loading base model...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quant_config if config.use_4bit else None,
        device_map="auto",
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    print(" Setting up LoRA...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        report_to="none",
        fp16=True,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    print(" Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f" LoRA adapter saved to: {config.output_dir}")

if __name__ == "__main__":
    cfg = SlothConfig()
    train_sidsloth(cfg)
