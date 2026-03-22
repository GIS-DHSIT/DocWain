from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch


@dataclass
class GraftConfig:
    vision_model: str = "google/siglip-so400m-patch14-384"
    text_model: str = "unsloth/Qwen3-14B-bnb-4bit"
    image_size: int = 384
    patch_size: int = 14
    vision_dim: int = 1152
    text_dim: int = 5120
    hidden_dim: int = 4096
    max_image_tokens: int = 729
    freeze_vision: bool = True
    freeze_text: bool = True

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


class VisionGraftedModel:
    def __init__(self, config: GraftConfig, device: str = "auto"):
        self._config = config
        self._device = device
        self._vision_encoder = None
        self._projection = None
        self._text_model = None
        self._tokenizer = None
        self._image_processor = None

    def load_vision_encoder(self):
        from transformers import SiglipVisionModel, SiglipImageProcessor
        self._vision_encoder = SiglipVisionModel.from_pretrained(self._config.vision_model)
        self._image_processor = SiglipImageProcessor.from_pretrained(self._config.vision_model)
        if self._config.freeze_vision:
            for p in self._vision_encoder.parameters():
                p.requires_grad = False
        return self

    def load_projection(self, checkpoint: Optional[Path] = None):
        from .projection import ProjectionMLP
        self._projection = ProjectionMLP(
            vision_dim=self._config.vision_dim,
            text_dim=self._config.text_dim,
            hidden_dim=self._config.hidden_dim,
        )
        if checkpoint and checkpoint.exists():
            self._projection.load_state_dict(torch.load(checkpoint, weights_only=True))
        return self

    def load_text_model(self):
        from unsloth import FastLanguageModel
        self._text_model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self._config.text_model, max_seq_length=4096, dtype=None, load_in_4bit=True,
        )
        return self

    def add_lora(self, r: int = 16, lora_alpha: int = 16):
        from unsloth import FastLanguageModel
        self._text_model = FastLanguageModel.get_peft_model(
            self._text_model, r=r, lora_alpha=lora_alpha, lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none", use_gradient_checkpointing="unsloth",
        )
        return self

    def encode_image(self, images) -> torch.Tensor:
        inputs = self._image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self._vision_encoder.device) for k, v in inputs.items()}
        with torch.no_grad():
            vision_outputs = self._vision_encoder(**inputs)
        visual_tokens = vision_outputs.last_hidden_state
        return self._projection(visual_tokens)

    def save_projection(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._projection.state_dict(), path)

    def save_all(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.save_projection(output_dir / "projection.pt")
        if self._text_model:
            self._text_model.save_pretrained(str(output_dir / "lora_adapter"))
        if self._tokenizer:
            self._tokenizer.save_pretrained(str(output_dir / "lora_adapter"))
