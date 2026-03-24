# tests/test_v2_vision_graft.py
import pytest
import torch


class TestProjectionMLP:
    def test_projection_output_shape(self):
        from src.finetune.v2.projection import ProjectionMLP
        proj = ProjectionMLP(vision_dim=1152, text_dim=5120, hidden_dim=4096)
        x = torch.randn(1, 196, 1152)
        out = proj(x)
        assert out.shape == (1, 196, 5120)

    def test_projection_is_trainable(self):
        from src.finetune.v2.projection import ProjectionMLP
        proj = ProjectionMLP(vision_dim=1152, text_dim=5120)
        trainable = sum(p.numel() for p in proj.parameters() if p.requires_grad)
        assert trainable > 0
        assert trainable < 100_000_000

    def test_projection_gelu_activation(self):
        from src.finetune.v2.projection import ProjectionMLP
        proj = ProjectionMLP(vision_dim=1152, text_dim=5120)
        has_gelu = any("GELU" in str(m) for m in proj.modules())
        assert has_gelu


class TestVisionGraft:
    def test_graft_config_defaults(self):
        from src.finetune.v2.vision_graft import GraftConfig
        cfg = GraftConfig()
        assert cfg.vision_model == "google/siglip-so400m-patch14-384"
        assert cfg.text_model == "unsloth/Qwen3-14B-bnb-4bit"
        assert cfg.image_size == 384
        assert cfg.vision_dim == 1152
        assert cfg.text_dim == 5120

    def test_graft_config_custom(self):
        from src.finetune.v2.vision_graft import GraftConfig
        cfg = GraftConfig(vision_model="custom/model", text_dim=4096)
        assert cfg.vision_model == "custom/model"
        assert cfg.text_dim == 4096

    def test_num_patches(self):
        from src.finetune.v2.vision_graft import GraftConfig
        cfg = GraftConfig()
        assert cfg.patch_size == 14
        assert cfg.num_patches == (384 // 14) ** 2
