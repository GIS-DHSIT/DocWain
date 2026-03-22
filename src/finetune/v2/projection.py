import torch
import torch.nn as nn


class ProjectionMLP(nn.Module):
    def __init__(self, vision_dim: int = 1152, text_dim: int = 5120, hidden_dim: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
