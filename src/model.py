import torch
from torch import nn


class DQModel(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
