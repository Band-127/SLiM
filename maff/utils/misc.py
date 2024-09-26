import torch
from torch import nn


class Unsqueeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(self.dim)
