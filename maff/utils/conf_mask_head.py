import torch
from torch import nn
from einops.einops import rearrange


class ConfMaskHead(nn.Module):
    def __init__(self, d_model: int):
        """
        Confidence mask head, a 3-layers MLP followed by a sigmoid activation.

        Args:
            d_model (int): Dimension of the input feature.
        """
        super(ConfMaskHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=int(d_model**0.5), bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(in_features=int(d_model**0.5), out_features=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> (b h w) c")

        # [(B H W), C] -> [(B H W), 1]
        x = self.head(x)

        # [(B H W), 1] -> [B, (H W)]
        x = rearrange(x, "(b h w) 1 -> b (h w)", b=B, h=H, w=W)

        return x
