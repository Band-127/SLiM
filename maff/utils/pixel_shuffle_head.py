import torch
from torch import nn
from einops.einops import rearrange


class PixelShuffleHead(nn.Module):
    def __init__(self, d_model, upsample_scale):
        super().__init__()

        self.upsample_scale = upsample_scale
        self.expansion_head = nn.Sequential(
            nn.Linear(
                in_features=int(d_model),
                out_features=int(d_model * self.upsample_scale),
                bias=True,
            ),
            nn.GELU(approximate="tanh"),
            nn.Linear(
                in_features=int(d_model * self.upsample_scale),
                out_features=int(d_model * (self.upsample_scale)**2),
                bias=True,
            ),
            nn.GELU(approximate="tanh"),
        )
        self.pixel_shuffle = nn.PixelShuffle(upsample_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, C, H, W]

        Returns:
            torch.Tensor: [B, C, H*upsample_scale, W*upsample_scale]
        """
        B, C, H, W = x.shape
        # [B, C, H, W] -> [(B H W), C]
        x = rearrange(x, "b c h w -> (b h w) c")
        # [(B H W), C] -> [(B H W), C * upsample_scale^2]
        x = self.expansion_head(x)
        # [(B H W), C * upsample_scale^2] -> [B, C * upsample_scale^2, H, W]
        x = rearrange(
            x,
            "(b h w) c -> b c h w",
            b=B,
            h=H,
            w=W,
        )

        return self.pixel_shuffle(x)
