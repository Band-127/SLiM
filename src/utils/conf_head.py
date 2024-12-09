import torch
from torch import nn
from einops.einops import rearrange
from src.backbone.vssm.vmamba import LayerNorm2d

class ConfHead(nn.Module):
    def __init__(self, d_model: int):
        """
        Confidence mask head, a 3-layers MLP followed by a sigmoid activation.

        Args:
            d_model (int): Dimension of the input feature.
        """
        super(ConfHead, self).__init__()
        self.head = nn.Sequential(
            LayerNorm2d(d_model),
            nn.Conv2d(
                in_channels=d_model,
                out_channels=int(d_model // 2),
                kernel_size=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=int(d_model // 2),
                out_channels=int(d_model // 2),
                kernel_size=3,
                padding=1,
            ),
            nn.Conv2d(
                in_channels=int(d_model // 2),
                out_channels=int(d_model // 2),
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(
                in_channels=int(d_model // 2),
                out_channels=1,
                kernel_size=1,
                padding=0,
            ),
            nn.Sigmoid(),
            nn.Flatten(1, -1),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # # x: [B, C, H, W]
        # B, C, H, W = x.shape
        # x = rearrange(x, "b c h w -> (b h w) c")

        # # [(B H W), C] -> [(B H W), 1]
        # x = self.head(x)

        # # [(B H W), 1] -> [B, (H W)]
        # x = rearrange(x, "(b h w) 1 -> b (h w)", b=B, h=H, w=W)

        return self.head(x)
