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
                out_features=int(d_model * (self.upsample_scale) ** 2),
                bias=True,
            ),
            nn.GELU(approximate="tanh"),
        )
        self.pixel_shuffle = nn.PixelShuffle(upsample_scale)

        # Initialize weights
        with torch.no_grad():
            for m in self.modules():
                # Use this initial weight for better performance when deal with torch official PixelShuffle.
                if isinstance(m, nn.Linear):
                    scale = int(m.weight.shape[0] / m.weight.shape[1])
                    """
                    This weight is like 
                    1 0 0 0 ...
                    1 0 0 0 ...
                    . . . . ...
                    0 1 0 0 ...                
                    0 1 0 0 ...
                    . . . . ...
                    0 0 1 0 ...
                    0 0 1 0 ...
                    . . . . ...
                    
                    When deal with torch official PixelShuffle, it will initially copy, or you can say
                    uniformly divides all channels into the whole shuffled window. (If we ignore the activation function)
                    Example:
                    1. feat = [[[0]], [[1]], [[2]], [[3]], [[4]]      # C H W = 5 1 1
                    2. after the MLP and pixel shuffle with scale 2.
                    3. feat_shuffle:
                        at H, W = 0, 0:
                            feat ≈ [0, 1, 2, 3] * 0.25
                        at H, W = 0, 1:
                            feat ≈ [0, 1, 2, 3] * 0.25
                        at H, W = 1, 0:
                            feat ≈ [0, 1, 2, 3] * 0.25
                        at H, W = 1, 1:
                            feat ≈ [0, 1, 2, 3] * 0.25
                        which are almost same as original feat, but with a 0.25 (1/4) scale, and a some difference cause by activation function.
                    """
                    m.weight = nn.Parameter(
                        rearrange(
                            (
                                (1 / scale)
                                * torch.eye(m.weight.shape[1], dtype=m.weight.dtype)
                            )
                            .unsqueeze(0)
                            .repeat([scale, 1, 1]),
                            "a b c -> (b a) c",
                        )
                    )
                    nn.init.constant_(m.bias, 0)

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
