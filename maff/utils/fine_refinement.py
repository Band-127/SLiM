import torch
from torch import nn
from typing import Tuple
from einops.einops import rearrange

from maff.mamba.MambaEncoder import QuadConcatMambaLayer


class FineRefinement(nn.Module):
    def __init__(
        self,
        in_output_dim: int,
        num_layers: int,
        inner_expansion: int,
        conv_dim: int,
        delta: int,
        using_mamba2: bool,
    ):
        super(FineRefinement, self).__init__()
        self.layers = nn.ModuleList(
            [
                QuadConcatMambaLayer(
                    in_output_dim=in_output_dim,
                    inner_expansion=inner_expansion,
                    conv_dim=conv_dim,
                    delta=delta,
                    using_mamba2=using_mamba2,
                )
                for _ in range(num_layers)
            ]
        )

        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x0 (torch.Tensor): (M, C, H, W)
            x1 (torch.Tensor): (M, C, H, W)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (M, C, H, W)
        """
        x0_shape = x0.shape[2:]  # H, W
        x1_shape = x1.shape[2:]  # H, W

        for layer in self.layers:
            x0_hw = rearrange(x0, "b c h w -> b (h w) c")
            x1_hw = rearrange(x1, "b c h w -> b (h w) c")
            x0_wh = rearrange(x0, "b c h w -> b (w h) c")
            x1_wh = rearrange(x1, "b c h w -> b (w h) c")

            x0_hw, x1_hw, x0_wh, x1_wh = layer(x0_hw, x1_hw, x0_wh, x1_wh)

            x0_hw = rearrange(
                x0_hw, "b (h w) c -> b c h w", h=x0_shape[0], w=x0_shape[1]
            )
            x1_hw = rearrange(
                x1_hw, "b (h w) c -> b c h w", h=x1_shape[0], w=x1_shape[1]
            )
            x0_wh = rearrange(
                x0_wh, "b (w h) c -> b c h w", h=x0_shape[0], w=x0_shape[1]
            )
            x1_wh = rearrange(
                x1_wh, "b (w h) c -> b c h w", h=x1_shape[0], w=x1_shape[1]
            )

            x0 = 0.5 * (x0_hw + x0_wh)
            x1 = 0.5 * (x1_hw + x1_wh)

        return x0, x1
