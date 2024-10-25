import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple
from einops.einops import rearrange
from typing import List
from maff.mamba.MambaEncoder import MambaLayer, QuadConcatMambaLayer


class QuadTreeFineEncoder(nn.Module):
    """
    Fine feature encoder(Multi-scale quadtree)
    """

    def __init__(
        self,
        in_output_dim: int,
        num_layers: int,
        inner_expansion: int,
        conv_dim: int,
        delta: int,
        using_mamba2: bool,
    ) -> None:
        super(QuadTreeFineEncoder, self).__init__()

        self.num_layers = num_layers
        self.layers_fw = nn.ModuleList(
            [
                MambaLayer(
                    in_output_dim=in_output_dim,
                    inner_expansion=inner_expansion,
                    conv_dim=conv_dim,
                    delta=delta,
                    using_mamba2=using_mamba2,
                )
                for _ in range(num_layers)
            ]
        )
        self.layers_bw = nn.ModuleList(
            [
                MambaLayer(
                    in_output_dim=in_output_dim,
                    inner_expansion=inner_expansion,
                    conv_dim=conv_dim,
                    delta=delta,
                    using_mamba2=using_mamba2,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(in_output_dim) for _ in range(num_layers)]
        )

        # Initialize weights
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )

    def forward(
        self,
        feat0_quadtrees: torch.Tensor,
        feat1_quadtrees: torch.Tensor,
        all_window_size: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Input: [M x L x C], [M x L x C]
        x = torch.concat([feat0_quadtrees, feat1_quadtrees], dim=-2)  # [M x 2L x C]

        # 2. Mamba
        for i in range(self.num_layers):
            # 1. Bidirectional scan
            xb = x.flip(dims=(-2,))
            # 2. Mamba
            delta_xf = self.layers_fw[i](x)
            delta_xb = self.layers_bw[i](xb)
            # 3. Fuse
            delta_x = 0.5 * (delta_xf + delta_xb.flip(dims=(-2,)))
            # 4. Layernorm
            delta_x = self.layer_norms[i](delta_x)
            # 5. Residual
            x = delta_x + x
        new_feat0_quadtrees = x[:, : feat0_quadtrees.shape[-2], :]  # [M x L x C]
        new_feat1_quadtrees = x[:, feat0_quadtrees.shape[-2] :, :]  # [M x L x C]

        # 3. Fuse all scales like fpn
        all_window_size_squared = [w**2 for w in all_window_size]
        # [M x SHW x C] -> S x [M x HW x C]
        new_feat0_quadtrees = list(
            torch.split(new_feat0_quadtrees, all_window_size_squared, dim=-2)
        )
        new_feat1_quadtrees = list(
            torch.split(new_feat1_quadtrees, all_window_size_squared, dim=-2)
        )
        # start from the coarse scale
        feat0 = rearrange(
            new_feat0_quadtrees[0],
            "m (h w) c -> m c h w",
            h=all_window_size[0],
            w=all_window_size[0],
        )  # [M x C x H x W]
        feat1 = rearrange(
            new_feat1_quadtrees[0],
            "m (h w) c -> m c h w",
            h=all_window_size[0],
            w=all_window_size[0],
        )  # [M x C x H x W]
        for idx in range(1, len(new_feat0_quadtrees)):
            feat0 = F.interpolate(feat0, scale_factor=2) + rearrange(
                new_feat0_quadtrees[idx],
                "m (h w) c -> m c h w",
                h=all_window_size[idx],
                w=all_window_size[idx],
            )
            feat1 = F.interpolate(feat1, scale_factor=2) + rearrange(
                new_feat1_quadtrees[idx],
                "m (h w) c -> m c h w",
                h=all_window_size[idx],
                w=all_window_size[idx],
            )
        return feat0, feat1


class FineEncoder(nn.Module):
    """
    Fine feature encoder(Single scale features)
    """

    def __init__(
        self,
        in_output_dim: int,
        num_layers: int,
        inner_expansion: int,
        conv_dim: int,
        delta: int,
        using_mamba2: bool,
    ) -> None:
        super(FineEncoder, self).__init__()

        self.num_layers = num_layers
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
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(in_output_dim) for _ in range(num_layers)]
        )

        # Initialize weights
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x0 (torch.Tensor): [M, C, H, W]
            x1 (torch.Tensor): [M, C, H, W]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: [M, C, H, W] x0, x1
        """
        x0_shape = x0.shape[2:]  # H, W
        x1_shape = x1.shape[2:]  # H, W

        for idx, layer in enumerate(self.layers):
            # Rearrange features into (B, H*W, C) and (B, W*H, C), two directions
            x0_hw = rearrange(x0, "b c h w -> b (h w) c")
            x1_hw = rearrange(x1, "b c h w -> b (h w) c")
            x0_wh = rearrange(x0, "b c h w -> b (w h) c")
            x1_wh = rearrange(x1, "b c h w -> b (w h) c")

            # Layer norms
            x0_hw = self.layer_norms[idx](x0_hw)
            x1_hw = self.layer_norms[idx](x1_hw)
            x0_wh = self.layer_norms[idx](x0_wh)
            x1_wh = self.layer_norms[idx](x1_wh)

            # Mamba forward, including another two directions, which is the flip of the original two directions
            x0_hw, x1_hw, x0_wh, x1_wh = layer(x0_hw, x1_hw, x0_wh, x1_wh)

            # Rearrange features back to (B, C, H, W)
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

            # Fusion
            x0 = 0.5 * (x0_hw + x0_wh)
            x1 = 0.5 * (x1_hw + x1_wh)

        return x0, x1
