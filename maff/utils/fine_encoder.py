import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple
from einops.einops import rearrange
from typing import List
from maff.mamba.MambaEncoder import MambaLayer


class FineEncoder(nn.Module):
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
        super(FineEncoder, self).__init__()

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

        # # For all fine scales, interpolate and add together for forcelly fusion
        # all_window_size_squared = [w**2 for w in all_window_size]
        # # [M x SHW x C] -> S x [M x HW x C]
        # new_feat0_quadtrees = list(
        #     torch.split(new_feat0_quadtrees, all_window_size_squared, dim=-2)
        # )
        # new_feat1_quadtrees = list(
        #     torch.split(new_feat1_quadtrees, all_window_size_squared, dim=-2)
        # )
        # # the finest
        # finest_feat0 = new_feat0_quadtrees.pop(-1)
        # finest_feat1 = new_feat1_quadtrees.pop(-1)
        # finest_feat0 = rearrange(
        #     finest_feat0,
        #     "m (h w) c -> m c h w",
        #     h=all_window_size[-1],
        #     w=all_window_size[-1],
        # )  # [M x C x H x W]
        # finest_feat1 = rearrange(
        #     finest_feat1,
        #     "m (h w) c -> m c h w",
        #     h=all_window_size[-1],
        #     w=all_window_size[-1],
        # )  # [M x C x H x W]
        # # the others
        # scale = 2 ** len(new_feat0_quadtrees)
        # for idx in range(len(new_feat0_quadtrees)):
        #     feat0 = new_feat0_quadtrees[idx]
        #     feat1 = new_feat1_quadtrees[idx]
        #     feat0 = rearrange(
        #         feat0,
        #         "m (h w) c -> m c h w",
        #         h=all_window_size[idx],
        #         w=all_window_size[idx],
        #     )  # [M x C x H x W]
        #     feat1 = rearrange(
        #         feat1,
        #         "m (h w) c -> m c h w",
        #         h=all_window_size[idx],
        #         w=all_window_size[idx],
        #     )  # [M x C x H x W]
        #     finest_feat0 = finest_feat0 + F.interpolate(feat0, scale_factor=scale)
        #     finest_feat1 = finest_feat1 + F.interpolate(feat1, scale_factor=scale)
        #     scale /= 2

        # finest_windows_size = all_window_size[-1]
        # ww = finest_windows_size**2
        # start = new_feat0_quadtrees.shape[-2] - ww
        # finest_feat0 = new_feat0_quadtrees[:, start:, :]  # [M x WW x C]
        # finest_feat1 = new_feat1_quadtrees[:, start:, :]  # [M x WW x C]
        # finest_feat0 = rearrange(
        #     finest_feat0,
        #     "m (h w) c -> m c h w",
        #     h=finest_windows_size,
        #     w=finest_windows_size,
        # )  # [M x C x H x W]
        # finest_feat1 = rearrange(
        #     finest_feat1,
        #     "m (h w) c -> m c h w",
        #     h=finest_windows_size,
        #     w=finest_windows_size,
        # )  # [M x C x H x W]
        # Output: single scale fine feature, M x C x H x W, at the end of quadtrees using finest_windows_size
        return feat0, feat1
