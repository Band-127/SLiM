import torch
import torch.nn as nn
from typing import Tuple
from einops.einops import rearrange
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
        finest_windows_size,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input: [M x L x C], [M x L x C]
        x = torch.concat([feat0_quadtrees, feat1_quadtrees], dim=-2)  # [M x 2L x C]

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

        ww = finest_windows_size**2
        start = new_feat0_quadtrees.shape[-2] - ww
        finest_feat0 = new_feat0_quadtrees[:, start:, :]  # [M x WW x C]
        finest_feat1 = new_feat1_quadtrees[:, start:, :]  # [M x WW x C]

        finest_feat0 = rearrange(
            finest_feat0,
            "m (h w) c -> m c h w",
            h=finest_windows_size,
            w=finest_windows_size,
        )  # [M x C x H x W]
        finest_feat1 = rearrange(
            finest_feat1,
            "m (h w) c -> m c h w",
            h=finest_windows_size,
            w=finest_windows_size,
        )  # [M x C x H x W]
        # Output: single scale fine feature, M x C x H x W, at the end of quadtrees using finest_windows_size
        return finest_feat0, finest_feat1
