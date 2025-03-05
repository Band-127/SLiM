import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Sequence
from einops.einops import rearrange
from typing import List
from timm.models.layers import trunc_normal_, DropPath

from src.utils.misc import LayerNorm2d
from src.mamba.MambaEncoder import MambaLayer


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


class FineEncoder_conv(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, num_layers: int, drop_rate: float
    ) -> None:
        super(FineEncoder_conv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.input_conv = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.output_dim,
            kernel_size=1,
            padding=0,
        )

        self.main_conv = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm2d(self.output_dim),
                    nn.Conv2d(
                        self.output_dim, self.output_dim * 2, kernel_size=3, padding=1
                    ),
                    LayerNorm2d(self.output_dim * 2),
                    nn.GELU(approximate="tanh"),
                    nn.Conv2d(
                        self.output_dim * 2, self.output_dim, kernel_size=3, padding=1
                    ),
                )
                for _ in range(num_layers)
            ]
        )

        self.drop_path = DropPath(drop_rate) if drop_rate > 0.0 else nn.Identity()

        # Initialize weights
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    @torch.no_grad
    def initial_forward(self, size: Sequence[int], batch_size: int):
        for i in range(5):
            random_data_0 = torch.zeros(batch_size, self.input_dim, size[0], size[1]).to(
                self.input_conv.weight.device
            )
            random_data_1 = torch.zeros(batch_size, self.input_dim, size[0], size[1]).to(
                self.input_conv.weight.device
            )
            _ = self.forward(random_data_0, random_data_1)
        torch.cuda.empty_cache()

    def forward(self, x0, x1):
        B = x0.size(0)
        # Concatenate x0 and x1 along the batch dimension
        x = torch.cat([x0, x1], dim=0)
        x = self.input_conv(x)

        for i in range(self.num_layers):
            x = self.drop_path(self.main_conv[i](x)) + x

        x0 = x[:B]
        x1 = x[B:]

        return x0, x1


class FineEncoder_upsample(nn.Module):
    def __init__(self, dim: int) -> None:
        super(FineEncoder_upsample, self).__init__()
        self.dim = int(dim)

    def forward(self, x0, x1):
        B = x0.size(0)

        x = torch.cat([x0, x1], dim=0)

        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)

        x0 = x[:B]
        x1 = x[B:]

        return x0, x1
