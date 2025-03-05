import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Sequence
from collections import OrderedDict
from timm.models.layers import trunc_normal_, DropPath

from src.utils.misc import LayerNorm2d
from src.backbone.vssm.vmamba import VSSBlock
from src.mamba.MambaEncoder import MambaEncoderLayer


class InceptionNeXt(nn.Module):
    def __init__(
        self,
        in_output_dim,
        kernel_size=7,
        split_ratio=8,
        aggregation_size=4,
    ):
        super(InceptionNeXt, self).__init__()
        self.in_output_dim = in_output_dim
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) // 2)
        self.aggregation_size = aggregation_size

        self.split_dim = int(self.in_output_dim // split_ratio)
        self.split_idices = (
            self.split_dim,
            self.split_dim,
            self.split_dim,
            self.in_output_dim - 3 * self.split_dim,
        )

        self.branch_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.split_dim,
                out_channels=self.split_dim,
                kernel_size=3,
                padding=1,
                groups=self.split_dim,
            ),
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.split_dim,
                out_channels=self.split_dim,
                kernel_size=(self.kernel_size, 1),
                padding=(self.padding, 0),
                groups=self.split_dim,
            ),
        )
        self.branch_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.split_dim,
                out_channels=self.split_dim,
                kernel_size=(1, self.kernel_size),
                padding=(0, self.padding),
                groups=self.split_dim,
            ),
        )

        self.mlp = nn.Sequential(
            LayerNorm2d(self.in_output_dim),
            nn.Conv2d(
                in_channels=self.in_output_dim,
                out_channels=4 * self.in_output_dim,
                kernel_size=1,
                padding=0,
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=4 * self.in_output_dim,
                out_channels=self.in_output_dim,
                kernel_size=1,
                padding=0,
            ),
        )

        self.downsample_conv = nn.Conv2d(
            in_channels=self.in_output_dim,
            out_channels=self.in_output_dim,
            kernel_size=self.aggregation_size,
            stride=self.aggregation_size,
        )

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.downsample_conv(x)
        x_hw, x_h, x_w, x_id = torch.split(x, self.split_idices, dim=1)
        x = self.mlp(
            torch.concat(
                [
                    self.branch_1(x_hw),
                    self.branch_2(x_h),
                    self.branch_3(x_w),
                    x_id,
                ],
                dim=1,
            )
        )
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)


class CoarseEncoder(nn.Module):
    """
    Coarse feature encoder(single scale)
    """

    def __init__(
        self,
        in_output_dim: int,
        num_layers: int,
        inner_expansion: int,
        conv_dim: int,
        delta: int,
        using_mamba2: bool,
        drop_rate: float,
    ) -> None:
        super(CoarseEncoder, self).__init__()

        self.in_output_dim = in_output_dim
        self.num_layers = num_layers
        self.inner_expansion = inner_expansion
        self.conv_dim = conv_dim
        self.delta = delta
        self.using_mamba2 = using_mamba2

        self.convs = nn.ModuleList(
            [
                InceptionNeXt(
                    in_output_dim=self.in_output_dim,
                    kernel_size=7,
                    split_ratio=8,
                    aggregation_size=4,
                )
                for _ in range(num_layers)
            ]
        )
        self.mambas = nn.ModuleList(
            [
                MambaEncoderLayer(
                    in_output_dim=self.in_output_dim,
                    inner_expansion=self.inner_expansion,
                    conv_dim=self.conv_dim,
                    delta=self.delta,
                    using_mamba2=self.using_mamba2,
                    aggregation_size=4,
                )
                for _ in range(num_layers)
            ]
        )

        self.drop_path = DropPath(drop_rate) if drop_rate > 0.0 else nn.Identity()

        # Initialize weights
        with torch.no_grad():
            for m in self.convs.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    @torch.no_grad
    def initial_forward(self, size: Sequence[int], batch_size: int):
        for i in range(5):
            random_data_0 = torch.zeros(
                batch_size, self.in_output_dim, size[0], size[1]
            ).to(self.mambas[0].mamba.mamba_forward.mamba_layer.in_proj.weight.device)
            random_data_1 = torch.zeros(
                batch_size, self.in_output_dim, size[0], size[1]
            ).to(self.mambas[0].mamba.mamba_forward.mamba_layer.in_proj.weight.device)
            _ = self.forward(random_data_0, random_data_1)
        torch.cuda.empty_cache()

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x0 (torch.Tensor): (B, C, H, W)
            x1 (torch.Tensor): (B, C, H, W)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (B, C, H, W)
        """
        for idx in range(self.num_layers):
            # 1. Input conv + skip
            x0 = self.drop_path(self.convs[idx](x0)) + x0
            x1 = self.drop_path(self.convs[idx](x1)) + x1

            # 2. Mamba + skip
            _x0, _x1 = self.mambas[idx](x0, x1)
            x0 = self.drop_path(_x0) + x0
            x1 = self.drop_path(_x1) + x1
        return x0, x1


class CoarseEncoder_vssm(nn.Module):
    def __init__(self, in_output_dim: int, num_layers: int) -> None:
        super(CoarseEncoder_vssm, self).__init__()
        self.in_output_dim = in_output_dim
        self.num_layers = num_layers

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"layer_{i}",
                        VSSBlock(
                            hidden_dim=self.in_output_dim,
                            drop_path=0.1,
                            norm_layer=LayerNorm2d,
                            channel_first=True,
                            ssm_d_state=1,
                            ssm_ratio=1.0,
                            ssm_dt_rank="auto",
                            ssm_act_layer=nn.SiLU,
                            ssm_conv=3,
                            ssm_conv_bias=False,
                            ssm_drop_rate=0.0,
                            ssm_init="v0",
                            forward_type="v05_noz",
                            mlp_ratio=4.0,
                            mlp_act_layer=nn.GELU,
                            mlp_drop_rate=0.0,
                            gmlp=False,
                            use_checkpoint=False,
                        ),
                    )
                    for i in range(self.num_layers)
                ]
            )
        )
        # Initialize weights
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Concat the features in W dimension
        x = torch.cat([x0, x1], dim=-1)  # (B, C, H, 2W)
        x_backwards = x.flip(dims=[-1])  # (B, C, H, 2W)
        x = self.layers(x)  # (B, C, H, 2W)
        x_backwards = self.layers(x_backwards)  # (B, C, H, 2W)
        x0 = 0.5 * (
            x[:, :, :, : x0.shape[-1]] + x_backwards.flip(-1)[:, :, :, : x0.shape[-1]]
        )
        x1 = 0.5 * (
            x[:, :, :, x0.shape[-1] :] + x_backwards.flip(-1)[:, :, :, x0.shape[-1] :]
        )
        return x0, x1
