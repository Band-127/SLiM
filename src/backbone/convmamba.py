from collections import OrderedDict
from typing import Sequence
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from yacs.config import CfgNode as CN
from timm.models.layers import trunc_normal_
from .vssm.csm_triton import cross_scan_fn, cross_merge_fn
from .vssm.csms6s import selective_scan_fn
from .vssm.vmamba import mamba_init
from .mamba.MambaEncoder import MambaEncoderLayer


class SizeInvariantPE(nn.Module):
    @torch.no_grad
    def __init__(self, d_model: int, pe_resolution: int = 256):
        super().__init__()
        self.d_model = d_model
        self.pe_resolution = pe_resolution
        # Size invariant position encoding, original resolution: 256*256, interpolate into input size during input
        pos_x = torch.arange(self.pe_resolution, dtype=torch.float32) * torch.pi
        pos_y = torch.arange(self.pe_resolution, dtype=torch.float32) * torch.pi
        d_pos_embeddings = int(math.ceil(d_model / 2))
        inv_freq = 1.0 / (
            (32768)
            ** (
                torch.arange(0, d_pos_embeddings, 2, dtype=torch.float32)
                / d_pos_embeddings
            )
        )
        pos_emb = torch.zeros(
            size=(self.pe_resolution, self.pe_resolution, 4 * inv_freq.shape[0]),
            dtype=torch.float32,
        )  # H, W, C'
        temp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
        temp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
        sin_x = torch.sin(temp_x).unsqueeze(0)
        sin_y = torch.sin(temp_y).unsqueeze(1)
        cos_x = torch.cos(temp_x).unsqueeze(0)
        cos_y = torch.cos(temp_y).unsqueeze(1)
        pos_emb[:, :, 0::4] = sin_x
        pos_emb[:, :, 1::4] = cos_x
        pos_emb[:, :, 2::4] = sin_y
        pos_emb[:, :, 3::4] = cos_y
        # Crop the exceeding channels, C' -> C
        pos_emb = pos_emb[:, :, 0:d_model]
        # H, W, C -> B, C, H, W
        pos_emb = pos_emb.swapaxes(0, -1).swapaxes(-1, -2).unsqueeze(0)
        self.register_buffer("pos_emb", pos_emb, persistent=False)

        # self.gamma = nn.Parameter(torch.ones(1, d_model, 1, 1) * 0.1)
        # Better Initialization, empirical, which means the center freq got the most attention
        def create_gaussian_kernel(kernel_size, sigma):
            center = kernel_size / 2
            x = torch.linspace(-center, center, kernel_size)
            gaussian_kernel = (1 / (2 * torch.pi * sigma**2) ** 0.5) * torch.exp(
                -0.5 * (x / sigma) ** 2
            )
            gaussian_kernel /= gaussian_kernel.sum()
            return gaussian_kernel

        self.gamma = create_gaussian_kernel(d_model, d_model / 12).view(1, -1, 1, 1)
        # Last Channel for image index, so give more attention too
        self.gamma[0, -1, 0, 0] = self.gamma.max()
        self.gamma = nn.Parameter(self.gamma)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        _, C, H, W = x0.shape
        with torch.no_grad():
            pe = F.interpolate(
                self.pos_emb[:, :C], size=(H, W), mode="bilinear", align_corners=False
            )
            pe0 = pe.clone()
            pe1 = pe.clone()
            pe0[:, -1, :, :] = 0
            pe1[:, -1, :, :] = 1
        return x0 + pe0 * self.gamma, x1 + pe1 * self.gamma


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNeXt_Block(nn.Module):
    def __init__(self, dim: int = 64, kernel_size: int = 7):
        super(ConvNeXt_Block, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) // 2)

        self.main_path = nn.Sequential(
            OrderedDict(
                # Depth-wise Conv
                dwconv=nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    groups=dim,
                ),
                # MLP
                ln=LayerNorm2d(dim),
                fc1=nn.Conv2d(
                    in_channels=self.dim,
                    out_channels=4 * self.dim,
                    kernel_size=1,
                    padding=0,
                ),
                act=nn.GELU(),
                fc2=nn.Conv2d(
                    in_channels=4 * self.dim,
                    out_channels=self.dim,
                    kernel_size=1,
                    padding=0,
                ),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        return x + self.main_path(x)


class Easy_SS2D(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 1,
        d_conv: int = 3,
        ssm_ratio: float = 2.0,
    ):
        super(Easy_SS2D, self).__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(d_model * ssm_ratio)
        self.dt_rank = int(math.ceil(self.d_model / 16))
        self.d_conv = int(d_conv)

        self.in_proj = nn.Sequential(
            OrderedDict(
                in_proj=nn.Conv2d(
                    in_channels=self.d_model,
                    out_channels=self.d_inner,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                ),
                dwconv=nn.Conv2d(
                    in_channels=self.d_inner,
                    out_channels=self.d_inner,
                    kernel_size=self.d_conv,
                    padding=(self.d_conv - 1) // 2,
                    groups=self.d_inner,
                ),
                act=nn.SiLU(inplace=True),
            )
        )

        self.out_proj = nn.Sequential(
            OrderedDict(
                ln=LayerNorm2d(self.d_inner),
                out_proj=nn.Conv2d(
                    in_channels=self.d_inner,
                    out_channels=self.d_model,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                ),
            )
        )

        self.x_proj_weight = nn.Parameter(
            torch.randn(self.k_group, (self.dt_rank + self.d_state * 2), self.d_inner)
        )  # (K, N, inner)

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = (
            mamba_init.init_dt_A_D(
                d_state=self.d_state,
                dt_rank=self.dt_rank,
                d_inner=self.d_inner,
                dt_scale=1.0,
                dt_init="random",
                dt_min=0.001,
                dt_max=0.1,
                dt_init_floor=1e-4,
                k_group=4,
            )
        )

    def forwardSS2D(self, x: torch.Tensor):
        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W
        xs = cross_scan_fn(
            x,
            in_channel_first=True,
            out_channel_first=True,
            scans=0,
            force_torch=False,
        )
        x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(
            dts.contiguous().view(B, -1, L),
            self.dt_projs_weight.view(K * D, -1, 1),
            groups=K,
        )
        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
        Ds = self.Ds.to(torch.float)  # (K * c)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        ys: torch.Tensor = selective_scan_fn(
            u=xs,
            delta=dts,
            A=As,
            B=Bs,
            C=Cs,
            D=Ds,
            delta_bias=delta_bias,
            delta_softplus=True,
            oflex=True,
            backend=None,
        ).view(B, K, -1, H, W)
        y: torch.Tensor = cross_merge_fn(
            ys,
            in_channel_first=True,
            out_channel_first=True,
            scans=0,
            force_torch=False,
        )
        y = y.view(B, -1, H, W).to(x.dtype)
        return y

    def forward(self, x: torch.Tensor):
        x = self.in_proj(x)
        x = self.forwardSS2D(x)
        return self.out_proj(x)


class Easy_VSSBlock(nn.Module):
    def __init__(self, dim: int = 64):
        super(Easy_VSSBlock, self).__init__()
        self.dim = dim
        self.ssm = nn.Sequential(
            OrderedDict(
                ln=LayerNorm2d(self.dim),
                ss2d=Easy_SS2D(
                    d_model=self.dim,
                    d_state=1,
                    ssm_ratio=1.0,
                    d_conv=3,
                ),
            )
        )

        self.mlp = nn.Sequential(
            OrderedDict(
                ln=LayerNorm2d(self.dim),
                fc1=nn.Conv2d(
                    in_channels=self.dim,
                    out_channels=4 * self.dim,
                    kernel_size=1,
                    padding=0,
                ),
                act=nn.GELU(),
                fc2=nn.Conv2d(
                    in_channels=4 * self.dim,
                    out_channels=self.dim,
                    kernel_size=1,
                    padding=0,
                ),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(x)
        x = x + self.mlp(x)
        return x


class InceptionNeXt_Block(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=7,
        split_ratio=8,
        aggregation_size=4,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) // 2)
        self.split_dim = int(self.dim // split_ratio)
        self.split_idices = (
            self.split_dim,
            self.split_dim,
            self.split_dim,
            self.dim - 3 * self.split_dim,
        )
        self.aggregation_size = aggregation_size

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
            LayerNorm2d(self.dim),
            nn.Conv2d(
                in_channels=self.dim,
                out_channels=4 * self.dim,
                kernel_size=1,
                padding=0,
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=4 * self.dim,
                out_channels=self.dim,
                kernel_size=1,
                padding=0,
            ),
        )

        self.downsample_conv = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.aggregation_size,
            stride=self.aggregation_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
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


class Aggregated_InceptionMamba(nn.Module):
    def __init__(
        self,
        d_model: int = 192,
        depth: int = 2,
        aggregation_size: int = 4,
        conv_kernel_size: int = 7,
    ):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.aggregation_size = aggregation_size
        self.conv_kernel_size = conv_kernel_size

        self.convs = nn.ModuleList(
            [
                InceptionNeXt_Block(
                    dim=self.d_model,
                    kernel_size=self.conv_kernel_size,
                    split_ratio=8,
                    aggregation_size=self.aggregation_size,
                )
                for _ in range(self.depth)
            ]
        )
        self.mambas = nn.ModuleList(
            [
                MambaEncoderLayer(
                    in_output_dim=self.d_model,
                    inner_expansion=2,
                    conv_dim=3,
                    delta=4,
                    using_mamba2=True,
                    aggregation_size=self.aggregation_size,
                )
                for _ in range(self.depth)
            ]
        )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, x1 = torch.chunk(x, 2, dim=-1)
        for idx in range(self.depth):
            # 1. Input conv + skip
            x0 = self.convs[idx](x0) + x0
            x1 = self.convs[idx](x1) + x1

            # 2. Mamba + skip
            _x0, _x1 = self.mambas[idx](x0, x1)
            x0 = _x0 + x0
            x1 = _x1 + x1

        return torch.concat([x0, x1], dim=-1)


# class InceptionNeXt_Block(nn.Module):
#     def __init__(
#         self,
#         dim,
#         kernel_size=7,
#         split_ratio=8,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.kernel_size = kernel_size
#         self.padding = int((kernel_size - 1) // 2)
#         self.split_dim = int(self.dim // split_ratio)
#         self.split_idices = (
#             self.split_dim,
#             self.split_dim,
#             self.split_dim,
#             self.dim - 3 * self.split_dim,
#         )

#         self.branch_1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=self.split_dim,
#                 out_channels=self.split_dim,
#                 kernel_size=3,
#                 padding=1,
#                 groups=self.split_dim,
#             ),
#         )
#         self.branch_2 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=self.split_dim,
#                 out_channels=self.split_dim,
#                 kernel_size=(self.kernel_size, 1),
#                 padding=(self.padding, 0),
#                 groups=self.split_dim,
#             ),
#         )
#         self.branch_3 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=self.split_dim,
#                 out_channels=self.split_dim,
#                 kernel_size=(1, self.kernel_size),
#                 padding=(0, self.padding),
#                 groups=self.split_dim,
#             ),
#         )

#         self.mlp = nn.Sequential(
#             LayerNorm2d(self.dim),
#             nn.Conv2d(
#                 in_channels=self.dim,
#                 out_channels=4 * self.dim,
#                 kernel_size=1,
#                 padding=0,
#             ),
#             nn.GELU(),
#             nn.Conv2d(
#                 in_channels=4 * self.dim,
#                 out_channels=self.dim,
#                 kernel_size=1,
#                 padding=0,
#             ),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x_hw, x_h, x_w, x_id = torch.split(x, self.split_idices, dim=1)
#         return x + self.mlp(
#             torch.concat(
#                 [
#                     self.branch_1(x_hw),
#                     self.branch_2(x_h),
#                     self.branch_3(x_w),
#                     x_id,
#                 ],
#                 dim=1,
#             )
#         )


# class Aggregated_InceptionMamba(nn.Module):
#     def __init__(
#         self,
#         d_model: int = 192,
#         depth: int = 2,
#         aggregation_size: int = 4,
#         conv_kernel_size: int = 7,
#     ):
#         super().__init__()
#         self.d_model = d_model
#         self.depth = depth
#         self.aggregation_size = aggregation_size
#         self.conv_kernel_size = conv_kernel_size
#         self.aggregation = nn.Sequential(
#             LayerNorm2d(self.d_model),
#             nn.Conv2d(
#                 in_channels=self.d_model,
#                 out_channels=self.d_model,
#                 kernel_size=self.aggregation_size,
#                 stride=self.aggregation_size,
#             ),
#         )

#         self.blocks = []
#         for _ in range(self.depth):
#             self.blocks.append(
#                 nn.Sequential(
#                     OrderedDict(
#                         conv=InceptionNeXt_Block(
#                             dim=self.d_model,
#                             kernel_size=self.conv_kernel_size,
#                         ),
#                         mamba=Easy_VSSBlock(dim=self.d_model),
#                     )
#                 )
#             )
#         self.blocks = nn.Sequential(*self.blocks)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         _, _, H, W = x.shape
#         _x = x
#         x = self.aggregation(x)
#         x = self.blocks(x)
#         return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) + _x


class ConvVMamba(nn.Module):
    def __init__(self, **kwargs):
        super(ConvVMamba, self).__init__()
        if "config" in kwargs:
            self._init_config(kwargs["config"])
        else:
            self._init_normal(**kwargs)

        with torch.no_grad():
            self.apply(self._init_weights)

    def _init_config(self, config: CN):
        self._init_normal(
            rgb_input=config["RGB_INPUT"],
            patch_size=config["PATCH_SIZE"],
            layers_dims=config["DIMS"],
            layers_depth=config["DEPTHS"],
            extra_depth=config["EXTRA_DEPTH"],
            extra_aggregation=config["EXTRA_AGGREGATION"],
            conv_kernel_size=config["CONV_KERNEL_SIZE"],
        )

    def _init_normal(
        self,
        rgb_input: bool = False,
        patch_size: int = 2,
        layers_dims: Sequence[int] = [48, 96, 192],
        layers_depth: Sequence[int] = [1, 1, 1],
        extra_depth: int = 2,
        extra_aggregation: int = 4,
        conv_kernel_size: int = 7,
    ):
        self.in_channel = 3 if rgb_input else 1
        self.patch_size = patch_size
        self.layers_dims = layers_dims
        self.layers_depth = layers_depth
        self.extra_depth = extra_depth
        self.extra_aggregation = extra_aggregation
        self.conv_kernel_size = conv_kernel_size

        # 1. Patch embedding, ConvNeXt layer only for first stage, without too much context information
        input_ksize = (
            self.patch_size + 1 if self.patch_size % 2 == 0 else self.patch_size
        )
        input_padding = int((input_ksize - 1) // 2)
        self.first_stage = nn.Sequential(
            OrderedDict(
                patch_embed=nn.Sequential(
                    OrderedDict(
                        conv=nn.Conv2d(
                            self.in_channel,
                            self.layers_dims[0],
                            kernel_size=input_ksize,
                            padding=input_padding,
                            stride=self.patch_size,
                            bias=False,
                        ),
                        ln=LayerNorm2d(self.layers_dims[0]),
                    )
                ),
                conv=nn.Sequential(
                    *[
                        ConvNeXt_Block(
                            dim=self.layers_dims[0],
                            kernel_size=self.conv_kernel_size,
                        )
                        for i in range(self.layers_depth[0])
                    ]
                ),
            )
        )
        # 2. PE
        self.pe = SizeInvariantPE(d_model=self.layers_dims[0])

        # 3. For other stages: Downsample -> Conv -> SS2D
        self.stages = nn.ModuleList([])
        for i in range(0, len(self.layers_depth) - 1):
            downsample = nn.Sequential(
                OrderedDict(
                    ln=LayerNorm2d(self.layers_dims[i]),
                    conv=nn.Conv2d(
                        in_channels=self.layers_dims[i],
                        out_channels=self.layers_dims[i + 1],
                        kernel_size=3,
                        padding=1,
                        stride=2,
                    ),
                )
            )
            blocks = []
            for _ in range(self.layers_depth[i + 1]):
                blocks.append(
                    nn.Sequential(
                        OrderedDict(
                            conv=ConvNeXt_Block(
                                dim=self.layers_dims[i + 1],
                                kernel_size=self.conv_kernel_size,
                            ),
                            mamba=Easy_VSSBlock(dim=self.layers_dims[i + 1]),
                        )
                    )
                )

            # 4. Extra depths for last stage
            if i == len(self.layers_depth) - 2:
                blocks.append(
                    Aggregated_InceptionMamba(
                        d_model=self.layers_dims[i + 1],
                        depth=self.extra_depth,
                        aggregation_size=self.extra_aggregation,
                        conv_kernel_size=self.conv_kernel_size,
                    )
                )

            self.stages.append(
                nn.Sequential(
                    OrderedDict(
                        downsample=downsample,
                        blocks=nn.Sequential(*blocks),
                    )
                )
            )

        # 4. FPN
        self.lateral_conv_1 = nn.Conv2d(
            self.layers_dims[0],
            self.layers_dims[1],
            kernel_size=1,
            padding=0,
            stride=1,
        )
        self.lateral_conv_2 = nn.Conv2d(
            self.layers_dims[1],
            self.layers_dims[2],
            kernel_size=1,
            padding=0,
            stride=1,
        )
        self.output_conv_1 = nn.Sequential(
            nn.Conv2d(
                self.layers_dims[1],
                self.layers_dims[1],
                kernel_size=3,
                padding=1,
                groups=self.layers_dims[1],
            ),
            nn.BatchNorm2d(self.layers_dims[1]),
            nn.GELU(),
            nn.Conv2d(
                self.layers_dims[1],
                self.layers_dims[0],
                kernel_size=1,
                padding=0,
            ),
        )
        self.output_conv_2 = nn.Sequential(
            nn.Conv2d(
                self.layers_dims[2],
                self.layers_dims[2],
                kernel_size=3,
                padding=1,
                groups=self.layers_dims[2],
            ),
            nn.BatchNorm2d(self.layers_dims[2]),
            nn.GELU(),
            nn.Conv2d(
                self.layers_dims[2],
                self.layers_dims[1],
                kernel_size=1,
                padding=0,
            ),
        )
        self.output_conv_3 = nn.Conv2d(
            self.layers_dims[2],
            self.layers_dims[2],
            kernel_size=1,
            padding=0,
            stride=1,
        )

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    @torch.jit.script
    def upsample_sum(
        not_upsample_feature: torch.Tensor, upsample_feature: torch.Tensor
    ):
        _, _, H, W = not_upsample_feature.shape
        return not_upsample_feature + F.interpolate(
            upsample_feature, size=(H, W), mode="bilinear", align_corners=False
        )

    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        features_0 = []
        features_1 = []

        # 1. First stage
        features = []
        x = torch.concat([x0, x1], dim=0)  # Concat in batch
        x = self.first_stage(x)
        x0, x1 = torch.chunk(x, 2, dim=0)
        features_0.append(x0)
        features_1.append(x1)

        # 2. PE
        x0, x1 = self.pe(x0, x1)

        # 3. Others
        x = torch.concat([x0, x1], dim=-1)  # Concat in W
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        # 4. Split back into two
        for feature in features:
            x0, x1 = torch.chunk(feature, 2, dim=-1)
            features_0.append(x0)
            features_1.append(x1)

        # 5. FPN
        # Concat in batch
        x = [
            torch.concat([_x0, _x1], dim=0) for _x0, _x1 in zip(features_0, features_1)
        ]

        f3 = self.output_conv_3(x[2])
        f2 = self.output_conv_2(
            self.upsample_sum(
                not_upsample_feature=self.lateral_conv_2(x[1]),
                upsample_feature=f3,
            )
        )
        f1 = self.output_conv_1(
            self.upsample_sum(
                not_upsample_feature=self.lateral_conv_1(x[0]),
                upsample_feature=f2,
            )
        )
        features = [f1, f2, x[2]]

        # Split back into two
        features_0 = []
        features_1 = []
        for feature in features:
            x0, x1 = torch.chunk(feature, 2, dim=0)
            features_0.append(x0)
            features_1.append(x1)

        return features_0, features_1

    def initial_forward(self):
        for i in range(10):
            image0 = torch.zeros(1, self.in_channel, 480, 640).to(
                self.output_conv_3.weight.device
            )
            image1 = torch.zeros(1, self.in_channel, 480, 640).to(
                self.output_conv_3.weight.device
            )
            self.forward(image0, image1)
        torch.cuda.empty_cache()
