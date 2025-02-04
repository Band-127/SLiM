from collections import OrderedDict
from typing import Sequence
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from yacs.config import CfgNode as CN
from timm.models.layers import DropPath, trunc_normal_
from src.backbone.vssm.csm_triton import cross_scan_fn, cross_merge_fn
from src.backbone.vssm.csms6s import selective_scan_fn


class SizeInvariantPE(nn.Module):
    @torch.no_grad
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Size invariant position encoding, original resolution: 512*512, interpolate into input size during input
        pos_x = torch.arange(512, dtype=torch.float32) * torch.pi
        pos_y = torch.arange(512, dtype=torch.float32) * torch.pi
        d_pos_embeddings = int(math.ceil(d_model / 2))
        inv_freq = 1.0 / (
            (32768)
            ** (
                torch.arange(0, d_pos_embeddings, 2, dtype=torch.float32)
                / (d_pos_embeddings - 1)
            )
        )
        pos_emb = torch.zeros(
            size=(512, 512, 4 * inv_freq.shape[0]), dtype=torch.float32
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
        self.register_buffer("pos_emb", pos_emb)

        self.gamma = nn.Parameter(torch.ones(1, d_model, 1, 1) * 0.5)

    def forward(self, x):
        _, C, H, W = x.shape
        return (
            x
            + F.interpolate(
                self.pos_emb[:, :C], size=(H, W), mode="bilinear", align_corners=False
            )
            * self.gamma
        )


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNeXt_Block(nn.Module):
    def __init__(self, dim: int = 64, kernel_size: int = 7, drop_rate: float = 0.0):
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

        self.drop_path = DropPath(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.main_path(x))


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
        self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
        self.A_logs = nn.Parameter(
            torch.zeros((self.k_group * self.d_inner, self.d_state))
        )  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_weight = nn.Parameter(
            0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank))
        )
        self.dt_projs_bias = nn.Parameter(
            0.1 * torch.rand((self.k_group, self.d_inner))
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
    def __init__(self, dim: int = 64, drop_rate: float = 0.0):
        super(Easy_VSSBlock, self).__init__()
        self.dim = dim
        self.drop_path = DropPath(drop_rate) if drop_rate > 0 else nn.Identity()
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

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.ssm(x))
        x = x + self.drop_path(self.mlp(x))
        return x


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
            rgb_input=False,
            layers_dims=[48, 96, 192],
            layers_depth=[1, 1, 1],
            conv_kernel_size=7,
        )

    def _init_normal(
        self,
        rgb_input: bool = False,
        layers_dims: Sequence[int] = [48, 96, 192],
        layers_depth: Sequence[int] = [1, 1, 1],
        conv_kernel_size: int = 7,
    ):
        self.in_channel = 3 if rgb_input else 1
        self.layers_dims = layers_dims
        self.layers_depth = layers_depth
        self.conv_kernel_size = conv_kernel_size
        # 1. Patch embedding/Stem, 2x2 patch + ConvNeXt layer only for first stage
        self.first_stage = nn.Sequential(
            OrderedDict(
                patch_embed=nn.Sequential(
                    OrderedDict(
                        conv=nn.Conv2d(
                            self.in_channel,
                            self.layers_dims[0],
                            kernel_size=3,
                            padding=1,
                            stride=2,
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
                            drop_rate=0.01,
                        )
                        for i in range(self.layers_depth[0])
                    ]
                ),
            )
        )

        # 2. For other stages: Downsample -> Conv -> VMamba
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
                                drop_rate=0.01,
                            ),
                            mamba=Easy_VSSBlock(
                                dim=self.layers_dims[i + 1], drop_rate=0.01
                            ),
                        )
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

        # 3. FPN
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

        # 4. PE
        self.pe = SizeInvariantPE(d_model=self.layers_dims[0])

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
        return not_upsample_feature + F.interpolate(
            upsample_feature, scale_factor=2.0, mode="bilinear", align_corners=False
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

        x0 = self.pe(x0)
        x1 = self.pe(x1)
        # 2. Others
        x = torch.concat([x0, x1], dim=-1)  # Concat in W
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        # 3. Split back into two
        for feature in features:
            x0, x1 = torch.chunk(feature, 2, dim=-1)
            features_0.append(x0)
            features_1.append(x1)

        return features_0, features_1

    def fpn(self, x0: torch.Tensor, x1: torch.Tensor):
        # Concat in batch
        x = [torch.concat([_x0, _x1], dim=0) for _x0, _x1 in zip(x0, x1)]

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
        features = [f1, f2, f3]

        # Split back into two
        features_0 = []
        features_1 = []
        for feature in features:
            x0, x1 = torch.chunk(feature, 2, dim=0)
            features_0.append(x0)
            features_1.append(x1)

        return features_0, features_1
