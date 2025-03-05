from typing import Tuple, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba, Mamba2
from einops.einops import rearrange
from typing import List

from src.backbone.vssm.vmamba import LayerNorm2d


class MambaLayer(nn.Module):
    def __init__(
        self,
        in_output_dim: int = 1024,
        inner_expansion: float = 2.0,
        conv_dim: int = 4,
        delta: int = 16,
        device: torch.device = None,
        dtype: torch.dtype = None,
        using_mamba2: bool = False,
    ):
        """
        Default constructor for MambaLayer using vanilla Mamba.

        input -> (B, L, in_output_dim)
        (inner) -> (B, L, in_output_dim * inner_expansion)
        output -> (B, L, in_output_dim)

        Args:
            in_output_dim (int, optional): Input and output dimension. Defaults to 1024.
            inner_expansion (float, optional): Expansion for inner dimension. Defaults to 2.0.
            conv_dim (int, optional): Local convolution dimension. Defaults to 4.
            device (torch.device, optional): Device. Defaults to None.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            using_mamba2 (bool, optional): Whether using Mamba2 or not. Defaults to False.
        """
        super(MambaLayer, self).__init__()

        self.in_output_dim: int = in_output_dim
        self.inner_expansion: float = inner_expansion
        self.conv_dim: int = conv_dim
        self.delta: int = delta
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.using_mamba2: bool = using_mamba2

        # vanilla Mamba builder
        self.mamba_base = Mamba2 if self.using_mamba2 else Mamba

        self.mamba_layer = (
            self.mamba_base(
                d_model=in_output_dim,
                d_state=16 if self.delta is None else self.delta,
                d_conv=4 if self.conv_dim is None else self.conv_dim,
                expand=inner_expansion,
                device=self.device,
                dtype=self.dtype,
                headdim=int(min(64, in_output_dim // 4)),
            )
            if self.using_mamba2
            else self.mamba_base(
                d_model=in_output_dim,
                d_state=16 if self.delta is None else self.delta,
                d_conv=4 if self.conv_dim is None else self.conv_dim,
                expand=inner_expansion,
                device=self.device,
                dtype=self.dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for MambaLayer.
        Args:
            x (torch.Tensor): input tensor(Batch, Sequence, Dimension)

        Returns:
            torch.Tensor: output tensor(Batch, Sequence, Dimension)
        """
        assert x.shape[-1] == self.in_output_dim

        x = self.mamba_layer(x)

        return x


class QuadDualInputMambaLayer(nn.Module):
    def __init__(
        self,
        in_output_dim: int = 1024,
        inner_expansion: float = 2.0,
        conv_dim: int = 4,
        delta: int = 16,
        device: torch.device = None,
        dtype: torch.dtype = None,
        using_mamba2: bool = False,
    ):
        super(QuadDualInputMambaLayer, self).__init__()
        self.in_output_dim = in_output_dim
        self.inner_expansion = inner_expansion
        self.conv_dim = conv_dim
        self.delta = delta
        self.device = device
        self.dtype = dtype
        self.using_mamba2 = using_mamba2

        # Four mamba layers for four directions
        self.mamba_hw_0 = MambaLayer(
            in_output_dim=self.in_output_dim,
            inner_expansion=self.inner_expansion,
            conv_dim=self.conv_dim,
            delta=self.delta,
            device=self.device,
            dtype=self.dtype,
            using_mamba2=self.using_mamba2,
        )
        self.mamba_wh_0 = MambaLayer(
            in_output_dim=self.in_output_dim,
            inner_expansion=self.inner_expansion,
            conv_dim=self.conv_dim,
            delta=self.delta,
            device=self.device,
            dtype=self.dtype,
            using_mamba2=self.using_mamba2,
        )
        self.mamba_hw_1 = MambaLayer(
            in_output_dim=self.in_output_dim,
            inner_expansion=self.inner_expansion,
            conv_dim=self.conv_dim,
            delta=self.delta,
            device=self.device,
            dtype=self.dtype,
            using_mamba2=self.using_mamba2,
        )
        self.mamba_wh_1 = MambaLayer(
            in_output_dim=self.in_output_dim,
            inner_expansion=self.inner_expansion,
            conv_dim=self.conv_dim,
            delta=self.delta,
            device=self.device,
            dtype=self.dtype,
            using_mamba2=self.using_mamba2,
        )

        # LayerNorm
        self.layer_norm = nn.LayerNorm(in_output_dim)

    def forward(
        self,
        x0_hw: torch.Tensor,
        x1_hw: torch.Tensor,
        x0_wh: torch.Tensor,
        x1_wh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Quad directional scan
        _x0_hw = torch.concat([x0_hw, x0_hw.flip(dims=(-2,))], dim=-2)
        _x1_hw = torch.concat([x1_hw, x1_hw.flip(dims=(-2,))], dim=-2)
        _x0_wh = torch.concat([x0_wh, x0_wh.flip(dims=(-2,))], dim=-2)
        _x1_wh = torch.concat([x1_wh, x1_wh.flip(dims=(-2,))], dim=-2)

        # 2. Enter mamba layers
        _x0_hw = self.mamba_hw_0(_x0_hw)
        _x1_hw = self.mamba_hw_1(_x1_hw)
        _x0_wh = self.mamba_wh_0(_x0_wh)
        _x1_wh = self.mamba_wh_1(_x1_wh)

        # 3. Fuse
        _x0_hw = 0.5 * (
            _x0_hw[:, : x0_hw.shape[-2], :]
            + _x0_hw[:, x0_hw.shape[-2] :, :].flip(dims=(-2,))
        )
        _x1_hw = 0.5 * (
            _x1_hw[:, : x1_hw.shape[-2], :]
            + _x1_hw[:, x1_hw.shape[-2] :, :].flip(dims=(-2,))
        )
        _x0_wh = 0.5 * (
            _x0_wh[:, : x0_wh.shape[-2], :]
            + _x0_wh[:, x0_wh.shape[-2] :, :].flip(dims=(-2,))
        )
        _x1_wh = 0.5 * (
            _x1_wh[:, : x1_wh.shape[-2], :]
            + _x1_wh[:, x1_wh.shape[-2] :, :].flip(dims=(-2,))
        )

        # 4. LayerNorm
        _x0_hw = self.layer_norm(_x0_hw)
        _x1_hw = self.layer_norm(_x1_hw)
        _x0_wh = self.layer_norm(_x0_wh)
        _x1_wh = self.layer_norm(_x1_wh)

        # 5. Residual connections
        x0_hw = _x0_hw + x0_hw
        x1_hw = _x1_hw + x1_hw
        x0_wh = _x0_wh + x0_wh
        x1_wh = _x1_wh + x1_wh

        return x0_hw, x1_hw, x0_wh, x1_wh


class QuadConcatMambaLayer(nn.Module):
    def __init__(
        self,
        in_output_dim: int = 1024,
        inner_expansion: float = 2.0,
        conv_dim: int = 4,
        delta: int = 16,
        device: torch.device = None,
        dtype: torch.dtype = None,
        using_mamba2: bool = False,
    ):
        super(QuadConcatMambaLayer, self).__init__()

        self.in_output_dim: int = in_output_dim
        self.inner_expansion: float = inner_expansion
        self.conv_dim: int = conv_dim
        self.delta: int = delta
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.using_mamba2: bool = using_mamba2

        # # Four mamba layers for four directions
        # self.mamba_hw_forward = MambaLayer(
        #     in_output_dim=self.in_output_dim,
        #     inner_expansion=self.inner_expansion,
        #     conv_dim=self.conv_dim,
        #     delta=self.delta,
        #     device=self.device,
        #     dtype=self.dtype,
        #     using_mamba2=self.using_mamba2,
        # )
        # self.mamba_wh_forward = MambaLayer(
        #     in_output_dim=self.in_output_dim,
        #     inner_expansion=self.inner_expansion,
        #     conv_dim=self.conv_dim,
        #     delta=self.delta,
        #     device=self.device,
        #     dtype=self.dtype,
        #     using_mamba2=self.using_mamba2,
        # )
        # self.mamba_hw_backward = MambaLayer(
        #     in_output_dim=self.in_output_dim,
        #     inner_expansion=self.inner_expansion,
        #     conv_dim=self.conv_dim,
        #     delta=self.delta,
        #     device=self.device,
        #     dtype=self.dtype,
        #     using_mamba2=self.using_mamba2,
        # )
        # self.mamba_wh_backward = MambaLayer(
        #     in_output_dim=self.in_output_dim,
        #     inner_expansion=self.inner_expansion,
        #     conv_dim=self.conv_dim,
        #     delta=self.delta,
        #     device=self.device,
        #     dtype=self.dtype,
        #     using_mamba2=self.using_mamba2,
        # )

        # Two mambas
        self.mamba_forward = MambaLayer(
            in_output_dim=self.in_output_dim,
            inner_expansion=self.inner_expansion,
            conv_dim=self.conv_dim,
            delta=self.delta,
            device=self.device,
            dtype=self.dtype,
            using_mamba2=self.using_mamba2,
        )
        self.mamba_backward = MambaLayer(
            in_output_dim=self.in_output_dim,
            inner_expansion=self.inner_expansion,
            conv_dim=self.conv_dim,
            delta=self.delta,
            device=self.device,
            dtype=self.dtype,
            using_mamba2=self.using_mamba2,
        )

    def forward(
        self,
        x0_hw: torch.Tensor,
        x1_hw: torch.Tensor,
        x0_wh: torch.Tensor,
        x1_wh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # # 1. Quad directional scan
        # x_hw_f = torch.concat([x0_hw, x1_hw], dim=-2)
        # x_hw_b = x_hw_f.flip(dims=(-2,))
        # x_wh_f = torch.concat([x0_wh, x1_wh], dim=-2)
        # x_wh_b = x_wh_f.flip(dims=(-2,))

        # # 2. Enter mamba layers
        # x_hw_f = self.mamba_hw_forward(x_hw_f)
        # x_hw_b = self.mamba_hw_backward(x_hw_b).flip(dims=(-2,))
        # x_wh_f = self.mamba_wh_forward(x_wh_f)
        # x_wh_b = self.mamba_wh_backward(x_wh_b).flip(dims=(-2,))

        # # 3. Fuse
        # _x0_hw = x_hw_f[:, : x0_hw.shape[-2], :] + x_hw_b[:, : x0_hw.shape[-2], :]
        # _x1_hw = x_hw_f[:, x0_hw.shape[-2] :, :] + x_hw_b[:, x0_hw.shape[-2] :, :]
        # _x0_wh = x_wh_f[:, : x0_wh.shape[-2], :] + x_wh_b[:, : x0_wh.shape[-2], :]
        # _x1_wh = x_wh_f[:, x0_wh.shape[-2] :, :] + x_wh_b[:, x0_wh.shape[-2] :, :]

        # 1. Forward and backward sequences
        x_forward = torch.concat([x0_hw, x1_hw, x0_wh, x1_wh], dim=-2)
        x_backward = x_forward.flip(dims=(-2,))

        # 2. Enter mamba layers
        x_fused = (
            self.mamba_forward(x_forward)
            + self.mamba_backward(x_backward).flip(dims=(-2,))
        ) / 2

        # 3. Defuse
        x0_hw, x1_hw, x0_wh, x1_wh = x_fused.chunk(4, dim=-2)

        return x0_hw, x1_hw, x0_wh, x1_wh


class MambaEncoderLayer(nn.Module):
    def __init__(
        self,
        in_output_dim: int = 1024,
        inner_expansion: float = 2.0,
        conv_dim: int = 4,
        delta: int = 16,
        device: torch.device = None,
        dtype: torch.dtype = None,
        using_mamba2: bool = False,
        aggregation_size: int = 4,
    ):
        super(MambaEncoderLayer, self).__init__()
        self.in_output_dim: int = in_output_dim
        self.inner_expansion: float = inner_expansion
        self.conv_dim: int = conv_dim
        self.delta: int = delta
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.using_mamba2: bool = using_mamba2
        self.aggregation_size: int = aggregation_size

        self.mamba = QuadConcatMambaLayer(
            in_output_dim=self.in_output_dim,
            inner_expansion=self.inner_expansion,
            conv_dim=self.conv_dim,
            delta=self.delta,
            device=self.device,
            dtype=self.dtype,
            using_mamba2=self.using_mamba2,
        )

        self.layer_norm = LayerNorm2d(self.in_output_dim)

        self.downsample_conv = nn.Conv2d(
            in_channels=self.in_output_dim,
            out_channels=self.in_output_dim,
            kernel_size=self.aggregation_size,
            stride=self.aggregation_size,
        )

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, H0, W0 = x0.shape
        _, _, H1, W1 = x1.shape
        # 1. Aggregation
        x0 = self.downsample_conv(x0)
        x1 = self.downsample_conv(x1)

        x0_shape = x0.shape[2:]  # H, W
        x1_shape = x1.shape[2:]  # H, W

        # 2. Layer norm
        x0 = self.layer_norm(x0)
        x1 = self.layer_norm(x1)

        # 3. Rearrange features into (B, H*W, C) and (B, W*H, C), two directions
        # Same effect as following, but a bit more efficient
        # x0_hw = rearrange(x0, "b c h w -> b (h w) c")
        # x1_hw = rearrange(x1, "b c h w -> b (h w) c")
        # x0_wh = rearrange(x0, "b c h w -> b (w h) c")
        # x1_wh = rearrange(x1, "b c h w -> b (w h) c")
        x0_hw = x0.flatten(2).transpose(1, 2)
        x1_hw = x1.flatten(2).transpose(1, 2)
        x0_wh = x0.permute(0, 2, 3, 1).flatten(1, 2)
        x1_wh = x1.permute(0, 2, 3, 1).flatten(1, 2)

        # 4. Mamba forward, including another two directions, which is the flip of the original two directions
        x0_hw, x1_hw, x0_wh, x1_wh = self.mamba(x0_hw, x1_hw, x0_wh, x1_wh)

        # 5. Rearrange features back to (B, C, H, W)
        # Same effect as following, but a bit more efficient
        # x0_hw = rearrange(x0_hw, "b (h w) c -> b c h w", h=x0_shape[0], w=x0_shape[1])
        # x1_hw = rearrange(x1_hw, "b (h w) c -> b c h w", h=x1_shape[0], w=x1_shape[1])
        # x0_wh = rearrange(x0_wh, "b (w h) c -> b c h w", h=x0_shape[0], w=x0_shape[1])
        # x1_wh = rearrange(x1_wh, "b (w h) c -> b c h w", h=x1_shape[0], w=x1_shape[1])
        x0_hw = x0_hw.view(x0_hw.shape[0], x0_shape[0], x0_shape[1], -1).permute(
            0, 3, 1, 2
        )
        x1_hw = x1_hw.view(x1_hw.shape[0], x1_shape[0], x1_shape[1], -1).permute(
            0, 3, 1, 2
        )
        x0_wh = x0_wh.view(x0_wh.shape[0], x0_shape[1], x0_shape[0], -1).permute(
            0, 3, 2, 1
        )
        x1_wh = x1_wh.view(x1_wh.shape[0], x1_shape[1], x1_shape[0], -1).permute(
            0, 3, 2, 1
        )
        x0 = x0_hw + x0_wh
        x1 = x1_hw + x1_wh

        # 6. Back to original resolution
        x0 = F.interpolate(x0, size=(H0, W0), mode="bilinear", align_corners=False)
        x1 = F.interpolate(x1, size=(H1, W1), mode="bilinear", align_corners=False)

        return x0, x1


class MultiScaleMambaEncoder(nn.Module):
    def __init__(
        self,
        in_output_dim: int = 1024,
        inner_expansion: float = 2.0,
        conv_dim: int = 4,
        delta: int = 16,
        device: torch.device = None,
        dtype: torch.dtype = None,
        using_mamba2: bool = False,
        layer_types: Sequence[str] = ["self", "cross"],
        scales_selection: Sequence[int] = [1, 1, 1, 1],
    ):
        """
        Default constructor for MambaEncoder using "self attn." mamba layer and "cross attn." mamba layer.

        Args:
            in_output_dim (int, optional): Input and output dimension. Defaults to 1024.
            inner_expansion (float, optional): Expansion for inner dimension. Defaults to 2.0.
            conv_dim (int, optional): Local convolution dimension. Defaults to 4.
            device (torch.device, optional): Device. Defaults to None.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            using_mamba2 (bool, optional): Whether using Mamba2 or not. Defaults to False.
            layer_types (Sequence[str], optional): Layer types for all layers, option: ["self", "cross"]
            scales_selection (Sequence[int], optional): Scales selection for all layers, Default: [1, 1, 1, 1]
        """
        super(MultiScaleMambaEncoder, self).__init__()

        self.in_output_dim: int = in_output_dim
        self.inner_expansion: float = inner_expansion
        self.conv_dim: int = conv_dim
        self.delta: int = delta
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.using_mamba2: bool = using_mamba2
        self.layer_types: Sequence[str] = layer_types
        self.scales_selection: Sequence[int] = scales_selection[::-1]

        def self_encoding_layer_builder():
            return QuadDualInputMambaLayer(
                in_output_dim=self.in_output_dim,
                inner_expansion=self.inner_expansion,
                conv_dim=self.conv_dim,
                delta=self.delta,
                device=self.device,
                dtype=self.dtype,
                using_mamba2=self.using_mamba2,
            )

        def cross_encoding_layer_builder():
            return QuadConcatMambaLayer(
                in_output_dim=self.in_output_dim,
                inner_expansion=self.inner_expansion,
                conv_dim=self.conv_dim,
                delta=self.delta,
                device=self.device,
                dtype=self.dtype,
                using_mamba2=self.using_mamba2,
            )

        self.layers = []
        for layer_type in self.layer_types:
            if layer_type.lower() == "self":
                self.layers.append(self_encoding_layer_builder())
            elif layer_type.lower() == "cross":
                self.layers.append(cross_encoding_layer_builder())

        self.layers = nn.ModuleList(self.layers)

        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x0: List[torch.Tensor],
        x1: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward function for MambaEncoder, using bidirectional scan.
        Args:
            x0 (torch.Tensor): input multi-scale feature 0: S x [B x C x H x W]
            x1 (torch.Tensor): input multi-scale feature 1: S x [B x C x H x W]
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: output multi-scale feature 0: S x [B x C x H x W], output multi-scale feature 1: S x [B x C x H x W]
        """
        # Inverse feature sequence, from high semantic level to low semantic level
        x0, x1 = x0[::-1], x1[::-1]

        # Choose the scales according to scales_selection
        x0_chosen: List[torch.Tensor] = []
        x1_chosen: List[torch.Tensor] = []
        for idx, select in enumerate(self.scales_selection):
            if select:
                x0_chosen.append(x0[idx])
                x1_chosen.append(x1[idx])

        x0_shape = [i.shape[2:] for i in x0_chosen]  # H, W of each scale
        x1_shape = [i.shape[2:] for i in x1_chosen]  # H, W of each scale
        x0_length = [i[0] * i[1] for i in x0_shape]  # HW of each scale
        x1_length = [i[0] * i[1] for i in x1_shape]  # HW of each scale

        for i, layer in enumerate(self.layers):
            # flatten in two directions into sequence
            # S x [B x C x H x W] -> S x [B x HW or WH x C] -> [SB x HW or WH x C]
            x0_hw = torch.concat(
                [rearrange(i, "b c h w -> b (h w) c") for i in x0_chosen], dim=1
            )
            x1_hw = torch.concat(
                [rearrange(i, "b c h w -> b (h w) c") for i in x1_chosen], dim=1
            )
            x0_wh = torch.concat(
                [rearrange(i, "b c h w -> b (w h) c") for i in x0_chosen], dim=1
            )
            x1_wh = torch.concat(
                [rearrange(i, "b c h w -> b (w h) c") for i in x1_chosen], dim=1
            )

            # mamba layer
            x0_hw, x1_hw, x0_wh, x1_wh = layer(
                x0_hw=x0_hw, x1_hw=x1_hw, x0_wh=x0_wh, x1_wh=x1_wh
            )

            # unflatten into 2d
            # [SB x HW or WH x C] -> S x [B x HW or WH x C]
            x0_hw = torch.split(x0_hw, x0_length, dim=1)
            x1_hw = torch.split(x1_hw, x1_length, dim=1)
            x0_wh = torch.split(x0_wh, x0_length, dim=1)
            x1_wh = torch.split(x1_wh, x1_length, dim=1)
            # S x [B x HW or WH x C] -> S x [B x C x H x W]
            x0_hw = [
                rearrange(
                    x, "b (h w) c -> b c h w", h=x0_shape[idx][0], w=x0_shape[idx][1]
                )
                for idx, x in enumerate(x0_hw)
            ]
            x1_hw = [
                rearrange(
                    x, "b (h w) c -> b c h w", h=x1_shape[idx][0], w=x1_shape[idx][1]
                )
                for idx, x in enumerate(x1_hw)
            ]
            x0_wh = [
                rearrange(
                    x, "b (w h) c -> b c h w", h=x0_shape[idx][0], w=x0_shape[idx][1]
                )
                for idx, x in enumerate(x0_wh)
            ]
            x1_wh = [
                rearrange(
                    x, "b (w h) c -> b c h w", h=x1_shape[idx][0], w=x1_shape[idx][1]
                )
                for idx, x in enumerate(x1_wh)
            ]
            # Fuse as S x [B x C x H x W]
            x0_chosen = [0.5 * (x0_hw[i] + x0_wh[i]) for i in range(len(x0_chosen))]
            x1_chosen = [0.5 * (x1_hw[i] + x1_wh[i]) for i in range(len(x1_chosen))]

        # Replace the original features with the chosen scales
        idx_chosen = 0
        for idx, select in enumerate(self.scales_selection):
            if select:
                x0[idx] = x0_chosen[idx_chosen]
                x1[idx] = x1_chosen[idx_chosen]
                idx_chosen += 1

        # Inverse back the feature sequence
        x0, x1 = x0[::-1], x1[::-1]

        return x0, x1
