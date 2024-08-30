from typing import Tuple, Sequence
import torch
from torch import nn
from mamba_ssm import Mamba, Mamba2


class MambaLayer(nn.Module):
    def __init__(
        self,
        in_output_dim: int = 1024,
        inner_expansion: float = 2.0,
        conv_dim: int = 4,
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
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.using_mamba2: bool = using_mamba2

        # vanilla Mamba builder
        self.mamba_base = Mamba2 if self.using_mamba2 else Mamba

        self.mamba_layer = self.mamba_base(
            d_model=in_output_dim,
            d_state=64 if self.using_mamba2 else 16,
            d_conv=4 if self.conv_dim is None else self.conv_dim,
            expand=inner_expansion,
            device=self.device,
            dtype=self.dtype,
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


class DualInputMambaLayer(nn.Module):
    def __init__(
        self,
        in_output_dim: int = 1024,
        inner_expansion: float = 2.0,
        conv_dim: int = 4,
        device: torch.device = None,
        dtype: torch.dtype = None,
        using_mamba2: bool = False,
    ):
        """
        Default constructor for DualInputMambaLayer using vanilla Mamba.

        input: x1, x2 -> (B, L1, in_output_dim), (B, L2, in_output_dim)
        (inner) h1, h2 -> (B, L1, in_output_dim * inner_expansion), (B, L2, in_output_dim * inner_expansion)
        output: y1, y2 -> (B, L1, in_output_dim), (B, L2, in_output_dim)

        Args:
            in_output_dim (int, optional): Input and output dimension. Defaults to 1024.
            inner_expansion (float, optional): Expansion for inner dimension. Defaults to 2.0.
            conv_dim (int, optional): Local convolution dimension. Defaults to 4.
            device (torch.device, optional): Device. Defaults to None.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            using_mamba2 (bool, optional): Whether using Mamba2 or not. Defaults to False.
        """
        super(DualInputMambaLayer, self).__init__()

        self.in_output_dim: int = in_output_dim
        self.inner_expansion: float = inner_expansion
        self.conv_dim: int = conv_dim
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.using_mamba2: bool = using_mamba2

        # vanilla Mamba builder
        self.mamba_base = Mamba2 if self.using_mamba2 else Mamba

        self.mamba_layer = self.mamba_base(
            d_model=in_output_dim,
            d_state=64 if self.using_mamba2 else 16,
            d_conv=4 if self.conv_dim is None else self.conv_dim,
            expand=inner_expansion,
            device=self.device,
            dtype=self.dtype,
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward function for DualInputMambaLayer.

        Args:
            x1 (torch.Tensor): input tensor 1(Batch, Sequence, Dimension)
            x2 (torch.Tensor): input tensor 2(Batch, Sequence, Dimension)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: output tensor 1(Batch, Sequence, Dimension), output tensor 2(Batch, Sequence, Dimension)
        """
        assert x1.shape[-1] == self.in_output_dim and x2.shape[-1] == self.in_output_dim

        x1 = self.mamba_layer(x1)
        x2 = self.mamba_layer(x2)

        return x1, x2


class ConcatMambaLayer(nn.Module):
    def __init__(
        self,
        in_output_dim: int = 1024,
        inner_expansion: float = 2.0,
        conv_dim: int = 4,
        device: torch.device = None,
        dtype: torch.dtype = None,
        using_mamba2: bool = False,
    ):
        """
        Default constructor for ConcatMambaLayer using vanilla Mamba.

        input: x1, x2 -> (B, L1, in_output_dim), (B, L2, in_output_dim)
        (inner) h1, h2 -> (B, L1, in_output_dim * inner_expansion), (B, L2, in_output_dim * inner_expansion)
        output: y1, y2 -> (B, L1, in_output_dim), (B, L2, in_output_dim)

        For Mamba:
            input: (x1, x2) -> (B, L1 + L2, in_output_dim)
            (inner) (h1, h2) -> (B, L1 + L2, in_output_dim * inner_expansion)
            output: (y1, y2) -> (B, L1 + L2, in_output_dim), (B, L2, in_output_dim)

        Args:
            in_output_dim (int, optional): Input and output dimension. Defaults to 1024.
            inner_expansion (float, optional): Expansion for inner dimension. Defaults to 2.0.
            conv_dim (int, optional): Local convolution dimension. Defaults to 4.
            device (torch.device, optional): Device. Defaults to None.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            using_mamba2 (bool, optional): Whether using Mamba2 or not. Defaults to False.
        """
        super(ConcatMambaLayer, self).__init__()

        self.in_output_dim: int = in_output_dim
        self.inner_expansion: float = inner_expansion
        self.conv_dim: int = conv_dim
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.using_mamba2: bool = using_mamba2

        self.mamba_layer = DualInputMambaLayer(
            in_output_dim=self.in_output_dim,
            inner_expansion=self.inner_expansion,
            conv_dim=self.conv_dim,
            device=self.device,
            dtype=self.dtype,
            using_mamba2=self.using_mamba2,
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward function for ConcatMambaLayer, using bidirectional scan.

        Args:
            x1 (torch.Tensor): input tensor 1(Batch, Sequence, Dimension)
            x2 (torch.Tensor): input tensor 2(Batch, Sequence, Dimension)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: output tensor 1(Batch, Sequence, Dimension), output tensor 2(Batch, Sequence, Dimension)
        """
        assert x1.shape[-1] == self.in_output_dim and x2.shape[-1] == self.in_output_dim

        # 1. Bidirectional scan
        x1_ = torch.concat(tensors=(x1, x2), dim=-2)
        x2_ = torch.concat(tensors=(x2, x1), dim=-2)

        # 2. Enter mamba layer
        x1_, x2_ = self.mamba_layer(x1_, x2_)

        # 3. Fuse both scanning direction into one and restore into two features
        L1 = x1.shape[-2]
        L2 = x2.shape[-2]
        x1 = 0.5 * (x1_[:, 0:L1, :] + x2_[:, L1:, :])
        x2 = 0.5 * (x1_[:, L2:, :] + x2_[:, 0:L2:, :])

        return x2, x2


class MambaEncoder(nn.Module):
    def __init__(
        self,
        in_output_dim: int = 1024,
        inner_expansion: float = 2.0,
        conv_dim: int = 4,
        device: torch.device = None,
        dtype: torch.dtype = None,
        using_mamba2: bool = False,
        layer_types: Sequence[str] = ["self", "cross"],
    ):
        """
        Default constructor for MambaEncoder using "self attn." mamba layer and "cross attn." mamba layer.

        input: x1, x2 -> (B, L1, in_output_dim), (B, L2, in_output_dim)
        (inner) h1, h2 -> (B, L1, in_output_dim * inner_expansion), (B, L2, in_output_dim * inner_expansion)
        output: y1, y2 -> (B, L1, in_output_dim), (B, L2, in_output_dim)

        For Mamba:
            input: (x1, x2) -> (B, L1 + L2, in_output_dim)
            (inner) (h1, h2) -> (B, L1 + L2, in_output_dim * inner_expansion)
            output: (y1, y2) -> (B, L1 + L2, in_output_dim), (B, L2, in_output_dim)

        Args:
            in_output_dim (int, optional): Input and output dimension. Defaults to 1024.
            inner_expansion (float, optional): Expansion for inner dimension. Defaults to 2.0.
            conv_dim (int, optional): Local convolution dimension. Defaults to 4.
            device (torch.device, optional): Device. Defaults to None.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            using_mamba2 (bool, optional): Whether using Mamba2 or not. Defaults to False.
            layer_types (Sequence[str], optional): Layer types for all layers, option: ["self", "cross"]
        """
        super(MambaEncoder, self).__init__()

        self.in_output_dim: int = in_output_dim
        self.inner_expansion: float = inner_expansion
        self.conv_dim: int = conv_dim
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.using_mamba2: bool = using_mamba2
        self.layer_types: Sequence[str] = layer_types

        def self_encoding_layer_builder():
            return DualInputMambaLayer(
                in_output_dim=self.in_output_dim,
                inner_expansion=self.inner_expansion,
                conv_dim=self.conv_dim,
                device=self.device,
                dtype=self.dtype,
                using_mamba2=self.using_mamba2,
            )

        def cross_encoding_layer_builder():
            return ConcatMambaLayer(
                in_output_dim=self.in_output_dim,
                inner_expansion=self.inner_expansion,
                conv_dim=self.conv_dim,
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

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward function for MambaEncoder, using bidirectional scan.
        Args:
            x1 (torch.Tensor): input tensor 1(Batch, Sequence, Dimension)
            x2 (torch.Tensor): input tensor 2(Batch, Sequence, Dimension)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: output tensor 1(Batch, Sequence, Dimension), output tensor 2(Batch, Sequence, Dimension)
        """
        assert x1.shape[-1] == self.in_output_dim and x2.shape[-1] == self.in_output_dim

        for i, layer in enumerate(self.layers):
            x1, x2 = layer(x1, x2)

        return x1, x2
