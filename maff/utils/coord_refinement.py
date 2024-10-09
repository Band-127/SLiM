import torch
import torch.nn as nn
from collections import OrderedDict
from mamba_ssm import Mamba, Mamba2


class BidirectionalMambaLayer(nn.Module):
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
        Default constructor for BidirectionalMambaLayer using vanilla Mamba.

        Args:
            in_output_dim (int, optional): Input and output dimension. Defaults to 1024.
            inner_expansion (float, optional): Expansion for inner dimension. Defaults to 2.0.
            conv_dim (int, optional): Local convolution dimension. Defaults to 4.
            device (torch.device, optional): Device. Defaults to None.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            using_mamba2 (bool, optional): Whether using Mamba2 or not. Defaults to False.
        """
        super(BidirectionalMambaLayer, self).__init__()

        self.in_output_dim: int = in_output_dim
        self.inner_expansion: float = inner_expansion
        self.conv_dim: int = conv_dim
        self.delta: int = delta
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.using_mamba2: bool = using_mamba2

        # vanilla Mamba builder
        self.mamba_base = Mamba2 if self.using_mamba2 else Mamba

        self.mamba_layer_forward = self.mamba_base(
            d_model=in_output_dim,
            d_state=16 if self.delta is None else self.delta,
            d_conv=4 if self.conv_dim is None else self.conv_dim,
            expand=inner_expansion,
            device=self.device,
            dtype=self.dtype,
        )

        self.mamba_layer_backward = self.mamba_base(
            d_model=in_output_dim,
            d_state=16 if self.delta is None else self.delta,
            d_conv=4 if self.conv_dim is None else self.conv_dim,
            expand=inner_expansion,
            device=self.device,
            dtype=self.dtype,
        )

        self.sigmoid = nn.Sigmoid()

        # LayerNorm
        self.layer_norm = nn.LayerNorm(in_output_dim)

    def forward(self, x) -> torch.Tensor:
        """
        Forward function for ConcatMambaLayer, using bidirectional scan.

        Args:
            x (torch.Tensor): input tensor(Batch, Sequence, Dimension)

        Returns:
            torch.Tensor: output tensor(Batch, Sequence, Dimension)
        """
        # 1. Bidirectional scan
        xf: torch.Tensor = x
        xb: torch.Tensor = xf.flip(dims=(-2,))

        # 2. Enter mamba layer
        delta_xf: torch.Tensor = self.sigmoid(self.mamba_layer_forward(xf))
        delta_xb: torch.Tensor = self.sigmoid(self.mamba_layer_backward(xb))

        # 3. Fuse
        delta_x = 0.5 * (delta_xf + delta_xb.flip(dims=(-2,)))

        # 4. Apply LayerNorm
        delta_x = self.layer_norm(delta_x)

        # 5. Restore into two features
        x = delta_x + x  # Residual connections

        return x


class CoordRefinementHead(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        inner_expansion: int = 2,
        conv_dim: int = 4,
        delta: int = 16,
        using_mamba2: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super(CoordRefinementHead, self).__init__()

        self.num_layers = num_layers
        self.inner_expansion = inner_expansion
        self.conv_dim = conv_dim
        self.delta = delta
        self.using_mamba2 = using_mamba2
        self.device = device
        self.dtype = dtype

        self.mamba_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"layer_{i}",
                        BidirectionalMambaLayer(
                            in_output_dim=2,
                            inner_expansion=self.inner_expansion,
                            conv_dim=self.conv_dim,
                            delta=self.delta,
                            using_mamba2=self.using_mamba2,
                        ),
                    )
                    for i in range(self.num_layers)
                ]
            )
        )

        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 0.01) # set weight to a small value to supress delta 
                    nn.init.constant_(m.bias, 0)

    def forward(self, coord: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        sorted_batch_unique = batch_idx.unique(sorted=True)
        for b_idx in sorted_batch_unique:
            coord[batch_idx == b_idx] = self.mamba_layer(
                coord[batch_idx == b_idx].unsqueeze(0)
            ).squeeze(0)
        return coord
