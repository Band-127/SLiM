import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple
from einops.einops import rearrange
from typing import List
from src.mamba.MambaEncoder import MambaLayer


class RecurrentRefinementUnit(nn.Module):
    def __init__(
        self,
        in_output_dim: int,
        num_layers: int,
        inner_expansion: int,
        conv_dim: int,
        delta: int,
        using_mamba2: bool,
    ) -> None:
        super(RecurrentRefinementUnit, self).__init__()
        self.num_layers = num_layers
        self.inner_expansion = inner_expansion
        self.conv_dim = conv_dim
        self.delta = delta
        self.using_mamba2 = using_mamba2

        self.layers = nn.ModuleList(
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

        self.conv_offset = nn.Sequential(
            nn.Conv2d(in_output_dim, 2 * in_output_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * in_output_dim, 2, kernel_size=1),
        )

        # Initialize weights
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for m in self.conv_offset.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="relu"
                    )

    def forward(
        self,
        feat0_window: torch.Tensor,
        feat1_window: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward function of Recurrent Refinement Unit
        Args:
            feat0_window (torch.Tensor): (M, C, H, W)
            feat1_window (torch.Tensor): (M, C, H, W)
            hidden_state (torch.Tensor): (M, 1, C)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: output coord offset (M, 2), updated hidden state (M, 1, C)
        """
        feat0_shape = feat0_window.shape[2:]  # H, W
        feat0_length = feat0_shape[0] * feat0_shape[1]
        feat1_shape = feat1_window.shape[2:]  # H, W
        zeros = torch.zeros_like(hidden_state)
        for i in range(self.num_layers):
            # Rearrange features into (M, H*W, C)
            feat0_hw = rearrange(feat0_window, "m c h w -> m (h w) c")
            feat1_hw = rearrange(feat1_window, "m c h w -> m (h w) c")

            # Concat hidden state with both features
            input = torch.cat(
                [hidden_state, feat0_hw, feat1_hw, zeros], dim=1
            )  # (M, 1+2*HW+1, C)

            # Mamba forward
            output = self.layers[i](self.layer_norms[i](input))

            # Update hidden state
            hidden_state = output[:, -1, :].unsqueeze(1)

            if i != self.num_layers - 1:
                # Rearrange features back to (M, C, H, W)
                feat0_window = rearrange(
                    output[:, 1 : 1 + feat0_length, :],
                    "m (h w) c -> m c h w",
                    h=feat0_shape[0],
                    w=feat0_shape[1],
                )
                feat1_window = rearrange(
                    output[:, 1 + feat0_length : -1, :],
                    "m (h w) c -> m c h w",
                    h=feat1_shape[0],
                    w=feat1_shape[1],
                )
            else:
                # Rearrange features back to (M, C, H, W)
                feat1_window = rearrange(
                    output[:, 1 + feat0_length : -1, :],
                    "m (h w) c -> m c h w",
                    h=feat1_shape[0],
                    w=feat1_shape[1],
                )

        # Decode feature into coord offset using conv
        offset = self.conv_offset(feat1_window)  # (M, 2, H, W)
        # Get mean offset of each window
        offset = offset.mean(dim=(-1, -2))  # (M, 2)

        return offset, hidden_state
