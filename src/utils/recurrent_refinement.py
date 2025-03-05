import torch
import torch.nn as nn
from torch.nn import functional as F
import kornia.geometry.subpix.dsnt as dsnt
from typing import Tuple
from collections import OrderedDict
from einops.einops import rearrange
from timm.models.layers import trunc_normal_

from src.utils.misc import LayerNorm2d, Alpha


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(ConvGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv_gates = nn.Conv2d(
            hidden_dim + input_dim, 2 * hidden_dim, 5, padding=2
        )
        self.conv_update = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 5, padding=2)

    def forward(self, h, x):
        combined = torch.cat([h, x], dim=1)  # [h, x]

        combined_conv = self.conv_gates(combined)
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)  # r
        update_gate = torch.sigmoid(beta)  # z

        combined_update = torch.cat([reset_gate * h, x], dim=1)
        h_tilde = torch.tanh(self.conv_update(combined_update))  # q

        return (1 - update_gate) * h + update_gate * h_tilde  # h = (1 - z) * h + z * q


class ConvContextBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=5):
        super(ConvContextBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Conv+MLP
        self.dwconv_mlp = nn.Sequential(
            nn.Conv2d(
                input_dim,
                input_dim,
                kernel_size=self.kernel_size,
                padding=self.padding,
            ),
            LayerNorm2d(input_dim),
            nn.Conv2d(input_dim, 2 * output_dim, kernel_size=1),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(2 * output_dim, output_dim, kernel_size=1),
        )

    def forward(self, x):
        return self.dwconv_mlp(x)


class RecurrentRefinementUnit(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        lookup_radius: int,
        context: bool = True,
    ) -> None:
        super(RecurrentRefinementUnit, self).__init__()
        self.input_dim = input_dim
        self.input_divider = input_dim**0.5
        self.hidden_dim = hidden_dim
        self.lookup_window_size = lookup_radius * 2
        self.context = context

        # Feat -> flow feature encoder
        self.flow_encoder = (
            nn.Sequential(
                nn.Conv2d(
                    2 * self.input_dim,
                    2 * self.input_dim,
                    kernel_size=7,
                    padding=3,
                ),
                nn.Conv2d(
                    2 * self.input_dim,
                    2 * self.hidden_dim,
                    kernel_size=1,
                ),
                LayerNorm2d(2 * self.hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Conv2d(
                    2 * self.hidden_dim,
                    self.hidden_dim,
                    kernel_size=1,
                ),
            )
            if not context
            else None
        )

        self.context_network = (
            ConvContextBlock(
                self.hidden_dim + 2 * self.input_dim, self.hidden_dim, kernel_size=7
            )
            # ConvGRU(hidden_dim=self.hidden_dim, input_dim=2 * self.input_dim)
            if context
            else None
        )

        # Flow feature -> offset decoder
        self.conv_offset = nn.Sequential(
            # DWConv
            nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=3,
                padding=1,
                groups=self.hidden_dim,
            ),
            # MLP
            nn.Conv2d(self.hidden_dim, 2 * self.hidden_dim, kernel_size=1),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(2 * self.hidden_dim, 3, kernel_size=1),
            LayerNorm2d(3),
        )

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
            nn.init.constant_(self.conv_offset[-1].weight, 0.001)

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
            hidden_state (torch.Tensor): (M, C, H, W)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: output coord offset (M, 2), updated hidden state (M, C, H, W)
        """

        """
        ################ Direct way ################
        
        # 1. Center of feat0_window
        grid = torch.zeros(feat0_window.shape[0], 1, 1, 2, device=feat0_window.device)
        center_feat0 = (
            F.grid_sample(feat0_window, grid, mode="bilinear", align_corners=False)[
                :, :, 0, 0
            ]
            / self.input_divider
        )  # (M, C)

        # 2. Correlation
        corr = torch.einsum(
            "mc,mchw->mhw", center_feat0, feat1_window / self.input_divider
        )  # (M, H, W)
        h, w = corr.shape[-2:]
        corr = corr.view(corr.shape[0], -1)
        corr = torch.softmax(corr, dim=1)
        corr = corr.view(corr.shape[0], h, w)

        # Offset
        spatial_expectation = (
            dsnt.spatial_expectation2d(corr[None], True)[0]
        )  # M, 2

        return spatial_expectation, hidden_state
        """
        

        # Input Encoding
        feat = torch.cat([feat0_window, feat1_window], dim=1)  # (M, 2*C, H, W)

        # Context injection
        if self.context:
            hidden_state = self.context_network(torch.cat([hidden_state, feat], dim=1))
            # hidden_state = self.context_network(h=hidden_state, x=feat)
        else:
            flow = self.flow_encoder(feat)  # (M, C, H, W)
            hidden_state = flow

        # Decode hidden state into coord offset using conv head
        offset_window = self.conv_offset(hidden_state)  # (M, 3, H, W)
        # Get sum of weighted offset window
        offset = (offset_window[:, 0:2, :, :] * offset_window[:, 2:3, :, :]).sum(
            dim=(2, 3)
        )

        return offset, hidden_state

    @torch.no_grad
    def initial_forward(self):
        for i in range(5):
            feat0_window = torch.zeros(10000, self.input_dim, 6, 6).to(
                self.conv_offset[0].weight.device
            )
            feat1_window = torch.zeros(10000, self.input_dim, 6, 6).to(
                self.conv_offset[0].weight.device
            )
            hidden_state = torch.zeros(10000, self.hidden_dim, 6, 6).to(
                self.conv_offset[0].weight.device
            )
            self.forward(feat0_window, feat1_window, hidden_state)
        torch.cuda.empty_cache()
