import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple
from collections import OrderedDict
from einops.einops import rearrange
from src.utils.misc import LayerNorm2d


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


class RecurrentRefinementUnit(nn.Module):
    def __init__(
        self,
        in_output_dim: int,
        gru: bool = True,
    ) -> None:
        super(RecurrentRefinementUnit, self).__init__()
        self.in_output_dim = in_output_dim

        self.conv_input = nn.Sequential(
            LayerNorm2d(in_output_dim * 2),
            nn.Conv2d(in_output_dim * 2, in_output_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_output_dim * 2, in_output_dim, kernel_size=3, padding=1),
        )

        self.gru = (
            ConvGRU(hidden_dim=in_output_dim, input_dim=in_output_dim) if gru else None
        )

        self.conv_offset = nn.Sequential(
            nn.Conv2d(in_output_dim, in_output_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_output_dim, in_output_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_output_dim, 2, kernel_size=1, padding=0),
            nn.BatchNorm2d(2),
        )

        # Initialize weights
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
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
        # Input decoding
        concat_feat = torch.cat([feat0_window, feat1_window], dim=1)  # (M, 2*C, H, W)
        concat_feat = self.conv_input(concat_feat)  # (M, C, H, W)

        # GRU
        if self.gru is not None:
            hidden_state = self.gru(h=hidden_state, x=concat_feat)
        else:
            hidden_state = concat_feat
        # Decode hidden state into coord offset using conv head
        offset_window = self.conv_offset(hidden_state)  # (M, 3, H, W)
        # # Get sum of weighted offset window
        # offset = (offset_window[:, 0:2, :, :] * offset_window[:, 2:3, :, :]).sum(
        #     dim=(2, 3)
        # )
        offset = offset_window.sum(dim=(2, 3))

        # # Sample offset from offset window
        # grid = torch.zeros(hidden_state.shape[0], 1, 1, 2, device=hidden_state.device)
        # offset = F.grid_sample(
        #     self.conv_offset(hidden_state),
        #     grid,
        #     mode="bilinear",
        #     align_corners=False,
        # )[:, :, 0, 0]  # M, 2

        return offset, hidden_state

    def initial_forward(self):
        for i in range(5):
            feat0_window = torch.randn(10, self.in_output_dim, 6, 6).to(
                self.conv_input[0].weight.device
            )
            feat1_window = torch.randn(10, self.in_output_dim, 6, 6).to(
                self.conv_input[0].weight.device
            )
            hidden_state = torch.zeros_like(feat0_window).to(
                self.conv_input[0].weight.device
            )
            self.forward(feat0_window, feat1_window, hidden_state)
        torch.cuda.empty_cache()
