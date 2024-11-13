import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple
from einops.einops import rearrange
from typing import List
from src.mamba.MambaEncoder import MambaLayer
from src.utils.misc import LayerNorm2d


def mean_2d(x: torch.Tensor):
    """
    Get mean of a 2D partial of 4D tensor
    Args:
        x (torch.Tensor): B x C x H x W
    Returns:
        (torch.Tensor): B x C
    """
    return rearrange(x, "b c h w -> b c (h w)").mean(dim=-1)


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

        # self.layers_forward = nn.ModuleList(
        #     [
        #         MambaLayer(
        #             in_output_dim=in_output_dim,
        #             inner_expansion=inner_expansion,
        #             conv_dim=conv_dim,
        #             delta=delta,
        #             using_mamba2=using_mamba2,
        #         )
        #         for _ in range(num_layers)
        #     ]
        # )
        # self.layers_backward = nn.ModuleList(
        #     [
        #         MambaLayer(
        #             in_output_dim=in_output_dim,
        #             inner_expansion=inner_expansion,
        #             conv_dim=conv_dim,
        #             delta=delta,
        #             using_mamba2=using_mamba2,
        #         )
        #         for _ in range(num_layers)
        #     ]
        # )
        # self.layer_norms = nn.ModuleList(
        #     [nn.LayerNorm(in_output_dim) for _ in range(num_layers)]
        # )

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * in_output_dim, in_output_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_output_dim, int(in_output_dim // 2), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_output_dim // 2), 3, kernel_size=3, padding=1),
            LayerNorm2d(3),
        )

        # self.hidden_state_extract_param = nn.Parameter(torch.zeros(1, 1, in_output_dim))

        # Initialize weights
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for m in self.conv_offset.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

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
        # feat0_shape = feat0_window.shape[2:]  # H, W
        # feat0_length = feat0_shape[0] * feat0_shape[1]
        # feat1_shape = feat1_window.shape[2:]  # H, W
        # for i in range(self.num_layers):
        #     # Rearrange features into (M, H*W, C)
        #     feat0_hw = rearrange(feat0_window, "m c h w -> m (h w) c")
        #     feat1_hw = rearrange(feat1_window, "m c h w -> m (h w) c")

        #     # Concat hidden state with both features
        #     input_forward = torch.cat(
        #         [
        #             hidden_state,
        #             feat0_hw,
        #             feat1_hw,
        #             self.hidden_state_extract_param.expand(hidden_state.size(0), 1, -1),
        #         ],
        #         dim=1,
        #     )  # (M, 1+2*HW+1, C)
        #     input_backward = torch.cat(
        #         [
        #             hidden_state,
        #             feat1_hw.flip(-1),
        #             feat0_hw.flip(-1),
        #             self.hidden_state_extract_param.expand(hidden_state.size(0), 1, -1),
        #         ],
        #         dim=1,
        #     )  # (M, 1+2*HW+1, C)

        #     # Mamba
        #     output_forward = self.layers_forward[i](
        #         self.layer_norms[i](input_forward)
        #     )
        #     output_backward = self.layers_backward[i](
        #         self.layer_norms[i](input_backward)
        #     )
            

        #     # Update hidden state
        #     hidden_state = 0.5 * (output_forward[:, -1, :] + output_backward[:, -1, :]).unsqueeze(1)
            
        #     # Fuse both directions
        #     output = input_forward[]

        #     # Rearrange features back to (M, C, H, W)
        #     feat0_window = rearrange(
        #         output[:, 1 : 1 + feat0_length, :],
        #         "m (h w) c -> m c h w",
        #         h=feat0_shape[0],
        #         w=feat0_shape[1],
        #     )
        #     feat1_window = rearrange(
        #         output[:, 1 + feat0_length : -1, :],
        #         "m (h w) c -> m c h w",
        #         h=feat1_shape[0],
        #         w=feat1_shape[1],
        #     )

        # Decode feature into coord offset using conv
        concat_feat = torch.cat([feat0_window, feat1_window], dim=1)  # (M, 2*C, H, W)
        offset_window = self.conv_offset(concat_feat)  # (M, 3, H, W)
        weight = offset_window[:, 2:3, :, :]  # (M, 1, H, W)

        # Get mean of offset window
        offset = mean_2d(offset_window[:, 0:2, :, :] * weight)

        # # Sample offset of center pixel of each window using bilinear sampling
        # grid = torch.zeros(
        #     offset_window.size(0), 1, 1, 2, device=offset_window.device
        # )  # (M, 1, 1, 2)
        # offset = (
        #     F.grid_sample(
        #         offset_window,
        #         grid,  # (M, 1, 1, 2)
        #         mode="bilinear",
        #         align_corners=False,
        #     )
        #     .squeeze(-1)
        #     .squeeze(-1)
        # )  # (M, 2)

        return offset, hidden_state
