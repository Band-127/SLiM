import torch
from torch import nn
from typing import List
from timm.models.layers import trunc_normal_


class CrossPixelRefinement(nn.Module):
    def __init__(self, kernel_size: int):
        super(CrossPixelRefinement, self).__init__()
        self.kernel_size = kernel_size

        self.conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=1),
            nn.Conv2d(
                8,
                8,
                kernel_size=(self.kernel_size, 1),
                padding=(self.kernel_size // 2, 0),
            ),
            nn.Conv2d(
                8,
                8,
                kernel_size=(1, self.kernel_size),
                padding=(0, self.kernel_size // 2),
            ),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(8, 2, kernel_size=1),
        )

        # Initialize
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    trunc_normal_(m.weight, std=0.002)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, data: dict):
        coord_0 = data["fine_coord_0"]  # [N, 2]
        coord_1 = data["fine_coord_1"]  # [N, 2]
        batch_size = data["batch_size"]  # B
        hw0_f = data["hw0_f"]  # H, W
        b_idx_it = data["b_idx_it"]  # N
        fine_scale = data["fine_scale"]
        scale0 = (
            data["scale0"][b_idx_it] * fine_scale if "scale0" in data else fine_scale
        )  # N, 2
        scale1 = (
            data["scale1"][b_idx_it] * fine_scale if "scale1" in data else fine_scale
        )  # N, 2

        # 1. Create a grid for coordinates, detach the gradient
        with torch.no_grad():
            grid_coord = torch.zeros(
                batch_size, 2, hw0_f[0], hw0_f[1], device=coord_0.device
            )  # B, C, H, W; C=2
            indices = (coord_0 / scale0 - 0.5).round().long()
            grid_coord[b_idx_it, :, indices[:, 1], indices[:, 0]] = (
                coord_1 / scale1
            ).detach()

        # 2. Enter small conv with residual connection
        grid_coord = self.conv(grid_coord) + grid_coord

        # 3. Extract the refined coordinates using indices
        refined_coord = grid_coord[b_idx_it, :, indices[:, 1], indices[:, 0]] * scale1

        return refined_coord
