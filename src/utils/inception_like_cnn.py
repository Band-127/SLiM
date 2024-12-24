import torch
import torch.nn as nn
from src.utils.misc import LayerNorm2d


class InceptionNeXt(nn.Module):
    def __init__(self, in_output_dim, kernel_size=7, split_ratio=8):
        super(InceptionNeXt, self).__init__()
        self.in_output_dim = in_output_dim
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) // 2)

        self.split_dim = int(self.in_output_dim // split_ratio)
        self.split_idices = (
            self.split_dim,
            self.split_dim,
            self.split_dim,
            self.in_output_dim - 3 * self.split_dim,
        )

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
            LayerNorm2d(self.in_output_dim),
            nn.Conv2d(
                in_channels=self.in_output_dim,
                out_channels=4 * self.in_output_dim,
                kernel_size=1,
                padding=0,
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=4 * self.in_output_dim,
                out_channels=self.in_output_dim,
                kernel_size=1,
                padding=0,
            ),
        )

    def forward(self, x):
        x_hw, x_h, x_w, x_id = torch.split(x, self.split_idices, dim=1)
        return self.mlp(
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

        # fut1 = torch.jit.fork(self.branch_1, x)
        # fut2 = torch.jit.fork(self.branch_2, x)
        # fut3 = torch.jit.fork(self.branch_3, x)

        # branch1_out = torch.jit.wait(fut1)
        # branch2_out = torch.jit.wait(fut2)
        # branch3_out = torch.jit.wait(fut3)

        # return self.output_activation(
        #     torch.concat([branch1_out, branch2_out, branch3_out], dim=1)
        # )
