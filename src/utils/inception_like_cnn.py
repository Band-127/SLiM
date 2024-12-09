import torch
import torch.nn as nn
from src.utils.misc import LayerNorm2d


class InceptionLikeCNN(nn.Module):
    def __init__(self, in_output_dim, kernel_size=7):
        super(InceptionLikeCNN, self).__init__()
        self.in_output_dim = in_output_dim
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) // 2)

        self.branch_1_dim = int(self.in_output_dim // 3)
        self.branch_2_dim = int(self.in_output_dim // 3)
        self.branch_3_dim = int(
            self.in_output_dim - self.branch_1_dim - self.branch_2_dim
        )

        self.branch_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_output_dim,
                out_channels=self.branch_1_dim,
                kernel_size=1,
                padding=0,
            ),
            LayerNorm2d(self.branch_1_dim),
            nn.Conv2d(
                in_channels=self.branch_1_dim,
                out_channels=self.branch_1_dim,
                kernel_size=(1, self.kernel_size),
                padding=(0, self.padding),
            ),
            LayerNorm2d(self.branch_1_dim),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.branch_1_dim,
                out_channels=self.branch_1_dim,
                kernel_size=(self.kernel_size, 1),
                padding=(self.padding, 0),
            ),
            LayerNorm2d(self.branch_1_dim),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.branch_1_dim,
                out_channels=self.branch_1_dim,
                kernel_size=(1, self.kernel_size),
                padding=(0, self.padding),
            ),
            LayerNorm2d(self.branch_1_dim),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.branch_1_dim,
                out_channels=self.branch_1_dim,
                kernel_size=(self.kernel_size, 1),
                padding=(self.padding, 0),
            ),
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_output_dim,
                out_channels=self.branch_2_dim,
                kernel_size=1,
                padding=0,
            ),
            LayerNorm2d(self.branch_2_dim),
            nn.Conv2d(
                in_channels=self.branch_2_dim,
                out_channels=self.branch_2_dim,
                kernel_size=(self.kernel_size, 1),
                padding=(self.padding, 0),
            ),
            LayerNorm2d(self.branch_2_dim),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.branch_2_dim,
                out_channels=self.branch_2_dim,
                kernel_size=(1, self.kernel_size),
                padding=(0, self.padding),
            ),
        )
        self.branch_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_output_dim,
                out_channels=self.branch_3_dim,
                kernel_size=1,
                padding=0,
            ),
            LayerNorm2d(self.branch_3_dim),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
        )

        self.output_activation = nn.ReLU()

    def forward(self, x):
        return self.output_activation(
            torch.concat(
                [
                    self.branch_1(x),
                    self.branch_2(x),
                    self.branch_3(x),
                ],
                dim=1,
            )
        )
