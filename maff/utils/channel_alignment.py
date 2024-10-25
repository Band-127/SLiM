import torch
from torch import nn
from typing import Sequence


class ChannelAlignment(nn.Module):
    def __init__(
        self,
        d_model_input: Sequence[int],
        d_model_output: int,
        dtype: torch.dtype = torch.float32,
    ):
        super(ChannelAlignment, self).__init__()

        self.d_model_input = d_model_input
        self.d_model_output = d_model_output

        self.conv_layer = nn.ModuleList(
            [
                nn.Conv2d(
                    d,
                    self.d_model_output,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    dtype=dtype,
                )
                for d in self.d_model_input
            ]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.d_model_output) for _ in self.d_model_input]
        )

        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """
        Align channels for all feature map in input tensor.
        Args:
            x (Sequence[torch.Tensor]): Input tensor, Scale x [Batch x Channel x H x W]

        Returns:
            Sequence[torch.Tensor]: Output tensor, Scale x [Batch x Channel x H x W]
        """

        # For each scale
        for i, s in enumerate(x):
            # Align
            s = self.conv_layer[i](s)
            x[i] = self.layer_norms[i](s.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return x
