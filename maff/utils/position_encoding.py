import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Sequence


class MultiScaleSinePositionalEncoding(nn.Module):
    @torch.no_grad
    def __init__(
        self,
        d_model: int,
        max_hw: int,
        scales: Sequence[float],
        dtype: torch.dtype = torch.float32,
    ):
        """
        Default Constructor for multi-scale sinusoidal positional encoding, alike 3D PE, make sure features in all scales are transform into same channel number before input.
        Args:
            d_model (int): Typically the largest dimensions of features in all scale, e.g. for ResNet18: 512
            max_hw (int): the largest length of both side of image input. MAX(maximum_height, maximum_width)
            scales (Sequence[int]): Scales of features, e.g. for ResNet18: [0.5, 0.25, 0.125, 0.0625]
            dtype (torch.dtype): Data type, must be in float type
        """
        super(MultiScaleSinePositionalEncoding, self).__init__()

        self.d_model = d_model
        self.max_hw = max_hw
        self.scales = scales
        self.dtype = dtype
        self.num_scales = len(scales)

        pos_x = torch.arange(self.max_hw, dtype=self.dtype)
        pos_y = torch.arange(self.max_hw, dtype=self.dtype)
        pos_z = torch.arange(self.num_scales, dtype=self.dtype)

        d_pos_embeddings = int(np.ceil(self.d_model / 3))
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, d_pos_embeddings, 2, dtype=self.dtype)
                / d_pos_embeddings
            )
        )

        hw_pos_emb = torch.zeros(
            size=(self.max_hw, self.max_hw, 6 * inv_freq.shape[0]), dtype=self.dtype
        )  # H, W, C

        for i in range(self.num_scales):
            temp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
            temp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
            temp_z = pos_z[i] * inv_freq

            sin_x = torch.sin(temp_x).unsqueeze(0)
            sin_y = torch.sin(temp_y).unsqueeze(1)
            sin_z = torch.sin(temp_z).unsqueeze(0).unsqueeze(0)
            cos_x = torch.cos(temp_x).unsqueeze(0)
            cos_y = torch.cos(temp_y).unsqueeze(1)
            cos_z = torch.cos(temp_z).unsqueeze(0).unsqueeze(0)

            hw_pos_emb[:, :, 0::6] = sin_x
            hw_pos_emb[:, :, 1::6] = cos_x
            hw_pos_emb[:, :, 2::6] = sin_y
            hw_pos_emb[:, :, 3::6] = cos_y
            hw_pos_emb[:, :, 4::6] = sin_z
            hw_pos_emb[:, :, 5::6] = cos_z

            # Crop the exceeding channels
            hw_pos_emb_cropped = hw_pos_emb[:, :, 0 : self.d_model]

            # H, W, C -> C, H, W
            hw_pos_emb_cropped = hw_pos_emb_cropped.swapaxes(0, -1).swapaxes(-1, -2)

            # Interpolate into each scales
            hw_pos_emb_rescaled = F.interpolate(
                hw_pos_emb_cropped.unsqueeze(0),
                scale_factor=(scales[i], scales[i]),
            ).squeeze(0)

            self.register_buffer(
                f"all_scales_pos_emb_{i}", hw_pos_emb_rescaled, persistent=False
            )

    def forward(self, x: Sequence[torch.Tensor]):
        """
        Add PE into multi_scale_input. Make sure the scaling and number of channels is same as when you construct this PE
        Args:
            x (Sequence[[torch.Tensor]): Multi-scale input: Scales x tensor[B x C x H x W], make sure features in all scales are transform into same channel number before input.
        """
        for s, single_scale in enumerate(x):
            batch_size = single_scale.shape[0]
            for b in range(batch_size):
                single_scale[b] = single_scale[b] + getattr(
                    self, f"all_scales_pos_emb_{s}"
                )

        return x


class DualMultiScaleSinePositionalEncoding(nn.Module):
    @torch.no_grad
    def __init__(
        self,
        d_model: int,
        max_hw: int,
        scales: Sequence[float],
        dtype: torch.dtype = torch.float32,
    ):
        """
        Default Constructor for dual input multi-scale sinusoidal positional encoding, alike 3D PE, make sure features in all scales are transform into same channel number before input.
        Args:
            d_model (int): Typically the largest dimensions of features in all scale, e.g. for ResNet18: 512
            max_hw (int): the largest length of both side of image input. MAX(maximum_height, maximum_width)
            scales (Sequence[int]): Scales of features, e.g. for ResNet18: [0.5, 0.25, 0.125, 0.0625]
            dtype (torch.dtype): Data type, must be in float type
        """
        super(DualMultiScaleSinePositionalEncoding, self).__init__()

        self.d_model = d_model
        self.max_hw = max_hw
        self.scales = scales
        self.dtype = dtype
        self.num_scales = len(scales)

        # build multi-scale position embedding map
        self.all_scales_pos_emb = []  # scale, C, H, W

        pos_x = torch.arange(self.max_hw, dtype=self.dtype)
        pos_y = torch.arange(self.max_hw, dtype=self.dtype)
        pos_z = torch.arange(2 * self.num_scales, dtype=self.dtype)

        d_pos_embeddings = int(np.ceil(self.d_model / 3))
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, d_pos_embeddings, 2, dtype=self.dtype)
                / d_pos_embeddings
            )
        )

        hw_pos_emb = torch.zeros(
            size=(self.max_hw, self.max_hw, 6 * inv_freq.shape[0]), dtype=self.dtype
        )  # H, W, C

        for input in range(2):
            for i in range(self.num_scales):
                temp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
                temp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
                temp_z = pos_z[i + input * self.num_scales] * inv_freq

                sin_x = torch.sin(temp_x).unsqueeze(0)
                sin_y = torch.sin(temp_y).unsqueeze(1)
                sin_z = torch.sin(temp_z).unsqueeze(0).unsqueeze(0)
                cos_x = torch.cos(temp_x).unsqueeze(0)
                cos_y = torch.cos(temp_y).unsqueeze(1)
                cos_z = torch.cos(temp_z).unsqueeze(0).unsqueeze(0)

                hw_pos_emb[:, :, 0::6] = sin_x
                hw_pos_emb[:, :, 1::6] = cos_x
                hw_pos_emb[:, :, 2::6] = sin_y
                hw_pos_emb[:, :, 3::6] = cos_y
                hw_pos_emb[:, :, 4::6] = sin_z
                hw_pos_emb[:, :, 5::6] = cos_z

                # Crop the exceeding channels
                hw_pos_emb_cropped = hw_pos_emb[:, :, 0 : self.d_model]

                # H, W, C -> C, H, W
                hw_pos_emb_cropped = hw_pos_emb_cropped.swapaxes(0, -1).swapaxes(-1, -2)

                # Interpolate into each scales
                hw_pos_emb_rescaled = F.interpolate(
                    hw_pos_emb_cropped.unsqueeze(0),
                    scale_factor=(scales[i], scales[i]),
                ).squeeze(0)

                self.register_buffer(
                    f"all_scales_pos_emb_{i + input * self.num_scales}",
                    hw_pos_emb_rescaled,
                    persistent=False,
                )

    def forward(self, x1: Sequence[torch.Tensor], x2: Sequence[torch.Tensor]):
        """
        Add PE into multi_scale_input. Make sure the scaling and number of channels is same as when you construct this PE
        Args:
            x1 (Sequence[torch.Tensor]): Multi-scale input 1: Scales x tensor[B x C x H x W], make sure features in all scales are transform into same channel number before input.
            x2 (Sequence[torch.Tensor]): Multi-scale input 2: Scales x tensor[B x C x H x W], make sure features in all scales are transform into same channel number before input.
        """
        for s, single_scale in enumerate(x1):
            batch_size = single_scale.shape[0]
            for b in range(batch_size):
                single_scale[b] = single_scale[b] + getattr(
                    self, f"all_scales_pos_emb_{s}"
                )

        for s, single_scale in enumerate(x2):
            batch_size = single_scale.shape[0]
            for b in range(batch_size):
                single_scale[b] = single_scale[b] + getattr(
                    self, f"all_scales_pos_emb_{s + self.num_scales}"
                )
        return x1, x2

    def get_position_encodings(self):
        """
        Return position encodings for all scales
        """
        encodings = []
        for i in range(2 * self.num_scales):
            encodings.append(getattr(self, f"all_scales_pos_emb_{i}"))
        return encodings
