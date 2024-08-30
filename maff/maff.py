import torch
from torch import nn
from torch.nn import functional as F
from yacs.config import CfgNode as CN
from einops.einops import rearrange
from typing import Sequence

from .backbone import build_backbone
from .mamba.MambaEncoder import MambaEncoder
from .transformer.transformer import LocalFeatureTransformer
from .utils.position_encoding import DualMultiScaleSinePositionalEncoding
from .utils.channel_alignment import ChannelAlignment


class MAFF(nn.Module):
    def __init__(self, config: CN):
        super(MAFF, self).__init__()

        self.config = config

        self.dtype = torch.float32 if config["DTYPE"] == "float32" else "float64"
        self.d_model = config["BACKBONE"]["LAYER_DIMS"][-1]
        self.scales_selection = config["SCALES_SELECTION"]
        self.coarse_scale_idx = config["COARSE_SCALE_IDX"]

        # 1. Pyramid feature backbone
        self.feature_backbone = build_backbone(config["BACKBONE"])
        # 2. Feature channel alignment
        self.feature_channel_alignment = ChannelAlignment(
            d_model_input=config["BACKBONE"]["LAYER_DIMS"],
            d_model_output=self.d_model,
            dtype=self.dtype,
        )
        # 3. PE
        self.pe = DualMultiScaleSinePositionalEncoding(
            d_model=self.d_model,
            max_hw=config["BACKBONE"]["INPUT_SIZE"],
            scales=[1 / i for i in config["BACKBONE"]["RESOLUTION"]],
            dtype=self.dtype,
        )
        # 4. Mamba fusion encoder or Transformer fusion encoder
        self.fusion_encoder = (
            MambaEncoder(
                in_output_dim=self.d_model,
                inner_expansion=config["MAMBA_FUSION"]["INNER_EXPANSION"],
                conv_dim=config["MAMBA_FUSION"]["CONV_DIM"],
                dtype=self.dtype,
                layer_types=config["MAMBA_FUSION"]["LAYER_TYPES"],
                using_mamba2=config["MAMBA_FUSION"]["USING_MAMBA2"]
            )
            if config["FUSION_TYPE"] == "mamba"
            else LocalFeatureTransformer(config=config["TRANSFORMER_FUSION"])
        )

    def forward(self, data: dict):
        """
        Forward function of MAFF
            data (dict): {
                'image0': (torch.Tensor): (B, 1, H, W)
                'image1': (torch.Tensor): (B, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (B, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (B, H, W)
            }

        Args:
            data (dict): input data
        """
        data.update(
            {
                "batch_size": data["image0"].shape[0],
                "hw0_i": data["image0"].shape[2:],
                "hw1_i": data["image1"].shape[2:],
            }
        )

        # 1. Feature extraction
        x1, x2 = (
            self.feature_backbone(data["image0"]),
            self.feature_backbone(data["image1"]),
        )  # Scales, [B x C x H x W]
        mask0, mask1 = (
            data["mask0"].flatten(-2),
            data["mask1"].flatten(-2),
        )  # Flattened, [N x L]

        # 2. Feature channel alignment
        x1, x2 = (
            self.feature_channel_alignment(x1),
            self.feature_channel_alignment(x2),
        )  # Scales, [B x C x H x W]

        # # 3. Position Encoding
        # x1, x2 = self.pe(x1, x2)  # S x [B x C x H x W]

        # 4. Feature selection
        new_x1 = []
        new_x2 = []
        for i, selection in enumerate(self.scales_selection):
            if selection == 1:
                new_x1.append(x1[i])
                new_x2.append(x2[i])
        x1 = new_x1
        x2 = new_x2
        data.update(
            {
                "hw0_c": x1[0].shape[2:],
                "hw1_c": x2[0].shape[2:],
            }
        )

        # 5. Flatten S x [B x C x H x W] -> [B x (HW) x (sum(C))]
        x1, x2, x1_length, x2_length = self.flatten(x1, x2)

        # 6. Mamba
        x1, x2 = self.fusion_encoder(x1, x2)

        # 7. Unflatten into S x [B x (HW) x C]
        x1, x2 = self.unflatten(x1, x2, x1_length, x2_length)

        # 8. Correlation / Feature Matching
        conf_matrix = self.feature_matching(data, x1[self.coarse_scale_idx], x2[self.coarse_scale_idx], mask0, mask1)

        return conf_matrix

    def flatten(self, x1: Sequence[torch.Tensor], x2: Sequence[torch.Tensor]):
        # [B x C x H x W] -> [B x (HW) x C]
        for i, scale in enumerate(x1):
            x1[i] = rearrange(scale, "b c h w -> b (h w) c")
        for i, scale in enumerate(x2):
            x2[i] = rearrange(scale, "b c h w -> b (h w) c")

        # S x [B x (HW) x C] -> [B x (HW) x (sum(C))]
        x1_length = [i.shape[1] for i in x1]
        x2_length = [i.shape[1] for i in x2]
        x1 = torch.concat(x1, dim=1)
        x2 = torch.concat(x2, dim=1)

        return x1, x2, x1_length, x2_length

    def unflatten(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x1_length: torch.Size,
        x2_length: torch.Size,
    ):
        # [B x (HW) x (sum(C))] -> S x [B x (HW) x C]
        x1 = torch.split(x1, split_size_or_sections=x1_length, dim=1)
        x2 = torch.split(x2, split_size_or_sections=x2_length, dim=1)

        return x1, x2

    def feature_matching(
        self,
        data: dict,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask1: torch.Tensor = None,
        mask2: torch.Tensor = None,
    ):
        """
        Feature Matching using dual-softmax

        Args:
            data (dict): data batch
            x1 (torch.Tensor): [B, L1, C]
            x2 (torch.Tensor): [B, L2, C]
            mask1 (torch.Tensor, optional): [B, L1]. Defaults to None.
            mask2 (torch.Tensor, optional): [B, L2]. Defaults to None.
        """
        # Normalize
        x1, x2 = map(lambda x: x / x.shape[-1] ** 0.5, [x1, x2])

        # Similarity matrix without dustbin
        sim_matrix = torch.einsum("blc,bsc->bls", x1, x2) / 0.1

        # Mask the area on similarity matrix where the mask==False into -inf
        if mask1 is not None and mask2 is not None:
            sim_matrix.masked_fill_(
                ~(mask1.unsqueeze(-1) * mask2.unsqueeze(-2)).bool(), -1e9
            )

        # Dual softmax
        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        # Update into data batch
        data.update({"conf_matrix": conf_matrix})

        return conf_matrix

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith("maff."):
                state_dict[k.replace("maff.", "", 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
