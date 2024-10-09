import torch
from torch import nn
from torch.nn import functional as F
from yacs.config import CfgNode as CN
from einops.einops import rearrange
import kornia.geometry.subpix.dsnt as dsnt
from kornia.utils.grid import create_meshgrid

from maff.backbone import build_backbone
from maff.mamba.MambaEncoder import MultiScaleMambaEncoder
from maff.transformer.transformer import LocalFeatureTransformer
from maff.utils.position_encoding import DualMultiScaleSinePositionalEncoding
from maff.utils.channel_alignment import ChannelAlignment
from maff.utils.conf_mask_head import ConfMaskHead
from maff.utils.pixel_shuffle_head import PixelShuffleHead
from maff.utils.fine_refinement import FineRefinement


class MAFF_v2(nn.Module):
    def __init__(self, config: CN):
        super(MAFF_v2, self).__init__()

        self.config = config

        self.dtype = torch.float32 if config["DTYPE"] == "float32" else "float64"
        self.d_model = config["DIMENSION"]
        self.scales_selection = config["SCALES_SELECTION"]
        self.coarse_scale_idx = config["COARSE_SCALE_IDX"]
        self.fine_scale_idx = config["FINE_SCALE_IDX"]
        self.coarse_scale = config["COARSE_SCALE"]
        self.fine_scale = config["FINE_SCALE"]
        self.coarse_match_thres = config["COARSE_MATCHING"]["THRESHOLD"]
        self.max_coarse_matches = config["COARSE_MATCHING"]["MAX_MATCHES"]
        self.debug = config["DEBUG"]

        # some refinement configurations
        self.pixel_shuffle_refinement = config["PIXEL_SHUFFLE_REFINEMENT"]
        self.fine_refinement = config["FINE_REFINEMENT"]

        # 1. Pyramid feature backbone
        self.feature_backbone = build_backbone(config["BACKBONE"])

        # 2. Feature channel alignment
        self.feature_channel_alignment = (
            ChannelAlignment(
                d_model_input=config["BACKBONE"]["LAYER_DIMS"],
                d_model_output=self.d_model,
                dtype=self.dtype,
            )
            if "FPN" not in config["BACKBONE"]["BACKBONE_TYPE"]
            else None
        )

        # 3. PE
        self.pe = DualMultiScaleSinePositionalEncoding(
            d_model=self.d_model,
            max_hw=config["BACKBONE"]["INPUT_SIZE"],
            scales=[1 / i for i in config["BACKBONE"]["RESOLUTION"]],
            dtype=self.dtype,
        )

        # 4. Mamba fusion encoder or Transformer fusion encoder
        if config["FUSION_TYPE"] == "mamba":
            self.fusion_encoder = MultiScaleMambaEncoder(
                in_output_dim=self.d_model,
                inner_expansion=config["MAMBA_FUSION"]["INNER_EXPANSION"],
                conv_dim=config["MAMBA_FUSION"]["CONV_DIM"],
                delta=config["MAMBA_FUSION"]["DELTA"],
                dtype=self.dtype,
                layer_types=config["MAMBA_FUSION"]["LAYER_TYPES"],
                using_mamba2=config["MAMBA_FUSION"]["USING_MAMBA2"],
                scales_selection=self.scales_selection,
            )
        elif config["FUSION_TYPE"] == "transformer":
            self.fusion_encoder = LocalFeatureTransformer(
                config=config["TRANSFORMER_FUSION"]
            )
        elif config["FUSION_TYPE"] is None:
            self.fusion_encoder = None

        # 5. Fine Refinement
        self.fine_refinement_encoder = (
            FineRefinement(
                in_output_dim=self.d_model,
                num_layers=config["FINE_REFINEMENT_MODEL"]["NUM_LAYER"],
                inner_expansion=config["FINE_REFINEMENT_MODEL"]["INNER_EXPANSION"],
                conv_dim=config["FINE_REFINEMENT_MODEL"]["CONV_DIM"],
                delta=config["FINE_REFINEMENT_MODEL"]["DELTA"],
                using_mamba2=config["FINE_REFINEMENT_MODEL"]["USING_MAMBA2"],
            )
            if self.fine_refinement
            else None
        )

        # 6. Pixel shuffle head
        self.pixel_shuffle_head = (
            PixelShuffleHead(self.d_model, config["FINE_SCALE"])
            if self.pixel_shuffle_refinement
            else None
        )

        # 8. Confidence mask head
        self.conf_mask_head = ConfMaskHead(self.d_model)

    def forward(self, data):
        pass
