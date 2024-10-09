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


class MAFF(nn.Module):
    def __init__(self, config: CN):
        super(MAFF, self).__init__()

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

    def forward(self, data: dict, training: bool = False):
        """
        Forward function of MAFF
        Input:
            data (dict): {
                "image0": (torch.Tensor): (B, 1, H, W)
                "image1": (torch.Tensor): (B, 1, H, W)
                "mask0": (torch.Tensor, optional): (B, H, W): '0' indicates a padded position
                "mask1": (torch.Tensor, optional): (B, H, W): '0' indicates a padded position
                "scale0": (torch.Tensor, optional): (B, 2), Megadepth only, absolute scale from input size to original size
                "scale1": (torch.Tensor, optional): (B, 2), Megadepth only, absolute scale from input size to original size

                (Training only)
                "spv_b_ids" (torch.Tensor): (M)
                "spv_i_ids" (torch.Tensor): (M)
                "spv_j_ids" (torch.Tensor): (M)
            }
        Output:
            data (dict): {
                "batch_size": (int)
                "hw0_i": (torch.Size): Hi0, Wi0: input shape 0
                "hw1_i": (torch.Size): Hi1, Wi1: input shape 1

                "feat0_c": (torch.Tensor): (B, Lc0, C): coarse feature 0
                "feat1_c": (torch.Tensor): (B, Lc1, C): coarse feature 1
                "feat0_f": (torch.Tensor): (B, Lf0, C): fine feature 0
                "feat1_f": (torch.Tensor): (B, Lf1, C): fine feature 1
                "hw0_c": (torch.Size): Hc0, Wc0: coarse feature shape 0
                "hw1_c": (torch.Size): Hc1, Wc1: coarse feature shape 1
                "coarse_scale" (float): hw0_i / hw0_c
                "hw0_f": (torch.Size): Hf0, Wf0: fine feature shape 0
                "hw1_f": (torch.Size): Hf1, Wf1: fine feature shape 1
                "fine_scale" (float): hw0_i / hw0_f
                "conf_mask0": (torch.Tensor): (B, Lc0) coarse confidence mask, used to refine coarse match in similarity matrix
                "conf_mask1": (torch.Tensor): (B, Lc1) coarse confidence mask, used to refine coarse match in similarity matrix

                "sim_matrix": (torch.Tensor): (B, Lc0, Lc1) coarse similarity matrix formed in coarse matching
                "num_matches": (int): number of matches
                "b_idx_c": (torch.Tensor): (M) coarse match index, batch, in coarse coordinate
                "i_idx_c": (torch.Tensor): (M) coarse match index, in image0, in coarse coordinate
                "j_idx_c": (torch.Tensor): (M) coarse match index, in image1, in coarse coordinate
                "coarse_coord_0": (torch.Tensor): (M, 2) coarse matched coords in image 0, in absolute coordinate
                "coarse_coord_1": (torch.Tensor): (M, 2) coarse matched coords in image 1, in absolute coordinate

                "fine_coord_0": (torch.Tensor): (M, 2) fine matched coords in image 0, in absolute coordinate
                "fine_coord_1": (torch.Tensor): (M, 2) fine matched coords in image 1, in absolute coordinate
                "sim_matrix_f": (torch.Tensor): (M, W, W) fine similarity matrix formed in fine matching
                "coord_offset_f": (torch.Tensor): (M, 2) fine matched coords offset(normalized) in image 1
                "std": (torch.Tensor): (M)
            }
        Args:
            data (dict): input data
            training (bool, optional): training/testing. Defaults to False.
        """
        data.update(
            {
                "batch_size": data["image0"].shape[0],
                "hw0_i": data["image0"].shape[2:],
                "hw1_i": data["image1"].shape[2:],
            }
        )

        # 1. Feature extraction
        x0, x1 = (
            self.feature_backbone(data["image0"]),
            self.feature_backbone(data["image1"]),
        )  # S, [B x C x H x W]
        mask0, mask1 = (
            data["mask0"].flatten(-2),
            data["mask1"].flatten(-2),
        )  # Flattened, [B x L]

        # 2. Feature channel alignment
        if self.feature_channel_alignment is not None:
            x0, x1 = (
                self.feature_channel_alignment(x0),
                self.feature_channel_alignment(x1),
            )  # S, [B x C x H x W]

        # 3. Position Encoding
        if self.fusion_encoder is not None:
            x0, x1 = self.pe(x0, x1)  # S x [B x C x H x W]

            # 4. Mamba, S x [B x C x H x W] -> S x [B x C x H x W]
            x0, x1 = self.fusion_encoder(x0, x1)

        data.update(
            {
                "feat0_c": x0[self.coarse_scale_idx],  # B, C, Hc0, Wc0
                "feat1_c": x1[self.coarse_scale_idx],  # B, C, Hc1, Wc1
                "feat0_f": x0[self.fine_scale_idx],  # B, C, Hf0, Wf0
                "feat1_f": x1[self.fine_scale_idx],  # B, C, Hf1, Wf1
                "hw0_c": x0[self.coarse_scale_idx].shape[2:],
                "hw1_c": x1[self.coarse_scale_idx].shape[2:],
                "hw0_f": x0[self.fine_scale_idx].shape[2:],
                "hw1_f": x1[self.fine_scale_idx].shape[2:],
                "coarse_scale": self.coarse_scale,
                "fine_scale": self.fine_scale,
            }
        )

        # 5. Mask refinement
        conf_mask0 = self.conf_mask_head(data["feat0_c"])
        conf_mask1 = self.conf_mask_head(data["feat1_c"])

        data.update({"conf_mask0": conf_mask0, "conf_mask1": conf_mask1})

        # 6. Feature Matching(both coarse and fine)
        self._feature_matching(
            data,
            data["feat0_c"],  # B, C, Hc0, Wc0
            data["feat1_c"],  # B, C, Hc1, Wc1
            mask0,  # B, Lc0
            mask1,  # B, Lc1
            data["conf_mask0"],  # B, Lc0
            data["conf_mask1"],  # B, Lc1
            training,
        )

    def _feature_matching(
        self,
        data: dict,
        feat0_c: torch.Tensor,
        feat1_c: torch.Tensor,
        mask0: torch.Tensor = None,
        mask1: torch.Tensor = None,
        conf_mask0: torch.Tensor = None,
        conf_mask1: torch.Tensor = None,
        training: bool = False,
    ):
        """
        Feature Matching using full correlation, and getting the coordinate of the best match

        Args:
            data (dict): data batch
            feat0_c (torch.Tensor): [B, C, Hc0, Wc0]
            feat1_c (torch.Tensor): [B, C, Hc1, Wc1]
            mask0 (torch.Tensor, optional): [B, Lc0]. Defaults to None.
            mask1 (torch.Tensor, optional): [B, Lc1]. Defaults to None.
            conf_mask0 (torch.Tensor, optional): [B, L0]. Confidence mask generated by mask_refinement. Defaults to None.
            conf_mask1 (torch.Tensor, optional): [B, L1]. Confidence mask generated by mask_refinement. Defaults to None.
            training (bool, optional): training/testing. Defaults to False.
        """
        # 1. Full Correlation
        # Flatten
        feat0_c = rearrange(feat0_c, "b c h w -> b (h w) c")  # B, Lc0, C
        feat1_c = rearrange(feat1_c, "b c h w -> b (h w) c")  # B, Lc1, C

        # Normalize
        feat0_c, feat1_c = map(lambda x: x / x.shape[-1] ** 0.5, [feat0_c, feat1_c])

        # Similarity matrix without dustbin
        sim_matrix = torch.einsum("blc,bsc->bls", feat0_c, feat1_c) / 0.1

        # Mask the area on similarity matrix where the mask==False into -inf
        if mask0 is not None and mask1 is not None:
            sim_matrix.masked_fill_(
                ~(mask0.unsqueeze(-1) * mask1.unsqueeze(-2)).bool(), -1e9
            )

        # Multiply the similarity matrix with confidence mask
        sim_matrix = sim_matrix * conf_mask0.unsqueeze(-1) * conf_mask1.unsqueeze(-2)

        # Update similarity matrix into data batch, used in coarse supervision
        data.update({"sim_matrix": sim_matrix})

        # 2. Get coarse coordinates
        self._get_coarse_coord(data=data, training=training)

        # 3. Fine Matching
        if self.pixel_shuffle_refinement:
            self._fine_matching_with_pixel_shuffle(data=data)
        else:
            self._fine_matching(data=data)

    @torch.no_grad()
    def _get_coarse_coord(self, data: dict, training: bool = False):
        # If not in training, generate coarse matches using similarity matrix
        if not training:
            # # Get coarse matches > threshold, using dual softmax
            # coarse_match = torch.argwhere(
            #     conf_matrix / conf_matrix.max() > self.coarse_match_thres
            # )  # M, 3(B, I, J)
            # del conf_matrix
            # torch.cuda.empty_cache()

            sim_matrix = data["sim_matrix"]
            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
            # Get top max_coarse_matches matches with highest confidence
            top_k = min(self.max_coarse_matches, conf_matrix.numel())
            flat_conf = conf_matrix.view(-1)
            top_k_values, top_k_indices = torch.topk(flat_conf, k=top_k)

            # Apply threshold
            valid_matches = top_k_values > (conf_matrix.max() * self.coarse_match_thres)
            coarse_match = top_k_indices[valid_matches]

            # Manually calculate original indices
            b = coarse_match // (conf_matrix.shape[1] * conf_matrix.shape[2])
            residual = coarse_match % (conf_matrix.shape[1] * conf_matrix.shape[2])
            i = residual // conf_matrix.shape[2]
            j = residual % conf_matrix.shape[2]

            coarse_match = torch.stack([b, i, j], dim=-1)

            del conf_matrix  # Free memory

            b_idx_c = coarse_match[:, 0]
            i_idx_c = coarse_match[:, 1]
            j_idx_c = coarse_match[:, 2]
        # If in training, use all groundtruth coarse matches to supervise all fine matches
        else:
            b_idx_c = data["spv_b_ids"]
            i_idx_c = data["spv_i_ids"]
            j_idx_c = data["spv_j_ids"]

        # Match indices -> coordinates
        scale0 = (
            self.coarse_scale * data["scale0"][b_idx_c]
            if "scale0" in data
            else self.coarse_scale
        )
        scale1 = (
            self.coarse_scale * data["scale1"][b_idx_c]
            if "scale1" in data
            else self.coarse_scale
        )
        coarse_coord_0 = (
            torch.stack(
                (
                    (i_idx_c % data["hw0_c"][1]),
                    (i_idx_c // data["hw0_c"][1]),
                ),
                dim=1,
            )
            * scale0
        )
        coarse_coord_1 = (
            torch.stack(
                (
                    (j_idx_c % data["hw1_c"][1]),
                    (j_idx_c // data["hw1_c"][1]),
                ),
                dim=1,
            )
            * scale1
        )

        data.update(
            {
                "b_idx_c": b_idx_c,  # in coarse coordinate
                "i_idx_c": i_idx_c,  # in coarse coordinate
                "j_idx_c": j_idx_c,  # in coarse coordinate
                "coarse_coord_0": coarse_coord_0,  # in absolute coordinate, at the top left cornerof coarse window
                "coarse_coord_1": coarse_coord_1,  # in absolute coordinate, at the top left corner of coarse window
                "num_matches": b_idx_c.shape[0],
            }
        )

    def _fine_matching(self, data: dict):
        feat0 = data["feat0_f"]
        feat1 = data["feat1_f"]
        b_idx_c = data["b_idx_c"]
        coarse_coord_0 = data["coarse_coord_0"]
        coarse_coord_1 = data["coarse_coord_1"]
        coarse_scale = int(data["coarse_scale"])
        fine_scale = int(data["fine_scale"])
        absolute_coarse_scale0 = (
            coarse_scale * data["scale0"][b_idx_c] if "scale0" in data else coarse_scale
        )
        absolute_coarse_scale1 = (
            coarse_scale * data["scale1"][b_idx_c] if "scale1" in data else coarse_scale
        )
        C = feat0.shape[-1]
        window_size = int(coarse_scale // fine_scale)  # window size
        window_radius = window_size / 2  # window radius

        # no coarse match
        if b_idx_c.shape[0] == 0:
            with torch.no_grad():
                data.update(
                    {
                        "fine_coord_0": torch.zeros(
                            size=(0, 2), dtype=torch.float32, device=feat0.device
                        ),
                        "fine_coord_1": torch.zeros(
                            size=(0, 2), dtype=torch.float32, device=feat0.device
                        ),
                        "expec_f": torch.zeros(
                            size=(0, 2), dtype=torch.float32, device=feat0.device
                        ),
                        "std": torch.zeros(
                            size=(0, 1), dtype=torch.float32, device=feat0.device
                        ),
                    }
                )
            return

        fine_topleft0 = coarse_coord_0 / absolute_coarse_scale0 * window_size
        fine_topleft1 = coarse_coord_1 / absolute_coarse_scale1 * window_size
        # 1. Pick a window of feature from fine feature map x0 using coarse matched coordinates
        feat0_window_picked = self._extract_feat_window(
            _feat=feat0,
            b_idx=b_idx_c,
            coord=fine_topleft0,
            window_size=window_size,
        )  # M, C, window_size, window_size

        # 2. Pick a window of feature from fine feature map x1 using coarse matched coordinates
        feat1_window_picked = self._extract_feat_window(
            _feat=feat1,
            b_idx=b_idx_c,
            coord=fine_topleft1,
            window_size=window_size,
        )  # M, C, window_size, window_size

        # 3. Fine Refinement
        if self.fine_refinement:
            feat0_window_picked, feat1_window_picked = self.fine_refinement_encoder(
                feat0_window_picked, feat1_window_picked
            )

        # 4. Pick the center feature from fine feature map x0 using coarse matched coordinates
        # feat0_picked = feat0_window_picked[
        #     :, :, int(window_radius), int(window_radius)
        # ]  # M, C
        feat0_picked = feat0_window_picked[:, :, 0, 0]  # M, C

        # 5. Correlation between both feature
        window_heatmap = torch.einsum("mc,mchw->mhw", feat0_picked, feat1_window_picked)
        # Divide by C**0.5 to control gradients:
        # When the feature dimension C is large, dot product results can be very high,
        # causing softmax gradients to approach zero.
        # This scaling factor helps keep gradients in a reasonable range, aiding model training.
        window_heatmap = self.softmax2d((window_heatmap / (C**0.5)))

        # 6. Compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(window_heatmap[None], True)[0]
        grid_normalized = create_meshgrid(
            window_size, window_size, True, window_heatmap.device
        ).reshape(1, -1, 2)  # [1, WW, 2]

        # 7. Compute absolute coordinates, (coarse coor + fine coor) * scale
        fine_coord_0 = data["coarse_coord_0"]
        scale1 = (
            fine_scale * data["scale1"][b_idx_c] if "scale1" in data else fine_scale
        )
        fine_coord_1 = (coords_normalized * window_radius * scale1) + data[
            "coarse_coord_1"
        ]

        # 8. Compute std over <x, y> (used in loss)
        var = (
            torch.sum(
                grid_normalized**2 * window_heatmap.view(-1, window_size**2, 1), dim=1
            )
            - coords_normalized**2
        )  # [M, 2]
        std = torch.sum(
            torch.sqrt(torch.clamp(var, min=1e-10)), -1
        )  # [M]  clamp needed for numerical stability

        data.update(
            {
                "fine_coord_0": fine_coord_0,
                "fine_coord_1": fine_coord_1,
                "sim_matrix_f": window_heatmap,
                "coord_offset_f": coords_normalized,
                "std": std,
            }
        )

    def _fine_matching_with_pixel_shuffle(self, data: dict):
        feat0 = data["feat0_f"]
        feat1 = data["feat1_f"]
        b_idx_c = data["b_idx_c"]
        coarse_coord_0 = data["coarse_coord_0"]
        coarse_coord_1 = data["coarse_coord_1"]
        coarse_scale = int(data["coarse_scale"])
        fine_scale = int(data["fine_scale"])
        absolute_coarse_scale0 = (
            coarse_scale * data["scale0"][b_idx_c] if "scale0" in data else coarse_scale
        )
        absolute_coarse_scale1 = (
            coarse_scale * data["scale1"][b_idx_c] if "scale1" in data else coarse_scale
        )
        C = feat0.shape[-1]
        window_size = int(coarse_scale // fine_scale)  # window size

        # no coarse match
        if b_idx_c.shape[0] == 0:
            with torch.no_grad():
                data.update(
                    {
                        "fine_coord_0": torch.zeros(
                            size=(0, 2), dtype=torch.float32, device=feat0.device
                        ),
                        "fine_coord_1": torch.zeros(
                            size=(0, 2), dtype=torch.float32, device=feat0.device
                        ),
                        "expec_f": torch.zeros(
                            size=(0, 2), dtype=torch.float32, device=feat0.device
                        ),
                        "std": torch.zeros(
                            size=(0, 1), dtype=torch.float32, device=feat0.device
                        ),
                    }
                )
            return

        fine_topleft0 = coarse_coord_0 / absolute_coarse_scale0 * window_size
        fine_topleft1 = coarse_coord_1 / absolute_coarse_scale1 * window_size
        # 1. Pick a window of feature from both fine feature map using coarse match
        feat0_window_picked = self._extract_feat_window(
            _feat=feat0,
            b_idx=b_idx_c,
            coord=fine_topleft0,
            window_size=window_size,
        )  # M, C, window_size, window_size
        feat1_window_picked = self._extract_feat_window(
            _feat=feat1,
            b_idx=b_idx_c,
            coord=fine_topleft1,
            window_size=window_size,
        )  # M, C, window_size, window_size

        # 2. Fine Refinement
        if self.fine_refinement:
            feat0_window_picked, feat1_window_picked = self.fine_refinement_encoder(
                feat0_window_picked, feat1_window_picked
            )

        # 3. Pixel shuffle for both features
        # (M, C, H, W） -> (M, C, H*r, W*r)
        feat0_picked = self.pixel_shuffle_head(feat0_window_picked)
        # (M, C, H*r, W*r) -> (M, C) (center)
        r_half = coarse_scale / 2
        # feat0_picked = feat0_picked[:, :, int(r_half), int(r_half)]
        feat0_picked = feat0_picked[:, :, 0, 0]
        # (M, C, H, W） -> (M, C, H*r, W*r)
        feat1_picked = self.pixel_shuffle_head(feat1_window_picked)

        # 4. Correlation between both feature
        window_heatmap = torch.einsum("mc,mchw->mhw", feat0_picked, feat1_picked)
        # Divide by C**0.5 to control gradients:
        # When the feature dimension C is large, dot product results can be very high,
        # causing softmax gradients to approach zero.
        # This scaling factor helps keep gradients in a reasonable range, aiding model training.
        window_heatmap = self.softmax2d((window_heatmap / (C**0.5)))

        # 5. Compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(window_heatmap[None], True)[0]
        grid_normalized = create_meshgrid(
            coarse_scale, coarse_scale, True, window_heatmap.device
        ).reshape(1, -1, 2)  # [1, WW, 2]

        # 6. Compute absolute coordinates, coarse coor + fine coor * scale        
        fine_coord_0 = data["coarse_coord_0"]
        scale1 = data["scale1"][b_idx_c] if "scale1" in data else 1
        fine_coord_1 = (coords_normalized * r_half * scale1) + data[
            "coarse_coord_1"
        ]

        # 7. Compute std over <x, y> (used in loss)
        var = (
            torch.sum(
                grid_normalized**2 * window_heatmap.view(-1, coarse_scale**2, 1), dim=1
            )
            - coords_normalized**2
        )  # [M, 2]
        std = torch.sum(
            torch.sqrt(torch.clamp(var, min=1e-10)), -1
        )  # [M]  clamp needed for numerical stability

        data.update(
            {
                "fine_coord_0": fine_coord_0,
                "fine_coord_1": fine_coord_1,
                "sim_matrix_f": window_heatmap,
                "coord_offset_f": coords_normalized,
                "std": std,
            }
        )

    @staticmethod
    def softmax2d(x):
        # x: [M x H x W] -> [M x (H W)]
        h, w = x.shape[1:]
        x = x.view(x.shape[0], -1)
        x = torch.softmax(x, dim=1)
        x = x.view(x.shape[0], h, w)
        return x

    @staticmethod
    def _extract_feat_window(
        _feat: torch.Tensor,
        b_idx: torch.Tensor,
        coord: torch.Tensor,
        window_size: int,
    ):
        # Calculate row and column indices
        row_indices = coord[:, 1].long()
        col_indices = coord[:, 0].long()

        # Create grid indices
        row_offsets = torch.arange(window_size, device=_feat.device, dtype=torch.long)
        col_offsets = torch.arange(window_size, device=_feat.device, dtype=torch.long)

        row_indices = row_indices.unsqueeze(1) + row_offsets.unsqueeze(0)
        col_indices = col_indices.unsqueeze(1) + col_offsets.unsqueeze(0)

        # Extract windows using advanced indexing
        windows = _feat[
            b_idx[:, None, None], :, row_indices[:, :, None], col_indices[:, None, :]
        ]

        # Rearrange to [B, C, H, W]
        windows = rearrange(windows, "b h w c -> b c h w")

        return windows

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith("maff."):
                state_dict[k.replace("maff.", "", 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
