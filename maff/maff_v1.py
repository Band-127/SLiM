import time
import torch
from torch import nn
from torch.nn import functional as F
from yacs.config import CfgNode as CN
from einops.einops import rearrange
import kornia.geometry.subpix.dsnt as dsnt
from kornia.utils.grid import create_meshgrid

from maff.backbone import build_backbone
from maff.mamba.MambaEncoder import MultiScaleMambaEncoder
from maff.utils.position_encoding import DualMultiScaleSinePositionalEncoding
from maff.utils.channel_alignment import ChannelAlignment
from maff.utils.conf_mask_head import ConfHead
from maff.utils.pixel_shuffle_head import PixelShuffleHead
from maff.utils.any_input_identity import AnyInputIdentity
from maff.utils.coarse_encoder import CoarseEncoder
from maff.utils.fine_encoder import FineEncoder


class MAFF_v1(nn.Module):
    def __init__(self, config: CN):
        """
        Args:
            config (CN): root/model configuration
        """
        super(MAFF_v1, self).__init__()

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
        self.enable_fusion = config["ENABLE_FUSION"]
        self.disable_pe = config["DISABLE_PE"]
        self.pixel_shuffle_refinement = config["PIXEL_SHUFFLE_REFINEMENT"]

        # trainable_parameters
        self.coarse_temperature = nn.Parameter(torch.tensor(0.1))

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
            else nn.Identity()
        )

        # 3. PE
        self.pe = (
            DualMultiScaleSinePositionalEncoding(
                d_model=self.d_model,
                max_hw=config["BACKBONE"]["INPUT_SIZE"],
                scales=[1 / i for i in config["BACKBONE"]["RESOLUTION"]],
                dtype=self.dtype,
            )
            if not self.disable_pe
            else AnyInputIdentity()
        )

        # 4 Fusion
        self.fusion_encoder = (
            MultiScaleMambaEncoder(
                in_output_dim=self.d_model,
                inner_expansion=config["MAMBA_FUSION"]["INNER_EXPANSION"],
                conv_dim=config["MAMBA_FUSION"]["CONV_DIM"],
                delta=config["MAMBA_FUSION"]["DELTA"],
                dtype=self.dtype,
                layer_types=config["MAMBA_FUSION"]["LAYER_TYPES"],
                using_mamba2=config["MAMBA_FUSION"]["USING_MAMBA2"],
                scales_selection=self.scales_selection,
            )
            if self.enable_fusion
            else AnyInputIdentity()
        )

        # 5 Coarse encoder
        self.coarse_encoder = (
            CoarseEncoder(
                in_output_dim=self.d_model,
                num_layers=config["COARSE_ENCODER"]["NUM_LAYERS"],
                inner_expansion=config["COARSE_ENCODER"]["INNER_EXPANSION"],
                conv_dim=config["COARSE_ENCODER"]["CONV_DIM"],
                delta=config["COARSE_ENCODER"]["DELTA"],
                using_mamba2=config["COARSE_ENCODER"]["USING_MAMBA2"],
            )
            if config["COARSE_ENCODER"]["NUM_LAYERS"] > 0 and not self.enable_fusion
            else AnyInputIdentity()
        )

        # 6. Confidence score mask head, used to generate confidence/saliency score for coarse matching
        self.conf_mask_head = ConfHead(self.d_model)

        # 7. Fine encoder
        self.fine_encoder = (
            FineEncoder(
                in_output_dim=self.d_model,
                num_layers=config["FINE_ENCODER"]["NUM_LAYERS"],
                inner_expansion=config["FINE_ENCODER"]["INNER_EXPANSION"],
                conv_dim=config["FINE_ENCODER"]["CONV_DIM"],
                delta=config["FINE_ENCODER"]["DELTA"],
                using_mamba2=config["FINE_ENCODER"]["USING_MAMBA2"],
            )
            if config["FINE_ENCODER"]["NUM_LAYERS"] > 0
            else AnyInputIdentity()
        )

        # 8. Fine confidence score head
        self.fine_conf_head = ConfHead(self.d_model)

        # 8. Pixel shuffle
        self.pixel_shuffle_head = (
            PixelShuffleHead(self.d_model, self.fine_scale)
            if self.pixel_shuffle_refinement
            else nn.Identity()
        )
        self.scale_after_pixel_shuffle = (
            1 if self.pixel_shuffle_refinement else self.fine_scale
        )

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

                "batch_size": (int)
                "hw0_i": (torch.Size): Hi0, Wi0: input shape 0
                "hw1_i": (torch.Size): Hi1, Wi1: input shape 1
                "coarse_scale" (float): hw0_i / hw0_c
                "fine_scale" (float): hw0_i / hw0_f

                (Training only)
                "spv_b_ids" (torch.Tensor): (M)
                "spv_i_ids" (torch.Tensor): (M)
                "spv_j_ids" (torch.Tensor): (M)
            }
        Output:
            data (dict): {
                "feat0_all": (List[torch.Tensor]): S x [B, C, H, W]: feature pyramid of image 0
                "feat1_all": (List[torch.Tensor]): S x [B, C, H, W]: feature pyramid of image 1
                "feat0_c": (torch.Tensor): (B, C, Hc0, Wc0): coarse feature 0
                "feat1_c": (torch.Tensor): (B, C, Hc1, Wc1): coarse feature 1
                "hw0_c": (torch.Size): Hc0, Wc0: coarse feature shape 0
                "hw1_c": (torch.Size): Hc1, Wc1: coarse feature shape 1
                "conf_mask0": (torch.Tensor): (B, Lc0) coarse confidence mask, used to refine coarse match in similarity matrix
                "conf_mask1": (torch.Tensor): (B, Lc1) coarse confidence mask, used to refine coarse match in similarity matrix

                "sim_matrix": (torch.Tensor): (B, Lc0, Lc1) coarse similarity matrix formed in coarse matching
                "num_matches": (int): number of matches
                "b_idx_c": (torch.Tensor): (M) coarse match index, batch, in coarse coordinate
                "i_idx_c": (torch.Tensor): (M) coarse match index, in image0, in coarse coordinate
                "j_idx_c": (torch.Tensor): (M) coarse match index, in image1, in coarse coordinate
                "coarse_coord_0": (torch.Tensor): (M, 2) coarse matched coords in image 0, in absolute coordinate
                "coarse_coord_1": (torch.Tensor): (M, 2) coarse matched coords in image 1, in absolute coordinate

                "feat0_f": (torch.Tensor): (M, C, Hw, Ww): fine feature 0
                "feat1_f": (torch.Tensor): (M, C, Hw, Ww): fine feature 1

                "fine_coord_0": (torch.Tensor): (M, 2) fine matched coords in image 0, in absolute coordinate
                "fine_coord_1": (torch.Tensor): (M, 2) fine matched coords in image 1, in absolute coordinate
                "sim_matrix_f": (torch.Tensor): (M, W, W) fine similarity matrix formed in fine matching
                "coord_offset_f0": (torch.Tensor): (M, 2) fine matched coords offset(normalized) in image 0
                "coord_offset_f1": (torch.Tensor): (M, 2) fine matched coords offset(normalized) in image 1

                (Test only)
                "feat_extract_time": (float), in seconds
                "coarse_time": (float), in seconds
                "correlation_time": (float), in seconds
                "fine_time": (float), in seconds
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
                "coarse_scale": self.coarse_scale,
                "fine_scale": self.fine_scale,
            }
        )
        if training:
            return self._forward_train(data)
        else:
            return self._forward_test(data)

    @torch.no_grad()
    def _forward_test(self, data: dict):
        start_time = time.perf_counter()
        # 1. Feature extraction
        x0, x1 = (
            self.feature_backbone(data["image0"]),
            self.feature_backbone(data["image1"]),
        )  # S, [B x C x H x W], from fine to coarse, from shallow to deep
        mask0, mask1 = (
            (
                data["mask0"].flatten(-2),
                data["mask1"].flatten(-2),
            )
            if "mask0" in data
            else (None, None)
        )  # Flattened, [B x L]

        # 2. Feature channel alignment, align the channel dim of features
        x0, x1 = (
            self.feature_channel_alignment(x0),
            self.feature_channel_alignment(x1),
        )  # S, [B x C x H x W]
        feat_extraxt_end_time = time.perf_counter()

        # 3. PE
        x0, x1 = self.pe(x0, x1)  # S x [B x C x H x W]

        # 4. Feature Fusion
        x0, x1 = self.fusion_encoder(x0, x1)

        # 5. Coarse encoder
        coarse_x0, coarse_x1 = x0[self.coarse_scale_idx], x1[self.coarse_scale_idx] = (
            self.coarse_encoder(x0[self.coarse_scale_idx], x1[self.coarse_scale_idx])
        )

        # 6. Conf score mask generation
        conf_mask0 = self.conf_mask_head(coarse_x0)
        conf_mask1 = self.conf_mask_head(coarse_x1)
        coarse_end_time = time.perf_counter()

        # Update coarse features, shapes and conf mask
        data.update(
            {
                "feat0_all": x0,  # S x [B x C x H x W]
                "feat1_all": x1,  # S x [B x C x H x W]
                "feat0_c": coarse_x0,  # B, C, Hc0, Wc0
                "feat1_c": coarse_x1,  # B, C, Hc1, Wc1
                "hw0_c": coarse_x0.shape[2:],
                "hw1_c": coarse_x1.shape[2:],
                "conf_mask0": conf_mask0,
                "conf_mask1": conf_mask1,
            }
        )

        # 7. Coarse matching
        self._coarse_correlation(
            data=data,
            feat0_c=coarse_x0,
            feat1_c=coarse_x1,
            mask0=mask0,
            mask1=mask1,
            conf_mask0=conf_mask0,
            conf_mask1=conf_mask1,
        )  # (B x HW x HW)
        self._get_coarse_coord_test(data=data)  # Get coord from sim matrix
        correlation_end_time = time.perf_counter()

        # 8. Fine encoder
        fine_x0, fine_x1 = self._extract_fine_features(data=data)  # M x C x H x W
        fine_x0, fine_x1 = self.fine_encoder(fine_x0, fine_x1)
        data.update(
            {
                "feat0_f": fine_x0,  # M, C, H, W
                "feat1_f": fine_x1,  # M, C, H, W
            }
        )
        self._fine_matching(data=data)
        fine_end_time = time.perf_counter()

        data.update(
            {
                "feat_extract_time": (feat_extraxt_end_time - start_time)
                / data["batch_size"],
                "coarse_time": (coarse_end_time - feat_extraxt_end_time)
                / data["batch_size"],
                "correlation_time": (correlation_end_time - coarse_end_time)
                / data["batch_size"],
                "fine_scan_time": 0.0,
                "fine_time": (fine_end_time - correlation_end_time)
                / data["batch_size"],
            }
        )

    def _forward_train(self, data: dict):
        # 1. Feature extraction
        x0, x1 = (
            self.feature_backbone(data["image0"]),
            self.feature_backbone(data["image1"]),
        )  # S, [B x C x H x W], from fine to coarse, from shallow to deep
        mask0, mask1 = (
            (
                data["mask0"].flatten(-2),
                data["mask1"].flatten(-2),
            )
            if "mask0" in data
            else (None, None)
        )  # Flattened, [B x L]

        # 2. Feature channel alignment, align the channel dim of features
        x0, x1 = (
            self.feature_channel_alignment(x0),
            self.feature_channel_alignment(x1),
        )  # S, [B x C x H x W]

        # 3. PE
        x0, x1 = self.pe(x0, x1)  # S x [B x C x H x W]

        # 4. Feature Fusion
        x0, x1 = self.fusion_encoder(x0, x1)

        # 5. Coarse encoder
        coarse_x0, coarse_x1 = x0[self.coarse_scale_idx], x1[self.coarse_scale_idx] = (
            self.coarse_encoder(x0[self.coarse_scale_idx], x1[self.coarse_scale_idx])
        )

        # 6. Conf score mask generation
        conf_mask0 = self.conf_mask_head(coarse_x0)
        conf_mask1 = self.conf_mask_head(coarse_x1)

        # Update coarse features, shapes and conf mask
        data.update(
            {
                "feat0_all": x0,  # S x [B x C x H x W]
                "feat1_all": x1,  # S x [B x C x H x W]
                "feat0_c": coarse_x0,  # B, C, Hc0, Wc0
                "feat1_c": coarse_x1,  # B, C, Hc1, Wc1
                "hw0_c": coarse_x0.shape[2:],
                "hw1_c": coarse_x1.shape[2:],
                "conf_mask0": conf_mask0,
                "conf_mask1": conf_mask1,
            }
        )

        # 7. Coarse matching
        self._coarse_correlation(
            data=data,
            feat0_c=coarse_x0,
            feat1_c=coarse_x1,
            mask0=mask0,
            mask1=mask1,
            conf_mask0=conf_mask0,
            conf_mask1=conf_mask1,
        )  # (B x HW x HW)
        self._get_coarse_coord_train(data=data)  # Get all coord from gt

        # 8. Fine encoder
        fine_x0, fine_x1 = self._extract_fine_features(data=data)  # M x C x H x W
        fine_x0, fine_x1 = self.fine_encoder(fine_x0, fine_x1)
        data.update(
            {
                "feat0_f": fine_x0,  # M, C, H, W
                "feat1_f": fine_x1,  # M, C, H, W
            }
        )
        self._fine_matching(data=data)

    def _coarse_correlation(
        self,
        data: dict,
        feat0_c: torch.Tensor,
        feat1_c: torch.Tensor,
        mask0: torch.Tensor = None,
        mask1: torch.Tensor = None,
        conf_mask0: torch.Tensor = None,
        conf_mask1: torch.Tensor = None,
    ):
        # 1. Flatten
        feat0_c = rearrange(feat0_c, "b c h w -> b (h w) c")  # B, Lc0, C
        feat1_c = rearrange(feat1_c, "b c h w -> b (h w) c")  # B, Lc1, C

        # Divide by C**0.5 to control gradients:
        # When the feature dimension C is large, dot product results can be very high,
        # causing softmax gradients to approach zero.
        # This scaling factor helps keep gradients in a reasonable range, aiding model training.
        feat0_c, feat1_c = map(lambda x: x / x.shape[-1] ** 0.5, [feat0_c, feat1_c])

        # 2. Full correlation
        # Similarity matrix without dustbin, using a trainable temperature
        sim_matrix = (
            torch.einsum("blc,bsc->bls", feat0_c, feat1_c) / self.coarse_temperature
        )

        # 3. Multiply the similarity matrix with confidence mask
        sim_matrix = sim_matrix * conf_mask0.unsqueeze(-1) * conf_mask1.unsqueeze(-2)

        # 4. Mask the area on similarity matrix where the mask==False into -inf
        if mask0 is not None and mask1 is not None:
            sim_matrix.masked_fill_(
                ~(mask0.unsqueeze(-1) * mask1.unsqueeze(-2)).bool(), -1e9
            )

        # 5. Update sim matrix
        data.update({"sim_matrix": sim_matrix})

    @torch.no_grad()
    def _get_coarse_coord_train(self, data: dict):
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

    @torch.no_grad()
    def _get_coarse_coord_test(self, data: dict):
        conf_matrix = data["sim_matrix"]
        # No dual-softmax in optimized version
        conf_matrix = F.softmax(conf_matrix, 1) * F.softmax(conf_matrix, 2)
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
                "coarse_coord_0": coarse_coord_0,  # in absolute coordinate, at the top left corner of coarse window
                "coarse_coord_1": coarse_coord_1,  # in absolute coordinate, at the top left corner of coarse window
                "num_matches": b_idx_c.shape[0],
            }
        )

    def _extract_fine_features(self, data: dict):
        # 1. Fine feature
        feat0_fine = data["feat0_all"][self.fine_scale_idx]  # [B, C, H, W]
        feat1_fine = data["feat1_all"][self.fine_scale_idx]  # [B, C, H, W]

        # 2. Get index of coarse match(top left)
        coarse_scale = int(self.coarse_scale)
        fine_scale = int(self.fine_scale)
        b_idx_c = data["b_idx_c"]
        absolute_fine_scale0 = (
            fine_scale * data["scale0"][b_idx_c] if "scale0" in data else coarse_scale
        )
        absolute_fine_scale1 = (
            fine_scale * data["scale1"][b_idx_c] if "scale1" in data else coarse_scale
        )
        fine_idx_0 = (
            data["coarse_coord_0"] / absolute_fine_scale0
        )  # torch.Tensor(M x 2)
        fine_idx_1 = (
            data["coarse_coord_1"] / absolute_fine_scale1
        )  # torch.Tensor(M x 2)
        window_size = coarse_scale // fine_scale

        # 3. Extract windows from features
        feat0_fine = self._extract_feat_window(
            feat0_fine, b_idx_c, fine_idx_0, window_size
        )
        feat1_fine = self._extract_feat_window(
            feat1_fine, b_idx_c, fine_idx_1, window_size
        )

        return feat0_fine, feat1_fine

    @staticmethod
    def _extract_feat_window(
        _feat: torch.Tensor, b_idx: torch.Tensor, coord: torch.Tensor, window_size: int
    ):
        # Calculate row and column indices
        row_indices = coord[:, 1].round().long()
        col_indices = coord[:, 0].round().long()

        # Create grid indices
        row_offsets = torch.arange(window_size, device=_feat.device, dtype=torch.long)
        col_offsets = torch.arange(window_size, device=_feat.device, dtype=torch.long)

        row_indices = row_indices.unsqueeze(1) + row_offsets.unsqueeze(0)
        col_indices = col_indices.unsqueeze(1) + col_offsets.unsqueeze(0)

        # Extract windows using advanced indexing
        windows = _feat[
            b_idx[:, None, None], :, row_indices[:, :, None], col_indices[:, None, :]
        ]

        # Rearrange to [M x C x H x W]
        windows = rearrange(windows, "m h w c -> m c h w")

        return windows

    def _fine_matching(self, data: dict):
        feat0 = data["feat0_f"]  # M, C, H, W
        feat1 = data["feat1_f"]  # M, C, H, W
        b_idx_c = data["b_idx_c"]  # M
        C = feat0.shape[1]

        # 1. Pixel shuffle if exists
        feat0 = self.pixel_shuffle_head(feat0)
        feat1 = self.pixel_shuffle_head(feat1)

        # 2. Get confidence matrix on image 0 and the maximum point offset
        window_size = feat0.shape[-1]
        window_radius = window_size / 2
        conf_0 = F.log_softmax(self.fine_conf_head(feat0) / (C**0.5), dim=1)  # [M, HW]
        w = int(conf_0.shape[-1] ** 0.5)
        conf_0 = rearrange(conf_0, "m (h w) -> m h w", h=w, w=w)
        offset_f_0 = dsnt.spatial_expectation2d(conf_0[None], True)[0] + 1.0  # [M, 2]
        i, j = (
            (offset_f_0[:, 1] * window_radius),
            (offset_f_0[:, 0] * window_radius),
        )  # 2 x [M]

        grid = torch.stack([j, i], dim=-1).unsqueeze(1).unsqueeze(1)  # [M, 1, 1, 2]
        grid = 2 * grid / (window_size - 1) - 1  # Normalize into [-1, 1]

        # 3. Sample feature from feat0 using grid sample
        feat0_picked = (
            F.grid_sample(feat0, grid, align_corners=True).squeeze(2).squeeze(2)
        )  # [M, C]

        # 4. Correlation between both feature
        window_heatmap = torch.einsum("mc,mchw->mhw", feat0_picked, feat1)
        # Divide by C**0.5 to control gradients:
        # When the feature dimension C is large, dot product results can be very high,
        # causing softmax gradients to approach zero.
        # This scaling factor helps keep gradients in a reasonable range, aiding model training.
        window_heatmap = self._softmax2d((window_heatmap / (C**0.5)))

        # 5. Compute coordinate offsets from heatmap
        window_size = window_heatmap.shape[-1]
        window_radius = window_size / 2
        offset_f_1 = dsnt.spatial_expectation2d(window_heatmap[None], True)[0] + 1.0

        # 6. Compute absolute coordinates, coarse coor + fine coor * scale
        scale0 = (
            data["scale0"][b_idx_c] * self.scale_after_pixel_shuffle
            if "scale0" in data
            else self.scale_after_pixel_shuffle
        )
        scale1 = (
            data["scale1"][b_idx_c] * self.scale_after_pixel_shuffle
            if "scale1" in data
            else self.scale_after_pixel_shuffle
        )
        fine_coord_0 = (offset_f_0 * window_radius * scale0) + data["coarse_coord_0"]
        fine_coord_1 = (offset_f_1 * window_radius * scale1) + data["coarse_coord_1"]

        data.update(
            {
                "fine_coord_0": fine_coord_0,
                "fine_coord_1": fine_coord_1,
                "sim_matrix_f": window_heatmap,
                "coord_offset_f0": offset_f_0,
                "coord_offset_f1": offset_f_1,
            }
        )

    @staticmethod
    def _softmax2d(x: torch.Tensor):
        # x: [M x H x W] -> [M x (H W)]
        h, w = x.shape[1:]
        x = x.view(x.shape[0], -1)
        x = torch.softmax(x, dim=1)
        x = x.view(x.shape[0], h, w)
        return x

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith("maff."):
                state_dict[k.replace("maff.", "", 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)

    def reparameter(self):
        if hasattr(self.feature_backbone, "switch_to_deploy"):
            self.feature_backbone.switch_to_deploy()
