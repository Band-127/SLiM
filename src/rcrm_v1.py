import time
import torch
from torch import nn
from torch.nn import functional as F
from yacs.config import CfgNode as CN
from einops.einops import rearrange
import kornia.geometry.subpix.dsnt as dsnt
from src.backbone import build_backbone
from src.utils.position_encoding import DualMultiScaleSinePositionalEncoding
from src.utils.channel_alignment import ChannelAlignment
from src.utils.conf_head import ConfHead
from src.utils.any_input_identity import AnyInputIdentity
from src.utils.coarse_encoder import CoarseEncoder
from src.utils.recurrent_refinement import RecurrentRefinementUnit
from src.utils.misc import create_grid


class RCRM_v1(nn.Module):
    def __init__(self, config: CN):
        """
        Args:
            config (CN): root/model configuration
        """
        super(RCRM_v1, self).__init__()

        self.config = config

        self.dtype = getattr(torch, config["DTYPE"])
        self.d_model = config["DIMENSION"]
        self.refine_iters = int(config["REFINE_ITERS"])
        self.refine_lookup_radius = int(config["REFINE_LOOKUP_RADIUS"])
        self.coarse_scale_idx = config["COARSE_SCALE_IDX"]
        self.fine_scale_idx = config["FINE_SCALE_IDX"]
        self.coarse_scale = config["COARSE_SCALE"]
        self.fine_scale = config["FINE_SCALE"]
        self.coarse_match_thres = config["COARSE_MATCHING"]["THRESHOLD"]
        self.max_coarse_matches = config["COARSE_MATCHING"]["MAX_MATCHES"]
        self.max_intermediate_matches = config["INTERMEDIATE_MATCHING"]["MAX_MATCHES"]
        self.debug = config["DEBUG"]

        # some refinement configurations
        self.disable_pe = config["DISABLE_PE"]
        # trainable_parameters
        self.coarse_temperature = nn.Parameter(torch.tensor(0.1))
        self.fine_temperature = nn.Parameter(torch.tensor(0.1))
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

        # 4 Coarse encoder
        self.coarse_encoder = (
            CoarseEncoder(
                in_output_dim=self.d_model,
                num_layers=config["COARSE_ENCODER"]["NUM_LAYERS"],
                inner_expansion=config["COARSE_ENCODER"]["INNER_EXPANSION"],
                conv_dim=config["COARSE_ENCODER"]["CONV_DIM"],
                delta=config["COARSE_ENCODER"]["DELTA"],
                using_mamba2=config["COARSE_ENCODER"]["USING_MAMBA2"],
            )
            if config["COARSE_ENCODER"]["NUM_LAYERS"] > 0
            else AnyInputIdentity()
        )

        # 5. Confidence score mask head, used to generate confidence/saliency score for coarse matching
        self.conf_mask_head = ConfHead(self.d_model)

        # 6. Recurrent Refinement Unit
        self.recurrent_refine_unit = (
            RecurrentRefinementUnit(
                in_output_dim=self.d_model,
                num_layers=config["RRU"]["NUM_LAYERS"],
                inner_expansion=config["RRU"]["INNER_EXPANSION"],
                conv_dim=config["RRU"]["CONV_DIM"],
                delta=config["RRU"]["DELTA"],
                using_mamba2=config["RRU"]["USING_MAMBA2"],
            )
            if config["RRU"]["NUM_LAYERS"] > 0
            else AnyInputIdentity()
        )

        # 7. Fine confidence score head
        self.fine_conf_head = ConfHead(self.d_model)

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
                "refine_iters" (int): Refine iterations

                (Training only)
                "spv_b_ids" (torch.Tensor): (M)
                "spv_i_ids" (torch.Tensor): (M)
                "spv_j_ids" (torch.Tensor): (M)
                "spv_b_ids_it": (torch.Tensor): (N)
                "spv_m_ids_it": (torch.Tensor): (N)
                "spv_i_ids_it": (torch.Tensor): (N)
                "spv_j_ids_it": (torch.Tensor): (N)
                "intermediate_coord_0_gt": (torch.Tensor): (N, 2)
                "intermediate_coord_1_gt": (torch.Tensor): (N, 2)
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
                "feat0_f": (torch.Tensor): (B, C, Hf0, Wf0): fine feature 0
                "feat1_f": (torch.Tensor): (B, C, Hf1, Wf1): fine feature 1
                "hw0_f": (torch.Size): Hf0, Wf0: fine feature shape 0
                "hw1_f": (torch.Size): Hf1, Wf1: fine feature shape 1

                "sim_matrix": (torch.Tensor): (B, Lc0, Lc1) coarse similarity matrix formed in coarse matching
                "num_matches": (int): number of matches
                "b_idx_c": (torch.Tensor): (M) coarse match index, batch
                "i_idx_c": (torch.Tensor): (M) coarse match index, in image0, in coarse coordinate
                "j_idx_c": (torch.Tensor): (M) coarse match index, in image1, in coarse coordinate
                "coarse_coord_0": (torch.Tensor): (M, 2) coarse matched coords in image 0, in absolute coordinate, top left corner(for easy indexing)
                "coarse_coord_1": (torch.Tensor): (M, 2) coarse matched coords in image 1, in absolute coordinate, top left corner(for easy indexing)

                "sim_matrix_f": (torch.Tensor): (M, Lf0, Lf1) fine similarity matrix formed before fine matching

                "b_idx_it": (torch.Tensor): (N) intermediate match index, in B
                "m_idx_it": (torch.Tensor): (N) intermediate match index, in M
                "i_idx_it": (torch.Tensor): (N) intermediate match index, in image 0
                "j_idx_it": (torch.Tensor): (N) intermediate match index, in image 1
                "intermediate_coord_0": (torch.Tensor): (N, 2) intermediate matched coords in image 0, in absolute coordinate
                "intermediate_coord_1": (torch.Tensor): (N, 2) intermediate matched coords in image 1, in absolute coordinate

                "fine_coord_0": (torch.Tensor): (N, 2) fine matched coords in image 0, in absolute coordinate
                "fine_coord_1": (torch.Tensor): (N, 2) fine matched coords in image 1, in absolute coordinate
                "all_offset_1": (torch.Tensor): (ITER, N, 2) coordinate offsets in image 1, in absolute coordinate, storing all iterations

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
                "refine_iters": self.refine_iters,
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

        # 4. Coarse encoder
        coarse_x0, coarse_x1 = x0[self.coarse_scale_idx], x1[self.coarse_scale_idx] = (
            self.coarse_encoder(x0[self.coarse_scale_idx], x1[self.coarse_scale_idx])
        )

        # 5. Conf score mask generation
        conf_mask0 = self.conf_mask_head(coarse_x0)
        conf_mask1 = self.conf_mask_head(coarse_x1)
        coarse_end_time = time.perf_counter()

        # Update features, shapes and conf mask
        data.update(
            {
                "feat0_c": coarse_x0,  # B, C, Hc0, Wc0
                "feat1_c": coarse_x1,  # B, C, Hc1, Wc1
                "hw0_c": coarse_x0.shape[2:],
                "hw1_c": coarse_x1.shape[2:],
                "conf_mask0": conf_mask0,
                "conf_mask1": conf_mask1,
                "feat0_f": x0[self.fine_scale_idx],  # B, C, Hf0, Wf0
                "feat1_f": x1[self.fine_scale_idx],  # B, C, Hf1, Wf1
                "hw0_f": x0[self.fine_scale_idx].shape[2:],
                "hw1_f": x1[self.fine_scale_idx].shape[2:],
            }
        )

        # 6. Coarse matching
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

        # 7. Intermediate matching
        self._fine_correlation(data=data)
        self._get_intermediate_coord_test(data=data)

        # 8. Reccurent refinement for coordinates
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

        # 4. Coarse encoder
        coarse_x0, coarse_x1 = x0[self.coarse_scale_idx], x1[self.coarse_scale_idx] = (
            self.coarse_encoder(x0[self.coarse_scale_idx], x1[self.coarse_scale_idx])
        )

        # 5. Conf score mask generation
        conf_mask0 = self.conf_mask_head(coarse_x0)
        conf_mask1 = self.conf_mask_head(coarse_x1)
        coarse_end_time = time.perf_counter()

        # Update features, shapes and conf mask
        data.update(
            {
                "feat0_c": coarse_x0,  # B, C, Hc0, Wc0
                "feat1_c": coarse_x1,  # B, C, Hc1, Wc1
                "hw0_c": coarse_x0.shape[2:],
                "hw1_c": coarse_x1.shape[2:],
                "conf_mask0": conf_mask0,
                "conf_mask1": conf_mask1,
                "feat0_f": x0[self.fine_scale_idx],  # B, C, Hf0, Wf0
                "feat1_f": x1[self.fine_scale_idx],  # B, C, Hf1, Wf1
                "hw0_f": x0[self.fine_scale_idx].shape[2:],
                "hw1_f": x1[self.fine_scale_idx].shape[2:],
            }
        )

        # 6. Coarse matching
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
        correlation_end_time = time.perf_counter()

        # 7. Intermediate matching
        self._fine_correlation(data=data)
        self._get_intermediate_coord_train(data=data)

        # 8. Reccurent refinement for coordinates
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
                "coarse_coord_1": coarse_coord_1,  # in absolute coordinate, at the center of coarse window
                "num_matches": b_idx_c.shape[0],
            }
        )

    def _fine_correlation(self, data: dict):
        b_idx_c = data["b_idx_c"]  # M
        feat0 = data["feat0_f"]  # B, C, H, W
        feat1 = data["feat1_f"]  # B, C, H, W
        C = feat0.shape[1]
        absolute_fine_scale0 = (
            self.fine_scale * data["scale0"][b_idx_c]
            if "scale0" in data
            else self.fine_scale
        )
        absolute_fine_scale1 = (
            self.fine_scale * data["scale1"][b_idx_c]
            if "scale1" in data
            else self.fine_scale
        )

        # 1. Get feature windows from both feature maps
        window_size = int(self.coarse_scale / self.fine_scale)
        feat0_window = self._extract_feat_window(
            feat0, b_idx_c, data["coarse_coord_0"] / absolute_fine_scale0, window_size
        )  # M, C, H, W
        feat1_window = self._extract_feat_window(
            feat1, b_idx_c, data["coarse_coord_1"] / absolute_fine_scale1, window_size
        )  # M, C, H, W

        # 2. Flatten
        feat0_window = rearrange(feat0_window, "b c h w -> b (h w) c")  # M, Lf0, C
        feat1_window = rearrange(feat1_window, "b c h w -> b (h w) c")  # M, Lf1, C

        # 3. Divide by C**0.5 to control gradients:
        # When the feature dimension C is large, dot product results can be very high,
        # causing softmax gradients to approach zero.
        # This scaling factor helps keep gradients in a reasonable range, aiding model training.
        feat0_window, feat1_window = map(
            lambda x: x / C**0.5, [feat0_window, feat1_window]
        )

        # 4. Get similarity matrix
        sim_matrix = (
            torch.einsum("blc,bsc->bls", feat0_window, feat1_window)
            / self.fine_temperature
        )  # M, Lf0, Lf1

        # 5. Update sim matrix
        data.update({"sim_matrix_f": sim_matrix})

    @torch.no_grad()
    def _get_intermediate_coord_train(self, data: dict):
        N = data["spv_b_ids_it"].shape[0]
        if N > self.max_intermediate_matches:
            perm = torch.randperm(N, device=data["spv_b_ids_it"].device)
            idx = perm[: self.max_intermediate_matches]
            data.update(
                {
                    "b_idx_it": data["spv_b_ids_it"][idx],
                    "m_idx_it": data["spv_m_ids_it"][idx],
                    "i_idx_it": data["spv_i_ids_it"][idx],
                    "j_idx_it": data["spv_j_ids_it"][idx],
                    "intermediate_coord_0": data["intermediate_coord_0_gt"][idx],
                    "intermediate_coord_1": data["intermediate_coord_1_gt"][idx],
                }
            )
        else:
            data.update(
                {
                    "b_idx_it": data["spv_b_ids_it"],
                    "m_idx_it": data["spv_m_ids_it"],
                    "i_idx_it": data["spv_i_ids_it"],
                    "j_idx_it": data["spv_j_ids_it"],
                    "intermediate_coord_0": data["intermediate_coord_0_gt"],
                    "intermediate_coord_1": data["intermediate_coord_1_gt"],
                }
            )

    @torch.no_grad()
    def _get_intermediate_coord_test(self, data: dict):
        coarse_scale = self.coarse_scale
        fine_scale = self.fine_scale
        conf_matrix = data["sim_matrix_f"]
        b_idx_c = data["b_idx_c"]  # [M]
        absolute_fine_scale0 = (
            fine_scale * data["scale0"] if "scale0" in data else fine_scale
        )  # [B, 2]
        absolute_fine_scale1 = (
            fine_scale * data["scale1"] if "scale1" in data else fine_scale
        )  # [B, 2]
        coarse_coord_0, coarse_coord_1 = data["coarse_coord_0"], data["coarse_coord_1"]
        window_size = int(coarse_scale / fine_scale)  # w

        # Dual-softmax
        conf_matrix = F.softmax(conf_matrix, 1) * F.softmax(conf_matrix, 2)

        coarse_coord_0 = (coarse_coord_0 / absolute_fine_scale0).round() + 0.5  # [M, 2]
        coarse_coord_1 = (coarse_coord_1 / absolute_fine_scale1).round() + 0.5  # [M, 2]

        # intermediate_match = torch.argwhere(conf_matrix > self.coarse_match_thres)
        # del conf_matrix

        # m_idx_it = intermediate_match[:, 0]
        # i_idx_it = intermediate_match[:, 1]
        # j_idx_it = intermediate_match[:, 2]
        # b_idx_it = b_idx_c[m_idx_it]
        # Get top k matches for each coarse match
        flat_conf = conf_matrix.view(conf_matrix.shape[0], -1)  # [M, Lf0*Lf1]
        top_k = int(min(self.max_intermediate_matches / conf_matrix.shape[0], flat_conf.shape[1]))
        top_k_values, top_k_indices = torch.topk(flat_conf, k=top_k, dim=1)  # [M, K]

        # Apply threshold
        valid_matches = top_k_values > (
            conf_matrix.max() * self.coarse_match_thres
        )  # [M, K]

        # Convert linear indices to 2D indices
        i_idx_it = (top_k_indices // conf_matrix.shape[2]).long()  # [M, K]
        j_idx_it = (top_k_indices % conf_matrix.shape[2]).long()  # [M, K]

        # Create match indices
        m_idx_it = (
            torch.arange(conf_matrix.shape[0], device=conf_matrix.device)
            .unsqueeze(1)
            .expand(-1, top_k)
        )  # [M, K]

        # Filter using valid_matches mask
        m_idx_it = m_idx_it[valid_matches]  # [N]
        i_idx_it = i_idx_it[valid_matches]  # [N]
        j_idx_it = j_idx_it[valid_matches]  # [N]
        b_idx_it = b_idx_c[m_idx_it]  # [N]

        del conf_matrix  # Free memory

        intermediate_coord_0 = (
            torch.stack(
                (
                    (i_idx_it % window_size),
                    (i_idx_it // window_size),
                ),
                dim=1,
            )
            + coarse_coord_0[m_idx_it]
        ) * absolute_fine_scale0[b_idx_it]

        intermediate_coord_1 = (
            torch.stack(
                (
                    (j_idx_it % window_size),
                    (j_idx_it // window_size),
                ),
                dim=1,
            )
            + coarse_coord_1[m_idx_it]
        ) * absolute_fine_scale1[b_idx_it]

        data.update(
            {
                "b_idx_it": b_idx_it,
                "m_idx_it": m_idx_it,
                "i_idx_it": i_idx_it,
                "j_idx_it": j_idx_it,
                "intermediate_coord_0": intermediate_coord_0,
                "intermediate_coord_1": intermediate_coord_1,
            }
        )

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

    @staticmethod
    def _extract_feat_window_bilinear(
        _feat: torch.Tensor, b_idx: torch.Tensor, coord: torch.Tensor, window_size: int
    ):
        """
        Args:
            _feat (torch.Tensor): B x C x H x W
            b_idx (torch.Tensor): M
            coord (torch.Tensor): M x 2
            window_size (int): w

        Returns:
            (torch.Tensor): (M x C x w x w)
        """
        # Row and column indices
        offsets = torch.arange(window_size, device=_feat.device) - (window_size - 1) / 2
        row_indices = coord[:, 1, None] + offsets
        col_indices = coord[:, 0, None] + offsets

        # Prepare grid for grid_sample
        grid = create_grid(row_indices, col_indices).permute(
            0, 2, 1, 3
        )  # (M, H, W, 2) (x, y)

        # Normalize grid to [-1, 1] range
        grid = (
            grid
            / torch.tensor(
                [_feat.shape[3] - 1, _feat.shape[2] - 1], device=_feat.device
            )
            * 2
        ) - 1

        # Extract windows using bilinear sampling
        b_idx_unique = b_idx.unique()
        windows = torch.zeros(
            (b_idx.shape[0], _feat.shape[1], window_size, window_size),
            device=_feat.device,
            dtype=_feat.dtype,
        )
        for b in b_idx_unique:
            mask = b_idx == b
            # Concat all the grids with the same batch index
            grid_b = grid[mask]  # m x Hout x Wout x 2
            feat_b = _feat[b : b + 1]  # 1 x C x Hin x Win
            grid_b = rearrange(
                grid_b, "m h w c -> 1 (m h) w c"
            )  # 1 x (m x Hout) x Wout x 2
            window = F.grid_sample(
                feat_b, grid_b, mode="bilinear", align_corners=True
            )  # 1 x C x (m x Hout) x Wout
            window = rearrange(
                window,
                "b c (m h) w -> (b m) c h w",
                m=mask.sum(),
                h=window_size,
                w=window_size,
            )  # m x C x Hout x Wout
            windows[mask] = window

        return windows

    def _fine_matching(self, data: dict):
        feat0 = data["feat0_f"]  # B, C, H, W
        feat1 = data["feat1_f"]  # B, C, H, W
        b_idx_it = data["b_idx_it"]  # N
        scale0 = (
            data["scale0"][b_idx_it] * self.fine_scale
            if "scale0" in data
            else self.fine_scale
        )  # N, 2
        scale1 = (
            data["scale1"][b_idx_it] * self.fine_scale
            if "scale1" in data
            else self.fine_scale
        )  # N, 2
        fine_coord_0 = data["intermediate_coord_0"]  # N, 2

        # 1. Get feature window in image0 using intermediate matches
        lookup_window_size = self.refine_lookup_radius * 2
        feat0_window = self._extract_feat_window_bilinear(
            feat0, b_idx_it, fine_coord_0 / scale0, lookup_window_size
        )  # [N, C, H, W]

        # 4. Iteratively refine coords on image 1
        fine_coord_1_init = fine_coord_1 = data["intermediate_coord_1"]  # N, 2
        total_offset = torch.zeros((b_idx_it.shape[0], 2), device=feat0.device)  # N, 2
        hidden_state = torch.zeros(
            (b_idx_it.shape[0], 1, self.d_model), device=feat0.device
        )  # Initial hidden state, [N, 1, C]
        all_offset_1 = []
        for i in range(self.refine_iters):
            # 4.1 Get feature from coords
            feat1_window = self._extract_feat_window_bilinear(
                feat1, b_idx_it, fine_coord_1 / scale1, lookup_window_size
            )  # [N, C, H, W]

            # 4.2 Both features enter recurrent refinement unit
            offset_f_1, hidden_state = self.recurrent_refine_unit(
                feat0_window, feat1_window, hidden_state
            )  # [N, 2], [N, 1, C]

            # 4.3 Update total offset
            total_offset += offset_f_1
            all_offset_1.append(total_offset)

            # 4.4 Update refined coords
            fine_coord_1 = (
                total_offset * self.refine_lookup_radius * scale1
            ) + fine_coord_1_init

        all_offset_1 = torch.stack(all_offset_1, dim=0)  # [ITER, N, 2]

        data.update(
            {
                "fine_coord_0": fine_coord_0,
                "fine_coord_1": fine_coord_1,
                "all_offset_1": all_offset_1,
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
