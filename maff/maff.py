import torch
from torch import nn
from torch.nn import functional as F
from yacs.config import CfgNode as CN
from einops.einops import rearrange
from typing import Sequence
import kornia.geometry.subpix.dsnt as dsnt
from kornia.utils.grid import create_meshgrid

from maff.backbone import build_backbone
from maff.mamba.MambaEncoder import MambaEncoder
from maff.transformer.transformer import LocalFeatureTransformer
from maff.utils.position_encoding import DualMultiScaleSinePositionalEncoding
from maff.utils.channel_alignment import ChannelAlignment


class MAFF(nn.Module):
    def __init__(self, config: CN):
        super(MAFF, self).__init__()

        self.config = config

        self.dtype = torch.float32 if config["DTYPE"] == "float32" else "float64"
        self.d_model = config["BACKBONE"]["LAYER_DIMS"][-1]
        self.scales_selection = config["SCALES_SELECTION"]
        self.coarse_scale_idx = config["COARSE_SCALE_IDX"]
        self.coarse_match_thres = config["COARSE_MATCHING"]["THRESHOLD"]
        self.debug = config["DEBUG"]

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
                using_mamba2=config["MAMBA_FUSION"]["USING_MAMBA2"],
            )
            if config["FUSION_TYPE"] == "mamba"
            else LocalFeatureTransformer(config=config["TRANSFORMER_FUSION"])
        )

    def forward(self, data: dict, training: bool = True):
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
            training (bool, optional): Whether in training mode. Defaults to True.
        """
        data.update(
            {
                "batch_size": data["image0"].shape[0],
                "hw0_i": data["image0"].shape[2:],
                "hw1_i": data["image1"].shape[2:],
            }
        )

        if self.debug:
            print(
                f"Initial GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            )

        # 1. Feature extraction
        x0, x1 = (
            self.feature_backbone(data["image0"]),
            self.feature_backbone(data["image1"]),
        )  # Scales, [B x C x H x W]
        mask0, mask1 = (
            data["mask0"].flatten(-2),
            data["mask1"].flatten(-2),
        )  # Flattened, [N x L]
        if self.debug:
            print(
                f"Step: Feature extraction\nGPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            )

        # 2. Feature channel alignment
        x0, x1 = (
            self.feature_channel_alignment(x0),
            self.feature_channel_alignment(x1),
        )  # Scales, [B x C x H x W]
        if self.debug:
            print(
                f"Step: Feature channel alignment\nGPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            )

        # 3. Position Encoding
        x0, x1 = self.pe(x0, x1)  # S x [B x C x H x W]
        if self.debug:
            print(
                f"Step: Position Encoding\nGPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            )

        # 4. Feature selection
        new_x0 = []
        new_x1 = []
        for i, selection in enumerate(self.scales_selection):
            if selection == 1:
                new_x0.append(x0[i])
                new_x1.append(x1[i])
        x0 = new_x0
        x1 = new_x1
        data.update(
            {
                "hw0_c": x0[0].shape[2:],
                "hw1_c": x1[0].shape[2:],
            }
        )
        if self.debug:
            print(
                f"Step: Feature selection\nGPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            )

        # 5. Flatten S x [B x C x H x W] -> [B x (HW) x (sum(C))]
        x0, x1, x0_length, x1_length = self.flatten(x0, x1)
        if self.debug:
            print(
                f"Step: Flatten features\nGPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            )

        # 6. Mamba
        x0, x1 = self.fusion_encoder(x0, x1)
        if self.debug:
            print(
                f"Step: Fusion encoder\nGPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            )

        # 7. Unflatten into S x [B x (HW) x C]
        x0, x1 = self.unflatten(x0, x1, x0_length, x1_length)
        if self.debug:
            print(
                f"Step: Unflatten features\nGPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            )

        data.update(
            {"feat0_c": x0[self.coarse_scale_idx], "feat1_c": x1[self.coarse_scale_idx]}
        )

        # 8. Correlation / Feature Matching
        self.feature_matching(data, data["feat0_c"], data["feat1_c"], mask0, mask1, training)

    def flatten(self, x0: Sequence[torch.Tensor], x1: Sequence[torch.Tensor]):
        # [B x C x H x W] -> [B x (HW) x C]
        for i, scale in enumerate(x0):
            x0[i] = rearrange(scale, "b c h w -> b (h w) c")
        for i, scale in enumerate(x1):
            x1[i] = rearrange(scale, "b c h w -> b (h w) c")

        # S x [B x (HW) x C] -> [B x (HW) x (sum(C))]
        x0_length = [i.shape[1] for i in x0]
        x1_length = [i.shape[1] for i in x1]
        x0 = torch.concat(x0, dim=1)
        x1 = torch.concat(x1, dim=1)

        return x0, x1, x0_length, x1_length

    def unflatten(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x0_length: torch.Size,
        x1_length: torch.Size,
    ):
        # [B x (HW) x (sum(C))] -> S x [B x (HW) x C]
        x0 = torch.split(x0, split_size_or_sections=x0_length, dim=1)
        x1 = torch.split(x1, split_size_or_sections=x1_length, dim=1)

        return x0, x1

    def feature_matching(
        self,
        data: dict,
        x0: torch.Tensor,
        x1: torch.Tensor,
        mask1: torch.Tensor = None,
        mask2: torch.Tensor = None,
        training: bool = True,
    ):
        """
        Feature Matching using full correlation, and getting the coordinate of the best match

        Args:
            data (dict): data batch
            x0 (torch.Tensor): [B, L0, C]
            x1 (torch.Tensor): [B, L1, C]
            mask1 (torch.Tensor, optional): [B, L0]. Defaults to None.
            mask2 (torch.Tensor, optional): [B, L1]. Defaults to None.
            training (bool, optional): Whether in training mode. Defaults to True.
        """
        # 1. Full Correlation
        # Normalize
        x0, x1 = map(lambda x: x / x.shape[-1] ** 0.5, [x0, x1])

        # Similarity matrix without dustbin
        sim_matrix = torch.einsum("blc,bsc->bls", x0, x1) / 0.1

        # Mask the area on similarity matrix where the mask==False into -inf
        if mask1 is not None and mask2 is not None:
            sim_matrix.masked_fill_(
                ~(mask1.unsqueeze(-1) * mask2.unsqueeze(-2)).bool(), -1e9
            )

        # Update similarity matrix into data batch, used in coarse supervision
        data.update({"sim_matrix": sim_matrix})
        if self.debug:
            print(
                f"Step: Full Correlation\nGPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            )

        if not training:
            # 2. Get coarse coordinates
            self.get_coarse_coord(data=data)
            if self.debug:
                print(
                    f"Step: Get coarse coordinates\nGPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
                )

            # 3. Fine Matching
            self.fine_matching(data=data)
            if self.debug:
                print(
                    f"Step: Fine Matching\nGPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
                )

    @torch.no_grad()
    def get_coarse_coord(self, data: dict):
        sim_matrix = data["sim_matrix"]
        # Get coarse matches > threshold, not using dual softmax (reference to Efficient LoFTR)
        coarse_match = torch.argwhere(
            sim_matrix / sim_matrix.max() > self.coarse_match_thres
        )  # M, 3(B, I, J)

        b_idx_c = coarse_match[:, 0]
        i_idx_c = coarse_match[:, 1]
        j_idx_c = coarse_match[:, 2]
        torch.cuda.empty_cache()

        # Match indices -> coordinates
        scale = data["hw0_i"][0] / data["hw0_c"][0]
        coarse_coord_0 = (
            torch.stack(
                (
                    i_idx_c % data["hw0_c"][0],
                    i_idx_c // data["hw0_c"][0],
                ),
                dim=1,
            )
            * scale
        )
        coarse_coord_1 = (
            torch.stack(
                (
                    j_idx_c % data["hw1_c"][0],
                    j_idx_c // data["hw1_c"][0],
                ),
                dim=1,
            )
            * scale
        )

        data.update(
            {
                "b_idx_c": b_idx_c,  # in coarse coordinate
                "i_idx_c": i_idx_c,  # in coarse coordinate
                "j_idx_c": j_idx_c,  # in coarse coordinate
                "coarse_coord_0": coarse_coord_0,  # in absolute coordinate
                "coarse_coord_1": coarse_coord_1,  # in absolute coordinate
            }
        )

    @torch.no_grad()
    def fine_matching(self, data: dict):
        feat0 = data["feat0_c"]
        feat1 = data["feat1_c"]
        b_idx_c = data["b_idx_c"]
        i_idx_c = data["i_idx_c"]
        j_idx_c = data["j_idx_c"]
        scale = data["hw0_i"][0] / data["hw0_c"][0]
        H, W = data["hw0_c"]
        C = feat0.shape[-1]
        window_size = self.config.FINE_MATCHING.WINDOW_SIZE  # window size

        if b_idx_c.shape[0] == 0:
            data.update(
                {
                    "fine_coord_0": torch.zeros(
                        size=(0, 2), dtype=torch.float32, device=feat0.device
                    ),
                    "fine_coord_1": torch.zeros(
                        size=(0, 2), dtype=torch.float32, device=feat0.device
                    ),
                    "conf_map": torch.zeros(
                        size=(0, 3, 3), dtype=torch.float32, device=feat0.device
                    ),
                    "std": torch.zeros(
                        size=(0, 1), dtype=torch.float32, device=feat0.device
                    ),
                }
            )
            return

        # 1. Pick feature from coarse feature map x0 using coarse match
        feat0_picked = feat0[b_idx_c, i_idx_c]  # M, C
        if self.debug:
            print(
                f"Step: Pick coarse match features\nGPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            )

        # 2. Pick a window of feature from coarse feature map x1 using coarse match
        feat1_window_picked = self.extract_feat_window(
            _feat=feat1,
            feat_hw=data["hw1_c"],
            b_idx=b_idx_c,
            j_idx=j_idx_c,
            window_size=window_size,
        )  # M, C, window_size, window_size
        if self.debug:
            print(
                f"Step: Extract feature window\nGPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            )

        # 3. Correlation between both feature
        window_heatmap = torch.einsum("mc,mchw->mhw", feat0_picked, feat1_window_picked)
        conf_map = window_heatmap.detach().clone()
        window_heatmap = torch.softmax(window_heatmap / (C**0.5), dim=1)
        if self.debug:
            print(
                f"Step: Compute feature correlation\nGPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            )

        # 4. Compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(window_heatmap[None], True)[
            0
        ]  # M, 2
        grid_normalized = create_meshgrid(
            window_size, window_size, True, window_heatmap.device
        ).reshape(1, -1, 2)  # [1, WW, 2]
        if self.debug:
            print(
                f"Step: Compute normalized coordinates\nGPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            )

        # 5. Compute absolute coordinates, (coarse coor + fine coor) * scale
        fine_coord_0 = data["coarse_coord_0"]
        fine_coord_1 = (coords_normalized * (window_size // 2) * scale) + data[
            "coarse_coord_1"
        ]
        if self.debug:
            print(
                f"Step: Compute absolute coordinates\nGPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            )

        # 6. Compute std over <x, y> (used in loss)
        var = (
            torch.sum(
                grid_normalized**2 * window_heatmap.view(-1, window_size**2, 1), dim=1
            )
            - coords_normalized**2
        )  # [M, 2]
        std = torch.sum(
            torch.sqrt(torch.clamp(var, min=1e-10)), -1
        )  # [M]  clamp needed for numerical stability

        if self.debug:
            print(
                f"Step: Compute standard deviation\nGPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            )

        data.update(
            {
                "fine_coord_0": fine_coord_0,
                "fine_coord_1": fine_coord_1,
                "conf_map": conf_map,
                "std": std,
            }
        )

    @staticmethod
    def extract_feat_window(
        _feat: torch.Tensor,
        feat_hw: Sequence[int],
        b_idx: torch.Tensor,
        j_idx: torch.Tensor,
        window_size: int,
    ):
        H, W = feat_hw
        pad_size = window_size // 2
        # Rearrange
        feat = rearrange(_feat, "b (h w) c -> b c h w", h=H, w=W)

        # Padding
        feat_padded = F.pad(feat, (pad_size, pad_size, pad_size, pad_size))

        # # Old method
        # windows = []
        # for i in range(b_idx.shape[0]):
        #     window = feat_padded[
        #         b_idx[i],
        #         :,
        #         j_idx[i] // W : j_idx[i] // W + window_size,
        #         j_idx[i] % W : j_idx[i] % W + window_size,
        #     ]
        #     windows.append(window)

        # return torch.stack(windows)

        # Calculate row and column indices
        row_indices = j_idx // W
        col_indices = j_idx % W

        # Create grid indices
        row_offsets = torch.arange(window_size, device=feat.device)
        col_offsets = torch.arange(window_size, device=feat.device)

        row_indices = row_indices.unsqueeze(1) + row_offsets.unsqueeze(0)
        col_indices = col_indices.unsqueeze(1) + col_offsets.unsqueeze(0)

        # Extract windows using advanced indexing
        windows = feat_padded[
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
