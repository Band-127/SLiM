import torch
from torch import nn
from torch.nn import functional as F
from einops.einops import rearrange
from yacs.config import CfgNode as CN

from .backbone import build_backbone
from .recurrent_refinement import RecurrentRefinementUnit
from .utils.misc import create_grid, CudaTimer, Upsample


class SoMa(nn.Module):
    def __init__(self, config: CN):
        super().__init__()
        self.d_coarse = config["COARSE_DIM"]
        self.d_fine = config["FINE_DIM"]
        self.refine_iters = int(config["REFINE_ITERS"])
        self.refine_lookup_radius = int(config["REFINE_LOOKUP_RADIUS"])
        self.lookup_window_size = int(self.refine_lookup_radius * 2)
        self.coarse_scale_idx = config["COARSE_SCALE_IDX"]
        self.fine_scale_idx = config["FINE_SCALE_IDX"]
        self.coarse_match_thres = config["COARSE_THRES"]
        self.fine_match_thres = config["FINE_THRES"]
        self.max_coarse_matches = config["MAX_COARSE_MATCHES"]
        self.max_fine_matches = config["MAX_FINE_MATCHES"]
        self.train_noise_scale = config["TRAIN_NOISE_SCALE"]
        self.optimized_dual_softmax = config["OPTIMIZED_DUAL_SOFTMAX"]

        # 1. Backbone
        self.feature_backbone = build_backbone(config["BACKBONE"])

        # 2. Recurrent Refinement Unit
        self.recurrent_refine_unit = RecurrentRefinementUnit(
            input_dim=self.d_fine,
            # input_dim=self.lookup_window_size**2,
            hidden_dim=config["REFINEMENT"]["HIDDEN_DIM"],
            context=config["REFINEMENT"]["CONTEXT_INJECTION"],
        )

        # 3. Fine feature upsample, x2
        self.fine_upsample = Upsample(scale=2.0)

        # trainable_parameters
        self.coarse_temperature = nn.Parameter(torch.tensor(0.05))
        self.fine_temperature = nn.Parameter(torch.tensor(0.005))

    def forward(self, data: dict, training: bool = False):
        """
        Forward function
        Input:
            data (dict): {
                "image0": (torch.Tensor): (B, 1, H, W)
                "image1": (torch.Tensor): (B, 1, H, W)
                "mask0": (torch.Tensor, optional): (B, H, W): '0' indicates a padded position, in coarse scale
                "mask1": (torch.Tensor, optional): (B, H, W): '0' indicates a padded position, in coarse scale
                "scale0": (torch.Tensor, optional): (B, 2), Megadepth only, absolute scale from input size to original size
                "scale1": (torch.Tensor, optional): (B, 2), Megadepth only, absolute scale from input size to original size

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
                "batch_size": (int)
                "hw0_i": (torch.Size): Hi0, Wi0: input shape 0
                "hw1_i": (torch.Size): Hi1, Wi1: input shape 1
                "refine_iters" (int): Refine iterations

                "feat0_all": (List[torch.Tensor]): S x [B, C, H, W]: feature pyramid of image 0
                "feat1_all": (List[torch.Tensor]): S x [B, C, H, W]: feature pyramid of image 1
                "feat0_c": (torch.Tensor): (B, C, Hc0, Wc0): coarse feature 0
                "feat1_c": (torch.Tensor): (B, C, Hc1, Wc1): coarse feature 1
                "hw0_c": (torch.Size): Hc0, Wc0: coarse feature shape 0
                "hw1_c": (torch.Size): Hc1, Wc1: coarse feature shape 1
                "feat0_f": (torch.Tensor): (B, C, Hf0, Wf0): fine feature 0
                "feat1_f": (torch.Tensor): (B, C, Hf1, Wf1): fine feature 1
                "hw0_f": (torch.Size): Hf0, Wf0: fine feature shape 0
                "hw1_f": (torch.Size): Hf1, Wf1: fine feature shape 1
                "coarse_scale" (float): hw0_i / hw0_c
                "fine_scale" (float): hw0_i / hw0_f

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
                "upsample_time": (float), in seconds
                "coarse_match_time": (float), in seconds
                "fine_match_time": (float), in seconds
                "refine_time": (float), in seconds
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
                "refine_iters": self.refine_iters,
            }
        )
        if training:
            return self._forward_train(data)
        else:
            return self._forward_test(data)

    def _forward_train(self, data: dict):
        # 1. Feature
        with CudaTimer() as timer:
            x0, x1 = self.feature_backbone(data["image0"], data["image1"])
        mask0, mask1 = (
            (
                data["mask0"].flatten(-2),
                data["mask1"].flatten(-2),
            )
            if "mask0" in data
            else (None, None)
        )  # Flattened, [B x L]
        coarse_x0, coarse_x1 = x0[self.coarse_scale_idx], x1[self.coarse_scale_idx]
        feat_extract_time = timer.elapsed_time / data["batch_size"]

        # 2. Upsample
        with CudaTimer() as timer:
            fine_0, fine_1 = self.fine_upsample(
                x0[self.fine_scale_idx], x1[self.fine_scale_idx]
            )
        upsample_time = timer.elapsed_time / data["batch_size"]

        data.update(
            {
                "feat0_c": coarse_x0,  # B, C, Hc0, Wc0
                "feat1_c": coarse_x1,  # B, C, Hc1, Wc1
                "hw0_c": coarse_x0.shape[2:],
                "hw1_c": coarse_x1.shape[2:],
                "feat0_f": fine_0,  # B, C, Hf0, Wf0
                "feat1_f": fine_1,  # B, C, Hf1, Wf1
                "hw0_f": fine_0.shape[2:],
                "hw1_f": fine_1.shape[2:],
                "coarse_scale": data["image0"].shape[-1] / coarse_x0.shape[-1],
                "fine_scale": data["image0"].shape[-1] / fine_0.shape[-1],
            }
        )

        # 3. Coarse matching
        with CudaTimer() as timer:
            self._coarse_correlation(
                data=data,
                feat0_c=coarse_x0,
                feat1_c=coarse_x1,
                mask0=mask0,
                mask1=mask1,
            )  # (B x HW x HW)
            self._get_coarse_coord_train(data=data)  # Get coord from gt
        coarse_match_time = timer.elapsed_time / data["batch_size"]

        # 4. Fine matching
        with CudaTimer() as timer:
            self._fine_correlation(data=data)
            self._get_fine_coord_train(data=data)
        fine_match_time = timer.elapsed_time / data["batch_size"]

        # 5. RRU
        with CudaTimer() as timer:
            self._refinement(data=data)
        refine_time = timer.elapsed_time / data["batch_size"]

        data.update(
            {
                "feat_extract_time": feat_extract_time,
                "upsample_time": upsample_time,
                "coarse_match_time": coarse_match_time,
                "fine_match_time": fine_match_time,
                "refine_time": refine_time,
            }
        )

    def _forward_test(self, data: dict):
        # 1. Feature
        with CudaTimer() as timer:
            x0, x1 = self.feature_backbone(data["image0"], data["image1"])
        mask0, mask1 = (
            (
                data["mask0"].flatten(-2),
                data["mask1"].flatten(-2),
            )
            if "mask0" in data
            else (None, None)
        )  # Flattened, [B x L]
        coarse_x0, coarse_x1 = x0[self.coarse_scale_idx], x1[self.coarse_scale_idx]
        feat_extract_time = timer.elapsed_time / data["batch_size"]

        # 2. Upsample
        with CudaTimer() as timer:
            fine_0, fine_1 = self.fine_upsample(
                x0[self.fine_scale_idx], x1[self.fine_scale_idx]
            )
        upsample_time = timer.elapsed_time / data["batch_size"]

        data.update(
            {
                "feat0_c": coarse_x0,  # B, C, Hc0, Wc0
                "feat1_c": coarse_x1,  # B, C, Hc1, Wc1
                "hw0_c": coarse_x0.shape[2:],
                "hw1_c": coarse_x1.shape[2:],
                "feat0_f": fine_0,  # B, C, Hf0, Wf0
                "feat1_f": fine_1,  # B, C, Hf1, Wf1
                "hw0_f": fine_0.shape[2:],
                "hw1_f": fine_1.shape[2:],
                "coarse_scale": data["image0"].shape[-1] / coarse_x0.shape[-1],
                "fine_scale": data["image0"].shape[-1] / fine_0.shape[-1],
            }
        )

        # 3. Coarse matching
        with CudaTimer() as timer:
            self._coarse_correlation(
                data=data,
                feat0_c=coarse_x0,
                feat1_c=coarse_x1,
                mask0=mask0,
                mask1=mask1,
            )  # (B x HW x HW)
            # self._get_coarse_coord_test(data=data)  # Get coord from sim matrix
            self._coarse_match_norm_clue(
                data=data,
                feat0_c=coarse_x0,
                feat1_c=coarse_x1,
                mask0=mask0,
                mask1=mask1,
            )
        coarse_match_time = timer.elapsed_time / data["batch_size"]

        # 4. Fine matching
        with CudaTimer() as timer:
            self._fine_correlation(data=data)
            self._get_fine_coord_test(data=data)
        fine_match_time = timer.elapsed_time / data["batch_size"]

        # 5. RRU
        with CudaTimer() as timer:
            self._refinement(data=data)
        refine_time = timer.elapsed_time / data["batch_size"]

        data.update(
            {
                "feat_extract_time": feat_extract_time,
                "upsample_time": upsample_time,
                "coarse_match_time": coarse_match_time,
                "fine_match_time": fine_match_time,
                "refine_time": refine_time,
            }
        )

    def _coarse_correlation(
        self,
        data: dict,
        feat0_c: torch.Tensor,
        feat1_c: torch.Tensor,
        mask0: torch.Tensor = None,
        mask1: torch.Tensor = None,
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
        sim_matrix = torch.einsum("blc,bsc->bls", feat0_c, feat1_c) / (
            self.coarse_temperature + 1e-8
        )

        # 3. Mask the area on similarity matrix where the mask==False into -inf
        if mask0 is not None and mask1 is not None:
            sim_matrix = sim_matrix.to(torch.float32)
            sim_matrix.masked_fill_(
                ~(mask0.unsqueeze(-1) * mask1.unsqueeze(-2)).bool(), -1e9
            )

        # 4. Update sim matrix
        data.update({"sim_matrix": sim_matrix})

    @torch.no_grad
    def _get_coarse_coord_train(self, data: dict):
        b_idx_c = data["spv_b_ids"]
        i_idx_c = data["spv_i_ids"]
        j_idx_c = data["spv_j_ids"]

        # Match indices -> coordinates
        scale0 = (
            data["coarse_scale"] * data["scale0"][b_idx_c]
            if "scale0" in data
            else data["coarse_scale"]
        )
        scale1 = (
            data["coarse_scale"] * data["scale1"][b_idx_c]
            if "scale1" in data
            else data["coarse_scale"]
        )
        W0 = data["hw0_c"][1]
        W1 = data["hw1_c"][1]
        coarse_coord_0 = (
            torch.stack(
                (
                    (i_idx_c % W0),
                    (i_idx_c // W0),
                ),
                dim=1,
            )
            * scale0
        )
        coarse_coord_1 = (
            torch.stack(
                (
                    (j_idx_c % W1),
                    (j_idx_c // W1),
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
            }
        )

    @torch.no_grad
    def _get_coarse_coord_test(self, data: dict):
        sim_matrix = data["sim_matrix"]
        if not self.optimized_dual_softmax:
            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
            # Get top max_coarse_matches matches with highest confidence
            top_k = min(
                self.max_coarse_matches,
                conf_matrix.shape[-1] * conf_matrix.shape[0],
            )
            flat_conf = conf_matrix.view(-1)
            top_k_values, top_k_indices = torch.topk(flat_conf, k=top_k)

            # Apply threshold
            valid_matches = top_k_values > (conf_matrix.max() * self.coarse_match_thres)
            # valid_matches = top_k_values > self.coarse_match_thres
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
        else:
            candidate_b, candidate_i, candidate_j = torch.where(
                sim_matrix > (sim_matrix.max() * 0.3)
            )

            # Partial softmax
            b_pos_unique, inversed_b = candidate_b.unique(
                sorted=True, return_inverse=True
            )
            i_pos_unique, inversed_i = candidate_i.unique(
                sorted=True, return_inverse=True
            )
            j_pos_unique, inversed_j = candidate_j.unique(
                sorted=True, return_inverse=True
            )
            conf_all = torch.zeros(
                size=(candidate_i.shape[0],),
                dtype=sim_matrix.dtype,
                device=candidate_i.device,
            )
            for b in b_pos_unique:
                row_pos = F.softmax(sim_matrix[b, i_pos_unique, :], 1)
                col_pos = F.softmax(sim_matrix[b, :, j_pos_unique], 0)
                conf = (
                    row_pos[inversed_i, j_pos_unique[inversed_j]]
                    * col_pos[i_pos_unique[inversed_i], inversed_j]
                ).to(conf_all.dtype)
                conf_all[b_pos_unique[inversed_b] == b] = conf

            top_k = min(
                self.max_coarse_matches,
                conf_all.shape[0],
            )
            top_k_values, top_k_indices = torch.topk(conf_all, k=top_k)
            valid_matches = top_k_values > (top_k_values[0] * self.coarse_match_thres)

            b_idx_c = candidate_b[top_k_indices][valid_matches]
            i_idx_c = candidate_i[top_k_indices][valid_matches]
            j_idx_c = candidate_j[top_k_indices][valid_matches]

        scale0 = (
            data["coarse_scale"] * data["scale0"][b_idx_c]
            if "scale0" in data
            else data["coarse_scale"]
        )
        scale1 = (
            data["coarse_scale"] * data["scale1"][b_idx_c]
            if "scale1" in data
            else data["coarse_scale"]
        )
        W0 = data["hw0_c"][1]
        W1 = data["hw1_c"][1]
        coarse_coord_0 = (
            torch.stack(
                (
                    (i_idx_c % W0),
                    (i_idx_c // W0),
                ),
                dim=1,
            )
            * scale0
        )
        coarse_coord_1 = (
            torch.stack(
                (
                    (j_idx_c % W1),
                    (j_idx_c // W1),
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
            }
        )

    @torch.no_grad
    def _coarse_match_norm_clue(
        self,
        data: dict,
        feat0_c: torch.Tensor,
        feat1_c: torch.Tensor,
        mask0: torch.Tensor = None,
        mask1: torch.Tensor = None,
    ):
        B, C, H, W = feat0_c.shape
        feat0_flatten = rearrange(feat0_c, "b c h w -> b (h w) c") / C**0.5  # B, Lc0, C
        feat1_flatten = rearrange(feat1_c, "b c h w -> b (h w) c") / C**0.5  # B, Lc1, C

        # 1. Get norm
        norm0 = torch.norm(feat0_flatten, dim=-1)  # B, Lc0
        norm1 = torch.norm(feat1_flatten, dim=-1)  # B, Lc1

        # 2. Filter norm by mask if exists
        if mask0 is not None and mask1 is not None:
            norm0.masked_fill_(~mask0, 0.0)
            norm1.masked_fill_(~mask1, 0.0)

        # 3. NMS on norm, only on single image !!!
        # norm0 = norm0.view(B, 1, H, W).contiguous()
        # norm0 = self.simple_nms(norm0, 1)
        # norm0 = norm0.flatten(1)

        b_idx_c = []
        i_idx_c = []
        j_idx_c = []
        for b in range(B):
            # 4. Get candidate match for correlation
            candidate_i = torch.where(norm0[b] > norm0[b].max() * 0.65)[0]
            candidate_j = torch.where(norm1[b] > norm1[b].max() * 0.65)[0]
            # candidate_i = torch.where(norm0[b] < norm0[b].max() * 0.6)[0]
            # candidate_j = torch.where(norm1[b] < norm1[b].max() * 0.6)[0]
            min_len = min(candidate_i.shape[0], candidate_j.shape[0])

            if min_len == 0:
                candidate_i = torch.where(norm0[b] > norm0[b].mean())[0]
                candidate_j = torch.where(norm1[b] > norm0[b].mean())[0]
                # candidate_i = torch.where(norm0[b] < norm0[b].mean())[0]
                # candidate_j = torch.where(norm1[b] < norm0[b].mean())[0]
                min_len = min(candidate_i.shape[0], candidate_j.shape[0])
                if min_len == 0:
                    b_idx_c.append(
                        torch.zeros(
                            size=(0,),
                            device=norm0.device,
                            dtype=torch.long,
                        )
                    )
                    i_idx_c.append(
                        torch.zeros(
                            size=(0,),
                            device=norm0.device,
                            dtype=torch.long,
                        )
                    )
                    j_idx_c.append(
                        torch.zeros(
                            size=(0,),
                            device=norm0.device,
                            dtype=torch.long,
                        )
                    )
                    continue
            # Random pick
            # N = norm1[b].shape[0]
            # candidate_i = torch.randperm(N, device=norm1.device)[: N // 3]
            # candidate_j = torch.where(norm1[b] > 0.0)[0]
            # min_len = min(candidate_i.shape[0], candidate_j.shape[0])

            # 5. Correlation & DS
            sim_matrix = torch.einsum(
                "lc,sc->ls",
                feat0_flatten[b, candidate_i].contiguous(),
                feat1_flatten[b, candidate_j].contiguous(),
            ) / (self.coarse_temperature + 1e-8)
            conf_matrix = F.softmax(sim_matrix, 0) * F.softmax(sim_matrix, 1)

            # 6. Topk
            top_k = min(
                self.max_coarse_matches // B,
                min_len,
            )
            flat_conf = conf_matrix.view(-1)
            top_k_values, top_k_indices = torch.topk(flat_conf, k=top_k)

            # 7. Apply threshold
            valid_matches = top_k_values > (conf_matrix.max() * self.coarse_match_thres)
            coarse_match = top_k_indices[valid_matches]

            # 8. Manually calculate original indices
            b_idx_c.append(
                torch.ones(
                    size=(coarse_match.shape[0],),
                    device=coarse_match.device,
                    dtype=torch.long,
                )
                * b
            )
            i_idx_c.append(candidate_i[coarse_match // conf_matrix.shape[1]])
            j_idx_c.append(candidate_j[coarse_match % conf_matrix.shape[1]])
        b_idx_c = torch.concat(b_idx_c, dim=0)
        i_idx_c = torch.concat(i_idx_c, dim=0)
        j_idx_c = torch.concat(j_idx_c, dim=0)

        scale0 = (
            data["coarse_scale"] * data["scale0"][b_idx_c]
            if "scale0" in data
            else data["coarse_scale"]
        )
        scale1 = (
            data["coarse_scale"] * data["scale1"][b_idx_c]
            if "scale1" in data
            else data["coarse_scale"]
        )
        W0 = data["hw0_c"][1]
        W1 = data["hw1_c"][1]
        coarse_coord_0 = (
            torch.stack(
                (
                    (i_idx_c % W0),
                    (i_idx_c // W0),
                ),
                dim=1,
            )
            * scale0
        )
        coarse_coord_1 = (
            torch.stack(
                (
                    (j_idx_c % W1),
                    (j_idx_c // W1),
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
            }
        )

    def _fine_correlation(self, data: dict):
        b_idx_c = data["b_idx_c"]  # M
        feat0 = data["feat0_f"]  # B, C, H, W
        feat1 = data["feat1_f"]  # B, C, H, W

        C = feat0.shape[1]
        absolute_fine_scale0 = (
            data["fine_scale"] * data["scale0"][b_idx_c]
            if "scale0" in data
            else data["fine_scale"]
        )
        absolute_fine_scale1 = (
            data["fine_scale"] * data["scale1"][b_idx_c]
            if "scale1" in data
            else data["fine_scale"]
        )

        # 1. Get feature windows from both feature maps
        window_size = int(data["coarse_scale"] / data["fine_scale"])
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
        sim_matrix = torch.einsum("blc,bsc->bls", feat0_window, feat1_window) / (
            self.fine_temperature + 1e-8
        )  # M, Lf0, Lf1

        # 5. Update sim matrix
        data.update({"sim_matrix_f": sim_matrix})

    @torch.no_grad
    def _get_fine_coord_train(self, data: dict):
        N = data["spv_b_ids_it"].shape[0]

        if N > self.max_fine_matches:
            perm = torch.randperm(N, device=data["spv_b_ids_it"].device)
            idx = perm[: self.max_fine_matches]
        else:
            idx = slice(None)

        intermediate_coord_1 = data["intermediate_coord_1_gt"][idx].clone()
        noise_scale = self.train_noise_scale
        image_scale = data["scale1"][data["spv_b_ids_it"][idx]]
        noise = (
            (
                torch.randn(
                    intermediate_coord_1.shape[0], 2, device=intermediate_coord_1.device
                )
            )
            * noise_scale
            * image_scale
        )
        noisy_coord_1 = intermediate_coord_1 + noise

        data.update(
            {
                "b_idx_it": data["spv_b_ids_it"][idx],
                "m_idx_it": data["spv_m_ids_it"][idx],
                "i_idx_it": data["spv_i_ids_it"][idx],
                "j_idx_it": data["spv_j_ids_it"][idx],
                "intermediate_coord_0": data["intermediate_coord_0_gt"][idx],
                "intermediate_coord_1": noisy_coord_1,
                "num_matches": noisy_coord_1.shape[0],
            }
        )

    @torch.no_grad
    def _get_fine_coord_test(self, data: dict):
        coarse_scale = data["coarse_scale"]
        fine_scale = data["fine_scale"]
        conf_matrix = data["sim_matrix_f"]
        b_idx_c = data["b_idx_c"]  # [M]
        if b_idx_c.shape[0] == 0:
            data.update(
                {
                    "b_idx_it": torch.zeros(
                        size=(0,), device=b_idx_c.device, dtype=torch.long
                    ),
                    "m_idx_it": torch.zeros(
                        size=(0,), device=b_idx_c.device, dtype=torch.long
                    ),
                    "i_idx_it": torch.zeros(
                        size=(0,), device=b_idx_c.device, dtype=torch.long
                    ),
                    "j_idx_it": torch.zeros(
                        size=(0,), device=b_idx_c.device, dtype=torch.long
                    ),
                    "intermediate_coord_0": torch.zeros(
                        size=(0, 2), device=b_idx_c.device, dtype=torch.float32
                    ),
                    "intermediate_coord_1": torch.zeros(
                        size=(0, 2), device=b_idx_c.device, dtype=torch.float32
                    ),
                }
            )
            return
        absolute_fine_scale0 = (
            fine_scale * data["scale0"][b_idx_c] if "scale0" in data else fine_scale
        )  # [M, 2]
        absolute_fine_scale1 = (
            fine_scale * data["scale1"][b_idx_c] if "scale1" in data else fine_scale
        )  # [M, 2]
        coarse_coord_0, coarse_coord_1 = data["coarse_coord_0"], data["coarse_coord_1"]
        window_size = int(coarse_scale / fine_scale)  # w

        # Dual-softmax
        conf_matrix = F.softmax(conf_matrix, 1) * F.softmax(conf_matrix, 2)
        coarse_coord_0 = (coarse_coord_0 / absolute_fine_scale0).round() + 0.5  # [M, 2]
        coarse_coord_1 = (coarse_coord_1 / absolute_fine_scale1).round() + 0.5  # [M, 2]

        # Get top k matches for each coarse match
        flat_conf = conf_matrix.view(conf_matrix.shape[0], -1)  # [M, Lf0*Lf1]
        top_k = int(
            min(
                self.max_fine_matches / conf_matrix.shape[0],
                conf_matrix.shape[-1],
            )
        )
        top_k_values, top_k_indices = torch.topk(flat_conf, k=top_k, dim=1)  # [M, K]

        # Apply threshold
        valid_matches = top_k_values > (
            # self.fine_match_thres
            conf_matrix.max() * self.fine_match_thres
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

        absolute_fine_scale0 = (
            absolute_fine_scale0[b_idx_it]
            if isinstance(absolute_fine_scale0, torch.Tensor)
            else absolute_fine_scale0
        )
        absolute_fine_scale1 = (
            absolute_fine_scale1[b_idx_it]
            if isinstance(absolute_fine_scale1, torch.Tensor)
            else absolute_fine_scale1
        )
        intermediate_coord_0 = (
            torch.stack(
                (
                    (i_idx_it % window_size),
                    (i_idx_it // window_size),
                ),
                dim=1,
            )
            + coarse_coord_0[m_idx_it]
        ) * absolute_fine_scale0

        intermediate_coord_1 = (
            torch.stack(
                (
                    (j_idx_it % window_size),
                    (j_idx_it // window_size),
                ),
                dim=1,
            )
            + coarse_coord_1[m_idx_it]
        ) * absolute_fine_scale1

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

    def _refinement(self, data: dict, print_mem: bool = False):
        feat0 = data["feat0_f"]  # B, C, H, W
        feat1 = data["feat1_f"]  # B, C, H, W
        b_idx_it = data["b_idx_it"]  # N
        scale0 = (
            data["scale0"][b_idx_it] * data["fine_scale"]
            if "scale0" in data
            else data["fine_scale"]
        )  # N, 2
        scale1 = (
            data["scale1"][b_idx_it] * data["fine_scale"]
            if "scale1" in data
            else data["fine_scale"]
        )  # N, 2
        fine_coord_0 = data["intermediate_coord_0"]  # N, 2

        # 0. Padding
        pad_len = self.max_fine_matches - min(
            self.max_fine_matches, fine_coord_0.shape[0]
        )
        non_pad_len = self.max_fine_matches - pad_len
        pad_feat = (
            torch.zeros(
                size=(
                    pad_len,
                    feat0.shape[1],
                    # self.lookup_window_size**2,
                    self.lookup_window_size,
                    self.lookup_window_size,
                ),
                dtype=feat0.dtype,
                device=feat0.device,
            )
            if pad_len
            else None
        )

        # 1. Get feature window in image0 using intermediate matches
        feat0_window = self._extract_feat_window_bilinear(
            feat0, b_idx_it, fine_coord_0 / scale0, self.lookup_window_size
        )  # [N, C, H, W]
        feat0_window = (
            torch.concat([feat0_window, pad_feat], dim=0) if pad_len else feat0_window
        )  # [Nmax, C, H, W]

        # 2. Iteratively refine coords on image 1
        fine_coord_1 = (data["intermediate_coord_1"] / scale1).requires_grad_(
            False
        )  # N, 2
        hidden_state = torch.zeros(
            (
                self.max_fine_matches,
                self.recurrent_refine_unit.hidden_dim,
                self.lookup_window_size,
                self.lookup_window_size,
            ),
            device=feat0.device,
            dtype=feat0.dtype,
        )  # Initial hidden state, [Nmax, C, H, W]

        all_offset_1 = []
        for i in range(self.refine_iters):
            # 4.1 Get feature from coords
            feat1_window = self._extract_feat_window_bilinear(
                feat1, b_idx_it, fine_coord_1, self.lookup_window_size
            )  # [N, C, H, W]
            feat1_window = (
                torch.concat([feat1_window, pad_feat], dim=0)
                if pad_len
                else feat1_window
            )  # [Nmax, C, H, W]

            # corr = torch.einsum(
            #     "bcl,bcs->bls", feat0_window.flatten(-2), feat1_window.flatten(-2)
            # )
            # corr = corr.view(
            #     corr.shape[0],
            #     corr.shape[1],
            #     self.lookup_window_size,
            #     self.lookup_window_size,
            # )
            # corr = torch.concat([corr, pad_feat], dim=0) if pad_len else corr

            # 4.2 Both features enter recurrent refinement unit
            offset_f_1, hidden_state = self.recurrent_refine_unit(
                feat0_window, feat1_window, hidden_state
            )  # [Nmax, 2], [Nmax, 1, C]
            # offset_f_1, hidden_state = self.recurrent_refine_unit(
            #     corr, hidden_state
            # )  # [Nmax, 2], [Nmax, 1, C]
            offset_f_1 = offset_f_1[:non_pad_len]

            all_offset_1.append(offset_f_1)

            # 4.3 Update refined coords
            fine_coord_1 += offset_f_1 * self.refine_lookup_radius
            if print_mem:
                if torch.cuda.is_available():
                    print(
                        f"Iter {i} memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
                    )

        all_offset_1 = torch.stack(all_offset_1, dim=0)  # [ITER, N, 2]

        data.update(
            {
                "fine_coord_0": fine_coord_0,
                "fine_coord_1": fine_coord_1 * scale1,
                "all_offset_1": all_offset_1,
            }
        )

    @staticmethod
    @torch.jit.script
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
        windows = windows.permute(0, 3, 1, 2)

        return windows

    @staticmethod
    @torch.jit.script
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
            / torch.tensor([_feat.shape[3], _feat.shape[2]], device=_feat.device)
            * 2
        ) - 1

        # Extract windows using bilinear sampling
        b_idx_unique = torch.unique(b_idx)
        windows = torch.zeros(
            (b_idx.shape[0], _feat.shape[1], window_size, window_size),
            device=_feat.device,
            dtype=_feat.dtype,
        )
        for b in b_idx_unique:
            mask = b_idx == b
            # Concat all the grids with the same batch index
            grid_b = grid[mask]  # m x Hout x Wout x 2
            m = grid_b.shape[0]
            feat_b = _feat[b : b + 1]  # 1 x C x Hin x Win
            grid_b = grid_b.reshape(1, -1, window_size, 2)  # 1 x (m x Hout) x Wout x 2

            window = F.grid_sample(
                feat_b,
                grid_b,
                mode="bilinear",
                align_corners=False,
                padding_mode="zeros",
            )  # 1 x C x (m x Hout) x Wout
            window = window.view(-1, m, window_size, window_size).permute(
                1, 0, 2, 3
            )  # m x C x Hout x Wout
            windows[mask] = window.to(windows.dtype)

        return windows

    @staticmethod
    @torch.jit.script
    def simple_nms(scores, nms_radius: int):
        """Fast Non-maximum suppression to remove nearby points"""
        """ Code from https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superpoint.py """
        assert nms_radius >= 0
        nms_kernel_size = nms_radius * 2 + 1

        zeros = torch.zeros_like(scores)
        max_mask = scores == F.max_pool2d(
            scores, kernel_size=nms_kernel_size, stride=1, padding=nms_radius
        )
        for _ in range(2):
            supp_mask = (
                F.max_pool2d(
                    max_mask.float(),
                    kernel_size=nms_kernel_size,
                    stride=1,
                    padding=nms_radius,
                )
                > 0
            )
            supp_scores = torch.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == F.max_pool2d(
                supp_scores,
                kernel_size=nms_kernel_size,
                stride=1,
                padding=nms_radius,
            )
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, zeros)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith("soma."):
                state_dict[k.replace("soma.", "", 1)] = state_dict.pop(k)
        # for k in list(state_dict.keys()):
        #     if k.startswith("coarse_encoder"):
        #         state_dict[
        #             k.replace("coarse_encoder", "feature_backbone.stages.1.blocks.1", 1)
        #         ] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)

    def initial_forward(self):
        if hasattr(self.feature_backbone, "initial_forward"):
            self.feature_backbone.initial_forward()
        if hasattr(self.recurrent_refine_unit, "initial_forward"):
            self.recurrent_refine_unit.initial_forward(
                length=self.max_fine_matches, win_size=self.lookup_window_size
            )
