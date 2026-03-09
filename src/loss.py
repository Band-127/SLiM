from yacs.config import CfgNode as CN
import torch
from torch import nn
from torch.nn import functional as F


class SoMa_Loss(nn.Module):
    def __init__(self, config: CN):
        """
        Args:
            config (CN): root.LOSS configuration path
        """
        super(SoMa_Loss, self).__init__()

        self.config = config
        self.coarse_weight = self.config["COARSE_WEIGHT"]
        self.coarse_gamma = self.config["FOCAL_GAMMA_COARSE"]
        self.coarse_percent = self.config["COARSE_PERCENT"]

        self.fine_weight = self.config["FINE_WEIGHT"]
        self.fine_gamma = self.config["FOCAL_GAMMA_FINE"]

        self.refine_weight = self.config["REFINE_WEIGHT"]
        self.refine_thres = self.config["REFINE_THRES"]
        self.iter_decay_gamma = self.config["ITER_DECAY_GAMMA"]

    def compute_coarse_loss(self, data):
        conf = data["sim_matrix"]
        conf_gt = data["conf_matrix_gt"]
        gamma = self.coarse_gamma

        # 0. Compute element-wise loss weight and pos neg mask
        weight = self._compute_c_weight(data)
        # corner case: no gt coarse-level match at all
        if not conf_gt.any():  # assign a wrong gt
            conf_gt = conf
            loss = (conf - conf_gt).mean()
            return loss

        # 1. Get partial of all gt positive pairs
        b_pos, i_pos, j_pos = torch.where(conf_gt == 1)
        N = b_pos.shape[0]
        actual_count = int(N * self.coarse_percent)
        idx = torch.randperm(N, device=conf_gt.device)[:actual_count]
        b_pos, i_pos, j_pos = b_pos[idx], i_pos[idx], j_pos[idx]

        # 2. Get the partial softmax
        b_pos_unique, inversed_b = b_pos.unique(sorted=True, return_inverse=True)
        i_pos_unique, inversed_i = i_pos.unique(sorted=True, return_inverse=True)
        j_pos_unique, inversed_j = j_pos.unique(sorted=True, return_inverse=True)
        loss = torch.zeros(inversed_i.shape[0], device=conf_gt.device)
        for b in b_pos_unique:
            row_pos = F.softmax(conf[b, i_pos_unique, :], 1)
            col_pos = F.softmax(conf[b, :, j_pos_unique], 0)
            pos_conf = (
                row_pos[inversed_i, j_pos_unique[inversed_j]]
                * col_pos[i_pos_unique[inversed_i], inversed_j]
            )

            loss[b_pos_unique[inversed_b] == b] = (
                -1
                * torch.pow(1 - pos_conf, gamma)
                * torch.clamp_min(pos_conf, 1e-6).log()
            )

        # handle loss weights
        if weight is not None:
            # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
            # but only through manually setting corresponding regions in sim_matrix to '-inf'.
            loss = loss * weight[b_pos, i_pos, j_pos]
        loss = loss.mean()

        return loss

    def compute_fine_loss(self, data):
        # 1. Compute FOCAL loss for similarity matrix in fine matching
        conf_matrix_f = data["conf_matrix_f"]
        conf_matrix_f_gt = data["conf_matrix_f_gt"]
        pos_mask = conf_matrix_f_gt == 1
        gamma = self.fine_gamma
        pos_conf = conf_matrix_f[pos_mask]
        loss = (
            -1 * torch.pow(1 - pos_conf, gamma) * torch.clamp_min(pos_conf, 1e-6).log()
        ).mean()

        return loss

    def compute_refine_loss(self, data):
        # Compute L2 loss for offset prediction in fine matching
        offset = data["all_offset_1"]  # [ITER, M, 2]
        offset_gt = data["coord_offset_gt"]  # [M, 2]
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = data["correct_mask"]  # [M]

        # corner case: no correct fine match found
        if not correct_mask.any():
            loss = None
        else:
            loss = torch.tensor(0.0, device=offset.device)
            iters = offset.shape[0]
            total_offset = torch.zeros_like(offset[0])

            for i in range(iters):
                loss += self.iter_decay_gamma ** (
                    iters - i - 1
                ) * self._compute_fine_loss_l2(
                    offset[i] + total_offset, offset_gt, correct_mask
                )
                with torch.no_grad():
                    total_offset += offset[i].detach()
        return loss

    def _compute_fine_loss_l2(self, coord_offset_f, coord_offset_f_gt, correct_mask):
        """
        Args:
            coord_offset_f (torch.Tensor): [M, 2] <x, y>
            coord_offset_f_gt (torch.Tensor): [M, 2] <x, y>
            correct_mask (torch.Tensor): (M)
            normalize_scale (torch.Tensor): [M, 2]
        """
        offset_l2 = (
            (coord_offset_f_gt[correct_mask] - coord_offset_f[correct_mask]) ** 2
        ).sum(-1)
        return offset_l2.mean()

    @torch.no_grad
    def _compute_c_weight(self, data):
        if "mask0" in data:
            # Mask the area on similarity matrix where the mask==False
            c_weight = (
                data["mask0"].flatten(-2).unsqueeze(-1)
                * data["mask1"].flatten(-2).unsqueeze(-2)
            ).float()
        else:
            c_weight = None
        return c_weight

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # Get conf matrix from sim matrix using dual softmax operator, fine only, as the coarse one will conduct partial softmax
        sim_matrix_f = data["sim_matrix_f"]
        conf_matrix_f = F.softmax(sim_matrix_f, 1) * F.softmax(sim_matrix_f, 2)
        data.update({"conf_matrix_f": conf_matrix_f})
        loss = torch.tensor(0.0, device=conf_matrix_f.device)

        # 1. Coarse-level loss
        loss_c = self.compute_coarse_loss(data)
        if loss_c is not None:
            loss += loss_c * self.coarse_weight
            loss_scalars.update({"loss_c": loss_c.clone().detach().cpu() * 0.25})
        else:
            loss_scalars.update({"loss_c": torch.tensor(0.0).clone().detach().cpu()})

        # 2. Fine-level loss
        loss_f = self.compute_fine_loss(data=data)
        if loss_f is not None and loss_f.item() != torch.nan:
            loss += loss_f * self.fine_weight
            loss_scalars.update({"loss_f": loss_f.clone().detach().cpu() * 0.25})
        else:
            loss_scalars.update({"loss_f": torch.tensor(0.0).clone().detach().cpu()})

        # 3. Refinement loss
        loss_rf = self.compute_refine_loss(data=data)
        if loss_rf is not None and loss_rf.item() != torch.nan:
            loss += loss_rf * self.refine_weight
            loss_scalars.update({"loss_rf": loss_rf.clone().detach().cpu()})
        else:
            loss_scalars.update({"loss_rf": torch.tensor(0.0).clone().detach().cpu()})

        # 3. Total loss
        data.update({"loss": loss, "loss_scalars": loss_scalars})
