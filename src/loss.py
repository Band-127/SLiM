from yacs.config import CfgNode as CN
import torch
from torch import nn
from torch.nn import functional as F


class RCRM_Loss(nn.Module):
    def __init__(self, config: CN):
        """
        Args:
            config (CN): root.LOSS configuration path
        """
        super(RCRM_Loss, self).__init__()

        self.config = config
        self.turn_off_conf_mask_loss = not self.config["CONF_MASK_DEPTH_REFINEMENT"]
        self.coarse_weight = self.config["COARSE_WEIGHT"]
        self.intermediate_weight = self.config["INTERMEDIATE_WEIGHT"]
        self.fine_weight = self.config["FINE_WEIGHT"]
        self.iter_decay_gamma = self.config["ITER_DECAY_GAMMA"]
        self.version = self.config["VERSION"]
        self.coarse_alpha = self.config["FOCAL_ALPHA_COARSE"]
        self.coarse_gamma = self.config["FOCAL_GAMMA_COARSE"]
        self.intermediate_alpha = self.config["FOCAL_ALPHA_INTERMEDIATE"]
        self.intermediate_gamma = self.config["FOCAL_GAMMA_INTERMEDIATE"]

    def compute_coarse_loss(self, data):
        conf = data["conf_matrix"]
        conf_gt = data["conf_matrix_gt"]

        # 0. Compute element-wise loss weight and pos neg mask
        weight = self._compute_c_weight(data)
        pos_mask = conf_gt == 1
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            conf_gt = conf
            loss = (conf - conf_gt).mean()
            return loss

        # 1. Compute loss for similarity matrix in coarse matching
        # Focal Loss
        alpha = self.coarse_alpha
        gamma = self.coarse_gamma

        pos_conf = conf[pos_mask]
        loss = (
            -alpha
            * torch.pow(1 - pos_conf, gamma)
            * torch.clamp_min(pos_conf, 1e-6).log()
        )

        # handle loss weights
        if weight is not None:
            # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
            # but only through manually setting corresponding regions in sim_matrix to '-inf'.
            loss = loss * weight[pos_mask]
        loss = loss.mean()

        # 2. Compute loss for confidence mask with gt depth map(binary classification for background/foreground)
        if not self.turn_off_conf_mask_loss:
            loss += self._compute_conf_loss(data)

        return loss

    def compute_fine_loss_v1(self, data):
        # 1. Compute FOCAL loss for similarity matrix in fine matching
        conf_matrix_f = data["conf_matrix_f"]
        conf_matrix_f_gt = data["conf_matrix_f_gt"]
        pos_mask = conf_matrix_f_gt == 1
        alpha = self.intermediate_alpha
        gamma = self.intermediate_gamma
        pos_conf = conf_matrix_f[pos_mask]
        loss_f1 = (
            -alpha
            * torch.pow(1 - pos_conf, gamma)
            * torch.clamp_min(pos_conf, 1e-6).log()
        ).mean()

        # 2. Compute L2 loss for offset prediction in fine matching
        offset = data["all_offset_1"]  # [ITER, M, 2]
        offset_gt = data["coord_offset_gt"]  # [M, 2]
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = data["correct_mask"]  # [M]

        # corner case: no correct fine match found
        if not correct_mask.any():
            loss_f2 = None
        else:
            loss_f2 = torch.tensor(0.0, device=offset.device)
            iters = offset.shape[0]
            total_offset = torch.zeros_like(offset[0])
            for i in range(iters):
                loss_f2 += self.iter_decay_gamma ** (
                    iters - i - 1
                ) * self._compute_fine_loss_l2(
                    offset[i] + total_offset, offset_gt, correct_mask
                )
                with torch.no_grad():
                    total_offset += offset[i].detach()
        return loss_f1, loss_f2

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

    def _compute_conf_loss(self, data):
        """
        L2 loss between confidence mask and gt depth map
        Args:
            data (dict): input data
        """
        conf_mask0 = data["conf_mask0"]
        conf_mask1 = data["conf_mask1"]
        supervision_depth0 = data["supervision_depth0"]
        supervision_depth1 = data["supervision_depth1"]

        loss = ((conf_mask0 - supervision_depth0) ** 2).mean() + (
            (conf_mask1 - supervision_depth1) ** 2
        ).mean()

        return 0.5 * loss

    @torch.no_grad()
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

    def forward_v1(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # Get conf matrix from sim matrix using dual softmax operator
        sim_matrix = data["sim_matrix"]
        sim_matrix_f = data["sim_matrix_f"]
        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        conf_matrix_f = F.softmax(sim_matrix_f, 1) * F.softmax(sim_matrix_f, 2)
        data.update({"conf_matrix": conf_matrix})
        data.update({"conf_matrix_f": conf_matrix_f})
        loss = torch.tensor(0.0, device=sim_matrix.device)

        # 1. Coarse-level loss
        loss_c = self.compute_coarse_loss(data)
        if loss_c is not None:
            loss += loss_c * self.coarse_weight
            loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})
        else:
            loss_scalars.update({"loss_c": torch.tensor(0.0).clone().detach().cpu()})

        # 2. Fine-level loss
        loss_f1, loss_f2 = self.compute_fine_loss_v1(data=data)
        if loss_f1 is not None and loss_f1.item() != torch.nan:
            loss += loss_f1 * self.intermediate_weight
            loss_scalars.update({"loss_f1": loss_f1.clone().detach().cpu()})
        else:
            loss_scalars.update({"loss_f1": torch.tensor(0.0).clone().detach().cpu()})
        if loss_f2 is not None and loss_f2.item() != torch.nan:
            loss += loss_f2 * self.fine_weight
            loss_scalars.update({"loss_f2": loss_f2.clone().detach().cpu()})
        else:
            loss_scalars.update({"loss_f2": torch.tensor(0.0).clone().detach().cpu()})

        # 3. Total loss
        data.update({"loss": loss, "loss_scalars": loss_scalars})

    def forward(self, data):
        if self.version == "v1":
            self.forward_v1(data)
