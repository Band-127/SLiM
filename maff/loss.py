from yacs.config import CfgNode as CN
import torch
from torch import nn
from torch.nn import functional as F


class MAFF_Loss(nn.Module):
    def __init__(self, config: CN):
        """
        Args:
            config (CN): root.LOSS configuration path
        """
        super(MAFF_Loss, self).__init__()

        self.config = config
        self.correct_thr = self.config["FINE_THR"]
        self.turn_off_fine_feature_likelihood = False
        self.turn_off_conf_mask_loss = not self.config["CONF_MASK_DEPTH_REFINEMENT"]
        self.coarse_weight = self.config["COARSE_WEIGHT"]
        self.fine_weight = self.config["FINE_WEIGHT"]

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
        alpha = self.config["FOCAL_ALPHA"]
        gamma = self.config["FOCAL_GAMMA"]

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

    def compute_fine_loss(self, data):
        # 1. Compute loss for offset prediction in fine matching
        coord_offset = data["coord_offset_f"]
        coord_offset_gt = data["coord_offset_f_gt"]
        std = data["std"]
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = (
            torch.linalg.norm(coord_offset_gt, ord=float("inf"), dim=1)
            < self.correct_thr
        )

        # corner case: no correct fine match found
        if not correct_mask.any():
            return None, None
        else:
            offset_loss = None
            if self.config["FINE_TYPE"] == "l2_std":
                offset_loss = self._compute_fine_loss_l2_std(
                    coord_offset, coord_offset_gt, std, correct_mask
                )
            elif self.config["FINE_TYPE"] == "l2":
                offset_loss = self._compute_fine_loss_l2(
                    coord_offset, coord_offset_gt, correct_mask
                )

            # 2. Compute loss for log-likelihood using similarity matrix in fine matching
            likelihood_loss = None
            if not self.turn_off_fine_feature_likelihood:
                sim_matrix_f = data["sim_matrix_f"]
                sim_matrix_f_gt = data["sim_matrix_f_gt"]
                likelihood_loss = self._compute_fine_loss_feature_likelihood(
                    sim_matrix_f, sim_matrix_f_gt, correct_mask
                )
            return offset_loss, likelihood_loss

    def _compute_fine_loss_feature_likelihood(
        self, sim_matrix_f, sim_matrix_f_gt, correct_mask
    ):
        """
        Args:
            sim_matrix_f (torch.Tensor): (M, W, W) fine similarity matrix formed in fine matching
            sim_matrix_f_gt (torch.Tensor): (M, W, W) fine similarity matrix ground truth
            correct_mask (torch.Tensor): (M)
        """
        sim_matrix_f = sim_matrix_f[correct_mask]
        sim_matrix_f_gt = sim_matrix_f_gt[correct_mask]

        pos_mask = sim_matrix_f_gt == 1
        # corner case: no gt fine-level match at all
        if not pos_mask.any():
            return None
        alpha = self.config["FOCAL_ALPHA"]
        gamma = self.config["FOCAL_GAMMA"]

        pos_sim = sim_matrix_f[pos_mask]
        loss = (
            -alpha
            * torch.pow(1 - pos_sim, gamma)
            * torch.clamp_min(pos_sim, 1e-6).log()
        ).mean()
        return loss

    def _compute_fine_loss_l2_std(
        self, coord_offset_f, coord_offset_f_gt, std, correct_mask
    ):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
            std (torch.Tensor): [M]
            correct_mask (torch.Tensor): (M)
        """
        # use std as weight that measures uncertainty
        inverse_std = 1.0 / torch.clamp(std, min=1e-10)
        weight = (
            inverse_std / torch.mean(inverse_std)
        ).detach()  # avoid minizing loss through increase std
        # l2 loss with std
        offset_l2 = (
            (coord_offset_f_gt[correct_mask] - coord_offset_f[correct_mask]) ** 2
        ).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()
        return loss

    def _compute_fine_loss_l2(self, coord_offset_f, coord_offset_f_gt, correct_mask):
        """
        Args:
            coord_offset_f (torch.Tensor): [M, 2] <x, y>
            coord_offset_f_gt (torch.Tensor): [M, 2] <x, y>
            correct_mask (torch.Tensor): (M)
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

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 1. Coarse-level loss
        # Get conf matrix from sim matrix using dual softmax operator
        sim_matrix = data["sim_matrix"]
        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        data.update({"conf_matrix": conf_matrix})
        loss = torch.tensor(0.0, device=sim_matrix.device)

        # similarity loss
        loss_c = self.compute_coarse_loss(data)
        if loss_c is not None:
            loss += loss_c * self.coarse_weight
            loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})
        else:
            loss_scalars.update({"loss_c": torch.tensor(0.0).clone().detach().cpu()})

        # 2. Fine-level loss
        loss_f = torch.tensor(0.0, device=sim_matrix.device)
        loss_f1, loss_f2 = self.compute_fine_loss(data=data)
        if loss_f1 is not None:
            loss_f += loss_f1
            loss_scalars.update({"loss_f1": loss_f1.clone().detach().cpu()})
        else:
            loss_scalars.update({"loss_f1": torch.tensor(0.0).clone().detach().cpu()})
        if loss_f2 is not None:
            loss_f += loss_f2
            loss_scalars.update({"loss_f2": loss_f2.clone().detach().cpu()})
        else:
            loss_scalars.update({"loss_f2": torch.tensor(0.0).clone().detach().cpu()})
        
        loss += loss_f * self.fine_weight

        # 3. Total loss
        data.update({"loss": loss, "loss_scalars": loss_scalars})
