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
        # Positive & Negative weight
        self.c_pos_w = self.config["POS_WEIGHT"]
        self.c_neg_w = self.config["NEG_WEIGHT"]
        self.correct_thr = self.config["FINE_THR"]

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.0
            c_pos_w = 0.0
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.0
            c_neg_w = 0.0

        # Cross Entropy Loss
        if self.config["COARSE_TYPE"] == "cross_entropy":
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            loss_pos = -torch.log(conf[pos_mask])
            loss_neg = -torch.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()

        # Focal Loss
        elif self.config["COARSE_TYPE"] == "focal":
            # conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            alpha = self.config["FOCAL_ALPHA"]
            gamma = self.config["FOCAL_GAMMA"]

            pos_conf = conf[pos_mask]
            loss_pos = (
                -alpha
                * torch.pow(1 - pos_conf, gamma)
                * torch.clamp_min(pos_conf, 1e-6).log()
            )

            # handle loss weights
            if weight is not None:
                # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                loss_pos = loss_pos * weight[pos_mask]

            loss = c_pos_w * loss_pos.mean()
            return loss

        else:
            raise ValueError(
                "Unknown coarse loss: {type}".format(type=self.config["coarse_type"])
            )

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt, std):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
            std (torch.Tensor): [M]
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = (
            torch.linalg.norm(expec_f_gt, ord=float("inf"), dim=1) < self.correct_thr
        )

        # use std as weight that measures uncertainty
        inverse_std = 1.0 / torch.clamp(std, min=1e-10)
        weight = (
            inverse_std / torch.mean(inverse_std)
        ).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if not correct_mask.any():
            if (
                self.training
            ):  # this seldomly happen during training, since we pad prediction with gt
                # sometimes there is not coarse-level gt at all.
                correct_mask[0] = True
                weight[0] = 0.0
            else:
                return None

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss

    @torch.no_grad()
    def compute_c_weight(self, data):
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
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. Coarse-level loss
        # Get conf matrix from sim matrix using dual softmax operator
        sim_matrix = data["sim_matrix"]
        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        data.update({"conf_matrix": conf_matrix})

        loss_c = self.compute_coarse_loss(
            data["conf_matrix"],
            data["conf_matrix_gt"],
            weight=c_weight,
        )
        loss: torch.Tensor = loss_c * self.config["COARSE_WEIGHT"]
        loss_scalars.update({"loss_c": loss.clone().detach().cpu()})

        # 2. Fine-level loss
        if self.config["FINE_WEIGHT"] is not None:
            loss_f = self._compute_fine_loss_l2_std(
                expec_f=data["expec_f"],
                expec_f_gt=data["expec_f_gt"],
                std=data["std"],
            )
            if loss_f is not None:
                loss += loss_f * self.config["FINE_WEIGHT"]
                loss_scalars.update({"loss_f": loss_f.clone().detach().cpu()})
            else:
                loss_scalars.update({"loss_f": torch.tensor(0.0).clone().detach().cpu()})

        # 3. Total loss
        data.update({"loss": loss, "loss_scalars": loss_scalars})
