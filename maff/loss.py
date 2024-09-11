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
            loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * torch.clamp_min(pos_conf, 1e-6).log()

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

        # 3. Total loss
        data.update({"loss": loss, "loss_scalars": loss_scalars})
