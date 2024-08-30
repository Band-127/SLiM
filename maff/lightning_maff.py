from yacs.config import CfgNode as CN
from loguru import logger
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR
import pytorch_lightning as pl
from pytorch_lightning.profilers.profiler import Profiler
from pytorch_lightning.profilers import PassThroughProfiler

from maff.maff import MAFF
from maff.utils.supervision import compute_supervision_coarse
from maff.loss import MAFF_Loss


class PL_MAFF(pl.LightningModule):
    def __init__(
        self,
        config: CN,
        pretrained_ckpt: str = None,
        profiler: Profiler = None,
        dump_dir: str = None,
    ):
        """_summary_

        Args:
            config (CN): Root configuration
            pretrained_ckpt (str, optional): Path to pretrained checkpoint if exists. Defaults to None.
            profiler (Profiler, optional): Pytorch Lightning Profiler module. Defaults to None.
            dump_dir (str, optional): Dir to dump testing output. Defaults to None.
        """
        super(PL_MAFF, self).__init__()

        self.config = config
        self.pretrained_ckpt = pretrained_ckpt
        self.profiler = profiler or PassThroughProfiler()
        self.dump_dir = dump_dir

        self.maff = MAFF(config=config["MODEL"])
        self.loss = MAFF_Loss(config=config["LOSS"])

        # Read pretrained checkpoint if exists
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location="cpu")["state_dict"]
            self.maff.load_state_dict(state_dict, strict=True)
            logger.info(f"Load '{pretrained_ckpt}' as pretrained checkpoint")

        # Coarse scale
        self.coarse_scale = self.config.MODEL.COARSE_SCALE

        self.validation_outputs = []
        
        self.lr = self.config.TRAINER.TRUE_LR

    def configure_optimizers(self):
        # optimizer
        if self.config.TRAINER.OPTIMIZER == "Adam":
            optimizer = Adam(
                self.parameters(),
                lr=self.config.TRAINER.TRUE_LR,
                weight_decay=self.config.TRAINER.ADAM_DECAY,
            )
        elif self.config.TRAINER.OPTIMIZER == "AdamW":
            optimizer = AdamW(
                self.parameters(),
                lr=self.config.TRAINER.TRUE_LR,
                weight_decay=self.config.TRAINER.ADAMW_DECAY,
            )
        else:
            # Default: AdamW
            optimizer = AdamW(
                self.parameters(),
                lr=self.config.TRAINER.TRUE_LR,
                weight_decay=self.config.TRAINER.ADAMW_DECAY,
            )

        # learning rate scheduler
        scheduler = {"interval": self.config.TRAINER.SCHEDULER_INTERVAL}
        if self.config.TRAINER.SCHEDULER == "MultiStepLR":
            scheduler.update(
                {
                    "scheduler": MultiStepLR(
                        optimizer=optimizer,
                        milestones=self.config.TRAINER.MSLR_MILESTONES,
                        gamma=self.config.TRAINER.MSLR_GAMMA,
                    )
                }
            )
        elif self.config.TRAINER.SCHEDULER == "CosineAnnealing":
            scheduler.update(
                {
                    "scheduler": CosineAnnealingLR(
                        optimizer=optimizer, T_max=self.config.TRAINER.COSA_TMAX
                    )
                }
            )
        elif self.config.TRAINER.SCHEDULER == "ExponentialLR":
            scheduler.update(
                {
                    "scheduler": ExponentialLR(
                        optimizer=optimizer, gamma=self.config.TRAINER.ELR_GAMMA
                    )
                }
            )
        else:
            # Default: MultiStepLR
            scheduler.update(
                {
                    "scheduler": MultiStepLR(
                        optimizer=optimizer,
                        milestones=self.config.TRAINER.MSLR_MILESTONES,
                        gamma=self.config.TRAINER.MSLR_GAMMA,
                    )
                }
            )
        return [optimizer], [scheduler]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == "linear":
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + (
                    self.trainer.global_step / self.config.TRAINER.WARMUP_STEP
                ) * abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            elif self.config.TRAINER.WARMUP_TYPE == "constant":
                pass
            else:
                raise ValueError(
                    f"Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}"
                )
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, coarse_scale=self.coarse_scale)

        with self.profiler.profile("MAFF"):
            self.maff(batch)

        with self.profiler.profile("Compute losses"):
            self.loss(batch)

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        # logging
        if (
            self.trainer.global_rank == 0
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            # scalars
            for k, v in batch["loss_scalars"].items():
                self.logger.experiment.add_scalar(f"train/{k}", v, self.global_step)
            print(f"Loss: {batch['loss']}")

        return {"loss": batch["loss"]}

    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        self.validation_outputs.append(batch["loss"])

        return {
            "loss_scalars": batch["loss_scalars"],
        }

    def on_validation_epoch_end(self):
        # log
        total_loss = 0.0
        for l in self.validation_outputs:
            total_loss += l
        total_loss /= len(self.validation_outputs)
        self.log("loss", total_loss, sync_dist=True)

        self.validation_outputs.clear()
