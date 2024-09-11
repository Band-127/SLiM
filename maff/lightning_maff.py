import pprint
import numpy as np
from yacs.config import CfgNode as CN
from collections import defaultdict
from pathlib import Path
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
from utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics,
)
from utils.comm import all_gather, gather
from utils.misc import flattenList
from utils.plotting import make_matching_figures


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
        self.lr = self.config.TRAINER.TRUE_LR

        self.training_outputs = []
        self.validation_outputs = []
        self.test_outputs = []

        self.n_vals_plot = max(
            config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1
        )

    def configure_optimizers(self):
        # optimizer
        if self.config.TRAINER.OPTIMIZER == "Adam":
            optimizer = Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.config.TRAINER.ADAM_DECAY,
            )
        elif self.config.TRAINER.OPTIMIZER == "AdamW":
            optimizer = AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.config.TRAINER.ADAMW_DECAY,
            )
        else:
            # Default: AdamW
            optimizer = AdamW(
                self.parameters(),
                lr=self.lr,
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
                self.lr = base_lr + (
                    self.trainer.global_step / self.config.TRAINER.WARMUP_STEP
                ) * abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = self.lr
            elif self.config.TRAINER.WARMUP_TYPE == "constant":
                pass
            else:
                raise ValueError(
                    f"Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}"
                )
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch, training: bool = True):
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, coarse_scale=self.coarse_scale)

        with self.profiler.profile("MAFF"):
            self.maff(batch, training=training)

        if training:
            with self.profiler.profile("Compute losses"):
                self.loss(batch)
            
        torch.cuda.empty_cache()

    def _compute_metrics(self, batch):
        with self.profiler.profile("Compute metrics"):
            # compute epi_errs for each match
            compute_symmetrical_epipolar_errors(batch)
            # compute R_errs, t_errs, pose_errs for each pair
            compute_pose_errors(batch, self.config)

            rel_pair_names = list(zip(*batch["pair_names"]))
            bs = batch["image0"].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                "identifiers": ["#".join(rel_pair_names[b]) for b in range(bs)],
                "epi_errs": [
                    batch["epi_errs"][batch["b_idx_c"] == b].cpu().numpy()
                    for b in range(bs)
                ],
                "R_errs": batch["R_errs"],
                "t_errs": batch["t_errs"],
                "inliers": batch["inliers"],
            }
            ret_dict = {"metrics": metrics}
        return ret_dict, rel_pair_names

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch, training=True)
        
        if self.global_rank == 0:
            print(f"Loss :{batch['loss'].item(): .6f}", end="")

        # logging
        if (
            self.trainer.global_rank == 0
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            # scalars
            for k, v in batch["loss_scalars"].items():
                self.logger.experiment.add_scalar(f"train/{k}", v, self.global_step)

            self.training_outputs.append(batch["loss"])
        
        return {"loss": batch["loss"]}

    def on_train_epoch_end(self):
        # log
        total_loss = 0.0
        for loss in self.training_outputs:
            total_loss += loss
        total_loss /= len(self.training_outputs)

        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                "train/avg_loss_on_epoch", total_loss, self.current_epoch
            )

        self.training_outputs.clear()
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch, training=False)

        ret_dict, _ = self._compute_metrics(batch)

        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(
                batch, self.config, mode=self.config.TRAINER.PLOT_MODE
            )

        self.validation_outputs.append(
            {
                **ret_dict,
                "figures": figures,
            }
        )

    def on_validation_epoch_end(self):
        # handle multiple validation sets
        multi_outputs = (
            [self.validation_outputs]
            if not isinstance(self.validation_outputs[0], (list, tuple))
            else self.validation_outputs
        )
        multi_val_metrics = defaultdict(list)

        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if self.trainer.sanity_checking:
                self.validation_outputs.clear()
                torch.cuda.empty_cache()
                return

            # 2. val metrics: dict of list, numpy
            _metrics = [o["metrics"] for o in outputs]
            metrics = {
                k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics])))
                for k in _metrics[0]
            }
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR
            )
            for thr in [5, 10, 20]:
                multi_val_metrics[f"auc@{thr}"].append(val_metrics_4tb[f"auc@{thr}"])

            # 3. figures
            _figures = [o["figures"] for o in outputs]
            figures = {
                k: flattenList(gather(flattenList([_me[k] for _me in _figures])))
                for k in _figures[0]
            }

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(
                        f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch
                    )

                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f"val_match_{valset_idx}/{k}/pair-{plot_idx}",
                                fig,
                                cur_epoch,
                                close=True,
                            )

        self.validation_outputs.clear()
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        # inference
        with self.profiler.profile("MAFF"):
            self.maff(batch, training=False)

        # metrics
        ret_dict, rel_pair_names = self._compute_metrics(batch)

        # dump results
        keys_to_save = {"fine_coord_0", "fine_coord_1", "conf_map", "epi_errs"}

        pair_names = list(zip(*batch["pair_names"]))
        bs = batch["image0"].shape[0]
        dumps = []

        for b_id in range(bs):
            item = {}
            mask = batch["b_idx_c"] == b_id
            item["pair_names"] = pair_names[b_id]
            item["identifier"] = "#".join(rel_pair_names[b_id])
            for key in keys_to_save:
                item[key] = batch[key][mask].cpu().numpy()
            for key in ["R_errs", "t_errs", "inliers"]:
                item[key] = batch[key][b_id]
            dumps.append(item)
        ret_dict["dumps"] = dumps

        self.test_outputs.append(ret_dict)

    def on_test_epoch_end(self):
        # metrics: dict of list, numpy
        _metrics = [o["metrics"] for o in self.test_outputs]
        metrics = {
            k: flattenList(gather(flattenList([_me[k] for _me in _metrics])))
            for k in _metrics[0]
        }

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList(
                [o["dumps"] for o in self.test_outputs]
            )  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(
                f"Prediction and evaluation results will be saved to: {self.dump_dir}"
            )

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR
            )
            logger.info("\n" + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / "MAFF_pred_eval", dumps)

        self.test_outputs.clear()
