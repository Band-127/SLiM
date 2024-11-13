import pprint
import numpy as np
from yacs.config import CfgNode as CN
from collections import defaultdict
from pathlib import Path
from loguru import logger
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import (
    MultiStepLR,
    CosineAnnealingLR,
    ExponentialLR,
    CosineAnnealingWarmRestarts,
)
import pytorch_lightning as pl
from pytorch_lightning.profilers.profiler import Profiler
from pytorch_lightning.profilers import PassThroughProfiler
import torch.distributed as dist
from contextlib import contextmanager
import time

from src.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from src.rcrm_v1 import RCRM_v1
from src.loss import RCRM_Loss
from utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics,
)
from utils.comm import all_gather, gather
from utils.misc import flattenList, print_params_summary
from utils.plotting import make_matching_figures


test_timer_list = [
    "feat_extract_time",
    "coarse_time",
    "correlation_time",
    "fine_scan_time",
    "fine_time",
]


@contextmanager
def sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_mean_time_across_ranks(local_time):
    if dist.is_initialized():
        world_size = dist.get_world_size()
        all_times = [torch.zeros_like(local_time) for _ in range(world_size)]
        dist.all_gather(all_times, local_time)
        return torch.mean(torch.stack(all_times))
    return local_time


class PL_RCRM(pl.LightningModule):
    def __init__(
        self,
        config: CN,
        pretrained_ckpt: str = None,
        profiler: Profiler = None,
        dump_dir: str = None,
    ):
        """_summary_

        Args:
            config (CN): root configuration
            pretrained_ckpt (str, optional): Path to pretrained checkpoint if exists. Defaults to None.
            profiler (Profiler, optional): Pytorch Lightning Profiler module. Defaults to None.
            dump_dir (str, optional): Dir to dump testing output. Defaults to None.
        """
        super(PL_RCRM, self).__init__()

        self.config = config
        self.pretrained_ckpt = pretrained_ckpt
        self.profiler = profiler or PassThroughProfiler()
        self.dump_dir = dump_dir
        self.num_devices = None
        self.first_stage_epochs = config.TRAINER.FIRST_STAGE_EPOCHS
        self.reparameter = False
        self.show_gt_matched_fine_on_val = config.MODEL.SHOW_GT_MATCHED_FINE

        if config["MODEL"]["VERSION"] == "v1":
            _RCRM = RCRM_v1
        self.rcrm = _RCRM(config=config["MODEL"])
        self.loss = RCRM_Loss(config=config["LOSS"])

        # Read pretrained checkpoint if exists
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location="cpu")["state_dict"]
            self.rcrm.load_state_dict(state_dict, strict=True)
            logger.info(f"Load '{pretrained_ckpt}' as pretrained checkpoint")

        # Coarse scale
        self.coarse_scale = self.config.MODEL.COARSE_SCALE
        self.lr = self.config.TRAINER.TRUE_LR

        self.training_outputs = []
        self.validation_outputs = []
        self.test_outputs = []
        self.train_forward_times = []
        self.val_forward_times = []
        self.val_parts_times = {}
        self.test_forward_times = []
        self.test_parts_times = {}

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
        scheduler = {
            "interval": self.config.TRAINER.SCHEDULER_INTERVAL,
        }
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
        elif self.config.TRAINER.SCHEDULER == "CosineAnnealingWarmRestarts":
            scheduler.update(
                {
                    "scheduler": CosineAnnealingWarmRestarts(
                        optimizer=optimizer,
                        T_0=self.config.TRAINER.COSAWR_T0,
                        T_mult=self.config.TRAINER.COSAWR_TMULT,
                        eta_min=self.config.TRAINER.COSAWR_ETAMIN,
                    ),
                    "interval": "step",
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
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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
        if training:
            with self.profiler.profile("Compute coarse supervision"):
                compute_supervision_coarse(
                    batch, coarse_scale=self.coarse_scale, config=self.config
                )

        with self.profiler.profile("MAFF"):
            start_time = time.perf_counter()
            self.rcrm(batch, training=training)
            end_time = time.perf_counter()

        if training:
            with self.profiler.profile("Compute fine supervision"):
                compute_supervision_fine(batch, config=self.config)

            with self.profiler.profile("Compute losses"):
                self.loss(batch)

        # Timer
        train_time = torch.tensor(end_time - start_time, device=self.device)
        mean_train_time = get_mean_time_across_ranks(train_time)
        batch.update({"forward_time": mean_train_time.item()})
        if training:
            self.train_forward_times.append(mean_train_time.item())
        else:
            self.val_forward_times.append(mean_train_time.item())
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
                    batch["epi_errs"][batch["b_idx_it"] == b].cpu().numpy()
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
        # logging
        if (
            self.trainer.global_rank == 0
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            # scalars
            for k, v in batch["loss_scalars"].items():
                self.logger.experiment.add_scalar(
                    f"train/{k}", v, self.global_step * self.num_devices
                )

            # num matches
            self.logger.experiment.add_scalar(
                "train/num_matches",
                batch["num_matches"],
                self.global_step * self.num_devices,
            )

            # time
            self.logger.experiment.add_scalar(
                "train/forward_time",
                batch["forward_time"],
                self.global_step * self.num_devices,
            )
            keys = test_timer_list
            for k in keys:
                self.logger.experiment.add_scalar(
                    f"train/{k}",
                    batch[k],
                    self.global_step * self.num_devices,
                )

            self.training_outputs.append(batch["loss"])

        return {"loss": batch["loss"]}

    def on_train_epoch_start(self):
        if self.trainer.current_epoch == 0:
            # Print model summary
            if self.trainer.global_rank == 0:
                print(self.rcrm)
                print_params_summary(self.rcrm, recursive=False)
        if self.trainer.current_epoch == self.first_stage_epochs:
            self.loss.turn_off_conf_mask_loss = True
            # self.loss.fine_weight = 1.0
            # self.loss.intermediate_weight = 0.1
            # self.loss.coarse_weight = 1.0

    def on_train_epoch_end(self):
        self.training_outputs.clear()
        torch.cuda.empty_cache()

        # time
        if self.trainer.global_rank == 0 and len(self.train_forward_times) > 0:
            avg_forward_time = sum(self.train_forward_times) / len(
                self.train_forward_times
            )
            self.logger.experiment.add_scalar(
                "train/avg_forward_time", avg_forward_time, self.trainer.current_epoch
            )
        self.train_forward_times.clear()

    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch, training=self.show_gt_matched_fine_on_val)

        # All timers
        keys = test_timer_list
        if len(self.val_parts_times.keys()) == 0:
            self.val_parts_times = {k: [] for k in keys}
        for key in keys:
            if key in batch.keys():
                self.val_parts_times[key].append(batch[key])

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
                metric_name = f"auc@{thr}"
                multi_val_metrics[f"auc@{thr}"].append(val_metrics_4tb[f"auc@{thr}"])
                self.log(
                    metric_name,
                    val_metrics_4tb[metric_name],
                    prog_bar=True,
                    sync_dist=True,
                )

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
                # time
                if len(self.val_forward_times) > 0:
                    avg_forward_time = self.calculate_trimmed_mean(
                        self.val_forward_times
                    )
                    self.logger.experiment.add_scalar(
                        "test/avg_forward_time",
                        avg_forward_time,
                        self.trainer.current_epoch,
                    )
                    self.val_forward_times.clear()
                keys = test_timer_list
                if len(self.val_parts_times.keys()) and len(
                    self.val_parts_times[keys[0]]
                ):
                    for k in keys:
                        avg_time_for_this_part = self.calculate_trimmed_mean(
                            self.val_parts_times[k]
                        )
                        self.logger.experiment.add_scalar(
                            f"test/{k}",
                            avg_time_for_this_part,
                            self.trainer.current_epoch,
                        )
                        self.val_parts_times[k].clear()

        self.validation_outputs.clear()
        torch.cuda.empty_cache()

    def on_test_epoch_start(self):
        if not self.reparameter:
            self.rcrm.reparameter()
            self.reparameter = True

    def test_step(self, batch: dict, batch_idx):
        # inference
        start_time = time.perf_counter()
        with torch.inference_mode(True):
            self.rcrm(batch)
        end_time = time.perf_counter()

        # All timers
        keys = test_timer_list
        if len(self.test_parts_times.keys()) == 0:
            self.test_parts_times = {k: [] for k in keys}
        for key in keys:
            if key in batch.keys():
                self.test_parts_times[key].append(batch[key])

        # Timer
        local_time = torch.tensor(end_time - start_time, device=self.device)
        mean_time = get_mean_time_across_ranks(local_time)
        batch.update({"forward_time": mean_time.item()})
        self.test_forward_times.append(mean_time.item())
        torch.cuda.empty_cache()

        # metrics
        ret_dict, rel_pair_names = self._compute_metrics(batch)

        # dump results
        keys_to_save = {"fine_coord_0", "fine_coord_1", "epi_errs"}

        pair_names = list(zip(*batch["pair_names"]))
        bs = batch["image0"].shape[0]
        dumps = []

        for b_id in range(bs):
            item = {}
            mask = batch["b_idx_it"] == b_id
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
        with torch.inference_mode(False):
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
            # time
            if len(self.test_forward_times) > 0:
                avg_forward_time = self.calculate_trimmed_mean(self.test_forward_times)
                print(f"Avg forward time: {avg_forward_time*1000: 8.4}ms")
                self.test_forward_times.clear()
            keys = test_timer_list
            if len(self.test_parts_times.keys()) and len(
                self.test_parts_times[keys[0]]
            ):
                for k in keys:
                    avg_time_for_this_part = self.calculate_trimmed_mean(
                        self.test_parts_times[k]
                    )
                    print(
                        f"Avg forward time for {k}: {avg_time_for_this_part*1000: 8.4}ms"
                    )
                    self.test_parts_times[k].clear()
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR
            )
            logger.info("\n" + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / "MAFF_pred_eval", dumps)

        self.test_outputs.clear()

    @staticmethod
    def calculate_trimmed_mean(_list):
        n = len(_list)
        trim_count = int(0.1 * n)
        sorted_list = sorted(_list)
        trimmed_list = sorted_list[trim_count:-trim_count]
        avg = sum(trimmed_list) / len(trimmed_list)
        return avg
