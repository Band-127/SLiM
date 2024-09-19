import math
import torch
from pathlib import Path
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from loguru import logger as loguru_logger

from utils.profiler import build_profiler
from default_config import get_cfg_defaults
from utils.misc import setup_gpus, get_rank_zero_only_logger
from maff.lightning_maff import PL_MAFF
from datasets.overall_dataset import MAFF_Dataset

# get logger and set as rank zero only(means ony info from rank 1 gpu will be stated out)
loguru_logger = get_rank_zero_only_logger(loguru_logger)

config = None


def init_config():
    global config
    if config is None:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                config = get_cfg_defaults()
                pl.seed_everything(config.GLOBAL_SEED)

            config_list = [config]
            dist.broadcast_object_list(config_list, src=0)
            config = config_list[0]

            dist.barrier()
        else:
            config = get_cfg_defaults()
            pl.seed_everything(config.GLOBAL_SEED)
    return config


def main():
    global config
    config = init_config()

    torch.set_float32_matmul_precision("high")

    # set train/test
    config.OVERALL_MODE = "train"

    # setup exact gpus available and set CUDA_VISIBLE_DEVICES variable
    n_gpu_available = (
        setup_gpus(config.DEVICE.GPU_IDX) if config.DEVICE.ENABLE_GPU else 0
    )
    config.TRAINER.WORLD_SIZE = n_gpu_available * config.DEVICE.NUM_NODES
    config.TRAINER.TRUE_BATCH_SIZE = (
        config.TRAINER.WORLD_SIZE * config.LOADER.BATCH_SIZE
    )
    config.TRAINER.SCALING = (
        config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    )
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * config.TRAINER.SCALING
    config.TRAINER.WARMUP_STEP = math.floor(
        config.TRAINER.WARMUP_STEP / config.TRAINER.SCALING
    )

    # Profiler
    profiler = build_profiler(config.PROFILER.PROFILER_NAME)

    # Lightning module
    model = PL_MAFF(
        config=config, pretrained_ckpt=config.PRETRAINED_PATH, profiler=profiler
    )
    loguru_logger.info("MAFF Lightning Module initialized!")

    # Lightning data
    data_module = MAFF_Dataset(config=config)
    loguru_logger.info("MAFF Data Module initialized!")

    # TensorBoard Logger
    logger = TensorBoardLogger(
        save_dir="logs/tb_logs", name=config.LOGGER.LOGGER_NAME, default_hp_metric=False
    )
    ckpt_dir = Path(logger.log_dir) / "checkpoints"

    # Callbacks
    ckpt_callback = ModelCheckpoint(
        monitor="auc@10",
        verbose=True,
        save_top_k=5,
        mode="max",
        save_last=True,
        dirpath=str(ckpt_dir),
        filename="{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [ckpt_callback, lr_monitor]

    # Torch Lightning Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if config.DEVICE.ENABLE_GPU else "cpu",
        strategy=DDPStrategy(find_unused_parameters=True)
        if config.DEVICE.ENABLE_DDP
        else "auto",
        devices=n_gpu_available,
        num_nodes=config.DEVICE.NUM_NODES,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        sync_batchnorm=(config.TRAINER.WORLD_SIZE > 0),
        profiler=profiler,
    )
    loguru_logger.info("Trainer Initialized!")

    # # Finding best LR with linear progression
    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(
    #     model=model,
    #     datamodule=data_module,
    # )
    # print(
    #     f"Best LR found by LR finder with linear progression :{lr_finder.suggestion()}"
    # )

    # # Setting best LR
    # model.lr = lr_finder.suggestion()

    # Training
    loguru_logger.info("Start training!")
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
