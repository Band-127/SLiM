import math
import torch
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
from pytorch_lightning.strategies import DDPStrategy
from loguru import logger as loguru_logger

from utils.profiler import build_profiler
from default_config import get_cfg_defaults
from utils.misc import setup_gpus, get_rank_zero_only_logger
from maff.lightning_maff import PL_MAFF
from datasets.overall_dataset import MAFF_Dataset

# get logger and set as rank zero only(means ony info from rank 1 gpu will be stated out)
loguru_logger = get_rank_zero_only_logger(loguru_logger)


def main():
    config = get_cfg_defaults()
    pl.seed_everything(config.GLOBAL_SEED)

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
        monitor="auc@20",
        verbose=True,
        save_top_k=5,
        mode="max",
        save_last=True,
        dirpath=str(ckpt_dir),
        filename="{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    device_monitor = DeviceStatsMonitor()
    callbacks = [ckpt_callback, lr_monitor, device_monitor]

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

    if config.TRAINER.FIND_LR:
        # Finding best LR with linear progression
        temp = config.TRAINER.WARMUP_TYPE
        config.TRAINER.WARMUP_TYPE = "constant"     # set to constant to find lr
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            model=model,
            datamodule=data_module,
            num_training=300
        )
        print(
            f"Best LR found by LR finder with linear progression :{lr_finder.suggestion()}"
        )
        # Setting best LR
        config.TRAINER.TRUE_LR = lr_finder.suggestion()
        model.lr = lr_finder.suggestion()
        config.TRAINER.WARMUP_TYPE = temp

    # Training
    model.num_devices = trainer.num_devices
    loguru_logger.info("Start training!")
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
