import pprint
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from loguru import logger as loguru_logger

from utils.profiler import build_profiler
from default_config import get_cfg_defaults
from utils.misc import setup_gpus, get_rank_zero_only_logger
from src.lightning_slim import PL_SLiM
from src.datasets.slim_dataset import SLiM_Dataset

# get logger and set as rank zero only(means ony info from rank 1 gpu will be stated out)
loguru_logger = get_rank_zero_only_logger(loguru_logger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="2,3,4,5,6,7",
        help="Comma-separated list of GPU devices to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="outdoor_train",
        help="Config name. [outdoor_train, indoor_train]",
    )
    args = parser.parse_args()
    
    device = args.device
    seed = args.seed
    config_name = args.config_name
    
    # get configurations
    config: CN = get_config(config_name)
    
    # set seed
    if seed is None:
        import os
        import random

        seed = os.environ.get("GLOBAL_SEED")
        if seed:
            seed = int(seed)
        else:
            # set a random number with current time as random seed
            random.seed(a=None)
            seed = random.randint(0, 4294967295)
            os.environ["GLOBAL_SEED"] = str(seed)

    config.GLOBAL_SEED = seed
    pl.seed_everything(config.GLOBAL_SEED)
    torch.cuda.manual_seed_all(config.GLOBAL_SEED)

    # setup exact gpus available and set CUDA_VISIBLE_DEVICES variable
    n_gpu_available = (
        setup_gpus(config.DEVICE.GPU_IDX) if config.DEVICE.ENABLE_GPU else 0
    )
    config.TRAINER.WORLD_SIZE = n_gpu_available * config.DEVICE.NUM_NODES
    config.TRAINER.TRUE_BATCH_SIZE = (
        config.TRAINER.WORLD_SIZE
        * config.BATCH_SIZE
        * config.TRAINER.ACCUMULATE_GRAD_BATCHES
    )
    config.TRAINER.SCALING = (
        config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    )
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * config.TRAINER.SCALING
    loguru_logger.info("Config: \n" + pprint.pformat(config))

    # Profiler
    profiler = build_profiler(config.PROFILER.PROFILER_NAME)

    # Lightning module
    model = PL_SLiM(
        config=config, pretrained_ckpt=config.PRETRAINED_PATH, profiler=profiler
    )
    loguru_logger.info("Lightning Module initialized!")

    # Lightning data
    data_module = SLiM_Dataset(config=config)
    loguru_logger.info("Data Module initialized!")

    # TensorBoard Logger
    logger = TensorBoardLogger(
        save_dir="logs/tb_logs", name=config.LOGGER.LOGGER_NAME, default_hp_metric=False
    )
    ckpt_dir = Path(logger.log_dir) / "checkpoints"

    # Callbacks
    ckpt_callback = ModelCheckpoint(
        monitor="auc@5",
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
        accumulate_grad_batches=config.TRAINER.ACCUMULATE_GRAD_BATCHES,
        log_every_n_steps=int(50 / config.TRAINER.ACCUMULATE_GRAD_BATCHES),
        max_epochs=27
    )
    loguru_logger.info("Trainer Initialized!")

    if config.TRAINER.FIND_LR:
        # Finding best LR with linear progression
        temp = config.TRAINER.WARMUP_TYPE
        model.num_devices = n_gpu_available
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model=model, datamodule=data_module, num_training=300)
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
