import math
import torch
from yacs.config import CfgNode as CN
import pytorch_lightning as pl
from loguru import logger as loguru_logger

from utils.profiler import build_profiler
from default_config import get_cfg_defaults
from utils.misc import setup_gpus, get_rank_zero_only_logger
from maff.lightning_maff import PL_MAFF
from datasets.overall_dataset import MAFF_Dataset

# get logger and set as rank zero only(means ony info from rank 1 gpu will be stated out)
loguru_logger = get_rank_zero_only_logger(loguru_logger)


def main():
    torch.set_float32_matmul_precision("high")
    # get configurations
    config: CN = get_cfg_defaults()
    pl.seed_everything(config.GLOBAL_SEED)

    # set train/test
    config.OVERALL_MODE = "test"

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
        config=config,
        pretrained_ckpt=config.PRETRAINED_PATH,
        profiler=profiler,
        dump_dir=config.DUMP_DIR,
    )
    loguru_logger.info("MAFF Lightning Module initialized!")

    # Lightning data
    data_module = MAFF_Dataset(config=config)
    loguru_logger.info("MAFF Data Module initialized!")

    # Torch Lightning Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if config.DEVICE.ENABLE_GPU else "cpu",
        devices=n_gpu_available,
        num_nodes=config.DEVICE.NUM_NODES,
        profiler=profiler,
        logger=False,
    )
    loguru_logger.info("Trainer Initialized!")

    # Testing
    loguru_logger.info("Start testing!")
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
