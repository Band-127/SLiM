import sys
import importlib
import math
import torch
from yacs.config import CfgNode as CN
import pytorch_lightning as pl
from loguru import logger as loguru_logger

from utils.profiler import build_profiler
from utils.misc import setup_gpus, get_rank_zero_only_logger
from maff.lightning_maff import PL_MAFF
from datasets.overall_dataset import MAFF_Dataset

# get logger and set as rank zero only(means ony info from rank 1 gpu will be stated out)
loguru_logger = get_rank_zero_only_logger(loguru_logger)


def main():
    latest_ckpt_path = (
        "logs/tb_logs/MegaDepth_640_(0, 1, 1, 1, 1)_8_2_M2_VMamba_T_FPN_PD/version_0"
    )
    latest_ckpt = "checkpoints/last.ckpt"
    devices = "5,7"
    ransac_thres = 0.2

    sys.path.append(latest_ckpt_path)
    get_cfg_defaults = importlib.import_module("config").get_cfg_defaults

    torch.set_float32_matmul_precision("highest")

    # get configurations
    config: CN = get_cfg_defaults()
    pl.seed_everything(config.GLOBAL_SEED)

    # set train/test
    config.OVERALL_MODE = "test"

    # setup exact gpus available and set CUDA_VISIBLE_DEVICES variable
    config.DEVICE.GPU_IDX = devices
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

    # no limit for number of matching in test mode
    config.MODEL.COARSE_MATCHING.MAX_MATCHES = 10000
    # lower the ransac pixel threshold for better performance
    config.TRAINER.RANSAC_PIXEL_THR = ransac_thres

    # Profiler
    profiler = build_profiler("inference")

    # Lightning module
    model = PL_MAFF(
        config=config,
        pretrained_ckpt=latest_ckpt_path + "/" + latest_ckpt,
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
