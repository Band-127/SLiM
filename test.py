import sys
import importlib
import math
import torch
from yacs.config import CfgNode as CN
import pytorch_lightning as pl
from loguru import logger as loguru_logger

from utils.profiler import build_profiler
from utils.misc import setup_gpus, get_rank_zero_only_logger
from src.lightning_rcrm import PL_RCRM
from datasets.rcrm_dataset import RCRM_Dataset

# get logger and set as rank zero only(means ony info from rank 1 gpu will be stated out)
loguru_logger = get_rank_zero_only_logger(loguru_logger)


def main():
    # latest_ckpt_path = (
    #     "logs/tb_logs/MegaDepth_960_v1_8_2_ConvVMamba_C2F0_I4R3_/version_2/"
    # )
    # latest_ckpt = "checkpoints/epoch=13-auc@5=0.561-auc@10=0.720-auc@20=0.835.ckpt"
    latest_ckpt_path = (
        "logs/tb_logs/MegaDepth_1024_v1_8_2_ConvVMamba_C2F0_I4R3_/version_0/"
    )
    latest_ckpt = "checkpoints/epoch=0-auc@5=0.517-auc@10=0.677-auc@20=0.791.ckpt"
    devices = "6,7"
    ransac_thres = 0.5
    ransac_times = 1
    coarse_thres = 0.1
    intermediate_thres = 0.1
    coarse_max = 6000
    intermediate_max = 18000
    refine_iters = 4
    image_size = [1024, 1024]
    
    # ransac_thres = 0.5
    # ransac_times = 5
    # coarse_thres = 0.03
    # intermediate_thres = 0.03
    # coarse_max = 600
    # intermediate_max = 1800
    # refine_iters = 4
    # image_size = [480, 640]

    sys.path.append(latest_ckpt_path)
    get_cfg_defaults = importlib.import_module("config").get_cfg_defaults

    # torch.set_float32_matmul_precision("high")

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

    # lower the ransac pixel threshold for better performance
    config.TRAINER.RANSAC_PIXEL_THR = ransac_thres
    config.TRAINER.RANSAC_TIMES = ransac_times
    config.MODEL.COARSE_MATCHING.THRESHOLD = coarse_thres
    config.MODEL.INTERMEDIATE_MATCHING.THRESHOLD = intermediate_thres
    config.IMAGE_SIZE = config.DATASET.MGDPT_IMG_RESIZE = image_size[0]
    config.MODEL.BACKBONE.INPUT_SIZE = image_size
    config.MODEL.COARSE_MATCHING.MAX_MATCHES = coarse_max
    config.MODEL.INTERMEDIATE_MATCHING.MAX_MATCHES = intermediate_max
    config.MODEL.REFINE_ITERS = refine_iters

    # Profiler
    profiler = build_profiler("inference")

    # Lightning module
    model = PL_RCRM(
        config=config,
        pretrained_ckpt=latest_ckpt_path + "/" + latest_ckpt,
        profiler=profiler,
        dump_dir=config.DUMP_DIR,
    )
    loguru_logger.info("MAFF Lightning Module initialized!")

    # Lightning data
    data_module = RCRM_Dataset(config=config)
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
