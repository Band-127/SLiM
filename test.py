import argparse
import math
import pprint
import torch
from yacs.config import CfgNode as CN
import pytorch_lightning as pl
from loguru import logger as loguru_logger

from utils.profiler import build_profiler
from default_config import get_config
from utils.misc import setup_gpus, get_rank_zero_only_logger
from src.lightning_rcrm import PL_RCRM
from src.datasets.rcrm_dataset import RCRM_Dataset

# get logger and set as rank zero only(means ony info from rank 1 gpu will be stated out)
loguru_logger = get_rank_zero_only_logger(loguru_logger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="ckpt/megadepth_19epochs.ckpt",
        help="Path to checkpoint file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="2,3,4,5,6,7",
        help="Comma-separated list of GPU devices to use.",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=None,
        help="Threshold used in correlation filtering.",
    )
    parser.add_argument(
        "--refine_iters",
        type=int,
        default=4,
        help="Iterations number for refinement.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--optimized_ds",
        action="store_true",
        help="Partial dual softmax.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Auto mixed precision.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="outdoor_test",
        help="Config name. [outdoor_test, indoor_test]",
    )
    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    device = args.device
    thr = args.thr
    refine_iters = args.refine_iters
    seed = args.seed
    optimized_ds = args.optimized_ds
    amp = args.amp
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
    n_gpu_available = setup_gpus(device) if config.DEVICE.ENABLE_GPU else 0
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
    # Set threshold if provided
    if thr is not None:
        config.MODEL.COARSE_THRES = config.COARSE_THRES = thr
        config.MODEL.FINE_THRES = config.FINE_THRES = thr
    config.MODEL.REFINE_ITERS = config.REFINE_ITERS = refine_iters
    config.DUMP_DIR = {
        "outdoor_train": None,
        "outdoor_test": f"dump/rcrm_outdoor_I{config.REFINE_ITERS}",
        "indoor_train": None,
        "indoor_test": f"dump/rcrm_indoor_I{config.REFINE_ITERS}",
    }[config.MODE]
    config.MODEL.OPTIMIZED_DUAL_SOFTMAX = config.OPTIMIZED_DUAL_SOFTMAX = optimized_ds
    config.AMP = amp if amp is not None else config.AMP
    loguru_logger.info("Config: \n" + pprint.pformat(config))

    # Profiler
    profiler = build_profiler("inference")

    # Lightning module
    model = PL_RCRM(
        config=config,
        pretrained_ckpt=ckpt_path,
        profiler=profiler,
        dump_dir=config.DUMP_DIR,
    )
    loguru_logger.info("Lightning Module initialized!")

    # Lightning data
    data_module = RCRM_Dataset(config=config)
    loguru_logger.info("Data Module initialized!")

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
