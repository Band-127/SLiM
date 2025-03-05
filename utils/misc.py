import os
import time
import contextlib
import joblib
from typing import Union
from loguru import _Logger, logger
from itertools import chain

import torch
import torch.nn as nn
from yacs.config import CfgNode as CN
from pytorch_lightning.utilities import rank_zero_only


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def upper_config(dict_cfg):
    if not isinstance(dict_cfg, dict):
        return dict_cfg
    return {k.upper(): upper_config(v) for k, v in dict_cfg.items()}


def log_on(condition, message, level):
    if condition:
        assert level in ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]
        logger.log(level, message)


def get_rank_zero_only_logger(logger: _Logger):
    if rank_zero_only.rank == 0:
        return logger
    else:
        for _level in logger._core.levels.keys():
            level = _level.lower()
            setattr(logger, level, lambda x: None)
        logger._log = lambda x: None
    return logger


def setup_gpus(gpus: Union[str, int]) -> int:
    """A temporary fix for pytorch-lighting 1.3.x"""
    gpus = str(gpus)
    gpu_ids = []

    if "," not in gpus:
        n_gpus = int(gpus)
        return n_gpus if n_gpus != -1 else torch.cuda.device_count()
    else:
        gpu_ids = [i.strip() for i in gpus.split(",") if i != ""]

    # setup environment variables
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    return len(gpu_ids)


def flattenList(x):
    return list(chain(*x))


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument

    Usage:
        with tqdm_joblib(tqdm(desc="My calculation", total=10)) as progress_bar:
            Parallel(n_jobs=16)(delayed(sqrt)(i**2) for i in range(10))

    When iterating over a generator, directly use of tqdm is also a solutin (but monitor the task queuing, instead of finishing)
        ret_vals = Parallel(n_jobs=args.world_size)(
                    delayed(lambda x: _compute_cov_score(pid, *x))(param)
                        for param in tqdm(combinations(image_ids, 2),
                                          desc=f'Computing cov_score of [{pid}]',
                                          total=len(image_ids)*(len(image_ids)-1)/2))
    Src: https://stackoverflow.com/a/58936697
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def format_number(number):
    """Convert number to appropriate unit representation (K, M, G)"""
    units = ["", "K", "M", "G"]
    unit_index = 0

    while number >= 1000 and unit_index < len(units) - 1:
        number /= 1000
        unit_index += 1

    if number < 10:
        return f"{number:.2f}{units[unit_index]}"
    return f"{number:.2f}{units[unit_index]}"


def count_parameters(model, indent="", min_params=1, recursive=False):
    """Count parameters for all nn.Module objects in the model"""
    total_params = 0

    # Get all attributes
    for name, module in model.__dict__["_modules"].items():
        if not isinstance(module, nn.Module) or name.startswith("_"):
            continue

        # Calculate parameters for current module
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if params >= min_params:
            total_params += params
            formatted_params = format_number(params)
            print(f"{indent}{name}: {formatted_params:>8} parameters")

            # Recursively print submodules if enabled
            if recursive and hasattr(module, "__dict__"):
                _ = count_parameters(module, indent + "    ", min_params, recursive)

    return total_params


def print_params_summary(model, min_params=1, recursive=False):
    print("\n" + "=" * 90)
    print("Model Structure and Parameters:")
    print("-" * 90)
    total = count_parameters(model, min_params=min_params, recursive=recursive)
    print("-" * 90)
    print(f"Total Parameters: {format_number(total)}")
    print("=" * 90 + "\n", flush=True)


class Timer:
    def __init__(self, process_name=""):
        self.process_name = process_name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        self._print_time()

    def _print_time(self):
        process_str = f"[{self.process_name}] " if self.process_name else ""
        if self.interval < 1:
            print(f"{process_str}Elapsed time: {self.interval * 1000:.2f} ms")
        elif self.interval < 60:
            print(f"{process_str}Elapsed time: {self.interval:.2f} s")
        else:
            minutes, seconds = divmod(self.interval, 60)
            print(f"{process_str}Elapsed time: {int(minutes)} min {seconds:.2f} s")
