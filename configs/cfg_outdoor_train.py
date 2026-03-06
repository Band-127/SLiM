import os
import random
from datetime import datetime
from yacs.config import CfgNode as CN

_CN = CN()

########    General Configurations    ########
_CN.GLOBAL_SEED = 66
_CN.DTYPE = "float32"
_CN.RGB_INPUT = False               # True if input channel == 3, False if input channel == 1
_CN.MODE = "outdoor_train"          # options: ["outdoor_train", "outdoor_test", "indoor_train", "indoor_test"]
_CN.PRETRAINED_PATH = None
_CN.REFINE_ITERS = 4
_CN.BATCH_SIZE = 1
_CN.COARSE_SCALE_IDX = 2            # 1/8
_CN.FINE_SCALE_IDX = 0              # 1/2
_CN.MAX_COARSE_MATCHES = {
    "outdoor_train": 6000,
    "outdoor_test": 6000,
    "indoor_train": 600,
    "indoor_test": 600,
}[_CN.MODE]
_CN.MAX_FINE_MATCHES = {
    "outdoor_train": 6000,
    "outdoor_test": 6000,
    "indoor_train": 600,
    "indoor_test": 4800,
}[_CN.MODE]
_CN.IMAGE_SIZE = {
    "outdoor_train": 1024,
    "outdoor_test": 1184,
    "indoor_train": 640,
    "indoor_test": 640,
}[_CN.MODE]
_CN.COARSE_THRES = {
    "outdoor_train": 0.1,
    "outdoor_test": 0.1,
    "indoor_train": 0.1,
    "indoor_test": 0.0005,
}[_CN.MODE]
_CN.FINE_THRES = {
    "outdoor_train": 0.1,
    "outdoor_test": 0.03,
    "indoor_train": 0.1,
    "indoor_test": 0.0005,
}[_CN.MODE]
_CN.DUMP_DIR = {
    "outdoor_train": None,
    "outdoor_test": f"dump/rcrm_outdoor_I{_CN.REFINE_ITERS}",
    "indoor_train": None,
    "indoor_test": f"dump/rcrm_indoor_I{_CN.REFINE_ITERS}",
}[_CN.MODE]
_CN.RANSAC_TIMES = 5
_CN.OPTIMIZED_DUAL_SOFTMAX = False
_CN.AMP = True

########    Device Configurations    ########
# Support CUDA only!!!
_CN.DEVICE = CN()
_CN.DEVICE.ENABLE_GPU = True                    # Whether enable GPUs, default true
_CN.DEVICE.ENABLE_DDP = True                    # Whether enable distributed data parallel, default true
_CN.DEVICE.GPU_IDX = "4,5,6,7"              # GPUs indices, e.g. "0,1,2,3,4,5,6,7"
_CN.DEVICE.NUM_NODES = 1
_CN.DEVICE.MASTER_ADDR = "localhost"
_CN.DEVICE.MASTER_PORT = "29500"

########    Dataset Configurations    ########
_CN.DATASET = CN()
_CN.DATASET.DATA_SOURCE = "MegaDepth" if "outdoor" in _CN.MODE else "ScanNet"
if _CN.DATASET.DATA_SOURCE == "MegaDepth":
    """
    MegaDepth
    """
    # training
    _CN.DATASET.TRAINVAL_DATA_SOURCE = "MegaDepth"
    _CN.DATASET.TRAIN_DATA_BASE_PATH = "data/megadepth"
    _CN.DATASET.TRAIN_DATA_ROOT = f"{_CN.DATASET.TRAIN_DATA_BASE_PATH}/train"
    _CN.DATASET.TRAIN_POSE_ROOT = None                                                                      # (optional directory for poses)
    _CN.DATASET.TRAIN_NPZ_ROOT = f"{_CN.DATASET.TRAIN_DATA_BASE_PATH}/index/scene_info_0.1_0.7"             # None if val data from all scenes are bundled into a single npz file
    _CN.DATASET.TRAIN_LIST_PATH = f"{_CN.DATASET.TRAIN_DATA_BASE_PATH}/index/trainvaltest_list/train_list.txt"
    # _CN.DATASET.TRAIN_LIST_PATH = f"{_CN.DATASET.TRAIN_DATA_BASE_PATH}/index/trainvaltest_list/debug_list.txt"
    _CN.DATASET.TRAIN_INTRINSIC_PATH = None
    # validating
    _CN.DATASET.VAL_DATA_BASE_PATH = "data/megadepth"
    _CN.DATASET.VAL_DATA_ROOT = f"{_CN.DATASET.VAL_DATA_BASE_PATH}/test"
    _CN.DATASET.VAL_POSE_ROOT = None                                                                        # (optional directory for poses)
    _CN.DATASET.VAL_NPZ_ROOT = f"{_CN.DATASET.VAL_DATA_BASE_PATH}/index/scene_info_val_1500"
    _CN.DATASET.VAL_LIST_PATH = f"{_CN.DATASET.VAL_DATA_BASE_PATH}/index/trainvaltest_list/val_list.txt"    # None if val data from all scenes are bundled into a single npz file
    # _CN.DATASET.VAL_LIST_PATH = f"{_CN.DATASET.VAL_DATA_BASE_PATH}/index/trainvaltest_list/debug_list.txt"    # None if val data from all scenes are bundled into a single npz file
    _CN.DATASET.VAL_INTRINSIC_PATH = None
    # testing
    _CN.DATASET.TEST_DATA_SOURCE = "MegaDepth"
    _CN.DATASET.TEST_DATA_BASE_PATH = "data/megadepth"
    _CN.DATASET.TEST_DATA_ROOT = f"{_CN.DATASET.TEST_DATA_BASE_PATH}/test"
    _CN.DATASET.TEST_POSE_ROOT = None                                                                       # (optional directory for poses)
    _CN.DATASET.TEST_NPZ_ROOT = f"{_CN.DATASET.TEST_DATA_BASE_PATH}/index/scene_info_val_1500"
    _CN.DATASET.TEST_LIST_PATH = f"{_CN.DATASET.TEST_DATA_BASE_PATH}/index/trainvaltest_list/val_list.txt"  # None if test data from all scenes are bundled into a single npz file
    _CN.DATASET.TEST_INTRINSIC_PATH = None
elif _CN.DATASET.DATA_SOURCE == "ScanNet":
    """
    ScanNet
    """
    # training
    _CN.DATASET.TRAINVAL_DATA_SOURCE = "ScanNet"
    _CN.DATASET.TRAIN_DATA_BASE_PATH = "data/scannet"
    _CN.DATASET.TRAIN_DATA_ROOT = f"{_CN.DATASET.TRAIN_DATA_BASE_PATH}/train"
    _CN.DATASET.TRAIN_POSE_ROOT = None                                                                          # (optional directory for poses)
    _CN.DATASET.TRAIN_NPZ_ROOT = f"{_CN.DATASET.TRAIN_DATA_BASE_PATH}/index/scene_data/train"
    _CN.DATASET.TRAIN_LIST_PATH = f"{_CN.DATASET.TRAIN_DATA_BASE_PATH}/index/scene_data/train_list/scannet_all.txt"
    # _CN.DATASET.TRAIN_LIST_PATH = f"{_CN.DATASET.TRAIN_DATA_BASE_PATH}/index/scene_data/train_list/debug_21.txt"
    _CN.DATASET.TRAIN_INTRINSIC_PATH = f"{_CN.DATASET.TRAIN_DATA_BASE_PATH}/index/intrinsics.npz"
    # validating
    _CN.DATASET.VAL_DATA_BASE_PATH = "data/scannet"
    _CN.DATASET.VAL_DATA_ROOT = f"{_CN.DATASET.VAL_DATA_BASE_PATH}/test"
    _CN.DATASET.VAL_POSE_ROOT = None                                                                            # (optional directory for poses)
    _CN.DATASET.VAL_NPZ_ROOT = f"{_CN.DATASET.VAL_DATA_BASE_PATH}/index/scene_data/val_list"
    _CN.DATASET.VAL_LIST_PATH = f"{_CN.DATASET.VAL_DATA_BASE_PATH}/index/scene_data/val_list/scannet_test.txt"  # None if val data from all scenes are bundled into a single npz file
    _CN.DATASET.VAL_INTRINSIC_PATH = f"{_CN.DATASET.VAL_DATA_BASE_PATH}/index/scene_data/val_list/intrinsics.npz"
    # testing
    _CN.DATASET.TEST_DATA_SOURCE = "ScanNet"
    _CN.DATASET.TEST_DATA_BASE_PATH = "data/scannet"
    _CN.DATASET.TEST_DATA_ROOT = f"{_CN.DATASET.TEST_DATA_BASE_PATH}/test"
    _CN.DATASET.TEST_POSE_ROOT = None                                                                           # (optional directory for poses)
    _CN.DATASET.TEST_NPZ_ROOT = f"{_CN.DATASET.TEST_DATA_BASE_PATH}//index/scene_data/val_list"
    _CN.DATASET.TEST_LIST_PATH = f"{_CN.DATASET.TEST_DATA_BASE_PATH}/index/scene_data/val_list/scannet_test.txt"# None if test data from all scenes are bundled into a single npz file
    _CN.DATASET.TEST_INTRINSIC_PATH = f"{_CN.DATASET.TEST_DATA_BASE_PATH}/index/scene_data/val_list/intrinsics.npz"
# general options
_CN.DATASET.PARALLEL_WORKERS_NUM = 16
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0           # discard data with overlap_score < min_overlap_score
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = None                # options: [None, "dark", "mobile", "rcrm", "rcrm_lite"]
# MegaDepth options
_CN.DATASET.MGDPT_IMG_RESIZE = _CN.IMAGE_SIZE       # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.MGDPT_IMG_PAD = True                    # pad img to square with size = MGDPT_IMG_RESIZE
_CN.DATASET.MGDPT_DEPTH_PAD = True                  # pad depthmap to square with size = 2000
_CN.DATASET.MGDPT_DF = 1                            # image size division factor
_CN.DATASET.MGDPT_COARSE_SCALE = None

########    Dataset Sampler Configurations    ########
_CN.SAMPLER = CN()
_CN.SAMPLER.N_SAMPLES_PER_SUBSET = 100
_CN.SAMPLER.SUBSET_REPLACEMENT = True               # whether sample each scene with replacement or not
_CN.SAMPLER.SHUFFLE = True                          # whether shuffle samples within epoch
_CN.SAMPLER.REPEAT = 1                              # how many times to be repeated for training

########    Dataset Loader Configurations    ########
_CN.LOADER = CN()
_CN.LOADER.BATCH_SIZE = _CN.BATCH_SIZE              # how many samples per batch to load, per gpu
_CN.LOADER.NUM_WORKERS = 4                          # how many subprocesses to use for data loading.
_CN.LOADER.PIN_MEMORY = True                        # If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them.

########    Model Configurations    ########
_CN.MODEL = CN()
_CN.MODEL.REFINE_ITERS = _CN.REFINE_ITERS
_CN.MODEL.REFINE_LOOKUP_RADIUS = 3
_CN.MODEL.COARSE_SCALE = None
_CN.MODEL.COARSE_SCALE_IDX = _CN.COARSE_SCALE_IDX
_CN.MODEL.FINE_SCALE_IDX = _CN.FINE_SCALE_IDX
_CN.MODEL.MAX_COARSE_MATCHES = _CN.MAX_COARSE_MATCHES
_CN.MODEL.MAX_FINE_MATCHES = _CN.MAX_FINE_MATCHES
_CN.MODEL.COARSE_THRES = _CN.COARSE_THRES
_CN.MODEL.FINE_THRES = _CN.FINE_THRES
_CN.MODEL.TRAIN_NOISE_SCALE = 0.8
_CN.MODEL.OPTIMIZED_DUAL_SOFTMAX = _CN.OPTIMIZED_DUAL_SOFTMAX
# Backbone
_CN.MODEL.BACKBONE = CN()
_CN.MODEL.BACKBONE.RGB_INPUT = _CN.RGB_INPUT
_CN.MODEL.BACKBONE.PATCH_SIZE = 2
_CN.MODEL.BACKBONE.DIMS = [48, 96, 192]
_CN.MODEL.BACKBONE.DEPTHS = [1, 1, 1]
_CN.MODEL.BACKBONE.EXTRA_DEPTH = 2
_CN.MODEL.BACKBONE.EXTRA_AGGREGATION = 4
_CN.MODEL.BACKBONE.CONV_KERNEL_SIZE = 7
_CN.MODEL.FINE_DIM = _CN.MODEL.BACKBONE.DIMS[_CN.FINE_SCALE_IDX]
_CN.MODEL.COARSE_DIM = _CN.MODEL.BACKBONE.DIMS[_CN.COARSE_SCALE_IDX]
# Refinement
_CN.MODEL.REFINEMENT = CN()
_CN.MODEL.REFINEMENT.HIDDEN_DIM = 32
_CN.MODEL.REFINEMENT.CONTEXT_INJECTION = True

########    Loss Configurations    ########
_CN.LOSS = CN()
# COARSE MATCHING
_CN.LOSS.COARSE_WEIGHT = 0.25
_CN.LOSS.FOCAL_GAMMA_COARSE = 2.0
_CN.LOSS.COARSE_PERCENT = 0.9
# FINE MATCHING
_CN.LOSS.FINE_WEIGHT = 0.2
_CN.LOSS.FOCAL_GAMMA_FINE = 2.0
# REFINEMENT
_CN.LOSS.REFINE_WEIGHT = 1.0
_CN.LOSS.REFINE_THRES = 1.0
_CN.LOSS.ITER_DECAY_GAMMA = 0.8

########    Trainer Configurations    ########
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = None                       # Will be calculated using number of nodes and exact number of devices available
_CN.TRAINER.GRADIENT_CLIPPING = 0.8                 # Gradient clipping
_CN.TRAINER.CANONICAL_BS = 8
_CN.TRAINER.CANONICAL_LR = 1e-3                     # using LR finder provided by pytorch lightning if FIND_LR set to True
_CN.TRAINER.SCALING = None                          # this will be calculated automatically
_CN.TRAINER.FIND_LR = False                         # use learning rate finder from pytorch-lightning, TODO: fix lr finder
# gradient accumulation
_CN.TRAINER.ACCUMULATE_GRAD_BATCHES = 1
# optimizer
_CN.TRAINER.OPTIMIZER = "AdamW"                         # options: [Adam, AdamW, SGD]
_CN.TRAINER.TRUE_LR = None
_CN.TRAINER.ADAM_DECAY = 0.1
_CN.TRAINER.ADAMW_DECAY = 0.01
_CN.TRAINER.SGD_MOMENTUM = 0.9
# learning rate scheduler
_CN.TRAINER.SCHEDULER = "MultiStepLR"                   # options: [MultiStepLR, CosineAnnealing, ExponentialLR, CosineAnnealingWarmRestarts]
_CN.TRAINER.SCHEDULER_INTERVAL = "epoch"                # [epoch, step], automatically switch to step mode when using CosineAnnealing & CosineAnnealingWarmRestarts
_CN.TRAINER.MSLR_MILESTONES = [6, 10, 14, 16]           # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 15                              # COSA: CosineAnnealing, Tmax, in epoch, will be converted to step automatically
_CN.TRAINER.COSA_ETA_MIN = 1e-9                         # COSA: CosineAnnealing, eta_min
_CN.TRAINER.ELR_GAMMA = 0.999992                        # ELR: ExponentialLR, this value for "step" interval
_CN.TRAINER.COSAWR_T0 = 1                               # COSAWR: CosineAnnealingWarmRestarts, T0, in epoch, will be converted to step automatically
_CN.TRAINER.COSAWR_TMULT = 2                            # COSAWR: CosineAnnealingWarmRestarts, Tmult
_CN.TRAINER.COSAWR_ETAMIN = 1e-12
# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'                      # options: [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.01
_CN.TRAINER.WARMUP_STEP = 18400                         # in steps
# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 10                    # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'                    # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'
# For metric calculation
_CN.TRAINER.RANSAC_PIXEL_THR = 0.275
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_TIMES = _CN.RANSAC_TIMES
_CN.TRAINER.EPI_ERR_THR = 5e-4 if _CN.DATASET.TRAINVAL_DATA_SOURCE == "ScanNet" else 1e-4   # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)

# Calculate coarse scale
_CN.DATASET.MGDPT_COARSE_SCALE = _CN.MODEL.COARSE_SCALE = _CN.MODEL.BACKBONE.PATCH_SIZE * 2 ** (_CN.COARSE_SCALE_IDX)

########    Logger Configurations    ########
_CN.LOGGER = CN()
_CN.LOGGER.EXP_NAME = f"{datetime.isoformat(datetime.now(), sep='-', timespec='seconds')}"
_CN.LOGGER.LOGGER_NAME = (
    (f"{_CN.DATASET.TRAINVAL_DATA_SOURCE}_{_CN.IMAGE_SIZE}")
    + (f"_C{_CN.MODEL.COARSE_SCALE_IDX}")
    + (f"_F{_CN.MODEL.FINE_SCALE_IDX}")
    + (f"_Extra{_CN.MODEL.BACKBONE.EXTRA_DEPTH}")
    + (f"_I{_CN.MODEL.REFINE_ITERS}R{_CN.MODEL.REFINE_LOOKUP_RADIUS}_")
    + ("A" if _CN.DATASET.AUGMENTATION_TYPE is not None else "")
    + ("nCTXT" if not _CN.MODEL.REFINEMENT.CONTEXT_INJECTION else "")
)

########    Profiler Configurations    ########
_CN.PROFILER = CN()
_CN.PROFILER.PROFILER_NAME = None                  # options: [None, "inference", "pytorch"], defaults: None


def get_cfg_defaults():
    # Set the seed
    if _CN.GLOBAL_SEED is None:
        seed = os.environ.get("GLOBAL_SEED")
        if seed:
            _CN.GLOBAL_SEED = int(seed)
        else:
            # set a random number with current time as random seed
            random.seed(a=None)
            _CN.GLOBAL_SEED = random.randint(0, 4294967295)
            os.environ["GLOBAL_SEED"] = str(_CN.GLOBAL_SEED)

            # print out the random seed and logger name
            print("", flush=True)
            print("#" * 128, flush=True)
            print(f"\tRandom seed: {_CN.GLOBAL_SEED}", flush=True)
            print(f"\tLogger name: {_CN.LOGGER.LOGGER_NAME}", flush=True)
            print("#" * 128, flush=True)
            print("", flush=True)
            print(_CN)

    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()


if __name__ == "__main__":
    print("", flush=True)
    print("#" * 128, flush=True)
    print(f"\tRandom seed: {_CN.GLOBAL_SEED}", flush=True)
    print(f"\tLogger name: {_CN.LOGGER.LOGGER_NAME}", flush=True)
    print("#" * 128, flush=True)
    print("", flush=True)
    print(_CN)
