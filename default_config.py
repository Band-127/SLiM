import random
from datetime import datetime
from yacs.config import CfgNode as CN

_CN = CN()

########    General Configurations    ########
_CN.DEBUG = False
_CN.OVERALL_MODE = "train"          # options: ["train", "test"]
_CN.GLOBAL_SEED = None              # for reproducibility, None for random
_CN.IMAGE_SIZE = 640
_CN.DTYPE = "float32"
_CN.PRETRAINED_PATH = None

########    Device Configurations    ########
# Support CUDA/CPU only!!!
_CN.DEVICE = CN()
_CN.DEVICE.ENABLE_GPU = True        # Whether enable GPUs, default true
_CN.DEVICE.ENABLE_DDP = True        # Whether enable distributed data parallel, default true
_CN.DEVICE.GPU_IDX = "0,2,3,4,5,6"  # GPUs indices, e.g. "0,1,2,3,4,5,6,7"
_CN.DEVICE.NUM_NODES = 1
_CN.DEVICE.MASTER_ADDR = "localhost"
_CN.DEVICE.MASTER_PORT = "29500"

########    Dataset Configurations    ########
_CN.DATASET = CN()
# training
_CN.DATASET.TRAINVAL_DATA_SOURCE = "MegaDepth"                                                              # options: ["ScanNet", "MegaDepth"]
_CN.DATASET.TRAIN_DATA_BASE_PATH = "data/megadepth"
_CN.DATASET.TRAIN_DATA_ROOT = f"{_CN.DATASET.TRAIN_DATA_BASE_PATH}/train"
_CN.DATASET.TRAIN_POSE_ROOT = None                                                                          # (optional directory for poses)
_CN.DATASET.TRAIN_NPZ_ROOT = f"{_CN.DATASET.TRAIN_DATA_BASE_PATH}/index/scene_info_0.1_0.7"                 # None if val data from all scenes are bundled into a single npz file
_CN.DATASET.TRAIN_LIST_PATH = f"{_CN.DATASET.TRAIN_DATA_BASE_PATH}/index/trainvaltest_list/train_list.txt"
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
# validating
_CN.DATASET.VAL_DATA_BASE_PATH = "data/megadepth"
_CN.DATASET.VAL_DATA_ROOT = f"{_CN.DATASET.VAL_DATA_BASE_PATH}/test"
_CN.DATASET.VAL_POSE_ROOT = None                                                                            # (optional directory for poses)
_CN.DATASET.VAL_NPZ_ROOT = f"{_CN.DATASET.VAL_DATA_BASE_PATH}/index/scene_info_val_1500"
_CN.DATASET.VAL_LIST_PATH = f"{_CN.DATASET.VAL_DATA_BASE_PATH}/index/trainvaltest_list/val_list.txt"        # None if val data from all scenes are bundled into a single npz file
_CN.DATASET.VAL_INTRINSIC_PATH = None
# testing
_CN.DATASET.TEST_DATA_SOURCE = "MegaDepth"                                                                  # options: ["ScanNet", "MegaDepth"]
_CN.DATASET.TEST_DATA_BASE_PATH = "data/megadepth"
_CN.DATASET.TEST_DATA_ROOT = f"{_CN.DATASET.TEST_DATA_BASE_PATH}/test"
_CN.DATASET.TEST_POSE_ROOT = None                                                                           # (optional directory for poses)
_CN.DATASET.TEST_NPZ_ROOT = f"{_CN.DATASET.TEST_DATA_BASE_PATH}/index/scene_info_val_1500"
_CN.DATASET.TEST_LIST_PATH = f"{_CN.DATASET.TEST_DATA_BASE_PATH}/index/trainvaltest_list/val_list.txt"      # None if test data from all scenes are bundled into a single npz file
_CN.DATASET.TEST_INTRINSIC_PATH = None
# general options
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0           # discard data with overlap_score < min_overlap_score
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = None                # options: [None, "dark", "mobile"]
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
_CN.LOADER.BATCH_SIZE = 1                           # how many samples per batch to load, per gpu
_CN.LOADER.NUM_WORKERS = 4                          # how many subprocesses to use for data loading.
_CN.LOADER.PIN_MEMORY = True                        # If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them.

########    Trainer Configurations    ########
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = None                       # Will be calculated using number of nodes and exact number of devices available
_CN.TRAINER.GRADIENT_CLIPPING = 0.5                 # Gradient clipping
_CN.TRAINER.CANONICAL_BS = 64
_CN.TRAINER.CANONICAL_LR = 8e-3
_CN.TRAINER.SCALING = None                          # this will be calculated automatically
_CN.TRAINER.FIND_LR = True                          # use learning rate finder from pytorch-lightning
# optimizer
_CN.TRAINER.OPTIMIZER = "AdamW"                     # options: [Adam, AdamW]
_CN.TRAINER.TRUE_LR = 2e-3                          # using LR finder provided by pytorch lightning
_CN.TRAINER.ADAM_DECAY = 0.1
_CN.TRAINER.ADAMW_DECAY = 0.1
# learning rate scheduler
_CN.TRAINER.SCHEDULER = "MultiStepLR"               # options: [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = "epoch"            # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [2, 4, 8, 12, 16]     # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30                          # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992                    # ELR: ExponentialLR, this value for "step" interval
# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'constant'                # options: [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.1
_CN.TRAINER.WARMUP_STEP = 1875
# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32     # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'

########    Logging Configurations    ########
_CN.LOGGER = CN()
_CN.LOGGER.EXP_NAME = f"{datetime.isoformat(datetime.now(), sep='-', timespec='seconds')}"

########    Profiler Configurations    ########
_CN.PROFILER = CN()
_CN.PROFILER.PROFILER__NAME = None                  # options: [None, "inference", "pytorch"], defaults: None

########    Model Configurations    ########
_CN.MODEL = CN()
_CN.MODEL.DEBUG = _CN.DEBUG
_CN.MODEL.DTYPE = _CN.DTYPE
_CN.MODEL.FUSION_TYPE = "mamba"                     # options: ["mamba", "transformer"]
_CN.MODEL.SCALES_SELECTION = (0, 1, 1)              # E.g. if BACKBONE.RESOLUTION = (2, 4, 8), SCALES_SELECTION = (0, 1, 1), means only 1/4 and 1/8 feature maps are selected for feature fusion
_CN.MODEL.COARSE_SCALE_IDX = 1
_CN.MODEL.COARSE_SCALE = None                       # Will be calculated automatically
# Feature Backbone
_CN.MODEL.BACKBONE = CN()
_CN.MODEL.BACKBONE.BACKBONE_TYPE = "ResNet18"       # options: ["ResNet18"]
_CN.MODEL.BACKBONE.RESOLUTION = (2, 4, 8)           # options: [(2, 4, 8), (2, 4, 8, 16)] for ResNet18
_CN.MODEL.BACKBONE.LAYER_DIMS = (128, 196, 256)     # options: [(128, 196, 256)(Modified by LoFTR), (64, 128, 256, 512)] for ResNet18
_CN.MODEL.BACKBONE.INPUT_SIZE = _CN.IMAGE_SIZE
# Mamba Feature Fusion
_CN.MODEL.MAMBA_FUSION = CN()
_CN.MODEL.MAMBA_FUSION.USING_MAMBA2 = True          # Whether using mamba2 or not
_CN.MODEL.MAMBA_FUSION.INNER_EXPANSION = 2          # Inner dimension expansion rate for mamba, inner dimension=rate*input dimension
_CN.MODEL.MAMBA_FUSION.CONV_DIM = 4                 # Conv dimension for mamba
_CN.MODEL.MAMBA_FUSION.SELF_NUM_LAYER = 0           # number of "self attn." layer
_CN.MODEL.MAMBA_FUSION.CROSS_NUM_LAYER = 4          # number of "cross attn." layer
_CN.MODEL.MAMBA_FUSION.LAYER_TYPES = ["self"] * _CN.MODEL.MAMBA_FUSION.SELF_NUM_LAYER + \
                                     ["cross"] * _CN.MODEL.MAMBA_FUSION.CROSS_NUM_LAYER
# Transformer Feature Fusion (comparison)
_CN.MODEL.TRANSFORMER_FUSION = CN()
_CN.MODEL.TRANSFORMER_FUSION.D_MODEL = _CN.MODEL.BACKBONE.LAYER_DIMS[-1]
_CN.MODEL.TRANSFORMER_FUSION.NHEAD = 8
_CN.MODEL.TRANSFORMER_FUSION.ATTENTION = "linear"
_CN.MODEL.TRANSFORMER_FUSION.LAYERS = 1             # number of self+cross attn. layer
_CN.MODEL.TRANSFORMER_FUSION.LAYER_TYPES = ['self', 'cross'] * _CN.MODEL.TRANSFORMER_FUSION.LAYERS


########    Loss Configurations    ########
_CN.LOSS = CN()
_CN.LOSS.POS_WEIGHT = 1.0
_CN.LOSS.NEG_WEIGHT = 1.0
_CN.LOSS.COARSE_WEIGHT = 1.0
_CN.LOSS.COARSE_TYPE = 'focal'                      # options: ['focal', 'cross_entropy']
_CN.LOSS.FOCAL_ALPHA = 0.25
_CN.LOSS.FOCAL_GAMMA = 2.0

########    Profiler Configurations    ########
_CN.PROFILER = CN()
_CN.PROFILER.PROFILER_NAME = None                   # options: [None, "inference", "pytorch"], Default: None -> PassThroughProfiler

########    Logger Configurations    ########
_CN.LOGGER = CN()
_CN.LOGGER.LOGGER_NAME = (f"{_CN.DATASET.TRAINVAL_DATA_SOURCE}_{_CN.IMAGE_SIZE}_{_CN.MODEL.SCALES_SELECTION}_") + \
                        (f"{_CN.MODEL.COARSE_SCALE_IDX}_") + \
                        ("M" if _CN.MODEL.FUSION_TYPE == "mamba" else "T") + \
                        ("2" if _CN.MODEL.MAMBA_FUSION.USING_MAMBA2 else "")

# geometric metrics and pose solver
_CN.TRAINER.EPI_ERR_THR = 5e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
_CN.TRAINER.POSE_GEO_MODEL = "E"  # ["E", "F", "H"]
_CN.TRAINER.POSE_ESTIMATION_METHOD = "RANSAC"  # [RANSAC, DEGENSAC, MAGSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False


def get_cfg_defaults():
    if _CN.GLOBAL_SEED is None:
        # set a random number with current time as random seed 
        random.seed(a=None)
        _CN.GLOBAL_SEED = random.randint(0, 4294967295)

        # speak out the random seed
        print("#"*64 + f"\nRandom seed: {_CN.GLOBAL_SEED}\n" + "#"*64)
        
    # Calculate coarse scale
    reach = _CN.MODEL.COARSE_SCALE_IDX
    for idx, i in enumerate(_CN.MODEL.SCALES_SELECTION):
        if i:
            if reach:
                reach -= 1
                continue
            _CN.DATASET.MGDPT_COARSE_SCALE = _CN.MODEL.COARSE_SCALE = _CN.MODEL.BACKBONE.RESOLUTION[idx]
            break

    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
