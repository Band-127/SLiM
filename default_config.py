import os
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
_CN.DUMP_DIR = "dump/maff_baseline_outdoor"

########    Device Configurations    ########
# Support CUDA/CPU only!!!
_CN.DEVICE = CN()
_CN.DEVICE.ENABLE_GPU = True        # Whether enable GPUs, default true
_CN.DEVICE.ENABLE_DDP = True        # Whether enable distributed data parallel, default true
_CN.DEVICE.GPU_IDX = "1,2,3,4"      # GPUs indices, e.g. "0,1,2,3,4,5,6,7"
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
_CN.DATASET.AUGMENTATION_TYPE = "maff"              # options: [None, "dark", "mobile", "maff"]
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
_CN.TRAINER.CANONICAL_BS = 8
_CN.TRAINER.CANONICAL_LR = 5e-4                     # using LR finder provided by pytorch lightning
_CN.TRAINER.SCALING = None                          # this will be calculated automatically
_CN.TRAINER.FIND_LR = False                         # use learning rate finder from pytorch-lightning, TODO: fix lr finder
_CN.TRAINER.FIRST_STAGE_EPOCHS = 2                  # first stage epochs
# optimizer
_CN.TRAINER.OPTIMIZER = "AdamW"                     # options: [Adam, AdamW]
_CN.TRAINER.TRUE_LR = None
_CN.TRAINER.ADAM_DECAY = 0.1
_CN.TRAINER.ADAMW_DECAY = 0.1
# learning rate scheduler
_CN.TRAINER.SCHEDULER = "MultiStepLR"               # options: [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = "epoch"            # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [4, 8, 12, 16, 20]    # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30                          # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992                    # ELR: ExponentialLR, this value for "step" interval
# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'                  # options: [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.1
_CN.TRAINER.WARMUP_STEP = 500
# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 64                # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'                # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'
# For metric calculation
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.EPI_ERR_THR = 5e-4 if _CN.DATASET.TRAINVAL_DATA_SOURCE == "ScanNet" else 1e-4   # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)

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
_CN.MODEL.VERSION = "v2"                            # options: ["v1", "v2"]
_CN.MODEL.FUSION_TYPE = None                        # options: ["mamba", "transformer", None], None for no feature fusion
_CN.MODEL.SCALES_SELECTION = (1, 1, 1, 1)           # E.g. if BACKBONE.RESOLUTION = (2, 4, 8), SCALES_SELECTION = (0, 1, 1), means only 1/4 and 1/8 feature maps are selected for feature fusion
_CN.MODEL.COARSE_SCALE_IDX = 1
_CN.MODEL.COARSE_SCALE = None                       # Will be calculated automatically
_CN.MODEL.FINE_SCALE_IDX = 0
_CN.MODEL.FINE_SCALE = None                         # Will be calculated automatically
_CN.MODEL.DIMENSION = 256
_CN.MODEL.USING_MAMBA2 = True
# refinement
_CN.MODEL.DISABLE_PE = False                        # Whether using pe before encoder or not
_CN.MODEL.PIXEL_SHUFFLE_REFINEMENT = True           # Whether using pixel shuffle refinement for fine coordinates generation
_CN.MODEL.CONF_MASK_DEPTH_REFINEMENT = True         # Whether using depth map to refine conf mask(generate confidence mask from output feature using mlp to mask unwanted area in correlation)
_CN.MODEL.FINE_REFINEMENT = True                    # Whether using feature refinement network for fine feature
_CN.MODEL.COORD_REFINEMENT = False                  # Whether using coordinate refinement for fine coordinates generation

# Feature Backbone
_CN.MODEL.BACKBONE = CN()
_CN.MODEL.BACKBONE.BACKBONE_TYPE = "VMamba_T"
# backbone options: 
# [
#   "ResNet18", "ResNet18_modified", "ResNet18_pretrained", 
#   "VMamba_T", "VMamba_S", "VMamba_B", 
#   "ResNet18_pretrained_FPN" , "VMamba_T_FPN", "VMamba_S_FPN", "VMamba_B_FPN", 
#   "VMamba_T_cropped", "VMamba_S_cropped", "VMamba_B_cropped",
# ]
_CN.MODEL.BACKBONE.RESOLUTION = (4, 8, 16, 32)                          # options: [(2, 4, 8), (2, 4, 8, 16)] for ResNet18 and ResNet18_modified, will automatically set for ResNet18_pretrained and VMamba
_CN.MODEL.BACKBONE.LAYER_DIMS = (64, 128, 256, 512)                     # options: (128, 196, 256)(Modified by LoFTR), will automatically set for ResNet18_pretrained and VMamba
_CN.MODEL.BACKBONE.INPUT_SIZE = _CN.IMAGE_SIZE
_CN.MODEL.BACKBONE.FPN_OUT_CHANNELS = _CN.MODEL.DIMENSION

# Mamba Feature Fusion
_CN.MODEL.MAMBA_FUSION = CN()
_CN.MODEL.MAMBA_FUSION.USING_MAMBA2 = _CN.MODEL.USING_MAMBA2
_CN.MODEL.MAMBA_FUSION.INNER_EXPANSION = 2          # Inner dimension expansion rate for mamba, inner dimension=rate*input dimension
_CN.MODEL.MAMBA_FUSION.CONV_DIM = 3                 # Conv dimension for mamba
_CN.MODEL.MAMBA_FUSION.DELTA = 16                   # Delta dimension for mamba
_CN.MODEL.MAMBA_FUSION.SELF_NUM_LAYER = 0           # number of "self attn." layer
_CN.MODEL.MAMBA_FUSION.CROSS_NUM_LAYER = 0          # number of "cross attn." layer
_CN.MODEL.MAMBA_FUSION.LAYER_TYPES = ["self"] * _CN.MODEL.MAMBA_FUSION.SELF_NUM_LAYER + \
                                     ["cross"] * _CN.MODEL.MAMBA_FUSION.CROSS_NUM_LAYER

# Transformer Feature Fusion (comparison)
_CN.MODEL.TRANSFORMER_FUSION = CN()
_CN.MODEL.TRANSFORMER_FUSION.D_MODEL = _CN.MODEL.BACKBONE.LAYER_DIMS[-1]
_CN.MODEL.TRANSFORMER_FUSION.NHEAD = 8
_CN.MODEL.TRANSFORMER_FUSION.ATTENTION = "linear"
_CN.MODEL.TRANSFORMER_FUSION.LAYERS = 1             # number of self+cross attn. layer
_CN.MODEL.TRANSFORMER_FUSION.LAYER_TYPES = ['self', 'cross'] * _CN.MODEL.TRANSFORMER_FUSION.LAYERS

# Fine Feature Refinement
_CN.MODEL.FINE_REFINEMENT_MODEL = CN()
_CN.MODEL.FINE_REFINEMENT_MODEL.INNER_EXPANSION = 2
_CN.MODEL.FINE_REFINEMENT_MODEL.CONV_DIM = 4
_CN.MODEL.FINE_REFINEMENT_MODEL.DELTA = 16
_CN.MODEL.FINE_REFINEMENT_MODEL.NUM_LAYER = 2
_CN.MODEL.FINE_REFINEMENT_MODEL.USING_MAMBA2 = _CN.MODEL.USING_MAMBA2

# Coordinate Refinement
_CN.MODEL.COORD_REFINEMENT_MODEL = CN()
_CN.MODEL.COORD_REFINEMENT_MODEL.INNER_EXPANSION = 256
_CN.MODEL.COORD_REFINEMENT_MODEL.CONV_DIM = 4
_CN.MODEL.COORD_REFINEMENT_MODEL.DELTA = 16
_CN.MODEL.COORD_REFINEMENT_MODEL.NUM_LAYER = 2
_CN.MODEL.COORD_REFINEMENT_MODEL.USING_MAMBA2 = _CN.MODEL.USING_MAMBA2

# Coarse matching
_CN.MODEL.COARSE_MATCHING = CN()
_CN.MODEL.COARSE_MATCHING.THRESHOLD = 0.3
_CN.MODEL.COARSE_MATCHING.MAX_MATCHES = 2500

# v2 config
_CN.MODEL.COARSE_ENCODER = CN()
_CN.MODEL.COARSE_ENCODER.NUM_LAYERS = 2
_CN.MODEL.COARSE_ENCODER.INNER_EXPANSION = 2
_CN.MODEL.COARSE_ENCODER.CONV_DIM = 4
_CN.MODEL.COARSE_ENCODER.DELTA = 16
_CN.MODEL.COARSE_ENCODER.USING_MAMBA2 = _CN.MODEL.USING_MAMBA2

_CN.MODEL.FINE_ENCODER = CN()
_CN.MODEL.FINE_ENCODER.NUM_LAYERS = 2
_CN.MODEL.FINE_ENCODER.INNER_EXPANSION = 2
_CN.MODEL.FINE_ENCODER.CONV_DIM = 4
_CN.MODEL.FINE_ENCODER.DELTA = 16
_CN.MODEL.FINE_ENCODER.USING_MAMBA2 = _CN.MODEL.USING_MAMBA2

########    Loss Configurations    ########
_CN.LOSS = CN()
# COARSE MATCHING
_CN.LOSS.COARSE_WEIGHT = 1.0
_CN.LOSS.FOCAL_ALPHA = 0.25
_CN.LOSS.FOCAL_GAMMA = 2.0
# FINE MATCHING
_CN.LOSS.FINE_WEIGHT = 1.0
_CN.LOSS.FINE_TYPE = 'l2'                           # options: ['l2', 'l2_std']
_CN.LOSS.FINE_THR = 1.0
# CONFIDENCE MASK REFINEMENT
_CN.LOSS.CONF_MASK_DEPTH_REFINEMENT = _CN.MODEL.CONF_MASK_DEPTH_REFINEMENT

########    Profiler Configurations    ########
_CN.PROFILER = CN()
_CN.PROFILER.PROFILER_NAME = None                   # options: [None, "inference", "pytorch"], Default: None -> PassThroughProfiler

# Set model backbone settings for VMamba and pretrained ResNet18
if "VMamba_T" in _CN.MODEL.BACKBONE.BACKBONE_TYPE:
    _CN.MODEL.BACKBONE.RESOLUTION = (2, 4, 8, 16, 32)
    _CN.MODEL.BACKBONE.LAYER_DIMS = (24, 96, 192, 384, 768)
    _CN.MODEL.SCALES_SELECTION = (0, 1, 1, 1, 1)
    _CN.MODEL.COARSE_SCALE_IDX = 2
    _CN.MODEL.FINE_SCALE_IDX = 1
elif "VMamba_S" in _CN.MODEL.BACKBONE.BACKBONE_TYPE:
    _CN.MODEL.BACKBONE.RESOLUTION = (2, 4, 8, 16, 32)
    _CN.MODEL.BACKBONE.LAYER_DIMS = (24, 96, 192, 384, 768)
    _CN.MODEL.SCALES_SELECTION = (0, 1, 1, 1, 1)
    _CN.MODEL.COARSE_SCALE_IDX = 2
    _CN.MODEL.FINE_SCALE_IDX = 1
elif "VMamba_B" in _CN.MODEL.BACKBONE.BACKBONE_TYPE:
    _CN.MODEL.BACKBONE.RESOLUTION = (2, 4, 8, 16, 32)
    _CN.MODEL.BACKBONE.LAYER_DIMS = (32, 128, 256, 512, 1024)
    _CN.MODEL.SCALES_SELECTION = (0, 1, 1, 1, 1)
    _CN.MODEL.COARSE_SCALE_IDX = 2
    _CN.MODEL.FINE_SCALE_IDX = 1
elif "ResNet18_pretrained" in _CN.MODEL.BACKBONE.BACKBONE_TYPE:
    _CN.MODEL.BACKBONE.RESOLUTION = (2, 4, 8, 16, 32)
    _CN.MODEL.BACKBONE.LAYER_DIMS = (64, 64, 128, 256, 512)
    _CN.MODEL.SCALES_SELECTION = (0, 1, 1, 1, 1)
    _CN.MODEL.COARSE_SCALE_IDX = 2
    _CN.MODEL.FINE_SCALE_IDX = 1
if "cropped" in _CN.MODEL.BACKBONE.BACKBONE_TYPE:
    _CN.MODEL.BACKBONE.RESOLUTION = _CN.MODEL.BACKBONE.RESOLUTION[0: len(_CN.MODEL.BACKBONE.RESOLUTION) - 1]
    _CN.MODEL.BACKBONE.LAYER_DIMS = _CN.MODEL.BACKBONE.LAYER_DIMS[0: len(_CN.MODEL.BACKBONE.LAYER_DIMS) - 1]
    _CN.MODEL.SCALES_SELECTION = _CN.MODEL.SCALES_SELECTION[0: len(_CN.MODEL.SCALES_SELECTION) - 1]

# Calculate coarse scale
_CN.DATASET.MGDPT_COARSE_SCALE = _CN.MODEL.COARSE_SCALE = _CN.MODEL.BACKBONE.RESOLUTION[_CN.MODEL.COARSE_SCALE_IDX]
# Calculate fine scale
_CN.MODEL.FINE_SCALE = _CN.MODEL.BACKBONE.RESOLUTION[_CN.MODEL.FINE_SCALE_IDX]

########    Logger Configurations    ########
_CN.LOGGER = CN()
FUSION_TYPE_MARK  = "None"
if _CN.MODEL.FUSION_TYPE == "mamba":
    FUSION_TYPE_MARK = "M"
    if _CN.MODEL.USING_MAMBA2:
        FUSION_TYPE_MARK += "2"
elif _CN.MODEL.FUSION_TYPE == "transformer":
    FUSION_TYPE_MARK = "T"
if _CN.MODEL.VERSION == "v1":
    _CN.LOGGER.LOGGER_NAME = (f"{_CN.DATASET.TRAINVAL_DATA_SOURCE}_{_CN.IMAGE_SIZE}_{_CN.MODEL.SCALES_SELECTION}_") + \
                            (_CN.MODEL.VERSION + "_") + \
                            (f"{_CN.MODEL.COARSE_SCALE}_") + \
                            (f"{_CN.MODEL.FINE_SCALE}_") + \
                            (f"{FUSION_TYPE_MARK}") + \
                            (f"_{_CN.MODEL.BACKBONE.BACKBONE_TYPE}_") + \
                            ("P" if _CN.MODEL.PIXEL_SHUFFLE_REFINEMENT else "") + \
                            ("D" if _CN.MODEL.CONF_MASK_DEPTH_REFINEMENT else "") + \
                            ("F" if _CN.MODEL.FINE_REFINEMENT else "") + \
                            ("C" if _CN.MODEL.COORD_REFINEMENT else "") + \
                            ("A" if _CN.DATASET.AUGMENTATION_TYPE is not None else "")
elif _CN.MODEL.VERSION == "v2":
    _CN.LOGGER.LOGGER_NAME = (f"{_CN.DATASET.TRAINVAL_DATA_SOURCE}_{_CN.IMAGE_SIZE}_") + \
                            (_CN.MODEL.VERSION + "_") + \
                            (f"{_CN.MODEL.COARSE_SCALE}_") + \
                            (f"{_CN.MODEL.COARSE_ENCODER.NUM_LAYERS}+{_CN.MODEL.FINE_ENCODER.NUM_LAYERS}")+\
                            (f"_{_CN.MODEL.BACKBONE.BACKBONE_TYPE}_") + \
                            ("P" if _CN.MODEL.PIXEL_SHUFFLE_REFINEMENT else "") + \
                            ("D" if _CN.MODEL.CONF_MASK_DEPTH_REFINEMENT else "") + \
                            ("A" if _CN.DATASET.AUGMENTATION_TYPE is not None else "")

# geometric metrics and pose solver
_CN.TRAINER.POSE_GEO_MODEL = "E"  # ["E", "F", "H"]
_CN.TRAINER.POSE_ESTIMATION_METHOD = "RANSAC"  # [RANSAC, DEGENSAC, MAGSAC]
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False


def get_cfg_defaults():
    # Set the seed
    if _CN.GLOBAL_SEED is None:
        if os.environ.get('GLOBAL_SEED'):
            _CN.GLOBAL_SEED = int(os.environ.get('GLOBAL_SEED'))
        else:
            # set a random number with current time as random seed
            random.seed(a=None)
            _CN.GLOBAL_SEED = random.randint(0, 4294967295)
            os.environ['GLOBAL_SEED'] = str(_CN.GLOBAL_SEED)

            # print out the random seed and logger name
            print("#" * 64 + f"\nRandom seed: {_CN.GLOBAL_SEED}\nLogger name: {_CN.LOGGER.LOGGER_NAME}\n" + "#" * 64)

    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()

if __name__ == "__main__":
    print("#" * 64 + f"\nRandom seed: {_CN.GLOBAL_SEED}\nLogger name: {_CN.LOGGER.LOGGER_NAME}\n" + "#" * 64)
