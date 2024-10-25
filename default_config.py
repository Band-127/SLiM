import os
import random
from datetime import datetime
from yacs.config import CfgNode as CN

_CN = CN()

########    General Configurations    ########
_CN.DEBUG = False
_CN.OVERALL_MODE = "train"          # options: ["train", "test"]
_CN.GLOBAL_SEED = 66                # for reproducibility, None for random
_CN.IMAGE_SIZE = 640
_CN.DTYPE = "float32"
_CN.PRETRAINED_PATH = None
_CN.DUMP_DIR = "dump/maff_baseline_outdoor"

########    Device Configurations    ########
# Support CUDA/CPU only!!!
_CN.DEVICE = CN()
_CN.DEVICE.ENABLE_GPU = True        # Whether enable GPUs, default true
_CN.DEVICE.ENABLE_DDP = True        # Whether enable distributed data parallel, default true
_CN.DEVICE.GPU_IDX = "1,2,3,4,5,7"          # GPUs indices, e.g. "0,1,2,3,4,5,6,7"
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
_CN.DATASET.AUGMENTATION_TYPE = None                # options: [None, "dark", "mobile", "maff", "maff_lite"]
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
_CN.TRAINER.GRADIENT_CLIPPING = 1.0                 # Gradient clipping
_CN.TRAINER.CANONICAL_BS = 8
_CN.TRAINER.CANONICAL_LR = 3e-3                     # using LR finder provided by pytorch lightning
_CN.TRAINER.SCALING = None                          # this will be calculated automatically
_CN.TRAINER.FIND_LR = False                         # use learning rate finder from pytorch-lightning, TODO: fix lr finder
_CN.TRAINER.FIRST_STAGE_EPOCHS = 2                  # first stage epochs
# optimizer
_CN.TRAINER.OPTIMIZER = "AdamW"                     # options: [Adam, AdamW]
_CN.TRAINER.TRUE_LR = None
_CN.TRAINER.ADAM_DECAY = 0.1
_CN.TRAINER.ADAMW_DECAY = 0.1
# learning rate scheduler
_CN.TRAINER.SCHEDULER = "MultiStepLR"                   # options: [MultiStepLR, CosineAnnealing, ExponentialLR, CosineAnnealingWarmRestarts]
_CN.TRAINER.SCHEDULER_INTERVAL = "epoch"                # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [2, 4, 8, 12]                # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.2
_CN.TRAINER.COSA_TMAX = 30                              # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992                        # ELR: ExponentialLR, this value for "step" interval
_CN.TRAINER.COSAWR_T0 = 3000                            # COSAWR: CosineAnnealingWarmRestarts, T0 means the first restart step
_CN.TRAINER.COSAWR_TMULT = 2                            # COSAWR: CosineAnnealingWarmRestarts, Tmult in CosineAnnealingWarmRestarts, check by urself
_CN.TRAINER.COSAWR_ETAMIN = 1e-8
# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'                      # options: [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.1
_CN.TRAINER.WARMUP_STEP = 1000                          # first epoch as warm up epoch
# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 64                    # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'                    # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'
# For metric calculation
_CN.TRAINER.RANSAC_PIXEL_THR = 0.2
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
_CN.MODEL.SHOW_GT_MATCHED_FINE = False
_CN.MODEL.DTYPE = _CN.DTYPE
_CN.MODEL.VERSION = "v1"                            # options: ["v1", "v2"]
_CN.MODEL.SCALES_SELECTION = (1, 1, 1, 1)           # E.g. if BACKBONE.RESOLUTION = (2, 4, 8), SCALES_SELECTION = (0, 1, 1), means only 1/4 and 1/8 feature maps are selected for feature fusion
_CN.MODEL.COARSE_SCALE_IDX = 1
_CN.MODEL.COARSE_SCALE = None                       # Will be calculated automatically
_CN.MODEL.FINE_SCALE_IDX = 0
_CN.MODEL.FINE_SCALE = None                         # Will be calculated automatically
_CN.MODEL.DIMENSION = 256
_CN.MODEL.USING_MAMBA2 = True
# refinement
_CN.MODEL.ENABLE_FUSION = False                     # Whether using feature fusion or not
_CN.MODEL.DISABLE_PE = False                        # Whether using pe before encoder or not
_CN.MODEL.PIXEL_SHUFFLE_REFINEMENT = True           # Whether using pixel shuffle refinement for fine coordinates generation
_CN.MODEL.CONF_MASK_DEPTH_REFINEMENT = True         # Whether using depth map to refine conf mask(generate confidence mask from output feature using mlp to mask unwanted area in correlation)

# Feature Backbone
_CN.MODEL.BACKBONE = CN()
_CN.MODEL.BACKBONE.BACKBONE_TYPE = "VMamba_T_cropped_FPN"
# backbone options: 
# [
#   "ResNet18", "ResNet18_modified", "ResNet18_pretrained",                         <-- ResNet in LoFTR, changed batchnorm into layernorm, original ResNet and pretrained weights
#   "VMamba_T", "VMamba_S", "VMamba_B",                                             <-- means pretrained, patch size = 4, 1/2 is extracted after patch embedding with a pixel shuffle(x2)
#   "VMamba_T_modifed", "VMamba_S_modified", "VMamba_B_modified",                   <-- means non pretrained, patch size = 2
#   "ResNet18_pretrained_FPN" , "VMamba_T_FPN", "VMamba_S_FPN", "VMamba_B_FPN",     <-- pretrained and with FPN
#   "VMamba_T_cropped", "VMamba_S_cropped", "VMamba_B_cropped",                     <-- pretrained without last two layers, 1/2 is extracted after patch embedding with a pixel shuffle(x2)
#   "VMamba_T_cropped_FPN"
#   "RepVGG", "RepVGG_FPN", "RepVGG_cropped"                                        <-- first two normal RepVGG, the last one is same as Efficient LoFTR, which patch size = 2
#   "RepVGG_pretrained", "RepVGG_pretrained_FPN", "RepVGG_pretrained_cropped"       <-- pretrained, add a fpn, without last two layers
# ]
# Efficient LoFTR using RepVGG_cropped
_CN.MODEL.BACKBONE.RESOLUTION = (4, 8, 16, 32)                          # options: [(2, 4, 8), (2, 4, 8, 16)] for ResNet18 and ResNet18_modified, will automatically set for ResNet18_pretrained and VMamba
_CN.MODEL.BACKBONE.LAYER_DIMS = (64, 128, 256, 512)                     # options: (128, 196, 256)(Modified by LoFTR), will automatically set for ResNet18_pretrained and VMamba
_CN.MODEL.BACKBONE.INPUT_SIZE = _CN.IMAGE_SIZE
_CN.MODEL.BACKBONE.FPN_OUT_CHANNELS = _CN.MODEL.DIMENSION

# Mamba Feature Fusion
_CN.MODEL.MAMBA_FUSION = CN()
_CN.MODEL.MAMBA_FUSION.USING_MAMBA2 = _CN.MODEL.USING_MAMBA2
_CN.MODEL.MAMBA_FUSION.INNER_EXPANSION = 2          # Inner dimension expansion rate for mamba, inner dimension=rate*input dimension
_CN.MODEL.MAMBA_FUSION.CONV_DIM = 4                 # Conv dimension for mamba
_CN.MODEL.MAMBA_FUSION.DELTA = 16                   # Delta dimension for mamba
_CN.MODEL.MAMBA_FUSION.SELF_NUM_LAYER = 0           # number of "self attn." layer
_CN.MODEL.MAMBA_FUSION.CROSS_NUM_LAYER = 2          # number of "cross attn." layer
_CN.MODEL.MAMBA_FUSION.LAYER_TYPES = ["self"] * _CN.MODEL.MAMBA_FUSION.SELF_NUM_LAYER + \
                                     ["cross"] * _CN.MODEL.MAMBA_FUSION.CROSS_NUM_LAYER

# Coarse Encoder
_CN.MODEL.COARSE_ENCODER = CN()
_CN.MODEL.COARSE_ENCODER.NUM_LAYERS = 1
_CN.MODEL.COARSE_ENCODER.INNER_EXPANSION = 2
_CN.MODEL.COARSE_ENCODER.CONV_DIM = 4
_CN.MODEL.COARSE_ENCODER.DELTA = 16
_CN.MODEL.COARSE_ENCODER.USING_MAMBA2 = _CN.MODEL.USING_MAMBA2

# Fine Encoder
_CN.MODEL.FINE_ENCODER = CN()
_CN.MODEL.FINE_ENCODER.NUM_LAYERS = 2
_CN.MODEL.FINE_ENCODER.INNER_EXPANSION = 2
_CN.MODEL.FINE_ENCODER.CONV_DIM = 4
_CN.MODEL.FINE_ENCODER.DELTA = 16
_CN.MODEL.FINE_ENCODER.USING_MAMBA2 = _CN.MODEL.USING_MAMBA2

# Coarse matching
_CN.MODEL.COARSE_MATCHING = CN()
_CN.MODEL.COARSE_MATCHING.THRESHOLD = 0.3
_CN.MODEL.COARSE_MATCHING.MAX_MATCHES = 2500

########    Loss Configurations    ########
_CN.LOSS = CN()
_CN.LOSS.VERSION = _CN.MODEL.VERSION
# COARSE MATCHING
_CN.LOSS.COARSE_WEIGHT = 1.0
_CN.LOSS.FOCAL_ALPHA = 0.25
_CN.LOSS.FOCAL_GAMMA = 2.0
# FINE MATCHING
_CN.LOSS.FINE_WEIGHT = 1.0
_CN.LOSS.FINE_TYPE = 'l2'                       # options: ['l2']
_CN.LOSS.FINE_THR = 1.0
# CONFIDENCE MASK REFINEMENT
_CN.LOSS.CONF_MASK_DEPTH_REFINEMENT = _CN.MODEL.CONF_MASK_DEPTH_REFINEMENT

########    Profiler Configurations    ########
_CN.PROFILER = CN()
_CN.PROFILER.PROFILER_NAME = None                   # options: [None, "inference", "pytorch"], Default: None -> PassThroughProfiler

# Set model backbone settings for VMamba and pretrained ResNet18
if "VMamba_T" in _CN.MODEL.BACKBONE.BACKBONE_TYPE:
    if "modified" not in _CN.MODEL.BACKBONE.BACKBONE_TYPE:
        _CN.MODEL.BACKBONE.RESOLUTION = (2, 4, 8, 16, 32)
        _CN.MODEL.BACKBONE.LAYER_DIMS = (24, 96, 192, 384, 768)
        _CN.MODEL.SCALES_SELECTION = (0, 1, 1, 1, 1)
        _CN.MODEL.COARSE_SCALE_IDX = 2
        _CN.MODEL.FINE_SCALE_IDX = 0
    else:
        _CN.MODEL.BACKBONE.RESOLUTION = (2, 4, 8)
        _CN.MODEL.BACKBONE.LAYER_DIMS = (192, 384, 768)
        _CN.MODEL.SCALES_SELECTION = (0, 1, 1, 1)
        _CN.MODEL.COARSE_SCALE_IDX = 2
        _CN.MODEL.FINE_SCALE_IDX = 0
elif "VMamba_S" in _CN.MODEL.BACKBONE.BACKBONE_TYPE:
    if "modified" not in _CN.MODEL.BACKBONE.BACKBONE_TYPE:
        _CN.MODEL.BACKBONE.RESOLUTION = (2, 4, 8, 16, 32)
        _CN.MODEL.BACKBONE.LAYER_DIMS = (24, 96, 192, 384, 768)
        _CN.MODEL.SCALES_SELECTION = (0, 1, 1, 1, 1)
        _CN.MODEL.COARSE_SCALE_IDX = 2
        _CN.MODEL.FINE_SCALE_IDX = 0
    else:
        _CN.MODEL.BACKBONE.RESOLUTION = (2, 4, 8)
        _CN.MODEL.BACKBONE.LAYER_DIMS = (96, 192, 384)
        _CN.MODEL.SCALES_SELECTION = (0, 1, 1, 1)
        _CN.MODEL.COARSE_SCALE_IDX = 2
        _CN.MODEL.FINE_SCALE_IDX = 0
elif "VMamba_B" in _CN.MODEL.BACKBONE.BACKBONE_TYPE:
    if "modified" not in _CN.MODEL.BACKBONE.BACKBONE_TYPE:
        _CN.MODEL.BACKBONE.RESOLUTION = (2, 4, 8, 16, 32)
        _CN.MODEL.BACKBONE.LAYER_DIMS = (32, 128, 256, 512, 1024)
        _CN.MODEL.SCALES_SELECTION = (0, 1, 1, 1, 1)
        _CN.MODEL.COARSE_SCALE_IDX = 2
        _CN.MODEL.FINE_SCALE_IDX = 0
    else:
        _CN.MODEL.BACKBONE.RESOLUTION = (2, 4, 8)
        _CN.MODEL.BACKBONE.LAYER_DIMS = (128, 256, 512)
        _CN.MODEL.SCALES_SELECTION = (0, 1, 1, 1)
        _CN.MODEL.COARSE_SCALE_IDX = 2
        _CN.MODEL.FINE_SCALE_IDX = 0
elif "ResNet18_pretrained" in _CN.MODEL.BACKBONE.BACKBONE_TYPE:
    _CN.MODEL.BACKBONE.RESOLUTION = (2, 4, 8, 16, 32)
    _CN.MODEL.BACKBONE.LAYER_DIMS = (64, 64, 128, 256, 512)
    _CN.MODEL.SCALES_SELECTION = (0, 1, 1, 1, 1)
    _CN.MODEL.COARSE_SCALE_IDX = 2
    _CN.MODEL.FINE_SCALE_IDX = 0
elif "RepVGG" in _CN.MODEL.BACKBONE.BACKBONE_TYPE:
    _CN.MODEL.BACKBONE.RESOLUTION = (2, 4, 8, 16, 32)
    _CN.MODEL.BACKBONE.LAYER_DIMS = (64, 64, 128, 256, 1280)
    _CN.MODEL.SCALES_SELECTION = (0, 1, 1, 1, 1)
    _CN.MODEL.COARSE_SCALE_IDX = 2
    _CN.MODEL.FINE_SCALE_IDX = 0
if "cropped" in _CN.MODEL.BACKBONE.BACKBONE_TYPE:
    _CN.MODEL.BACKBONE.RESOLUTION = _CN.MODEL.BACKBONE.RESOLUTION[0: len(_CN.MODEL.BACKBONE.RESOLUTION) - 2]
    _CN.MODEL.BACKBONE.LAYER_DIMS = _CN.MODEL.BACKBONE.LAYER_DIMS[0: len(_CN.MODEL.BACKBONE.LAYER_DIMS) - 2]
    _CN.MODEL.SCALES_SELECTION = _CN.MODEL.SCALES_SELECTION[0: len(_CN.MODEL.SCALES_SELECTION) - 2]
if _CN.MODEL.BACKBONE.BACKBONE_TYPE == "RepVGG_cropped":
    _CN.MODEL.BACKBONE.RESOLUTION = (2, 4, 8)
    _CN.MODEL.BACKBONE.LAYER_DIMS = (64, 128, 256)
    _CN.MODEL.SCALES_SELECTION = (0, 1, 1)
    _CN.MODEL.COARSE_SCALE_IDX = 2
    _CN.MODEL.FINE_SCALE_IDX = 0
elif _CN.MODEL.BACKBONE.BACKBONE_TYPE == "RepVGG_pretrained_cropped":
    _CN.MODEL.BACKBONE.RESOLUTION = (2, 4, 8)
    _CN.MODEL.BACKBONE.LAYER_DIMS = (64, 64, 128)
    _CN.MODEL.SCALES_SELECTION = (1, 1, 1)
    _CN.MODEL.COARSE_SCALE_IDX = 2
    _CN.MODEL.FINE_SCALE_IDX = 0

# Calculate coarse scale
_CN.DATASET.MGDPT_COARSE_SCALE = _CN.MODEL.COARSE_SCALE = _CN.MODEL.BACKBONE.RESOLUTION[_CN.MODEL.COARSE_SCALE_IDX]
# Calculate fine scale
_CN.MODEL.FINE_SCALE = _CN.MODEL.BACKBONE.RESOLUTION[_CN.MODEL.FINE_SCALE_IDX]

########    Logger Configurations    ########
_CN.LOGGER = CN()
if _CN.MODEL.VERSION == "v1":
    _CN.LOGGER.LOGGER_NAME = (f"{_CN.DATASET.TRAINVAL_DATA_SOURCE}_{_CN.IMAGE_SIZE}_{_CN.MODEL.SCALES_SELECTION}_") + \
                            (_CN.MODEL.VERSION + "_") + \
                            (f"{_CN.MODEL.COARSE_SCALE}_") + \
                            (f"{_CN.MODEL.FINE_SCALE}_") + \
                            (f"{_CN.MODEL.COARSE_ENCODER.NUM_LAYERS if not _CN.MODEL.ENABLE_FUSION else _CN.MODEL.MAMBA_FUSION.CROSS_NUM_LAYER}+{_CN.MODEL.FINE_ENCODER.NUM_LAYERS}") + \
                            (f"_{_CN.MODEL.BACKBONE.BACKBONE_TYPE}_") + \
                            ("F" if _CN.MODEL.ENABLE_FUSION else "") + \
                            ("P" if _CN.MODEL.PIXEL_SHUFFLE_REFINEMENT else "") + \
                            ("D" if _CN.MODEL.CONF_MASK_DEPTH_REFINEMENT else "") + \
                            ("A" if _CN.DATASET.AUGMENTATION_TYPE is not None else "")
elif _CN.MODEL.VERSION == "v2":
    _CN.LOGGER.LOGGER_NAME = (f"{_CN.DATASET.TRAINVAL_DATA_SOURCE}_{_CN.IMAGE_SIZE}_") + \
                            (_CN.MODEL.VERSION + "_") + \
                            (f"{_CN.MODEL.COARSE_SCALE}_") + \
                            (f"{_CN.MODEL.COARSE_ENCODER.NUM_LAYERS}+{_CN.MODEL.FINE_ENCODER.NUM_LAYERS}") + \
                            (f"_{_CN.MODEL.BACKBONE.BACKBONE_TYPE}_") + \
                            ("P" if _CN.MODEL.PIXEL_SHUFFLE_REFINEMENT else "") + \
                            ("D" if _CN.MODEL.CONF_MASK_DEPTH_REFINEMENT else "") + \
                            ("A" if _CN.DATASET.AUGMENTATION_TYPE is not None else "")


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
            print("", flush=True)
            print("#" * 128, flush=True)
            print(f"\tRandom seed: {_CN.GLOBAL_SEED}", flush=True)
            print(f"\tLogger name: {_CN.LOGGER.LOGGER_NAME}", flush=True)
            print("#" * 128, flush=True)
            print("", flush=True)

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
