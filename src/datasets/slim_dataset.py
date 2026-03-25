import os.path as osp
from tqdm import tqdm
from yacs.config import CfgNode as CN
from torch.utils.data import (
    DataLoader,
    ConcatDataset,
    DistributedSampler,
    SequentialSampler,
)
import pytorch_lightning as pl
from .megadepth import MegaDepthDataset
from .scannet import ScanNetDataset
from .sampler import RandomConcatSampler
from utils.augment import get_augmentor_builder


class SLiM_Dataset(pl.LightningDataModule):
    def __init__(self, config: CN):
        super(SLiM_Dataset, self).__init__()

        # 1. Read vars
        self.mode = "train" if "train" in config.MODE else "val"
        self.seed = config.GLOBAL_SEED
        self.device_enable_ddp = config.DEVICE.ENABLE_DDP
        self.parallel_workers_num = config.DATASET.PARALLEL_WORKERS_NUM
        self.data_source = None
        ## train and val
        if self.mode == "train":
            self.data_source = self.train_data_source = (
                config.DATASET.TRAINVAL_DATA_SOURCE
            )
            self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
            self.train_npz_dir = config.DATASET.TRAIN_NPZ_ROOT
            self.train_list_path = config.DATASET.TRAIN_LIST_PATH
            self.train_intrinsic_path = config.DATASET.TRAIN_INTRINSIC_PATH
            self.val_data_root = config.DATASET.VAL_DATA_ROOT
            self.val_npz_dir = config.DATASET.VAL_NPZ_ROOT
            self.val_list_path = config.DATASET.VAL_LIST_PATH
            self.val_intrinsic_path = config.DATASET.VAL_INTRINSIC_PATH
            self.min_overlap_score = config.DATASET.MIN_OVERLAP_SCORE_TRAIN
        ## test
        else:
            self.data_source = self.test_data_source = config.DATASET.TEST_DATA_SOURCE
            self.test_data_root = config.DATASET.TEST_DATA_ROOT
            self.test_npz_dir = config.DATASET.TEST_NPZ_ROOT
            self.test_list_path = config.DATASET.TEST_LIST_PATH
            self.test_intrinsic_path = config.DATASET.TEST_INTRINSIC_PATH
            self.min_overlap_score = config.DATASET.MIN_OVERLAP_SCORE_TEST
        ## Other vars
        self.sampler_n_samples_per_subset = config.SAMPLER.N_SAMPLES_PER_SUBSET
        self.sampler_subset_replacement = config.SAMPLER.SUBSET_REPLACEMENT
        self.sampler_shuffle = config.SAMPLER.SHUFFLE
        self.sampler_repeat = config.SAMPLER.REPEAT
        self.loader_batch_size = config.LOADER.BATCH_SIZE
        self.loader_num_workers = config.LOADER.NUM_WORKERS
        self.loader_pin_memory = config.LOADER.PIN_MEMORY
        ## Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        ## Dataloaders
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None
        ## Augmentor
        self.augmentor_builder = get_augmentor_builder(config.DATASET.AUGMENTATION_TYPE)

        # 2, Read all npz names stated in list
        if self.mode == "train":
            ## train
            with open(self.train_list_path, "r") as f:
                self.train_npz_names = [name.split()[0] for name in f.readlines()]
            ## val
            with open(self.val_list_path, "r") as f:
                self.val_npz_names = [name.split()[0] for name in f.readlines()]
        else:
            ## test
            with open(self.test_list_path, "r") as f:
                self.test_npz_names = [name.split()[0] for name in f.readlines()]

        # 3. Add .npz at the back for megadepth dataset, read some megadepth"s vars
        if self.data_source.lower() == "megadepth":
            if self.mode == "train":
                self.train_npz_names = [f"{n}.npz" for n in self.train_npz_names]
                self.val_npz_names = [f"{n}.npz" for n in self.val_npz_names]
            else:
                self.test_npz_names = [f"{n}.npz" for n in self.test_npz_names]
            ## Read some megadepth"s vars
            self.megadapth_img_resize = config.DATASET.MGDPT_IMG_RESIZE
            self.megadepth_division_factor = config.DATASET.MGDPT_DF
            self.megadepth_image_padding = config.DATASET.MGDPT_IMG_PAD
            self.megadepth_depth_padding = config.DATASET.MGDPT_DEPTH_PAD
            self.megadepth_coarse_scale = 1 / config.DATASET.MGDPT_COARSE_SCALE

    def setup(self, stage=None, color=False):
        # train and val
        if stage == "fit":
            ## train
            datasets = []
            for npz_name in tqdm(
                self.train_npz_names,
                desc=f"Loading {self.data_source} dataset for training",
            ):
                npz_path = osp.join(self.train_npz_dir, npz_name)
                ### Megadepth
                if self.data_source.lower() == "megadepth":
                    datasets.append(
                        MegaDepthDataset(
                            root_dir=self.train_data_root,
                            npz_path=npz_path,
                            mode="train",
                            min_overlap_score=self.min_overlap_score,
                            img_resize=self.megadapth_img_resize,
                            df=self.megadepth_division_factor,
                            img_padding=self.megadepth_image_padding,
                            depth_padding=self.megadepth_depth_padding,
                            coarse_scale=self.megadepth_coarse_scale,
                            augmentor_builder=self.augmentor_builder,
                            color=color,
                        )
                    )
                ## ScanNet
                elif self.data_source.lower() == "scannet":
                    datasets.append(
                        ScanNetDataset(
                            root_dir=self.train_data_root,
                            npz_path=npz_path,
                            intrinsic_path=self.train_intrinsic_path,
                            mode="train",
                            min_overlap_score=self.min_overlap_score,
                            augmentor_builder=self.augmentor_builder,
                        )
                    )
            self.train_dataset = ConcatDataset(datasets)
            ## val
            datasets = []
            for npz_name in tqdm(
                self.val_npz_names,
                desc=f"Loading {self.data_source} dataset for validating",
            ):
                npz_path = osp.join(self.val_npz_dir, npz_name)
                ### Megadepth
                if self.data_source.lower() == "megadepth":
                    datasets.append(
                        MegaDepthDataset(
                            root_dir=self.val_data_root,
                            npz_path=npz_path,
                            mode="val",
                            min_overlap_score=self.min_overlap_score,
                            img_resize=self.megadapth_img_resize,
                            df=self.megadepth_division_factor,
                            img_padding=self.megadepth_image_padding,
                            depth_padding=self.megadepth_depth_padding,
                            coarse_scale=self.megadepth_coarse_scale,
                            color=color,
                        )
                    )
                ## ScanNet
                elif self.data_source.lower() == "scannet":
                    datasets.append(
                        ScanNetDataset(
                            root_dir=self.val_data_root,
                            npz_path=npz_path,
                            intrinsic_path=self.val_intrinsic_path,
                            mode="val",
                            min_overlap_score=self.min_overlap_score,
                        )
                    )
            self.val_dataset = ConcatDataset(datasets)
        # test
        else:
            datasets = []
            for npz_name in tqdm(
                self.test_npz_names,
                desc=f"Loading {self.data_source} dataset for testing",
            ):
                npz_path = osp.join(self.test_npz_dir, npz_name)
                ## Megadepth
                if self.data_source.lower() == "megadepth":
                    datasets.append(
                        MegaDepthDataset(
                            root_dir=self.test_data_root,
                            npz_path=npz_path,
                            mode="val",
                            min_overlap_score=self.min_overlap_score,
                            img_resize=self.megadapth_img_resize,
                            df=self.megadepth_division_factor,
                            img_padding=self.megadepth_image_padding,
                            depth_padding=self.megadepth_depth_padding,
                            coarse_scale=self.megadepth_coarse_scale,
                            color=color,
                        )
                    )
                ## ScanNet
                elif self.data_source.lower() == "scannet":
                    datasets.append(
                        ScanNetDataset(
                            root_dir=self.test_data_root,
                            npz_path=npz_path,
                            intrinsic_path=self.test_intrinsic_path,
                            mode="val",
                            min_overlap_score=self.min_overlap_score,
                        )
                    )
            self.test_dataset = ConcatDataset(datasets)

    def train_dataloader(self):
        sampler = RandomConcatSampler(
            data_source=self.train_dataset,
            n_samples_per_subset=self.sampler_n_samples_per_subset,
            subset_replacement=self.sampler_subset_replacement,
            shuffle=self.sampler_shuffle,
            repeat=self.sampler_repeat,
            seed=self.seed,
        )
        self._train_dataloader = DataLoader(
            dataset=self.train_dataset,
            sampler=sampler,
            batch_size=self.loader_batch_size,
            num_workers=self.loader_num_workers,
            pin_memory=self.loader_pin_memory,
        )
        return self._train_dataloader

    def val_dataloader(self):
        sampler = (
            DistributedSampler(dataset=self.val_dataset, shuffle=False)
            if self.device_enable_ddp
            else SequentialSampler(data_source=self.val_dataset)
        )
        self._val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.loader_batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.loader_num_workers,
            pin_memory=self.loader_pin_memory,
        )
        return self._val_dataloader

    def test_dataloader(self, *args, **kwargs):
        sampler = (
            DistributedSampler(dataset=self.test_dataset, shuffle=False)
            if self.device_enable_ddp
            else SequentialSampler(data_source=self.test_dataset)
        )
        self._test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.loader_batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.loader_num_workers,
            pin_memory=self.loader_pin_memory,
        )

        return self._test_dataloader
