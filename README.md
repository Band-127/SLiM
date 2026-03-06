# SoMa: Scalable Feature Matching via State Space Modeling and Sparse Correlation (CVPR 2026)
> Choo Sin Wai, Bo Li\* \
> \* corresponding author

<p align="center">
  <img src="assets/pipeline.png" width="400" alt="Method pipeline" />
  <br/>
  <sub>Figure 1: Method pipeline.</sub>
</p>

<p align="center">
  <img src="assets/intro_image.png" width="400" alt="Comparison" />
  <br/>
  <sub>Figure 2: Accuracy-efficiency tradeoff comparison on MegaDepth.</sub>
</p>

- [📰 News](#news)
- [✅ TODO](#todo)
- [⚙️ Installation](#installation-and-environment-setup)
- [📦 Pretrained model](#pretrained-model)
- [🏋️ Training](#training)
- [🧪 Testing](#testing)
- [📚 Citation](#citation)

<a id="news"></a>

## 📰 News

- [2026.02] Our paper is accepted by CVPR 2026.

<a id="todo"></a>

## ✅ TODO

- [x] Release installation steps.
- [ ] Release train and test demo.
- [ ] Release pre-trained models.

<a id="installation-and-environment-setup"></a>

## ⚙️ Installation and environment setup

### 1. 🗂️ Dataset Setup (thanks to LoFTR for the guides)

#### 1.1 📥 Download dataset indices

- Download the dataset indices (filename: data.7z) from this [Google Drive folder](https://drive.google.com/drive/folders/1fhAHN5tYr4yANkkFSHJ5kXjcSnepu0l3?usp=sharing)
- Unzip them and place them under the project root as follows:
```
Efficient Matching
└── data
     ├── megadepth
     │   └── index
     │        ├── scene_info_0.1_0.7
     │        ├── scene_info_val_1500
     │        └── trainvaltest_list
     └── scannet
          ├── index
          │   ├── intrinsics.npz
          │   ├── scene_data
          │   └── statistics.json
          └── test
               └── scenexxxx_xx (100 scenes here)
```

#### 1.2 📦 Download dataset

##### 1.2.1 🏔️ MegaDepth

Similar to LoFTR, we use depth maps provided in the original MegaDepth dataset as well as undistorted images, corresponding camera intrinsics and extrinsics preprocessed by D2-Net. You can download them separately from the following links:

- [MegaDepth undistorted images and processed depths](https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz)
    - Note that we only use depth maps.
    - Path of the downloaded data will be referred to as `/path/to/megadepth`
- [D2-Net preprocessed images](https://drive.google.com/drive/folders/1hxpOsqOZefdrba_BqnW490XpNX_LgXPB)
    - Images are undistorted manually in D2-Net since the undistorted images from MegaDepth do not come with corresponding intrinsics.
    - Path of the downloaded data will be referred to as `/path/to/megadepth_d2net`

##### 1.2.2 🏢 ScanNet

- Download the dataset following the official guide: [ScanNet](https://github.com/ScanNet/ScanNet), and use the Python-exported data.

#### 1.3 🔗 Build the dataset symlinks

We symlink the datasets into the `data` directory under the project root.
```bash
# scannet
# -- # train dataset
ln -s /path/to/scannet_train/* /path/to/Efficient_Matching/data/scannet/train

# megadepth
# -- # train and test dataset (train and test share the same dataset)
ln -sv /path/to/megadepth/phoenix /path/to/megadepth_d2net/Undistorted_SfM /path/to/Efficient_Matching/data/megadepth/train
ln -sv /path/to/megadepth/phoenix /path/to/megadepth_d2net/Undistorted_SfM /path/to/Efficient_Matching/data/megadepth/test
```

#### 1.4 🧱 Final data structure

```
Efficient Matching
└── data
      ├── megadepth
      │   ├── index
      │   │   ├── scene_info_0.1_0.7
      │   │   ├── scene_info_val_1500
      │   │   └── trainvaltest_list
      │   ├── test
      │   │   ├── phoenix
      │   │   └── Undistorted_SfM
      │   └── train
      │       ├── phoenix
      │       └── Undistorted_SfM
      └── scannet
          ├── index
          │   ├── intrinsics.npz
          │   ├── scene_data
          │   └── statistics.json
          ├── test
               └── scenexxxx_xx (100 scenes here)
          └── train
               └── scenexxxx_xx (1513 scenes here)
```

### 2. 🧰 Environment Setup

We have tested the following environment on **Ubuntu 22.04**.

- **CUDA version** does not have to be identical to ours, but make sure:
  - Your **NVIDIA driver** supports the CUDA version used by your **PyTorch build** (i.e., driver capability should be >= PyTorch CUDA version).
  - The **PyTorch CUDA version** should match the **CUDA toolkit** you compile against in the environment.
- **CUDA toolkit choice**
  - If your system already has a compatible `cuda-toolkit`, you can use the system installation.
  - Otherwise, install `cuda-toolkit` inside the conda environment (as shown below) and use it for compilation.
- **GCC/G++ compatibility**
  - `gcc/g++` must be compatible with the CUDA toolkit version you use.
  - If your system `gcc/g++` can compile CUDA extensions successfully, you do **not** need to install `gcc/g++` in conda.

```bash
conda create -n soma -y python=3.10
conda activate soma

# torch = 2.1.1+cu118
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118\

# mamba and causal-conv1d package, check mamba github page if installation/build fails
conda install cuda-toolkit==11.8 -c nvidia/label/cuda-11.8.0
pip install mamba-ssm==2.2.2
pip install causal-conv1d==1.2.1

# other dependencies: pytorch-lightning, albumentation, yacs etc.
pip install -r requirements.txt

# selective scan from vmamba
#     force use conda gcc and g++ (=11)
conda install -c conda-forge -y gcc_linux-64=11 gxx_linux-64=11
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
#     force use conda cuda
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
#     compilation takes time, please wait patiently
cd src/backbone/vssm/kernels/selective_scan
pip install . --no-build-isolation
cd ../../../../../
```


<a id="pretrained-model"></a>

## 📦 Pretrained model

<a id="training"></a>

## 🏋️ Training

<a id="testing"></a>

## 🧪 Testing

<a id="citation"></a>

## 📚 Citation

```bibtex
@inproceedings{choo2026soma,
  title={Scalable Feature Matching via State Space Modeling and Sparse Correlation},
  author={Choo, Sin Wai and Li, Bo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
